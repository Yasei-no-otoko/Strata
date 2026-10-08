// Compares dsv4/ops.cpp against tests/golden.txt (numpy port of the official model.py).  Usage: test_ops tests/golden.txt
#include "dsv4/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

static int g_fail = 0;
static std::map<std::string, std::vector<double>> G;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static bool near(const std::string& name, const std::vector<float>& got, double tol) {
    const auto& ref = G.at(name);
    if (ref.size() != got.size()) { std::fprintf(stderr, "FAIL %s: size %zu vs %zu\n", name.c_str(), got.size(), ref.size()); ++g_fail; return false; }
    double worst = 0;
    for (size_t i = 0; i < ref.size(); ++i) worst = std::max(worst, std::fabs(ref[i] - got[i]));
    if (worst > tol) { std::fprintf(stderr, "FAIL %s: max abs diff %.3g > %.3g\n", name.c_str(), worst, tol); ++g_fail; return false; }
    std::printf("ok  %-28s max|diff| = %.3g\n", name.c_str(), worst);
    return true;
}
static void exact_floats(const char* name, const std::vector<float>& got, const std::vector<float>& ref) {
    bool ok = got.size() == ref.size();
    for (size_t i = 0; ok && i < got.size(); ++i) {
        uint32_t a = 0, b = 0;
        std::memcpy(&a, &got[i], sizeof(a));
        std::memcpy(&b, &ref[i], sizeof(b));
        ok = a == b;
    }
    if (!ok) { std::fprintf(stderr, "FAIL %s (scalar/optimized bits differ)\n", name); ++g_fail; }
    else std::printf("ok  %-28s scalar bits exact\n", name);
}
static void same_ints(const std::string& name, const std::vector<int>& got) {
    const auto& ref = G.at(name);
    bool ok = ref.size() == got.size();
    for (size_t i = 0; ok && i < ref.size(); ++i) ok = (int) ref[i] == got[i];
    if (!ok) { std::fprintf(stderr, "FAIL %s (indices differ)\n", name.c_str()); ++g_fail; } else std::printf("ok  %-28s exact\n", name.c_str());
}
static std::vector<float> F(const std::string& n) { std::vector<float> v; for (double d : G.at(n)) v.push_back((float) d); return v; }

static void scalar_sparse_attn(const float* q, int h, int d, const float* kv, const int* idxs, int topk,
                               const float* sink, float scale, float* o) {
    std::vector<int> v;
    for (int t = 0; t < topk; ++t) if (idxs[t] >= 0) v.push_back(idxs[t]);
    for (int hh = 0; hh < h; ++hh) {
        float* oo = o + (size_t) hh * d;
        std::fill(oo, oo + d, 0.f);
        if (v.empty()) continue;
        std::vector<double> s(v.size());
        const float* qq = q + (size_t) hh * d;
        double mx = -1e300;
        for (size_t t = 0; t < v.size(); ++t) {
            const float* kk = kv + (size_t) v[t] * d;
            double a = 0;
            for (int i = 0; i < d; ++i) a += (double) qq[i] * kk[i];
            s[t] = a * scale;
            mx = std::max(mx, s[t]);
        }
        double den = std::exp((double) sink[hh] - mx);
        for (auto& e : s) { e = std::exp(e - mx); den += e; }
        for (size_t t = 0; t < v.size(); ++t) {
            const float* kk = kv + (size_t) v[t] * d;
            for (int i = 0; i < d; ++i) oo[i] += (float) (s[t] / den * kk[i]);
        }
    }
}

static double scalar_dot(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += (double) a[i] * b[i];
    return sum;
}

static void scalar_hc_post(const float* x, const float* residual, const float* post, const float* comb,
                           int hc, int d, float* out) {
    for (int k = 0; k < hc; ++k)
        for (int i = 0; i < d; ++i) {
            double a = (double) post[k] * x[i];
            for (int j = 0; j < hc; ++j) a += (double) comb[j * hc + k] * residual[j * d + i];
            out[k * d + i] = (float) a;
        }
}

static void scalar_hc_head(const float* x, int hc, int d, const float* fn, const float* scale,
                           const float* base, float norm_eps, float hc_eps, float* y) {
    const int n = hc * d;
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const double rs = 1.0 / std::sqrt(ss / n + norm_eps);
    std::vector<double> pre((size_t) hc);
    for (int m = 0; m < hc; ++m) {
        const double a = scalar_dot(fn + (size_t) m * n, x, n);
        pre[(size_t) m] = 1.0 / (1.0 + std::exp(-(a * rs * scale[0] + base[m]))) + hc_eps;
    }
    for (int i = 0; i < d; ++i) {
        double a = 0;
        for (int j = 0; j < hc; ++j) a += pre[(size_t) j] * x[j * d + i];
        y[i] = (float) a;
    }
}

static void scalar_hc_pre(const float* x, int hc, int d, const float* fn, const float* scale,
                          const float* base, float norm_eps, int iters, float eps,
                          float* y, float* post, float* comb) {
    const int n = hc * d, mix = (2 + hc) * hc;
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const double rs = 1.0 / std::sqrt(ss / n + norm_eps);
    std::vector<float> mixes((size_t) mix), pre((size_t) hc);
    for (int m = 0; m < mix; ++m)
        mixes[(size_t) m] = (float) (scalar_dot(fn + (size_t) m * n, x, n) * rs);
    dsv4::hc_split_sinkhorn(mixes.data(), scale, base, hc, iters, eps, pre.data(), post, comb);
    for (int i = 0; i < d; ++i) {
        double a = 0;
        for (int j = 0; j < hc; ++j) a += (double) pre[(size_t) j] * x[j * d + i];
        y[i] = (float) a;
    }
}

static int run_bench() {
    using clock_type = std::chrono::steady_clock;
    constexpr int reps = 32;
#ifdef _OPENMP
    const int workers = omp_get_max_threads();
#else
    const int workers = 1;
#endif
    uint32_t state = 0xC0FFEE12u;
    auto randf = [&]() {
        state = state * 1664525u + 1013904223u;
        return (float)((int)((state >> 8) % 20001u) - 10000) / 10000.0f;
    };
    volatile double checksum = 0.0;
    auto report = [&](const char* name, auto&& f) {
        for (int i = 0; i < 2; ++i) f();
        const auto t0 = clock_type::now();
        for (int i = 0; i < reps; ++i) f();
        const double elapsed = std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
        std::printf("bench %-18s workers=%d reps=%d total_ms=%.3f ms_per_call=%.4f\n",
                    name, workers, reps, elapsed, elapsed / reps);
    };

    {
        constexpr int h = 64, d = 512, topk = 128;
        std::vector<float> q((size_t)h * d), kv((size_t)topk * d), sink((size_t)h), out((size_t)h * d);
        std::vector<int> idx((size_t)topk);
        for (auto& x : q) x = randf();
        for (auto& x : kv) x = randf();
        for (auto& x : sink) x = randf();
        for (int& x : idx) x = &x - idx.data();
        report("sparse_attn", [&] {
            dsv4::sparse_attn_token(q.data(), h, d, kv.data(), idx.data(), topk, sink.data(), 0.0441941738f, out.data());
            checksum += out[(size_t)(state & 4095u)];
        });
    }
    {
        constexpr int hc = 4, d = 4096, n = hc * d, mix = (2 + hc) * hc;
        std::vector<float> x((size_t)n), fn((size_t)mix * n), scale(3), base((size_t)mix),
                           y((size_t)d), post((size_t)hc), comb((size_t)hc * hc),
                           residual((size_t)n), out((size_t)n);
        for (auto& v : x) v = randf();
        for (auto& v : fn) v = randf();
        for (auto& v : scale) v = randf();
        for (auto& v : base) v = randf();
        for (auto& v : residual) v = randf();
        dsv4::hc_split_sinkhorn(fn.data(), scale.data(), base.data(), hc, 7, 1e-6f,
                                post.data(), post.data(), comb.data());
        report("hc_pre", [&] {
            dsv4::hc_pre(x.data(), hc, d, fn.data(), scale.data(), base.data(), 1e-6f, 7, 1e-6f,
                         y.data(), post.data(), comb.data());
            checksum += y[(size_t)(state & 4095u)];
        });
        report("hc_post", [&] {
            dsv4::hc_post(x.data(), residual.data(), post.data(), comb.data(), hc, d, out.data());
            checksum += out[(size_t)(state & 16383u)];
        });
        report("hc_head", [&] {
            dsv4::hc_head(x.data(), hc, d, fn.data(), scale.data(), base.data(), 1e-6f, 1e-6f, y.data());
            checksum += y[(size_t)(state & 4095u)];
        });
    }
    std::printf("bench checksum=%.9g\n", (double)checksum);
    return 0;
}

int main(int argc, char** argv) {
    std::ifstream f(argc > 1 ? argv[1] : "tests/golden.txt");
    if (!f) { std::fprintf(stderr, "cannot open golden file\n"); return 2; }
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line); std::string name; size_t n; is >> name >> n;
        std::vector<double> v(n); for (auto& x : v) is >> x;
        G[name] = v;
    }
    if (argc > 2 && std::string(argv[2]) == "--bench") return run_bench();
    using namespace dsv4;
    struct GC { const char* n; Score fn; bool bias, hash; } cases[] = {
        {"sqrtsoftplus", Score::SqrtSoftplus, true, false}, {"hash", Score::SqrtSoftplus, false, true},
        {"softmax", Score::Softmax, false, false}, {"sigmoid", Score::Sigmoid, true, false}};
    for (auto& c : cases) {
        const std::string p = std::string("gate_") + c.n;
        auto lg = F(p + "_logits"); std::vector<float> bias; if (c.bias) bias = F(p + "_bias");
        const int32_t h[6] = {17, 3, 200, 41, 99, 250};
        int32_t idx[6]; float w[6];
        gate_route(lg.data(), 256, 6, c.fn, c.bias ? bias.data() : nullptr, c.hash ? h : nullptr, 1.5f, idx, w);
        same_ints(p + "_idx", std::vector<int>(idx, idx + 6));
        near(p + "_w", std::vector<float>(w, w + 6), 1e-6);
    }
    for (const char* tag : {"yarn", "plain"}) {
        std::vector<float> cs, sn;
        const bool y = std::string(tag) == "yarn";
        yarn_table(64, 64, y ? 65536 : 0, y ? 160000.0 : 10000.0, 16.0, 32, 1, cs, sn);
        near(std::string("rope_") + tag + "_cos", cs, 1e-4);
        near(std::string("rope_") + tag + "_sin", sn, 1e-4);
        for (bool inv : {false, true}) {
            auto x = F(std::string("rope_") + tag + "_x");
            rotary(x.data(), 64, 64, &cs[37 * 32], &sn[37 * 32], inv);
            near(std::string("rope_") + tag + (inv ? "_inv" : "_fwd"), x, 1e-4);
        }
    }
    {   // rotary only touches the LAST rope_dim elements of a longer head vector
        std::vector<float> x(128, 1.f), cs(32, 0.f), sn(32, 1.f);  // rotate by 90 degrees
        rotary(x.data(), 128, 64, cs.data(), sn.data(), false);
        CHECK(x[0] == 1.f && x[63] == 1.f && x[64] == -1.f && x[65] == 1.f);
    }
    struct W { const char* n; int w, s, p; } wins[] = {{"win_prefill", 8, 11, 0}, {"win_short", 8, 1, 3}, {"win_wrap", 8, 1, 13}, {"win_exact", 8, 1, 7}};
    for (auto& t : wins) {
        std::vector<int> o; int r, c; window_topk(t.w, t.s, t.p, o, r, c);
        CHECK(G.at(std::string(t.n) + "_shape")[0] == r && G.at(std::string(t.n) + "_shape")[1] == c);
        same_ints(t.n, o);
    }
    struct C { const char* n; int r, s, p, o; } cmps[] = {{"cmp_prefill", 4, 11, 0, 11}, {"cmp_decode", 4, 1, 11, 8}, {"cmp_decode2", 128, 1, 300, 128}};
    for (auto& t : cmps) {
        std::vector<int> o; int r, c; compress_topk(t.r, t.s, t.p, t.o, o, r, c);
        CHECK(G.at(std::string(t.n) + "_shape")[0] == r && G.at(std::string(t.n) + "_shape")[1] == c);
        same_ints(t.n, o);
    }
    {
        auto g = F("swiglu_gate"), u = F("swiglu_up"); std::vector<float> o(64);
        swiglu_clamped(g.data(), u.data(), 64, 10.f, o.data());
        near("swiglu_out", o, 1e-5);
    }
    {   // rmsnorm sanity: unit weights, [3,4] -> rms = sqrt(12.5)
        float x[2] = {3, 4}, y[2];
        rmsnorm(x, nullptr, 0.f, 2, y);
        CHECK(std::fabs(y[0] - 3.f / std::sqrt(12.5f)) < 1e-6f);
    }
    {   // hyper-connections: pre / post / comb / hc_post / hc_head
        const int HC = 4, D = 16;
        auto x = F("hc_x"), fn = F("hc_fn"), sc = F("hc_scale"), bs = F("hc_base"), sub = F("hc_sub");
        std::vector<float> y(D), post(HC), comb(HC * HC), out(HC * D);
        hc_pre(x.data(), HC, D, fn.data(), sc.data(), bs.data(), 1e-6f, 20, 1e-6f, y.data(), post.data(), comb.data());
        near("hc_pre_out", y, 1e-5); near("hc_post_w", post, 1e-5); near("hc_comb", comb, 1e-5);
        hc_post(sub.data(), x.data(), post.data(), comb.data(), HC, D, out.data());
        near("hc_post_out", out, 1e-5);
        // Sinkhorn property: comb is (almost) doubly stochastic
        for (int j = 0; j < HC; ++j) { double r = 0; for (int k = 0; k < HC; ++k) r += comb[j * HC + k]; CHECK(std::fabs(r - 1.0) < 1e-3); }
        auto hf = F("hch_fn"), hs = F("hch_scale"), hb = F("hch_base");
        std::vector<float> hy(D);
        hc_head(x.data(), HC, D, hf.data(), hs.data(), hb.data(), 1e-6f, 1e-6f, hy.data());
        near("hch_out", hy, 1e-5);
    }
    {   // sparse attention with sink and masked (-1) slots
        auto q = F("sa_q"), kv = F("sa_kv"), sk = F("sa_sink"); std::vector<int> ix;
        for (double v : G.at("sa_idx")) ix.push_back((int) v);
        std::vector<float> o(4 * 16);
        sparse_attn_token(q.data(), 4, 16, kv.data(), ix.data(), (int) ix.size(), sk.data(), 0.25f, o.data());
        near("sa_out", o, 1e-5);
        std::vector<int> none(3, -1); std::vector<float> z(4 * 16, 9.f);
        sparse_attn_token(q.data(), 4, 16, kv.data(), none.data(), 3, sk.data(), 0.25f, z.data());
        CHECK(z[0] == 0.f && z[63] == 0.f);
    }
    {   // AVX2 kernels must preserve the scalar float/double operation order, including tails.
        uint32_t state = 0xA341316Cu;
        auto randf = [&]() {
            state = state * 1664525u + 1013904223u;
            return (float)((int)((state >> 8) % 20001u) - 10000) / 10000.0f;
        };
        for (int d : {1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 511, 512}) {
            const int h = 3;
            std::vector<float> q((size_t)h * d), kv((size_t)16 * d), sink((size_t)h);
            for (auto& x : q) x = randf();
            for (auto& x : kv) x = randf();
            for (auto& x : sink) x = randf();
            for (int topk : {0, 1, 2, 5, 11, 19}) {
                std::vector<int> idx((size_t)topk);
                for (int t = 0; t < topk; ++t) idx[(size_t)t] = (t % 4 == 1) ? -1 : (t * 7 + 3) % 16;
                std::vector<float> got((size_t)h * d), ref((size_t)h * d);
                sparse_attn_token(q.data(), h, d, kv.data(), idx.data(), topk, sink.data(), 0.125f, got.data());
                scalar_sparse_attn(q.data(), h, d, kv.data(), idx.data(), topk, sink.data(), 0.125f, ref.data());
                exact_floats("sparse-attn scalar", got, ref);
            }

            const int hc = 4, n = hc * d, mix = (2 + hc) * hc;
            std::vector<float> x((size_t)n), fn((size_t)mix * n), scale(3),
                               base((size_t)mix), post((size_t)hc), comb((size_t)hc * hc),
                               got((size_t)d), ref((size_t)d), got_post((size_t)n), ref_post((size_t)n);
            for (auto& v : x) v = randf();
            for (auto& v : fn) v = randf();
            for (auto& v : scale) v = randf();
            for (auto& v : base) v = randf();
            for (auto& v : post) v = randf();
            for (auto& v : comb) v = randf();
            std::vector<float> gpost((size_t)hc), rpost((size_t)hc), gcomb((size_t)hc * hc), rcomb((size_t)hc * hc);
            hc_pre(x.data(), hc, d, fn.data(), scale.data(), base.data(), 1e-6f, 7, 1e-6f,
                   got.data(), gpost.data(), gcomb.data());
            scalar_hc_pre(x.data(), hc, d, fn.data(), scale.data(), base.data(), 1e-6f, 7, 1e-6f,
                          ref.data(), rpost.data(), rcomb.data());
            exact_floats("hc-pre output", got, ref);
            exact_floats("hc-pre post", gpost, rpost);
            exact_floats("hc-pre comb", gcomb, rcomb);
            hc_post(x.data(), x.data(), post.data(), comb.data(), hc, d, got_post.data());
            scalar_hc_post(x.data(), x.data(), post.data(), comb.data(), hc, d, ref_post.data());
            exact_floats("hc-post", got_post, ref_post);
            hc_head(x.data(), hc, d, fn.data(), scale.data(), base.data(), 1e-6f, 1e-6f, got.data());
            scalar_hc_head(x.data(), hc, d, fn.data(), scale.data(), base.data(), 1e-6f, 1e-6f, ref.data());
            exact_floats("hc-head", got, ref);
        }
    }
    std::printf("%s\n", g_fail ? "OPS TESTS FAILED" : "OPS TESTS PASSED");
    return g_fail ? 1 : 0;
}
