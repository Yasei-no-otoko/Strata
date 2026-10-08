cmake_minimum_required(VERSION 3.24)
project(dsv4_strata_ext LANGUAGES CXX)
enable_testing()

# Eredita o imposta lo standard C++ (Strata usa C++20, manteniamo la coerenza)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Backend device: di default quello EMULATO su CPU (src/gpu.cpp), che non richiede CUDA.
# Con -DDSV4_WITH_CUDA=ON si compila src/gpu_cuda.cu AL POSTO di src/gpu.cpp: i due backend
# definiscono gli stessi simboli dsv4::gpu, quindi non possono mai stare nello stesso binario.
option(DSV4_WITH_CUDA "Build the real CUDA device layer (src/gpu_cuda.cu) instead of the CPU-emulated one" OFF)
option(DSV4_WITH_HIP "Build the real AMD HIP device layer" OFF)
option(DSV4_AVX2 "Opt in to AVX2 CPU kernels (requires an AVX2-capable host at runtime)" OFF)
option(DSV4_NATIVE_CPU "Use ggml-cpu quantized activation dots for MoE host rows (opt-in, AVX2 floor)" OFF)
option(DSV4_CPU_BMI2 "Enable ggml BMI2 kernels with DSV4_NATIVE_CPU (may be slow on some Zen CPUs)" OFF)
option(DSV4_CPU_AVX512 "Use the guarded AVX-512 VNNI IQ1_M CPU MoE dot kernel (opt-in)" OFF)
set(DSV4_MATVEC_THREADS "256" CACHE STRING "HIP/CUDA fallback block width (32, 64, 128, 256, or 512; shape policy may use 32 for measured Q8_0/Q4_K cases)")
set_property(CACHE DSV4_MATVEC_THREADS PROPERTY STRINGS 32 64 128 256 512)
if(NOT DSV4_MATVEC_THREADS MATCHES "^(32|64|128|256|512)$")
    message(FATAL_ERROR "DSV4_MATVEC_THREADS must be 32, 64, 128, 256, or 512")
endif()
set(DSV4_GGML_DIR "" CACHE PATH "llama.cpp checkout to reuse for DSV4_NATIVE_CPU (empty: fetch pinned llama.cpp only when enabled)")
if(DSV4_WITH_CUDA AND DSV4_WITH_HIP)
    message(FATAL_ERROR "Select only one of DSV4_WITH_CUDA and DSV4_WITH_HIP")
endif()

# ggml-cpu supplies optimized IQ1_M and other native format row dots that the standalone reference
# implementation does not have. It stays opt-in so normal extension builds remain self-contained.
if(DSV4_NATIVE_CPU)
    # Do not reuse a ggml target created by a parent project: its ISA flags are already fixed,
    # and may exceed the AVX2 runtime floor guarded by native_cpu.cpp.
    if(TARGET ggml-cpu OR TARGET ggml-base)
        message(FATAL_ERROR "DSV4_NATIVE_CPU requires this directory to configure its own ggml targets; pre-existing ggml targets have an unverifiable ISA contract")
    endif()
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
        message(FATAL_ERROR "DSV4_NATIVE_CPU requires an x86 CPU build with an AVX2-capable host")
    endif()
    enable_language(C)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(GGML_STATIC ON CACHE BOOL "" FORCE)
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
    set(GGML_AVX ON CACHE BOOL "" FORCE)
    set(GGML_AVX2 ON CACHE BOOL "" FORCE)
    # The pinned ggml fallback stays at the AVX2 ISA floor. The optional AVX-512
    # IQ1_M override is compiled separately and called only after its own runtime guard.
    foreach(_dsv4_ggml_advanced_isa
            GGML_AVX_VNNI GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI
            GGML_AVX512_BF16 GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
        set(${_dsv4_ggml_advanced_isa} OFF CACHE BOOL "" FORCE)
    endforeach()
    set(GGML_FMA ON CACHE BOOL "" FORCE)
    set(GGML_F16C ON CACHE BOOL "" FORCE)
    set(GGML_BMI2 ${DSV4_CPU_BMI2} CACHE BOOL "" FORCE)
    if(NOT TARGET ggml-cpu OR NOT TARGET ggml-base)
        if(DSV4_GGML_DIR)
            add_subdirectory(${DSV4_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
        else()
            include(FetchContent)
            FetchContent_Declare(dsv4_llamacpp
                GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
                GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
                GIT_SHALLOW FALSE)
            FetchContent_GetProperties(dsv4_llamacpp)
            if(NOT dsv4_llamacpp_POPULATED)
                FetchContent_Populate(dsv4_llamacpp)
            endif()
            add_subdirectory(${dsv4_llamacpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
        endif()
    endif()
    if(NOT TARGET ggml-cpu OR NOT TARGET ggml-base)
        message(FATAL_ERROR "DSV4_NATIVE_CPU needs ggml-cpu and ggml-base; set DSV4_GGML_DIR to a llama.cpp checkout")
    endif()
endif()
if(DSV4_CPU_BMI2 AND NOT DSV4_NATIVE_CPU)
    message(FATAL_ERROR "DSV4_CPU_BMI2 requires DSV4_NATIVE_CPU=ON")
endif()
if(DSV4_CPU_AVX512 AND NOT DSV4_NATIVE_CPU)
    message(FATAL_ERROR "DSV4_CPU_AVX512 requires DSV4_NATIVE_CPU=ON")
endif()

if(DSV4_WITH_HIP)
    enable_language(HIP)
    find_package(hip CONFIG REQUIRED)
    set(DSV4_GPU_SRC src/gpu_cuda.cu)
    set_source_files_properties(${DSV4_GPU_SRC} PROPERTIES LANGUAGE HIP)
elseif(DSV4_WITH_CUDA)
    # sm_89 = RTX 4090 di questa macchina. Per un'altra scheda: -DCMAKE_CUDA_ARCHITECTURES="80;89".
    # Va impostato PRIMA di enable_language(CUDA): dopo, cmake ha gia' messo il suo default (52) nella
    # cache e "NOT DEFINED" non lo vede piu', quindi il kernel verrebbe compilato per la scheda sbagliata.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()
    # nvcc 12.8 e' in /usr/local/cuda-12.8: se cmake non lo trova da solo, diglielo.
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
        find_program(DSV4_NVCC NAMES nvcc nvcc-12.8 PATHS /usr/local/cuda/bin /usr/local/cuda-12.8/bin
                                                /usr/local/cuda-12/bin ENV PATH)
        if(DSV4_NVCC)
            set(CMAKE_CUDA_COMPILER "${DSV4_NVCC}")
        endif()
    endif()
    enable_language(CUDA)
    set(DSV4_GPU_SRC src/gpu_cuda.cu)
else()
    set(DSV4_GPU_SRC src/gpu.cpp)
endif()

# OpenMP: senza, i "#pragma omp parallel for" di cpu_mv/gpu::matvec/experts_hit sono inerti e il MoE
# gira su un core solo. Obbligatorio se trovato, con avviso esplicito altrimenti.
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP not found: dsv4_run will run SINGLE-THREADED (roughly 20x slower).")
endif()

# 1. Definiamo la Libreria Core Additiva
# src/iq_tables.cpp e' generato: python3 tools/extract_iq_tables.py
add_library(dsv4_core
    src/gguf_header.cpp
    src/config.cpp
    src/mem_plan.cpp
    src/ops.cpp
    src/native_cpu.cpp
    src/cpu_pool.cpp
    src/cpu_moe.cpp
    src/dequant.cpp
    src/iq_tables.cpp
    ${DSV4_GPU_SRC}
    src/sampler.cpp
    src/serve_loop.cpp
    src/model.cpp
)
if(DSV4_CPU_AVX512)
    target_sources(dsv4_core PRIVATE src/native_cpu_avx512.cpp)
    target_include_directories(dsv4_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(dsv4_core PRIVATE DSV4_NATIVE_CPU_AVX512=1)
    if(DSV4_GGML_DIR)
        target_include_directories(dsv4_core PRIVATE ${DSV4_GGML_DIR}/ggml/src)
    else()
        target_include_directories(dsv4_core PRIVATE ${dsv4_llamacpp_SOURCE_DIR}/ggml/src)
    endif()
    if(MSVC)
        set_source_files_properties(src/native_cpu_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        set_source_files_properties(src/native_cpu_avx512.cpp PROPERTIES
            COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni")
    else()
        message(FATAL_ERROR "DSV4_CPU_AVX512 is not configured for ${CMAKE_CXX_COMPILER_ID}")
    endif()
endif()
find_package(Threads REQUIRED)

# Riferiamo i percorsi d'inclusione interni al modulo
target_include_directories(dsv4_core PUBLIC include)
target_link_libraries(dsv4_core PUBLIC Threads::Threads)
if(WIN32)
    target_compile_definitions(dsv4_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()
if(DSV4_WITH_HIP)
    target_compile_definitions(dsv4_core PRIVATE DSV4_USE_HIP=1)
    target_link_libraries(dsv4_core PUBLIC hip::host)
    set_target_properties(dsv4_core PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
endif()
if(DSV4_WITH_CUDA OR DSV4_WITH_HIP)
    target_compile_definitions(dsv4_core PRIVATE DSV4_MATVEC_THREADS=${DSV4_MATVEC_THREADS})
endif()
if(DSV4_NATIVE_CPU)
    target_compile_definitions(dsv4_core PUBLIC DSV4_NATIVE_CPU=1)
    if(DSV4_CPU_BMI2)
        target_compile_definitions(dsv4_core PRIVATE DSV4_NATIVE_CPU_BMI2=1)
    endif()
    target_link_libraries(dsv4_core PUBLIC ggml-cpu ggml-base)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(dsv4_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Keep the baseline portable by default. This option only changes the C++ host-op
# translation unit; never pass x86 flags into HIP/CUDA compilation. No fast-math or FMA is enabled.
if(DSV4_AVX2)
    if(MSVC)
        set_source_files_properties(src/ops.cpp PROPERTIES
            COMPILE_DEFINITIONS DSV4_ENABLE_AVX2=1
            COMPILE_OPTIONS "/arch:AVX2;/fp:precise")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        set_source_files_properties(src/ops.cpp PROPERTIES
            COMPILE_DEFINITIONS DSV4_ENABLE_AVX2=1
            COMPILE_OPTIONS "-mavx2;-mno-fma;-ffp-contract=off")
    else()
        message(FATAL_ERROR "DSV4_AVX2 is not configured for ${CMAKE_CXX_COMPILER_ID}")
    endif()
endif()

# Consente a dsv4_core di accedere agli header di Strata qualora servissero in Phase 2
target_include_directories(dsv4_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# 2. Definiamo l'eseguibile principale dsv4_run
add_executable(dsv4_run src/dsv4_run_main.cpp)
target_link_libraries(dsv4_run PRIVATE dsv4_core)   # + strata_core in Phase 2 (read-only use)

# 2b. Il planner di residenza degli esperti (CPU only)
add_executable(dsv4_plan src/dsv4_plan_main.cpp)
target_link_libraries(dsv4_plan PRIVATE dsv4_core)

# 3. Definiamo i Test interni del modulo
add_executable(test_dsv4_core tests/test_dsv4_core.cpp)
target_link_libraries(test_dsv4_core PRIVATE dsv4_core)
add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE dsv4_core)
add_executable(test_cpu_pool tests/test_cpu_pool.cpp)
target_link_libraries(test_cpu_pool PRIVATE dsv4_core)
add_executable(test_cpu_moe tests/test_cpu_moe.cpp)
target_link_libraries(test_cpu_moe PRIVATE dsv4_core)
# La matematica di decode dei kernel CUDA (include/dsv4/dq_traits.hpp), verificata su CPU contro
# dequant_row()/row_dot(): nessun touch della GPU, nessun nvcc.
add_executable(test_dq_traits tests/test_dq_traits.cpp)
target_link_libraries(test_dq_traits PRIVATE dsv4_core)
if(DSV4_NATIVE_CPU)
    add_executable(test_native_cpu tests/test_native_cpu.cpp)
    target_link_libraries(test_native_cpu PRIVATE dsv4_core)
    if(DSV4_CPU_AVX512)
        target_compile_definitions(test_native_cpu PRIVATE DSV4_NATIVE_CPU_AVX512=1)
    endif()
endif()
add_test(NAME core COMMAND test_dsv4_core)
add_test(NAME ops COMMAND test_ops ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden.txt)
add_test(NAME cpu_pool COMMAND test_cpu_pool)
add_test(NAME cpu_moe COMMAND test_cpu_moe)
add_test(NAME dq_traits COMMAND test_dq_traits)
if(DSV4_NATIVE_CPU)
    add_test(NAME native_cpu COMMAND test_native_cpu)
endif()
if(DSV4_WITH_HIP OR DSV4_WITH_CUDA)
    add_executable(test_gpu tests/test_gpu.cpp)
    target_link_libraries(test_gpu PRIVATE dsv4_core)
    add_test(NAME gpu COMMAND test_gpu)
endif()
# -DDSV4_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89   (usa src/gpu_cuda.cu al posto di src/gpu.cpp)
# -DSTRATA_ROOT=/path/to/Strata   (Strata e' solo LETTO: include paths + link, mai modificato)
