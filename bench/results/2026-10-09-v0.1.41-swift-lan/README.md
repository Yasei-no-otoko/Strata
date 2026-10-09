# Swift 1.5 LAN access on Windows

On 2026-10-09 (Asia/Tokyo), the installed Strata 0.1.41 server for Huihui
Swift 1.5 IQ3_XXS was started on port 8081 for the LAN. The current Ethernet
address is `192.168.0.4`, so the web UI is at `http://192.168.0.4:8081/`
and the OpenAI-compatible base URL is `http://192.168.0.4:8081/v1`.
The existing local UI remains at `http://127.0.0.1:8081/`.

`tools/start_swift_lan.ps1` reads `STRATA_API_KEY` and `STRATA_LAN_HOST`
from the ignored local `.env` file. The generated 256-bit key is held only
in that file and the server process environment; it is not on the command
line or in this report. The file has an owner-only access rule. Clients use
the key with `Authorization: Bearer` or `x-api-key`. The launcher overrides
the bind address at startup, while the existing model config remains set to
`127.0.0.1` and is byte-identical to its backup. Starting the original
`run-swift-huihui-iq3_xxs.bat` retains its local-only setting.

To restart LAN mode after stopping the current server, from the repository
folder run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\start_swift_lan.ps1
```

The host listens on `0.0.0.0` so both LAN and loopback addresses work. This
also includes other active interfaces, such as Tailscale; the API key is
required there too. The existing Windows Firewall rule for Python 3.12
allows inbound connections on Private networks; the Ethernet profile is
Private. The Wi-Fi profile is Public. No firewall setting was changed.

| Check | Result |
| --- | --- |
| Swift and image capability | Loaded, images enabled, Strata 0.1.41 |
| Listener | `0.0.0.0:8081` |
| `/health` via LAN and loopback addresses | Both OK |
| Web UI via LAN address | HTTP 200 |
| `/v1/models` without key or with incorrect key | HTTP 401 in both cases |
| `/v1/models` with correct key via LAN and loopback | HTTP 200 in both cases |
| `/v1/chat/completions` via LAN address | Exact `LAN ready.` response |
| Existing Swift model config | SHA-256 unchanged |
| Firewall and credential file | Private Python rule active; one owner-only file access rule |

The LAN request was made from this PC to its LAN address. A second physical
device was not available for testing, so reachability from another machine
remains unverified. An attempted WSL check could not start because its VHDX
file is missing. The key and raw logs are kept only under the ignored local
files; [verification.json](verification.json) contains no credential value.
