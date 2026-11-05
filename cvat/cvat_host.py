"""Where the CVAT server is: used by every script in this folder.

Priority:
  1. CVAT_HOST (and CVAT_PORT) from the environment, e.g. CVAT_HOST=http://localhost CVAT_PORT=8181
  2. a running ngrok tunnel to port CVAT_PORT (default 8181), found through ngrok's local API
  3. http://localhost:8181
"""
import json
import os
import urllib.request


def resolve_cvat_host():
    port = int(os.environ.get("CVAT_PORT", "8181"))
    host = os.environ.get("CVAT_HOST")
    if host:
        return host, port
    try:
        with urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=2) as r:
            for t in json.loads(r.read())["tunnels"]:
                if str(port) in t.get("config", {}).get("addr", ""):
                    print(f"ngrok tunnel found: {t['public_url']}")
                    return t["public_url"], None
    except Exception:
        pass
    print(f"CVAT at http://localhost:{port}")
    return "http://localhost", port
