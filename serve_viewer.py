"""
AerX Labs — Local viewer server for the CesiumJS 3D mission viewer.

CesiumJS uses Web Workers for its 3D geometry, and browsers refuse to load those
from a `file://` page — so the Cesium viewer (`*_cesium.html`) must be opened over
http. This helper starts a tiny local web server rooted at the project folder and
opens the most recently generated Cesium viewer in your default browser.

Run it (from D:\\Projects\\AerXlabs\\AerX_Prototype):

    .venv\\Scripts\\python.exe serve_viewer.py

Leave it running while you use the viewer; press Ctrl+C to stop. (The Plotly
`*_3d.html` viewer works from a plain file double-click and doesn't need this.)
"""

import functools
import glob
import http.server
import os
import socketserver
import sys
import webbrowser

PORT = 8777
ROOT = os.path.dirname(os.path.abspath(__file__))


def newest_cesium() -> str:
    files = glob.glob(os.path.join(ROOT, "results", "**", "*_cesium.html"), recursive=True)
    if not files:
        return ""
    latest = max(files, key=os.path.getmtime)
    return os.path.relpath(latest, ROOT).replace(os.sep, "/")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else newest_cesium()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=ROOT)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
        base = f"http://127.0.0.1:{PORT}/"
        print("=" * 70)
        print(f"  Serving {ROOT}")
        print(f"  at      {base}")
        if target:
            url = base + target
            print(f"  Opening {url}")
            webbrowser.open(url)
        else:
            print("  No *_cesium.html found yet — run `python main.py ghats` first,")
            print(f"  then browse to {base}")
        print("  Press Ctrl+C to stop.")
        print("=" * 70)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
