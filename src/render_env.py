"""
AerX Labs — Matplotlib Rendering Environment Configuration
src/render_env.py

Centralizes Matplotlib backend selection and the Windows Tcl/Tk runtime
wiring so the pipeline runs reliably in two modes:

  * Headless / batch (PNG export, unit tests, CI) — how every `main.py` run
    renders -> forces the non-interactive "Agg" backend. No Tcl/Tk required.

  * Interactive GUI (opt-in, for ad-hoc `plt.show()` exploration)
    -> locates the Tcl/Tk runtime that ships with the CPython install and
       exports TCL_LIBRARY / TK_LIBRARY so the TkAgg backend can start.

IMPORTANT: `configure_matplotlib()` must be called *before* the first
`import matplotlib.pyplot`, because the backend can only be selected prior
to pyplot initialization.
"""

import os
import sys
import glob


def _wire_tcl_tk() -> bool:
    """
    Ensure TCL_LIBRARY / TK_LIBRARY point at a valid Tcl/Tk runtime.

    Returns True if a usable init.tcl was located (or the environment was
    already valid), False otherwise.
    """
    # Respect an already-valid, user-provided configuration.
    existing = os.environ.get("TCL_LIBRARY")
    if existing and os.path.isfile(os.path.join(existing, "init.tcl")):
        return True

    # Candidate roots that ship Tcl/Tk alongside CPython on Windows.
    base = os.path.dirname(sys.executable)          # e.g. .venv/Scripts
    candidates = [
        os.path.join(sys.base_prefix, "tcl"),        # C:\Program Files\Python313\tcl
        os.path.join(os.path.dirname(base), "tcl"),  # venv-local tcl (rare)
        os.path.join(sys.prefix, "tcl"),
    ]

    for root in candidates:
        if not os.path.isdir(root):
            continue
        # Find a tcl8.x directory containing init.tcl.
        for tcl_dir in sorted(glob.glob(os.path.join(root, "tcl8*"))):
            if os.path.isfile(os.path.join(tcl_dir, "init.tcl")):
                os.environ["TCL_LIBRARY"] = tcl_dir
                # Pair the matching tk8.x directory when present.
                tk_dir = tcl_dir.replace("tcl8", "tk8")
                if os.path.isdir(tk_dir):
                    os.environ["TK_LIBRARY"] = tk_dir
                return True
    return False


def configure_matplotlib(interactive: bool) -> bool:
    """
    Select and lock in the Matplotlib backend.

    Parameters:
        interactive: True to request an on-screen GUI window (TkAgg),
                     False for headless PNG rendering (Agg).

    Returns:
        True if an interactive backend is active, False if headless.
    """
    import matplotlib

    if interactive and _wire_tcl_tk():
        try:
            matplotlib.use("TkAgg", force=True)
            return True
        except Exception:
            pass  # Fall through to headless.

    matplotlib.use("Agg", force=True)
    return False
