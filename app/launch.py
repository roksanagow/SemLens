"""The ``semlens`` command: start the app in the browser from any folder."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run the Streamlit app; extra arguments go to Streamlit (for example ``--server.port 8502``)."""
    from app.paths import STORAGE
    credentials = Path.home() / ".streamlit" / "credentials.toml"
    if not credentials.exists():
        # Without this file Streamlit stops on first run to ask for an email address.
        credentials.parent.mkdir(parents=True, exist_ok=True)
        credentials.write_text('[general]\nemail = ""\n', encoding="utf-8")
    print(f"SemLens keeps saved corpora and analyses in {STORAGE}", flush=True)
    command = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("app.py")),
               "--browser.gatherUsageStats", "false", "--theme.base", "dark", *sys.argv[1:]]
    try:
        raise SystemExit(subprocess.call(command))
    except KeyboardInterrupt:
        raise SystemExit(0)


if __name__ == "__main__":
    main()
