"""Launch the Stem Similarity Search interface."""

from pathlib import Path
import subprocess
import sys


APP_PATH = Path(__file__).resolve().parent / "dp3" / "app.py"


def main() -> int:
    """Run the Streamlit app with the current Python interpreter."""
    try:
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(APP_PATH)])
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
