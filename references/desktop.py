"""Optional desktop window: runs the Streamlit app headless and shows it via pywebview.

Not a new server: Streamlit's own server is bound to 127.0.0.1 on a free port.
Needs `pip install -r requirements-desktop.txt` (pywebview; WebView2 on Windows).
Streamlit output goes to logs/streamlit.log because pythonw has no console.
"""
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / 'app.py'
LOG = ROOT / 'logs' / 'streamlit.log'
HOST = '127.0.0.1'
TITLE = 'RP Studio'


def free_port() -> int:
    """A port the OS reports free right now (Streamlit fails loudly if it is taken first)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def build_command(port: int) -> list[str]:
    return [sys.executable, '-m', 'streamlit', 'run', str(APP),
            '--server.address', HOST, '--server.port', str(port),
            '--server.headless', 'true', '--browser.gatherUsageStats', 'false',
            '--client.toolbarMode', 'minimal', '--server.runOnSave', 'false',
            '--server.fileWatcherType', 'none']


def start_server(port: int, log_file) -> subprocess.Popen:
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)  # no console flash on Windows
    return subprocess.Popen(build_command(port), cwd=ROOT, stdin=subprocess.DEVNULL,
                            stdout=log_file, stderr=subprocess.STDOUT, creationflags=flags)


def wait_for_server(process: subprocess.Popen, port: int, timeout: float = 30) -> None:
    """Return once *our* process answers the health check; raise if it exits or stalls."""
    url = f'http://{HOST}:{port}/_stcore/health'
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Streamlit exited during startup (code {process.returncode}).')
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(0.2)
    raise TimeoutError(f'Streamlit did not become ready within {timeout:.0f}s.')


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()  # Windows: immediate; posts are already fsynced on append
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def show_error(message: str) -> None:
    """Surface failures even under pythonw, where stderr is invisible."""
    print(message, file=sys.stderr)
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, message, TITLE, 0x10)


def main() -> int:
    try:
        import webview
    except ImportError:
        show_error('pywebview is not installed.\n\n'
                   'Run: pip install -r requirements-desktop.txt\n'
                   'Or use run_app.bat to open RP Studio in your browser.')
        return 1
    LOG.parent.mkdir(exist_ok=True)
    port = free_port()
    with LOG.open('w', encoding='utf-8') as log_file:
        process = start_server(port, log_file)
        try:
            wait_for_server(process, port)
            webview.create_window(TITLE, f'http://{HOST}:{port}', width=1600, height=900,
                                  min_size=(900, 600), resizable=True)
            webview.start(debug=False)
        except Exception as exc:
            show_error(f'RP Studio could not start: {exc}\n\nDetails: {LOG}')
            return 1
        finally:
            stop_server(process)
    return 0


if __name__ == '__main__':
    sys.exit(main())
