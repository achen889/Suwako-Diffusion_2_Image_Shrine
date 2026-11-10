"""Optional desktop window: runs the Streamlit app headless and shows it via pywebview.

Not a new server: Streamlit's own server is bound to 127.0.0.1 on a free port.
Needs `pip install -r requirements-desktop.txt` (pywebview; WebView2 on Windows).
Streamlit output is relayed to the launcher console and a timestamped file in logs/.
"""
import os
import socket
import subprocess
import sys
from datetime import datetime
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / 'app.py'
LOG_DIR = ROOT / 'logs'
HOST = '127.0.0.1'
TITLE = 'Suwako Diffusion'
PROJECT_PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'


def ensure_project_interpreter() -> bool:
    if not PROJECT_PYTHON.is_file():
        show_error(
            f"Suwako's project virtual environment is missing: {PROJECT_PYTHON}\n\n"
            "Create it with: py -3.12 -m venv .venv\n"
            "Then install CUDA-enabled PyTorch and requirements-desktop.txt into .venv."
        )
        return False
    active = os.path.normcase(os.path.abspath(sys.executable))
    expected = os.path.normcase(os.path.abspath(PROJECT_PYTHON))
    if active != expected:
        os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
    return True


def check_cuda() -> bool:
    result = subprocess.run(
        [sys.executable, '-m', 'suwako.cli', '--check-cuda'], cwd=ROOT
    )
    if result.returncode:
        show_error(
            f"CUDA preflight failed under {sys.executable}.\n\n"
            "Repair the project .venv before using Suwako's local generator."
        )
        return False
    return True


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
    process = subprocess.Popen(
        build_command(port), cwd=ROOT, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=flags,
        text=True, encoding='utf-8', errors='replace', bufsize=1,
    )
    relay = threading.Thread(
        target=relay_server_output, args=(process.stdout, log_file),
        name='suwako-log-relay', daemon=True,
    )
    process._log_relay_thread = relay
    relay.start()
    return process


def relay_server_output(stream, log_file) -> None:
    """Write child output immediately to disk and relay it to an available console."""
    for line in stream:
        try:
            log_file.write(line)
            log_file.flush()
        except OSError as exc:
            if sys.stderr is not None:
                print(f'Could not persist Streamlit log line: {exc}', file=sys.stderr)
        if sys.stdout is not None:
            try:
                print(line, end='', file=sys.stdout, flush=True)
            except (OSError, ValueError):
                pass


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
    if process.poll() is None:
        process.terminate()  # Windows: immediate; output is flushed by the relay.
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    relay = getattr(process, '_log_relay_thread', None)
    if relay:
        relay.join(timeout=2)


def show_error(message: str) -> None:
    """Surface failures even under pythonw, where stderr is invisible."""
    if sys.stderr is not None:
        print(message, file=sys.stderr)
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, message, TITLE, 0x10)


def main() -> int:
    if not ensure_project_interpreter() or not check_cuda():
        return 1
    try:
        import webview
    except ImportError:
        show_error('pywebview is not installed.\n\n'
                   'Run: pip install -r requirements-desktop.txt\n'
                   'Or use run_studio.bat to open Suwako in your browser.')
        return 1
    LOG_DIR.mkdir(exist_ok=True)
    port = free_port()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = LOG_DIR / f'streamlit_{stamp}_{os.getpid()}.log'
    with log_path.open('a', encoding='utf-8') as log_file:
        process = start_server(port, log_file)
        try:
            wait_for_server(process, port)
            webview.create_window(TITLE, f'http://{HOST}:{port}', width=1600, height=900,
                                  min_size=(900, 600), resizable=True)
            webview.start(debug=False)
        except Exception as exc:
            show_error(f'Suwako Studio could not start: {exc}\n\nDetails: {log_path}')
            return 1
        finally:
            stop_server(process)
    return 0


if __name__ == '__main__':
    sys.exit(main())
