import os
import subprocess
import sys
import time
import webbrowser
import socket
from threading import Thread

def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def stream_process(stream):
    for line in iter(stream.readline, ''):
        print(line, end='')

base_path = os.path.dirname(__file__)
app_path = os.path.join(base_path, "app.py")

# Find a free port
port = find_free_port()
url = f"http://localhost:{port}"

print(f" Starting Streamlit on port {port} (URL: {url})")

# Launch Streamlit process with pipes
process = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", str(port)],
    cwd=base_path,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# Start threads to stream stdout and stderr
Thread(target=stream_process, args=(process.stdout,)).start()
Thread(target=stream_process, args=(process.stderr,)).start()

# Wait a bit for server to start then open browser
time.sleep(5)
webbrowser.open(url)

# Wait for process to finish
process.wait()
