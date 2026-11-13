import subprocess
import uuid

import pandas as pd
import pytest

from agent import execution_env as ee

# Skip this if we don't have Docker; build the image on first use
@pytest.fixture(scope="module", autouse=True)
def image():
    if not ee.docker_available():
        pytest.skip("Docker is not available")
    error = ee.prepare_sandbox()
    assert error is None, error

def run_in_container(script, memory_mb=1024):
    # Same flags as the sandbox,
    # but a plain script (no audit hook, no import limits)
    # We => check only container
    cmd = ee._docker_base(f"sandbox-test-{uuid.uuid4().hex[:8]}", memory_mb)
    cmd += [ee.IMAGE, "python", "-I", "-c", script]
    return subprocess.run(cmd, capture_output=True, timeout=90)

# container-level isolation
def test_container_has_no_network():
    script = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    print('connected')\n"
        "except OSError:\n"
        "    print('blocked')"
    )
    assert run_in_container(script).stdout.decode().strip() == "blocked"

def test_container_filesystem_is_read_only():
    script = (
        "try:\n"
        "    open('/usr/local/hacked.txt', 'w').write('x')\n"
        "    print('written')\n"
        "except OSError:\n"
        "    print('blocked')"
    )
    assert run_in_container(script).stdout.decode().strip() == "blocked"

def test_container_runs_as_non_root():
    assert run_in_container("import os; print(os.getuid())").stdout.decode().strip() != "0"

def test_container_does_not_see_host_env(monkeypatch):
    monkeypatch.setenv("NRP_API_KEY", "super-secret")
    out = run_in_container("import os; print(os.environ.get('NRP_API_KEY'))")
    assert out.stdout.decode().strip() == "None"

def test_container_memory_limit_is_enforced():
    # touch 2 GB inside a ~1.5 GB container => Fail!
    out = run_in_container("x = b'x' * (2 * 1024**3); print('allocated')")
    assert b"allocated" not in out.stdout

# timeout must stop the container, not just the docker client
def test_timeout_kills_container(monkeypatch):
    monkeypatch.setenv("SANDBOX_BACKEND", "docker")
    df = pd.DataFrame({"a": [1, 2]})
    out = ee.execute_code("while True:\n    pass", df, timeout=3)
    assert not out["success"] and "timed out" in out["error"]
    running = subprocess.run(["docker", "ps", "-q", "--filter", "name=sandbox-"], capture_output=True)
    assert running.stdout.decode().strip() == ""