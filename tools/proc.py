"""Run a subprocess with a timeout that actually kills everything it spawned.

subprocess.run(timeout=...) kills only the direct child. Gradle's runServer forks a
Minecraft server JVM that keeps running and holds the output pipe open, so run() blocks
forever waiting to read to EOF. That is the hours-long hang. Here we put the child in its
own process group and, on timeout, kill the whole group so the pipes close.
"""
import os
import signal
import subprocess
import time


def run_capped(cmd, cwd, timeout: int, env=None) -> tuple[int, str, bool]:
    """Return (returncode, combined_output, timed_out). Never blocks past ~timeout+15s."""
    posix = os.name != "nt"
    popen_kwargs = dict(cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, env=env)
    if posix:
        popen_kwargs["start_new_session"] = True  # new process group we can kill as a whole
    else:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(cmd, **popen_kwargs)

    def kill_tree():
        try:
            if posix:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        except (ProcessLookupError, OSError):
            pass

    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out or "", False
    except subprocess.TimeoutExpired:
        kill_tree()
        try:
            # Grandchildren are dead now, so the pipe closes and this returns promptly.
            out, _ = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            kill_tree()
            out = ""
        return -1, out or "", True
    finally:
        if proc.poll() is None:
            kill_tree()
