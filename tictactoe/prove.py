"""Detached job runner for minsat.py on one machine: a queue of solver jobs, each under `timeout` and
`nice`, a few at a time, with a PID file and a status JSON.

    nohup python3 prove.py --jobs jobs.json --dir runs/run1 > runs/run1/runner.log 2>&1 &
    cat runs/run1/status.json

jobs.json is a list of {"name": ..., "args": [minsat.py args], "timeout": seconds}. It is re-read before each job
starts, so jobs can be appended while the runner is alive; finished names (in results.jsonl) are skipped, so a
restart resumes. Each job appends one JSON row to results.jsonl (status SAT/UNSAT/UNKNOWN, or TIMEOUT/ERROR from
here) and writes its log to logs/<name>.log and, if SAT, its net to nets/<name>.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def done_names(path):
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                out.add(json.loads(line)["name"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--nice", type=int, default=19)
    ap.add_argument("--min-free-gb", type=float, default=12.0, help="don't start a job below this MemAvailable")
    a = ap.parse_args()
    a.dir, a.jobs = os.path.abspath(a.dir), os.path.abspath(a.jobs)
    os.makedirs(os.path.join(a.dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(a.dir, "nets"), exist_ok=True)
    res_path = os.path.join(a.dir, "results.jsonl")
    with open(os.path.join(a.dir, "runner.pid"), "w") as f:
        f.write(str(os.getpid()))
    running = {}  # name -> (Popen, t0, job)
    started = now()
    state = {}

    def mem_free_gb():
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable"):
                        return int(line.split()[1]) / 1e6
        except OSError:
            pass
        return 1e9

    def write_status(final=False):
        jobs = json.load(open(a.jobs))
        done = done_names(res_path)
        st = {"state": "done" if final else "running", "pid": os.getpid(), "started": started, "updatedAt": now(),
              "total": len(jobs), "finished": len(done), "running": [
                  {"name": n, "seconds": round(time.time() - t0), "timeout": j["timeout"], "pid": p.pid}
                  for n, (p, t0, j) in running.items()],
              "queued": [j["name"] for j in jobs if j["name"] not in done and j["name"] not in running],
              "memFreeGB": round(mem_free_gb(), 1), "last": state.get("last")}
        tmp = os.path.join(a.dir, "status.json.tmp")
        with open(tmp, "w") as f:
            json.dump(st, f, indent=1)
        os.replace(tmp, os.path.join(a.dir, "status.json"))

    def finish(name, p, t0, job):
        log = os.path.join(a.dir, "logs", name + ".log")
        row = None
        with open(log) as f:
            for line in f:
                if line.startswith("{"):
                    try:
                        row = json.loads(line)
                    except ValueError:
                        pass
        if row is None:
            row = {"status": "TIMEOUT" if p.returncode in (124, -9, 137) else "ERROR", "returncode": p.returncode}
            with open(log) as f:
                rounds = [line for line in f if " round " in line]
            if rounds:
                row["last_round"] = rounds[-1].strip()
        row.update({"name": name, "args": job["args"], "timeout": job["timeout"],
                    "wall": round(time.time() - t0, 1), "finishedAt": now()})
        with open(res_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        state["last"] = {"name": name, "status": row["status"]}

    while True:
        for name, (p, t0, job) in list(running.items()):
            if p.poll() is not None:
                finish(name, p, t0, job)
                del running[name]
        jobs = json.load(open(a.jobs))
        done = done_names(res_path)
        todo = [j for j in jobs if j["name"] not in done and j["name"] not in running]
        while todo and len(running) < a.parallel and mem_free_gb() >= a.min_free_gb:
            j = todo.pop(0)
            log = open(os.path.join(a.dir, "logs", j["name"] + ".log"), "w")
            cmd = ["nice", "-n", str(a.nice), "timeout", "-s", "KILL", str(int(j["timeout"])), sys.executable,
                   os.path.join(HERE, "minsat.py")] + j["args"] + ["--out", os.path.join(a.dir, "nets", j["name"] + ".json")]
            running[j["name"]] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE), time.time(), j)
        write_status()
        if not running and not todo:
            write_status(final=True)
            return 0
        time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())
