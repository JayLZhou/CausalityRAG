"""A2/A3: run one command on a dedicated GPU and record its peak memory and wall time.

Polls the GPU's used memory every 0.2 s (nothing else runs on that GPU), subtracts
the idle baseline, and appends {name, peak_mib, seconds, n_queries, ...} to a JSONL.
Usage: measure_gpu.py --gpu G --name NAME --n N --out OUT -- CMD ...
"""
import argparse
import json
import subprocess
import threading
import time


def used_mib(gpu):
    out = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                                   "--format=csv,noheader,nounits"], text=True)
    return int(out.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    base = used_mib(args.gpu)
    peak = [base]
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            peak[0] = max(peak[0], used_mib(args.gpu))
            time.sleep(0.2)

    t = threading.Thread(target=poll, daemon=True)
    t.start()
    t0 = time.time()
    rc = subprocess.call(cmd)
    seconds = time.time() - t0
    stop.set()
    t.join()
    row = {"name": args.name, "gpu": args.gpu, "returncode": rc, "baseline_mib": base,
           "peak_mib": peak[0] - base, "seconds": round(seconds, 1), "n_queries": args.n,
           "seconds_per_query_incl_load": round(seconds / args.n, 2), "cmd": " ".join(cmd)}
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
