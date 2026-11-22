#!/usr/bin/env python3
"""measure_load.py – Record live ZMQ load stats (num_running, num_waiting, num_tokens)
while running a steady Poisson load at a target saturation level.

For each (length, sat) cell, this script:
  1. Flushes and warms the node with the same L_docs used by find_msp.py
  2. Starts a Poisson arrival stream at sat% of the known MSP
  3. Subscribes to the node's ZMQ load publisher for STEP_DURATION_S seconds
  4. Records every LoadStat tick to JSONL

This gives real observed num_tokens / num_running_reqs values to calibrate
the proxy health thresholds (token_loaded_threshold, loaded_threshold).

Usage:
  python measure_load.py --length 8192 --sat 80
  python measure_load.py --lengths 2048,8192,16384,24576 --sat 40,80,100

Output: phase-1/load_stats/load_{length}_{sat}_{ts}.jsonl
"""

import argparse
import asyncio
import json
import math
import random
import sys
import time
import threading
from pathlib import Path

import aiohttp
import msgpack
import zmq

HERE = Path(__file__).parent.resolve()

# ── constants (must match find_msp.py) ────────────────────────────────────────
L_DOCS          = 40
SUFFIX_TOKENS   = 128
OUTPUT_TOKENS   = 256
TOKEN_MIN       = 100
TOKEN_MAX       = 50_000

STEP_DURATION_S = 120     # seconds to send load per cell
REST_S          = 15      # drain time between cells
WARM_IN_S       = 10      # wait after load starts before recording

# MSP values from find_msp.py runs
MSP_TABLE = {
    2048:  2.3283,
    8192:  0.75,
    16384: 0.20,
    24576: 0.1875,
}
# ──────────────────────────────────────────────────────────────────────────────


def rand_ids(rng: random.Random, n: int) -> list[int]:
    return [rng.randint(TOKEN_MIN, TOKEN_MAX) for _ in range(n)]


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def load_tokenizer_path(config_path: str) -> str:
    cfg = load_config(config_path)
    model = cfg["model"]
    rel = cfg["models"][model]["tokenizer_path"]
    return str(Path(config_path).parent / rel)


def node_urls(config_path: str, node_id: str, host_override: str | None = None):
    cfg = load_config(config_path)
    nodes = {n["id"]: n for n in cfg["nodes"]}
    n = nodes[node_id]
    host = host_override or n["ip"]
    sglang_url = f"http://{host}:{n['sglang']}"
    load_addr   = f"tcp://{host}:{n['load']}"
    return sglang_url, load_addr


def docs_path(l_docs_dir: str, length: int) -> Path:
    return Path(l_docs_dir) / f"L_docs_{length}.json"


def load_docs(length: int, l_docs_dir: str, tokenizer_path: str, seed: int) -> list[dict]:
    p = docs_path(l_docs_dir, length)
    if p.exists():
        with open(p) as f:
            docs = json.load(f)
        print(f"  Loaded {len(docs)} docs from {p}")
        return docs
    # generate if missing (same logic as find_msp.py)
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(tokenizer_path)
    rng = random.Random(seed)
    docs = []
    for _ in range(L_DOCS):
        ids = rand_ids(rng, length)
        text = tk.decode(ids)
        docs.append({"ids": ids, "text": text})
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(docs, f)
    print(f"  Generated {L_DOCS} docs → {p}")
    return docs


def flush_cache(sglang_url: str):
    import requests
    requests.post(f"{sglang_url}/flush_cache", json={}, timeout=10).raise_for_status()


async def _warm_async(sglang_url: str, docs: list[dict], rng: random.Random):
    connector = aiohttp.TCPConnector(limit=0)
    timeout   = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def _one(doc):
            suffix = rand_ids(rng, SUFFIX_TOKENS)
            payload = {
                "input_ids": doc["ids"] + suffix,
                "sampling_params": {"temperature": 0, "max_new_tokens": OUTPUT_TOKENS},
                "stream": False,
            }
            async with session.post(f"{sglang_url}/generate", json=payload) as resp:
                resp.raise_for_status()
                await resp.read()
        tasks = [asyncio.create_task(_one(doc)) for doc in docs]
        for i, t in enumerate(asyncio.as_completed(tasks)):
            await t
            print(f"  warmed {i+1}/{len(docs)}", end="\r", flush=True)
    print()


def warm_node(sglang_url: str, docs: list[dict], rng: random.Random):
    print(f"  Warming with {len(docs)} docs…")
    asyncio.run(_warm_async(sglang_url, docs, rng))


# ── Poisson load sender (runs in a background thread) ─────────────────────────

async def _send_load_async(sglang_url: str, docs: list[dict], rate: float,
                           duration: float, rng: random.Random):
    connector = aiohttp.TCPConnector(limit=0)
    timeout   = aiohttp.ClientTimeout(total=300)
    deadline  = time.perf_counter() + duration
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        while time.perf_counter() < deadline:
            doc  = docs[rng.randrange(len(docs))]
            seed = rng.randint(0, 2**31)
            suffix = rand_ids(random.Random(seed), SUFFIX_TOKENS)
            payload = {
                "input_ids": doc["ids"] + suffix,
                "sampling_params": {"temperature": 0, "max_new_tokens": OUTPUT_TOKENS,
                                    "ignore_eos": True},
                "stream": True,
            }
            asyncio.create_task(
                session.post(f"{sglang_url}/generate", json=payload).__aenter__()
            )
            gap = -math.log(max(rng.random(), 1e-9)) / rate
            await asyncio.sleep(max(0.0, gap))


def run_load_thread(sglang_url: str, docs: list[dict], rate: float,
                    duration: float, rng: random.Random) -> threading.Thread:
    def _run():
        asyncio.run(_send_load_async(sglang_url, docs, rate, duration, rng))
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


# ── ZMQ subscriber ─────────────────────────────────────────────────────────────

def decode_load_stat(payload: bytes) -> dict | None:
    try:
        raw = msgpack.unpackb(payload, raw=False)
        if not isinstance(raw, list) or len(raw) < 5:
            return None
        if raw[0] != "LoadStat":
            return None
        return {
            "running":    int(raw[1]),
            "waiting":    int(raw[2]),
            "tokens":     int(raw[3]),
            "max_tokens": int(raw[4]),
        }
    except Exception:
        return None


def record_zmq(load_addr: str, duration: float, out_path: Path,
               length: int, sat: int, rate: float):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.connect(load_addr)
    sock.setsockopt(zmq.RCVTIMEO, 2000)  # 2s recv timeout

    deadline = time.time() + duration
    records = []

    print(f"  Subscribing to {load_addr} for {duration}s…")
    while time.time() < deadline:
        try:
            frames = sock.recv_multipart()
            if len(frames) < 3:
                continue
            stat = decode_load_stat(frames[2])
            if stat is None:
                continue
            stat["ts"]     = time.time()
            stat["length"] = length
            stat["sat"]    = sat
            stat["rate"]   = rate
            records.append(stat)
            print(f"  running={stat['running']:3d}  waiting={stat['waiting']:3d}  "
                  f"tokens={stat['tokens']:7d}  max={stat['max_tokens']:7d}", end="\r", flush=True)
        except zmq.Again:
            continue

    print()
    sock.close()
    ctx.term()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    if records:
        tokens_vals  = [r["tokens"]  for r in records]
        running_vals = [r["running"] for r in records]
        waiting_vals = [r["waiting"] for r in records]
        print(f"  tokens:  min={min(tokens_vals)}  max={max(tokens_vals)}  "
              f"mean={sum(tokens_vals)/len(tokens_vals):.0f}")
        print(f"  running: min={min(running_vals)}  max={max(running_vals)}  "
              f"mean={sum(running_vals)/len(running_vals):.1f}")
        print(f"  waiting: min={min(waiting_vals)}  max={max(waiting_vals)}  "
              f"mean={sum(waiting_vals)/len(waiting_vals):.1f}")
        print(f"  ticks recorded: {len(records)}")
    else:
        print("  WARNING: no LoadStat ticks received — check ZMQ address and node is running")

    return records


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).parent.resolve()
    ap.add_argument("--config",    default=str(here / "../config.json"))
    ap.add_argument("--node-id",   default="A", dest="node_id")
    ap.add_argument("--host",      default=None,
                    help="Override node IP (e.g. localhost when on same machine)")
    ap.add_argument("--lengths",   default="2048,8192,16384,24576")
    ap.add_argument("--sat",       default="40,80,100",
                    help="Saturation levels as %% of MSP (comma-separated)")
    ap.add_argument("--l-docs-dir", default=str(here / "data"), dest="l_docs_dir")
    ap.add_argument("--out-dir",   default=str(here / "load_stats"), dest="out_dir")
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    sats    = [int(x) for x in args.sat.split(",")]

    tokenizer_path = load_tokenizer_path(args.config)

    for length in lengths:
        if length not in MSP_TABLE:
            print(f"No MSP entry for length={length}, skipping")
            continue

        docs = load_docs(length, args.l_docs_dir, tokenizer_path, args.seed)
        sglang_url, load_addr = node_urls(args.config, args.node_id, args.host)

        for sat in sats:
            rate = (sat / 100.0) * MSP_TABLE[length]
            run_ts = int(time.time())
            out_path = Path(args.out_dir) / f"load_{length}_{sat}_{run_ts}.jsonl"

            print(f"\n── length={length}  sat={sat}%  rate={rate:.4f} req/s ──")

            # flush and warm
            print("  Flushing cache…")
            flush_cache(sglang_url)
            time.sleep(3)
            rng = random.Random(args.seed)
            warm_node(sglang_url, docs, rng)
            time.sleep(5)  # let KV events propagate

            # start load in background thread
            load_rng = random.Random(args.seed + 1)
            load_thread = run_load_thread(sglang_url, docs, rate,
                                          STEP_DURATION_S + WARM_IN_S, load_rng)

            print(f"  Load started, warming in {WARM_IN_S}s before recording…")
            time.sleep(WARM_IN_S)

            # record ZMQ ticks for STEP_DURATION_S
            records = record_zmq(load_addr, STEP_DURATION_S, out_path,
                                 length, sat, rate)

            load_thread.join(timeout=10)

            print(f"  → {out_path}")
            print(f"  Resting {REST_S}s…")
            time.sleep(REST_S)


if __name__ == "__main__":
    main()
