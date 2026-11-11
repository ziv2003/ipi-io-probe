#!/usr/bin/env python3
"""
Measure what trajectory output actually costs, before optimising anything.

Mimics i-PI's behaviour: one xyz file per ring-polymer bead, one write per
frame per file, persistent handles. Then repeats the same volume of data with
larger, coalesced writes, so you can see whether buffering is worth it on
*your* filesystem.

Four modes are measured, because the honest answer differs by a lot:

  per-frame          write() per frame, no fsync   (what i-PI does by default)
  per-frame+sync     write() + fsync per frame     (i-PI with <flush>1</flush>)
  buffered           coalesce to --chunk-mb, no fsync
  buffered+sync      coalesce to --chunk-mb, fsync per chunk

Without fsync you may only be timing a memcpy into the page cache, so read
all four columns before drawing conclusions.

    python probe.py --path /scratch/$USER/iotest --natoms 192 --beads 32 --frames 2000

Run it on a compute node, on the filesystem you actually write to.
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import time


# --------------------------------------------------------------------------
def make_frame_body(natoms: int) -> str:
    """One xyz frame body, ~12 kB for 192 atoms.

    Built once and reused: string formatting would otherwise pollute the
    timing with interpreter cost that has nothing to do with I/O.
    """
    symbols = ("O", "H", "H")
    rows = []
    for i in range(natoms):
        x = (i % 7) * 1.23456789
        y = (i % 5) * 0.98765432
        z = (i % 3) * 1.11111111
        rows.append(f"{symbols[i % 3]:2s} {x:14.8f} {y:14.8f} {z:14.8f}")
    return "\n".join(rows) + "\n"


def frame_bytes(body: str, natoms: int, step: int, bead: int) -> bytes:
    head = f"{natoms}\nstep={step} bead={bead}\n"
    return (head + body).encode()


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


# --------------------------------------------------------------------------
def run_mode(mode, path, body, natoms, beads, frames, chunk_bytes):
    """Write `frames` frames to `beads` files. Returns (elapsed, latencies)."""
    files = []
    for b in range(beads):
        files.append(open(os.path.join(path, f"pos_{b}.xyz"), "wb"))

    payloads = [[frame_bytes(body, natoms, step, b) for b in range(beads)]
                for step in range(frames)]

    latencies = []
    do_sync_per_frame = mode == "per-frame+sync"
    chunked = mode.startswith("buffered")
    sync_per_chunk = mode == "buffered+sync"

    t_all = time.perf_counter()

    if not chunked:
        for step in range(frames):
            t0 = time.perf_counter()
            for b in range(beads):
                files[b].write(payloads[step][b])
                if do_sync_per_frame:
                    files[b].flush()
                    os.fsync(files[b].fileno())
            latencies.append(time.perf_counter() - t0)
    else:
        pending = [bytearray() for _ in range(beads)]
        for step in range(frames):
            for b in range(beads):
                pending[b].extend(payloads[step][b])
            if sum(len(p) for p in pending) >= chunk_bytes:
                t0 = time.perf_counter()
                for b in range(beads):
                    files[b].write(pending[b])
                    pending[b] = bytearray()
                    if sync_per_chunk:
                        files[b].flush()
                        os.fsync(files[b].fileno())
                latencies.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        for b in range(beads):
            if pending[b]:
                files[b].write(pending[b])
                files[b].flush()
                os.fsync(files[b].fileno())
        latencies.append(time.perf_counter() - t0)

    for f in files:
        f.close()

    return time.perf_counter() - t_all, latencies


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--path", required=True, help="directory to write into")
    ap.add_argument("--natoms", type=int, default=192)
    ap.add_argument("--beads", type=int, default=32,
                    help="number of per-bead files (i-PI writes one per bead)")
    ap.add_argument("--frames", type=int, default=2000,
                    help="output events; each event writes all beads")
    ap.add_argument("--chunk-mb", type=float, default=1.0)
    ap.add_argument("--keep", action="store_true", help="keep the files")
    ap.add_argument("--step-ms", type=float, default=None,
                    help="your measured ms/step, to express I/O as %% of wall time")
    args = ap.parse_args(argv)

    body = make_frame_body(args.natoms)
    one_frame = len(frame_bytes(body, args.natoms, 0, 0))
    total_mb = one_frame * args.beads * args.frames / 1e6
    chunk_bytes = int(args.chunk_mb * 1e6)

    print(f"atoms/file      {args.natoms}")
    print(f"files (beads)   {args.beads}")
    print(f"output events   {args.frames}")
    print(f"frame size      {one_frame / 1024:.1f} kB")
    print(f"total volume    {total_mb:.1f} MB")
    print(f"target          {args.path}")
    print()

    os.makedirs(args.path, exist_ok=True)
    results = {}

    for mode in ("per-frame", "per-frame+sync", "buffered", "buffered+sync"):
        sub = os.path.join(args.path, mode.replace("+", "_"))
        if os.path.isdir(sub):
            shutil.rmtree(sub)
        os.makedirs(sub)

        elapsed, lat = run_mode(mode, sub, body, args.natoms, args.beads,
                                args.frames, chunk_bytes)
        results[mode] = (elapsed, lat)
        print(f"  {mode:16s} done", flush=True)
        if not args.keep:
            shutil.rmtree(sub)

    print()
    print("Latencies below are per *write call*, which is not the same unit for")
    print("every row: buffered modes aggregate several frames into one write. The")
    print("last two columns normalise back to one output event so the tail can be")
    print("compared like for like.")
    print()
    print(f"{'mode':16s} {'total (s)':>10s} {'MB/s':>8s} {'writes':>7s} "
          f"{'p50/ev':>8s} {'p99/ev':>8s} {'max/ev':>8s}")
    print("-" * 76)
    for mode, (elapsed, lat) in results.items():
        per_event = args.frames / max(1, len(lat))
        ms = [v * 1e3 / per_event for v in lat]
        print(f"{mode:16s} {elapsed:10.3f} {total_mb / elapsed:8.1f} "
              f"{len(lat):7d} {pct(ms, 50):8.3f} {pct(ms, 99):8.3f} "
              f"{max(ms):8.3f}")

    base = results["per-frame"][0]
    buffered = results["buffered"][0]
    print()
    print(f"per-frame -> buffered: {base / buffered:.2f}x on total wall time")

    if args.step_ms is not None:
        overhead = base / args.frames * 1e3  # ms per output event
        print(f"I/O per output event: {overhead:.3f} ms "
              f"= {100.0 * overhead / (args.step_ms + overhead):.2f}% of a "
              f"{args.step_ms:.2f} ms step")
        if overhead / args.step_ms < 0.05:
            print("  -> under 5%: buffering is not your problem, tune stride instead")

    if not args.keep:
        print("\n(files removed; pass --keep to inspect them)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
