#!/usr/bin/env python3
"""Validate GPU-to-GPU NVLink P2P bandwidth on the local machine.

This is an environment gate, not part of the prototype. It answers one question:
can two GPUs move data directly over NVLink, and at what speed? Use it to tell a
hardware problem (no P2P, PCIe-only bandwidth) apart from a software limitation.

Context: the Mooncake Store rejects the ``nvlink_intra`` protocol at init
(``unsupported_protocol``), which is a Store build limitation, not a hardware one.
This script confirms the NVLink path itself is healthy: on the 8x A100 box it
measures about 270 GB/s GPU0 to GPU1, full NVLink speed. A PCIe-only path or a host
bounce would be roughly 20 GB/s.

Requires torch with CUDA and at least two visible GPUs, so run it where both are
visible, for example inside the project image:

    docker run --rm --gpus '"device=0,1"' mloss-vllm-kvcache:mooncake \\
      -lc 'python3 scripts/check_nvlink_p2p.py'
"""

from __future__ import annotations

import argparse
import time


def measure(src: int, dst: int, size_mb: int, iters: int) -> float:
    """Return the GB/s of a device-to-device copy from ``src`` to ``dst``."""
    import torch

    elems = size_mb * 1024 * 1024 // 4  # float32 bytes
    x = torch.empty(elems, dtype=torch.float32, device=f"cuda:{src}")
    gb = x.numel() * 4 / 1e9

    x.to(f"cuda:{dst}")  # warmup; also triggers P2P enablement
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        x.to(f"cuda:{dst}")
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / iters
    return gb / seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GPU-to-GPU NVLink P2P bandwidth")
    parser.add_argument("--src", type=int, default=0, help="Source GPU index")
    parser.add_argument("--dst", type=int, default=1, help="Destination GPU index")
    parser.add_argument("--size-mb", type=int, default=1024, help="Transfer size per copy, MB")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations")
    parser.add_argument("--nvlink-floor", type=float, default=100.0,
                        help="GB/s above which the path is judged NVLink rather than PCIe")
    args = parser.parse_args()

    import torch

    count = torch.cuda.device_count()
    print(f"visible GPUs: {count}")
    if count <= max(args.src, args.dst):
        raise SystemExit(
            f"need GPUs {args.src} and {args.dst} visible; only {count} present. "
            "Run with both GPUs exposed (e.g. --gpus '\"device=0,1\"')."
        )

    peer = torch.cuda.can_device_access_peer(args.src, args.dst)
    print(f"can_device_access_peer({args.src},{args.dst}): {peer}")

    bw = measure(args.src, args.dst, args.size_mb, args.iters)
    print(f"GPU{args.src}->GPU{args.dst} copy: {bw:.1f} GB/s "
          f"({args.size_mb} MB x {args.iters} iters)")

    if bw >= args.nvlink_floor:
        print(f"VERDICT: NVLink P2P active ({bw:.0f} GB/s >= {args.nvlink_floor:.0f} GB/s).")
    else:
        print(f"VERDICT: no NVLink P2P; {bw:.0f} GB/s looks like PCIe or a host bounce. "
              "Check `nvidia-smi topo -m` and `nvidia-smi nvlink --status`.")


if __name__ == "__main__":
    main()
