#!/usr/bin/env python3
"""
MoE Profiling Script

Usage:
    python profile_moe.py                    # Basic profiling with table output
    python profile_moe.py --trace            # Generate tensorboard trace
    python profile_moe.py --ops              # Profile individual ops
    python profile_moe.py --detailed         # Show more rows in table

Then for trace viewing:
    tensorboard --logdir=./profile_traces
"""

import argparse
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from megablocks import ops
from megablocks.layers.arguments import Arguments
from megablocks.layers.moe import MoE, clear_load_balancing_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Profile MoE operations")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num-experts", type=int, default=64, help="Number of experts")
    parser.add_argument("--top-k", type=int, default=8, help="Top-k experts per token")
    parser.add_argument("--iterations", type=int, default=20, help="Profiling iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--trace", action="store_true", help="Generate tensorboard trace")
    parser.add_argument("--ops", action="store_true", help="Profile individual ops")
    parser.add_argument("--detailed", action="store_true", help="Show more detail in table")
    parser.add_argument("--backward", action="store_true", help="Include backward pass")
    return parser.parse_args()


def create_moe_layer(hidden_size, ffn_hidden_size, num_experts, top_k):
    """Create a standard megablocks MoE layer."""
    init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.02)
    args = Arguments(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        moe_num_experts=num_experts,
        moe_capacity_factor=1,
        moe_top_k=top_k,
        init_method=init_method,
    )
    moe = MoE(args)
    moe.cuda().half()
    return moe, args


def profile_individual_ops(args):
    """Profile individual megablocks operations."""
    print("\n" + "=" * 70)
    print("PROFILING INDIVIDUAL OPS")
    print("=" * 70)

    sl = args.batch_size * args.seq_len
    hs = args.hidden_size
    ne = args.num_experts
    top_k = args.top_k
    block_size = 128

    # Setup tensors
    x = torch.randn((sl, hs), device="cuda", dtype=torch.float16)
    top_expert = torch.randint(0, ne, (sl * top_k,), device="cuda", dtype=torch.int32)

    # Sort
    sort_end_bit = max(int(math.ceil(math.log2(ne))), 1)
    bin_ids, indices = ops.sort(top_expert, sort_end_bit)

    # Histogram
    tokens_per_expert = ops.histogram(top_expert, ne)

    # Cumsum
    padded_tokens_per_expert = ops.round_up(tokens_per_expert, block_size)
    padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
    bins = ops.inclusive_cumsum(tokens_per_expert, 0)

    # Weights for scatter
    weights = torch.rand((sl * top_k,), device="cuda", dtype=torch.float16)

    # Gather output for scatter test
    gathered = ops.padded_gather(x, indices, bin_ids, bins, padded_bins, top_k)

    ops_to_profile = {
        "sort": lambda: ops.sort(top_expert, sort_end_bit),
        "histogram": lambda: ops.histogram(top_expert, ne),
        "cumsum": lambda: ops.inclusive_cumsum(tokens_per_expert, 0),
        "padded_gather": lambda: ops.padded_gather(x, indices, bin_ids, bins, padded_bins, top_k),
        "padded_scatter": lambda: ops.padded_scatter(gathered, indices, bin_ids, weights, bins, padded_bins, top_k),
    }

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    for op_name, op_fn in ops_to_profile.items():
        # Warmup
        for _ in range(args.warmup):
            op_fn()
        torch.cuda.synchronize()

        with torch.profiler.profile(activities=activities) as prof:
            for _ in range(args.iterations):
                op_fn()
            torch.cuda.synchronize()

        avg_cuda_time = sum(
            e.cuda_time_total for e in prof.key_averages()
        ) / args.iterations / 1000  # Convert to ms

        print(f"{op_name:20s}: {avg_cuda_time:.3f} ms")

    print()


def profile_moe_layer(args):
    """Profile full MoE layer forward (and optionally backward)."""
    print("\n" + "=" * 70)
    print("PROFILING FULL MOE LAYER")
    print(f"Config: batch={args.batch_size}, seq={args.seq_len}, hidden={args.hidden_size}")
    print(f"        experts={args.num_experts}, top_k={args.top_k}")
    print("=" * 70 + "\n")

    moe, moe_args = create_moe_layer(
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.hidden_size * 4,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )

    x = torch.randn(
        args.seq_len, args.batch_size, args.hidden_size,
        device="cuda", dtype=torch.float16,
        requires_grad=args.backward,
    )

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    def run_forward():
        out = moe(x)
        clear_load_balancing_loss()
        return out

    def run_forward_backward():
        out = moe(x)
        loss = out.sum()
        loss.backward()
        moe.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        clear_load_balancing_loss()
        return out

    run_fn = run_forward_backward if args.backward else run_forward
    mode_str = "forward+backward" if args.backward else "forward only"

    # Warmup
    print(f"Warming up ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        run_fn()
    torch.cuda.synchronize()

    # Profile
    print(f"Profiling ({args.iterations} iterations, {mode_str})...")

    if args.trace:
        # Generate tensorboard trace
        trace_handler = torch.profiler.tensorboard_trace_handler("./profile_traces")
        with torch.profiler.profile(
            activities=activities,
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(args.iterations):
                run_fn()
                prof.step()

        print("\nTrace saved to ./profile_traces/")
        print("View with: tensorboard --logdir=./profile_traces")
    else:
        # Table output
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(args.iterations):
                run_fn()
        torch.cuda.synchronize()

        row_limit = 50 if args.detailed else 25

        print("\n--- CUDA Time by Kernel ---")
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=row_limit,
        ))

        print("\n--- CPU Time (shows sync points) ---")
        print(prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=15,
        ))

        # Summary stats
        total_cuda_time = sum(e.cuda_time_total for e in prof.key_averages())
        avg_cuda_time = total_cuda_time / args.iterations / 1000

        print(f"\n--- Summary ---")
        print(f"Average iteration time: {avg_cuda_time:.3f} ms")

        # Look for CPU-GPU sync indicators
        print("\n--- Potential Sync Points (CPU waiting) ---")
        for event in prof.key_averages():
            # High CPU time relative to CUDA time can indicate sync
            if event.cpu_time_total > 1000 and event.cuda_time_total < event.cpu_time_total * 0.1:
                print(f"  {event.key}: CPU={event.cpu_time_total/1000:.2f}ms, CUDA={event.cuda_time_total/1000:.2f}ms")


def profile_with_cuda_events(args):
    """Simple timing with CUDA events for quick comparison."""
    print("\n" + "=" * 70)
    print("SIMPLE TIMING (CUDA Events)")
    print("=" * 70 + "\n")

    moe, moe_args = create_moe_layer(
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.hidden_size * 4,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )

    x = torch.randn(
        args.seq_len, args.batch_size, args.hidden_size,
        device="cuda", dtype=torch.float16,
        requires_grad=args.backward,
    )

    def run_fn():
        out = moe(x)
        if args.backward:
            out.sum().backward()
            moe.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad = None
        clear_load_balancing_loss()

    # Warmup
    for _ in range(args.warmup):
        run_fn()
    torch.cuda.synchronize()

    # Time with CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(args.iterations):
        start.record()
        run_fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    import numpy as np
    mean_time = np.mean(times)
    std_time = np.std(times)

    mode_str = "forward+backward" if args.backward else "forward only"
    print(f"Mode: {mode_str}")
    print(f"Mean time: {mean_time:.3f} ms")
    print(f"Std time:  {std_time:.3f} ms")
    print(f"Throughput: {args.batch_size * args.seq_len / mean_time * 1000:.0f} tokens/sec")


def main():
    args = parse_args()

    print("=" * 70)
    print("MoE PROFILING")
    print("=" * 70)
    print(f"Config:")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Sequence len:  {args.seq_len}")
    print(f"  Hidden size:   {args.hidden_size}")
    print(f"  Num experts:   {args.num_experts}")
    print(f"  Top-k:         {args.top_k}")
    print(f"  Tokens total:  {args.batch_size * args.seq_len}")
    print(f"  Include bwd:   {args.backward}")

    if args.ops:
        profile_individual_ops(args)

    profile_with_cuda_events(args)
    profile_moe_layer(args)


if __name__ == "__main__":
    main()
