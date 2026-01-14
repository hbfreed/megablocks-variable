#!/usr/bin/env python3
"""
MoE Profiling Script - Find training bottlenecks

Usage:
    python profile_moe.py                     # Profile both dMoE and variable MoE
    python profile_moe.py --num-experts 128   # Custom expert count

Outputs:
    - Console table showing where time is spent
    - moe_trace.json for detailed visualization (open in chrome://tracing)
"""

import argparse
from functools import partial

import torch

from gpt_model import GPT, GPTConfig
from megablocks.layers.arguments import Arguments
from megablocks.layers.dmoe import dMoE


def parse_args():
    parser = argparse.ArgumentParser(description="Profile MoE training bottlenecks")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--variable-only", action="store_true", help="Only profile variable MoE"
    )
    parser.add_argument("--dmoe-only", action="store_true", help="Only profile dMoE")
    parser.add_argument("--mlp-impl", type=str, default="grouped", choices=["grouped", "sparse"],
                        help="MLP implementation for dMoE (grouped=default, sparse=stk ops)")
    return parser.parse_args()


def profile_dmoe(args, num_tokens):
    """Profile MegaBlocks dMoE (uniform expert sizes)."""
    print("\n" + "=" * 70)
    print("PROFILING: MegaBlocks dMoE (uniform experts)")
    print("=" * 70)

    init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.02)
    moe_args = Arguments(
        hidden_size=args.hidden_size,
        ffn_hidden_size=256,  # Per-expert FFN size (must match variable MoE)
        moe_num_experts=args.num_experts,
        moe_capacity_factor=0,  # Dropless
        moe_top_k=args.top_k,
        init_method=init_method,
        mlp_impl=args.mlp_impl,
    )
    print(f"MLP implementation: {args.mlp_impl}")
    dmoe = dMoE(moe_args)
    dmoe.cuda().bfloat16()

    x = torch.randn(
        args.seq_len,
        args.batch_size,
        args.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    def train_step():
        out = dmoe(x)
        output = out[0] if isinstance(out, tuple) else out
        loss = output.sum()
        loss.backward()
        dmoe.zero_grad(set_to_none=True)
        x.grad = None

    # Warmup
    for _ in range(args.warmup):
        train_step()
    torch.cuda.synchronize()

    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(args.iterations):
        start.record()
        train_step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean_time = sum(times) / len(times)
    print(f"→ Mean step time: {mean_time:.3f} ms")
    print(f"→ Throughput: {num_tokens / mean_time * 1000:.0f} tokens/sec")

    # Detailed profiling for dMoE too
    print(f"\nDetailed profiling...")
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
    ) as prof:
        for _ in range(args.iterations):
            train_step()
    torch.cuda.synchronize()

    print("\n" + "=" * 70)
    print("TOP BOTTLENECKS (by CUDA time)")
    print("=" * 70)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    return mean_time


def profile_variable_moe(args, num_tokens):
    """Profile variable-size MoE layer only (fair comparison)."""
    print("\n" + "=" * 70)
    print("PROFILING: Variable-size MoE layer (your implementation)")
    print("=" * 70)

    from gpt_model import GPTConfig, MoEMLP

    config = GPTConfig(
        sequence_len=args.seq_len,
        n_embd=args.hidden_size,
        use_moe=True,
        expert_sizes=[(args.num_experts, 256)],  # Must be divisible by 128
        num_active_experts=args.top_k,
    )
    print(f"Expert config: {config.expert_sizes}")

    moe = MoEMLP(config)
    moe.cuda().bfloat16()
    moe.train()

    x = torch.randn(
        args.batch_size,
        args.seq_len,
        args.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    def train_step():
        out, aux_loss, f_i = moe(x)
        loss = out.sum()
        loss.backward()
        moe.zero_grad(set_to_none=True)
        x.grad = None

    # Warmup
    for _ in range(args.warmup):
        train_step()
    torch.cuda.synchronize()

    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(args.iterations):
        start.record()
        train_step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean_time = sum(times) / len(times)
    print(f"→ Mean step time: {mean_time:.3f} ms")
    print(f"→ Throughput: {num_tokens / mean_time * 1000:.0f} tokens/sec")

    # Detailed profiling
    print(f"\nDetailed profiling...")
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(args.iterations):
            train_step()
    torch.cuda.synchronize()

    trace_path = "./moe_trace.json"
    prof.export_chrome_trace(trace_path)

    print("\n" + "=" * 70)
    print("TOP BOTTLENECKS (by CUDA time)")
    print("=" * 70)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    print(f"\n→ Trace saved to {trace_path}")

    return mean_time


def main():
    args = parse_args()
    num_tokens = args.batch_size * args.seq_len

    print("=" * 70)
    print("MoE TRAINING PROFILER")
    print("=" * 70)
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Sequence len:  {args.seq_len}")
    print(f"  Hidden size:   {args.hidden_size}")
    print(f"  Num experts:   {args.num_experts}")
    print(f"  Top-k:         {args.top_k}")
    print(f"  Total tokens:  {num_tokens}")

    dmoe_time = None
    var_time = None

    if not args.variable_only:
        dmoe_time = profile_dmoe(args, num_tokens)

    if not args.dmoe_only:
        var_time = profile_variable_moe(args, num_tokens)

    # Summary
    if dmoe_time and var_time:
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        print(
            f"  dMoE (uniform):     {dmoe_time:.3f} ms  ({num_tokens / dmoe_time * 1000:.0f} tok/s)"
        )
        print(
            f"  Variable MoE:       {var_time:.3f} ms  ({num_tokens / var_time * 1000:.0f} tok/s)"
        )
        print(f"  Slowdown:           {var_time / dmoe_time:.2f}x")


if __name__ == "__main__":
    main()
