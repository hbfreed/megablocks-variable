#!/usr/bin/env python3
"""
MoE Profiling Script - Find training bottlenecks

Usage:
    python profile_moe.py                     # Profile the variable MoE layer
    python profile_moe.py --num-experts 128   # Custom expert count

Profiles gpt_model.MoEMLP, the STK block-sparse reference layer, which needs the
nanomoe_ops extension. Upstream's uniform dMoE layer, which this script used to
compare against, is no longer part of the package.

Outputs:
    - Console table showing where time is spent
    - moe_trace.json for detailed visualization (open in chrome://tracing)
"""

import argparse

import torch

from gpt_model import GPTConfig, MoEMLP


def parse_args():
    parser = argparse.ArgumentParser(description="Profile MoE training bottlenecks")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--token-rounding",
        choices=["ceil", "nearest"],
        default="ceil",
        help="Per-expert 128-row block rounding policy for the variable MoE",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--variable-only",
        action="store_true",
        help="Accepted for compatibility; the variable MoE is all this profiles",
    )
    return parser.parse_args()


def profile_variable_moe(args, num_tokens):
    """Profile the variable-size MoE layer."""
    print("\n" + "=" * 70)
    print("PROFILING: Variable-size MoE layer")
    print("=" * 70)

    config = GPTConfig(
        sequence_len=args.seq_len,
        n_embd=args.hidden_size,
        use_moe=True,
        expert_sizes=[(args.num_experts, 256)],  # Must be divisible by 128
        num_active_experts=args.top_k,
        token_rounding=args.token_rounding,
    )
    print(
        f"Expert config: {config.expert_sizes}, "
        f"token rounding: {config.token_rounding}",
    )

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

    with torch.no_grad():
        _, routing_stats, _ = moe(x)
    print(
        "Route accounting: "
        f"{float(routing_stats['dropped_route_fraction']) * 100:.2f}% dropped, "
        f"{float(routing_stats['padding_route_fraction']) * 100:.2f}% padded",
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
    print("\nDetailed profiling...")
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

    profile_variable_moe(args, num_tokens)


if __name__ == "__main__":
    main()
