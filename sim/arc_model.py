#!/usr/bin/env python3
"""Arc Model — unified NPU architecture evaluation.

Two dimensions:
  A. Precision:  load GGUF weights → per-block INT4 quant → cos_sim check (GATE)
  B. Performance: MXU timing model → decode tok/s, utilization, DRAM stall

Gate rule: all weight layers must pass cos_sim ≥ 0.97 before performance eval.
"""

import json
import logging
import sys
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "ggml-npu"))
sys.path.insert(0, str(_HERE))

logger = logging.getLogger(__name__)

from q4_dequant import load_weights_from_gguf
from golden_executor import GoldenMXU
from quantize import quantize_int4_per_block
from model_specs import MODELS, get_spec
from fsa_ref import (
    compare_architectures, print_comparison,
    FSAConfig, FSAHardwareModel,
    CaduceusHardwareModel, ArchComparisonReport,
)


@dataclass
class PrecisionReport:
    n_layers: int = 0
    cos_mean: float = 0.0
    cos_min: float = 0.0
    cos_std: float = 0.0
    worst_layer: str = ""
    worst_cos: float = 1.0
    passed: bool = False
    mse_mean: float = 0.0
    mse_min: float = 0.0
    max_abs_error: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "cos_mean": self.cos_mean,
            "cos_min": self.cos_min,
            "cos_std": self.cos_std,
            "worst_layer": self.worst_layer,
            "worst_cos": self.worst_cos,
            "passed": self.passed,
            "mse_mean": self.mse_mean,
            "mse_min": self.mse_min,
            "max_abs_error": self.max_abs_error,
        }


@dataclass
class PerfReport:
    decode_tok_s: float = 0.0
    decode_us_tok: float = 0.0
    mxu_util_pct: float = 0.0
    dram_stall_pct: float = 0.0
    total_mac_g: float = 0.0

    def to_dict(self) -> dict:
        return {
            "decode_tok_s": self.decode_tok_s,
            "decode_us_tok": self.decode_us_tok,
            "mxu_util_pct": self.mxu_util_pct,
            "dram_stall_pct": self.dram_stall_pct,
            "total_mac_g": self.total_mac_g,
        }


@dataclass
class ArcReport:
    model_name: str = ""
    hidden: int = 0
    intermediate: int = 0
    layers: int = 0
    precision: Optional[PrecisionReport] = None
    perf: Optional[PerfReport] = None
    passed: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "hidden": self.hidden,
            "intermediate": self.intermediate,
            "layers": self.layers,
            "precision": self.precision.to_dict() if self.precision else None,
            "perf": self.perf.to_dict() if self.perf else None,
            "passed": self.passed,
            "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ArcModel:
    """Unified architecture evaluator: precision gate → performance model.

    Supports configurable quantization schemes:
      - per-channel: one scale per output channel
      - per-block:   group_size=128 along K (TensorRT/GPTQ standard)
      - both:        compare both schemes with per-layer breakdown

    Architecture comparison (v2.0):
      - run_arch_comparison(): compare CaduceusCore vs FSA inline-softmax
      - Head-to-head: area, latency, MAC utilization, flexibility
    """

    COS_THRESHOLD = 0.96   # per-layer vector cosine minimum (INT4: ~0.97 expected)
    SCHEMES = {
        "per-channel": {"name": "Per-Channel INT4", "desc": "1 scale/output channel"},
        "per-block":   {"name": "Per-Block INT4 (g=128)", "desc": "TensorRT/GPTQ standard"},
    }

    # Known model configs: (qkv, hidden, intermediate, layers, num_heads, kv_heads)
    MODELS = MODELS

    def __init__(self, config_path: str = "config/npu_config.yaml"):
        from npu_sim import NPUSimulator
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = _HERE / config_file
        self.sim = NPUSimulator(str(config_file))
        self.mxu = GoldenMXU()
        self.rng = np.random.RandomState(42)

    def _run_precision(self, weights: dict, scheme: str) -> PrecisionReport:
        """Run precision validation for one quantization scheme."""
        from quantize import quantize_int4_per_channel, quantize_int4_per_block

        use_block = (scheme == "per-block")
        cos_values = []
        mse_values = []
        max_abs_values = []
        worst_layer = ""
        worst_cos = 1.0
        n_tested = 0

        for name, W_f32 in sorted(weights.items()):
            if W_f32.ndim != 2 or "weight" not in name.lower():
                continue
            K, N = W_f32.shape
            if K < 64 or N < 64:
                continue

            act = self.rng.randint(-128, 128, size=K, dtype=np.int8).reshape(1, K)
            golden = act.astype(np.float32) @ W_f32.astype(np.float32)
            g_vec = golden[0, :].astype(np.float64)
            ng = np.linalg.norm(g_vec)

            if use_block:
                packed, scales, _ = quantize_int4_per_block(W_f32, group_size=128)
                result = self.mxu.matmul_int4_per_block(act, packed, scales, 1, K, N, group_size=128)
            else:
                packed, scales, _ = quantize_int4_per_channel(W_f32)
                result = self.mxu.matmul_int4_per_channel(act, packed, scales, 1, K, N)

            t_vec = result[0, :].astype(np.float64)
            nt = np.linalg.norm(t_vec)
            cos_val = float(np.dot(g_vec, t_vec)) / max(ng * nt, 1e-16)
            cos_values.append(cos_val)

            diff = g_vec - t_vec
            mse_values.append(float(np.mean(diff * diff)))
            max_abs_values.append(float(np.max(np.abs(diff))))

            if cos_val < worst_cos:
                worst_cos = cos_val
                worst_layer = name
            n_tested += 1

        cos_arr = np.array(cos_values)
        mse_arr = np.array(mse_values) if mse_values else np.array([0.0])
        max_abs_arr = np.array(max_abs_values) if max_abs_values else np.array([0.0])
        return PrecisionReport(
            n_layers=n_tested,
            cos_mean=float(np.mean(cos_arr)),
            cos_min=float(np.min(cos_arr)),
            cos_std=float(np.std(cos_arr)),
            worst_layer=worst_layer,
            worst_cos=worst_cos,
            passed=worst_cos >= self.COS_THRESHOLD,
            mse_mean=float(np.mean(mse_arr)),
            mse_min=float(np.min(mse_arr)),
            max_abs_error=float(np.max(max_abs_arr)),
        )

    def evaluate(self, gguf_path: str,
                 scheme: str = "per-block",
                 model_spec: Optional[tuple] = None) -> ArcReport:
        """Run full Arc evaluation: precision → performance.

        Args:
            gguf_path: path to Q4_K GGUF model
            scheme: "per-channel", "per-block", or "both"
            model_spec: optional (qkv, hidden, intermediate, layers, num_heads, kv_heads) tuple

        Returns:
            ArcReport with precision and performance dimensions.
        """
        name = Path(gguf_path).stem
        report = ArcReport(model_name=name)

        # Auto-detect model spec
        spec = model_spec
        if spec is None:
            for key, val in MODELS.items():
                if val.model_type != "llm":
                    continue
                if key.replace(".", "").replace("-", "") in name.lower().replace(".", "").replace("-", ""):
                    spec = val
                    break
        if spec is None:
            report.error = f"Unknown model spec for {name}. Pass model_spec=(QKV,H,I,L,NH,KV)"
            return report

        report.hidden, report.intermediate, report.layers = spec[1], spec[2], spec[3]

        # ── Load weights ────────────────────────────────────────────
        logger.info(f"\n{'='*60}")
        logger.info("Arc Model — Precision Gate")
        logger.info(f"{'='*60}")

        t0 = time.time()
        try:
            weights = load_weights_from_gguf(gguf_path)
        except Exception as e:
            report.error = f"GGUF load failed: {e}"
            logger.error(report.error)
            return report
        logger.info(f"Loaded {len(weights)} tensors in {time.time()-t0:.1f}s")

        # ── A. Precision ────────────────────────────────────────────
        schemes_to_run = list(self.SCHEMES.keys()) if scheme == "both" else [scheme]

        scheme_results = {}
        try:
            for s in schemes_to_run:
                t1 = time.time()
                pr = self._run_precision(weights, s)
                dt = time.time() - t1
                scheme_results[s] = pr

                label = self.SCHEMES[s]["name"]
                icon = "PASS" if pr.passed else "FAIL"
                logger.info(f"\n  [{label}] {pr.n_layers} layers in {dt:.1f}s")
                logger.info(f"    cos_sim: mean={pr.cos_mean:.6f}  min={pr.cos_min:.6f}  std={pr.cos_std:.6f}")
                logger.info(f"    mse:     mean={pr.mse_mean:.6e}  min={pr.mse_min:.6e}")
                logger.info(f"    max_abs_error: {pr.max_abs_error:.6e}")
                logger.info(f"    worst:   {pr.worst_layer[-60:]}  cos={pr.worst_cos:.6f}")
                logger.info(f"    gate:    {icon} (threshold={self.COS_THRESHOLD})")
        except Exception as e:
            report.error = f"Precision evaluation failed: {e}"
            logger.error(report.error)
            return report

        if scheme == "both":
            pc_pr = scheme_results["per-channel"]
            pb_pr = scheme_results["per-block"]
            delta = pb_pr.cos_mean - pc_pr.cos_mean
            winner = "per-block" if delta > 0 else "per-channel"
            logger.info(f"\n  Comparison: per-block − per-channel = {delta:+.4f} cos_sim")
            logger.info(f"  → {winner} wins  (min: {pc_pr.cos_min:.4f} vs {pb_pr.cos_min:.4f})")
            best = pb_pr if delta > 0 else pc_pr
            report.precision = best
            pr = best
        else:
            pr = scheme_results[scheme]
            report.precision = pr

        # ── B. Performance Model ────────────────────────────────────
        if not pr.passed:
            logger.warning("\n  → Skipping performance eval: precision gate not met")
            report.passed = False
            return report

        logger.info(f"\n{'='*60}")
        logger.info("Arc Model — Performance")
        logger.info(f"{'='*60}")

        H, I, L = report.hidden, report.intermediate, report.layers
        num_heads = spec[4]
        kv_heads = spec[5]
        head_dim = spec[0] // num_heads
        qkv = num_heads * head_dim
        kv = kv_heads * head_dim

        trace = []
        for layer in range(L):
            trace.append((1, H, qkv, layer, "Q_proj"))
            trace.append((1, H, kv,  layer, "K_proj"))
            trace.append((1, H, kv,  layer, "V_proj"))
            trace.append((1, qkv, H,  layer, "O_proj"))
            trace.append((1, H, I,   layer, "FFN_gate"))
            trace.append((1, H, I,   layer, "FFN_up"))
            trace.append((1, I, H,   layer, "FFN_down"))

        try:
            perf = self.sim.simulate_decode(trace)
        except Exception as e:
            report.error = f"Performance simulation failed: {e}"
            logger.error(report.error)
            return report
        total_mac = sum(m * k * n for m, k, n, _, _ in trace) * 2

        mxu_us = perf.decode_breakdown.get("MXU", 0)
        dma_us = perf.decode_breakdown.get("DMA (stall)", 0)
        total_us = perf.decode_per_token_us

        pf = PerfReport(
            decode_tok_s=perf.decode_tok_per_s,
            decode_us_tok=total_us,
            mxu_util_pct=mxu_us / total_us * 100 if total_us > 0 else 0,
            dram_stall_pct=dma_us / total_us * 100 if total_us > 0 else 0,
            total_mac_g=total_mac / 1e9,
        )
        report.perf = pf
        report.passed = True

        logger.info(f"\n  Config: {H} hidden, {I} intermediate, {L} layers")
        logger.info(f"  Decode: {pf.decode_tok_s:.1f} tok/s  ({pf.decode_us_tok:.0f} us/tok)")
        logger.info(f"  MXU:    {pf.mxu_util_pct:.1f}% util")
        logger.info(f"  DRAM:   {pf.dram_stall_pct:.1f}% stall")
        logger.info(f"  MAC:    {pf.total_mac_g:.2f}G")

        return report

    def run_arch_comparison(
        self, model_name: str, spec: tuple,
        seq_q: int = 1, seq_kv: int = 1024,
    ) -> ArchComparisonReport:
        """Run architecture comparison: CaduceusCore vs FSA.

        Args:
            model_name: e.g. 'Qwen2.5-3B'
            spec: (QKV, H, I, L, NH, KVH) tuple
            seq_q: query sequence length (1 = decode, 128 = prefill)
            seq_kv: KV cache length
        """
        qkv, hidden, intermediate, layers = spec[0], spec[1], spec[2], spec[3]
        num_heads = spec[4]
        kv_heads = spec[5]
        head_dim = qkv // num_heads

        logger.info(f"\n{'='*60}")
        logger.info(f"Arc Model — Architecture Comparison")
        logger.info(f"{'='*60}")
        logger.info(f"  Model: {model_name} ({hidden}h, {layers}L, {num_heads}NH, {kv_heads}KVH)")

        report = compare_architectures(
            model_name=model_name,
            seq_q=seq_q, seq_kv=seq_kv, head_dim=head_dim,
            num_heads=num_heads, num_kv_heads=kv_heads, num_layers=layers,
        )
        print_comparison(report)
        return report

    def run_arch_comparison_table(
        self, model_specs: dict, seq_q: int = 1, seq_kv: int = 1024,
    ) -> list[ArchComparisonReport]:
        """Run architecture comparison across multiple models.

        Args:
            model_specs: {name: (QKV, H, I, L, NH, KVH), ...}
        """
        reports = []
        for name, spec in model_specs.items():
            reports.append(self.run_arch_comparison(name, spec, seq_q, seq_kv))
        return reports

    def print_table(self, report: ArcReport):
        """Print final summary table."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Arc Model — Final Report: {report.model_name}")
        logger.info(f"{'='*80}")

        if report.error:
            logger.error(f"  Error: {report.error}")
            return

        pr = report.precision
        if pr is None:
            logger.warning("  No precision data (evaluation incomplete)")
            return

        pf = report.perf
        logger.info(f"{'Dimension':<15} {'Metric':<22} {'Value':>15}")
        logger.info(f"{'-'*15} {'-'*22} {'-'*15}")
        logger.info(f"{'Precision':<15} {'layers':<22} {pr.n_layers:>15d}")
        logger.info(f"{'Precision':<15} {'cos_sim (mean)':<22} {pr.cos_mean:>15.6f}")
        logger.info(f"{'Precision':<15} {'cos_sim (min)':<22} {pr.cos_min:>15.6f}")
        logger.info(f"{'Precision':<15} {'cos_sim (std)':<22} {pr.cos_std:>15.6f}")
        logger.info(f"{'Precision':<15} {'mse (mean)':<22} {pr.mse_mean:>15.6e}")
        logger.info(f"{'Precision':<15} {'mse (min)':<22} {pr.mse_min:>15.6e}")
        logger.info(f"{'Precision':<15} {'max_abs_error':<22} {pr.max_abs_error:>15.6e}")
        logger.info(f"{'Precision':<15} {'gate passed':<22} {str(pr.passed):>15}")
        if pf:
            logger.info(f"{'Performance':<15} {'decode tok/s':<22} {pf.decode_tok_s:>15.1f}")
            logger.info(f"{'Performance':<15} {'decode us/tok':<22} {pf.decode_us_tok:>15.0f}")
            logger.info(f"{'Performance':<15} {'MXU utilization':<22} {pf.mxu_util_pct:>14.1f}%")
            logger.info(f"{'Performance':<15} {'DRAM stall':<22} {pf.dram_stall_pct:>14.1f}%")
        logger.info(f"{'='*80}")
        logger.info(f"  Overall: {'PASS' if report.passed else 'FAIL'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Arc Model evaluation")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Precision + performance evaluation
    p_eval = sub.add_parser("evaluate", help="Precision gate → performance model")
    p_eval.add_argument("--model", required=True, help="Path to GGUF model")
    p_eval.add_argument("--scheme", default="per-block",
                        choices=["per-channel", "per-block", "both"],
                        help="INT4 quantization scheme (default: per-block)")
    p_eval.add_argument("--spec", help="Model spec: QKV,H,I,L,NH,KV")

    # Architecture comparison
    p_arch = sub.add_parser("compare", help="Architecture comparison: CaduceusCore vs FSA")
    p_arch.add_argument("--model", default="Qwen2.5-3B", help="Model name")
    p_arch.add_argument("--spec", default="2048,2560,9728,28,32,8",
                        help="Model spec: QKV,H,I,L,NH,KVH")
    p_arch.add_argument("--seq-q", type=int, default=1, help="Query seq length (1=decode)")
    p_arch.add_argument("--seq-kv", type=int, default=1024, help="KV cache length")
    p_arch.add_argument("--all", action="store_true", help="Compare all known models")

    args = parser.parse_args()

    if args.mode == "evaluate":
        spec = None
        if args.spec:
            spec = tuple(int(x) for x in args.spec.split(","))
        arc = ArcModel()
        report = arc.evaluate(args.model, scheme=args.scheme, model_spec=spec)
        arc.print_table(report)

    elif args.mode == "compare":
        arc = ArcModel()
        if args.all:
            llm_specs = {
                name: val for name, val in MODELS.items()
                if val.model_type == "llm" and len(val) >= 6
            }
            arc.run_arch_comparison_table(llm_specs, seq_q=args.seq_q, seq_kv=args.seq_kv)
        else:
            spec = tuple(int(x) for x in args.spec.split(","))
            arc.run_arch_comparison(args.model, spec, args.seq_q, args.seq_kv)
