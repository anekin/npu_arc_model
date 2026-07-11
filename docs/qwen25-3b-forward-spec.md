# Qwen2.5-3B 36-Layer Forward Pass Test Specification

> **Version:** 1.0 | **Date:** 2026-07-06
> **Status:** Draft Specification (not executable)
> **Scope:** Define the golden reference, tensor shapes, tolerances, and execution methodology for the full 36-layer Qwen2.5-3B forward pass on CaduceusCore RTL.

---

## 1. Model Identity

| Field | Value |
|-------|-------|
| Model | Qwen/Qwen2.5-3B-Instruct-GGUF |
| GGUF File | `Qwen2.5-3B-Instruct-Q4_K_M.gguf` |
| SHA256 | Obtain via `sha256sum` after download from HuggingFace. Example: `wget https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf` then compute. |
| Source | `https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF` |
| Quantization | Q4_K_M (per-block INT4, group_size=128) |

### 1.1 Model Architecture Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_hidden_layers` | 36 | Number of transformer blocks |
| `hidden_size` | 2048 | Embedding dimension |
| `intermediate_size` | 11008 | FFN intermediate dimension (SwiGLU) |
| `num_attention_heads` | 16 | Query heads |
| `num_key_value_heads` | 16 | Key/value heads (full attention, no GQA reduction) |
| `head_dim` | 128 | `hidden_size / num_attention_heads = 2048 / 16` |
| `q_dim` | 2048 | `num_attention_heads * head_dim` |
| `k_dim` | 2048 | `num_key_value_heads * head_dim` |
| `v_dim` | 2048 | `num_key_value_heads * head_dim` |
| `vocab_size` | 151936 | Vocabulary size |
| `max_position_embeddings` | 32768 | Maximum sequence length (RoPE) |
| `rms_norm_eps` | 1e-6 | RMS normalization epsilon |
| `rope_theta` | 1000000.0 | RoPE base frequency |

### 1.2 Weight Tensors per Layer (GGUF keys)

Each layer `blk.{N}` contains 7 weight tensors:

| Tensor | Shape | Elements | GGUF Key Example (N=0) |
|--------|-------|----------|------------------------|
| Q_proj.weight | (2048, 2048) | 4,194,304 | `blk.0.attn_q.weight` |
| K_proj.weight | (2048, 2048) | 4,194,304 | `blk.0.attn_k.weight` |
| V_proj.weight | (2048, 2048) | 4,194,304 | `blk.0.attn_v.weight` |
| O_proj.weight | (2048, 2048) | 4,194,304 | `blk.0.attn_output.weight` |
| gate_proj.weight | (11008, 2048) | 22,544,384 | `blk.0.ffn_gate.weight` |
| up_proj.weight | (11008, 2048) | 22,544,384 | `blk.0.ffn_up.weight` |
| down_proj.weight | (2048, 11008) | 22,544,384 | `blk.0.ffn_down.weight` |
| **Per layer total** | | **84,759,364** | |

With 36 layers plus embedding + output norm: **~3.05B parameters** total.

---

## 2. Per-Layer Operation Chain

Each of the 36 layers follows the same 17-operation sequence, derived from the verified FM-SOC-027 blk.0 17-op chain.

### 2.1 Single-Layer Op Sequence (17 ops, building block FM-SOC-027)

The sequence below uses the canonical Qwen2.5-3B shapes. INT4 weights are stored per-block (g=128) with FP32 scale factors; the MXU engine handles dequantization internally via scale buffers.

| Op # | Name | Opcode | Engine | Input Shape | Output Shape | Key Tensor | Dimensions (M, K, N) |
|------|------|--------|--------|-------------|--------------|------------|----------------------|
| 00 | RMSNorm pre-attn | RMSNORM | SFU | (1, 2048) FP16 | (1, 2048) FP16 | hidden_states | elements=2048 |
| 01 | Q_proj MMUL | MMUL | MXU | Act (1, 2048) INT8, Wgt (2048, 2048) INT4 | (1, 2048) INT32 | W_Q | M=1, K=2048, N=2048 |
| 02 | K_proj MMUL | MMUL | MXU | Act (1, 2048) INT8, Wgt (2048, 2048) INT4 | (1, 2048) INT32 | W_K | M=1, K=2048, N=2048 |
| 03 | V_proj MMUL | MMUL | MXU | Act (1, 2048) INT8, Wgt (2048, 2048) INT4 | (1, 2048) INT32 | W_V | M=1, K=2048, N=2048 |
| 04 | RoPE | ROPE | SFU | Q (1, 2048) FP16, K (1, 2048) FP16 | Q_rot (1, 2048) FP16, K_rot (1, 2048) FP16 | sin/cos LUT | q_len=2048, k_len=2048 |
| 05 | attn_score MMUL | MMUL | MXU | Q_rot (16, 128) INT8, K^T (128, 16) INT4 | (16, 16) INT32 per head | | M=16, K=128, N=16 |
| 06 | attn_softmax | SOFTMAX | SFU | (16×16) FP16 per head group | (16×16) FP16 per head group | | elements=256 |
| 07 | attn_weight MMUL | MMUL | MXU | attn_probs (16, 16) INT8, V (16, 128) INT4 | (1, 2048) INT32 concat | | M=16, K=16, N=128 |
| 08 | O_proj MMUL | MMUL | MXU | Act (1, 2048) INT8, Wgt (2048, 2048) INT4 | (1, 2048) INT32 | W_O | M=1, K=2048, N=2048 |
| 09 | VRESID pre-attn | VRESID | Vector | hidden (1, 2048) FP16 + attn_out (1, 2048) INT32 | (1, 2048) INT32 | | elements=2048 |
| 10 | RMSNorm post-attn | RMSNORM | SFU | (1, 2048) FP16 | (1, 2048) FP16 | | elements=2048 |
| 11 | gate MMUL | MMUL | MXU | Act (1, 2048) INT8, Wgt (2048, 11008) INT4 | (1, 11008) INT32 | W_gate | M=1, K=2048, N=11008 |
| 12 | up MMUL | MMUL | MXU | Act (1, 2048) INT8, Wgt (2048, 11008) INT4 | (1, 11008) INT32 | W_up | M=1, K=2048, N=11008 |
| 13 | SiLU | SILU | SFU | (1, 11008) FP16 | (1, 11008) FP16 | | elements=11008 |
| 14 | VMUL gate×up | VMUL | Vector | gate (1, 11008) INT32, up (1, 11008) INT32 | (1, 11008) INT32 | | elements=11008 |
| 15 | down MMUL | MMUL | MXU | Act (1, 11008) INT8, Wgt (11008, 2048) INT4 | (1, 2048) INT32 | W_down | M=1, K=11008, N=2048 |
| 16 | VRESID post-FFN | VRESID | Vector | hidden (1, 2048) FP16 + ffn_out (1, 2048) INT32 | (1, 2048) INT32 | | elements=2048 |

### 2.2 Engine Dispatch Summary (per layer)

| Engine | Count | Ops |
|--------|:-----:|-----|
| MXU | 9 | Q_proj, K_proj, V_proj, attn_score, attn_weight, O_proj, gate, up, down |
| SFU | 5 | RMSNorm (×2), RoPE, SOFTMAX, SiLU |
| Vector | 3 | VRESID (×2), VMUL |
| **Total** | **17** | |

### 2.3 36-Layer Full Forward Pass

Total operations across all 36 layers: **36 × 17 = 612 ops**.

Each layer i (0 ≤ i < 36):
- Reads `hidden_states` from previous layer (or embedding layer for i=0)
- Executes the 17-op chain from §2.1
- Produces `layer_output` (shape (1, 2048) INT32), which becomes the input RMSNorm for layer i+1
- Layer 35 output is the final hidden state, optionally followed by output RMSNorm + lm_head

---

## 3. Per-Layer Tensor Shapes Table (36 Layers)

For each layer N (0-35), the operation sequence, input shapes, and output shapes are **identical in structure** but differ in the **specific weight values** loaded from GGUF keys `blk.{N}.attn_*.weight`, `blk.{N}.ffn_*.weight`.

The table below uses a compact notation. All dimensions are M (batch=1), K (input dim), N (output dim) unless otherwise noted.

| Layer Index | Ops | Attention Dims | FFN Dims | Total MMUL Tiles | Key GGUF Tensors |
|:-----------:|:---:|:---------------|:---------|:----------------:|------------------|
| Layer 0 | 17-op chain | qkvo=(2048,2048) | gate/up=(2048,11008), down=(11008,2048) | 3,136 | blk.0.attn_{q,k,v,output}.weight, blk.0.ffn_{gate,up,down}.weight |
| Layer 1 | 17-op chain | same | same | 3,136 | blk.1.* |
| Layer 2 | 17-op chain | same | same | 3,136 | blk.2.* |
| Layer 3 | 17-op chain | same | same | 3,136 | blk.3.* |
| Layer 4 | 17-op chain | same | same | 3,136 | blk.4.* |
| Layer 5 | 17-op chain | same | same | 3,136 | blk.5.* |
| Layer 6 | 17-op chain | same | same | 3,136 | blk.6.* |
| Layer 7 | 17-op chain | same | same | 3,136 | blk.7.* |
| Layer 8 | 17-op chain | same | same | 3,136 | blk.8.* |
| Layer 9 | 17-op chain | same | same | 3,136 | blk.9.* |
| Layer 10 | 17-op chain | same | same | 3,136 | blk.10.* |
| Layer 11 | 17-op chain | same | same | 3,136 | blk.11.* |
| Layer 12 | 17-op chain | same | same | 3,136 | blk.12.* |
| Layer 13 | 17-op chain | same | same | 3,136 | blk.13.* |
| Layer 14 | 17-op chain | same | same | 3,136 | blk.14.* |
| Layer 15 | 17-op chain | same | same | 3,136 | blk.15.* |
| Layer 16 | 17-op chain | same | same | 3,136 | blk.16.* |
| Layer 17 | 17-op chain | same | same | 3,136 | blk.17.* |
| Layer 18 | 17-op chain | same | same | 3,136 | blk.18.* |
| Layer 19 | 17-op chain | same | same | 3,136 | blk.19.* |
| Layer 20 | 17-op chain | same | same | 3,136 | blk.20.* |
| Layer 21 | 17-op chain | same | same | 3,136 | blk.21.* |
| Layer 22 | 17-op chain | same | same | 3,136 | blk.22.* |
| Layer 23 | 17-op chain | same | same | 3,136 | blk.23.* |
| Layer 24 | 17-op chain | same | same | 3,136 | blk.24.* |
| Layer 25 | 17-op chain | same | same | 3,136 | blk.25.* |
| Layer 26 | 17-op chain | same | same | 3,136 | blk.26.* |
| Layer 27 | 17-op chain | same | same | 3,136 | blk.27.* |
| Layer 28 | 17-op chain | same | same | 3,136 | blk.28.* |
| Layer 29 | 17-op chain | same | same | 3,136 | blk.29.* |
| Layer 30 | 17-op chain | same | same | 3,136 | blk.30.* |
| Layer 31 | 17-op chain | same | same | 3,136 | blk.31.* |
| Layer 32 | 17-op chain | same | same | 3,136 | blk.32.* |
| Layer 33 | 17-op chain | same | same | 3,136 | blk.33.* |
| Layer 34 | 17-op chain | same | same | 3,136 | blk.34.* |
| Layer 35 | 17-op chain | same | same | 3,136 | blk.35.* |

### 3.1 Tile Count Derivation (per layer)

| MMUL Op | M | K | N | Tiles (64×64 MXU) |
|---------|---|----|----|:------------------:|
| Q_proj | 1 | 2048 | 2048 | (1×64)×(2048×64)×(2048×64) = 1×32×32 = 1024 |
| K_proj | 1 | 2048 | 2048 | 1×32×32 = 1024 |
| V_proj | 1 | 2048 | 2048 | 1×32×32 = 1024 |
| attn_score | 16 | 128 | 16 | 1×2×1 = 2 (batched) |
| attn_weight | 16 | 16 | 128 | 1×1×2 = 2 (batched) |
| O_proj | 1 | 2048 | 2048 | 1024 |
| gate | 1 | 2048 | 11008 | 1×32×172 = 5504 |
| up | 1 | 2048 | 11008 | 5504 |
| down | 1 | 11008 | 2048 | 1×172×32 = 5504 |
| **Total** | | | | **21,712 tiles** |

Across 36 layers: **21,712 × 36 = 781,632 tiles** total.

---

## 4. Tolerance Thresholds

### 4.1 Per-Engine Tolerances

| Engine | Metric | Threshold | Notes |
|--------|--------|:---------:|-------|
| MXU (INT4 per-block) | Bit-exact | 0 LSB | `matmul_int4_per_block` output must match RTL INT32 output exactly |
| MXU (INT32 path) | Bit-exact | 0 LSB | `matmul_int32` output identical |
| SFU (FP16 path) | Cosine similarity | ≥ 0.999 | `cos_sim(golden_sfu, rtl_sfu)` |
| SFU (FP16 path) | Max relative error | ≤ 1e-4 | `max_rel_err = max(|a-b|/max(|a|,|b|, 1e-8))` |
| Vector (INT32) | Bit-exact | 0 LSB | ADD, MUL, REDUCE, RESID |
| Vector (CONV) | Bit-exact | 0 LSB | INT32→FP16 matches numpy float16 |
| INTER-LAYER propagation | Max relative error | ≤ 1e-3 | Accumulated error across 36 layers must stay bounded |

### 4.2 Per-Layer Output Tolerance

Each layer's final output (VRESID post-FFN, op 16) shall be compared against the Func Model golden reference:

```
cos_sim(layer_output_golden, layer_output_rtl)  >= 0.999
max_rel_err(layer_output_golden, layer_output_rtl) <= 1e-4
```

Per-op intermediate outputs (all 17 ops) must also be compared individually to prevent error masking.

### 4.3 36-Layer End-to-End Tolerance

The final hidden state (layer 35 output) must meet:

```
cos_sim(final_hidden_golden, final_hidden_rtl) >= 0.995
max_rel_err(final_hidden_golden, final_hidden_rtl) <= 1e-2
```

The relaxed tolerance accounts for FP16 accumulation across 36 layers. If all per-layer comparisons pass their individual thresholds, the end-to-end tolerance is a derived guarantee.

---

## 5. Golden Reference

### 5.1 Func Model Golden `.npz` Path

The golden reference for all 36 layers is generated by the Func Model and stored as:

```
rtl/test_vectors/soc_e2e/qwen25-3b-36layer/expected.npz
```

Generation command (planned):

```bash
PYTHONPATH=sim python sim/gen_soc_rtl_vectors.py --case qwen25-3b-36layer
```

### 5.2 `.npz` Format

`expected.npz` contains per-layer goldens (one key per layer) plus aggregate:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `layer_{N}_output` | (1, 2048) | int32 | VRESID output of layer N after all 17 ops |
| `layer_{N}_op_{M}` | varies | varies | Per-op intermediate output for each of 17 ops |
| `final_hidden` | (1, 2048) | int32 | Layer 35 output |
| `metadata` | — | str (JSON) | Generation parameters, SHA256 of GGUF weight extract |

### 5.3 Expected File Size

| Component | Size |
|-----------|:----:|
| Per-layer VRESID output | 36 × 2048 × 4 = 288 KB |
| Per-op intermediate outputs | 36 × 17 × avg 8 KB ≈ 4.9 MB |
| Metadata + overhead | ~50 KB |
| **Total expected.npz (compressed)** | **~2-3 MB** |

### 5.4 Golden Reference Generation Methodology

1. Load GGUF `Qwen2.5-3B-Instruct-Q4_K_M.gguf` via `q4_dequant.load_weights_from_gguf()`
2. For each layer 0..35:
   a. Extract 7 weight tensors, quantize INT4 per-block (g=128), pack tile-major
   b. Execute 17-op chain through `GoldenExecutor` (bit-exact FP16/INT32)
   c. Save per-op intermediate output and final VRESID output
3. Write all 36 layers to `expected.npz` with per-layer and per-op keys
4. Compute SHA256 of the generated `.npz` for integrity verification

---

## 6. Execution Methodology

### 6.1 Execution Models

| Mode | Platform | Engine | Purpose |
|------|----------|--------|---------|
| **Func Model** | Python (`FuncModel`) | GoldenExecutor | Generate golden reference; debug op sequences |
| **Spike E2E** | Spike + firmware | Ibex CPU + GoldenExecutor | Validate firmware dispatch, doorbell, IRQ path |
| **Cocotb** | VCS + Cocotb | RTL (MXU+SFU+Vector) | Full RTL verification per layer |
| **RTL SoC** | VCS + Ibex RTL | Full SoC | Final sign-off; all 36 layers |

### 6.2 Layer-by-Layer Strategy

Execute layers sequentially. Each layer completes all 17 ops before the next layer starts. Intermediate results are saved per-op for isolation.

```python
for layer in range(36):
    for op in manifest["ops"]:
        rtl_out, passed = bridge.run_step(op)
        assert passed, f"Layer {layer}, Op {op['idx']}: {op['name']} FAILED"
```

### 6.3 Per-Layer Verification Steps

1. **Pre-condition**: SRAM and DRAM initialized with layer N weights and previous layer output
2. **Execute**: 17-op chain through doorbell dispatch or direct MMIO
3. **Compare**: Per-op output vs golden reference within per-engine tolerances
4. **Accumulate**: Track cos_sim and max_rel_err across all 17 ops
5. **Layer gate**: All 17 ops must PASS before proceeding to layer N+1
6. **Evidence**: Each layer saves a per-op comparison report

### 6.4 Anti-Vacuous Gating

Following FM-SOC-007/008 methodology, each layer test MUST include a corrupted-weight verification:

- Corrupt one byte of a random weight tensor in the layer
- Re-run forward pass for that layer
- Verify that at least one op output differs from the original golden
- Prevents vacuous PASS from stale SRAM or unloaded weights

### 6.5 Build / Rebuild Protocol

| Step | Script / Command | Output |
|------|-----------------|--------|
| 1. Generate 36-layer golden | `gen_soc_rtl_vectors.py --case qwen25-3b-36layer` | `rtl/test_vectors/soc_e2e/qwen25-3b-36layer/expected.npz` |
| 2. Verify Func Model dry-run | `PYTHONPATH=sim python -c "run_36layer_chain()"` | Console: 612/612 PASS |
| 3. Generate RTL test vectors | `gen_soc_rtl_vectors.py` | Per-layer `input.npz` |
| 4. Run VCS Cocotb layer N | `./simv_multimodule +layer={N}` | Per-layer PASS/FAIL |
| 5. Regression | `run_fm_soc_all.sh --extended qwen25-3b-36layer` | Summary report |

### 6.6 Integration with FM-SOC Case Hierarchy

| Case ID | Coverage | Status |
|---------|----------|:------:|
| FM-SOC-027 | blk.0 single-layer 17-op chain (single-tile MMUL workaround) | ✅ PASS |
| FM-SOC-032 | 28-block chain (scaled weights, per-block fingerprint) | ✅ PASS |
| FM-SOC-10X | Full host→PCIe→doorbell→firmware→17-op blk.0 chain | ✅ PASS |
| **FM-SOC-036** | **Full 36-layer Qwen2.5-3B forward pass** (planned) | 📋 Spec defined |
| **FM-SOC-037** | **36-layer anti-vacuous: corrupted layer 17 weight** (planned) | 📋 Spec defined |

---

## 7. Per-Layer Weight Volume

Each layer processes **84,759,364 weight elements** across 7 tensors. At Q4_K_M (4-bit with g=128 FP32 scales), the packed weight footprint per layer is:

| Tensor | Elements | Packed Size (INT4) | Scales Size (FP32) | Total |
|--------|:--------:|:------------------:|:------------------:|:-----:|
| Q_proj | 4,194,304 | 2,097,152 B | 32 × 2048 × 4 = 262,144 B | 2.25 MB |
| K_proj | 4,194,304 | 2,097,152 B | 262,144 B | 2.25 MB |
| V_proj | 4,194,304 | 2,097,152 B | 262,144 B | 2.25 MB |
| O_proj | 4,194,304 | 2,097,152 B | 262,144 B | 2.25 MB |
| gate_proj | 22,544,384 | 11,272,192 B | 86 × 11008 × 4 = 3,786,752 B | 14.32 MB |
| up_proj | 22,544,384 | 11,272,192 B | 3,786,752 B | 14.32 MB |
| down_proj | 22,544,384 | 11,272,192 B | 11008 × 16 × 4 = 704,512 B | 11.42 MB |
| **Per layer** | **84,759,364** | **42,205,184 B** | **~9.3 MB** | **~49 MB** |

> **Note**: At 49 MB per layer and 4 MB SRAM, tile streaming via DMA is mandatory. Each MMUL is split into 64×64 tiles and streamed through the MXU. The Func Model's `tile_scheduler.py` manages the double-buffered DMA schedule.

---

## 8. Known Limitations / Gaps

| # | Gap | Impact | Mitigation |
|---|-----|--------|------------|
| 1 | 4 MB SRAM insufficient for full layer weights | Requires tile streaming | Func Model `tile_scheduler.py` handles this; verify in Cocotb |
| 2 | Single-tile MMUL workaround in FM-SOC-027 | Not representative of real MMUL throughput | FM-SOC-036 must use multi-tile path |
| 3 | No lm_head (unembedding) in current scope | Tests stop at final hidden state | Add in Phase 6 if needed |
| 4 | Quantization error accumulates across layers | Layer 35 tolerance is looser than per-layer | Model-level accuracy (perplexity) measured separately in Arc Model |

---

## 9. References

- **Blk.0 building block**: FM-SOC-027 (17-op chain, 9 MMUL + 5 SFU + 3 Vector) — PASS on RTL SoC + Spike
- **28-layer chain**: FM-SOC-032 (28-block, per-block weight scaling, fingerprint isolation) — PASS
- **Host→PCIe E2E**: FM-SOC-10X (17-op blk.0 via PCIe TLP, firmware dispatch) — PASS
- **Func Model golden path**: `sim/golden_executor.py` (GoldenMXU, GoldenSFU, GoldenVector)
- **Tile scheduler**: `sim/tile_scheduler.py` (DMA double-buffered streaming)
- **RTL development plan**: `docs/rtl_development_plan.md` §4.4.1 (single-layer op sequence)
- **Verification lessons**: `docs/caduceus-verification-lessons.md` (14 principles including anti-vacuous, per-op isolation)
- **SoC FM test plan**: `rtl/testcase-list-soc-fm.md` (FM-SOC-001 through FM-SOC-10X)
