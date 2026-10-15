# Benchmark results: Confucius4-TTS vs ZONOS2, Vietnamese pilot

Run on 2026-10-15 on the A100 box (`vm-rnd-dev-a100`). This is a **pilot**, 5 items per
block and one voice per item. The numbers show direction, not magnitude; see
[Caveats](#caveats) before quoting any of them.

Raw data sits next to this file in `scores/a100_pilot_20261015/`:

| file | what it holds |
|---|---|
| `summary.md` | the generated per-block table, every metric |
| `confucius4_tts__small.csv`, `zonos2__small.csv` | per-file scores, including each ASR transcript |
| `*.summary.json` | the same aggregates as JSON |
| `run_confucius4.json`, `run_zonos2.json` | provenance: commits, sampling parameters, test-set hash |

## What was run

| | Confucius4-TTS | ZONOS2 |
|---|---|---|
| model commit | `fb43a46` | `71750ef` |
| model path | local checkpoints | `Zyphra/ZONOS2` |
| sampling | server defaults: temp 0.8, top_p 0.8, top_k 30, beams 3, rep 10, nfe 25, cfg 0.7 | server defaults: temp 1.15, topk 106, min_p 0.18, rep 1.2/50 |
| files generated | 45 of 45, 0 failed | 45 of 45, 0 failed |
| checkpoint load | 28 s | 2 to 3 min, 15 GB |

Benchmark commit `7ebb818`. Test set `testset.jsonl`, sha256 `a9d91289c7f1…`, 350 items,
capped to 5 per block for this run. One sample per cell, one voice per item.

Scoring used `PhoWhisper-small` and `whisper-small` rather than the large defaults,
which is proportionate for 45 files and keeps the download near 1 GB instead of 6.5.
UTMOS was skipped.

## Headline

| metric | ZONOS2 | Confucius4 | better |
|---|---|---|---|
| cloning CER, V1 clean reference | 0.041 | **0.008** | Confucius4, 5× |
| cloning CER, V1 noisy reference | 0.038 | **0.015** | Confucius4 |
| long-form CER, V2 | 0.072 | **0.031** | Confucius4 |
| English word recall, CS1 | **0.700** | 0.300 | ZONOS2 |
| CER, CS3 | **0.186** | 0.306 | ZONOS2 |
| WER, CS2 | 0.158 | **0.022** | Confucius4 |
| speaker similarity, mean across blocks | 0.94 | **0.96** | Confucius4 |
| similarity to ground truth, V1 | 0.967 | 0.965 | tie |
| foreign-voice similarity, XL3 | 0.687 | **0.760** | Confucius4 |
| real-time factor | **0.62** | 1.29 | ZONOS2, 2× faster |
| noisy-reference leakage | **-0.003** | 0.025 | ZONOS2 |

Language identification was correct for both models on every cross-lingual item.

## Reading it

**Confucius4 wins intelligibility and speaker fidelity.** A fifth of the character error
on voice cloning, and higher speaker similarity on every block except the noisy-reference
case.

**ZONOS2 wins throughput and reference robustness.** It generates faster than real time,
0.62 against 1.29, so roughly twice the speed. Its similarity does not move when the
reference is noisy, where Confucius4 loses 0.025.

**Code-switching splits, which is the interesting result.** ZONOS2 recovers 70% of CS1's
English words against Confucius4's 30%, and has lower character error on CS3. But
Confucius4 is far better on CS2 word error, 0.022 against 0.158. So ZONOS2 pronounces
English loanwords more recognisably while Confucius4 reads the surrounding Vietnamese
more accurately. This is consistent with the architectures: ZONOS2 takes UTF-8 bytes as
text input, Confucius4 takes Mistral BPE, which fragments Vietnamese at 2.75 to 3.29
tokens per syllable.

Confucius4's failure mode is audible in the transcripts. English words come out with
Vietnamese phonology:

```
deploy   -> "đi ploy"
backlog  -> "bắc lộc"
branch   -> "nhanh"
```

**XL2 is a problem for both.** Word error 0.400 and 0.600 with correct language
identification, so both speak the right language with the wrong words. Inspect those five
items before drawing any cross-lingual conclusion.

## Caveats

- **n=5 per cell.** Intervals are wide. CS1 English recall spans 0.30 to 1.00 for ZONOS2
  and 0.00 to 0.70 for Confucius4, so that 70-versus-30 gap is suggestive, not established.
- **ZONOS2's CS1 real-time factor of 2.26 has an interval from 0.65 to 5.46.** One outlier
  generation dominates it. Its other blocks sit near 0.62.
- **One voice per item**, so nothing here says whether either model's behaviour holds
  across speakers.
- **No listening test yet.** These are objective proxies. `make_ab_pairs.py` builds the
  blind AB sets, and 45 files per model is enough for a 30 to 40 pair test.

## Environment, which matters for reproducing

The two models **cannot share a Python environment**:

| | interpreter | torch |
|---|---|---|
| Confucius4 | `~/miniconda3/envs/test/bin/python` | 2.7.0+cu126 |
| ZONOS2 | `~/ZONOS2/.venv/bin/python` | 2.9.1+cu128, SGLang kernels |
| scoring | `~/miniconda3/envs/test/bin/python` | the ASR models are set up there |

ZONOS2 also JIT-compiles a CUDA indexing kernel on first use, so it needs `ninja` and
`nvcc` **on PATH**. Invoking `.venv/bin/python` directly does not put the venv's `bin` on
PATH, which is what made it fail with `FileNotFoundError: 'ninja'`. Export both:

```bash
export PATH=$HOME/ZONOS2/.venv/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
```

## Reproducing this run

```bash
cd ~/tts-vi-benchmark

# Confucius4
~/miniconda3/envs/test/bin/python scripts/generate_all.py \
    --model confucius4_tts --condition small \
    --max-items-per-block 5 --max-speakers 1 --samples 1

# ZONOS2 (needs the PATH exports above)
~/ZONOS2/.venv/bin/python scripts/generate_all.py \
    --model zonos2 --condition small \
    --max-items-per-block 5 --max-speakers 1 --samples 1

# score both together
~/miniconda3/envs/test/bin/python scripts/score.py \
    --run zonos2/small --run confucius4_tts/small \
    --asr-vi vinai/PhoWhisper-small --whisper openai/whisper-small --skip-utmos
```

## Changes made to the benchmark scripts

Originals backed up on the box at `/tmp/generate_all.py.bak` and `/tmp/score.py.bak`.

1. **`generate_all.py` gained `--max-speakers`.** Capping items alone did not cap work:
   five blocks carry no speaker list, and the default for those is every Vietnamese
   speaker, so 5 items silently became 40 generations each.
2. **`score.py`: word-timestamp failure now falls back.** Whisper's word-level timestamps
   crash inside the transformers ASR pipeline on some outputs, and that exception was
   discarding *every* metric for all 15 code-switching files. It now retries without
   timestamps, losing only the optional consistency metric.
3. **`score.py`: the consistency metric is non-fatal**, same reasoning, and its minimum
   span length went from 0.25 s to 0.5 s.

## Scale, if you want the full run later

Measured generation rate, driven by audio seconds rather than item count:

| scope | generations per model | Confucius4 | ZONOS2 |
|---|---|---|---|
| this pilot | 45 | 7 min | 4 min plus load |
| all 350 items, 1 sample, 2 voices | ~690 | 1.6 h | ~0.8 h |
| full default: 8 voices, 3 samples | 4,950 | 9.1 h | ~4.5 h |

Scoring adds roughly 3 to 5 s per file, plus a 6.5 GB download the first time the large
ASR models are used.

## Next steps

1. Look at the five XL2 items; both models fail them the same way.
2. Build the blind AB listening sets with `make_ab_pairs.py` for cloning and code-switching.
3. Re-run with all 350 items and two voices if the pilot's direction holds, roughly 2.5
   hours of generation for both models.
