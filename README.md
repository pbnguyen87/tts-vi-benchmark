# benchmark — Confucius4-TTS vs ZONOS2 on Vietnamese

Resources for comparing the two models on three abilities: voice cloning, code-switching,
cross-lingual cloning. The full protocol is in `PLAN.md`.

## Layout

```
benchmark/
├── PLAN.md                     protocol, metrics, decision rule
├── testset/
│   ├── texts/                  one JSONL per block (V1, V2, CS1–CS3, XL1–XL3), normalized text
│   │   └── testset.jsonl       merged, hashed file that drives every run
│   └── references/             one clean 8–12 s clip + one noisy 5 s clip per speaker
│       └── speakers.json       speaker id, language, clip paths
├── outputs/
│   ├── confucius4_tts/<condition>/    wavs + manifest.jsonl + run.json; condition = pretrained, or any checkpoint label
│   └── zonos2/<condition>/
├── scores/                     per-utterance CSVs and summary tables from the scoring script
├── listening/                  AB pair lists, rater sheets, results
└── scripts/                    build_testset_from_s7.py, generate_all.py, score.py, make_ab_pairs.py
```

## Conventions

- Item id: `<block>_<nnn>`, e.g. `CS2_014`. Speaker id: `spk_<name>`; foreign speakers `spk_en_<name>`, `spk_zh_<name>`.
- Output file: `<item_id>__<speaker_id>__<clean|noisy>__s<k>.wav`, `k` = sample index 0–2, seed = 1000 + k.
  Each run folder has a `manifest.jsonl` with one row per file (text, reference, duration, generation time, or `error`).
- Every run folder carries `run.json`: model path, git commit of the model repo, generation
  parameters, testset hash, date.
- Text in `testset.jsonl` is already normalized with `normalize_vietnamese` from
  `ZONOS2/scripts/generate_vi.py`; both models run with their own normalizers disabled.
- Sample rates: Confucius4-TTS 22.05 kHz, ZONOS2 44.1 kHz. `score.py` resamples to 16 kHz for
  ASR and speaker models and to 22.05 kHz for UTMOS.

## Building the test set

```bash
python benchmark/scripts/build_testset_from_s7.py \
    --s7-dir /path/work_XXX/s7_loudnorm \
    --speakers 8 --v1 100 --v2 50 --xl3 40 --seed 42 [--dry-run]
```

Selects `--speakers` Vietnamese speakers (random, seeded), their references (clean 8–12 s,
noisy 5 s) and the V1/V2/XL3 items, all drawn at random from s7 segments; segments whose wav
is missing on disk are skipped. There is no training split: every voice is zero-shot, writes empty `#`-commented templates for CS1–CS3, XL1, XL2, merges everything into
`testset/testset.jsonl` and records its SHA-256 in `testset/summary.json`. Fill the template
blocks by hand, then re-run the script (same seed) to refresh the merged file and hash.

Noisy-reference options (`--noisy-mode`):

| Mode | Noisy reference is |
|---|---|
| `lowest_snr` (default) | the speaker's real clip with the lowest estimated SNR, cut to 5 s |
| `dnsmos` | the real clip with the lowest DNSMOS; falls back to SNR when DNSMOS is absent |
| `synthetic` | the clean reference cut to 5 s with pink (or `--synthetic-noise white`) noise mixed at `--synthetic-snr-db` (default 10 dB): a controlled degradation, same content as the clean clip |

`--copy-ground-truth` copies each V1/V2 source wav into `testset/ground_truth/<item_id>.wav`
so the test set does not depend on the pipeline workdir.

## Running the benchmark (GPU box)

Each script uses the model's own virtualenv for generation and any env with `transformers`,
`torchaudio`, `soundfile` for scoring. Everything is resumable: rerun the same command and
finished files are skipped.

### 1. Generate, one command per (model, condition)

`--condition` is a free label naming the checkpoint; outputs land in `outputs/<model>/<condition>/`.

```bash
# released checkpoints, all blocks, 3 samples, noisy reference on V1 only
Confucius4-TTS/.venv/bin/python benchmark/scripts/generate_all.py --model confucius4_tts --condition pretrained
ZONOS2/.venv/bin/python        benchmark/scripts/generate_all.py --model zonos2          --condition pretrained

# later, another checkpoint of the same model on the SAME test set: only the label and the weights change
Confucius4-TTS/.venv/bin/python benchmark/scripts/generate_all.py --model confucius4_tts --condition vi_ft_v1 \
    --t2s-checkpoint Confucius4-TTS/checkpoints/t2s_vi/model.safetensors
ZONOS2/.venv/bin/python benchmark/scripts/generate_all.py --model zonos2 --condition vi_ft_v1 \
    --model-path ZONOS2/finetune/runs/vi_lora/merged

# useful flags
#   --blocks V1,CS1 --samples 1 --max-items-per-block 5   quick pass
#   --noisy-ref-blocks V1,V2                              which blocks also run with the noisy reference
#   --zonos-normalize-foreign                             let ZONOS2 normalize XL1/XL2 text (English/Mandarin) itself
#   --dry-run                                             print the job list only
```

Speakers per block: V1/V2/CS*/XL1/XL2 default to every Vietnamese speaker in
`speakers.json`, XL3 to every foreign speaker; an item's `speakers` list overrides this and
accepts wildcards (`spk_en_*`).

### 2. Score

```bash
python benchmark/scripts/score.py --run confucius4_tts/pretrained --run zonos2/pretrained   # thêm --run <model>/<label> cho checkpoint khác
```

Writes `scores/<model>__<condition>.csv` (per file), `.summary.json` (mean and bootstrap 95 % CI
per block and reference, derived checks) and `scores/summary.md` with all runs side by side.
Metric to flag mapping:

| Metric | Source | Flag to skip / change |
|---|---|---|
| `cer_vi` | PhoWhisper-large | `--asr-vi ''` or another HF id |
| `wer_ws`, `cer_ws`, `en_recall`, `lid_ok` | Whisper-large-v3 | `--whisper ''` |
| `sim_wavlm`, `sim_gt_wavlm`, `cs_consistency` | microsoft/wavlm-base-plus-sv | `--wavlm-sv ''`, `--skip-consistency` |
| `sim_ecapa` | speechbrain/spkrec-ecapa-voxceleb | `--skip-ecapa` |
| `utmos` | tarepan/SpeechMOS utmos22_strong | `--skip-utmos` |

Derived checks in the summary: `noise_leakage_sim_drop` (clean minus noisy similarity on the
same items), `sample_std_sim_wavlm` (spread across the 3 samples), `crosslingual_delta_XL1_vs_V1`
and `_XL2_vs_V1` (similarity penalty when a Vietnamese voice speaks English or Mandarin).
For CS blocks, put the English words of each item in an `english_spans` list in the JSONL;
without it the script guesses them from ASCII tokens.

PhoWhisper ships `pytorch_model.bin`; `transformers` refuses to load it on torch < 2.6, so
score on the GPU box (torch 2.9) or pass a local safetensors copy to `--asr-vi`.

### 3. Listening tests

```bash
python benchmark/scripts/make_ab_pairs.py --run-a confucius4_tts/pretrained --run-b zonos2/pretrained --question cloning     --n 40
python benchmark/scripts/make_ab_pairs.py --run-a confucius4_tts/pretrained --run-b zonos2/pretrained --question codeswitch  --n 30
python benchmark/scripts/make_ab_pairs.py --run-a confucius4_tts/pretrained --run-b zonos2/pretrained --question crosslingual --n 30
python benchmark/scripts/make_ab_pairs.py --run-a confucius4_tts/pretrained --run-b zonos2/pretrained --question accent      --n 40
```

Each call writes `listening/<question>_<seed>/` with 16 kHz peak-normalized `pairs/`, a
`sheet.csv` for raters and a `key.csv` (A/B assignment) to keep hidden until scoring.

## Direct generation entry points

- Confucius4-TTS: `Confucius4-TTS/scripts/generate_vi.py --no-normalize ...`
- ZONOS2: `ZONOS2/scripts/generate_vi.py --no-normalize ...`

Both accept `--text-file`, a reference clip, and a model path for fine-tuned checkpoints.
