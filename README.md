# benchmark — Confucius4-TTS vs ZONOS2 on Vietnamese

Resources for comparing the two models on three abilities: voice cloning, code-switching,
cross-lingual cloning. The full protocol (blocks, metrics, decision rule) is in `PLAN.md`.

Every voice is zero-shot and there is one condition per released model. A fine-tuned
checkpoint is benchmarked later by rerunning the same scripts with a new `--condition` label
and the checkpoint path; the test set never changes.

## Workflow

1. `build_testset_from_s7.py` — pick Vietnamese speakers, references and V1/V2/XL3 items from s7.
2. `add_foreign_references.py` — add the English and Mandarin voices that XL3 needs (already committed; rerun only to change them).
3. Fill the hand-written blocks (CS1–CS3, XL1, XL2) in `testset/texts/`, rerun step 1 to refresh the merged file and its hash.
4. `generate_all.py` — one run per (model, condition).
5. `score.py` — objective metrics and the side-by-side summary.
6. `make_ab_pairs.py` — blind listening pairs.

Steps 1–3 run anywhere the s7 folder is reachable; steps 4–6 run on the GPU box.

## Layout

```
benchmark/
├── PLAN.md                     protocol, metrics, decision rule
├── testset/
│   ├── texts/                  one JSONL per block (V1, V2, CS1–CS3, XL1–XL3), normalized text
│   ├── testset.jsonl           merged, hashed file that drives every run (written by step 1)
│   ├── summary.json            counts per block + SHA-256 of testset.jsonl
│   ├── ground_truth/           optional copies of the V1/V2 source wavs (--copy-ground-truth)
│   └── references/             one clean 8–12 s clip + one noisy 5 s clip per speaker
│       ├── speakers.json       speaker id, language, gender, clip paths, source
│       ├── spk_<name>_*.wav    Vietnamese voices from s7 (git-ignored, podcast audio)
│       └── spk_en_*/spk_zh_*   English / Mandarin voices from public data (committed)
├── outputs/
│   ├── confucius4_tts/<condition>/    wavs + manifest.jsonl + run.json; condition = pretrained or any checkpoint label
│   └── zonos2/<condition>/
├── scores/                     per-utterance CSVs and summary tables from score.py
├── listening/                  AB pair lists, rater sheets, answer keys
└── scripts/
    ├── build_testset_from_s7.py
    ├── add_foreign_references.py
    ├── generate_all.py
    ├── score.py
    └── make_ab_pairs.py
```

## Conventions

- Item id: `<block>_<nnn>`, e.g. `CS2_014`. Speaker id: `spk_<name>` for Vietnamese voices,
  `spk_en_<name>` / `spk_zh_<name>` for foreign voices.
- Output file: `<item_id>__<speaker_id>__<clean|noisy>__s<k>.wav`, `k` = sample index 0–2, seed = 1000 + k.
  Each run folder has a `manifest.jsonl` with one row per file (text, reference, duration, generation time, or `error`).
- Every run folder carries `run.json`: model path, git commit of the model repo, generation
  parameters, testset hash, date.
- Text in `testset.jsonl` is already normalized with `normalize_vietnamese` from
  `ZONOS2/scripts/generate_vi.py`; both models run with their own normalizers disabled.
- Sample rates: references 24 kHz; Confucius4-TTS outputs 22.05 kHz, ZONOS2 44.1 kHz. `score.py`
  resamples to 16 kHz for ASR and speaker models and to 22.05 kHz for UTMOS.
- All scripts are resumable: rerun the same command and finished files are skipped.

## 1. Building the test set

### Vietnamese speakers and items (from s7)

```bash
pip install numpy soundfile scipy
python scripts/build_testset_from_s7.py \
    --s7-dir /path/work_XXX/s7_loudnorm \
    --speakers 8 --v1 100 --v2 50 --xl3 40 --seed 42 [--dry-run]
```

| Flag | Meaning |
|---|---|
| `--s7-dir` | `<workdir>/s7_loudnorm`; repeatable to pool several workdirs |
| `--speakers 8` | Vietnamese speakers picked at random (seeded) among those with at least `--min-utts-per-speaker` (12) usable segments |
| `--v1 100` | in-domain items: sentences of the selected speakers, read by their own voice |
| `--v2 50` | 8–15 s items from any Vietnamese speaker (prosody over longer text) |
| `--xl3 40` | Vietnamese texts to be read by the foreign voices |
| `--seed 42` | same seed, same s7 folder → same selection |

Only segments that pass the quality gates are candidates (`--max-cer 0.05`, `--min-snr-db 15`,
`--min-dnsmos 2.8`, `--max-clipping 0.001`, 3–15 s); segments whose wav is missing on disk
are skipped and another one is drawn. Speaker ids come from s7's per-file clusters, so they
are voices, not necessarily distinct people.

The script writes, per speaker, one clean 8–12 s reference and one noisy 5 s reference
(`--noisy-mode`):

| Mode | Noisy reference is |
|---|---|
| `lowest_snr` (default) | the speaker's real clip with the lowest estimated SNR, cut to 5 s |
| `dnsmos` | the real clip with the lowest DNSMOS; falls back to SNR when DNSMOS is absent |
| `synthetic` | the clean reference cut to 5 s with pink (or `--synthetic-noise white`) noise mixed at `--synthetic-snr-db` (default 10 dB): a controlled degradation, same content as the clean clip |

It then writes empty `#`-commented templates for the hand-written blocks (CS1–CS3, XL1, XL2),
merges every block into `testset/testset.jsonl` and records its SHA-256 in
`testset/summary.json`. `--copy-ground-truth` also copies each V1/V2 source wav into
`testset/ground_truth/<item_id>.wav` so scoring does not depend on the pipeline workdir.

### Foreign reference voices (for XL3)

s7 only has Vietnamese voices. `scripts/add_foreign_references.py` adds the English and Mandarin
references from public datasets and appends them to `speakers.json` without touching the
Vietnamese entries:

```bash
pip install requests numpy soundfile scipy duckdb pyarrow
python scripts/add_foreign_references.py --seed 42   # default: 1 EN male + 1 EN female + 1 ZH
```

| Voice | Source | How the clean 8–12 s clip is made |
|---|---|---|
| `spk_en_m*`, `spk_en_f*` | LibriTTS-R `test.clean` (CC BY 4.0); gender and SNR from `ylacombe/libritts_r_tags` | DuckDB reads the parquet shards over HTTPS (no token, no full download); among the cleanest speakers, one utterance that falls in the window |
| `spk_zh_*` | AISHELL-3 `test` (Apache 2.0), northern accent | consecutive utterances of one speaker concatenated with 0.3 s gaps |

Clips are 24 kHz mono PCM16, level-matched to the Vietnamese references (RMS −23 dBFS,
peak ≤ −3 dBFS, close to the s7 loudnorm target); the noisy reference is always synthetic
(pink noise, `--snr-db 10`). These clips are small and redistributable, so they are committed;
the Vietnamese references stay git-ignored. Same seed, same speakers.

### Hand-written blocks

Fill `testset/texts/CS1.jsonl` … `XL2.jsonl` by replacing the commented template rows. Each
row needs `item_id`, `text` and, for CS blocks, an `english_spans` list with the English words
of the item (without it `score.py` guesses them from ASCII tokens). Then rerun
`build_testset_from_s7.py` with the same arguments: the random parts are unchanged, the merged
file and hash are refreshed.

## 2. Generate (GPU box)

Each model generates inside its own virtualenv. `--condition` is a free label naming the
checkpoint; outputs land in `outputs/<model>/<condition>/`.

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
#   --noisy-ref-blocks V1,V2                              which blocks also run with the noisy reference ('' = none)
#   --zonos-normalize-foreign                             let ZONOS2 normalize XL1/XL2 text (English/Mandarin) itself
#   --dry-run                                             print the job list only
```

Speakers per block: V1/V2/CS*/XL1/XL2 default to every Vietnamese speaker in
`speakers.json`, XL3 to every foreign speaker; an item's `speakers` list overrides this and
accepts wildcards (`spk_en_*`).

## 3. Score

Any env with `transformers`, `torchaudio`, `soundfile` (plus `speechbrain` for ECAPA).

```bash
python benchmark/scripts/score.py --run confucius4_tts/pretrained --run zonos2/pretrained
# add --run <model>/<label> for every further checkpoint
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

PhoWhisper ships `pytorch_model.bin`; `transformers` refuses to load it on torch < 2.6, so
score on the GPU box (torch 2.9) or pass a local safetensors copy to `--asr-vi`.

## 4. Listening tests

```bash
python benchmark/scripts/make_ab_pairs.py --run-a confucius4_tts/pretrained --run-b zonos2/pretrained --question cloning      --n 40
python benchmark/scripts/make_ab_pairs.py --run-a confucius4_tts/pretrained --run-b zonos2/pretrained --question codeswitch   --n 30
python benchmark/scripts/make_ab_pairs.py --run-a confucius4_tts/pretrained --run-b zonos2/pretrained --question crosslingual --n 30
python benchmark/scripts/make_ab_pairs.py --run-a confucius4_tts/pretrained --question accent --n 40   # single run, no --run-b
```

Each call writes `listening/<question>_<seed>/` with 16 kHz peak-normalized `pairs/`, a
`sheet.csv` for raters and a `key.csv` (A/B assignment) to keep hidden until scoring.
`--ref clean|noisy|both` picks which reference condition the pairs come from.

## Direct generation entry points

- Confucius4-TTS: `Confucius4-TTS/scripts/generate_vi.py --no-normalize ...`
- ZONOS2: `ZONOS2/scripts/generate_vi.py --no-normalize ...`

Both accept `--text-file`, a reference clip, and a model path for other checkpoints.
