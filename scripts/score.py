"""Score benchmark outputs for the three abilities (PLAN.md §3, §4 step 5).

Reads ``outputs/<model>/<condition>/manifest.jsonl`` (from generate_all.py), computes
per-file metrics, writes ``scores/<model>__<condition>.csv`` and a summary JSON with
bootstrap 95% confidence intervals per block. Given several runs it also writes
``scores/summary.md`` with the side-by-side table.

Metrics (all optional components degrade gracefully when a package is missing):

  intelligibility   cer_vi   PhoWhisper CER on Vietnamese text (V*, CS*, XL3)
                    wer_ws   Whisper-large-v3 WER (all blocks), cer_ws CER
                    en_recall / en_exact   CS blocks: fraction of English words recovered by Whisper
  language          lid, lid_ok            Whisper language id of the output vs the target language (XL blocks)
  cloning           sim_wavlm, sim_ecapa   cosine(out, reference) with WavLM-SV and SpeechBrain ECAPA
                    sim_gt_wavlm           cosine(out, ground-truth recording) when available
  consistency       cs_consistency         CS blocks: WavLM cosine between the Vietnamese part and the
                                           English part of the same output (Whisper word timestamps)
  quality           utmos                  UTMOS22-strong on 16 kHz audio
  cost              duration_s, gen_time_s, rtf   from the manifest

Aggregation: mean and 95% bootstrap CI per (block, ref) plus the derived checks
  noise_leakage  = sim_wavlm(clean ref) - sim_wavlm(noisy ref) on the same items
  sample_var     = std of sim_wavlm across the k samples of one (item, speaker)
  crosslingual_delta = sim(XL1/XL2) - sim(V1) for the same speakers

Examples::

    python benchmark/scripts/score.py --run confucius4_tts/A_pretrained --run zonos2/A_pretrained
    python benchmark/scripts/score.py --run zonos2/B_finetuned --asr-vi vinai/PhoWhisper-small --whisper openai/whisper-small --skip-utmos
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import statistics
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

BENCH_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("score")

VI_BLOCKS = {"V1", "V2", "CS1", "CS2", "CS3", "XL3"}
CS_BLOCKS = {"CS1", "CS2", "CS3"}
XL_TARGET = {"XL1": "en", "XL2": "zh", "XL3": "vi"}
_VI_DIACRITIC = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)


# --------------------------------------------------------------------------- args
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="<model>/<condition> under outputs/, repeatable")
    ap.add_argument("--outputs-root", default=str(BENCH_ROOT / "outputs"))
    ap.add_argument("--scores-dir", default=str(BENCH_ROOT / "scores"))
    ap.add_argument("--speakers", default=str(BENCH_ROOT / "testset" / "references" / "speakers.json"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: score only the first N files per run")
    ap.add_argument("--blocks", default="all")

    m = ap.add_argument_group("models")
    m.add_argument("--asr-vi", default="vinai/PhoWhisper-large", help="HF id; '' to skip")
    m.add_argument("--whisper", default="openai/whisper-large-v3", help="HF id; '' to skip")
    m.add_argument("--wavlm-sv", default="microsoft/wavlm-base-plus-sv", help="HF id; '' to skip")
    m.add_argument("--skip-ecapa", action="store_true", help="skip SpeechBrain ECAPA similarity")
    m.add_argument("--skip-utmos", action="store_true")
    m.add_argument("--skip-consistency", action="store_true", help="skip CS within-utterance consistency (needs Whisper word timestamps)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--recompute", action="store_true", help="ignore cached per-file rows in the CSV")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


# --------------------------------------------------------------------------- text utils
def norm_text(s: str, lang: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def edit_distance(a: List[str], b: List[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    r = list(ref.replace(" ", ""))
    return edit_distance(r, list(hyp.replace(" ", ""))) / max(len(r), 1)


def wer(ref: str, hyp: str) -> float:
    r = ref.split()
    return edit_distance(r, hyp.split()) / max(len(r), 1)


def english_words(item: dict) -> List[str]:
    """English span words for CS items: explicit 'english_spans' field, else heuristic
    (ASCII-only tokens of 3+ letters without Vietnamese diacritics, excluding common Vietnamese ASCII syllables)."""
    spans = item.get("english_spans")
    if spans:
        return [w for s in spans for w in norm_text(s, "en").split()]
    vi_ascii = {"ta", "con", "an", "ban", "toi", "anh", "em", "la", "va", "co", "khong", "nha", "cho", "nay", "cai", "ma", "hay", "thi", "de", "tren", "trong", "voi", "nhu", "ra", "vao", "lam", "them", "cua", "day", "tam", "nam", "sau", "ba", "hai", "sang"}
    out = []
    for w in norm_text(item["text"], "vi").split():
        if len(w) >= 3 and w.isascii() and w.isalpha() and not _VI_DIACRITIC.search(w) and w not in vi_ascii:
            out.append(w)
    return out


# --------------------------------------------------------------------------- audio utils
def load_16k(path: str) -> np.ndarray:
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        x = AF.resample(torch.from_numpy(x), sr, 16000).numpy()
    return x


# --------------------------------------------------------------------------- scorers (lazy, optional)
class Scorers:
    def __init__(self, a: argparse.Namespace, device: str):
        import torch

        self.device = device
        self.torch = torch
        self.asr_vi = self._pipeline(a.asr_vi, "asr-vi") if a.asr_vi else None
        self.whisper = self._pipeline(a.whisper, "whisper") if a.whisper else None
        self.wavlm = self.wavlm_fe = None
        if a.wavlm_sv:
            try:
                from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

                self.wavlm_fe = AutoFeatureExtractor.from_pretrained(a.wavlm_sv)
                self.wavlm = AutoModelForAudioXVector.from_pretrained(a.wavlm_sv).to(device).eval()
            except Exception as e:  # noqa: BLE001
                log.warning("WavLM-SV unavailable: %s", e)
        self.ecapa = None
        if not a.skip_ecapa:
            try:
                from speechbrain.inference.speaker import EncoderClassifier

                self.ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
            except Exception as e:  # noqa: BLE001
                log.warning("SpeechBrain ECAPA unavailable: %s", e)
        self.utmos = None
        if not a.skip_utmos:
            try:
                self.utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()
            except Exception as e:  # noqa: BLE001
                log.warning("UTMOS unavailable: %s", e)
        self.want_consistency = not a.skip_consistency and self.whisper is not None and self.wavlm is not None
        self._emb_cache: Dict[str, np.ndarray] = {}

    def _pipeline(self, name: str, tag: str):
        try:
            from transformers import pipeline

            dev = 0 if self.device.startswith("cuda") else -1
            return pipeline("automatic-speech-recognition", model=name, device=dev, torch_dtype=self.torch.float16 if dev >= 0 else None)
        except Exception as e:  # noqa: BLE001
            log.warning("%s (%s) unavailable: %s", tag, name, e)
            return None

    # ---- ASR
    def transcribe(self, pipe, wav: np.ndarray, lang: Optional[str], timestamps: bool = False) -> Tuple[str, Optional[str], list]:
        kw = {"generate_kwargs": {"task": "transcribe"}}
        if lang:
            kw["generate_kwargs"]["language"] = {"vi": "vietnamese", "en": "english", "zh": "chinese"}.get(lang, lang)
        if timestamps:
            kw["return_timestamps"] = "word"
        out = pipe({"raw": wav, "sampling_rate": 16000}, **kw)
        text = out.get("text", "")
        chunks = out.get("chunks", []) if timestamps else []
        return text, None, chunks

    def detect_language(self, wav: np.ndarray) -> Optional[str]:
        """Whisper language id via the model's detect_language when exposed; else None."""
        try:
            model, proc = self.whisper.model, self.whisper.feature_extractor
            feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features.to(model.device, model.dtype)
            tok = self.whisper.tokenizer
            with self.torch.no_grad():
                logits = model(feats, decoder_input_ids=self.torch.tensor([[model.config.decoder_start_token_id]], device=model.device)).logits[0, -1]
            lang_ids = {tok.convert_tokens_to_ids(f"<|{c}|>"): c for c in ("vi", "en", "zh", "ja", "ko", "fr", "de", "es", "th", "id")}
            best = max(lang_ids, key=lambda i: logits[i].item())
            return lang_ids[best]
        except Exception:
            return None

    # ---- speaker embeddings
    def embed_wavlm(self, wav: np.ndarray) -> Optional[np.ndarray]:
        if self.wavlm is None:
            return None
        inputs = self.wavlm_fe(wav, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            e = self.wavlm(**inputs).embeddings[0]
        e = self.torch.nn.functional.normalize(e, dim=-1)
        return e.float().cpu().numpy()

    def embed_ecapa(self, wav: np.ndarray) -> Optional[np.ndarray]:
        if self.ecapa is None:
            return None
        with self.torch.no_grad():
            e = self.ecapa.encode_batch(self.torch.from_numpy(wav).unsqueeze(0).to(self.device)).squeeze()
        e = self.torch.nn.functional.normalize(e, dim=-1)
        return e.float().cpu().numpy()

    def cached_ref(self, path: str, kind: str) -> Optional[np.ndarray]:
        key = f"{kind}:{path}"
        if key not in self._emb_cache:
            wav = load_16k(path)
            self._emb_cache[key] = self.embed_wavlm(wav) if kind == "wavlm" else self.embed_ecapa(wav)
        return self._emb_cache[key]

    def mos(self, wav: np.ndarray) -> Optional[float]:
        if self.utmos is None:
            return None
        with self.torch.no_grad():
            return float(self.utmos(self.torch.from_numpy(wav).unsqueeze(0).to(self.device), 16000).item())


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# --------------------------------------------------------------------------- per-file scoring
def score_file(row: dict, item: dict, S: Scorers) -> dict:
    blk = row["block"]
    lang = row.get("lang") or item.get("language") or "vi"
    wav = load_16k(row["path"])
    ref_text = norm_text(row["text"], lang)
    r: dict = {}

    # intelligibility
    if S.asr_vi is not None and blk in VI_BLOCKS:
        hyp, _, _ = S.transcribe(S.asr_vi, wav, "vi")
        hyp = norm_text(hyp, "vi")
        r["cer_vi"] = cer(ref_text, hyp)
        r["hyp_vi"] = hyp
    chunks = []
    if S.whisper is not None:
        want_ts = S.want_consistency and blk in CS_BLOCKS
        hyp, _, chunks = S.transcribe(S.whisper, wav, lang, timestamps=want_ts)
        hypn = norm_text(hyp, lang)
        r["wer_ws"] = wer(ref_text, hypn)
        r["cer_ws"] = cer(ref_text, hypn)
        r["hyp_ws"] = hypn
        if blk in CS_BLOCKS:
            ew = english_words(item)
            if ew:
                hyp_words = set(hypn.split())
                hits = [w for w in ew if w in hyp_words]
                r["en_recall"] = len(hits) / len(ew)
                r["en_words"] = len(ew)
        if blk in XL_TARGET:
            lid = S.detect_language(wav)
            r["lid"] = lid
            r["lid_ok"] = None if lid is None else float(lid == XL_TARGET[blk])

    # speaker similarity
    ref_wav_path = row.get("reference_wav")
    if ref_wav_path:
        e_out = S.embed_wavlm(wav)
        r["sim_wavlm"] = cosine(e_out, S.cached_ref(ref_wav_path, "wavlm"))
        r["sim_ecapa"] = cosine(S.embed_ecapa(wav), S.cached_ref(ref_wav_path, "ecapa"))
        gt = row.get("ground_truth_wav")
        if gt and Path(gt).is_file() and e_out is not None:
            r["sim_gt_wavlm"] = cosine(e_out, S.cached_ref(gt, "wavlm"))
        # within-utterance consistency for code-switching
        if blk in CS_BLOCKS and chunks and e_out is not None:
            ew = set(english_words(item))
            en_parts, vi_parts = [], []
            for c in chunks:
                w = norm_text(c.get("text", ""), "en")
                ts = c.get("timestamp") or (None, None)
                if ts[0] is None or ts[1] is None:
                    continue
                seg = wav[int(ts[0] * 16000): int(ts[1] * 16000)]
                (en_parts if w in ew else vi_parts).append(seg)
            if en_parts and vi_parts:
                en_wav, vi_wav = np.concatenate(en_parts), np.concatenate(vi_parts)
                if len(en_wav) > 4000 and len(vi_wav) > 4000:  # >= 0.25 s each
                    r["cs_consistency"] = cosine(S.embed_wavlm(en_wav), S.embed_wavlm(vi_wav))

    # quality and cost
    r["utmos"] = S.mos(wav)
    r["duration_s"] = row.get("duration_s")
    r["gen_time_s"] = row.get("gen_time_s")
    if row.get("duration_s") and row.get("gen_time_s"):
        r["rtf"] = row["gen_time_s"] / row["duration_s"]
    return r


# --------------------------------------------------------------------------- aggregation
METRICS = ["cer_vi", "wer_ws", "cer_ws", "en_recall", "lid_ok", "sim_wavlm", "sim_ecapa", "sim_gt_wavlm", "cs_consistency", "utmos", "rtf"]


def bootstrap_ci(vals: List[float], n: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return (math.nan,) * 3
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    means = [float(rng.choice(v, len(v), replace=True).mean()) for _ in range(n)]
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(rows: List[dict], n_boot: int) -> dict:
    rng = np.random.default_rng(0)
    out: dict = {"by_block_ref": {}, "derived": {}}
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["block"], r["ref"])].append(r)
    for (blk, ref), rs in sorted(groups.items()):
        g = {"n": len(rs)}
        for m in METRICS:
            vals = [r[m] for r in rs if r.get(m) is not None and not (isinstance(r[m], float) and math.isnan(r[m]))]
            if vals:
                mean, lo, hi = bootstrap_ci(vals, n_boot, rng)
                g[m] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(vals)}
        out["by_block_ref"][f"{blk}/{ref}"] = g

    # noise leakage: same (item, speaker, sample) clean vs noisy
    pairs = defaultdict(dict)
    for r in rows:
        if r.get("sim_wavlm") is not None:
            pairs[(r["item"], r["speaker"], r["sample"])][r["ref"]] = r["sim_wavlm"]
    deltas = [p["clean"] - p["noisy"] for p in pairs.values() if "clean" in p and "noisy" in p]
    if deltas:
        mean, lo, hi = bootstrap_ci(deltas, n_boot, rng)
        out["derived"]["noise_leakage_sim_drop"] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(deltas)}

    # sample variance of similarity across k samples
    per_key = defaultdict(list)
    for r in rows:
        if r.get("sim_wavlm") is not None and r["ref"] == "clean":
            per_key[(r["item"], r["speaker"])].append(r["sim_wavlm"])
    stds = [statistics.pstdev(v) for v in per_key.values() if len(v) >= 2]
    if stds:
        out["derived"]["sample_std_sim_wavlm"] = {"mean": round(float(np.mean(stds)), 4), "n": len(stds)}

    # cross-lingual delta: sim on XL1/XL2 minus sim on V1 for the same speaker
    spk_sim = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("sim_wavlm") is not None and r["ref"] == "clean":
            spk_sim[r["speaker"]][r["block"]].append(r["sim_wavlm"])
    for xl in ("XL1", "XL2"):
        ds = [np.mean(b[xl]) - np.mean(b["V1"]) for b in spk_sim.values() if b.get(xl) and b.get("V1")]
        if ds:
            mean, lo, hi = bootstrap_ci([float(d) for d in ds], n_boot, rng)
            out["derived"][f"crosslingual_delta_{xl}_vs_V1"] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(ds)}
    return out


def write_summary_md(path: Path, summaries: Dict[str, dict]) -> None:
    runs = list(summaries)
    keys = sorted({k for s in summaries.values() for k in s["by_block_ref"]})
    lines = ["# Benchmark summary", "", "Mean [95% CI] per block/reference. Lower is better for CER/WER/RTF; higher for sim, en_recall, lid_ok, utmos, cs_consistency.", ""]
    for m in METRICS:
        rows_m = []
        for k in keys:
            cells = []
            for run in runs:
                g = summaries[run]["by_block_ref"].get(k, {}).get(m)
                cells.append(f"{g['mean']:.3f} [{g['ci95'][0]:.3f}, {g['ci95'][1]:.3f}] (n={g['n']})" if g else "–")
            if any(c != "–" for c in cells):
                rows_m.append(f"| {k} | " + " | ".join(cells) + " |")
        if rows_m:
            lines += [f"## {m}", "", "| block/ref | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs), *rows_m, ""]
    lines += ["## derived", "", "| check | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
    dkeys = sorted({k for s in summaries.values() for k in s["derived"]})
    for k in dkeys:
        cells = []
        for run in runs:
            g = summaries[run]["derived"].get(k)
            cells.append(f"{g['mean']:.3f}" + (f" [{g['ci95'][0]:.3f}, {g['ci95'][1]:.3f}]" if g and "ci95" in g else "") + (f" (n={g['n']})" if g else "") if g else "–")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- main
def main() -> None:
    a = parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    import torch

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    scores_dir = Path(a.scores_dir).resolve()
    scores_dir.mkdir(parents=True, exist_ok=True)
    blocks = None if a.blocks == "all" else {b.strip() for b in a.blocks.split(",")}

    S: Optional[Scorers] = None
    summaries: Dict[str, dict] = {}
    for run in a.run:
        run_dir = Path(a.outputs_root).resolve() / run
        manifest = run_dir / "manifest.jsonl"
        if not manifest.is_file():
            log.warning("no manifest in %s, skipped", run_dir)
            continue
        rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [r for r in rows if "error" not in r and (blocks is None or r["block"] in blocks)]
        if a.limit:
            rows = rows[: a.limit]
        run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8")) if (run_dir / "run.json").is_file() else {}
        items = {}
        if run_meta.get("testset") and Path(run_meta["testset"]).is_file():
            items = {it["id"]: it for it in (json.loads(l) for l in Path(run_meta["testset"]).read_text(encoding="utf-8").splitlines() if l.strip())}

        csv_path = scores_dir / f"{run.replace('/', '__')}.csv"
        cached: Dict[str, dict] = {}
        if csv_path.is_file() and not a.recompute:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    cached[r["file"]] = {k: (float(v) if v not in ("", "None") and k in METRICS + ["duration_s", "gen_time_s"] else (None if v in ("", "None") else v)) for k, v in r.items()}
        todo = [r for r in rows if r["file"] not in cached]
        log.info("%s: %d files, %d cached, %d to score", run, len(rows), len(rows) - len(todo), len(todo))
        if todo and S is None:
            S = Scorers(a, device)
        scored: List[dict] = []
        for i, r in enumerate(rows, 1):
            base = {"file": r["file"], "item": r["item"], "block": r["block"], "speaker": r["speaker"], "ref": r["ref"], "sample": r["sample"]}
            if r["file"] in cached:
                scored.append({**base, **{k: v for k, v in cached[r["file"]].items() if k not in base}})
                continue
            r = dict(r)
            r["path"] = str(run_dir / r["file"])
            try:
                res = score_file(r, items.get(r["item"], {"text": r["text"], "language": r.get("lang")}), S)
            except Exception as exc:  # noqa: BLE001
                log.warning("score failed for %s: %s", r["file"], exc)
                res = {}
            scored.append({**base, **res})
            if i % 25 == 0 or i == len(rows):
                log.info("  %s: %d/%d", run, i, len(rows))
        cols = ["file", "item", "block", "speaker", "ref", "sample"] + METRICS + ["duration_s", "gen_time_s", "lid", "en_words", "hyp_vi", "hyp_ws"]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(scored)
        summary = aggregate(scored, a.bootstrap)
        summary["run"] = run
        summary["n_files"] = len(scored)
        summary["run_meta"] = {k: run_meta.get(k) for k in ("model", "condition", "model_path", "t2s_checkpoint", "testset_sha256", "repo_commit", "rtf", "date")}
        (scores_dir / f"{run.replace('/', '__')}.summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        summaries[run] = summary
        log.info("wrote %s", csv_path)

    if summaries:
        write_summary_md(scores_dir / "summary.md", summaries)
        log.info("wrote %s", scores_dir / "summary.md")
        for run, s in summaries.items():
            v1 = s["by_block_ref"].get("V1/clean", {})
            print(f"{run:32s} n={s['n_files']:4d}  V1 cer_vi={v1.get('cer_vi', {}).get('mean', 'n/a')}  sim_wavlm={v1.get('sim_wavlm', {}).get('mean', 'n/a')}  utmos={v1.get('utmos', {}).get('mean', 'n/a')}")


if __name__ == "__main__":
    main()
