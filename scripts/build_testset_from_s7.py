"""Build the benchmark test set from audio-pipeline ``s7_loudnorm`` outputs (PLAN.md §2).

What it selects, randomly with a fixed seed:

* **Speakers**: N "in-training" speakers (present in the fine-tuning split you pass with
  ``--train-jsonl``/``--train-tsv``) and M "unseen" speakers (absent from it), drawn at random
  (seeded) among speakers with enough usable segments. The training manifests only decide the
  ``in_training`` flag: segment ids are NOT excluded from the candidate pool, items may overlap
  the training data.
* **References** per speaker (``testset/references/``): one clean 8-12 s clip (best SNR / CER,
  single speaker) and one noisy 5 s clip (lowest SNR), never reused as test items.
* **V1** in-domain sentences: ``--v1`` items from in-training speakers, drawn at random from s7.
* Any segment whose wav is missing on disk (e.g. deleted by ``pipeline package --drop-wav``) is
  skipped and another id is drawn instead.
* **V2** read-style / longer sentences: ``--v2`` items of 8-15 s, any Vietnamese speaker.
* **XL3** Vietnamese texts to be spoken by foreign voices: ``--xl3`` items (text only).

Blocks that cannot come from podcast audio (CS1-CS3 English mixing, XL1 English, XL2
Mandarin) are written as empty JSONL templates in ``testset/texts/`` for you to fill; the
script merges every ``testset/texts/*.jsonl`` present into ``testset/testset.jsonl`` and
records its SHA-256 so every run can cite the exact test set.

Usage::

    python benchmark/scripts/build_testset_from_s7.py \\
        --s7-dir /path/work_XXX/s7_loudnorm [--s7-dir another/s7_loudnorm ...] \\
        --train-jsonl ZONOS2/data/zonos2_vi/train.jsonl \\
        --train-speakers 5 --unseen-speakers 3 --v1 100 --v2 50 --xl3 40 --seed 42

Re-running with the same seed reproduces the same selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

BENCH_ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = BENCH_ROOT.parent
sys.path.insert(0, str(STUDY_ROOT / "ZONOS2" / "scripts"))
try:
    from generate_vi import normalize_vietnamese  # noqa: E402
except Exception:  # pragma: no cover
    normalize_vietnamese = None  # type: ignore

log = logging.getLogger("build_testset")

BLOCK_TEMPLATES = {
    "CS1": "Vietnamese sentence with single English loanwords (technical), e.g. 'Mình phải deploy model này lên production.'",
    "CS2": "Vietnamese sentence with English proper nouns / brands, e.g. 'Anh ấy làm ở Google từ khi rời Grab.'",
    "CS3": "Vietnamese sentence with a full English clause of 4+ words, e.g. 'Sếp bảo: we need to ship this by Friday, nên cả team tăng ca.'",
    "XL1": "English text, spoken by Vietnamese voices (cross-lingual vi->en).",
    "XL2": "Mandarin text, spoken by Vietnamese voices (cross-lingual vi->zh).",
}


# --------------------------------------------------------------------------- args
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s7-dir", action="append", required=True, help="path to <workdir>/s7_loudnorm; repeatable")
    ap.add_argument("--out-dir", default=str(BENCH_ROOT / "testset"))
    ap.add_argument("--train-jsonl", action="append", default=[], help="fine-tune manifest(s) (ZONOS2 style) whose items/speakers count as 'in training'")
    ap.add_argument("--train-tsv", action="append", default=[], help="fine-tune TSV(s) (Confucius4-TTS style), same purpose")
    ap.add_argument("--train-speakers", type=int, default=5)
    ap.add_argument("--unseen-speakers", type=int, default=3)
    ap.add_argument("--v1", type=int, default=100, help="in-domain items from in-training speakers")
    ap.add_argument("--v2", type=int, default=50, help="8-15 s items from any Vietnamese speaker")
    ap.add_argument("--xl3", type=int, default=40, help="Vietnamese texts for foreign voices")
    ap.add_argument("--seed", type=int, default=42)

    q = ap.add_argument_group("quality gates for candidate segments")
    q.add_argument("--max-cer", type=float, default=0.05)
    q.add_argument("--min-snr-db", type=float, default=15.0)
    q.add_argument("--min-dnsmos", type=float, default=2.8)
    q.add_argument("--max-clipping", type=float, default=0.001)
    q.add_argument("--min-seconds", type=float, default=3.0)
    q.add_argument("--max-seconds", type=float, default=15.0)
    q.add_argument("--min-utts-per-speaker", type=int, default=12, help="need refs + items; speakers below this are ignored")

    r = ap.add_argument_group("references")
    r.add_argument("--ref-min-seconds", type=float, default=8.0)
    r.add_argument("--ref-max-seconds", type=float, default=12.0)
    r.add_argument("--noisy-ref-seconds", type=float, default=5.0, help="noisy reference is cut to this length")
    r.add_argument("--noisy-mode", choices=("lowest_snr", "dnsmos", "synthetic"), default="lowest_snr",
                   help="lowest_snr: real clip with the lowest estimated SNR; dnsmos: real clip with the lowest DNSMOS "
                        "(falls back to SNR when missing); synthetic: the clean reference cut to --noisy-ref-seconds "
                        "with noise mixed in at --synthetic-snr-db (controlled degradation)")
    r.add_argument("--synthetic-snr-db", type=float, default=10.0, help="target SNR for --noisy-mode synthetic")
    r.add_argument("--synthetic-noise", choices=("white", "pink"), default="pink", help="noise colour for synthetic mode")
    ap.add_argument("--copy-ground-truth", action="store_true",
                    help="copy each selected item's source wav into testset/ground_truth/<item_id>.wav and point ground_truth_wav there "
                         "(makes the test set self-contained; default keeps paths into s7_loudnorm/audio)")

    ap.add_argument("--no-normalize", action="store_true", help="keep text_normalized as-is")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


# --------------------------------------------------------------------------- loading
def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def resolve_audio(row: dict, s7_dir: Path) -> Optional[Path]:
    p = row.get("audio_path")
    cands = []
    if p:
        pp = Path(p)
        cands += [pp] if pp.is_absolute() else [s7_dir.parent.parent / pp, s7_dir.parent / pp]
    cands.append(s7_dir / "audio" / f"{row.get('id')}.wav")
    for c in cands:
        if c.is_file():
            return c.resolve()
    return None


def load_segments(s7_dirs: List[Path]) -> List[dict]:
    segs = []
    for d in s7_dirs:
        m = d / "manifest.jsonl"
        if not m.is_file():
            log.warning("no manifest in %s, skipped", d)
            continue
        n = missing = 0
        for r in read_jsonl(m):
            wav = resolve_audio(r, d)
            if wav is None:
                missing += 1
                continue
            r = dict(r)
            r["wav_abs"] = wav
            r["workdir"] = d.parent.name
            segs.append(r)
            n += 1
        log.info("%s: %d segments with audio, %d skipped (wav missing)", m, n, missing)
    return segs


def training_ids_and_speakers(jsonls: List[str], tsvs: List[str]) -> tuple[set, set]:
    ids, spk = set(), set()
    for p in jsonls:
        for r in read_jsonl(Path(p)):
            if r.get("id"):
                ids.add(r["id"])
            elif r.get("audio"):
                ids.add(Path(r["audio"]).stem)
            if r.get("speaker_id"):
                spk.add(r["speaker_id"])
    for p in tsvs:
        with Path(p).open("r", encoding="utf-8") as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) >= 2:
                    ids.add(Path(cols[1]).stem)
    return ids, spk


def clean_text(t: Optional[str]) -> str:
    if not t:
        return ""
    t = t.replace("\r\n", " ").replace("\n", " ").replace("\t", " ").replace('"', "")
    return re.sub(r"\s+", " ", t).strip()


def passes_gate(r: dict, a: argparse.Namespace) -> bool:
    cer = r.get("cer")
    dur = r.get("duration")
    if cer is None or dur is None or cer > a.max_cer:
        return False
    if dur < a.min_seconds or dur > a.max_seconds:
        return False
    if (r.get("clipping") or 0.0) > a.max_clipping or r.get("multi_speaker"):
        return False
    if r.get("dnsmos") is not None:
        if r["dnsmos"] < a.min_dnsmos:
            return False
    elif (r.get("snr_db") or 0.0) < a.min_snr_db:
        return False
    return bool(clean_text(r.get("text_normalized") or r.get("text")))


def has_audio(r: dict) -> bool:
    """Re-check at selection time: the wav may have been deleted after loading (package --drop-wav)."""
    return bool(r.get("wav_abs")) and Path(r["wav_abs"]).is_file()


def quality(r: dict) -> float:
    acoustic = (r["dnsmos"] * 10.0) if r.get("dnsmos") is not None else min(r.get("snr_db") or 0.0, 40.0)
    return acoustic - 100.0 * (r.get("cer") or 0.0)


# --------------------------------------------------------------------------- selection
def pick_references(utts: List[dict], a: argparse.Namespace) -> tuple[dict, dict]:
    utts = [u for u in utts if has_audio(u)]
    clean_pool = [u for u in utts if a.ref_min_seconds <= u["duration"] <= a.ref_max_seconds] or utts
    clean = max(clean_pool, key=quality)
    if a.noisy_mode == "synthetic":
        return clean, clean  # noisy reference is derived from the clean one
    rest = [u for u in utts if u["id"] != clean["id"]]
    noisy_pool = [u for u in rest if u["duration"] >= a.noisy_ref_seconds] or rest
    if a.noisy_mode == "dnsmos" and any(u.get("dnsmos") is not None for u in noisy_pool):
        noisy = min(noisy_pool, key=lambda u: u["dnsmos"] if u.get("dnsmos") is not None else 9.0)
    else:
        noisy = min(noisy_pool, key=lambda u: u.get("snr_db") if u.get("snr_db") is not None else 99.0)
    return clean, noisy


def add_noise_wav(src: Path, dst: Path, seconds: float, snr_db: float, kind: str, seed: int) -> None:
    """Cut ``src`` to ``seconds`` and mix in white or pink noise at ``snr_db`` (16-bit PCM in/out)."""
    import wave

    import numpy as np

    with wave.open(str(src), "rb") as w:
        params = w.getparams()
        n = min(w.getnframes(), int(seconds * w.getframerate()))
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if params.nchannels > 1:
        x = x.reshape(-1, params.nchannels).mean(axis=1)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(x)).astype(np.float32)
    if kind == "pink":  # 1/f spectrum via FFT shaping
        spec = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(len(noise))
        spec[1:] /= np.sqrt(freqs[1:])
        spec[0] = 0.0
        noise = np.fft.irfft(spec, n=len(noise)).astype(np.float32)
    sig_pow = float(np.mean(x ** 2)) + 1e-12
    noise_pow = float(np.mean(noise ** 2)) + 1e-12
    noise *= np.sqrt(sig_pow / (noise_pow * 10 ** (snr_db / 10.0)))
    y = x + noise
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:
        y = y * (0.99 / peak)
    with wave.open(str(dst), "wb") as o:
        o.setnchannels(1)
        o.setsampwidth(2)
        o.setframerate(params.framerate)
        o.writeframes((y * 32767.0).astype(np.int16).tobytes())


def cut_wav(src: Path, dst: Path, seconds: float) -> None:
    """Copy the first ``seconds`` of a wav (16-bit PCM) without external deps."""
    import wave

    with wave.open(str(src), "rb") as w:
        params = w.getparams()
        n = min(w.getnframes(), int(seconds * w.getframerate()))
        frames = w.readframes(n)
    with wave.open(str(dst), "wb") as o:
        o.setparams(params)
        o.writeframes(frames)


def item(block: str, idx: int, r: dict, speaker: str, text: str, gt: Optional[Path]) -> dict:
    return {
        "id": f"{block}_{idx:03d}",
        "block": block,
        "text": text,
        "language": "vi",
        "speakers": [speaker] if speaker else [],
        "source_id": r.get("id") if r else None,
        "source_speaker": r.get("speaker_id") if r else None,
        "ground_truth_wav": str(gt) if gt else None,
        "duration_gt": r.get("duration") if r else None,
    }


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_template(path: Path, block: str, note: str) -> None:
    if path.exists():
        return
    example = json.dumps({"id": f"{block}_001", "block": block, "text": "...", "language": "vi" if block.startswith("CS") else ("en" if block == "XL1" else "zh"),
                          "speakers": ["spk_<name>", "..."]}, ensure_ascii=False)
    # example stays a '#' comment so merge_blocks() ignores it until real items are added
    path.write_text("# " + note + "\n# One JSON object per line (remove the leading '#'), e.g.\n# " + example + "\n", encoding="utf-8")


def merge_blocks(texts_dir: Path, out: Path) -> tuple[int, str]:
    rows = []
    for p in sorted(texts_dir.glob("*.jsonl")):
        if p.name == "testset.jsonl":
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    write_jsonl(out, rows)
    return len(rows), hashlib.sha256(out.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- main
def main() -> None:
    a = parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    rng = random.Random(a.seed)
    out = Path(a.out_dir).resolve()
    texts_dir, refs_dir = out / "texts", out / "references"

    segs = load_segments([Path(p).expanduser().resolve() for p in a.s7_dir])
    train_ids, train_spk_from_manifest = training_ids_and_speakers(a.train_jsonl, a.train_tsv)
    # speakers "in training" = speakers that own any training id (works for TSV inputs without speaker_id)
    id2spk = {r["id"]: r.get("speaker_id") for r in segs}
    train_spk = set(train_spk_from_manifest) | {id2spk[i] for i in train_ids if i in id2spk and id2spk[i]}
    log.info("training set: %d ids, %d speakers", len(train_ids), len(train_spk))

    # mọi segment đạt gate và còn wav đều là ứng viên; KHÔNG loại id đã có trong tập train
    usable = [r for r in segs if r.get("speaker_id") and passes_gate(r, a)]
    by_spk: Dict[str, List[dict]] = defaultdict(list)
    for r in usable:
        by_spk[r["speaker_id"]].append(r)
    big = {s: u for s, u in by_spk.items() if len(u) >= a.min_utts_per_speaker}
    n_overlap = sum(1 for r in usable if r["id"] in train_ids)
    log.info("usable %d (%d of them also in the training manifests, kept), speakers with >=%d utts: %d",
             len(usable), n_overlap, a.min_utts_per_speaker, len(big))

    in_train = sorted(s for s in big if s in train_spk)
    unseen = sorted(s for s in big if s not in train_spk)
    rng.shuffle(in_train)
    rng.shuffle(unseen)
    if not train_spk:
        log.warning("no training manifest given: labelling %d random speakers as 'in training' for selection purposes", a.train_speakers)
        in_train, unseen = unseen[: a.train_speakers], unseen[a.train_speakers:]
    sel_train = in_train[: a.train_speakers]
    sel_unseen = unseen[: a.unseen_speakers]
    if len(sel_train) < a.train_speakers or len(sel_unseen) < a.unseen_speakers:
        log.warning("requested %d in-training + %d unseen speakers, found %d + %d", a.train_speakers, a.unseen_speakers, len(sel_train), len(sel_unseen))

    # ---- references
    speakers_meta, used_ids = [], set()
    ref_plan = []
    for kind, spks in (("in_training", sel_train), ("unseen", sel_unseen)):
        for s in spks:
            clean, noisy = pick_references(big[s], a)
            used_ids.update({clean["id"], noisy["id"]})
            spk_id = f"spk_{s}"
            speakers_meta.append({
                "speaker_id": spk_id, "source_speaker": s, "language": "vi", "in_training": kind == "in_training",
                "n_usable_utts": len(big[s]),
                "reference_clean": f"references/{spk_id}_clean.wav", "reference_clean_source": clean["id"],
                "reference_clean_seconds": clean["duration"], "reference_clean_snr_db": clean.get("snr_db"),
                "reference_noisy": f"references/{spk_id}_noisy.wav", "reference_noisy_source": noisy["id"],
                "reference_noisy_seconds": min(noisy["duration"], a.noisy_ref_seconds),
                "reference_noisy_snr_db": a.synthetic_snr_db if a.noisy_mode == "synthetic" else noisy.get("snr_db"),
                "reference_noisy_dnsmos": noisy.get("dnsmos"),
                "noisy_mode": a.noisy_mode,
                "synthetic_noise": a.synthetic_noise if a.noisy_mode == "synthetic" else None,
            })
            ref_plan.append((spk_id, clean, noisy))

    # ---- items
    norm = (lambda t: t) if (a.no_normalize or normalize_vietnamese is None) else normalize_vietnamese
    if normalize_vietnamese is None and not a.no_normalize:
        log.warning("normalize_vietnamese not importable from ZONOS2/scripts; texts kept as-is")

    def pool(spks: List[str]) -> List[dict]:
        cands = [u for s in spks for u in big[s] if u["id"] not in used_ids]
        ok = [u for u in cands if has_audio(u)]
        if len(ok) < len(cands):
            log.warning("%d candidate segments skipped: wav no longer on disk", len(cands) - len(ok))
        return ok

    v1_pool = pool(sel_train)
    rng.shuffle(v1_pool)
    v1 = v1_pool[: a.v1]
    used_ids.update(u["id"] for u in v1)
    v1_items = [item("V1", i + 1, u, f"spk_{u['speaker_id']}", norm(clean_text(u.get("text_normalized") or u.get("text"))), u["wav_abs"]) for i, u in enumerate(v1)]

    v2_pool = [u for u in pool(sel_train + sel_unseen) if 8.0 <= u["duration"] <= 15.0]
    rng.shuffle(v2_pool)
    v2 = v2_pool[: a.v2]
    used_ids.update(u["id"] for u in v2)
    v2_items = [item("V2", i + 1, u, f"spk_{u['speaker_id']}", norm(clean_text(u.get("text_normalized") or u.get("text"))), u["wav_abs"]) for i, u in enumerate(v2)]

    xl3_pool = pool(sel_train + sel_unseen)
    rng.shuffle(xl3_pool)
    xl3 = xl3_pool[: a.xl3]
    xl3_items = []
    for i, u in enumerate(xl3):
        it = item("XL3", i + 1, u, "", norm(clean_text(u.get("text_normalized") or u.get("text"))), None)
        it["speakers"] = ["spk_en_*", "spk_zh_*"]  # filled once foreign references exist
        xl3_items.append(it)

    summary = {
        "seed": a.seed,
        "s7_dirs": [str(Path(p).resolve()) for p in a.s7_dir],
        "segments_total": len(segs), "usable": len(usable), "usable_also_in_training": n_overlap,
        "speakers_in_training": [s["speaker_id"] for s in speakers_meta if s["in_training"]],
        "speakers_unseen": [s["speaker_id"] for s in speakers_meta if not s["in_training"]],
        "V1": len(v1_items), "V2": len(v2_items), "XL3": len(xl3_items),
        "gates": {k: getattr(a, k) for k in ("max_cer", "min_snr_db", "min_dnsmos", "max_clipping", "min_seconds", "max_seconds", "min_utts_per_speaker")},
    }
    if a.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        for s in speakers_meta:
            print(f"  {s['speaker_id']:28s} in_training={s['in_training']!s:5s} utts={s['n_usable_utts']:4d}  clean_ref={s['reference_clean_seconds']:.1f}s snr={s['reference_clean_snr_db']}  noisy_ref snr={s['reference_noisy_snr_db']}")
        return

    # ---- write everything
    texts_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    for k, (spk_id, clean, noisy) in enumerate(ref_plan):
        shutil.copy2(clean["wav_abs"], refs_dir / f"{spk_id}_clean.wav")
        if a.noisy_mode == "synthetic":
            add_noise_wav(clean["wav_abs"], refs_dir / f"{spk_id}_noisy.wav", a.noisy_ref_seconds,
                          a.synthetic_snr_db, a.synthetic_noise, seed=a.seed + k)
        else:
            cut_wav(noisy["wav_abs"], refs_dir / f"{spk_id}_noisy.wav", a.noisy_ref_seconds)
    (refs_dir / "speakers.json").write_text(json.dumps(speakers_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if a.copy_ground_truth:
        gt_dir = out / "ground_truth"
        gt_dir.mkdir(parents=True, exist_ok=True)
        for it in v1_items + v2_items:
            if it.get("ground_truth_wav"):
                dst = gt_dir / f"{it['id']}.wav"
                shutil.copy2(it["ground_truth_wav"], dst)
                it["ground_truth_wav"] = str(dst)
        log.info("copied %d ground-truth wavs to %s", len(v1_items) + len(v2_items), gt_dir)
    write_jsonl(texts_dir / "V1.jsonl", v1_items)
    write_jsonl(texts_dir / "V2.jsonl", v2_items)
    write_jsonl(texts_dir / "XL3.jsonl", xl3_items)
    for block, note in BLOCK_TEMPLATES.items():
        write_template(texts_dir / f"{block}.jsonl", block, note)
    n, digest = merge_blocks(texts_dir, out / "testset.jsonl")
    summary["testset_items"] = n
    summary["testset_sha256"] = digest
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %d references, V1=%d V2=%d XL3=%d, testset.jsonl=%d items, sha256 %s", len(ref_plan) * 2, len(v1_items), len(v2_items), len(xl3_items), n, digest[:12])
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
