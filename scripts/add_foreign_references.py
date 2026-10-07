"""Add English and Mandarin reference voices to ``testset/references/`` from public datasets.

The Vietnamese references come from ``build_testset_from_s7.py``; the foreign voices needed by
XL3 (foreign voice reading Vietnamese) are not in the podcast data, so this script pulls them:

* **English** (default 1 male + 1 female): LibriTTS-R ``test.clean`` (CC BY 4.0). The parquet
  shards are read in place over HTTPS with DuckDB (range requests on the row groups of the chosen
  speaker, no token, no full download). Gender and SNR per speaker come from the small
  ``ylacombe/libritts_r_tags`` parquet. For each chosen speaker the clean reference is the single
  utterance closest to the 8-12 s window (measured after download, not guessed).
* **Mandarin** (default 1 speaker): AISHELL-3 ``test`` (Apache 2.0); utterances are 2-6 s, so the
  clean reference concatenates consecutive utterances of one speaker (0.3 s gap) until it reaches
  the window. Gender/accent from ``spk-info.txt``; speakers are filtered to ``north`` accent.

Every clip is written as 24 kHz mono 16-bit PCM, level-matched to the Vietnamese ones (RMS -23 dBFS,
peak <= -3 dBFS, close to the s7 loudnorm target); the noisy reference
is synthetic (pink noise at ``--snr-db``) made with ``add_noise_wav`` from the test-set builder,
so the same degradation applies to every language. Entries are appended to
``testset/references/speakers.json`` with ids ``spk_en_<id>`` / ``spk_zh_<id>``.

Usage::

    python benchmark/scripts/add_foreign_references.py [--en-male 1 --en-female 1 --zh 1] [--seed 42]

Needs: requests, numpy, soundfile, scipy (resampling), duckdb, pyarrow. Re-running replaces the foreign entries
(same seed -> same speakers) and leaves the Vietnamese entries untouched.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import random
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
import soundfile as sf

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT / "scripts"))
from build_testset_from_s7 import add_noise_wav  # noqa: E402

log = logging.getLogger("foreign_refs")
HF = "https://huggingface.co/datasets"
LIBRITTS_SHARDS = [f"{HF}/mythicinfinity/libritts_r/resolve/main/data/test.clean/test.clean-0000{i}-of-00003.parquet" for i in range(3)]
LIBRITTS_TAGS = f"{HF}/ylacombe/libritts_r_tags/resolve/main/clean/test.clean-00000-of-00001.parquet"
AISHELL3 = "AISHELL/AISHELL-3"
AISHELL3_RAW = f"https://huggingface.co/datasets/{AISHELL3}/resolve/main"
TARGET_SR = 24000
HEADERS = {"User-Agent": "tts-vi-benchmark/add_foreign_references"}


# --------------------------------------------------------------------------- helpers
def download(url: str) -> bytes:
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.content


def to_mono_24k(data: np.ndarray, sr: int) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != TARGET_SR:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(sr, TARGET_SR)
        data = resample_poly(data, TARGET_SR // g, sr // g).astype(np.float32)
    return data


TARGET_RMS_DBFS = -23.0  # xấp xỉ loudnorm -23 LUFS của s7 (AISHELL-3 gốc rất nhỏ, peak ~0.08)
PEAK_DBFS = -3.0


def write_pcm16(path: Path, x: np.ndarray) -> None:
    """Level-match to the Vietnamese references: RMS -> -23 dBFS, then peak ceiling -3 dBFS."""
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    if rms > 0:
        x = x * (10 ** (TARGET_RMS_DBFS / 20) / rms)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    ceiling = 10 ** (PEAK_DBFS / 20)
    if peak > ceiling:
        x = x * (ceiling / peak)
    sf.write(str(path), x, TARGET_SR, subtype="PCM_16")


def wav_bytes_to_24k(b: bytes) -> np.ndarray:
    data, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    return to_mono_24k(data, sr)


# --------------------------------------------------------------------------- English: LibriTTS-R
def libritts_speakers_by_gender() -> Dict[str, dict]:
    """speaker_id -> {gender, n, snr_mean} from the 1.3 MB tags parquet (no audio)."""
    import pyarrow.parquet as pq

    t = pq.read_table(io.BytesIO(download(LIBRITTS_TAGS)), columns=["speaker_id", "gender", "snr"]).to_pylist()
    acc: Dict[str, dict] = {}
    for row in t:
        s = str(row["speaker_id"])
        a = acc.setdefault(s, {"gender": row.get("gender"), "n": 0, "snr": 0.0})
        a["n"] += 1
        a["snr"] += float(row.get("snr") or 0.0)
    for a in acc.values():
        a["snr_mean"] = a["snr"] / max(1, a["n"])
    return acc


def libritts_pick_clip(speaker_id: str, ref_min: float, ref_max: float, out: Path) -> Optional[dict]:
    """Read the speaker's row groups from the parquet shards (DuckDB over HTTPS), keep the utterance
    closest to the window. Word count prefilters candidates (LibriTTS runs ~2.5-3 words/s)."""
    import duckdb

    con = duckdb.connect()
    urls = ", ".join(f"'{u}'" for u in LIBRITTS_SHARDS)
    rows = con.execute(
        f"SELECT id, text_normalized, audio.bytes AS b FROM read_parquet([{urls}]) "
        f"WHERE speaker_id = ? AND len(string_split(text_normalized, ' ')) BETWEEN 16 AND 40 "
        f"ORDER BY abs(len(string_split(text_normalized, ' ')) - 27), id LIMIT 12", [speaker_id]).fetchall()
    best = None
    for rid, text, b in rows:
        try:
            x = wav_bytes_to_24k(bytes(b))
        except Exception as e:
            log.warning("skip %s: %s", rid, e)
            continue
        r = {"id": rid, "text_normalized": text}
        dur = len(x) / TARGET_SR
        dist = 0.0 if ref_min <= dur <= ref_max else min(abs(dur - ref_min), abs(dur - ref_max))
        if best is None or dist < best["dist"]:
            best = {"dist": dist, "dur": dur, "x": x, "id": r["id"], "text": r["text_normalized"]}
        if dist == 0.0:
            break
    if best is None:
        return None
    write_pcm16(out, best["x"])
    return {"source_utterances": [best["id"]], "text": best["text"], "seconds": round(best["dur"], 2)}


# --------------------------------------------------------------------------- Mandarin: AISHELL-3
def aishell_speakers() -> Dict[str, dict]:
    txt = download(f"{AISHELL3_RAW}/spk-info.txt").decode("utf-8")
    info = {}
    for line in txt.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            info[parts[0]] = {"age": parts[1], "gender": parts[2], "accent": parts[3]}
    content = download(f"{AISHELL3_RAW}/test/content.txt").decode("utf-8")
    utts: Dict[str, List[tuple]] = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        fname, _, trans = line.partition("\t")
        spk = fname[:7]
        hanzi = "".join(t for t in trans.split() if re.match(r"^[^\x00-\x7F]+$", t))
        utts.setdefault(spk, []).append((fname.strip(), hanzi))
    for spk, lst in utts.items():
        if spk in info:
            info[spk]["utts"] = sorted(lst)
    return {s: v for s, v in info.items() if v.get("utts")}


def aishell_build_clip(spk: str, utts: List[tuple], ref_min: float, ref_max: float, out: Path, gap_s: float = 0.3) -> Optional[dict]:
    pieces, ids, texts, total = [], [], [], 0.0
    gap = np.zeros(int(gap_s * TARGET_SR), dtype=np.float32)
    for fname, hanzi in utts[:12]:
        try:
            x = wav_bytes_to_24k(download(f"{AISHELL3_RAW}/test/wav/{spk}/{fname}"))
        except Exception as e:
            log.warning("skip %s: %s", fname, e)
            continue
        if pieces:
            pieces.append(gap)
            total += gap_s
        pieces.append(x)
        ids.append(fname.replace(".wav", ""))
        texts.append(hanzi)
        total += len(x) / TARGET_SR
        if total >= ref_min:
            break
    if not pieces or total < ref_min * 0.8:
        return None
    x = np.concatenate(pieces)
    if total > ref_max:
        x = x[: int(ref_max * TARGET_SR)]
    write_pcm16(out, x)
    return {"source_utterances": ids, "text": " ".join(texts), "seconds": round(len(x) / TARGET_SR, 2)}


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(BENCH_ROOT / "testset"))
    ap.add_argument("--en-male", type=int, default=1)
    ap.add_argument("--en-female", type=int, default=1)
    ap.add_argument("--zh", type=int, default=1)
    ap.add_argument("--ref-min-seconds", type=float, default=8.0)
    ap.add_argument("--ref-max-seconds", type=float, default=12.0)
    ap.add_argument("--noisy-ref-seconds", type=float, default=5.0)
    ap.add_argument("--snr-db", type=float, default=10.0, help="synthetic pink noise level for the noisy reference")
    ap.add_argument("--min-utts", type=int, default=20, help="LibriTTS-R: ignore speakers with fewer test utterances")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    rng = random.Random(a.seed)
    refs_dir = Path(a.out_dir).resolve() / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)
    spk_json = refs_dir / "speakers.json"
    speakers: List[dict] = json.loads(spk_json.read_text(encoding="utf-8")) if spk_json.exists() else []
    speakers = [s for s in speakers if s.get("language") == "vi"]  # foreign entries are rebuilt below
    new_entries: List[dict] = []

    def finish(spk_id: str, lang: str, clean_path: Path, meta: dict, source: dict) -> None:
        noisy_path = refs_dir / f"{spk_id}_noisy.wav"
        add_noise_wav(clean_path, noisy_path, a.noisy_ref_seconds, a.snr_db, "pink", seed=a.seed + len(new_entries))
        new_entries.append({
            "speaker_id": spk_id, "language": lang, **source,
            "reference_text": meta["text"],
            "reference_clean": f"references/{clean_path.name}", "reference_clean_seconds": meta["seconds"],
            "reference_clean_source": meta["source_utterances"],
            "reference_noisy": f"references/{noisy_path.name}", "reference_noisy_seconds": min(meta["seconds"], a.noisy_ref_seconds),
            "reference_noisy_snr_db": a.snr_db, "noisy_mode": "synthetic", "synthetic_noise": "pink", "sample_rate": TARGET_SR,
        })
        log.info("%s: %.1f s clean (%s) + %.0f s noisy", spk_id, meta["seconds"], ",".join(meta["source_utterances"]), a.noisy_ref_seconds)

    # ---- English
    if a.en_male or a.en_female:
        log.info("LibriTTS-R: reading speaker tags (gender, SNR) ...")
        tags = libritts_speakers_by_gender()
        for gender, n_want in (("male", a.en_male), ("female", a.en_female)):
            pool = [s for s, t in tags.items() if t.get("gender") == gender and t["n"] >= a.min_utts]
            pool.sort(key=lambda s: -tags[s]["snr_mean"])
            pool = pool[: max(6, 3 * n_want)]  # cleanest speakers, then random among them
            rng.shuffle(pool)
            got = 0
            for s in pool:
                if got >= n_want:
                    break
                spk_id = f"spk_en_{gender[0]}{s}"
                meta = libritts_pick_clip(s, a.ref_min_seconds, a.ref_max_seconds, refs_dir / f"{spk_id}_clean.wav")
                if meta is None:
                    log.warning("LibriTTS-R speaker %s: no usable clip, trying another", s)
                    continue
                finish(spk_id, "en", refs_dir / f"{spk_id}_clean.wav", meta,
                       {"source_dataset": "mythicinfinity/libritts_r (LibriTTS-R test.clean, CC BY 4.0)", "source_speaker": s,
                        "gender": gender, "source_snr_mean": round(tags[s]["snr_mean"], 1)})
                got += 1

    # ---- Mandarin
    if a.zh:
        log.info("AISHELL-3: reading spk-info and test transcripts ...")
        info = aishell_speakers()
        pool = [s for s, v in info.items() if v.get("accent") == "north" and v.get("age") in ("B", "C") and len(v["utts"]) >= 8]
        pool.sort()
        rng.shuffle(pool)
        got = 0
        for s in pool:
            if got >= a.zh:
                break
            spk_id = f"spk_zh_{s}"
            meta = aishell_build_clip(s, info[s]["utts"], a.ref_min_seconds, a.ref_max_seconds, refs_dir / f"{spk_id}_clean.wav")
            if meta is None:
                log.warning("AISHELL-3 speaker %s: no usable clip, trying another", s)
                continue
            finish(spk_id, "zh", refs_dir / f"{spk_id}_clean.wav", meta,
                   {"source_dataset": "AISHELL/AISHELL-3 (test, Apache 2.0)", "source_speaker": s,
                    "gender": info[s]["gender"], "age_group": info[s]["age"], "accent": info[s]["accent"]})
            got += 1

    speakers += new_entries
    spk_json.write_text(json.dumps(speakers, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %d foreign + %d Vietnamese speakers to %s", len(new_entries), len(speakers) - len(new_entries), spk_json)


if __name__ == "__main__":
    main()
