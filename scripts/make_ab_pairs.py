"""Build blind AB listening pairs from two benchmark runs (PLAN.md §3, §4 step 6).

For each (item, speaker, ref, sample) present in both runs, one pair is drawn with the
side order randomized. Output under ``listening/<name>/``:

  pairs/<pair_id>_A.wav, <pair_id>_B.wav   copies (converted to 16 kHz mono so raters cannot tell
                                           the models apart by sample rate)
  pairs/<pair_id>_ref.wav                  the reference clip (cloning and cross-lingual questions)
  key.csv                                  pair_id, item, speaker, ref, sample, which run is A/B  -- keep hidden from raters
  sheet.csv                                pair_id, question, answer (empty) for the raters
  README.txt                               instructions and the question for each block group

Question sets (``--question``):
  cloning        V1/V2: "which sounds more like the reference voice"          (default 40 pairs)
  codeswitch     CS1-3: "which reads the English words more naturally"        (default 30 pairs)
  crosslingual   XL1-3: "which sounds more like the reference voice"          (default 30 pairs)
  accent         XL3:   single-stimulus 1-5 Vietnamese accent rating, both runs, no pairing

Example::

    python benchmark/scripts/make_ab_pairs.py --run-a confucius4_tts/pretrained --run-b zonos2/pretrained \
        --question cloning --n 40 --seed 7 --name cloning_A
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = {
    "cloning": ({"V1", "V2"}, "Which clip sounds more like the REFERENCE voice? (A / B / same)", True),
    "codeswitch": ({"CS1", "CS2", "CS3"}, "Which clip reads the English words more naturally for a Vietnamese speaker? (A / B / same)", False),
    "crosslingual": ({"XL1", "XL2", "XL3"}, "Which clip sounds more like the REFERENCE voice? (A / B / same)", True),
    "accent": ({"XL3"}, "Rate the Vietnamese accent of this clip: 1 = strong foreign accent ... 5 = native Vietnamese", False),
}


def load_rows(run: str, root: Path):
    d = root / run
    rows = [json.loads(l) for l in (d / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return d, {(r["item"], r["speaker"], r["ref"], r["sample"]): r for r in rows if "error" not in r}


def to_16k(src: Path, dst: Path) -> None:
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    x, sr = sf.read(str(src), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        x = AF.resample(torch.from_numpy(x), sr, 16000).numpy()
    peak = abs(x).max() or 1.0
    sf.write(str(dst), 0.9 * x / peak, 16000)  # peak-normalize so loudness does not give the model away


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-a", required=True, help="<model>/<condition> under outputs/")
    ap.add_argument("--run-b", help="second run; omit only for --question accent")
    ap.add_argument("--question", choices=list(QUESTIONS), default="cloning")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--ref", default="clean", help="which reference condition to use: clean|noisy|both")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--name", help="folder name under listening/ (default: <question>_<seed>)")
    ap.add_argument("--outputs-root", default=str(BENCH_ROOT / "outputs"))
    ap.add_argument("--listening-root", default=str(BENCH_ROOT / "listening"))
    a = ap.parse_args()

    blocks, question, needs_ref = QUESTIONS[a.question]
    rng = random.Random(a.seed)
    root = Path(a.outputs_root).resolve()
    out = Path(a.listening_root).resolve() / (a.name or f"{a.question}_{a.seed}")
    (out / "pairs").mkdir(parents=True, exist_ok=True)

    dir_a, rows_a = load_rows(a.run_a, root)
    if a.question == "accent":
        cands = [(k, r) for k, r in rows_a.items() if k[0].split("_")[0] in blocks and (a.ref == "both" or k[2] == a.ref)]
        if a.run_b:
            dir_b, rows_b = load_rows(a.run_b, root)
            cands += [(k, r) for k, r in rows_b.items() if k[0].split("_")[0] in blocks and (a.ref == "both" or k[2] == a.ref)]
        rng.shuffle(cands)
        cands = cands[: a.n]
        with (out / "key.csv").open("w", newline="", encoding="utf-8") as kf, (out / "sheet.csv").open("w", newline="", encoding="utf-8") as sf_:
            kw = csv.writer(kf); sw = csv.writer(sf_)
            kw.writerow(["clip_id", "run", "item", "speaker", "ref", "sample", "file"])
            sw.writerow(["clip_id", "question", "rating_1_to_5"])
            for i, (k, r) in enumerate(cands):
                run = a.run_a if k in rows_a and rows_a[k] is r else a.run_b
                cid = f"c{i:03d}"
                to_16k((root / run / r["file"]), out / "pairs" / f"{cid}.wav")
                kw.writerow([cid, run, *k, r["file"]])
                sw.writerow([cid, question, ""])
        (out / "README.txt").write_text(f"Single-stimulus rating. Listen to pairs/<clip_id>.wav and fill rating_1_to_5 in sheet.csv.\n\n{question}\n", encoding="utf-8")
        print(f"{len(cands)} clips -> {out}")
        return

    if not a.run_b:
        ap.error("--run-b is required for AB questions")
    dir_b, rows_b = load_rows(a.run_b, root)
    keys = [k for k in rows_a if k in rows_b and k[0].split("_")[0] in blocks and (a.ref == "both" or k[2] == a.ref)]
    if not keys:
        raise SystemExit("no common items between the two runs for these blocks")
    rng.shuffle(keys)
    # one pair per (item, speaker) first, so the set covers many items before reusing samples
    seen, chosen, rest = set(), [], []
    for k in keys:
        (chosen if (k[0], k[1]) not in seen else rest).append(k)
        seen.add((k[0], k[1]))
    keys = (chosen + rest)[: a.n]

    with (out / "key.csv").open("w", newline="", encoding="utf-8") as kf, (out / "sheet.csv").open("w", newline="", encoding="utf-8") as sf_:
        kw = csv.writer(kf); sw = csv.writer(sf_)
        kw.writerow(["pair_id", "item", "speaker", "ref", "sample", "A_is", "B_is", "file_A", "file_B", "text"])
        sw.writerow(["pair_id", "question", "answer_A_B_same"])
        for i, k in enumerate(keys):
            pid = f"p{i:03d}"
            ra, rb = rows_a[k], rows_b[k]
            flip = rng.random() < 0.5
            first, second = ((a.run_b, dir_b, rb), (a.run_a, dir_a, ra)) if flip else ((a.run_a, dir_a, ra), (a.run_b, dir_b, rb))
            to_16k(first[1] / first[2]["file"], out / "pairs" / f"{pid}_A.wav")
            to_16k(second[1] / second[2]["file"], out / "pairs" / f"{pid}_B.wav")
            if needs_ref and ra.get("reference_wav"):
                to_16k(Path(ra["reference_wav"]), out / "pairs" / f"{pid}_ref.wav")
            kw.writerow([pid, *k, first[0], second[0], first[2]["file"], second[2]["file"], ra.get("text", "")])
            sw.writerow([pid, question, ""])
    (out / "README.txt").write_text(
        "Blind AB test. For each pair_id listen to pairs/<pair_id>_A.wav and _B.wav"
        + (" after the reference pairs/<pair_id>_ref.wav" if needs_ref else "")
        + f", then write A, B or same in sheet.csv.\n\n{question}\n\nDo not open key.csv.\n",
        encoding="utf-8",
    )
    print(f"{len(keys)} pairs ({a.run_a} vs {a.run_b}, blocks {sorted(blocks)}) -> {out}")


if __name__ == "__main__":
    main()
