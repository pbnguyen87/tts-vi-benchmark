"""Run one model over the benchmark test set (PLAN.md §1-2, §4 step 2/4).

Covers the three abilities by block:

* voice cloning        V1, V2   : Vietnamese text, Vietnamese speakers, clean reference
                                  (+ noisy reference for --noisy-ref-blocks, default V1)
* code-switching       CS1-CS3  : Vietnamese text with English, Vietnamese speakers
* cross-lingual        XL1, XL2 : English / Mandarin text, Vietnamese speakers  (vi -> en/zh)
                       XL3      : Vietnamese text, foreign speakers                (en/zh -> vi)

Each (item, speaker, reference kind) is generated ``--samples`` times with seed 1000+k and
saved as ``<item>__<speaker>__<ref>__s<k>.wav`` under
``outputs/<model>/<condition>/``, plus ``manifest.jsonl`` (one line per file with timing)
and ``run.json`` (model path, git commit, parameters, testset hash). Existing files are
skipped, so an interrupted run can be resumed.

Examples::

    # released checkpoints
    python benchmark/scripts/generate_all.py --model confucius4_tts --condition pretrained
    python benchmark/scripts/generate_all.py --model zonos2 --condition pretrained

    # another checkpoint of the same model, same test set: new label + weights
    python benchmark/scripts/generate_all.py --model confucius4_tts --condition vi_ft_v1 \\
        --t2s-checkpoint ../Confucius4-TTS/checkpoints/t2s_model_vi.safetensors
    python benchmark/scripts/generate_all.py --model zonos2 --condition vi_ft_v1 \\
        --model-path ../ZONOS2/runs/vi_podcast

    # plan only
    python benchmark/scripts/generate_all.py --model zonos2 --condition pretrained --dry-run

Run each model inside its own environment (Confucius4-TTS/.venv or conda env; ZONOS2 via
``uv run``), because their dependency pins differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

BENCH_ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = BENCH_ROOT.parent
REPOS = {"confucius4_tts": STUDY_ROOT / "Confucius4-TTS", "zonos2": STUDY_ROOT / "ZONOS2"}

log = logging.getLogger("generate_all")

# language code of the *text*, per block
BLOCK_LANG = {"V1": "vi", "V2": "vi", "CS1": "vi", "CS2": "vi", "CS3": "vi", "XL1": "en", "XL2": "zh", "XL3": "vi"}
CONFUCIUS_LANG = {"vi": "vi", "en": "en", "zh": "zh"}
ZONOS_LANG = {"vi": "en_us", "en": "en_us", "zh": "cmn"}  # only used when normalization is on


# --------------------------------------------------------------------------- args
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(REPOS))
    ap.add_argument("--condition", required=True, help="free label for the checkpoint, e.g. pretrained, vi_ft_v1")
    ap.add_argument("--testset", default=str(BENCH_ROOT / "testset" / "testset.jsonl"))
    ap.add_argument("--speakers", default=str(BENCH_ROOT / "testset" / "references" / "speakers.json"))
    ap.add_argument("--out-root", default=str(BENCH_ROOT / "outputs"))
    ap.add_argument("--blocks", default="all", help="comma list, e.g. V1,CS1,XL1 (default all present)")
    ap.add_argument("--samples", type=int, default=3, help="samples per (item, speaker, ref); seeds 1000+k")
    ap.add_argument("--noisy-ref-blocks", default="V1", help="blocks also generated with the noisy reference ('' = none)")
    ap.add_argument("--max-items-per-block", type=int, default=0, help="debug: cap items per block (0 = all)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")

    m = ap.add_argument_group("model paths")
    m.add_argument("--model-path", default=None, help="zonos2: HF id or checkpoint dir (default Zyphra/ZONOS2); confucius: config yaml dir root")
    m.add_argument("--t2s-checkpoint", default=None, help="confucius: local T2S safetensors (fine-tuned)")
    m.add_argument("--s2a-checkpoint", default=None, help="confucius: local S2A .pt")
    m.add_argument("--w2v-bert-path", default=None, help="confucius: local w2v-BERT dir (default pretrained/w2v-bert-2.0 if present)")

    g = ap.add_argument_group("generation (server defaults unless given)")
    g.add_argument("--zonos-normalize-foreign", action="store_true", help="zonos2: run its NeMo normalizer for en/zh blocks (text arrives pre-normalized by default)")
    g.add_argument("--confucius-segment-tokens", type=int, default=80)
    g.add_argument("--confucius-config", default=None, help="confucius: inference yaml (default config/inference_config.yaml); e.g. vistral_finetune/config/inference_config_vistral.yaml")
    g.add_argument("--max-seconds", type=float, default=40.0, help="skip generations longer than this when computing RTF stats (sanity only)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


# --------------------------------------------------------------------------- planning
def read_jsonl(p: Path) -> List[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def expand_speakers(item: dict, speakers: Dict[str, dict]) -> List[str]:
    """Resolve item['speakers']; wildcards like 'spk_en_*' expand by language prefix."""
    out = []
    for s in item.get("speakers") or []:
        if s.endswith("*"):
            prefix = s[:-1]
            out += sorted(k for k in speakers if k.startswith(prefix))
        elif s in speakers:
            out.append(s)
        else:
            log.warning("%s: unknown speaker %s, skipped", item["id"], s)
    if not out:  # default: all Vietnamese speakers for vi/CS/XL1/XL2 blocks
        blk = item["block"]
        lang_needed = "vi" if blk != "XL3" else None
        if lang_needed:
            out = sorted(k for k, v in speakers.items() if v.get("language") == lang_needed)
        else:
            out = sorted(k for k, v in speakers.items() if v.get("language") != "vi")
    return out


def plan_jobs(items: List[dict], speakers: Dict[str, dict], a: argparse.Namespace) -> List[dict]:
    blocks = None if a.blocks == "all" else {b.strip() for b in a.blocks.split(",") if b.strip()}
    noisy_blocks = {b.strip() for b in a.noisy_ref_blocks.split(",") if b.strip()}
    per_block: Counter = Counter()
    jobs = []
    for it in items:
        blk = it["block"]
        if blocks and blk not in blocks:
            continue
        if a.max_items_per_block and per_block[blk] >= a.max_items_per_block:
            continue
        per_block[blk] += 1
        for spk in expand_speakers(it, speakers):
            refs = ["clean"] + (["noisy"] if blk in noisy_blocks and speakers[spk].get("reference_noisy") else [])
            for ref in refs:
                for k in range(a.samples):
                    jobs.append({"item": it, "speaker": spk, "ref": ref, "k": k, "seed": 1000 + k,
                                 "lang": it.get("language") or BLOCK_LANG.get(blk, "vi")})
    return jobs


def ref_path(speakers: Dict[str, dict], spk: str, kind: str, speakers_json: Path) -> Path:
    rel = speakers[spk]["reference_clean" if kind == "clean" else "reference_noisy"]
    p = Path(rel)
    return p if p.is_absolute() else (speakers_json.parent.parent / rel).resolve()


def git_commit(repo: Path) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- backends
class ConfuciusBackend:
    def __init__(self, a: argparse.Namespace, device: str):
        repo = REPOS["confucius4_tts"]
        sys.path.insert(0, str(repo / "scripts"))
        sys.path.insert(0, str(repo))
        os.chdir(repo)  # inference_config.yaml paths are repo-relative
        import yaml
        from generate_vi import _patch_local_checkpoints  # reuse the local-checkpoint resolver

        cfg_path = Path(a.confucius_config).resolve() if a.confucius_config else repo / "config" / "inference_config.yaml"
        if not cfg_path.is_absolute():
            cfg_path = (repo / cfg_path).resolve()
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        w2v = a.w2v_bert_path or (str(repo / "pretrained/w2v-bert-2.0") if (repo / "pretrained/w2v-bert-2.0").is_dir() else None)
        if w2v:
            cfg["paths"]["w2v_bert_path"] = w2v
        explicit = {}
        if a.t2s_checkpoint:
            explicit[cfg["paths"]["t2s_checkpoint"]] = str(Path(a.t2s_checkpoint).resolve())
        if a.s2a_checkpoint:
            explicit[cfg["paths"]["s2a_checkpoint"]] = str(Path(a.s2a_checkpoint).resolve())
        _patch_local_checkpoints(explicit)
        tmp = repo / "config" / ".inference_config_benchmark.yaml"
        tmp.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

        from confuciustts.cli.inference import ConfuciusTTS

        self.tts = ConfuciusTTS(config_path=str(tmp), device=device)
        self.sample_rate = self.tts.sample_rate
        self.segment_tokens = a.confucius_segment_tokens
        self.describe = {"model": "confucius4_tts", "config": str(tmp), "source_config": str(cfg_path), "t2s_checkpoint": a.t2s_checkpoint or "HF default",
                         "s2a_checkpoint": a.s2a_checkpoint or "HF default", "sample_rate": self.sample_rate,
                         "sampling": "server defaults: temp 0.8, top_p 0.8, top_k 30, beams 3, rep 10, nfe 25, cfg 0.7",
                         "max_text_tokens_per_segment": self.segment_tokens}

    def generate(self, text: str, lang: str, ref: Path, seed: int, out: Path) -> float:
        import torch
        import torchaudio

        torch.manual_seed(seed)
        audio = self.tts.generate(text, CONFUCIUS_LANG.get(lang, lang), str(ref), raw=True,
                                  max_text_tokens_per_segment=self.segment_tokens, verbose=False)
        torchaudio.save(str(out), audio.cpu(), self.sample_rate)
        return audio.shape[-1] / self.sample_rate


class ZonosBackend:
    def __init__(self, a: argparse.Namespace, device: str):
        repo = REPOS["zonos2"]
        sys.path.insert(0, str(repo / "python"))
        from zonos2.message import TTSSamplingParams
        from zonos2.tts import TTSLLM

        self.model_path = a.model_path or "Zyphra/ZONOS2"
        self.tts = TTSLLM(model_path=self.model_path)
        self.TTSSamplingParams = TTSSamplingParams
        self.sample_rate = 44100
        self.normalize_foreign = a.zonos_normalize_foreign
        self._emb_cache: Dict[str, object] = {}
        self.describe = {"model": "zonos2", "model_path": self.model_path, "sample_rate": self.sample_rate,
                         "sampling": "server defaults: temp 1.15, topk 106, min_p 0.18, rep 1.2/50",
                         "accurate_mode": True, "clean_speaker_background": False,
                         "text_normalization": f"off (foreign: {'on' if self.normalize_foreign else 'off'})"}

    def _embed(self, ref: Path):
        key = str(ref)
        if key not in self._emb_cache:
            self._emb_cache[key] = self.tts.embed_speaker_file(key)
        return self._emb_cache[key]

    def generate(self, text: str, lang: str, ref: Path, seed: int, out: Path) -> float:
        normalize = self.normalize_foreign and lang in ("en", "zh")
        res = self.tts.generate_one(
            text, self.TTSSamplingParams(seed=seed),
            language=ZONOS_LANG.get(lang, "en_us"), text_normalization=normalize,
            speaker_embedding=self._embed(ref), clean_speaker_background=False, accurate_mode=True,
        )
        self.tts.save_audio(res["audio"], str(out))
        return len(res["audio"]) / 4 / self.sample_rate


# --------------------------------------------------------------------------- main
def main() -> None:
    a = parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    testset = Path(a.testset).resolve()
    speakers_json = Path(a.speakers).resolve()
    items = read_jsonl(testset)
    speakers = {s["speaker_id"]: s for s in json.loads(speakers_json.read_text(encoding="utf-8"))}
    testset_hash = hashlib.sha256(testset.read_bytes()).hexdigest()
    jobs = plan_jobs(items, speakers, a)

    out_dir = Path(a.out_root).resolve() / a.model / a.condition
    by_block = Counter(j["item"]["block"] for j in jobs)
    log.info("testset %s (%d items, sha256 %s): %d generations planned %s", testset.name, len(items), testset_hash[:12], len(jobs), dict(by_block))
    if a.dry_run:
        for j in jobs[:5]:
            print(f"  {j['item']['id']}__{j['speaker']}__{j['ref']}__s{j['k']}  lang={j['lang']}  text={j['item']['text'][:60]!r}")
        print(f"  ... total {len(jobs)} -> {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    import torch

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    backend = ConfuciusBackend(a, device) if a.model == "confucius4_tts" else ZonosBackend(a, device)
    log.info("backend ready on %s in %.1fs", device, time.time() - t0)

    run_meta = {
        "model": a.model, "condition": a.condition, "device": device, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "testset": str(testset), "testset_sha256": testset_hash, "speakers_json": str(speakers_json),
        "repo_commit": git_commit(REPOS[a.model]), "benchmark_commit": git_commit(BENCH_ROOT),
        "samples": a.samples, "noisy_ref_blocks": a.noisy_ref_blocks, "blocks": a.blocks, **backend.describe,
    }
    (out_dir / "run.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest_path = out_dir / "manifest.jsonl"
    done = {json.loads(l)["file"] for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()} if manifest_path.exists() else set()
    n_new = n_skip = n_fail = 0
    gen_time = audio_time = 0.0
    with manifest_path.open("a", encoding="utf-8") as mf:
        for i, j in enumerate(jobs, 1):
            it = j["item"]
            fname = f"{it['id']}__{j['speaker']}__{j['ref']}__s{j['k']}.wav"
            out = out_dir / fname
            if fname in done or out.exists():
                n_skip += 1
                continue
            ref = ref_path(speakers, j["speaker"], j["ref"], speakers_json)
            try:
                t1 = time.time()
                dur = backend.generate(it["text"], j["lang"], ref, j["seed"], out)
                dt = time.time() - t1
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                log.warning("FAILED %s: %s", fname, exc)
                mf.write(json.dumps({"file": fname, "item": it["id"], "block": it["block"], "speaker": j["speaker"], "ref": j["ref"],
                                     "sample": j["k"], "seed": j["seed"], "error": str(exc)}, ensure_ascii=False) + "\n")
                mf.flush()
                continue
            n_new += 1
            gen_time += dt
            audio_time += dur
            mf.write(json.dumps({"file": fname, "item": it["id"], "block": it["block"], "speaker": j["speaker"], "ref": j["ref"],
                                 "reference_wav": str(ref), "sample": j["k"], "seed": j["seed"], "lang": j["lang"],
                                 "text": it["text"], "ground_truth_wav": it.get("ground_truth_wav"),
                                 "duration_s": round(dur, 3), "gen_time_s": round(dt, 3)}, ensure_ascii=False) + "\n")
            mf.flush()
            if n_new % 20 == 0 or i == len(jobs):
                log.info("  %d/%d  new %d skip %d fail %d  RTF so far %.2f", i, len(jobs), n_new, n_skip, n_fail, gen_time / max(audio_time, 1e-6))
    run_meta.update({"generated": n_new, "skipped_existing": n_skip, "failed": n_fail,
                     "rtf": round(gen_time / max(audio_time, 1e-6), 3), "audio_seconds": round(audio_time, 1)})
    (out_dir / "run.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("done: %d generated, %d skipped, %d failed, RTF %.2f -> %s", n_new, n_skip, n_fail, run_meta["rtf"], out_dir)


if __name__ == "__main__":
    main()
