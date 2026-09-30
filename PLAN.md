# Plan: Confucius4-TTS vs ZONOS2 on Vietnamese voice cloning, code-switching and cross-lingual cloning

**Goal.** Measure three abilities of the two released checkpoints for Vietnamese, then again
after fine-tuning on the same podcast data: how faithfully each clones a Vietnamese voice,
how it reads English mixed into Vietnamese, and how well a voice crosses languages in both
directions. Same text, same references, same metrics, same GPU.

## 1. Conditions

| | Confucius4-TTS | ZONOS2 |
|---|---|---|
| A. Pretrained | `netease-youdao/Confucius4-TTS`, reference mode | `Zyphra/ZONOS2`, speaker embedding |
| B. Fine-tuned | T2S full fine-tune via `config/train_t2s_vi.yaml` | LoRA via `finetune/train.py` |

Inference at each model's server defaults, bf16, fixed seed, 3 samples per item. Vietnamese
text normalized once upstream with the shared `normalize_vietnamese`; both models run with
their own normalizers off. Both receive the same reference clip per speaker; no reference
transcripts are used, since neither released code needs one.

## 2. Test material

**Speakers.** 5 Vietnamese podcast hosts present in the fine-tuning data, 3 Vietnamese
speakers absent from it, 2 English and 1 Mandarin speaker from public data for
cross-lingual. One clean 8–12 s reference per speaker, plus one noisy 5 s reference for the
robustness check.

**Texts, all held out of training.**

| Block | Items | Serves |
|---|---|---|
| V1 Vietnamese in-domain sentences from `val.jsonl` | 100 | cloning |
| V2 Vietnamese read-style, 8–15 s | 50 | cloning, prosody |
| CS1 Vietnamese with single English loanwords, technical | 40 | code-switching |
| CS2 Vietnamese with English proper nouns and brands | 30 | code-switching |
| CS3 Vietnamese with full English clauses of 4+ words | 30 | code-switching |
| XL1 English text | 40 | cross-lingual, Vietnamese voice speaking English |
| XL2 Mandarin text | 20 | cross-lingual, second target |
| XL3 Vietnamese text | 40 | cross-lingual, English and Mandarin voices speaking Vietnamese |

## 3. Metrics per ability

**Voice cloning** (V1, V2; Vietnamese references)
- Speaker similarity: cosine to the reference from WavLM-SV and from SpeechBrain ECAPA, both
  reported so neither model's internal encoder is favoured.
- Intelligibility: CER from PhoWhisper-large.
- Noise leakage: similarity and UTMOS with the noisy reference versus the clean one, same text.
- Consistency: similarity variance across the 3 samples and across sentences of one speaker.
- Blind AB on 40 pairs: "which sounds more like the reference".

**Code-switching** (CS1–CS3; Vietnamese references)
- Mixed error rate from Whisper-large-v3, with CER on Vietnamese spans and WER on English
  spans scored separately after aligning the ASR output to the script.
- English-span language accuracy: Whisper language ID on the cut span, plus a per-word manual
  tag with four labels: correct English, Vietnamese-accented but correct, spelled-out
  letters, wrong or dropped. The manual tag is the metric of record.
- Within-utterance speaker consistency: WavLM similarity between the Vietnamese part and the
  English part of the same output (accent-leakage check).
- Blind AB on 30 pairs: "which reads the English words more naturally for a Vietnamese
  speaker".

**Cross-lingual cloning** (XL1–XL3)
- vi→en and vi→zh: Vietnamese voice, foreign text. Similarity to the Vietnamese reference,
  WER by Whisper for English and CER by Whisper for Mandarin, plus language-ID confidence
  that the output is the target language.
- en→vi and zh→vi: foreign voice, Vietnamese text. Similarity to the foreign reference, CER
  by PhoWhisper, and accent rating by 3 native listeners on a 5-point scale.
- Compare to each model's own same-language baseline so the cross-lingual penalty is a
  delta, not an absolute.

**Cost, recorded alongside:** RTF, first-packet latency, VRAM, and for condition B the
GPU-hours spent.

## 4. Protocol

1. Freeze speakers, references and normalized texts in `testset/testset.jsonl`; hash it; the
   same file drives both models.
2. Run condition A for both models, all blocks; keep audio, parameters and logs under
   `outputs/`.
3. Fine-tune both on the identical training split of the 5 in-domain speakers; log data
   hours and compute.
4. Run condition B for both; unseen Vietnamese speakers and foreign speakers stay zero-shot,
   which shows whether fine-tuning helped generalization or only the trained voices.
5. Score with one script (`scripts/score.py`); bootstrap 95% intervals over items;
   overlapping intervals count as a tie.
6. Listening tests last, raters blind to model, balanced order; sheets under `listening/`.

## 5. Decision rule

One winner per ability, so the outcome may be mixed. Cloning: WavLM similarity and PhoWhisper
CER. Code-switching: English-span accuracy and within-utterance consistency. Cross-lingual:
similarity delta and target-language WER. A lead counts only with non-overlapping intervals
or a listening-test preference above 60%.

## 6. Timeline and risks

Test set and evaluation script one day, zero-shot runs half a day, fine-tunes one to two days
each including data prep, scoring and listening one day; about a week on one A100.

Risks: ZONOS2's fine-tune recipe is untested on GPU, keep a debugging day; ASR scoring of
English spans inside Vietnamese is noisy, hence the manual four-label tag; the Mandarin
block depends on Whisper's Mandarin CER being reliable, drop it if not; Confucius4-TTS's
data loader silently substitutes bad rows, so verify `summary.json` counts before training.
