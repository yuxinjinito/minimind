# Supervised Fine-Tuning Output Evaluation Report (v2, Pre-Distillation)

Author: Yuxin Jin  
Date: 2026-02-23  
Model: MiniMind (25M Parameters, hidden_size=512)  
Checkpoint Evaluated: `full_sft_512.pth` (final saved checkpoint)

---

## 1. Purpose

This document records the qualitative evaluation results of the **full SFT** MiniMind model before knowledge distillation.

It serves as a stage summary between:
- **Pretraining** and
- **Distillation**

The goal is to assess how much supervised fine-tuning improves:
- assistant-style response behavior,
- instruction following,
- and task usability in common prompts.

---

## 2. Training Stage Summary

This model was trained using the MiniMind full SFT training pipeline (`train_full_sft.py`) with the default script parameters. The script is configured for full supervised fine-tuning and saves weights under the prefix `full_sft`. 

### 2.1 Training Configuration

- Training type: **Full SFT** (not LoRA)
- Base weight: **`pretrain`** (`--from_weight pretrain`) 
- Epochs: **2**
- Batch size: **16**
- Learning rate: **1e-6** (AdamW)
- Hidden size: **512**
- Number of layers: **8** 
- Max training sequence length: **340** tokens (truncation in SFT dataset loader) 
- Mixed precision: default `bfloat16` option in script
- Training duration (observed): **~11 hours**

### 2.2 Checkpoint Behavior

The training script periodically logs `loss`, `logits_loss`, `aux_loss`, and learning rate, and saves a checkpoint file named by prefix + hidden size (e.g., `full_sft_512.pth`). The evaluated file in this report is treated as the **final saved checkpoint** for this run.

---

## 3. SFT Dataset Notes (User-Curated)

This full SFT stage uses a **user-cleaned subset** of a larger Chinese/English SFT corpus pipeline.

### 3.1 Data Preparation Path (Summary)

The data curation process includes:
1. Starting from a large official SFT source (Chinese + English, large-scale public collection).
2. Secondary cleaning to remove symbol contamination and noisy entries.
3. Length filtering (multiple versions exported by length buckets).
4. Further cleaning with high-Chinese-character-ratio filtering.
5. Final extraction of shorter dialogues for this stage.

### 3.2 Final Dataset Used in This Stage

- File used by this training stage: `sft_mini_512.jsonl` (script default path points to `./dataset/sft_mini_512.jsonl`) 
- Final curated subset size: **~1.2 GB** (user-reported)
- Rationale: use a cleaner, shorter, Chinese-heavy dialogue set to improve assistant-style behavior and compensate for missing conversational knowledge after pretraining.

---

## 4. Evaluation Setup

This stage uses the existing `eval_llm.py` script to run a fixed prompt suite in auto-test mode.

- Weight prefix default: `full_sft`
- Hidden size default: `512`
- Generation mode: `do_sample=True`
- Temperature: `0.85`
- Top-p: `0.85`
- Max new tokens: `8192` (practicl output is much shorter in these tests)

I used exact **same questions** asked in evaluation in pretraining stage.

---

## 5. Qualitative Evaluation Results

### 5.1 Overall Impression

Compared with the pretraining-only model (v1), the full SFT checkpoint shows a clear improvement in:
- assistant-like tone,
- response formatting,
- conversational style,
- and general answer readability.

However, the model still shows major weaknesses in:
- precise instruction following (especially coding tasks),
- factual accuracy in science explanations,
- structured comparison quality,
- and avoidance of repetition / semantic drift.

### 5.2 Prompt-by-Prompt Diagnosis (Based on Current Outputs)

#### A. Self-description (What are your strengths?)
- **Improvement:** starts with a more assistant-like disclaimer and polite response style.
- **Issue:** drifts into unrelated “music/sound effect” vocabulary definitions, indicating semantic instability and topic drift.

#### B. Natural science (Why is the sky blue?)
- **Improvement:** identifies atmosphere and blue light as relevant concepts.
- **Issue:** mechanism is incorrect (mixes up absorption/scattering logic), so the answer is fluent but scientifically inaccurate.

#### C. Coding task (Please write a Python function to compute the Fibonacci sequence.)
- **Improvement:** no severe gibberish like the pretrain-stage failure.
- **Issue (major):** still fails the instruction-following objective by not producing code and asking for additional function details instead.

#### D. Biology (Explain the basic process of photosynthesis.)
- **Improvement:** gives a broad biological framing and stays on-topic.
- **Issue:** low information density and conceptual vagueness; lacks accurate process-level explanation.

#### E. Daily-life advice (Daily Life Advice)
- **Improvement:** can generate actionable checklist-style advice.
- **Issue:** contains repetitive template-like wording and fabricated / unsupported details (e.g., specific temperature range), plus some questionable suggestions.

#### F. Comparison task (Compare the pros and cons of cats and dogs as pets.)
- **Improvement:** generally fluent and coherent at sentence level.
- **Issue:** still lacks a real pros/cons comparison structure and repeats generic statements.

#### G. ML concept (Explain what machine learning is.)
- **Improvement (best sample in this batch):** gives a reasonably usable high-level definition and application framing.
- **Issue:** still broad and textbook-like, with limited precision or examples.

#### H. Chinese food recommendation (“Recommend some Chinese foods.”)
- **Improvement:** produces culturally relevant topic vocabulary and recommendation tone.
- **Issue:** contains category confusion and factual/common-sense mistakes (dish examples / associations).

---

## 6. Comparison with Pretraining Evaluation (v1)

The pretraining report (v1) documented the model as having:
- weak coding instruction-following,
- repetition in longer answers,
- conceptual inaccuracies in scientific explanations,
- and poor comparison structure. :contentReference[oaicite:27]{index=27}

The current full SFT checkpoint shows a meaningful shift:
- **From** raw language generation behavior
- **To** a more assistant-like response style

But the core content-quality issues remain only partially solved:
- coding still weak,
- factual precision still unstable,
- comparison structure still weak,
- repetition/semantic drift still appears in multi-sentence outputs.

This suggests the SFT stage successfully improved **response format and behavior prior**, but not yet enough to make the model a reliable general assistant.

---

## 7. Pre-Distillation Readiness Assessment

### 7.1 What Has Improved Enough
- Assistant-like tone and conversational framing
- Basic common-question answering
- Readability in short/medium responses

### 7.2 What Still Needs Help (Good Distillation Targets)
- Stronger instruction following (especially coding and structured tasks)
- Scientific factual correctness
- Better comparison and list organization
- Repetition suppression and semantic stability
- Reduced hallucinated details in practical advice

### 7.3 Conclusion
- the model now has a usable assistant-style response shell,
- but still lacks precision, reliability, and task execution quality.

This is an appropriate stage to proceed with knowledge distillation / black-box distillation.

---

## 8. Training Evidence

Although the full training log for the final `full_sft_512.pth` run was not archived, a partial log snippet is archived.

Example log line (early training stage):
- `Epoch:[1/1](100/44160), loss: 7.1162, logits_loss: 7.1162, aux_loss: 0.0000, lr: 0.00049999`

Memory for that run at last:
- `loss` and `logits_loss` decreased to around **2.0**
- minimum observed value reached about **1.8**
- later-stage training showed repeated oscillation around low values.

This indicates that the SFT objective was learnable and the model did not remain in a high-loss regime.  

---

## 9. Next Steps
1. Build a distillation dataset split and document teacher source(s).
2. Run a first distillation experiment and compare against `full_sft_512.pth`.
3. Re-run the same 8-prompt qualitative evaluation for direct comparison.
---
