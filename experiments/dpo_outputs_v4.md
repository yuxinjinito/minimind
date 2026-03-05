# DPO Stage Evaluation Report (v4, Direct Preference Optimization)

Author: Yuxin Jin
Date: 2026-02-25
Project: MiniMind v4 (25M Parameters, hidden_size=512)

1. Purpose

This report documents the v4 DPO (Direct Preference Optimization) stage of MiniMind.

v4 is a **parallel branch** from the same full SFT base as v3 (LoRA). The two stages are independent:

- v3: full SFT → LoRA domain adaptation (identity / medical)
- v4: full SFT → DPO preference alignment

```
v1 (pretrain)
  └→ v2 (full SFT)
       ├→ v3 (LoRA identity / medical)
       └→ v4 (DPO)  ← this report
```

The goal is to evaluate whether DPO-based preference alignment can improve response quality (reduce repetition, improve style, reduce hallucination) on the same base model, and to compare the effect with v3's LoRA adaptation.

2. Stage Decision and Rationale

v3 showed that LoRA can shift domain-specific response style, but with cross-domain contamination and limited factual improvement.

For v4, DPO was chosen because it:

- directly optimizes for preferred vs rejected response pairs
- does not require a separate reward model (unlike PPO-based RLHF)
- is compatible with the existing `train_dpo.py` script in the repo
- allows a clean parallel comparison with v3 on the same base

3. DPO Training Setup
3.1 Training Configuration

- Algorithm: **DPO** (Direct Preference Optimization)
- Base weight: **`full_sft`** (`full_sft_512.pth`)
- Reference model: same `full_sft` checkpoint (frozen)
- Epochs: **1**
- Batch size: **4**
- Learning rate: **4e-8** (AdamW, intentionally conservative to avoid catastrophic forgetting)
- Beta (DPO temperature): **0.1**
- Hidden size: **512**
- Number of layers: **8**
- Max sequence length: **1024** tokens
- Mixed precision: `bfloat16`
- Gradient clipping: **1.0**
- Accumulation steps: **1**

3.2 DPO Dataset

- File: `dataset/dpo.jsonl`
- Size: **17,166** chosen/rejected pairs
- Language: **primarily English**
- Format: each entry contains a `chosen` and `rejected` conversation (user + assistant turn)
- Source: general-purpose preference pairs covering diverse topics (business planning, movie analysis, advice, etc.)

3.3 Training Log Summary

(To be filled with loss curve observations after training.)

4. Evaluation Setup

Same evaluation script and decoding settings as v1–v3:

- Script: `eval_llm.py`
- Weight prefix: `dpo` (for post-DPO) / `full_sft` (for baseline)
- LoRA: **None** (not loaded)
- Generation mode: `do_sample=True`
- Temperature: `0.85`
- Top-p: `0.85`
- Seed: `2026` (deterministic)
- Same 8 fixed prompts as v1/v2

5. v4 Evaluation Worklog
5.1 Pre-DPO Baseline (full SFT)

The baseline model (`full_sft_512.pth`) was tested on all 8 prompts. This is the same checkpoint as v3's pre-LoRA baseline.

Observed baseline pattern:

- Self-description: generic, mentions "no human specialty," provides broad goal statement.
- Sky: partially correct (mentions scattering) but contains factual error ("蓝色光波看起来是黄色的").
- Fibonacci: complete failure, incoherent math-like text, no code generated.
- Photosynthesis: covers broad concepts, repetitive across paragraphs but stays on-topic.
- Rain advice: mostly reasonable but contains contradictions ("阳光明媚" in a rain context, "不要在雨天出门").
- Cats vs dogs: generic, no real comparison structure.
- Machine learning: decent high-level definition, best response in the batch.
- Food: lists dishes but repeats "糖醋排骨" three times.

5.2 After DPO

The DPO-trained model (`dpo_512.pth`) was tested on the same 8 prompts.

**A. Self-description (你有什么特长？)**
- **Change:** added specific capability list (NLP, image recognition, machine translation, intelligent Q&A) and a sentence about continuous improvement.
- **Assessment:** minor improvement in specificity. Style remains similar.

**B. Sky color (为什么天空是蓝色的)**
- **Change:** severe regression. Output repeats "分子间相互作用" (molecular interaction) multiple times in a loop pattern.
- **Assessment:** DPO introduced or amplified repetition for this prompt. Factual content degraded significantly.

**C. Fibonacci (请用Python写一个计算斐波那契数列的函数)**
- **Change:** still no code generated. Mentions "Python" and "function" more explicitly but remains incoherent. Contains garbled text (`j�、wo`), suggesting encoding or tokenization artifacts.
- **Assessment:** no meaningful improvement. Garbled output is a new issue.

**D. Photosynthesis (解释一下"光合作用"的基本过程)**
- **Change:** identical output to baseline.
- **Assessment:** DPO had zero effect on this prompt.

**E. Rain advice (如果明天下雨，我应该如何出门)**
- **Change:** identical output to baseline.
- **Assessment:** DPO had zero effect on this prompt.

**F. Cats vs dogs (比较一下猫和狗作为宠物的优缺点)**
- **Change:** identical output to baseline.
- **Assessment:** DPO had zero effect on this prompt.

**G. Machine learning (解释什么是机器学习)**
- **Change:** identical output to baseline.
- **Assessment:** DPO had zero effect on this prompt.

**H. Food recommendation (推荐一些中国的美食)**
- **Change:** more diverse dish listing (广东菜, 红烧肉, 大董烤鸭, 烤鸡翅, 湘菜土豆). Better recommendation tone and structure. No repeated dish names.
- **Assessment:** noticeable improvement in diversity and naturalness.

5.3 Change Summary Table

| Prompt | Change | Direction |
|--------|--------|-----------|
| Self-description | Added specific capabilities | Minor improvement |
| Sky color | Severe repetition loop | Regression |
| Fibonacci | Still no code, garbled text | No improvement / slight regression |
| Photosynthesis | Identical | No change |
| Rain advice | Identical | No change |
| Cats vs dogs | Identical | No change |
| Machine learning | Identical | No change |
| Food recommendation | More diverse, better tone | Improvement |

Result: 4/8 identical, 2/8 improved, 2/8 regressed.

6. Key Findings
6.1 DPO Had Minimal Impact on Most Prompts

4 out of 8 prompts produced byte-identical outputs before and after DPO training. This suggests the extremely conservative learning rate (4e-8) and the language mismatch (English preference data vs Chinese evaluation prompts) severely limited DPO's effect on the model's Chinese generation behavior.

6.2 Where DPO Helped: Diversity and Specificity

The two prompts that improved (self-description, food recommendation) both showed the same pattern: more specific content and less repetition. This is consistent with DPO's intended effect of preferring richer, more informative responses over generic ones.

6.3 Where DPO Hurt: Repetition Amplification

The sky-color prompt showed a dramatic repetition loop that did not exist in the baseline. This suggests DPO may have destabilized the model's generation for certain prompt types, possibly because the preference data's distribution conflicted with the model's existing Chinese knowledge patterns.

6.4 Encoding Artifacts

The Fibonacci output contained garbled characters (`j�、wo`), which may indicate tokenizer/encoding edge cases triggered by DPO weight updates. This did not appear in the baseline.

7. Cross-Branch Comparison (v3 LoRA vs v4 DPO)

Both v3 and v4 start from the same full SFT base and are tested on overlapping prompt sets.

| Dimension | v3 (LoRA) | v4 (DPO) |
|-----------|-----------|----------|
| Target | Domain adaptation (identity, medical) | Preference alignment (general) |
| Data language | Chinese | Primarily English |
| Affected prompts | Identity/medical prompts strongly affected | 4/8 unchanged, 2/8 mildly affected |
| Repetition | Increased in identity responses | Amplified in sky-color response |
| Cross-domain side effects | Strong (medical LoRA contaminated identity) | Minimal (most prompts untouched) |
| Style shift | Strong domain style transfer | Weak overall style change |
| Factual improvement | Minimal | Minimal |
| Overall impact magnitude | High (but with side effects) | Very low |

Key contrast: LoRA produced strong but noisy changes with cross-domain contamination. DPO produced almost no change, likely due to language mismatch and ultra-conservative learning rate.

8. What Worked in v4

- The parallel branch design (v3 vs v4 from same base) made comparison clean and interpretable.
- The fixed 8-prompt evaluation set revealed that DPO's effect was near-zero on most prompts, which is itself an informative result.
- The two prompts that did improve (self-description, food) showed DPO's intended direction: more specific, less repetitive.

9. What Did Not Work Well

- **Language mismatch:** the DPO dataset is primarily English, but evaluation is in Chinese. The preference signal likely did not transfer well across languages for a 25M parameter model.
- **Learning rate too conservative:** 4e-8 was chosen to avoid forgetting, but resulted in negligible parameter updates for most generation patterns.
- **Repetition not suppressed:** DPO was expected to reduce repetition (since rejected samples tend to be repetitive), but actually amplified it in one case.
- **No code improvement:** the Fibonacci prompt remains completely broken, and DPO introduced garbled text.

10. Next Steps

- Re-run DPO with a **Chinese preference dataset** (or at least a mixed Chinese/English set) to test whether language alignment is the bottleneck.
- Experiment with a **higher learning rate** (e.g., 1e-7 or 5e-7) to see if stronger updates produce more visible effects without catastrophic forgetting.
- Try **multiple epochs** (2–3) on the current dataset to increase exposure.
- Consider building a small **Chinese DPO dataset** from the v1–v3 evaluation outputs (using the better/worse response pairs already observed).
- Compare with RLAIF: use an LLM-as-judge to generate preference labels on Chinese prompts, then train DPO on those.

11. Summary

v4 demonstrates that DPO with an English-dominant preference dataset and an ultra-conservative learning rate has **near-zero impact** on a 25M Chinese-language model's generation behavior. 4 out of 8 test prompts produced identical outputs, and the two regressions (repetition loop, garbled text) suggest that even minimal DPO updates can destabilize specific generation patterns.

The cross-branch comparison with v3 reveals a clear contrast: LoRA produces strong but noisy domain shifts, while DPO (under these conditions) barely moves the model at all. The most likely bottleneck is the language mismatch between training data and evaluation, not the DPO algorithm itself.

This result motivates a follow-up experiment with language-matched preference data and a moderately higher learning rate.
