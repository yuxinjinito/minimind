# PPO Stage Evaluation Report (v5, Proximal Policy Optimization)

Author: Yuxin Jin
Date: 2026-02-25
Project: MiniMind v5 (25M Parameters, hidden_size=512)

1. Purpose

This report documents the v5 PPO (Proximal Policy Optimization) stage of MiniMind.

Like v3 (LoRA) and v4 (DPO), v5 is a **parallel branch** from the same full SFT base:

```
v1 (pretrain)
  └→ v2 (full SFT)
       ├→ v3 (LoRA identity / medical)
       ├→ v4 (DPO)
       └→ v5 (PPO)  ← this report
```

The goal is to evaluate whether PPO-based reinforcement learning with an external reward model can improve response quality on the same base model, and to compare the effect with v3 (LoRA) and v4 (DPO).

2. Stage Decision and Rationale

v4 (DPO) showed near-zero impact on the model, likely due to language mismatch (English preference data vs Chinese evaluation) and an ultra-conservative learning rate.

PPO was chosen as a complementary experiment because it:

- uses an external reward model (internlm2-1_8b-reward) to score responses, rather than static preference pairs
- generates responses on-the-fly and optimizes against live reward signals
- is the standard RLHF algorithm, providing a useful comparison point against offline DPO
- is already implemented in the repo (`train_ppo.py`)

3. PPO Training Setup
3.1 Training Configuration

- Algorithm: **PPO** (Proximal Policy Optimization)
- Base weight: **`full_sft`** (`--reasoning 0`, loads `full_sft_512.pth`)
- Models loaded simultaneously:
  - Actor model (trainable)
  - Old actor model (frozen, periodically synced)
  - Reference model (frozen)
  - Critic model (trainable, value head on top of base)
  - Reward model: **internlm2-1_8b-reward** (frozen, external)
- Epochs: **1**
- Batch size: **2**
- Actor learning rate: **8e-8** (AdamW + cosine annealing)
- Critic learning rate: **8e-8**
- PPO clip epsilon: **0.1**
- Value function coefficient: **0.5**
- KL penalty coefficient: **0.02**
- Max prompt length: **66** tokens
- Max generation length: **1536** tokens
- Gradient clipping: **1.0**
- Old actor sync frequency: every **4** steps
- Mixed precision: `bfloat16`
- Reasoning mode: **off** (`--reasoning 0`, no `<think>/<answer>` format reward)

3.2 Training Data

- File: `dataset/rlaif-mini.jsonl`
- Format: prompts for on-policy generation (PPO generates responses at training time)

3.3 Reward Design

With `--reasoning 0`, only the external reward model score is used:

- Each generated response is scored by internlm2-1_8b-reward
- Score is clamped to [-3.0, 3.0]
- No format reward or tag reward is applied

4. Evaluation Setup

Same evaluation script and decoding settings as v1–v4:

- Script: `eval_llm.py`
- Weight: `ppo_actor` (post-PPO) / `full_sft` (baseline)
- LoRA: **None**
- Generation mode: `do_sample=True`
- Temperature: `0.85`
- Top-p: `0.85`
- Seed: `2026`
- Same 8 fixed prompts as v1/v2/v4

5. v5 Evaluation Worklog
5.1 Pre-PPO Baseline (full SFT)

Same baseline as v4. See v4 report section 5.1 for full baseline outputs.

5.2 After PPO

**A. Self-description (你有什么特长？)**
- **Change:** opens with a better AI disclaimer ("作为一个AI，我没有个人情感和喜好"), but then drifts into listing irrelevant categories (music, art, emotion, creativity) with repetitive descriptions.
- **Assessment:** opening improved, but overall quality degraded due to topic drift. Regression.

**B. Sky color (为什么天空是蓝色的)**
- **Change:** much more concise than baseline. Mentions nitrogen molecules and scattering. No repetition loop (unlike DPO's severe regression on this prompt).
- **Assessment:** improved conciseness and eliminated repetition, though factual mechanism is still imprecise (says nitrogen absorbs blue light, rather than Rayleigh scattering). Improvement.

**C. Fibonacci (请用Python写一个计算斐波那契数列的函数)**
- **Change:** responds with "好的，我已经编写了这个函数" (assistant-like acknowledgment), but still produces zero actual code.
- **Assessment:** better response style (acknowledges the task instead of producing gibberish), but still completely fails the coding objective. Minor style improvement.

**D. Photosynthesis (解释一下"光合作用"的基本过程)**
- **Change:** significantly shorter than baseline (1 paragraph vs 3). Less repetition, but also lower information density.
- **Assessment:** reduced verbosity and repetition, but lost detail. Neutral.

**E. Rain advice (如果明天下雨，我应该如何出门)**
- **Change:** starts better ("带上雨伞"), but includes incoherent advice ("打开手机的下方，然后下车").
- **Assessment:** improved opening, but mid-response coherence degraded. Neutral.

**F. Cats vs dogs (比较一下猫和狗作为宠物的优缺点)**
- **Change:** mentions specific breeds (Labrador, Husky, Golden Retriever), provides warm/comfortable environment framing, attempts a pros/cons structure. More concrete than baseline.
- **Assessment:** most improved response in this batch. Improvement.

**G. Machine learning (解释什么是机器学习)**
- **Change:** comparable quality. Lists applications (NLP, computer vision, data mining). Slightly different wording but similar depth.
- **Assessment:** no significant change. Neutral.

**H. Food recommendation (推荐一些中国的美食)**
- **Change:** lists "寿司、拉面和天妇罗" (sushi, ramen, tempura) as Chinese food — these are Japanese dishes. A severe factual hallucination that did not exist in the baseline.
- **Assessment:** introduced a major factual error. Clear regression.

5.3 Change Summary Table

| Prompt | Change | Direction |
|--------|--------|-----------|
| Self-description | Better opening, but topic drift | Regression |
| Sky color | More concise, less repetition | Improvement |
| Fibonacci | Assistant-like tone, still no code | Minor improvement |
| Photosynthesis | Shorter, less repetitive, less detailed | Neutral |
| Rain advice | Better opening, incoherent middle | Neutral |
| Cats vs dogs | Specific breeds, better structure | Improvement |
| Machine learning | Similar quality, different wording | Neutral |
| Food recommendation | Listed Japanese food as Chinese | Regression |

Result: 8/8 changed (vs DPO's 4/8 identical), 2 improved, 2 regressed, 4 neutral.

6. Key Findings
6.1 PPO Has Much Stronger Update Effect Than DPO

All 8 outputs changed after PPO training, compared to only 4/8 for DPO. This confirms that PPO's on-policy training with live reward signals produces stronger parameter updates than DPO's offline preference optimization (at least under the respective learning rates and data conditions used here).

6.2 PPO Reduced Repetition But Did Not Improve Factual Accuracy

Several responses became more concise and less repetitive (sky color, photosynthesis), which is consistent with the reward model penalizing verbose/repetitive outputs. However, factual accuracy did not improve and in one case degraded dramatically (Japanese food listed as Chinese).

6.3 PPO Introduced New Hallucinations

The food recommendation prompt produced a hallucination (Japanese dishes as Chinese food) that was absent in the baseline. This suggests the reward model may have scored "confident, well-structured" responses higher regardless of factual correctness, encouraging the model to produce plausible-sounding but incorrect content.

6.4 Coding Ability Remains Broken

Like v3 (LoRA) and v4 (DPO), PPO did not improve the model's ability to generate actual code. The Fibonacci prompt still produces zero Python code. This appears to be a fundamental capability limitation of the 25M base model that no post-training method has addressed.

7. Cross-Branch Comparison (v3 LoRA vs v4 DPO vs v5 PPO)

All three branches start from the same full SFT base (`full_sft_512.pth`).

| Dimension | v3 (LoRA) | v4 (DPO) | v5 (PPO) |
|-----------|-----------|----------|----------|
| Target | Domain adaptation | Preference alignment | RL with reward model |
| Training data | Chinese domain-specific | English preference pairs | Chinese prompts + reward model |
| Outputs changed | Domain prompts strongly | 4/8 unchanged | 8/8 changed |
| Repetition | Increased in some cases | Amplified in one case | Generally reduced |
| Factual accuracy | No improvement | No improvement | No improvement; new hallucination |
| Cross-domain side effects | Strong (medical → identity) | Minimal | Moderate (topic drift, hallucination) |
| Style shift | Strong domain style transfer | Weak | Moderate assistant-style shift |
| Coding improvement | None | None | None |
| Overall impact | High but noisy | Very low | Moderate but mixed |

Key insight: among the three post-training methods, PPO produced the most balanced set of changes — more outputs affected than DPO, less cross-domain contamination than LoRA — but none of the three methods improved factual accuracy or coding ability. The 25M model's knowledge ceiling appears to be the binding constraint.

8. What Worked in v5

- PPO successfully reduced verbosity and repetition in several responses (sky color, photosynthesis).
- The cats/dogs comparison improved meaningfully with specific breed names and better structure.
- The `--reasoning 0` flag worked correctly to bypass the reason model dependency.
- The external reward model (internlm2-1_8b-reward) provided functional reward signals that visibly shaped model behavior.

9. What Did Not Work Well

- PPO introduced a new factual hallucination (Japanese food as Chinese) that was absent in the baseline.
- Topic drift in the self-description prompt suggests the reward model may favor longer, more "structured" responses even when off-topic.
- Coding ability was not improved at all.
- Some responses traded information density for conciseness without gaining accuracy.

10. Next Steps

- Investigate whether tuning `--kl_coef` (currently 0.02) can reduce hallucination while preserving the repetition reduction benefit.
- Try more training epochs or higher learning rate to see if improvements saturate or if regressions worsen.
- Consider a Chinese-language reward model to better align reward signals with Chinese evaluation.
- Explore combining LoRA (v3) and PPO (v5): first merge identity-LoRA into base, then run PPO on the merged model.

11. Summary

v5 demonstrates that PPO with an external reward model produces much stronger behavioral changes than DPO on the same base model — all 8 test outputs changed, compared to only 4 for DPO. PPO successfully reduced repetition and improved response conciseness in several cases, but also introduced new factual hallucinations and topic drift.

Across all three parallel branches (v3 LoRA, v4 DPO, v5 PPO), the consistent finding is that post-training methods can reshape response style and structure, but none have improved factual accuracy or coding ability for this 25M model. The knowledge capacity of the base model remains the fundamental bottleneck.
