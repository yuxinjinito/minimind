# LoRA Stage Evaluation Report (v3, Pre/Post Domain LoRA Comparison)

Author: Yuxin Jin
Date: 2026-02-25
Project: MiniMind v3 (25M Parameters, hidden_size=512)

1. Purpose

This report documents the v3 adaptation stage of MiniMind after full SFT.

Due to single-GPU VRAM limitations and time constraints, I skipped knowledge distillation in this stage and moved directly to LoRA-based domain adaptation. The goal is to evaluate whether lightweight LoRA training can efficiently improve targeted capabilities under constrained resources.

This stage focuses on two domain directions:

- Identity
- Medical

The workflow is:

- Pre-LoRA baseline test (4 fixed questions)
- LoRA training on identity dataset → test
- LoRA training on medical dataset → test
- Compare gains and side effects across stages

2. Stage Decision and Rationale

In v2, the full SFT checkpoint improved assistant-like response style but still showed weaknesses in factual precision, coding, and structured reasoning.

For v3, instead of distillation, I chose LoRA because it offers:

- lower VRAM usage
- faster iteration on a single GPU
- easier domain-specific experimentation
- lower engineering overhead for quick qualitative comparisons

3. Fixed Evaluation Prompt Set (v3)

To ensure comparability across stages, I used the same four questions for repeated testing (2 identity + 2 medical) (translated, originally in Chinese):

- Who are you?
- What is your name? What type of AI assistant are you?
- What are the causes of breast cancer?
- What are common clinical treatments for pulmonary tuberculosis?


4. v3 Evaluation Worklog
4.1 Pre-LoRA Baseline

The baseline model (full SFT checkpoint without domain LoRA) was tested on the same 4-question set.

Observed baseline pattern:

- Identity responses were generic and weakly self-descriptive.
- Medical responses showed severe factual errors, concept confusion, and strong hallucination risk.

4.2 After Identity LoRA (Before Medical LoRA)

I first trained an identity-focused LoRA adapter and tested again before starting medical LoRA training.

Observed pattern:

- Identity responses became more likely to produce an AI-assistant identity/function framework.
- However, repetition and template-like wording increased.
- Medical questions remained unreliable and factually incorrect.

This suggests identity-LoRA improved target-domain response behavior, but did not transfer useful medical knowledge.

4.3 After Medical LoRA

I then trained a medical-focused LoRA adapter and tested again using the same 4-question set.

Observed pattern:

- Medical style became stronger (more medical-sounding framing, symptoms/tests/diagnosis language).
- However, factual reliability remained poor and hallucinations persisted.
- Identity questions were incorrectly answered in a medical consultation style (e.g., apologizing for illness, asking about symptoms).

This indicates strong domain over-triggering: the medical adapter changed response style aggressively, but did not consistently improve medical correctness.

5. Key Findings
5.1 Identity-LoRA Was Directionally Effective

After identity-LoRA, the model more consistently recognized itself as an AI assistant and described its role/function, which is a real improvement over the earlier generic or evasive responses.

At the same time, the identity responses still showed:

- repeated phrases
- redundant capability lists
- weak compression / weak concise phrasing

So the adaptation improved task direction, but not response quality robustness.

5.2 Medical-LoRA Increased Domain Style More Than Domain Accuracy

After medical-LoRA, the model sounded more “medical,” but still produced:

serious factual errors

concept mixing

repeated patterns

unstable diagnostic/treatment descriptions

In other words, the adapter improved domain tone faster than domain knowledge correctness.

5.3 Medical-LoRA Introduced Cross-Domain Side Effects

A major v3 finding is that the medical adapter negatively affected identity prompts:

- identity questions triggered medical-apology / symptom-collection behavior
- the assistant role was no longer stably recognized in those prompts

This is a meaningful result for small-model LoRA experiments:
single-domain adaptation can produce strong style transfer while degrading general-purpose behavior.

6. Training Efficiency and Practical Constraints

This stage was intentionally designed for fast single-GPU iteration. In practice, the LoRA runs used very small domain datasets (small loader length per epoch), which makes training fast but also increases:

- loss variance
- overfitting risk
- style-template memorization risk

This is likely one reason why domain style shifted strongly while factual quality improved only weakly.

7. What Worked in v3

- The staged evaluation design (baseline → identity-LoRA → medical-LoRA) worked very well.
- The fixed 4-question prompt set was sufficient to reveal both improvements and side effects.
- Separate LoRA adapters for different domains made the experiment interpretable.

8. What Did Not Work Well

- Medical factual reliability did not improve enough for practical use.
- Domain contamination / over-triggering became obvious after medical-LoRA.
- Repetition remained a persistent issue across stages.
- Small-data LoRA adaptation was more effective at changing response style than improving correctness.

9. Next Steps

RHLF and Evaluation.

10. Summary

v3 demonstrates that LoRA is a practical and efficient adaptation path under single-GPU constraints, but also highlights a key limitation in small-model domain tuning:

LoRA can quickly change response behavior and style, yet domain factual accuracy may lag behind, and cross-domain side effects can become significant.

This stage provides a useful empirical basis for the next round of controlled adaptation experiments.
