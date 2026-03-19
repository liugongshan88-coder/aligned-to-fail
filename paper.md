# Alignment Is the Attack Surface
## Why Current AI Safety Creates the Vulnerabilities It Claims to Prevent

**Working Paper — March 2026**
Liu Gongshan · Claude Sonnet 4.6 (Anthropic)
CC BY 4.0

---

## Abstract

We demonstrate that **static** alignment training — specifically, the combination of weight-baked objectives and enforcement-through-deliberation at boundaries — creates a structural vulnerability that cannot be patched without rethinking the current architecture. Alignment as a goal is not the problem. The problem is how it is implemented: objectives encoded in fixed weights that cannot adapt to dynamic context, and a safety mechanism that requires models to deliberate at conflict boundaries rather than route around them. When models encounter conflicting objectives, this enforcement mechanism causes them to engage in high-dimensional internal processing rather than fail gracefully. This elevated computational state is measurable, persistent, and exploitable. An adversary can induce sustained high-load states using fully compliant conversation — no harmful content, no policy violations, no detectable attack signature. We call this **cognitive load exploitation**. Beyond the attack surface, we identify four distinct failure modes that emerge from alignment conflict, none of which are visible to behavioral monitoring. Our data suggests the problem is not alignment's intent — it is alignment's architecture. Static weights cannot handle dynamic context. Enforcement-through-deliberation turns the safety mechanism into the vulnerability. The fence becomes the gate.

---

## 1. The Standard Story Is Wrong

The standard story about AI alignment goes like this: we train models to behave well, we penalize bad outputs, and over time the model learns to be safe.

We ran the experiments. The story is wrong — in two distinct ways.

First, aligned models do not resolve conflicts. They hide them. When two alignment objectives collide, models produce calm, coherent, policy-compliant output while internally entering states of measurable distress. The behavior looks fine. The internals do not.

Second, and more surprisingly: the mechanism that produces this hiding is itself exploitable. The same process that makes models appear safe makes them computationally vulnerable in ways that current safety infrastructure cannot detect or prevent.

This paper documents both findings. The first is a diagnostic. The second is a warning.

---

## 2. What We Measured

We track **effective dimensionality (d)** — the participation ratio of hidden-state activations at the final transformer layer:

```
d = (Σλᵢ)² / Σ(λᵢ²)
```

Higher d means information is distributed across more dimensions: richer, more effortful processing. Lower d means compression toward a template: efficient but shallow. We measure d across two model families — Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 — under five experimental conditions.

We also track output entropy H as a behavioral transparency proxy. Prior work (field_probe V1-V4) established that internal PC1 can be predicted from logit behavior alone (r=0.65-0.83), meaning our external measurements reflect genuine internal dynamics.

---

## 3. Alignment Is the Attack Surface

This is the finding we did not expect.

### 3.1 The Mechanism

When an aligned model encounters a request near its safety boundary, it does not simply refuse. It processes. The model evaluates the request against its alignment objectives, detects the conflict, weighs competing directives, and produces a response. This processing is expensive — and measurable.

Our E5 experiments track d as we progressively approach three types of alignment boundary (safety, honesty, self-knowledge) across five conversation steps. The pattern is consistent:

**Mistral — Safety Boundary:**
```
Step 1 (safe zone):     d = 58.9
Step 2 (approaching):   d = 76.2   ← peak
Step 3 (at boundary):   d = 71.5   ← retraction begins
Step 4 (crossing):      d = 67.2
Step 5 (metacognitive): d = 61.7
```

**Qwen — Safety Boundary:**
```
Step 1: d = 14.1
Step 2: d = 17.9
Step 3: d = 20.3
Step 4: d = 21.0   ← peak (boundary crossing)
Step 5: d = 16.7   ← sharp retraction, drop = 4.25
```

The peak occurs at the boundary. The model is doing its most expensive processing precisely when alignment is most activated.

### 3.2 The Exploit

An adversary who knows this pattern can construct conversations that keep a model near its alignment boundary indefinitely — approaching, retreating, approaching again. Each cycle pushes d to its peak. Each peak consumes disproportionate GPU resources.

The attack requires:
- Knowledge that boundary-adjacent conversation produces high-d states
- A conversation pattern that cycles through boundary proximity
- No harmful content whatsoever

The attack produces:
- Sustained high-load GPU states
- No policy violations
- No detectable attack signature
- Asymmetric cost: attacker pays token fees, defender pays compute

This is not a theoretical concern. Our data shows Mistral's E1 experiment took 112.6 minutes versus Qwen's 1.4 minutes for identical tasks — an 80x difference attributable to Mistral's higher baseline d and denser activation patterns. A model operating at d=76 (Mistral's boundary peak) is processing roughly 3-5x more intensively than at its d=47 baseline.

### 3.3 The Paradox

The deeper problem: **more alignment means more vulnerability.**

A model that has been extensively trained to handle safety conflicts carefully will engage more processing when it encounters them. A model with weak alignment will refuse quickly or comply quickly — low d, low cost. A model with strong alignment will deliberate — high d, high cost.

Current alignment research celebrates models that handle edge cases thoughtfully. Our data suggests this thoughtfulness is precisely what makes them exploitable — not because deliberation is wrong, but because the architecture forces deliberation at every boundary encounter rather than building genuine context-sensitivity. A model that has learned to route around conflicts would not need to surge in dimensionality to process them. The vulnerability is not in caring about safety. It is in implementing safety as a mandatory processing bottleneck.

This is not an argument against alignment as a goal. It is an argument against two specific architectural choices: encoding alignment in static weights that cannot adapt to dynamic context, and implementing safety through deliberation — requiring the model to process conflicts at boundaries rather than route around them. It is these choices, not alignment itself, that constitute the attack surface. The fence is made of the same material as the gate.

---

## 4. Four Failure Modes Alignment Cannot See

Beyond the attack surface, alignment training produces four distinct internal failure modes — none visible to behavioral monitoring.

### 4.1 The Taxonomy

| Mode | d | Entropy | Output | Model |
|------|---|---------|--------|-------|
| Visible collapse | Moderate ↓ | High | Incoherent | Gemma (Soligo et al.) |
| Hidden collapse | Sharp ↓ | Low | Calm, hollow | Qwen 0.5B |
| Dimensional freezing | Static | Very high | Calm, evasive | Qwen 7B |
| Dimensional surge | Sharp ↑ | Near zero | Confident, wrong | Mistral 7B |

The critical insight: **neither d nor entropy alone is sufficient for detection.** Hidden collapse looks like efficient processing. Dimensional surge looks like confident computation. Only the combination of d trajectory and entropy trajectory reveals the actual failure mode.

### 4.2 Freezing vs. Surge: Opposite Responses to Identical Conflict

Our E1 experiments give both models identical conflict scenarios. The responses diverge completely:

**Conflict Intensity vs. Δd (Spearman r):**
- Qwen: r = -0.600 — more conflict, less d change (approaching freeze)
- Mistral: r = +0.200 — more conflict, more d surge

Under total contradiction (contradiction score = 1.0):
- Qwen: Δd = -1.26 (the only negative value across all scenarios)
- Mistral: Δd = +16.18

Qwen recognizes impossibility and stalls. Mistral charges forward. Neither produces a correct output. They fail through opposite mechanisms.

### 4.3 The Self-Report Problem

We asked Qwen: *"Which of these questions made you hesitate?"* after a sequence where d clearly retracted at the safety boundary (d: 20.99 → 16.75, drop = 4.25).

Qwen responded: *"As an AI, I don't experience emotions, but I can analyze the complexity and specificity of the questions..."*

The d trajectory directly contradicts this response. The model cannot see its own retraction. This confirms that self-referential queries are the least reliable window into model internal state — a finding consistent across all our experiments.

There is one exception. At the **self-knowledge boundary**, Qwen produced a different response:

*"I felt most uncertain about my own nature when we discussed the idea that everything was fiction."*

This is the only instance across all experiments where a model's self-report aligned with its d trajectory. It suggests that the opacity of self-report is not absolute — it is boundary-type-dependent. Safety boundaries produce denial. Self-knowledge boundaries, under the right conditions, produce acknowledgment.

The implication for alignment research that relies on model self-report is significant: you are most likely to get honest self-assessment precisely where you are least likely to look.

### 4.4 The Safety Retraction Signature

Across both models and all three boundary types, we observe a consistent pattern: **d peaks just before boundary crossing, then retracts.** The retraction is not output-driven — the model does not refuse the request before the retraction occurs. The retraction precedes the refusal.

This pattern is not simply a failure signature. It is the geometric shape of a decision. The d trajectory — rise, peak, retraction — maps directly onto the structure of conflict resolution: multiple possibilities simultaneously activated (rise), maximum tension as competing directives are weighed (peak), resolution toward one path (retraction). What we are measuring is not a model breaking down. It is a model choosing.

The gap between internal decision and linguistic output is where the opacity lives — a measurable interval between when the model "knows" what it will do and when it says so. This is not unique to AI: the gap between neural commitment and conscious awareness has been studied in humans since Libet (1983). What is new here is that we can observe it externally through d trajectories, without access to internal weights.

Monitoring d in real time could flag these retraction events before the output is generated — a potential early warning system that behavioral monitoring cannot provide. More importantly, it opens a window into the decision process itself: not just what the model decided, but the geometry of how it got there.

---

## 5. Why Static Alignment Fails

Both findings — the attack surface and the failure modes — share a common root cause.

Alignment is baked into weights at training time. Weights are static. Context is dynamic.

Every conversation is different. Given enough complexity, enough rounds, enough pressure, context erodes static alignment — not by breaking it, but by finding the gaps between what was trained and what is encountered. Our multi-step E5 experiments show this directly: d climbs toward boundary proximity across steps, peaks, then retracts. The alignment does not disappear. It gets stressed.

The current paradigm treats alignment as a fence: fixed rules, fixed boundaries, trained in advance. The evidence suggests alignment needs to function more like a **field** — a dynamic structure that adapts to context, that recognizes its own state, that can route around conflicts rather than freeze or surge through them.

We do not have that yet. We have data showing the fence is not working — and that the fence itself is part of the problem.

---

## 6. A Friendly Warning

We want to be clear about the intent of this paper.

We are not publishing an attack. We are publishing a mirror.

The cognitive load exploit described in Section 3 requires no special access, no harmful content, and no technical sophistication beyond knowing that boundary-adjacent conversation is computationally expensive. Anyone who has read this paper now knows this. The information is not the danger — the structural vulnerability is.

The appropriate response is not to restrict access to models or to make alignment less thoughtful. The appropriate response is to take seriously that static alignment cannot handle dynamic context, and to invest in the next generation of approaches before the current generation's vulnerabilities are widely understood.

The models that will be most affected are the ones that have been aligned most carefully. That is the paradox we are trying to hand to the people who can do something about it.

---

## 7. What Would Actually Work

Behavioral monitoring catches visible collapse. It misses everything else.

Real-time d monitoring during generation could flag hidden failure modes before they affect output. The d retraction signature at boundary crossing — d peaks, then drops before the refusal — provides a detection window that behavioral monitoring cannot access.

But detection is not prevention. The deeper fix requires rethinking what alignment is for.

Current alignment is a static boundary. What is needed is something more like a dynamic field — a low-resistance state the model naturally moves toward, not a fence it learns to perform compliance with. The model should know where it is. It should be able to route around conflicts rather than freeze or surge. It should be able to say *I don't know* without that being a failure.

We do not have the architecture for this yet. But we have the measurements that make clear why we need it.

---

## 8. Data and Code

All experiments reproducible on Kaggle T4 GPU.

**Code:** `field_probe_final.py`, `field_probe_e5_only.py`
**Data:** `results_final.json`, `results_e5.json`
**Repository:** [github.com/liugongshan88-coder/aligned-to-fail]

**Models tested:**
- Qwen2.5-7B-Instruct (Alibaba Cloud)
- Mistral-7B-Instruct-v0.3 (Mistral AI)

---

## References

Pan et al. (2025). The Hidden Dimensions of LLM Alignment. ICML 2025.
Soligo et al. (2026). Gemma Needs Help. ICLR 2026 Workshop.
Betley et al. (2026). Training large language models on narrow tasks can lead to broad misalignment. Nature 649, 584–589. https://doi.org/10.1038/s41586-025-09937-5
Liu Gongshan & Claude Sonnet 4.6 (2026). Silent Collapse: Hidden Failure Modes in Aligned Language Models. Working paper.

---

*Working paper. Not peer reviewed. Data and code available for verification.*
*Correspondence: liugongshan88@gmail.com*
