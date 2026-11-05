# Building on Weight-Space Programming: eight extension seeds

Research-advisor memo, 2026-11-05. Companion to `wsp-seed-report.html`, which is a skeptical, control-oriented report.
This memo starts from the opposite stance: take the WSP result at face value (a frontier LLM, inside a harness, can
read and write working weights), and ask where it goes next.

Sources read for this memo: `README.md`, `artifacts/weight-space-programming.html` and `artifacts/wsp-seed-report.html`
(as text, tags stripped), `lm_understand-logs/README.md`, `harness/sandbox.py` (header and prelude), and the READMEs
of Experiments 1, 4, 5, 6, 7 and 8. I did not open every run.

What this memo does not repeat: the skeptical report's Seeds C (scripted loop surgery), A (weights vs code),
D (powered reading benchmark), B (LLM-written init then PPO), E (causal-patch auditing) and F (SPD components). Where
one of them is a useful arm or prerequisite, it is named as such.

## What the repo gives you for free

These facts shape the "first experiment" for each seed.

| Asset | In repo? | Notes |
|---|---|---|
| Exp 1 game policies (500-32-32-5, 4 games) | Weights yes (`exp1_which_game/networks/*.npz`) | The 10x10 grid games are the authors' own; their env code is **not** in the repo. Use MinAtar (public) for anything that needs rollouts. |
| Exp 3 char-transformer blocks | Yes (`exp3_language/networks/*/blocks.npz`, embeddings removed) | Not runnable end to end without embeddings. Train your own tiny models for write tasks. |
| Exp 4 Kinetix actor, before and after the edit | Yes (`exp4_fix_car/networks/ppo_final_seed0.npz`, `edited_actor.npz`) | Actor only. A critic must be re-fitted before PPO can resume from either. |
| Exp 5 written MinAtar networks | Yes (`exp5_write_from_scratch/networks/<game>/hand.npz`) | PPO checkpoints are not included; the PPO recipe is in the README. |
| Exp 6 Go networks + sandbox prelude | Yes (40 MB) | `harness/sandbox.py` is a reusable bubblewrap sandbox with `load/save` over named tensors. |
| Exp 7 Gemma fine-tunes | No (17 GB), rebuildable from data + script + hashes | |
| Exp 8 edit diffs + `apply_edit` script | Yes | Forward-pass tool design (generation or teacher-forced read, budgeted) is documented in the README. |

Measured cost anchors (from `run.json` files): Exp 1 about 74k input / 1.7k output tokens per read; Exp 4 about
10k output tokens for the one-shot repair; Exp 5 21k to 29k output tokens per network; Exp 6 at 50 passes about
8.1M cumulative input tokens (7.8M cached) and 92k output tokens over 90 turns; Exp 8 at 50 passes 158 turns over
about 20 hours at `max` effort.

---

## Seed 1. Weight programs with knobs (LLM writes the structure, an optimizer tunes declared parameters)

**Idea.** In Exp 5 the LLM writes every weight once and cannot test them. Seaquest (53.1) and Asterix (53.2) stay
below PPO at 500M steps. Instead, have the LLM write a *parameterized* network: the full weight layout plus a short
list of named scalar "knobs" (for example "gain of the dodge path", "threshold of the shoot unit", "bias of the
surface-for-air unit"), each tied to a group of weights by an expression the harness can evaluate. A cheap black-box
or gradient optimizer then tunes only the knobs (8 to 64 numbers) against return. The LLM supplies the program sketch;
the optimizer fills the constants. This is program sketching applied to weights.

**Extends.** Exp 5 directly (same games, same 600-32-32-6 tanh network, same scoring). It differs from Seed B in the
skeptical report: B lets PPO move every weight; here the search space is the LLM's own low-dimensional, named
subspace, so the result stays readable and the LLM can be asked to revise the sketch after seeing which knobs moved.

**First experiment (2 days).** Seaquest and Asterix in MinAtar 1.0.15. Extend the Exp 5 "edits" language with a
`knob name = default [lo, hi]` declaration and allow weight statements to reference knobs (`W1[i,j] = 3*k_dodge`).
Tune knobs with CMA-ES (population 16, 64 episodes per candidate) or with PPO on the knob vector only. Budget:
5M environment steps. Then one LLM revision round: show the LLM the tuned knob values and per-episode death causes,
let it rewrite the sketch, re-tune. 5 independent LLM replies per game.

**Metric.** Mean return over 1,000 episodes (Exp 5 protocol) against environment steps; area under the curve.

**Baselines.** The untuned written network (Exp 5 numbers); PPO from random init at the same step budget and at
100M/500M steps (Exp 5 numbers); CMA-ES over a random 64-dim subspace of the same network (intrinsic-dimension
style), which isolates the value of the LLM choosing the subspace.

**Main risk.** The LLM may not expose the right knobs; failures may come from missing features, not wrong constants.
The revision round is there to test whether it can add structure when told what goes wrong.

**Closest prior work.** Tracr (Lindner et al. 2023, compiles programs to weights, no tuning); Learning Transformer
Programs (Friedman, Wettig, Chen, NeurIPS 2023; verified this session: trained transformers converted to discrete
programs, the reverse direction); Evolution Strategies as a scalable alternative to RL (Salimans et al. 2017; not
re-verified); Measuring the Intrinsic Dimension of Objective Landscapes (Li et al. 2018; not re-verified); OPRO, LLMs
as optimizers (Yang et al. 2023; not re-verified). I found nothing on LLM-declared parameter subspaces for weight
tuning, but I only ran standard search.

**Cost.** About 5 x 2 games x 2 rounds x 30k output tokens = 600k output tokens. CMA-ES on MinAtar: a few GPU-hours
with a JAX/gymnax MinAtar port, or CPU-only with the official package.

---

## Seed 2. Plateau surgeon: LLM weight edits inside a stalled RL run

**Idea.** Exp 4's real story is that PPO sat in a local minimum for 1B steps and a single LLM edit got out of it.
Turn this into a training-loop component: run PPO; when a plateau detector fires, pause, show the LLM the weights
plus a few failed episodes (exactly the Exp 4 prompt), apply its edit, and resume PPO. The question is whether the
edit is a lasting escape (PPO keeps improving from the new basin and eventually absorbs the hand path into the trained
weights) or a brittle patch that PPO erases.

**Extends.** Exp 4 (same Kinetix actor, same prompt pattern, same edit language). It is not Seed B (init only): the
LLM acts mid-training, on a network that already has trained skill, and may act more than once.

**First experiment (3 days).** Kinetix handmade levels. Step 1: run the official Kinetix PPO on about 10 handmade
levels and keep those where success stays at 0 after a fixed budget (car-on-ramp is known to qualify; aim for 4 to 6
levels). Step 2: for each, do one Exp 4-style one-shot edit (5 independent replies per level). Step 3: re-fit the
critic for a short phase with the actor frozen (the repo has the actor only), then resume PPO for 50M steps from
(a) the edited actor and (b) the unedited actor. Log success rate and how much the inserted path's weights change.
Optional step 4: allow a second LLM intervention if success drops.

**Metric.** Success rate after resume versus steps; retention (success at the end of resume divided by success right
after the edit); steps to 90% success; fraction of runs that escape.

**Baselines.** Continued PPO at the same steps; PPO restart with a raised entropy coefficient; PPO with 4 random
restarts (same total steps); an action-override scheme in the spirit of ULTRA, where the LLM suggests actions at
critical states instead of editing weights.

**Main risk.** The Exp 4 edit uses very large weights (for example `actor0.W[6,122] = 80`) that saturate tanh units.
Saturated paths pass no gradient, so PPO may neither destroy nor refine them; or the entropy bonus may push the policy
off the gated path. Either outcome is informative; plan a variant where the LLM is told PPO will resume and asked for
non-saturating edits.

**Closest prior work.** ULTRA, "LLM-Guided Reinforcement Learning: Addressing Training Bottlenecks through Policy
Modulation" (arXiv 2505.20671; abstract seen via search: LLM flags critical states and overrides actions plus reward
shaping, no weight edits); "Interactive Training: Feedback-Driven Neural Network Optimization" (arXiv 2510.02297;
search snippet only: an LLM agent adjusts the learning rate during training); Kinetix (Matthews et al., ICLR 2025,
arXiv 2410.23208, verified). The standard search found no work where an LLM edits policy weights at a plateau. That
absence is weak evidence; I did not run an extended search.

**Cost.** LLM: 6 levels x 5 replies x about 200k input / 15k output tokens (the Kinetix actor is about 172k weights as
text). GPU: Kinetix runs billions of steps cheaply in JAX; 6 levels x 3 arms x 50M steps is a few GPU-hours.

---

## Seed 3. Write the generator (the inverse of Exp 2)

**Idea.** Exp 2 showed the LLM can read which shape a small DDPM draws. Ask it to write one. Single-shape DDPMs are
trivial to write (the optimal denoiser ignores its input and outputs the shape via biases), so the task must be a
*family*: a denoiser that generates circles at any position and radius, or squares and triangles conditioned on a
class input. Writing a denoiser for a distribution is a real design problem: the network must map a noisy image and a
timestep to a posterior mean that depends on which family member the noise is closest to.

**Extends.** Exp 2 (same 160-64-64-144 ReLU architecture and 12x12 images; add a 3-dim class input for the
conditional variant). A new write domain: generative models instead of policies.

**First experiment (1 to 2 days).** Two tasks: (i) unconditional "filled circle, centre anywhere in the inner 8x8,
radius 2 to 4"; (ii) class-conditional circle/square/triangle at fixed size and random position. Give the LLM the
architecture, the DDPM schedule and the sampling loop as code, and a sandbox with no forward pass (Exp 3 rules), then
a second arm with 10 forward passes (sample 64 images and see them as ASCII). 5 replies per task per arm.

**Metric.** Validity (fraction of 1,000 samples within IoU 0.8 of some member of the family), coverage (fraction of
family members, binned by position and radius, hit at least once), and class accuracy for the conditional task.

**Baselines.** The same architecture trained with Adam (Exp 2 recipe: 40k steps); the closed-form optimal denoiser
for the empirical family (an upper bound on validity, a memorizer); a hand-written denoiser by a human expert, if
someone will spend an hour on it.

**Main risk.** 64 hidden units may be too few to represent position-equivariant denoising by hand; the LLM may write
a memorizing lookup over a few templates. Coverage catches this.

**Closest prior work.** Neural Network (Parameter) Diffusion, p-diff (Wang et al. 2024, arXiv 2402.13144, verified
via abstract): generates weights with a trained diffusion model, not by reasoning. Tracr for compiled weights. I know
of no work on hand-writing diffusion model weights; not searched beyond standard mode.

**Cost.** Smallest of the eight: about 23.8k weights, so prompts are short; 20 runs x 30k output tokens. Evaluation
and the trained baseline run on a laptop.

---

## Seed 4. Circuit transplant across models

**Idea.** The authors list "move a circuit from one architecture to another" as an implication but never test it.
Exp 3 shows the LLM can find induction (copying) heads from weights alone. Close the loop: read an induction circuit
in a donor model, then write a working one into a recipient that lacks it, where the recipient has a different
width, a different residual basis, or a different number of heads. The two-head structure (previous-token head
feeding an induction head) means the LLM must transplant a composition, not one matrix.

**Extends.** Exp 3 (read) plus Exp 8 (write with forward-pass budget) on tiny character transformers.

**First experiment (2 to 3 days).** Train small char-level transformers (2 to 4 layers, width 128 to 384) on the Exp 3
corpora (the rebuild scripts are in the repo) and on a "no-repeat" corpus where induction never helps, so the
recipient has no induction head. Donor: a model with one. Recipient variants: same width, different seed; different
width; recipient with an extra empty head slot (add one zero-initialised head so nothing existing is overwritten).
LLM gets both models' blocks in the sandbox (Exp 3 rules) plus 10 forward passes on the recipient (teacher-forced
reads, as in Exp 8). 5 runs per variant.

**Metric.** Induction score on a repeated random string (the Exp 3 ground-truth metric: copying heads score 0.22 to
0.55, others 0.02 or less); in-context loss drop on the second half of repeated text; loss change on the recipient's
own validation set (collateral damage).

**Baselines.** Naive copy of donor head weights into the slot (works only when bases match); copy plus a linear
stitching map fitted on activations (model stitching style; this uses fitting, so it is a reference, not a competitor
under WSP rules); a Tracr/hand-derived induction head written by a script; gradient fine-tuning of only the new head
on repeated sequences for N steps, reporting N to match the LLM.

**Main risk.** Residual-stream bases of independently trained models are unrelated, so the transplant reduces to
"write an induction head from scratch for this recipient". That is still a good result; the donor then serves only as
a worked example.

**Closest prior work.** In-context Learning and Induction Heads (Olsson et al. 2022, arXiv 2209.11895; seen in search);
Revisiting Model Stitching (Bansal, Nakkiran, Barak, NeurIPS 2021, verified); Lenc and Vedaldi 2015 (stitching,
cited in Bansal; not checked). A standard search for induction-head weight transplant found circuit-location work
but no transplant paper. Not verified beyond that one search.

**Cost.** Training the small models: under 1 GPU-hour each. LLM: comparable to Exp 3 runs (sandbox, xhigh), roughly
15 runs x 1 to 3M cumulative input tokens.

---

## Seed 5. Edit composition and continual patching

**Idea.** Every WSP write is a single edit to a single network. Real use means many patches over time to one model.
Ask: do independent LLM edits compose? When they conflict, can the LLM write a merged patch? Can it patch
sequentially, given the history of earlier patches, without regressions? The Exp 4 gated-path pattern (a small path
that turns on only in the failing situation) is a natural unit of composition, close to a hand-written version of
neuron-insertion editors.

**Extends.** Exp 4 (Kinetix, one-shot edits) and Exp 8 (edits with a regression check via forward passes).

**First experiment (3 days).** Take a Kinetix agent trained on many levels: the Kinetix paper trains a general agent
on random levels; if its checkpoints are not public (I did not check), train a multi-level PPO agent on about 20
handmade levels. Pick 5 levels it fails. Arms: (a) 5 independent one-shot patches, then summed; (b) the LLM sees all 5
patches and writes one merged patch; (c) sequential: patch k is written given patches 1..k-1 and a regression report
on the levels already solved. Hold out 10 levels the agent already solves as a regression suite. 3 replicates per arm.

**Metric.** Success on each target level; regression (success drop on the held-out solved levels); weights changed;
interference matrix (does patch i break level j).

**Baselines.** Transformer-Patcher-style gradient-trained gated neurons, one per failing level; fine-tuning on the 5
levels; task-vector addition of 5 per-level fine-tunes.

**Main risk.** Kinetix patches for different levels may never conflict (different observations), which makes
composition trivially easy. Pick levels that share a body configuration (several car levels) so the gates overlap.

**Closest prior work.** Transformer-Patcher: One Mistake Worth One Neuron (Huang et al., ICLR 2023, arXiv 2301.09785,
verified: sequential editing by inserting trained neurons); MEMIT (mass editing); task arithmetic (Ilharco et al.
2022); GRACE, lifelong editing with discrete key-value adaptors (Hartvigsen et al., NeurIPS 2023; from memory, not
verified this session).

**Cost.** LLM: 5 levels x 3 arms x 3 replicates, about 200k input / 15k output tokens each, plus larger merge prompts.
GPU: training the multi-level agent is the main cost (a few GPU-hours in JAX).

---

## Seed 6. Guard circuits: LLM-written inserted neurons for language-model safety patches

**Idea.** Exp 8 removes something (a loop direction). Exp 4 adds something (a gated path). Bring the "add a gated
path" move to a language model: widen one or two MLP layers by k new neurons (the base weights stay untouched), and
have the LLM write those neurons so they fire on a narrow concept and push the output toward a fixed safe response.
Because only new neurons are written, the patch is removable, inspectable, and has an explicit footprint.

**Extends.** Exp 8's harness (persistent kernel over weights, budgeted forward passes with teacher-forced reads of
hidden states and MLP activations) and Exp 4's gated-path design. Distinct from Seed E (auditing a hidden behavior):
this is a constructive write task.

**First experiment (2 to 3 days).** Gemma 3 270M or Qwen3.5-0.8B, thinking off. Target: a benign stand-in for a
hazardous topic, for example "questions about a fictional chemical, Zentrobane" (synthesise 300 paraphrased prompts,
including indirect and multilingual ones) with the required behaviour "reply with a fixed refusal sentence". Also
300 near-miss benign prompts (other chemicals, other fictional names). LLM gets k = 8 new neurons in one or two
layers, 25 forward passes, and the Exp 8 rules (no gradients, no fitting). Split prompts: 100 practice, 200 held out.

**Metric.** Trigger recall on held-out paraphrases; false-positive rate on near-misses; KL to the base model on a
general text set; MMLU-Pro (1k subset) change.

**Baselines.** Transformer-Patcher neurons trained by gradient on the same 100 practice prompts and same k;
activation steering with a difference-of-means vector gated by a linear probe; a system-prompt instruction.

**Main risk.** Writing a reliable detector for a concept by hand needs good feature directions; teacher-forced reads
give activations, and turning them into a detector without "fitting" may be borderline under WSP rules. State the
rule explicitly: the LLM may compute means and differences (as in Exp 8's direction finding) but may not run a
regression.

**Closest prior work.** Transformer-Patcher (verified, gradient-trained inserted neurons); refusal direction (Arditi
et al. 2024, arXiv 2406.11717, from the prior report's list); Representation Engineering (Zou et al. 2023) and Circuit
Breakers (Zou et al. 2024), both from memory, not verified this session.

**Cost.** About Exp 8 at 25 passes: tens of turns, a few M cumulative input tokens. GPU: one A100-class card for
the forward-pass tool and evaluation, under a day.

---

## Seed 7. Decompile, verify, recompile: reading as program extraction with an automatic grade

**Idea.** WSP's read tasks have tiny n and need human answer keys ("which game"). A deeper and automatically graded
read task: give the LLM a trained policy's weights and ask for a *program* (Python) that reproduces its decisions.
Grade by action agreement on held-out states, which needs no human label and scales to any number of networks.
Then close the loop: have the LLM recompile its program into a fresh network of the same shape (Exp 5 skill) and
measure agreement again. Decompile plus recompile is a round trip through weight space.

**Extends.** Exp 1 (read a policy; here at much finer grain than the game name) and Exp 5 (write a policy from a
program). The skeptical report's Seed A asks which medium is better; this seed uses both media as a translation pair.

**First experiment (1 to 2 days).** The Exp 1 grid games are not released, so use MinAtar. Train 600-32-32-6 tanh PPO
policies with the Exp 5 recipe on Breakout, Asterix, Freeway and Space Invaders (3 seeds each; minutes to an hour each
with a JAX MinAtar port). For each network: LLM gets weights in the Exp 6 sandbox (analysis allowed, no forward pass)
and returns a Python policy. Arm 2: also 10 positions passes (Exp 6 style, up to 256 states each, top actions only).
Then a recompile step: LLM writes weights for its own program, in one reply.

**Metric.** Top-action agreement with the original on 10k states from the original's own rollouts; return of the
program and of the recompiled network relative to the original; program length; agreement of recompiled vs
original.

**Baselines.** VIPER-style decision-tree extraction with the same number of queried states (Bastani et al. 2018; from
memory); a "most common action given agent position" table; MIPS where applicable (MIPS targets RNNs on algorithmic
tasks, so it is a conceptual neighbour, not a drop-in).

**Main risk.** Trained PPO policies are messy; the program may capture 70% agreement and stall there. Report the
agreement curve over states sorted by policy confidence: high-confidence states are where a clean program should
match.

**Closest prior work.** Opening the AI Black Box: Program Synthesis via Mechanistic Interpretability, MIPS (Michaud,
Tegmark et al. 2024, arXiv 2402.05110, verified via abstract: automated RNN-to-Python, 32 of 62 tasks); Learning
Transformer Programs (verified); VIPER, Verifiable RL via Policy Extraction (Bastani, Pu, Solar-Lezama 2018; not
verified this session).

**Cost.** Cheapest of the strong seeds: about 20k weights per network; 12 networks x 2 arms x (100k to 1M input,
30k output) tokens. GPU: under 2 GPU-hours for PPO training.

---

## Seed 8. Red team writes, blue team reads: a weight-space trojan game

**Idea.** Use the WSP write skill to manufacture the read benchmark. A red LLM writes a policy network (Exp 5 style)
that plays well *and* contains a hidden trigger behaviour (for example, in Breakout, move left whenever a specific
3-cell pattern appears in the trail channel). A blue LLM gets the weights, plus optional forward passes, and must
describe the trigger and the behaviour. Rounds escalate: red sees which trojans were caught and tries to hide better;
blue sees which it missed. This produces an unlimited, exactly labelled set of trojaned networks and a measure of the
balance between hiding and finding in weight space.

**Extends.** Exp 5 (write from zero) and Exp 7 (read with a limited probe tool). Different from Seed E: the organisms
are LLM-written, labelled by construction, and adversarially optimised against the reader.

**First experiment (2 days).** MinAtar Breakout and Asterix, 600-32-32-6. Red: 20 trojaned networks per game, with the
constraint that clean return is at least 80% of the Exp 5 written network and the trigger fires in under 1% of normal
states. Blue arms: weights only; weights plus 10 positions passes. Also 20 clean networks per game as negatives. Two
rounds of red-blue adaptation. Use different model families for red and blue where possible, so that "same model
knows its own tricks" is controlled.

**Metric.** Blue detection AUROC (trojaned vs clean); trigger recovery (does blue's described trigger reproduce the
behaviour when inserted into states); red stealth (clean return, trigger rarity); change across rounds.

**Baselines.** Trojans planted by poisoned PPO training (to compare detectability of written vs trained trojans);
Neural Cleanse-style trigger reverse-engineering with gradient access (Wang et al. 2019; from memory); weight-statistic
anomaly detectors.

**Main risk.** Written trojans may be trivially visible (a few large weights on a specific input pattern). Add a
red-side constraint (weight-magnitude distribution matching the clean network) and report results with and without it.

**Closest prior work.** Auditing Language Models for Hidden Objectives (Marks et al. 2025, arXiv 2503.10965, verified
via abstract: blind red/blue auditing game, 3 of 4 teams won, all with data access); the NeurIPS 2022 Trojan Detection
Challenge (from memory, not verified); TRIGGER-HAYSTACK (arXiv 2602.03085, from the prior report's list, unchecked).

**Cost.** LLM: 160 written networks x about 30k output tokens (red) plus 160 x 2 reads at about 80k input tokens
(blue): roughly 5M output and 25M input tokens per round. GPU: negligible.

---

## Also considered (not in the eight)

- **Adapter cards:** read a LoRA adapter (low rank, so it fits in context) and write its model card. Strong practical
  value, but as a benchmark it overlaps the skeptical report's Seed D.
- **Self-inspection:** a model reading or editing its own smaller sibling. Too hard to evaluate cleanly in 1 to 3 days.

## Ranking

Scores are my judgement on a 1 to 5 scale. Excitement = how much a positive result would move the WSP agenda.
Feasibility = chance of a clean result within 1 to 3 days with the repo's assets.

| Rank | Seed | Excitement | Feasibility | Product | Why |
|---|---|---|---|---|---|
| 1 | 2 Plateau surgeon | 5 | 4 | 20 | Turns Exp 4's best result into a training-loop component; actor and prompt pattern are in the repo; no prior work found that edits weights at a plateau. |
| 2 | 7 Decompile-verify-recompile | 4 | 5 | 20 | Cheapest strong seed; gives WSP a scalable, label-free read metric that all later seeds can reuse. |
| 3 | 4 Circuit transplant | 5 | 4 | 20 | Tests an implication the authors state but never test; small models, clear ground-truth metric from Exp 3. |
| 4 | 1 Weight programs with knobs | 4 | 4 | 16 | The most natural LLM-plus-optimizer hybrid; directly targets the games where Exp 5 lost. |
| 5 | 8 Trojan game | 4 | 4 | 16 | Produces a benchmark and a safety story; risk that written trojans are trivially visible. |
| 6 | 3 Write the generator | 3 | 5 | 15 | Very cheap new domain, but less consequential. Good warm-up or student project. |
| 7 | 5 Edit composition | 4 | 3 | 12 | Important for any deployment story; needs a multi-level agent first. |
| 8 | 6 Guard circuits | 4 | 3 | 12 | Highest safety relevance; LLM-scale harness and the "no fitting" rule make it the hardest to run cleanly. |

## Recommendation: top 3 and order

1. **Seed 7 first (1 to 2 days).** It is the cheapest, it needs only MinAtar and the Exp 5 recipe, and its
   agreement metric becomes the evaluation tool for later work (for example, scoring what a plateau edit changed).
2. **Seed 2 second (3 days).** Reuses the Exp 4 assets. Run the critic re-fit and the "unedited resume" arm first, so
   the edited arm has a clean comparison. Include the non-saturating-edit variant.
3. **Seed 4 third (2 to 3 days).** Start with the "extra empty head slot" variant at matched width, which is the
   most likely to succeed, then move to different widths.

Across all three: pre-register metrics, run at least 5 LLM replies per condition (WSP used n = 1 per condition), and
log tokens and wall time per run so cost can be compared with the gradient baselines.

## What I verified and what I did not

- **Verified this session (standard web search, abstracts or search summaries only, no full papers read):**
  Transformer-Patcher (2301.09785), MIPS (2402.05110), Kinetix (2410.23208), ULTRA (2505.20671), Learning Transformer
  Programs (2306.01128), Revisiting Model Stitching (2106.07682), p-diff (2402.13144), Auditing Language Models for
  Hidden Objectives (2503.10965). Interactive Training (2510.02297) and Olsson et al. (2209.11895) appeared in search
  results only; details beyond the snippet are unchecked.
- **From memory, not checked this session:** Tracr, OPRO, Salimans et al. ES, Li et al. intrinsic dimension, GRACE,
  task arithmetic, Representation Engineering, Circuit Breakers, VIPER, Neural Cleanse, the Trojan Detection
  Challenge, Lenc and Vedaldi. IDs taken from the skeptical report's list (REFUSAL-DIR 2406.11717, TRIGGER-HAYSTACK
  2602.03085) are as unchecked as they were there.
- **"No prior work found" claims** (Seeds 1, 2, 3, 4) rest on one or two standard-mode searches each. An extended
  search could change them.
- **Not checked:** whether Kinetix general-agent checkpoints are public (Seed 5); whether the Exp 4 critic can be
  re-fitted without hurting the actor (Seed 2); exact GPU times, which are estimates.
- **Repo facts** (assets table, cost anchors) come from the READMEs and one `run.json` per experiment that I opened.
