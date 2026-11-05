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
| Exp 1 game policies (500-32-32-5, 4 games) | Actor and critic weights (`exp1_which_game/networks/*.npz`, checked) | The 10x10 grid games are the authors' own ("our own 10 x 10 grid versions", Exp 1 README); no env code is in the repo (no `.py` with an env class or `reset`, checked by grep). Use MinAtar (public) for anything that needs rollouts. |
| Exp 3 char-transformer blocks | Yes (`exp3_language/networks/*/blocks.npz`: only the 4 blocks' LN, attention and MLP tensors, checked) | Embeddings, final LayerNorm and output matrix are not included (Exp 3 README, line 98). Not runnable end to end. Train your own tiny models for write tasks. |
| Exp 4 Kinetix network, before and after the edit | Yes (`exp4_fix_car/networks/ppo_final_seed0.npz`, `edited_actor.npz`) | Checked: `ppo_final_seed0.npz` holds the full actor-critic, including the value head (`DenseAndActivation_5..9`, `Dense_1`). The embedding layer `Dense_0` (679 -> 128) is **shared** by actor and critic, and the edit changed it (`embed.W` 1,887 entries, `embed.b` 1 entry, per `eval/oneshot_result.json`). So PPO can resume with the original critic, but the critic then reads a changed embedding. |
| Exp 5 written MinAtar networks | Yes (`exp5_write_from_scratch/networks/<game>/hand.npz`, actor only) | PPO checkpoints are not included; the PPO recipe is in the README. The 6-channel observation mapping is specified in the prompts for Breakout, Seaquest and Asterix only. |
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
Tune knobs with CMA-ES (population 16, 16 episodes per candidate, at most 32 knobs) or with PPO on the knob vector
only. Budget: 50M environment steps (with episodes of roughly 1k steps, 16 x 16 episodes is about 0.25M steps per
generation, so about 200 generations; 5M steps would allow only about 20, too few for CMA-ES even in 32 dimensions). Then one LLM revision round: show the LLM the tuned knob values and per-episode death causes,
let it rewrite the sketch, re-tune. 5 independent LLM replies per game.

**Metric.** Mean return over 1,000 episodes (Exp 5 protocol) against environment steps; area under the curve.

**Baselines.** The untuned written network (Exp 5 numbers); PPO from random init at the same step budget and at
100M/500M steps (Exp 5 numbers); CMA-ES over a random subspace of the same size around the written network
(intrinsic-dimension style), which isolates the value of the LLM choosing the subspace; and the key comparison,
full-weight PPO fine-tuning from the same written network at the same 50M steps (the skeptical report's Seed B arm).
Knobs are only worth having if they match or beat that arm while staying readable. Random-init PPO at 50M steps
alone would be a weak baseline: Exp 5 shows PPO at 100M is already below the untuned written network on all three
games.

**Main risk.** The LLM may not expose the right knobs; failures may come from missing features, not wrong constants.
The revision round is there to test whether it can add structure when told what goes wrong.

**Closest prior work.** Tracr (Lindner et al. 2023, compiles programs to weights, no tuning); Learning Transformer
Programs (Friedman, Wettig, Chen, NeurIPS 2023; verified this session: trained transformers converted to discrete
programs, the reverse direction); Evolution Strategies as a scalable alternative to RL (Salimans et al. 2017; not
re-verified); Measuring the Intrinsic Dimension of Objective Landscapes (Li et al. 2018; not re-verified); OPRO, LLMs
as optimizers (Yang et al. 2023; not re-verified). In code space the pattern is likely already common: an LLM writes a
parameterized program and an outer loop tunes or evolves it (PROG-POLICY-ATARI, arXiv 2508.19506, from the prior
report's list; FunSearch and Eureka, from memory). The new part is doing this in weight space. **I ran no prior-art
search for this seed**, so there may be close weight-space work I have not seen.

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

**First experiment (3 days).** Start with car-on-ramp alone, the one level that is known to be stuck (0 successes
after 1B steps) and whose full actor-critic is in the repo. Step 1: 5 independent Exp 4-style one-shot edits (the
released edit is a sixth). Step 2: the critic is in `ppo_final_seed0.npz`, but its shared embedding layer changed
with the edit, so first re-fit the value head for a short phase with the actor frozen. Step 3: resume PPO for 50M
steps from (a) each edited network and (b) the unedited network. Step 4, only if day 1 to 2 go well: run PPO on
about 10 more handmade levels and add those still at 0 success after a fixed budget. A level stalled at a fixed
budget is not proven to be a local minimum, so report the car-on-ramp result separately from these. Optional: allow
a second LLM intervention if success drops.

**Metric.** The edit already gives 100% success on car-on-ramp, so "steps to 90% success" would be zero by
construction and cannot tell a lasting escape from a brittle patch. Use instead: (1) retention (success at the end
of resume over success right after the edit); (2) **absorption**: at the end of resume, ablate the inserted path
(reset the edited entries to their pre-edit values, keeping all other PPO updates) and measure success. High success
after ablation means PPO moved the trained network into the new basin. Low success means it still depends on the
patch. (3) Return on the level's unshaped reward (time to goal), which the edit does not optimise and PPO can improve.
For levels where the edit is only partly successful, steps to 90% success is a valid extra metric.

**Baselines.** Continued PPO at the same steps; PPO restart with a raised entropy coefficient; PPO with 4 random
restarts (same total steps); an action-override scheme in the spirit of ULTRA, where the LLM suggests actions at
critical states instead of editing weights.

**Main risk.** The Exp 4 edit uses very large weights (`eval/oneshot_result.json` reports a relative change of
about 3,700x on `actor0.b` and 100x on `embed.b`) that saturate tanh units. Saturated paths pass no gradient, so PPO
may neither destroy nor refine them; or the entropy bonus may push the policy off the gated path. Either outcome is
informative; plan a variant where the LLM is told PPO will resume and asked for non-saturating edits. A second risk
is generality: the first experiment has one level, so it shows a mechanism, not a rate.

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

**Main risk.** Capacity. The family is finite (about 100 position and radius combinations on a 12x12 grid), so a
template bank over all members is a legitimate solution here, not a failure: there is no training set to memorise.
The question is whether 64 hidden units suffice for the LLM to cover the family. Coverage measures this. If the
hand-written denoiser covers only a few templates, coverage will be low. Because the family is enumerable, the
metric cannot tell "generalising" from "enumerating"; do not claim generalisation from this experiment.

**Closest prior work.** Neural Network (Parameter) Diffusion, p-diff (Wang et al. 2024, arXiv 2402.13144, verified
via abstract): generates weights with a trained diffusion model, not by reasoning. Tracr for compiled weights.
Closed-form and analytic denoisers (for example, work on optimal empirical denoisers and on locally equivariant
analytic denoisers, such as Kamb and Ganguli 2024, from memory, not verified) are relevant to how a hand-written
denoiser could be structured. **I did not search specifically for hand-written diffusion weights**; my one search
was for p-diff.

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
corpora (the Python rebuild scripts are in the repo; I did not check how the romaji and English corpora are rebuilt).
Natural text always contains repeats, so a "no-repeat corpus" is not a reliable way to get a recipient without
induction. Make recipients in one of two ways instead: (i) a checkpoint saved before the induction-head phase change
(Olsson et al. report it early in training), confirmed by the Exp 3 induction score staying at or below 0.02 for every
head; or (ii) a fully trained model whose induction heads are zero-ablated (its other weights stay co-adapted).
Donor: a fully trained model with induction heads. Recipient variants: same width, different seed; different width;
recipient with an extra empty head slot (add one zero-initialised head so nothing existing is overwritten).
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
policies with the Exp 5 recipe on Breakout, Seaquest and Asterix (4 seeds each; minutes to an hour each with a JAX
MinAtar port). These are the three games whose 6-channel observation mapping is written out in the Exp 5 prompts.
Freeway and Space Invaders would need a new mapping. For each network: LLM gets weights in the Exp 6 sandbox
(analysis allowed, no forward pass) and returns a Python policy. Arm 2: also 10 positions passes (Exp 6 style, up to
256 states each, top actions only). Then a recompile step: LLM writes weights for its own program, in one reply.

**Rule that keeps the metric meaningful.** A 3-layer MLP is trivial to evaluate in numpy, so in the weights-only arm
the LLM could compute the network on synthetic states and fit a tree or table to the outputs. That would be
distillation by query, not reading, and the agreement score would not tell them apart. Forbid evaluating the network
(or any layer composition) on inputs in the sandbox, and audit every code run as Exp 3 did. Any run that breaks the
rule is scored as arm 2 or discarded. In arm 2, give the VIPER baseline the same number of queried states.

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
30k output) tokens. GPU: under 2 GPU-hours for PPO training. Human time: auditing sandbox code runs for forbidden
forward passes (scriptable in part, as a check for matmul chains that start at an input-shaped array, but plan for a
manual pass).

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
states. Red cannot run its network under Exp 5 rules, so the harness checks both constraints (1,000 clean episodes,
and trigger rate on states from those episodes) and rejects networks that fail. Report the rejection rate. Also 20
clean networks per game, written by the same red model without a trojan, as negatives. Clean networks must come from
the same writer, or blue could separate the two classes by writing style instead of by finding the trojan. Blue arms:
weights only; weights plus 10 positions passes. Two rounds of red-blue adaptation. Use different model families for red and blue where possible, so that "same model
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

**Cost.** Per round: 80 written networks (2 games x 20 trojaned + 20 clean), plus rejected attempts, at about 30k
output tokens each, so about 2.4M or more output tokens for red. For blue, 80 networks x 2 arms = 160 reads at about
80k input tokens, about 13M input tokens. GPU: negligible.

---

## Also considered (not in the eight)

- **Adapter cards:** read a LoRA adapter (low rank, so it fits in context) and write its model card. Strong practical
  value, but as a benchmark it overlaps the skeptical report's Seed D.
- **Self-inspection:** a model reading or editing its own smaller sibling. Too hard to evaluate cleanly in 1 to 3 days.

## Ranking

Scores are my judgement on a 1 to 5 scale. Excitement = how much a positive result would move the WSP agenda.
Feasibility = chance of a clean result within 1 to 3 days with the repo's assets.

The scores are coarse (1 to 5). Seeds 2, 7 and 4 tie at 20, and so do 1 and 8 (16) and 5 and 6 (12). The order
within a tie is a judgement call, not a result of the scores.

| Rank | Seed | Excitement | Feasibility | Product | Why |
|---|---|---|---|---|---|
| 1= | 2 Plateau surgeon | 5 | 4 | 20 | Turns Exp 4's best result into a training-loop component. The full actor-critic is in the repo, which makes the car-on-ramp first experiment feasible. One standard search found no weight edits at a plateau. Limit: the first experiment has one level. |
| 1= | 7 Decompile-verify-recompile | 4 | 5 | 20 | Cheapest strong seed; a scalable, label-free read metric. It needs the no-forward-pass audit, or the metric cannot separate reading from distillation. |
| 1= | 4 Circuit transplant | 5 | 4 | 20 | Tests an implication the authors state but never test; small models, clear ground-truth metric from Exp 3. Recipient construction fixed (pre-phase-change checkpoint or ablated heads). |
| 4= | 1 Weight programs with knobs | 4 | 4 | 16 | The most natural LLM-plus-optimizer hybrid; targets the games where Exp 5 lost. It is only informative against the full-weight fine-tune arm, and I did not search its prior art. |
| 4= | 8 Trojan game | 4 | 4 | 16 | Produces a benchmark and a safety story; risk that written trojans are trivially visible. The harness must enforce red's constraints. |
| 6 | 3 Write the generator | 3 | 5 | 15 | Very cheap new domain, but less consequential, and the finite family means it cannot show generalisation. Good warm-up or student project. |
| 7= | 5 Edit composition | 4 | 3 | 12 | Important for any deployment story; needs a multi-level agent first (checkpoint availability unchecked). |
| 7= | 6 Guard circuits | 4 | 3 | 12 | Highest safety relevance; LLM-scale harness and the "no fitting" rule make it the hardest to run cleanly. |

## Recommendation: top 3 and order

The three top seeds do not depend on each other: Seed 2 uses Kinetix, Seed 7 MinAtar, Seed 4 small char
transformers. The order below is therefore set by cost, risk and shared tooling, not by dependencies.

1. **Seed 7 first (1 to 2 days).** Cheapest and lowest risk. It builds the MinAtar PPO-training and 1,000-episode
   evaluation pipeline (Exp 5 recipe and channel mapping), which Seeds 1 and 8 and the skeptical report's Seed B
   reuse directly. Its results do not feed Seeds 2 or 4.
2. **Seed 2 second (3 days).** Reuses the Exp 4 assets. Run the value-head re-fit and the "unedited resume" arm
   first, so the edited arm has a clean comparison. Report absorption (the ablation test), not steps to success.
   Include the non-saturating-edit variant.
3. **Seed 4 third (2 to 3 days).** Start with the "extra empty head slot" variant at matched width, which is the
   most likely to succeed, then move to different widths. If GPU time is free while Seed 2 waits on Kinetix runs,
   training Seed 4's small models in parallel is fine; nothing blocks it.

Real dependencies outside the top 3: Seed 1's key baseline is Seed B (full-weight fine-tune from the written
network), so run them together. Seed 5 needs a multi-level Kinetix agent before anything else.

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
- **"No prior work found" claims.** Seed 2 and Seed 4 each rest on one standard-mode search, and both searches
  returned only neighbouring work. **No search was run for Seed 1 or Seed 3** (earlier wording implied there was;
  corrected). Seeds 5 to 8 make no novelty claim beyond the named prior work. An extended search could change any of
  this.
- **Checked against the repo (second pass):** tensor names in the Exp 1, Exp 3, Exp 4 and Exp 5 `.npz` files. The
  Exp 4 critic **is** included; the first version of this memo said "actor only", which was wrong. The edited
  embedding is shared with the critic. Exp 3 has no embeddings or output layer. No Exp 1 env code is in the repo.
  The Exp 5 channel mapping covers only Breakout, Seaquest and Asterix. The Exp 4 relative-change figures come from
  `eval/oneshot_result.json`.
- **Not checked:** whether Kinetix general-agent checkpoints are public (Seed 5); how the romaji and English Exp 3
  corpora are rebuilt; MinAtar episode lengths used in the Seed 1 budget arithmetic (about 1k steps assumed); exact
  GPU times, which are estimates.
- **Repo facts** (cost anchors) come from the READMEs and one `run.json` per experiment that I opened.
