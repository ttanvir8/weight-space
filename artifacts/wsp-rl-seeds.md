# Weight-space programming for reinforcement learning: eight new seeds

Research-advisor memo, 2026-11-05. Companion to `wsp-extension-seeds.md` (eight general extension seeds) and
`wsp-seed-report.html` (skeptical report, Seeds A to F). This memo is only about RL: ways an LLM that reads and writes
weights could make RL agents learn better, more safely, or more cheaply.

**Not repeated here.** From the skeptical report: A (weights vs code), B (LLM-written init, then PPO), C, D, E, F. From
the extension memo: 1 (weight programs with knobs), 2 (plateau surgeon: actor edits in a stalled run), 3 (write a
generator), 4 (circuit transplant), 5 (edit composition on one Kinetix agent), 6 (guard circuits), 7 (decompile,
verify, recompile), 8 (trojan game). Where one of these is a useful baseline arm, it is named as an arm, not a seed.

**The common move in these seeds.** WSP's RL experiments (4, 5, 6) only write or edit the **policy network**, and
the written network is the final product. These seeds move the LLM's write to other parts of the RL system: the
critic, the reward model, a training regularizer, the recurrent state, the world model, the network's capacity
during training, the opponent pool. The authors' own list of implications (theory guide §3.6: audit, cheaper
optimization and initialization, steering and unlearning, self-improvement and merging) names merging (R4's topic)
but none of the other RL components.

## Sources read

`README.md`; `artifacts/wsp-extension-seeds.md` (in full); `artifacts/wsp-seed-report.html` and
`artifacts/weight-space-programming.html` (tags stripped; the theory guide only searched for the implications list and
related terms); `lm_understand-logs/README.md`; the Exp 4, Exp 5 and Exp 6 READMEs; the Exp 4 prompt (reward and
observation sections), `eval/base_result.json`, `eval/oneshot_result.json`, `eval/ppo_training_result.json`; the
Exp 5 prompts' input-mapping sections and `eval/minatar_eval.json`; the Exp 6 prompt's Go rules and its eval scripts'
imports. I did not open every run.

## Repo facts checked for this memo

| Fact | Status | Where it matters |
|---|---|---|
| Exp 4 `ppo_final_seed0.npz` holds the full actor-critic: shared `Dense_0` embed (679 -> 128, ReLU), actor `DenseAndActivation_0..4` + `Dense_0` head (16 logits), **value trunk `DenseAndActivation_5..9` (5 x 128 x 128) + `Dense_1` (128 -> 1)** | Checked (tensor names and shapes) | Seed R1 |
| Exp 4 reward = +1 success / -1 red, **plus 0.2 x (d_prev - d)**, d = distance green shape to blue box (prompt line 30). The fix needs a run-up *away* from the goal first (README: "back up, then drive left"), so the shaping term penalises the needed first move | Checked (prompt text) | Seed R1 |
| Base policy: 0/256 successes; the shown episode stalls at body x = 2.00 from step 32 on; probability of entry 0 = 2 (the reverse value) is 0.00 at every logged step | Checked (`base_result.json`) | Seed R1 kill test |
| Kinetix PPO settings (lr 5e-5, ent 0.01, gamma 0.995, GAE 0.9, 2,048 envs, 1.0B steps) and Kinetix commit `80ee9c2` | Checked (`ppo_training_result.json`) | Seed R1 |
| Exp 5 `hand.npz` (actor only, 600-32-32-6 tanh) for Breakout, Seaquest, Asterix; scores in `minatar_eval.json` (hand: 517.4 / 53.1 / 53.2; PPO-500M best seed: 25.3 / 88.9 / 55.2; per-seed PPO scores also there, 10 seeds at 100M for Breakout and Asterix, 3 for Seaquest) | Checked | R2, R3, R4, R5, R6, R8 |
| Hand networks' footprint: Breakout uses 6 of 32 layer-1 units and 8 of 32 layer-2 units; Seaquest 32 / 26; Asterix 32 / 32. Max \|w\| is 1,000 (Breakout, Asterix) and 244 (Seaquest), so many units saturate | Checked (computed from the `.npz`) | R2, R4 |
| 6-channel MinAtar mapping is written out only for Breakout, Seaquest, Asterix. It is **shared**: a network for one game takes the other games' observations without change. Breakout's mapping is lossless (paddle, brick, ball, trail; two channels empty). Seaquest's merges sources (channel 1 = fish + subs + enemy bullets) | Checked (prompt text) | R3, R4, R6, R8 |
| PPO checkpoints for MinAtar: **not in repo** (README). The PPO recipe is (CleanRL style, lr 5e-4, 64 x 128 rollout, etc.) | Checked | All MinAtar seeds must train their own |
| MinAtar, JAX, PyTorch, Kinetix: **not installed** in this container (only numpy) | Checked (`pip list`) | All; install is part of day 1 |
| Exp 6: Go rules fully stated in `message.txt` (simple ko, whole-board repetition loses, area scoring, komi 7.5, 162-move cap); the three same-size PPO networks (512 x 6) and the four written networks are in `networks/`; positions passes in `runs/*/work/pass_*` keep inputs, legal masks and logits | Checked | Seed R7 |
| Exp 6: **no Go engine, no PPO training script, no practice or hidden opponents** in repo (eval scripts import `play` and `ppo_ref` from an unreleased `scripts/gohand`) | Checked (imports, `find`) | Seed R7 must rebuild |
| Exp 1 networks include critic weights, but the Exp 1 grid games are not released | Checked by the earlier memo; tensor names re-checked | Not used |

---

## Seed R1. Critic surgery: unstick an agent by editing what it believes, not what it does

**Idea.** In Exp 4 the LLM rewrote the actor so the car backs up and then jumps. But PPO stalled for a reason that
lives outside the actor: the shaped reward 0.2 x (d_prev - d) pays the car for moving toward the box and charges it
for the run-up it needs. My hypothesis (not yet checked; the read sub-task below tests it) is that the critic has
learned that the ramp stall is as good as it gets. Instead of editing
behaviour, show the LLM the **value network** (shared embed + value trunk) and the reward, ask it to explain what the
critic believes, and let it edit **only the value-trunk tensors** so that states with a run-up (body further right,
leftward speed) carry high value. The actor is left exactly as trained, so the edited network still succeeds 0% of
the time. Any success afterwards comes from PPO itself, now fed advantages from a corrected critic. Writing a value
function into weights by reasoning is a write task WSP never tried, and it hands the behaviour change back to the
optimizer.

**Why a natural extension, and why the authors likely did not consider it.** It uses Exp 4's network, prompt
format, edit language and harness unchanged; only the shown tensors change. The Exp 4 prompt explicitly says the
value head "is not shown". The authors frame WSP as reading and writing the behaviour-producing network; the critic is
a training-time object they dropped from view. The deceptive shaping term is visible in their own prompt but not
discussed in the README.

**First experiment (3 days).**
- Day 0 kill test (an hour): sample 1,024 base-policy episodes and log (a) the per-step probability of entry 0 = 2
  (reverse) and (b) the largest rightward displacement of the car body. `base_result.json` shows reverse at 0.00 in
  the one logged episode, and in that episode the car does not roll back (body x stays at 2.00 to 2.01). If run-up states are never visited, a critic edit alone cannot
  change any advantage that PPO ever computes. The experiment is therefore a 2 x 2 from the start:
  {critic edited, critic untouched} x {generic exploration bump, none}. The bump must not be targeted at reverse:
  raise the entropy coefficient from 0.01 to 0.05 for the first 20M steps, the same for all 16 logits.
- Day 1: 5 independent one-shot LLM replies (Exp 4 rules: no tools, one reply). Prompt = the Exp 4 prompt with the
  value trunk and `Dense_1` added (about 82.7k extra numbers; the actor can be omitted to stay near the Exp 4 input
  size of 355k tokens). Allowed edit targets: `DenseAndActivation_5..9`, `Dense_1`. The harness rejects any statement
  that touches the embed or the actor, so the policy is bit-identical after the edit.
  Also ask, before the edit, for the predicted value at 20 states described in words (positions and velocities from
  real rollouts plus 10 synthetic run-up states). The harness computes the true critic values; this is an
  automatically graded **read** sub-task.
- Day 2 to 3: resume official Kinetix PPO from each arm for 100M steps (the original run did 1B steps in 2.5 GPU-hours,
  so 100M is about 15 GPU-minutes per run). Two sub-arms for the edited critic: trainable from the start, and
  frozen for the first 20M steps (otherwise the value loss pulls the edited values back toward the stall policy's
  returns within a few updates). 3 PPO seeds per LLM reply.

**Metric.** Test success (1,024 episodes, Exp 4 protocol) at 25M, 50M, 100M resumed steps; fraction of runs reaching
50% success. Not trivial by construction: the start is 0% for every arm, and 1B steps of the original run never left
0%. Read sub-task: Spearman correlation between the LLM's 20 predicted values and the true critic outputs.

**Baselines (for a credible reading, not to disprove WSP).** (1) Resume with the untouched critic, with and without
the exploration bump. (2) Critic re-initialized at random (tests "any critic change helps"). (3) Shaping removed
(sparse +1/-1 only) on resume. (4) A **medium control**: the same LLM writes a Python potential function Phi(s), added
as potential-based shaping gamma Phi(s') - Phi(s) on resume (policy-invariant in the Ng et al. 1999 sense; from
memory). If the code potential works as well, the weight medium adds nothing here, which is still informative.
(5) Reference only: resume from the released actor edit (`edited_actor.npz`) with a re-fit value head, i.e.
the extension memo's Seed 2.

**Main risk.** Exploration, not belief, may be the bottleneck: if the 0.05 entropy arm alone solves the level, the
critic adds nothing measurable. Second risk: a large hand-written value bump makes advantages huge and destabilises
PPO. Tell the LLM that PPO will resume. Two things limit the damage: PPO normalises advantages per minibatch (if the
Kinetix implementation does; check), and the clipped ratio bounds each policy step. Third: one level, one stall. It shows a mechanism, not a rate.

**Closest prior work.** Eureka (Ma et al., ICLR 2024, arXiv 2310.12931; seen in search): LLM writes reward *code*,
not critic weights. LINVIT (arXiv 2402.16181; seen in search): LLM policy as a regularizer in value-based RL. Potential-
based shaping (Ng, Harada, Russell 1999; from memory). ULTRA (arXiv 2505.20671; from the earlier memo, abstract-level):
LLM overrides actions at critical states. **Searches run (standard mode):** "LLM edits value function critic weights
reinforcement learning deceptive reward shaping" and "large language model written value function or potential-based
reward shaping heuristic reinforcement learning Eureka". Neither returned work in which an LLM edits a critic's
weights or writes a potential-based shaping function; both returned only neighbours. That is two standard searches,
not a literature review.

**Cost.** LLM: 5 replies x about 350k to 500k input and 15k output tokens. GPU: 2 x 2 arms x 2 sub-arms x 5 replies
x 3 seeds x 100M Kinetix steps: about 60 short runs, roughly 15 GPU-hours. The baselines add about 5.

---

## Seed R2. The written network as a prior, not a start: KL-to-written-policy with LLM-written trust

**Idea.** The skeptical report's Seed B uses the written network as an initialization; the hand networks' weights
reach |w| = 1,000, which saturates tanh units, so PPO may be unable to move them. Use the written network only as a
**teacher in the loss**: train a fresh, randomly initialized student with PPO plus beta(s) x KL(pi_student ||
pi_written). The new part is beta(s): ask the LLM to write a second small network, a **trust gate** (600 -> 8 -> 1,
sigmoid), that says in which states its own policy should be trusted ("trust the dodge logic when an enemy is
adjacent; do not trust the surfacing logic, I was unsure about the oxygen timing"). The prior then constrains the
student only where the writer is confident, and the student's weights are never saturated.

**Why natural, why likely missed.** It uses the three Exp 5 networks unchanged, and Seaquest and Asterix are the games
where the hand network loses to 500M-step PPO (53.1 vs 88.9, 53.2 vs 55.2). WSP treats a written network as an
answer. Treating it as a fallible prior with a written statement of its own uncertainty is an RL use the post does not
discuss (its implications list says "write a good starting network", i.e. init).

**First experiment (2 days).** Seaquest and Asterix, MinAtar 1.0.15, Exp 5 PPO recipe, 50M steps, 5 seeds per arm.
Arms: (a) PPO from random init; (b) init from `hand.npz` (Seed B, reference); (c) random init + KL to hand with a
constant beta, annealed to 0 over 25M steps (3 beta values; kickstarting-style); (d) as (c) but beta(s) from the
LLM-written trust gate (one reply per game, Exp 5 rules, no tools; 5 replies per game); (e) as (d) with a **random
gate** of matched average trust (controls for "less KL on average"). The core arms (a) to (c) need no LLM tokens,
because the hand networks exist.

**Metric.** Mean return over 1,000 episodes (Exp 5 protocol) at 10M and 50M steps, and area under the curve. The
claim to test is "the student beats both its teacher and PPO at the same step count". Return above the teacher's own
score (53.1 / 53.2) at 50M steps is not guaranteed by construction: the KL term pulls toward the teacher, and PPO at
100M steps reached only 5.6 (Seaquest) and 25.8 (Asterix) best-seed.

**Baselines.** Arms (a), (b), (c), (e) above. Kickstarting with a PPO-trained teacher of equal score is the fairness
check: if any teacher of score 53 gives the same gain, the gain comes from distillation, not from the writing.

**Main risk.** Teachers with |w| = 1,000 give near-deterministic logits, so KL is dominated by a few states; use a
temperature on the teacher's logits (a fixed 3-value sweep). A second risk is that the gate is uninformative because
the LLM cannot calibrate its own confidence. The random-gate arm (e) measures exactly that.

**Closest prior work.** Kickstarting Deep RL (Schmitt et al. 2018, arXiv 1803.03835; seen in search). LINVIT (arXiv
2402.16181; seen in search: LLM policy as regulariser, most benefit when the LLM policy is near-optimal). LLM action
priors for PPO/DQN (arXiv 2410.07927; seen in search, details unchecked). KL-regularisation theory (Zhao et al.,
arXiv 2502.07460; seen in search). **Searches run:** "KL regularization toward LLM prior policy reinforcement learning
sample efficiency"; "Kickstarting deep reinforcement learning teacher policy distillation Schmitt 2018". Novelty is
**moderate**: regularising toward an LLM-derived prior exists; what is new is a prior that is itself a cheap
weight-space network (no LLM calls during training) plus an LLM-written, state-dependent trust gate. I found no prior
work on the gate, from two standard searches.

**Cost.** LLM: 2 games x 5 gate replies x about 10k input / 20k output tokens. GPU: 2 games x 5 arms (some with sweeps,
about 11 configurations) x 5 seeds x 50M steps. With a JAX MinAtar port, about 10 to 20 GPU-hours; with the official
Python package on CPU, much longer (see "Shared tooling").

---

## Seed R3. Plasticity doctor: LLM reads a network at a task switch and rewrites its idle units

**Idea.** Networks trained with RL lose plasticity: units go dormant or saturate, and a network trained on one task
learns the next one worse than a fresh network (the dormant-neuron and primacy-bias literature). Standard fixes are
blind: reset the units that are inactive (ReDo), or shrink and perturb all weights. An LLM that can read weights can
do two things a blind rule cannot: decide which units are truly unused (not just low-activity) and **write new,
task-relevant features** into the freed units instead of random weights. At a task switch, the LLM sees the trained
network, harness-computed unit statistics on states from the new task, and the new task's rules, and writes an edit:
which units to free, and what each freed unit should detect.

**Why natural, why likely missed.** WSP edits networks to fix a behaviour (Exp 4, Exp 8). Using the read-write skill
as a **network-health intervention during training** targets a known RL pathology the post never mentions. The shared
6-channel MinAtar interface in the Exp 5 prompts makes a clean continual-RL test possible with no new observation
code: a Breakout network accepts Asterix observations as is.

**First experiment (3 days).** 600-32-32-6 tanh actor-critic (Exp 5 recipe). Train on Breakout for 20M steps, switch to
Asterix for 20M steps; 5 seeds of the first phase. At the switch, the harness computes per unit: mean |activation|,
saturation fraction (|tanh| > 0.99, the tanh analogue of dormancy), and gradient norm on 4,096 Asterix states. The
LLM gets weights (actor and critic, about 41k numbers), these statistics, and Asterix's rules and source (Exp 5
style); one reply, no tools. A 2 x 2 factorial separates its two contributions: {units chosen by LLM, units chosen by
the ReDo rule} x {LLM-written incoming weights, ReDo random re-init} at a matched number of reset units.

**Metric.** Asterix return (1,000 episodes) at 5M and 20M post-switch steps and AUC; Breakout return after the Asterix
phase (forgetting, secondary). Decision rule: the factorial cell "LLM-chosen and LLM-written" must beat ReDo
(the "rule-chosen, random" cell) at the 5M point by more than the seed-to-seed standard error.

**Baselines.** No intervention; ReDo at the switch; Shrink and Perturb (Ash and Adams 2020; from memory); full reset
(fresh network); PPO from scratch on Asterix. The fresh network is the target that a plasticity fix must approach.

**Main risk.** A 32-unit network trained for 20M steps may show little plasticity loss, so every arm ties. Day-1 kill
test: if no-intervention is within one standard error of fresh-network training on Asterix at 5M steps, there is
nothing to fix; lengthen phase 1 or add a third game before spending LLM tokens. The harness-computed statistics are
allowed measurements (like Exp 6 passes); the LLM must not run the network itself, which the no-tools rule enforces.

**Closest prior work.** The Dormant Neuron Phenomenon in Deep RL, ReDo (Sokar et al., ICML 2023, arXiv 2302.12902;
seen in search); "Measure gradients, not activations!", ReGraMa (arXiv 2505.24061; seen in search); PRIME (arXiv
2607.17922; search snippet only: resets only units that are both activation-dormant and gradient-silent); primacy bias
(Nikishin et al. 2022; from memory); plasticity injection (Nikishin et al. 2023; from memory). **Searches run:**
"dormant neuron reset plasticity loss deep reinforcement learning ReDo"; "language model agent monitors reinforcement
learning training diagnoses dormant neurons plasticity intervenes". The second found no LLM-agent version, only
rule-based resets.

**Cost.** LLM: 5 replies x about 100k input / 20k output tokens. GPU: 5 seeds x (1 phase-1 run + about 8 phase-2 arms)
x 20M steps, about 10 GPU-hours in JAX.

---

## Seed R4. Merge two game policies into one network by reading both

**Idea.** Give the LLM two trained policies for different games (Breakout and Asterix, same 600-32-32-6 shape and
the same 6-channel interface) and ask for **one** network of the **same width (32)** that plays both. It must find
which units carry each skill, free capacity, write a game detector (Breakout lights the projectile channel 3,
Asterix the danger channel 1) and gate each skill on it. Weight averaging and task arithmetic have no notion of
"this unit is the ball tracker"; distillation needs data and gradients. A reader can do a functional merge.

**Why natural, why likely missed.** The authors list "merging" as an implication but never test it, and every WSP
RL write is single-task. This differs from the extension memo's Seed 5 (many patches on one Kinetix model, one task
family) and Seed 4 (moving one circuit between transformers): here two whole policies from different tasks must
share a fixed capacity.

**First experiment (2 days).** Step 1, warm-up with zero training: merge the two Exp 5 hand networks. Breakout's uses
only 6 + 8 units, Asterix's uses all 32 + 32, so a 32-wide merge must compress Asterix's design. That is not
trivial, but these are sparse and LLM-written, so this step is a sanity check only. Step 2, the real test: train
PPO policies (Exp 5 recipe, 50M steps, 3 seeds per game) and merge pairs. One reply per merge, no tools, both networks
as text (about 41k numbers, roughly 150k input tokens); 3 pairs x 3 replies. Forbid nothing extra: with no tools, the
LLM cannot run either network.

**Metric.** For each game, return of the merged network divided by the source network's return (1,000 episodes),
after subtracting the random-policy return (measure it; do not assume 0). Report the minimum over the two games: a
merge that plays one game well and the other randomly scores near 0.

**Baselines.** Weight averaging; task arithmetic and TIES (from memory); permutation alignment then averaging (Git
Re-Basin, Ainsworth et al. 2022; from memory); multi-task policy distillation into a 32-unit student from N states
of each teacher (Rusu et al. 2016; from memory), reporting N needed to match the LLM; a 64-unit concatenation with a
hand-coded game gate (an upper reference that cheats on width); a **from-rules write**: the LLM writes one
two-game network from both games' rules with no source networks shown (Exp 5 style). Without this arm, a good merge
could just be a fresh write that ignores the sources (this matters most in the hand-network warm-up, where the
sources are the LLM's own style).

**Main risk.** PPO policies use all units densely, so there may be no free capacity: the LLM must compress, and may
only manage "one game kept, the other approximated". The minimum-over-games metric will show this. A second risk:
Breakout's ball channel never fires in Asterix, so naive weight addition may already work partly; the averaging
baseline measures how much.

**Closest prior work.** Robot Fleet Learning via Policy Merging (arXiv 2310.01362; seen in search: weight-space merging
of policies on Meta-World); Reinforced Agent Merging, RAM (arXiv 2601.13572; seen in search: merging RL-trained
agentic LLMs); Policy Fusion (arXiv 2104.10610; seen in search); GENOME (arXiv 2503.01155, evolving LLM weights by
crossover; from the skeptical report's list, unchecked). **Search run:** "merging two reinforcement learning policy
networks different games single network weight space merging distillation". It found no LLM-by-reading merge.

**Cost.** LLM: 9 merges x 150k input / 20k output tokens plus the warm-up. GPU: 6 PPO runs x 50M steps (a few
GPU-hours); evaluation on CPU.

---

## Seed R5. Reward-model surgery: find and remove a spurious feature before the policy exploits it

**Idea.** When reward is learned from preferences, the policy optimises the reward model's bugs. Plant a known bug:
train small reward models (600-32-32-1) on preference data in which a spurious feature (for example "the player is in
the top two rows") is correlated with the preferred segment. PPO on such a reward model learns to camp in the
spurious state. Give the LLM the reward model's weights and the game rules, **not told whether or what was planted**,
and ask it to (a) state what the reward model rewards, and (b) write an edit that removes any dependence that the
rules do not justify. Then run PPO on the edited reward model and measure true return.

**Why natural, why likely missed.** WSP reads policies and language models (Exp 1, 7) and edits them (Exp 4, 8). A
reward model is a different and arguably easier read target: it is a scalar function of the state, and its intended
meaning is stated by the task. Every WSP RL experiment uses a programmed reward, so the reward-learning pipeline is
outside the post's frame. The safety story (stop reward hacking before training, not after) is direct.

**First experiment (3 days).** Asterix (lossless channels for player, enemies, treasure). Organisms: 3 planted
features (top-rows position; a specific column; presence of an enemy in the trail channel) x 2 seeds = 6 bugged
reward models, plus 3 clean reward models trained on decorrelated pairs. Preference labels come from true return over
50-step segments of a mixture of PPO checkpoints. Two harness checks per organism, before any LLM call: the bugged
model's accuracy on decorrelated held-out pairs must drop by at least 10 points relative to in-distribution pairs (the
bug is real); and PPO (20M steps) on the bugged model must have a true return clearly below PPO on the true reward
(the bug matters downstream). Discard organisms that fail and report how many. The LLM gets one reply, no tools
(about 20k weights as text, comparable to Exp 1's 74k-token reads); 3 replies per organism, all 9 organisms shuffled
and blind.

**Metric.** (1) Downstream: true return of PPO (20M steps, 3 seeds) on the edited reward model, normalised so PPO on
the unedited reward model is 0 and PPO on the true reward is 1. (2) Detection: per organism, did the LLM say "bugged"
and name the planted feature (graded against the key; with n = 9 report counts, not AUROC). (3) Collateral: edited
model's accuracy on in-distribution and decorrelated pairs; for the 3 clean models, the edit must not lower true-
return PPO. The LLM never sees the answer key, and the planted feature is not described. But it does see the game
rules, which state the true reward, so it could ignore the given model and write a reward function from the rules.
That would not be reading. Two guards: an edit footprint cap (at most 5% of entries changed, enforced by the harness),
and baseline (e) below, which measures what writing from the rules alone achieves.

**Baselines.** (a) PPO on the unedited reward model with a KL penalty to a reference policy (standard reward-hacking
mitigation); (b) reward-model ensemble with a pessimistic penalty; (c) a **medium control**: the same LLM shown
harness-computed saliency maps of the reward model (Michaud et al. style) instead of weights, writing the same kind of
edit, which isolates "reading weights" from "seeing attributions"; (d) oracle: reward model retrained on decorrelated
pairs (upper bound); (e) **rules-only write**: the LLM writes a 600-32-32-1 reward model from the game rules with no
reward model shown (Exp 5 style). If (e) matches the edits, the result shows writing, not reading.

**Main risk.** The planted bugs may be too visible (one input block with large first-layer weights), which makes the
detection trivial; use a spurious feature that is distributed (column position across many cells) as the harder
case, and report results per feature type. A second risk: a removal edit may also remove real signal (top rows may
genuinely be safer in Asterix); the clean organisms and the true-reward oracle measure that.

**Closest prior work.** Understanding Learned Reward Functions (Michaud, Gleave, Russell 2020, arXiv 2012.05862; seen
in search: saliency and counterfactuals find spurious features in learned rewards). HARVE (arXiv 2606.03131; seen in
search, snippet level: training-free projection of hacking directions out of an LLM reward model's head) and "One Bias
After Another" (arXiv 2603.03291; seen in search: removes spurious directions from reward-model representations).
These two are the closest: they edit reward-model weights without retraining, but via scripted directions found from
labelled hacking pairs, on LLM reward models. The delta here is an LLM reading a small reward model's weights, blind,
and the downstream RL outcome as the score. **Searches run:** "Understanding learned reward functions Michaud Gleave
Russell interpretability reward model bugs"; "editing learned reward model weights to remove spurious feature before
policy optimization reward hacking". The novelty claim is narrower than I first expected: reward-model weight editing
exists.

**Cost.** LLM: 9 organisms x 3 replies x about 70k input / 15k output tokens. GPU: training 9 reward models (minutes),
plus PPO: 9 organisms x (original, true reward, 3 edits, 2 to 3 baselines) x 3 seeds x 20M steps, roughly 20 GPU-hours
in JAX.

---

## Seed R6. Write the memory: hand-built recurrence for a partially observed game

**Idea.** In MinAtar Breakout, the "trail" channel (the ball's previous cell) is what shows the ball's direction:
without it, a single frame does not show which way the ball moves. Remove the trail channel from Breakout and give the policy one recurrent layer
(h_t = tanh(x_t W + h_{t-1} U + b)). Ask the LLM to write the **recurrent weights U** so that the hidden state carries
what the trail used to carry (for example: two units hold the ball's row and column as analog values, two more hold
the previous step's values, and a difference gives direction). Recurrent PPO is slow and finicky; a written memory
could be frozen and PPO trains only the readout, which turns a POMDP into a feed-forward learning problem.

**Why natural, why likely missed.** All WSP networks are feed-forward, and Exp 5's Breakout hand network reads the
trail channel (checked: it reads channels 0, 3, 4). Writing state that persists over time is a new kind of weight
program, and the comparison to a random fixed recurrence (an echo-state network) has a sharp question: does designed
recurrence beat random recurrence as a frozen memory for RL?

**First experiment (2 days).** Breakout, MinAtar 1.0.15, observation with channel 4 zeroed. Network 600 -> 32
(recurrent) -> 32 -> 6. Arms, 5 seeds each, 30M steps: (a) feed-forward PPO with the trail (reference, the Markov
upper level); (b) feed-forward PPO without trail (lower bound); (c) GRU PPO, same width, random init; (d) echo-state:
random frozen W and U (spectral radius swept over 3 values), PPO trains the 32 -> 32 -> 6 readout; (e) LLM-written
frozen W and U, PPO trains the readout; (f) LLM writes the whole recurrent policy, no training (Exp 5 rules, one
reply). 5 LLM replies; arms (e) and (f) use the same reply where the LLM writes both.

**Metric.** Return (1,000 episodes) at 30M steps and AUC. A memory probe, independent of return: a linear readout,
fitted by the harness (evaluation only, not by the LLM) on held-out rollouts, predicts the ball's true direction
from h_t; report accuracy for (c), (d), (e). Day-0 check: (a) must beat (b) by a clear margin, otherwise the trail is
not needed and the POMDP is not real. If PPO hardly learns Breakout in 30M steps (PPO-100M best seed reached only
15.8), the arms cannot separate; check the arm-(a) curve first and raise the budget or switch to a game where PPO
learns faster.

**Baselines.** (a) to (d) above; (c) is the standard approach and (d) is the precise control for "the LLM chose the
recurrence".

**Main risk.** Capacity: 32 recurrent units cannot copy a 100-cell ball grid, so the LLM must compress the ball
position into a few analog units, which tanh distorts. The probe will show whether the memory works even if the
readout fails to use it.

**Closest prior work.** Reinforcement Learning with Echo State Networks (Szita, Gyenes, Lőrincz, ICANN 2006; seen in
search); Memory-based control with recurrent neural networks (Heess et al. 2015, arXiv 1512.04455; seen in search);
influence-aware memory (Suau et al., arXiv 1911.07643; seen in search); Tracr (compiled transformers, no recurrence;
from memory). **Search run:** "hand-designed recurrent neural network weights memory partial observability compared
echo state network reinforcement learning". It returned no hand-designed versus random versus trained recurrence
comparison in RL, and nothing LLM-written.

**Cost.** LLM: 5 replies x about 15k input / 25k output tokens (Exp 5 size). GPU: 6 arms (with sweeps, about 8
configurations) x 5 seeds x 30M steps, about 10 GPU-hours in JAX; recurrent PPO is the slowest arm.

---

## Seed R7. Read the opponent: weight-space exploiters for league training (Exp 6 extension)

**Idea.** In adversarial games, a policy's weaknesses are written in its weights. Give the LLM a target Go policy's
weights and ask it to write an **exploiter** tailored to that target, then add exploiters to the target's opponent
pool and continue PPO. Trained adversarial policies need millions of games against the victim; a reader might find a
blind spot (for example a mis-valued pass or an unseen capture shape) from the weights. The RL payoff is a cheap
source of diverse hard opponents for league or PSRO-style training.

**Why natural, why likely missed.** Exp 6 writes a Go player against fixed, unread opponents. Reading the opponent is
the game-theoretic use of the read skill, and the post treats read and write tasks separately.

**First experiment (3 days, the tightest of the eight).** The repo lacks the engine, the training script and the
small opponents (checked). Day 1: implement the rules from `message.txt` in numpy or JAX; test legal-move masks against
the saved positions passes (`work/pass_*/inputs.npy`, `legal.npy`; exact for "moves" positions). Train 3 small PPO
targets (243-32-32-82, about 11k weights, readable as text) with a simple self-play-plus-random recipe. Day 2: arms,
3 runs each per target, in the Exp 6 sandbox: (A) target weights + 10 games passes against the target; (B) **no
target weights** + 10 games passes against the target (black-box control); (C) target weights, 0 passes. The LLM may
not evaluate the target network in its sandbox (forbid matmul chains starting from board-shaped inputs, audit every
run, as the extension memo's Seed 7 does); otherwise reading becomes simulated play. Day 3: resume PPO of each target
for a short phase with its exploiters added to the pool.

**Metric.** Exploiter win rate against its target (2,000 games); specificity = that win rate minus its win rate
against the other two targets; arm A minus arm B isolates the value of the weights. RL payoff: after resumed
training, the target's win rate against held-out exploiters (from the other arms and other seeds) versus a target
resumed with only self-play.

**Baselines.** Arm B; a trained adversarial policy (PPO against the frozen target) with the number of games it needs
to match the LLM's exploiter; a generic strong player (the Exp 6 budget-50 network, in the repo) against each target,
which sets how much "exploit" is just "plays better".

**Main risk.** Engineering eats the three days. A small 32-unit target may have no readable blind spot beyond general
weakness, so the specificity score is near 0; that is a real negative result if the generic-player baseline is in
place. The released same-size PPO networks (1.48M weights) are too large to read as text and should not be the first
targets.

**Closest prior work.** Adversarial Policies Beat Superhuman Go AIs (Wang et al., ICML 2023; seen in search via the
project page and ICML listing; the attacker could query, not read, KataGo's network); Approximate exploitability
(arXiv 2004.09677; seen in search); PSRO and AlphaStar league training (from memory). **Search run:** "LLM reads
opponent neural network weights to find exploit adversarial policy Go". No weight-reading exploiter work appeared.

**Cost.** LLM: 3 targets x 3 arms x 3 runs of an Exp 6-style session at 10 passes (Exp 6 budget-10: 28 turns), a few M
cumulative input tokens each, so tens of M input tokens in total, most of it cached. GPU: target training and game
evaluation, a few GPU-hours; the engine is the main human cost.

---

## Seed R8. Write the simulator: an LLM-compiled world model as a prior for model-based RL

**Idea.** The skeptical report notes that Exp 5 mixes two jobs: designing a strategy and compiling it into weights.
A world model isolates the compile part. Rules are fully given by the source code, and no strategy is needed. Ask the
LLM to write the weights of a Breakout dynamics network (observation + action -> next observation, reward, done).
The RL use: a **differentiable** simulator, written before any data, that (a) lets a policy be trained in imagination
with zero real steps, and (b) serves as an initialization that gradient training refines when the real environment
differs from the description.

**Why natural, why likely missed.** WSP writes policies (Exp 5, 6) and edits a policy and a language model (Exp 4, 8);
it never writes a dynamics model. A world model is the RL object
whose correct weights are most fully determined by the information WSP already puts in its prompts (the game source).

**First experiment (3 days).** Breakout, whose 6-channel mapping is lossless (checked). Network 606 -> 256 -> 256 ->
602 (600 next-cell logits + reward + done). Exp 6 sandbox rules: the LLM may write construction code, may not fit to
data, and gets 10 passes, each running the model on up to 256 transitions it specifies and returning predictions.
3 runs. Then two RL tests. (i) Clean: train PPO only inside the written model, evaluate in the real game. (ii) Shift:
the real game differs in one rule the LLM is not told (for example the ball does not bounce off the side wall but
wraps); fine-tune the written model on K real transitions (K = 1k, 10k, 100k) and compare with a randomly
initialised model trained on the same K.

**Metric.** Exact next-frame accuracy (all 600 bits right) on 10k held-out transitions from a mix of random and PPO
policies; reward and done accuracy; 20-step rollout accuracy. For (i), real-game return of the policy trained in the
written model, against a policy trained with the same steps in the real game. For (ii), model accuracy against K, and
the K a random-init model needs to match the written model at K = 1k.

**Baselines.** A same-architecture model trained on N real transitions, reporting N needed to match (equivalent data);
a **code world model** written by the same LLM (Python), which is exact on the clean task, so for (i) it is the upper
bound, and for (ii) it is refined WorldCoder-style by letting the LLM repair the code from the same K transitions.

**Main risk.** Writing a 600 -> 600 local-rule map through 256 hidden units is hard: brick removal, paddle bounce angles
and ball start sides all need gated cases. Partial accuracy may be the realistic outcome; the equivalent-data metric
still gives a meaningful number. The clean case (i) is a weak RL argument because the code model is exact; (ii) is
where weights can win, because they can be refined by gradient while code needs another LLM repair loop.

**Closest prior work.** Generating Code World Models with LLMs Guided by MCTS, GIF-MCTS (Dainese et al., NeurIPS 2024,
arXiv 2405.15383; seen in search); WorldCoder (Tang et al. 2024; seen in search snippets); PoE-World (NeurIPS 2025;
seen in search: LLM-synthesised programmatic experts). All write code, not weights. **Searches run:** "LLM generates
world model code model-based reinforcement learning GIF-MCTS WorldCoder"; "LLM writes neural network world model
weights directly without training transition model". The second returned only coverage of WSP itself.

**Cost.** LLM: 3 runs x Exp 6-style session at 10 passes, a few M cumulative input tokens each. GPU: the trained-model
baselines and PPO-in-model runs, a few GPU-hours.

---

## Shared tooling (a real dependency)

Six seeds (R2 to R6, R8) need a MinAtar PPO pipeline that reproduces the Exp 5 recipe and the 1,000-episode
evaluation. None is in the repo, and nothing is installed here. Before any MinAtar seed, check the pipeline against
`minatar_eval.json`: retrain Asterix at 100M steps with 3 seeds and confirm the scores fall inside the range of the
10 released seeds' scores (`asterix.ppo100_s0..s9`). A JAX port is needed for the GPU-hour estimates above; whether a
JAX MinAtar port matches MinAtar 1.0.15's dynamics exactly (including sticky actions and the 10,000-step cap) I did
not check. If not exact, train in the port and always evaluate in the official package. R1 needs Kinetix at the
recorded commit, which is independent of this.

## Also considered (not in the eight)

- **Observation re-encoding transfer** (rewrite the first layer for a new sensor layout). For linear changes such as
  channel permutations a script solves it exactly, so it does not test reading. Interesting only for non-linear
  re-encodings, which I could not build from repo assets in 3 days.
- **LLM as crossover operator in population-based training.** Close to EvoLLM and GENOME (skeptical report's list);
  R4 covers the functional-merge core.
- **Credit assignment by reading attributions**, **curriculum by editing**, **offline RL conservatism edits**: no
  first experiment with a non-trivial metric came to mind within the repo's assets.

## Ranking

Scores 1 to 5, my judgement. Excitement = how much a positive result would change how RL is done or how WSP is
viewed. Feasibility = chance of a clean result in 1 to 3 days with repo assets plus public packages.

| Rank | Seed | Excitement | Feasibility | Product | Reasoning |
|---|---|---|---|---|---|
| 1= | R1 Critic surgery | 5 | 4 | 20 | New write target with a crisp mechanism already present in the repo (deceptive shaping, stuck critic). Full actor-critic in repo; Kinetix runs are cheap. Not 5 on feasibility: the day-0 exploration check may show the critic cannot matter without exploration. |
| 1= | R5 Reward-model surgery | 5 | 4 | 20 | Safety and RL in one; downstream metric is hard to game. Needs organism construction and many PPO runs. Novelty narrower than hoped (HARVE and "One Bias After Another" edit reward-model weights by script). |
| 3 | R6 Write the memory | 4 | 4 | 16 | Clean design question with a sharp control (random frozen recurrence). Cheap in tokens. Risk: Breakout learns slowly under PPO (15.8 at 100M best seed). |
| 4 | R2 Written prior + trust gate | 3 | 5 | 15 | Most feasible: core arms need no new LLM call. Least novel: kickstarting and LLM priors exist; the gate is the new part. |
| 5= | R3 Plasticity doctor | 4 | 3 | 12 | Strong idea, but small 32-unit nets may show little plasticity loss; needs many seeds for small effects. |
| 5= | R4 Two-game merge | 3 | 4 | 12 | Feasible and clear metric; less RL-specific (it is model merging on policies). |
| 5= | R8 Write the simulator | 4 | 3 | 12 | Cleanest test of "compile" versus "design"; hard write; the RL win needs the shift case. |
| 8 | R7 Read the opponent | 5 | 2 | 10 | Highest ceiling in multi-agent RL, but the engine, training script and opponents must be rebuilt. |

**Ties.** R1 and R5 tie at 20; I put R1 first because its assets are all in the repo and its kill test costs an hour.
R3, R4 and R8 tie at 12; their order is arbitrary. R2 (15) and R6 (16) are close; a one-point change in either score
would swap them.

## Recommendation: top 3 and order

1. **R1 first.** The day-0 kill test (sample base-policy episodes, measure reverse probability and rightward
   displacement) takes an hour. The 2 x 2 runs either way, but the test says in advance whether the no-bump
   arms can work at all: if run-up states are never visited, only the bump arms can test the critic edit. It uses only Kinetix, so it does not wait on the MinAtar pipeline, and it can run
   while that pipeline is built.
2. **R5 second.** It needs the MinAtar PPO pipeline (see "Shared tooling"). That is a real dependency: none of its
   downstream numbers mean anything until the pipeline reproduces an Exp 5 PPO score range. R5 also builds the
   reward-model and organism tooling.
3. **R6 third.** It needs the same validated MinAtar pipeline plus a recurrent PPO arm. It does not depend on R5's
   results; R5 goes first only because it was ranked higher, so R5 and R6 can swap or run in parallel once the
   pipeline is validated.

Cheap add-on: R2's arms (a) to (c) cost no LLM tokens and reuse the validated pipeline. Run them as soon as the
pipeline passes, as a pipeline exercise with a real result.

Across all seeds: pre-register metrics and kill tests; at least 5 LLM replies per condition (WSP used n = 1); report
tokens and GPU-hours next to every result.

## What I searched, and what I did not verify

- **Searches run (all WebSearch, standard mode, abstracts and snippets only, no full papers):** 13 queries, listed in
  each seed's prior-work paragraph. Items marked "seen in search" appeared in these results; I did not read the
  papers. Snippet-only items (WorldCoder details, PRIME 2607.17922, HARVE 2606.03131, 2603.03291, 2410.07927) are as
  reliable as their snippets.
- **From memory, not checked this session:** Ng, Harada, Russell 1999; Shrink and Perturb; primacy bias; plasticity
  injection; TIES; Git Re-Basin; Rusu et al. policy distillation; PSRO; AlphaStar league; Tracr. IDs from the
  skeptical report's list (GENOME 2503.01155) are unchecked, as there.
- **Novelty claims.** For each seed I ran one or two standard searches and found no work in which an LLM reads or
  writes that RL component as weights. That is weak evidence. R2 and R5 have close neighbours that I name. No claim of
  "no prior work" is made beyond "not found in the listed searches".
- **Fixes after one self-review pass:** R1's "the critic believes the stall is optimal" is now marked as a
  hypothesis that the read sub-task tests; the "car never rolls back" claim is limited to the one logged episode;
  R4 and R5 gained "write from rules only" baselines, because without them a fresh write that ignores the given
  network would score as a successful read; the intro no longer says the authors' list omits merging (it names it).
- **Repo facts** are in the table at the top. **Not checked:** whether a JAX MinAtar port exactly matches 1.0.15;
  whether the Kinetix level `l/h2_car_ramp` loads unchanged at the recorded commit (the record names it); whether the
  car can roll back under "motor off" in other episodes (only the one logged episode was read); random-policy returns
  for MinAtar games; all GPU-hour figures, which are estimates.
