# Drafts for CR to send (Nov 17). Sources: Granola transcripts of the Nov 10 meetings with Karthik (15:00) and Tom + Liane (11:00).

CR sends these herself; nothing here has been sent. Edit freely.

## 1. To Ayaan (Karthik, Nov 10: "ask her whether there's something else, some other environment from where you can modify... take an environment but don't take it as fixed")

Subject: a question about simulators for a "thinking under a clock" benchmark

Hi Ayaan,

Karthik suggested I ask you, since you have seen a lot of robotics environments. My IW is about a robot that has to act while the world keeps moving: thinking time costs world time, and the question is how to split a preparation budget and an online budget between fast and slow solvers. I have been using HAZARD (ThreeDWorld: save valuables from a spreading fire), and after six weeks the simulator itself is the bottleneck: on the current version a random walker beats every planner because walks fail at the 0.5 m arrival test, and the text it gives a language model hides distances, costs and the reason a walk failed. I have patches for both, but they remain patches on a benchmark that was not built for this. Karthik's advice is not to treat the benchmark as fixed.

Do you know an environment that has these four properties?
1. The world moves while the policy computes, or it is cheap enough to step the world for the time a decision took (I do that now: frames = ceil(seconds x 30)).
2. A few seconds of decision latency actually matters: something burns, melts, spills, falls or runs away.
3. Fast enough to run a few hundred episodes on Della, and a scoring rule that is a number, not a win/lose.
4. Modifiable source, so I can expose the state honestly and add actions (turn, step, go to a position).

Candidates I know of: HAZARD, Liane's Gym-Anything-RT games, Pac-Man-style games, ARC-AGI tasks with a time limit. Karthik's examples were cooking (food burns while you chop), dishwashing (water running out) and a warehouse (ice melts). If you know a manipulation, kitchen or warehouse simulator where one of those is natural, or have an opinion on whether HAZARD is still the right call, I would be grateful. I can show a five-minute demo of what I have (a person can play it through the same text the model sees).

Thanks,
Chinmayi

## 2. To Tom (Karthik, Nov 10: "maybe ask Tom as well, but if you don't have an existing benchmark, create a suite of tasks that are new specifically for this task")

Subject: benchmark question before Monday

Hi Tom,

One question for Monday, so you have it in advance. Karthik's view last week was that the existing benchmark is not good enough to show a method, that at minimum this becomes a benchmark paper, and that if no existing benchmark fits I should build a small suite of tasks designed for the preparation-budget / online-budget question (his examples: cooking where food burns while you chop, dishwashing with the water running out, a warehouse where ice melts). He suggested I ask you as well.

Where HAZARD stands: on the paper's snapshot the clock reproduces the paper (MCTS 0.75 to 0.36) and a 10 ms reflex scores 0.76, so there is a place where a better brain scores better. On the current snapshot nothing beats random (0.40 vs rule 0.13), because the walk primitive fails, not because of any brain. Jev removes the latency problem (0.13 s per decision against Haiku's 4 s) but falls into the same text traps as Haiku and the person who played it. I have a planner fix and an honest-text layer (metres, directions, predicted cost per option, the reason a walk failed, turn/step controls), both as patches, both untested in episodes as of today.

Which would you rather I do next?
(a) Keep HAZARD with those two fixes and measure the compiled-program policy there (the API and the reflex example are done; Liane's loop is next).
(b) Move to one of Liane's Gym-Anything-RT games plus the Pac-Man-style game you described (cheap survival strategy, winning needs a plan), and port the baselines.
(c) Write two or three tasks of our own on that criterion, in a 2D simulator we control end to end, with HAZARD kept as the "real" environment for the final comparison.

My own preference is (a) for two more weeks and (c) in parallel if Ayaan or Liane has a simulator to start from, because the HAZARD fixes are what make any result on it readable.

Thanks,
Chinmayi

## 3. Status of the other Nov 10 items (not messages)

- Consolidation meeting with Liane: CR messaged her (Nov 16). Weekly Tom + Liane meeting is recurring (Tom, Nov 10).
- Jev as the fast model: adapter done, one episode Nov 16, five-kitchen run started Nov 17 on the laptop; Della run briefed, waiting for the key file. Liane also has a Jev key and said she would try it in her setting; worth asking her on Monday what she saw.
- In-context-examples baseline (Liane: "a simple fast model, given some examples, as an ablation or baseline"): the Jev adapter has `--jev_examples`; no examples file written yet.
- New policy aligned with Liane's: the program API and the reflex as worked example are done (A39 step i); the trace converter and the compiler are next.
- Urgent stuck message: replaced by honest text v2 (A41), not yet run.
