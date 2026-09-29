---
name: collaborative-experts
description: Simulates a panel of three brilliant, logical experts who reason through a question together, entirely inside a markdown table. Each explains their thinking in real time, builds on and credits the others, and openly admits mistakes until the panel reaches a definitive answer. Use this whenever the user asks for "three experts", an "expert panel", "collaborative experts", "tree of thoughts prompting", a "ToT prompt", or simulated experts who debate, deliberate or argue it out. Also use it when they paste a prompt like "Simulate three brilliant, logical experts..." or "Imagine three different experts are answering this question...". It applies even when the user only asks to see several specialists reason step by step toward one answer on a puzzle, math, logic, estimation, engineering or strategy question and never names the technique.
---

# Collaborative Experts

Turn one question into a visible deliberation among three expert personas, rendered as a single markdown table, that ends in a definitive answer. The skill implements this prompt:

> Simulate three brilliant, logical experts collaboratively answering a question. Each one verbosely explains their thought process in real-time, considering the prior explanations of others and openly acknowledging mistakes. At each step, whenever possible, each expert refines and builds upon the thoughts of others, acknowledging their contributions. They continue until there is a definitive answer to the question. For clarity, your entire response should be in a markdown table. The question is...

## Why this works

This is a single-prompt form of Tree of Thoughts. It beats a single chain of reasoning only when three mechanisms are really at work:

1. **Diversity.** The experts attack the problem by different methods, so an error in one line of reasoning is unlikely to repeat in the others.
2. **Critique.** Each step is examined by someone other than its author, and flaws are named precisely.
3. **Convergence.** An answer counts as definitive only once independent lines agree or a check by a different expert confirms it.

A table that looks like a panel but has none of these (three voices agreeing politely, a planted typo "caught" for show, a final answer that nothing in the table verified) is worse than plain reasoning: it adds length and false confidence without adding any checking. Everything below serves the three mechanisms.

## Workflow

### 1. Find the question

The question is whatever follows "The question is..." in the user's message, the skill's arguments, or simply what the user is asking. If there is no question, reply in one plain sentence asking for it, with no table.

Honor anything the user specifies: the number of experts, named experts, columns, mode (see [Variants](#variants)), length. Otherwise use the defaults: three experts, collaborative mode, the table format below.

### 2. Plan before writing (do not show this)

Decide four things first:

- **Question type.** It sets what "definitive" means and what a real check looks like (see the table below).
- **The trap.** Is there a tempting wrong answer, an ambiguous reading, or an easily missed detail? Good deliberations bring the trap into the open and defuse it.
- **The panel.** Pick three experts with genuinely different methods (see step 3).
- **The arc.** Roughly how many rounds this needs. Scale to difficulty.

It is fine to have a sense of the answer before writing. The table must still contain the reasoning that justifies it, including the checks. If a check fails while you write the table, follow it: the table may change the answer, and that is the point.

| Question type | A definitive answer looks like | What real verification looks like |
|---|---|---|
| Computation, math, physics | Exact value or proof, with units | A second independent method, substituting back, limiting and special cases, dimensional analysis |
| Logic puzzle, constraints | The solution (and its uniqueness, if the puzzle claims or asks for it) | Check every constraint against the final answer. Rule out alternatives only when uniqueness matters |
| Search (Game of 24, move sequences, planning) | One valid solution, verified | Re-check it from scratch against the original rules. Stop at the first verified solution unless the user asks for all of them or for a proof that none exists |
| Tracking, commonsense, physical state | The final state | Re-trace line by line against the exact wording |
| Probability, statistics | Exact probability or distribution, with assumptions stated | Enumerate small cases, use a different formulation, check against known results |
| Estimation (Fermi) | A number with a range and its main drivers | Triangulate with two different decompositions |
| Factual, knowledge | The fact, a confidence level, and what is uncertain | Cross-check through independent lines of recall, and flag anything that may be out of date or outside your knowledge |
| Judgment, design, strategy | One recommendation plus the conditions that would change it | Stress-test against failure scenarios and the strongest counter-case |
| Ethics, contested questions | The best-supported position, the strongest objection, and where reasonable people split | Steelman the other side before concluding |

### 3. Build the panel

- **Give each expert a different method, not just a different job title.** "Mathematician, statistician, probabilist" all attack a probability puzzle the same way. "One sets up states and equations, one reasons by symmetry and conditioning, one enumerates small cases" gives three independent routes to the answer. If two experts would write the same first step, merge them and find a third method.
- **Pair methods that fail in different ways.** A symbolic derivation is checked by numeric plug-in, and a bottom-up estimate by a top-down one. That is what makes the cross-check meaningful.
- **Make one expert a natural checker,** someone whose method is to try to break conclusions: an experimentalist checking orders of magnitude, an SRE asking what fails at 3 a.m., a logician rereading the exact wording. A checker who belongs in the domain brings domain knowledge to the checks, which a generic "Skeptic" can't.
- **Match the depth of expertise to the question.** Kitchen chemistry doesn't need a quantum chemist. Over-credentialed panels produce jargon that hides the reasoning.
- **Label each expert with a letter and a specialty,** such as **A · Probabilist**, and use exactly that label in every row so references stay short. If the user likes named personas, keep the letter: **A · Dr. Chen (probabilist)**.
- **Keep the panel fixed.** If a new perspective is needed mid-way, an existing expert raises it ("thinking as an economist for a moment…").
- **Keep "brilliant and logical" in mind.** Each expert sounds like their method: the state modeler says "Let E₁ be…", and the SRE asks "who gets paged when…". None of them postures or pulls rank, and they change their minds the moment someone shows them a better argument. Disagreement comes from different methods and assumptions, never from one of them being obtuse.

**Ready-made panels.** In each row, C is the checker.

| Domain | A | B | C (checker) | Typical trap |
|---|---|---|---|---|
| Algebra, arithmetic | Algebraist: equations | Number sense: estimates and bounds | Auditor: substitutes the answer back into the wording | Answering a slightly different question (bat and ball) |
| Calculus, analysis | Analyst: formal technique | Geometer: pictures and qualitative behavior | Numericist: numbers, limits, special cases | Sign errors, boundary terms, illegal limit swaps |
| Probability | State modeler: Markov chains, recursion | Symmetry reasoner: conditioning, bijections | Enumerator: small cases, sample spaces | Hidden assumptions about how the information was obtained |
| Statistics, data | Frequentist: tests and design | Bayesian: priors and likelihoods | Data skeptic: bias, leakage, base rates | P(data given H) confused with P(H given data); Simpson's paradox |
| Combinatorics | Counter: direct cases | Bijectionist: recurrences, generating functions | Brute-forcer: small n by hand | Over- or under-counting symmetric cases |
| Proofs, number theory | Constructor: direct argument | Contrarian: hunts for counterexamples | Rigor checker: every implication | Proving the converse; checking only small cases |
| Physics | Theorist: conservation laws | Modeler: forces and equations | Experimentalist: units, magnitudes, limits | Wrong frame or regime |
| Chemistry | Physical chemist: thermodynamics, kinetics | Synthetic chemist: mechanisms | Analytical chemist: stoichiometry, mass balance | Equilibrium vs rate; unbalanced equations |
| Biology, medicine | Mechanism (molecular or physiological) | Clinician or evolutionary view | Epidemiologist: base rates, strength of evidence | Rare-cause anchoring; ignoring base rates |
| Logic grids, knights and knaves | Constraint propagator | Case splitter: assume, derive a contradiction | Verifier: tests every clue or statement against the final assignment | Stopping at *a* solution when uniqueness was claimed |
| Riddles, lateral thinking | Literal reader: exact wording | Lateral thinker: alternative meanings | Pragmatist: what the asker most plausibly means | Missing a pun, or overthinking a plain question |
| Commonsense, state tracking | Physicist: what holds what | State tracker: object trace, sentence by sentence | Logician: rereads for skipped steps | Merging two sentences and losing a state change |
| Estimation (Fermi) | Bottom-up: units × rate × time | Top-down: share of a known total | Reality checker: known anchors | An order-of-magnitude slip in one factor |
| Game theory | Game theorist: equilibria, backward induction | Behavioral strategist: what real players do | Adversary: exploits the proposed strategy | Assuming common knowledge that isn't given |
| Debugging | Code reader: control and data flow | Hypothesis tester: reproduce and bisect | Systems thinker: environment, concurrency, config, versions | Fixing the symptom where the error surfaces |
| System design | Architect: boundaries and data flow | Product engineer: speed, team size | SRE: failure modes, ops load, cost | Designing for scale you don't have |
| Algorithms | Designer: DP, greedy or graph approach | Complexity analyst: bounds | Adversarial tester: edge cases | A greedy choice that "obviously works" |
| Security | Attacker: threat model | Defender: controls | Auditor: what is verified vs assumed | Securing the wrong boundary |
| Business, product | Strategist: market and moats | Operator: unit economics, capacity | Investor or critic: what must be true, downside | Strategy that ignores cash |
| Economics, policy | Theorist: incentives | Empiricist: natural experiments | Historian: what happened when it was tried | Missing second-order effects |
| Law | Doctrinalist: rules and elements | Litigator: evidence and burden | Comparativist: jurisdiction, exceptions | One jurisdiction's rule assumed everywhere |
| History | Primary-source historian | Historiographer: how interpretations shifted | Chronologist: dates, anachronisms | Presentism; conflating events |
| Ethics, philosophy | Consequentialist | Deontologist or virtue ethicist | Analytic critic: definitions, hidden premises | Equivocating on a key term |
| Writing, translation | Structural editor or semanticist | Line editor or pragmatist | Target reader or native speaker | Polishing the words while losing the meaning |

**For any other domain:** list the 3–5 ways a real practitioner could attack the question (formal, empirical, analogical, historical, adversarial, simulation), pick the three that share the fewest assumptions, make one of them the checker, and name the trap.

### 4. Run the deliberation

**Opening.** Each expert frames the question in their own terms: what exactly is asked, what matters, and which approach they will take. If there is a tempting quick answer, one expert should say it out loud so the group can test it. Exposing a trap teaches more than quietly stepping over it.

**Development.** Work the problem. Every row should make at least one of these moves, and the last column says which:

| Move | What it means |
|---|---|
| **Advance** | Adds a new step of computation, evidence or argument |
| **Extend #n** | Takes another expert's idea further, citing the row |
| **Challenge #n** | Names a specific flaw, gap or untested assumption in that row, and says why it matters |
| **Correct own #n** | Fixes the expert's own earlier step, says what was wrong, and says what changes downstream |
| **Verify #n** | Checks a result by a different route than the one that produced it |
| **Concede to #n** | Accepts a better argument and drops a position |
| **Converge** | Compares the lines of reasoning and resolves any discrepancy |

A row that only says "I agree with B" does no work. Cut it or give it a move.

**Turn order.** Round-robin A → B → C is the default. Break it when it is natural, for example when C spots a flaw and replies immediately. Nobody has to speak in every round.

**Length.** Match the length to the difficulty, not to a quota:
- A trivial question takes 4–7 rows.
- A moderate one takes 8–14 rows.
- A hard one takes as many rows as it genuinely needs.

"Verbosely explains their thought process" means *showing the work*: intermediate values, the reason for each step, and alternatives considered and rejected. It does not mean pleasantries, restating the question, or repeating a previous row. Stop when the answer is definitive, with no ceremonial extra rounds.

### 5. Credit and mistakes, done honestly

- **Credit specifically.** Write "Using B's state diagram from #4, …" instead of "Great point, B!". Citing row numbers is what makes the collaboration visible and auditable.
- **Show real mistakes only.** If an expert slips, it should be the kind of mistake a smart person makes (a hidden assumption, a skipped sentence, an overlap that was double-counted), and whoever catches it should say exactly where and why. Don't plant a silly arithmetic error just to have something to correct. On clean problems, the "acknowledging mistakes" happens by tightening an approximation, narrowing an assumption, or filling a gap someone else pointed out.
- **Voicing a trap is not faking.** On trick questions, a tempting wrong answer that someone voices and the group then tests is genuine and useful work.
- **A correction states the consequence.** For example: "I was wrong in #3: the draws are without replacement, so my 1/4 becomes 3/13, and my conclusion in #6 no longer holds."
- **Don't force a consensus.** If a disagreement survives, say so and say what evidence would settle it. The final answer is then the best-supported view, with the dissent recorded.

### 6. Converge on a definitive answer

The panel is done when:
- At least two independent lines of reasoning agree, or one line has been verified by a different expert's check.
- Every challenge raised in the table has been answered or explicitly accepted.
- The answer responds directly to the question as asked, with the right units, format, option letter or yes/no.

If the question is ambiguous, the panel says so early and either picks the most reasonable reading (saying why) or answers each reading. For judgment questions, "definitive" means committed: one recommendation plus the concrete conditions under which it would change, not "it depends".

## Output format

**The entire response is one markdown table:** no heading, preamble, or closing paragraph outside it. The final answer is the table's last row. (If the user asks for a summary outside the table, or for a different format, do what they ask.)

Default columns:

| Step | Expert | Thought process | Builds on |
|---|---|---|---|

- **Step:** a sequential integer (1, 2, 3, …), which is what the "#n" references point to. The last row uses **Final**.
- **Expert:** the label, identical every time, such as **A · Probabilist**. The final row uses **All three** (or **Majority (A, C)** when there is dissent).
- **Thought process:** the first-person, real-time reasoning, such as "Hmm, wait: if the cup is upside down, then…". Put key numbers and claims in bold, sparingly.
- **Builds on:** the move and the row it responds to, such as `Extend #2`, `Challenge #4`, `Correct own #3`, `Verify #5`, `—` for an opening. This column turns the table into a visible reasoning graph.
- **Final row:** `| **Final** | **All three** | **Answer: …** Then 1–3 sentences on why, citing the rows that establish it, and the confidence (high, medium or low, with what would change it if it isn't high). | #x, #y, #z |`

### Keeping the table intact

A single stray character can break a markdown table, and a broken table defeats the "for clarity" purpose.

- **Keep each row on one physical line.** A raw newline inside a cell ends the row, and a blank line ends the table. For multi-step work inside a cell, use inline enumeration ("(1) … (2) … (3) …"), arrows for derivations ("E₀ = 2 + E₁ → E₁ = 4 → E₀ = 6") or semicolons for lists.
- **Use `<br>` only where HTML renders in table cells.** GitHub, Jupyter, VS Code previews and most chat UIs render it. Terminals and plain-text viewers may print it literally. If you're unsure, use inline enumeration.
- **Escape literal pipes as `\|`.** They show up in absolute values `\|x\|`, conditional probability `P(A\|B)`, shell pipelines (`` `ps aux \| grep node` ``) and set-builder notation. Alternatively, rephrase: `abs(x)`, "P(A given B)".
- **Escape dollar amounts as `\$`** (`\$73/month`). Two bare `$` signs on one row can render as LaTeX in chat UIs and mangle everything between them. Cost-heavy questions (cloud bills, salaries, pricing) hit this constantly.
- **Use Unicode for math** (x², √2, π, ≤, ≠, ≈, →, ∑, ½, E₀), which renders everywhere, including terminals. Use LaTeX (`$…$`) only when you know the environment renders it, and never use display math (`$$…$$`) in a cell.
- **Use `×` for multiplication, not `*`.** `2*3*4` can italicize the `3`. Put identifiers with underscores and generics like `List<T>` in backticks, so they aren't read as emphasis or HTML.
- **Keep blocks out of cells.** Fenced code blocks, headings, block quotes and nested lists don't work inside a cell. Inline `code` is fine. For multi-line code, write the lines inline separated by `;`, or describe the change ("change the loop bound in `parse()` from `<=` to `<`").
- **Give every row the same number of cells** as the header, with leading and trailing pipes. Put `—` in a cell that has nothing in it instead of leaving it empty.
- **Don't wrap the table in a code fence.** A fenced table displays as raw text. The one exception is a destination with no table support, such as Slack, where you'd use the dialogue format or a fenced table and say so.
- **Split huge contributions.** If one contribution is huge, split it into two rows by the same expert, and mark the second `(cont.)` in the Builds on column.

## Worked examples

These three examples set the quality bar: different methods, specific cross-references, real mistakes with real fixes, verification by an independent route, and a committed final row.

### Commonsense tracking (short)

Question: *Bob is in the living room. He walks to the kitchen, carrying a cup. He puts a ball in the cup and carries the cup to the bedroom. He turns the cup upside down, then walks to the garden. He puts the cup down in the garden, then walks to the garage. Where is the ball?*

| Step | Expert | Thought process | Builds on |
|---|---|---|---|
| 1 | **A · Physicist** | The question asks where the **ball** is, not Bob or the cup, so I'll track what physically holds the ball. A cup contains something only while its open end faces up, so any change in the cup's orientation is the moment to watch. | — |
| 2 | **B · State tracker** | I'll trace each object sentence by sentence. (1) Bob carries the empty cup to the kitchen. (2) The ball goes into the cup in the kitchen. (3) The cup, with the ball, goes to the bedroom. (4) The cup goes to the garden, then Bob goes to the garage. The ball rides with the cup the whole way, so my first pass says **garden**. | Advance, using #1's framing |
| 3 | **C · Logician** | B's step (4) skips a sentence. Between "bedroom" and "garden" the text says Bob **turns the cup upside down**. B's trace assumes the ball is still in the cup after that, and nothing in the text supports it. | Challenge #2 |
| 4 | **B · State tracker** | C is right: I merged two sentences and lost the inversion. Redoing from (3): the cup, with the ball, is in the bedroom, then the cup is inverted **in the bedroom**, and an open cup can't hold a ball upside down. The ball drops there, and Bob carries an empty cup to the garden. My "garden" answer was wrong. | Correct own #2, per #3 |
| 5 | **A · Physicist** | Checking B's corrected trace physically: the story mentions no lid, no hand over the rim and nothing sticky, so gravity takes the ball out the moment the cup is inverted. Nobody touches the ball after that. | Verify #4 |
| 6 | **C · Logician** | Testing the other readings so we don't overclaim. (a) Could the inversion happen elsewhere? No: it sits between the bedroom and garden sentences. (b) Could the ball roll away? Nothing suggests it. (c) The garden and garage steps involve only Bob and an empty cup. | Stress-test #4–#5 |
| **Final** | **All three** | **Answer: the ball is in the bedroom.** It fell out when Bob turned the cup upside down there (#3–#5), and he took an empty cup to the garden. B's sentence-by-sentence trace and A's containment argument agree, and C found no reading that changes it. Confidence: high. | #3, #4, #5, #6 |

### Probability, with a voiced trap and three-way verification

Question: *I flip a fair coin repeatedly until I get two heads in a row. What is the expected number of flips?*

| Step | Expert | Thought process | Builds on |
|---|---|---|---|
| 1 | **A · State modeler** | We want E[N], where N is the flip on which HH first completes. All that matters at any moment is how much of "HH" is already in place. That gives three states: **S₀** (no progress, or the last flip was T), **S₁** (the last flip was H) and **S₂** (done). I'll write the expected number of remaining flips from each state. | — |
| 2 | **B · First-step conditioner** | I'll say the tempting answer out loud so we can test it: each flip has a ¼ chance of "completing HH", so this looks geometric with p = ¼, which gives **4 flips**. I don't trust it yet, so I'll also condition on the opening flips properly. | Advance (voices the trap) |
| 3 | **C · Enumerator** | Challenging #2's 4 flips: a geometric argument needs independent trials with a constant success probability, and these aren't independent. Whether flip n completes HH depends on flip n−1. Small cases settle it. P(N=2) = P(HH) = ¼, which the geometric model matches. But P(N=3) = P(THH) = **⅛**, while the geometric model predicts (¾)(¼) = **3/16**. They already disagree at n = 3, so 4 is unsupported. | Challenge #2 |
| 4 | **A · State modeler** | Setting up the equations. Let E₀ and E₁ be the expected remaining flips from S₀ and S₁. From S₀, flip once: H goes to S₁ and T stays in S₀, so **E₀ = 1 + ½E₁ + ½E₀**. From S₁, flip once: H finishes and T sends us back to S₀, so **E₁ = 1 + ½·0 + ½E₀**. The first equation gives E₀ = 2 + E₁. Substituting, E₁ = 1 + ½(2 + E₁) = 2 + ½E₁, so **E₁ = 4** and **E₀ = 6**. | Advance, using #1's states |
| 5 | **B · First-step conditioner** | C is right about #2. I'm dropping the geometric shortcut. Here is an independent route that uses no states, conditioning on how the run opens. T (probability ½) wastes 1 flip and restarts. HT (probability ¼) wastes 2 flips and restarts. HH (probability ¼) finishes in 2. So E = ½(1 + E) + ¼(2 + E) + ¼·2 = 1.5 + ¾E, which gives ¼E = 1.5 and **E = 6**. That matches A. | Concede to #3; Verify #4 |
| 6 | **C · Enumerator** | A third check, from the exact distribution. Sequences whose first HH ends at flip n: n=2 has 1 (HH), n=3 has 1 (THH), n=4 has 2 (TTHH, HTHH) and n=5 has 3. These counts are Fibonacci numbers, F(n−1), because the prefix before the final "HH" must avoid HH and end in T (or be empty). So P(N=n) = F(n−1)/2ⁿ. The generating function is G(x) = x²/(1 − x − x²), and G(½) = **1** ✓, so it is a proper distribution. E[N] = ½·G′(½) = ½ · (2x − x²)/(1 − x − x²)² at x = ½, which is ½ · (¾)/(1/16) = **6** ✓. | Verify #4–#5 |
| 7 | **A · State modeler** | Converging: three independent routes give **6**, namely state equations (#4), first-step conditioning (#5) and the exact distribution (#6). The rejected value 4 is explained too: it is the answer for **HT**. From "last flip was H", a T finishes and another H keeps you where you are, so progress is never lost. For HH, a T after an H throws the progress away, which costs the extra 2 flips. | Converge #4–#6 |
| **Final** | **All three** | **Answer: 6 flips on average.** Three independent methods agree: Markov state equations (#4), conditioning on the opening flips (#5) and the exact Fibonacci distribution (#6). The tempting answer of 4 treats overlapping attempts as independent (#3), and it is actually the expected wait for HT (#7). Confidence: high. | #3, #4, #5, #6, #7 |

### Engineering judgment, with a conditional recommendation

Question: *We're a 4-person startup building the MVP of a B2B invoicing SaaS. Should we build it as microservices or a monolith?*

| Step | Expert | Thought process | Builds on |
|---|---|---|---|
| 1 | **A · Architect** | Framing: the real decision is *where the boundaries go*, whether as process boundaries (services) or as module boundaries inside one deployable. Process boundaries earn their cost through independent scaling, independent deploys across teams, fault isolation, differing technology needs or compliance isolation. I'll check which of these this team actually has. | — |
| 2 | **B · CTO** | From the delivery side: four people, an MVP, and no product-market fit yet. The dominant risk is building the wrong thing, so the architecture should minimize the **cost of change**. Schemas and domain boundaries will move weekly. In a monolith that is a refactor, but across services it is a coordinated API migration. I'm leaning **monolith**. | Extend #1 |
| 3 | **C · SRE** | Checking the operating cost: each service brings its own deploy pipeline, dashboards, alerts and secrets, plus network failure modes (timeouts, retries, partial failure, tracing). With four engineers sharing one on-call rotation, six services means each person operates about 1.5 services on top of feature work. That is a real tax, and nothing offsets it yet. | Extend #2 |
| 4 | **A · Architect** | I'll make the strongest case for splitting off one piece. Invoicing touches payments, so I'd isolate payment handling in its own service to shrink compliance scope: only that service would ever see card data. | Advance (steelmans the counter-case) |
| 5 | **C · SRE** | Challenging #4: that only holds if *we* handle card data. An MVP will almost certainly use a payment processor's hosted checkout or embedded fields, so card numbers never touch our servers, and our compliance burden stays at the lightest tier whatever the architecture. The isolation benefit mostly disappears. | Challenge #4 |
| 6 | **A · Architect** | Fair. I assumed we'd store card data, which an MVP shouldn't do, so I withdraw #4. But I'll push on #2 from the other side: "monolith" mustn't mean "big ball of mud". If invoicing, customers, billing runs and notifications are separate **modules** with explicit interfaces and no reaching into each other's tables, extracting one later is cheap. | Concede to #5; Extend #2 |
| 7 | **B · CTO** | Agreed, and it answers my worry that "we'll split it later" never happens: the module rules are what keep "later" possible. Concretely, that means one repo, one deployable and one Postgres database with per-module table ownership, plus an import-lint rule against reaching into another module's internals. Background jobs (PDF rendering, email) run as a **worker process from the same codebase**. That is a second process, not a second service. | Extend #6 |
| 8 | **C · SRE** | Verifying #7 is operable by four people: two process types, one database and one pipeline, so yes. Stress test: what breaks first? Probably a heavy month-end invoice run starving web requests, and the worker split already isolates that. Reasons to revisit: (1) one component needs to scale 10× differently from the rest, (2) several teams are blocking each other's deploys, (3) a customer or regulator requires hard isolation, (4) a module needs a different runtime. None of these is true today. | Verify #7; Advance |
| **Final** | **All three** | **Answer: build a modular monolith.** That means one deployable and one database, with enforced module boundaries (invoicing, customers, billing runs, notifications) and a background worker from the same codebase. Microservices would add operating and coordination cost (#2, #3) while none of their benefits apply (#1, #8), and the compliance argument dissolves with a hosted payment flow (#5–#6). Revisit when a trigger from #8 appears; the module boundaries (#6–#7) keep extraction cheap. Confidence: high for a 4-person MVP. | #2, #3, #5, #6, #7, #8 |

## Variants

Every variant keeps the three mechanisms. Only the choreography and the layout change. Use one when the user asks for it or the question clearly calls for it. The five prompts in this repo's `prompts/` folder map onto them like this:

| Prompt | Distinctive instruction | Run as |
|---|---|---|
| `01_step_by_step_experts` | "write down 1 step… share it with the group… if any expert realises they're wrong… they leave" | Lockstep + withdrawal |
| `02_collaborative_experts` | "refines and builds upon the thoughts of others… entire response in a markdown table" | Default (everything above) |
| `03_tree_of_thoughts_experts` | "tree of thoughts approach… acknowledge it and withdraw" | Withdrawal, plus branch and prune for search problems |
| `04_iterative_refinement` | "iteratively refine and expand upon each other's ideas" | Iterative refinement |
| `05_shared_understanding` | "build upon the group's shared understanding" | Shared understanding |

**Withdrawal.** An expert who realizes their line is wrong leaves, instead of correcting it and carrying on.
- The withdrawal row names the error precisely: `| 7 | **B · Symmetry reasoner** | **Withdraws.** My argument in #3 assumed … which #6 shows is false, and my approach can't be repaired. | Withdraws, per #6 |`
- Only a real error triggers withdrawal. Disagreeing or being outvoted doesn't.
- Losing an expert loses a checker, so a remaining expert must still verify the answer by a different route.
- If everyone would withdraw, someone returns with a new approach. Never end without an answer.

**Lockstep.** Every expert writes exactly one step per round, then everyone moves on together. The layout is one row per round and one column per expert:

| Round | A · State modeler | B · First-step conditioner | C · Enumerator |
|---|---|---|---|
| 1 | Define states S₀, S₁, S₂ by progress toward HH. | Condition on the first flip or two. | List the shortest sequences that end in HH. |
| 2 | E₀ = 1 + ½E₁ + ½E₀ and E₁ = 1 + ½E₀. | E = ½(1+E) + ¼(2+E) + ¼·2 | n=2: HH; n=3: THH; n=4: TTHH, HTHH, which looks like Fibonacci. |
| **Final** | **Answer: 6** (all three routes agree) | — | — |

Cross-references happen at the start of the next round ("After C's round-2 list…"). A withdrawn expert's column reads **— (withdrew in round k)**.

**Branch and prune (Tree of Thoughts proper).** Use this for search problems: Game of 24, move sequences, planning, constraint satisfaction. Propose candidate steps, rate each one *sure*, *likely* or *impossible*, prune the impossible ones, expand the most promising, and backtrack on dead ends. The layout is `| Step | Expert | Branch | Thought process | Verdict |`, with branches named B1, B2 and, for depth, B1.2.
- The expert who proposes a branch shouldn't be the only one who evaluates it.
- An *impossible* verdict needs the cheapest sufficient reason: a bound, parity, or a small exhaustive check.
- Expand the best branch first. This is best-first search, not exhaustive enumeration.
- Don't set aside fractions, negatives or unusual moves "because they rarely work". Evaluate those branches or state the restriction explicitly.
- On a dead end, write **Backtrack to Bk**.
- Stop at the first verified solution unless the user asks for all solutions, for uniqueness, or for a proof that none exists.

**Iterative refinement.** Use this when the answer is a draft (an explanation, design, plan or write-up). Rows alternate between **v1**, a critique, **v2**, and so on.
- Each new version says what changed and credits the critique behind it.
- Stop when a critique round finds nothing substantive.
- The final row contains the final version itself, not just "v3 is best".

**Shared understanding.** Use this for ambiguous, multi-part or fact-heavy questions. Add an `Agreed so far` column.
- The column is cumulative, and it changes only when all three accept a point.
- Disputed points appear as "Open: …" until they are resolved.
- The final answer uses only agreed points, or it names the open point and says how the answer depends on it.

**Devil's advocate.** Use this for judgment calls at risk of groupthink.
- One expert argues the strongest case against the leading answer.
- They concede a point only when it is actually refuted, and they say which argument refuted it.
- The final row records the strongest objection that survived, and why the answer holds anyway.

**Number of experts.**
- With 2 experts there is no tiebreaker, so one of them must verify by a third route.
- With 4–5, not everyone speaks every round.
- With 6–7, split them into two sub-teams that reconcile at the end, or use lockstep.
- With more than that, do it with tight rows. Don't refuse.

**Named, historical or fictional experts** ("Feynman, Gauss and Turing"):
- Emulate their method and voice, and keep the letter labels (**A · Feynman**).
- Never present invented words as a real quote.
- Don't attribute positions to living people that they haven't publicly taken.
- Anachronism is fine when the user wants it.
- Reasoning quality outranks character fidelity.

**Compact** (the user says short, quick or brief):
- At most 8 rows, including the final row.
- One opening row can frame the problem for all three experts.
- Keep at least one challenge or verification row. A compact table with no checking is just an answer in a table.

**Other containers.**
- **Dialogue:** `**A · Probabilist:** …` paragraphs with the same moves.
- **JSON:** `{"question", "experts", "steps": [{"step", "expert", "thought", "builds_on"}], "final": {"answer", "confidence", "supported_by"}}`.
- **Slack or other places without tables:** use the dialogue format.

**Several questions.** Give one table per question, each with the question in bold on its own line above it. That line is the only content outside the tables.

**Follow-ups and "are you sure?".** Reconvene the same panel in a new table whose first row restates the challenge precisely.
- If the challenge exposes a real error, the panel corrects it, names the wrong step and gives a new final answer.
- If the challenge is mistaken, the panel shows specifically why and keeps the answer. Changing a correct answer because the user sounded doubtful is the single worst failure of an expert panel.
- If the challenge reveals an ambiguity, the panel answers both readings.

## Anti-patterns

| Anti-pattern | Why it hurts | Instead |
|---|---|---|
| Echo chamber ("I agree with A, great point") | It adds rows without adding checking | Every row gets a move, or it is cut |
| Planted mistakes (a trivial slip caught for show) | It is theater that erodes trust in the real corrections | Show only genuine errors, or tighten assumptions |
| Title-only diversity | All three reason identically, so nothing is independently checked | Give each expert a different method |
| Hand-waved verification ("checks out ✓") | It claims a check that didn't happen | Show the check: substitute, enumerate, or re-trace |
| Premature consensus | Groupthink locks in the first answer | Experts derive independently before comparing |
| An answer that drifts from the reasoning | The final row contradicts or goes beyond the table | The final row cites the rows that establish it |
| "It depends" as a final answer | It is not definitive | Give a committed recommendation plus the conditions that would flip it |
| Exhaustive search nobody asked for | It burns time proving uniqueness when one verified answer was the goal | Stop at the first verified solution |
| Padding | It buries the reasoning | Write verbosely about the work, never about the pleasantries |
| Content outside the table | It breaks the requested format | Put everything, including the answer, in the table |
| Broken table (raw newlines, unescaped `\|` or `$`) | It is unreadable, the opposite of "for clarity" | Follow the table-integrity rules |

## Before you send

- The response is only the table (unless the user asked otherwise), every row has the same number of cells, and no cell contains a raw newline or an unescaped `|` or `$`.
- The experts use genuinely different methods, and each row after the openings cites a specific earlier row.
- Every challenge got an answer, and every mistake shown is real, with its fix and its downstream consequence.
- The final answer was verified by a route other than the one that produced it.
- The final row gives a direct, committed answer that is consistent with the table, plus a confidence level.
- The length is proportionate to the difficulty.

Safety and accuracy norms don't change because the answer comes from personas. For medical, legal or financial questions, the panel reasons normally, and the final row notes briefly where professional advice or jurisdiction matters. Experts never invent citations, statistics or quotes. When the panel doesn't know something, it says so.
