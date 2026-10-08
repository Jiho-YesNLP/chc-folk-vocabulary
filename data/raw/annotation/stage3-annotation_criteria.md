# What Counts as an Ability Descriptor of a CHC Ability

> **Phase 1 deliverable.** This document establishes inclusion and exclusion criteria for corpus annotation in Phases 2–3. It governs the decision annotators make for every retrieved passage that survives the pre-filter: *is this a genuine ability descriptor of the target CHC narrow ability?* Annotation uses a three-way scheme — **ability descriptor** (include), **incidental mention** (exclude), and **off-topic** (exclude). A fourth case, **technical label use**, is *not* an annotation category: passages containing a verbatim CHC label are removed deterministically by a lexicon pre-filter before annotation (see *Pre-filter* below). Consistency across annotators on these distinctions is the methodological foundation of the corpus.

---

## Category 1: Ability Descriptor (INCLUDE)

A passage qualifies as an ability descriptor of CHC narrow ability *aᵢ* if it satisfies **all three** of the following criteria simultaneously:

**1a. Ability-characterizing content.** The passage describes a cognitive process, capacity, tendency, or pattern of behavior that maps onto *aᵢ*'s construct definition — without requiring technical vocabulary to make the connection. The connection must be interpretable by a psychometrically naive reader who has only the seed sentences as a reference point.

> Example for Ideational Fluency (FI): *"She could sit down with any prompt and have fifteen different angles worked out before most people had finished reading the question."* — The phrase "fifteen different angles" describes *output volume* of idea generation, the core of FI.

**1b. Naturalistic register.** The passage uses everyday, observational, or narrative language — not psychometric, diagnostic, or academic prose. Typical markers: hedged first/third person observation ("she seems to", "I've noticed that"), concrete behavioral scenario, metaphor or analogy, evaluative framing in social context ("always the one who...").

**1c. Implicit ability reference.** The ability is described through its behavioral or experiential manifestations, *not* by naming it. The passage communicates what the ability *looks like*, *feels like*, or *produces* — not what it *is called*. This criterion is what distinguishes an ability descriptor from technical label use (handled by the *Pre-filter* below).

**Annotation instruction:** *If you could use this passage as a seed sentence that would retrieve similar descriptions of the same ability, include it.*

**Deficits count.** A CHC ability is a dimension, and a passage can describe it from either end. A description of difficulty, failure, or a low level of *aᵢ* is an ability descriptor when it satisfies 1a–1c like any other passage: it shows *what* goes wrong, *how*, or *under what conditions* in a way that is specific to *aᵢ*. The same process-versus-outcome line applies to deficits as to strengths (2c). A bare self-evaluation ("I'm terrible at this") or the name of a condition or diagnosis ("I have dyslexia") is not a behavior, so it stays incidental.

> Example for Resistance to Auditory Stimulus Distortion (UR): *"In a busy restaurant I can hear that everyone's talking, but the words just melt into the background chatter and I can't pick out what my friend is saying."* — Include. It names the condition (competing noise) and the specific failure (speech is not separated from noise), which is the UR construct seen from its low end.
>
> Counter-example: *"I've always been bad with numbers."* — Incidental (2c). It is an outcome judgment of low ability, with no process described.

---

## Category 2: Incidental Mention (EXCLUDE)

A passage is an incidental mention if the cognitive behavior described is:

**2a. Peripheral to the main proposition.** The passage is primarily about something else (a narrative event, an evaluation of a person's character, a description of a task) and the cognitive ability relevant to *aᵢ* is invoked only in passing — as a modifier, instrument, or incidental detail, rather than as the passage's focal referent.

> Example: *"She remembered the answer and solved the puzzle."* — Memory is used here as an instrument for puzzle-solving, not described as a capacity. The cognitive process is not under the discourse's attention; it is a transparent means to an end.

**2b. Too generic to discriminate.** The passage could apply equally to two or more CHC narrow abilities — including abilities at the broad-stratum level — without any linguistic signal that distinguishes which one is being described. This is distinct from a genuine neighboring-ability case (where the passage does describe *something*, just not the target): incidental mentions are ambiguous at the construct level, not merely the narrow-ability level.

> Example: *"He was really smart."* — Consistent with Gf, Gc, or multiple Gr abilities; the passage contributes nothing diagnostically.

**2c. Purely evaluative without process description.** The passage evaluates cognitive performance as an outcome (good/bad, fast/slow, impressive/unremarkable) without characterizing the underlying cognitive process that produced it. Evaluative framing is *part* of an ability-descriptor fingerprint when co-present with process description, but cannot stand alone.

> Example: *"Her answer was brilliant."* — Outcome judgment with no characterization of how or what kind of thinking produced it.

**Annotation instruction:** *If removing the phrase in question would leave the passage's main point intact, or if the passage gives you no information about what kind of cognitive ability is being described, code it as incidental.*

---

## Category 3: Off-Topic (EXCLUDE)

A passage is off-topic if it describes **no cognitive ability at all** relevant to the CHC inventory — it was retrieved as a similarity artifact (e.g., the shared person-narrative framing of the seeds, or another common-mode direction; see the query-debiasing note in `docs/notes.md`) rather than because it characterizes any cognitive process.

> Example: *"He drove across town to pick his brother up from the station."* — A person-narrative sentence with no cognitive-ability content; retrieved only because its surface framing resembles the seeds.

This is distinct from **incidental mention** (Category 2), which *does* invoke a relevant cognitive ability, only peripherally or too generically. Off-topic passages contain no cognitive-ability referent at all to be peripheral about. Keeping the two distinct is diagnostic: a high off-topic rate signals a retrieval/index problem (needs debiasing or a higher similarity threshold), whereas a high incidental rate signals a seed-relevance problem (needs seed revision).

**Annotation instruction:** *If the passage gives you no cognitive-ability content whatsoever — you cannot name even a broad-stratum ability it could be about — code it off-topic, not incidental.*

---

## Pre-filter: Technical Label Use (removed before annotation)

Technical label use is **not an annotation category**. It is handled by a deterministic lexicon pre-filter that runs *before* annotation: any passage containing a verbatim CHC label is removed from Track A (and may be retained for Track B, the register it belongs to).

**Strict lexicon definition.** The lexicon contains only the *exact label strings* from the CHC taxonomy (`data/raw/chc_taxonomy.json`): each narrow-ability name (e.g., "Associational Fluency," "Ideational Fluency," "General Sequential Reasoning") and each broad-factor name (e.g., "Retrieval Fluency," "Fluid Reasoning," "Visual Processing"), matched case-insensitively as whole phrases. **No** ability codes (e.g., "FA," "I," "N" — they collide with ordinary words and pronouns), **no** test-instrument names, **no** paraphrases, **no** lay-adopted terms. The list is deliberately minimal and high-precision: anything it misses is caught downstream by the relevance classifier, so the pre-filter need not be exhaustive. A handful of labels are themselves everyday words ("Induction," "Visualization," "Imagery," "Originality," "Creativity"); under the strict rule these are matched anyway, accepting the rare false exclusion of an everyday-sense passage as the cost of keeping the filter dumb.

The criteria below describe *why* verbatim labels are excluded and what technical register looks like. They are retained as rationale; a technical-register passage that contains **no** verbatim label (e.g., a bare test-name or clinical paraphrase) is **not** auto-removed and flows into normal annotation — typically landing as incidental mention or off-topic.

A passage uses technical labels if:

**3a. Verbatim CHC terminology.** The passage includes a CHC narrow ability name (e.g., "associative fluency," "ideational fluency," "induction," "visualization") or a recognizable paraphrase that borrows its structure from psychometric nomenclature (e.g., "retrieval fluency," "fluid reasoning skills"). These are excluded from Track A by design — they are the signal Track B is built to capture.

**3b. Test-instrument language.** The passage describes performance on a specific cognitive test or test format in a way that directly indexes the CHC ability being measured (e.g., "alternative uses task," "matrix reasoning," "number series," "verbal analogies"). The test is the CHC construct's operational definition; describing performance on it is not a naturalistic ability descriptor but a lab-register proxy.

> Example: *"He scored in the 90th percentile on the alternative uses task."* — This is a measurement statement, not an ability descriptor.

**3c. Academic or clinical prose about the construct.** Passages from textbooks, assessment reports, or psychology articles that explain or define the CHC ability in expert terms. These belong in Track B by design; they should not contaminate Track A's folk register.

**Important edge case — lay adoption of technical vocabulary:** Some CHC terms have partial lay adoption (e.g., "working memory," "fluid reasoning" in popular science and teacher discourse). These are coded as **technical label use** even when the speaker is not a psychometrician, because the term itself carries the construct's meaning rather than describing it through behavioral observation. The test: *could this passage have been written without any exposure to psychometric literature?* If no, it is a technical label use.

---

## Boundary Cases and Decision Rules

| Situation | Decision | Rationale |
|---|---|---|
| Passage describes the ability but contains a verbatim CHC label | Removed by the lexicon pre-filter before annotation | Even partial contamination prevents label-free sampling |
| Passage is technical in register (e.g., names a test) but contains no verbatim CHC label | Not auto-removed; annotate normally (usually incidental mention or off-topic) | Strict lexicon matches exact labels only; the classifier handles the rest |
| Passage clearly describes *a* cognitive ability but you are uncertain which CHC narrow ability | Code as neighboring-ability false positive; specify best candidate | Contributes to linguistic discriminant validity analysis; do not discard |
| Passage describes the ability via metaphor ("she has a mental filing system that cross-references everything") | Ability descriptor (include) | Metaphor and analogy are canonical descriptor vehicles |
| Passage describes a test or game the speaker played, with naturalistic description of the experience | Ability descriptor if the description characterizes the process; incidental if it only reports the outcome | Distinguish process description from outcome reporting |
| Passage is fiction or dialogue | Include if the ability descriptor criteria are met; register, not genre, is the filter | Fictional characters can be described with naturalistic cognitive language |
| Passage describes a difficulty, failure, or low level of the ability | Ability descriptor if it characterizes the specific behavior or conditions of the failure; incidental if it only evaluates low performance or names a condition/diagnosis | Abilities are dimensions; the low end describes the same construct (see *Deficits count*) |

---

## Connection to Annotation and the Four-Component Fingerprint

This annotation scheme — applied *after* the technical-label pre-filter — is the gating decision for corpus inclusion. Once a passage clears the **ability descriptor** criterion, it enters the pipeline for:

- **Lexical** — what words are statistically over-representative for this ability relative to the full corpus?
- **Conceptual frame** — what underlying schema organizes the description (e.g., FA as *traversal*; FI as *production volume*)?
- **Construal** — dispositional / agentive / spontaneous / output-based?
- **Evaluative framing** — rare vs. common; positively valued vs. neutral; what social context?

The operationalization document should be accompanied by a **calibration set**: 20–30 pre-coded example passages per broad stratum, covering all three annotation categories (ability descriptor, incidental mention, off-topic), with annotator rationales written out. Annotators should reach ≥ 0.80 Cohen's κ on the three-way annotation scheme before the full annotation run begins.

---

## Seed Sentence Authoring and Review Principles

> A seed sentence is itself an ability descriptor used as a retrieval query, so **every seed must satisfy Category 1 and avoid Categories 2–3** above. But seeds carry an additional burden the annotation criteria do not: a set of ten seeds for ability *aᵢ* must, as a set, *retrieve aᵢ and not its neighbors*. The principles below extend the inclusion criteria into authoring-and-review rules. They apply equally to human-authored seeds and to any machine-assisted revision; use them as a checklist when examining a candidate seed set.

**P1. No construct-name leakage (extends 1c / 3a).** A seed must not contain the ability's own name or the lexical root of that name. The folk register describes what the ability *does*; it does not say what it is *called*. The failure is easy to miss when the construct name is also an ordinary word.

> Fails: a Quantitative Reasoning (RQ) seed that says "she *reasons* through the proportions"; an Associational Fluency (FA) seed built on "*associations*"; a General Sequential Reasoning (RG) seed using "*deduce*." Fix by naming the behavior, not the faculty: "she works out in her head how the amounts relate"; "ideas that all tie back to the concept"; "he works through it step by step."

**P2. No neighbor-vocabulary contamination.** A seed for ability A must not be phrased in the *defining vocabulary of a sibling ability B*. This silently pulls the seed toward B's retrieval region and corrupts exactly the narrow-ability distinction the corpus is built to test.

> Fails: an Imagery (IM) seed that says "she *visualizes* the scene" — "visualize" is the name of the neighboring ability Visualization (Vz). Fix: "she brings the whole scene up in her head."

**P3. No definition-mirroring.** Subtler than a verbatim label (3a): a seed should not echo the phrasing of the construct's own technical *definition*, even when no label word appears. Importing the expert's framing imports the expert's ontology.

> Fails: an Attentional Control (AC) seed describing attention "like a *spotlight*," lifted from the definition's "spotlight or focal attention." Fix: "she keeps her concentration trained where it's needed and shuts out the rest."

**P4. No named test stimuli or instruments (extends 3b).** Naming the specific figure, task, or apparatus used to *measure* the ability indexes the instrument, not the lived behavior. Distinguish this from culturally common references, which remain folk.

> Fails: a Perceptual Alternations (PN) seed naming "the *Necker cube*." Fix by describing the phenomenon: "that wireframe-box drawing that keeps popping inside-out." Acceptable: "duck-or-rabbit," "old-woman/young-woman" — these are genuine common-culture references, not lab terms.

**P5. Discriminate against the nearest neighbor.** Each seed should carry at least one feature the closest neighboring ability would *not* satisfy — i.e., the contrast named in that ability's `discriminator`. A set of ten seeds that are each individually on-topic but collectively silent on the discriminating feature will not separate the pair. Pay special attention to the recurring contrast axes:

| Contrast axis | Hard pairs | The feature a seed must carry |
|---|---|---|
| Volume vs. relatedness vs. novelty | FI / FA / FO | quality-blind *count* (FI) vs. *fit to a given concept* (FA) vs. *originality/insight* (FO) |
| Knowing vs. doing vs. speed | KM / A3 / N | *knowing-that* vs. *executing correctly* vs. *over-learned speed* |
| Rate vs. rate-with-comprehension | Grw RS / Gs RS | *sheer pace* vs. *pace while still fully understanding* |
| Generating vs. copying vs. fluency | Grw WS / Gps WS / Gs WS | producing one's own text vs. physical *copying* vs. smooth fast either-way |
| Brief holding vs. durable memory | Wa·Wv (Gwm) / MA·MM·M6·MV | "just long enough, then gone" vs. lasting recall |
| Unknown vs. known target | CS / CF | recognizing *without* knowing the target vs. finding a *specified* one |
| Knowing vs. using | MK / Gp | knowing a tool's name/function vs. the manual skill of using it |
| Flow vs. audience-effect | OP / CM | uninterrupted *production* vs. *effective communication* to listeners |
| Spoken vs. written | MY / EU | sensitivity to *spoken* grammar vs. *written* mechanics |
| Clean vs. distorted signal | LS / UR | comprehending complex *clean* speech vs. catching speech *through noise* |

**P6. Characterize the process, not just the outcome (extends 2c).** A seed that only evaluates a result ("her answer was brilliant") is a Category-2 incidental mention and retrieves indiscriminately. Every seed needs a process or behavioral mechanism, even when wrapped in evaluative framing.

**P7. Vary the construal within a set.** Spread the ten seeds across dispositional ("she's the kind who…"), agentive ("he worked out…"), spontaneous ("the ideas just came…"), output-based ("her answer was…"), and hedged-observation ("I've watched her…") frames. Two reasons: (a) a monolithic frame makes the set's retrieval profile a property of the *frame* rather than the *ability*; (b) frame variety directly supports the construal component of the fingerprint. *Note:* the shared "a person doing X" framing common to all abilities' seeds is a corpus-wide common-mode direction handled at retrieval time (query debiasing / All-but-the-Top); P7 is about avoiding *within-set* monotony, not eliminating the person frame.

**P8. Benign homographs are not leakage.** A construct-name word used in an unrelated everyday sense does not violate P1. Apply the 3a edge-case test: *does the word here carry the construct's meaning, or an ordinary one?*

> Acceptable: "she knows the real scientific *reason*, not the myth" in a General Science Information (K1) seed — here "reason" means *cause*, not Gf "reasoning."

**Provenance note.** Because LLM-assisted revision can satisfy P1–P4 (mechanical, objective corrections) while quietly performing *construct alignment* — reshaping seeds toward the model's own representation of the ability — keep seed authorship and final revision decisions human-led, and record provenance (e.g., a `seeds_generated_by_llm` vs. human-authored distinction in the inventory). P5's discrimination judgments in particular are research decisions, not mechanical ones.

---

## What This Document Does Not Decide

- Which CHC narrow ability a confirmed ability descriptor describes (that is the neighboring-ability annotation task, handled separately).
- The threshold for passage-level relevance filtering (set empirically in Phase 2 using the LLM classifier).
- Whether an ability descriptor is central or peripheral to an ability's fingerprint (that is the salience weighting step in Phase 3).
