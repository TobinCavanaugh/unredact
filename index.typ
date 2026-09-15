// Unredacting monospace documents using diffusion based LLMs
#import "@preview/tracl:0.8.1": *

#show: acl.with(
  anonymous: false,
  title: [Unredacting monospace documents using diffusion based LLMs],
  authors: make-authors(
    (
      name: "Tobin Cavanaugh",
      affiliation: [#email("tobincavanaugh@gmail.com")],
    ),
  ),
  meta-authors: "Tobin Cavanaugh",
)

#abstract[
  Redacted documents contain a potentially critical flaw: the use of bounding box redaction permanently removes text pixels, but inadvertently informs the character length of the concealed text.
  This paper investigates *unredaction*—the probabilistic recovery of redacted spans in monospace declassified documents.
  By combining non-autoregressive Diffusion Language Models (DLMs) with geometric bounds extracted from monospace font advances and surrounding bidirectional context, candidate fills can be generated, filtered, and ranked.
  We formalize a taxonomy of redaction recoverability (Type 1 Coreference, Type 2 Expungement, and Type 3 Constrained Situations) and demonstrate on a benchmark evaluation that character-range verification successfully prunes overlength hallucinations, recovering up to 83.3% of short constrained spans.
]

= Introduction
It's important to understand the capabilities of unredact, this approach is academically interesting, but likely to be ineffective in unredacting effectively processed or non-monospace files.
@fig:types describes the different types of redactions and the capability of general purpose programs to unredact information.


#figure(
  placement: auto,
  scope: "parent",
  table(
    columns: (1.1fr, 1.8fr, 2.2fr, 1.4fr),
    inset: (x: 10pt, y: 8pt),
    fill: (x, y) => if y == 0 { rgb("e0e0e0") } else if calc.even(y) { rgb("f9f9f9") } else { none },
    stroke: 0.5pt + luma(120),
    align: (col, row) => if row == 0 { center + horizon } else { left + top },

    // Header
    [*Classification*], [*Theoretical Foundation*], [*Mechanism & Context*], [*Feasibility*],

    // Row 1
    [*Type 1*\ Juxtaposition / Coreference],
    [The semantic meaning or raw information exists elsewhere within the document, but has been selectively redacted in a specific instance to prevent the unification of two contexts.],
    [Acts as a deliberate breaking of coreference. The redaction aims to isolate two ideas, but the underlying data leaks through the broader document context.],
    [ Achievable ],

    // Row 2
    [*Type 2*\ Total Expungement],
    [The targeted concept, entity, or value is completely purged from the entire document, leaving zero internal traces.],
    [High entropy scenario. Examples include full removal of names or isolated numeric values. Without external data, exact reconstruction is mathematically impossible from the document alone.],
    [ Impossible Generally. Requires outside information.],

    // Row 3
    [*Type 3*\ Constrained / Situational],
    [The exact answer is absent, but the surrounding contextual clues and domain rules heavily bound the solution space.],
    [Redaction of a country "sharing a border with China." The context rules out non-neighboring nations, leaving a discrete, searchable subset of valid candidates.],
    [ Partial. Solvable as a constraint satisfaction problem.],
  ),
  caption: [Redaction types by recoverability],
) <fig:types>

unredact particularly aims to solve Type 1 redactions, where the semantic meaning is in the document, however the coreference of the information has been eliminated.

#figure(
  align(center,
    grid(
      columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr),
      gutter: 4pt,
      align: center + horizon,
      rect(width: 100%, inset: 6pt, stroke: 0.8pt + luma(110), fill: luma(248))[#text(8pt)[Visible context]],
      [→],
      rect(width: 100%, inset: 6pt, stroke: 0.8pt + luma(110), fill: luma(248))[#text(8pt)[Box bounds]],
      [→],
      rect(width: 100%, inset: 6pt, stroke: 0.8pt + luma(110), fill: luma(248))[#text(8pt)[DLM infill]],
      [→],
      rect(width: 100%, inset: 6pt, stroke: 0.8pt + luma(110), fill: luma(248))[#text(8pt)[Verify + clean]],
      [→],
      rect(width: 100%, inset: 6pt, stroke: 0.8pt + luma(110), fill: luma(248))[#text(8pt)[Rank candidates]],
    ),
  ),
  caption: [The end-to-end unredact pipeline.],
) <fig:pipeline>

= Background <sec:background>

Document sanitization in public records—such as those released under the Freedom of Information Act (FOIA) or published in declassification repositories like CIA CREST—frequently relies on digital or manual bounding boxes superimposed over sensitive text. Historically, a substantial majority of mid-to-late 20th-century intelligence assessments, diplomatic telegrams, and military memoranda were composed using fixed-pitch mechanical typewriters or early line printers (typically standardized to 10-pitch Pica or 12-pitch Elite fonts). 

In monospace documents, every character glyph occupies an identical horizontal advance width ($w_c$). Consequently, while a solid black rectangular overlay successfully destroys the visual and rasterized pixel data of the underlying glyphs, the physical geometry of the box inadvertently leaks a tight bounding constraint on the character length of the concealed string. When combined with modern bidirectional language modeling techniques @nie2025llada @bavarian2022fim, this physical leak transforms unredaction from an intractable open-ended generation problem into a constrained fill-in-the-middle search.

#figure(
  placement: auto,
  scope: "parent",
  block(
    width: 100%,
    stroke: 0.5pt + luma(150),
    fill: luma(252),
    inset: 12pt,
    radius: 3pt,
    align(left, [
      #set text(font: "Courier New", size: 9.5pt)
      #text(fill: luma(100))[MEMORANDUM FOR: The Director of Central Intelligence \
      SUBJECT: South Asia Strategic Assessment (Excerpt)] \ \
      #text(fill: rgb("202020"))[
        1. Recent reconnaissance confirms that the neighboring state has deployed \
        an #box(fill: black, width: 36pt, height: 9pt, baseline: 10%)[] nuclear capability along the northern border sector. \
        2. Senior military officials maintain that such installations serve \
        primarily as a strategic #box(fill: black, width: 54pt, height: 9pt, baseline: 10%)[] against potential incursions.
      ]
      #v(6pt)
      #line(length: 100%, stroke: 0.4pt + luma(180))
      #set text(size: 8.5pt, fill: luma(90))
      _Geometric side-channel:_ Redaction 1 width leaks $approx 6$ characters; Redaction 2 width leaks $approx 9$ characters.
    ])
  ),
  caption: [Simulated monospace declassified document excerpt. Monospace typesetting enforces a rigid grid, leaking the concealed character count through box width.],
) <fig:example-doc>

= Methods <sec:methods>
unredact uses publicly available text diffusion models to infill redacted segments of text based on their surrounding text, and whatever incidental information we can get access to.
Redactions are primarily done through black bars covering the redacted text, which provides us with more information than may be immediately available.
unredact is solely focused towards monospace documents, where the physical size of characters on the page is uniform for the section being unredacted.
The reason for this is that given a known size redaction, a known character size, a maximum and minimum infill text length can be inferred.
This data gives us access to a subset of infill possibilities, when refined via the grammatical requirements of english, we can further compress the problem space to the following:
Can we come up with a grammatically correct, length approximate text infill that is logical in the situation?

Modern AI development has given us access to tools that logically map onto this problem, primarily Diffusion Language Models (DLMs) @nie2025llada.
Primarily the feature of DLMs that suits this problem space, is the capability for bidirectional context, or non-autoregressivity.
This describes the capability of a model to "see" all token positions concurrently, and generate based on the preceding and trailing context.
The benefit of this is obvious for our approach, where we have a given context already, and only a small subset must be filled in.

== Box Measurement

To translate a visual or PDF bounding box into usable constraints for generation, we measure the bounding box width $W$ in points or pixels and estimate the baseline monospace glyph advance width $w_c$. Because manual redaction placement and digital clipping introduce minor boundary margins, we incorporate a small padding tolerance $delta$:

$ L_("min") = max(1, floor((W - 2 delta) / w_c)), quad L_("max") = ceil((W + 2 delta) / w_c) $

This yields a discrete character interval $[L_("min"), L_("max")]$ that acts as a hard filter on candidate generations.

#figure(
  placement: auto,
  scope: "parent",
  block(
    width: 100%,
    stroke: 0.5pt + luma(180),
    fill: luma(250),
    inset: 10pt,
    radius: 4pt,
    align(center, [
      #grid(
        columns: (auto, auto, auto),
        gutter: 14pt,
        align: horizon,
        [
          #set text(size: 8.5pt)
          *Monospace Grid Alignment* \
          #v(4pt)
          #table(
            columns: (20pt, 20pt, 20pt, 20pt, 20pt, 20pt, 20pt, 20pt, 20pt),
            inset: 4pt,
            stroke: 0.4pt + luma(180),
            fill: (x, y) => if x >= 1 and x <= 7 { rgb("333333") } else { none },
            align: center,
            [a], [#text(fill: white)[d]], [#text(fill: white)[e]], [#text(fill: white)[t]], [#text(fill: white)[e]], [#text(fill: white)[r]], [#text(fill: white)[r]], [#text(fill: white)[e]], [n]
          )
        ],
        [#text(16pt, fill: luma(120))[→]],
        [
          #set text(size: 8.5pt)
          #align(left)[
            *Derived Parameters:* \
            $W_("box") = 63.0 "pt"$ \
            $w_c = 7.0 "pt / char"$ \
            Tolerance $delta = 3.5 "pt"$ \
            #text(fill: rgb("006600"), weight: "bold")[$[L_("min"), L_("max")] = [7, 11]$ chars]
          ]
        ]
      )
    ])
  ),
  caption: [Monospace character-grid projection and derivation of character-count search interval $[L_("min"), L_("max")]$.],
) <fig:box-measure>

== Candidate Generation

For each redaction, the evaluator sends only the visible `before` and `after`
text, the estimated character range, and generation settings to a masked-diffusion model. The local experiment used `GSAI-ML/LLaDA-8B-Base` @nie2025llada through a small HTTP server on an RTX 3060 Ti. Because the model fills a fixed number of token positions rather than stopping at a character boundary, the client estimates a token span from the upper character bound and tries nearby spans. This is deliberately only a heuristic: tokens and characters do not correspond one-to-one.

Candidates are generated independently, then normalized before evaluation. The client decodes HTML entities such as `&nbsp;`, rejects replacement-character output, trims obvious continuation lines, and records discarded artifacts. Character-range verification remains authoritative: a candidate is usable for selection only when its character count falls within the box's `[min_chars, max_chars]` interval. Ground truth, document-level metadata, and grading information remain client-side and are never sent to the model.

== Semantic Ranking

The first blind selector uses the character range and candidate order. For diagnostics, the evaluator also embeds a local context window around the redaction with `all-MiniLM-L6-v2` @reimers2019sentencebert. It compares each candidate slot with the ground-truth slot and normalizes the result against an unrelated control token, producing a contrastive score rather than reporting the highly saturated raw sentence cosine. This score is an oracle diagnostic and is not a deployable performance metric because it uses the answer.

An adjacency-echo guard penalizes candidates that mostly reproduce content immediately next to the gap. An optional document-keyness prior can further adjust ranking, but it is kept separate from generation. This separation is important: a candidate can be generated successfully yet lose during selection, as happened for `large` in the random-remasking run.

= Evaluation <sec:eval>

The primary evaluation set is the six-case India passage in `redactions.json`. It is primarily a Type 2-style set: the missing words are strongly constrained by grammar and context, but are not necessarily repeated elsewhere in the visible document. The broader `redactions_broad.json` collection contains 21 synthetic declassified-style cases across recoverable, theme-echo, low-information, and absent-entity tiers.

We report three complementary quantities. *Blind exact recovery* is whether the top candidate selected without ground truth exactly matches the known span. *Candidate recall at K* asks whether the correct span appeared anywhere in the generated candidate pool, separating generation failure from ranking failure. The contrastive semantic score, along with near-miss and off-target labels, is an oracle diagnostic used to study ranking rather than to claim blind performance.

#figure(
  placement: auto,
  scope: "parent",
  table(
    columns: (1.2fr, 0.8fr, 1.6fr, 1.4fr),
    inset: (x: 6pt, y: 6pt),
    stroke: 0.5pt + luma(120),
    fill: (x, y) => if y == 0 { rgb("e0e0e0") } else if calc.even(y) { rgb("f9f9f9") } else { none },
    align: (col, row) => if row == 0 { center + horizon } else { left + top },
    [*Candidate*], [*Length*], [*Filter Status*], [*Ground Truth Match*],
    [`deterrent`], [9 chars], [#text(fill: rgb("007700"))[*PASSED* $[7, 11]$]], [Exact Match (GT)],
    [`deterrent 1`], [11 chars], [#text(fill: rgb("007700"))[*PASSED* $[7, 11]$]], [Sub-optimal fill],
    [`"final deterrent"`], [17 chars], [#text(fill: rgb("aa0000"))[REJECTED (overlength)]], [—],
    [`credible deterrent`], [18 chars], [#text(fill: rgb("aa0000"))[REJECTED (overlength)]], [—],
    [`strategic deterrent ":`], [23 chars], [#text(fill: rgb("aa0000"))[REJECTED (overlength)]], [—],
  ),
  caption: [Demonstration of candidate generation and character-bound pruning on case `ind2_deterrent` (Target: `deterrent`, search bounds $[7, 11]$ chars). Out of 6 raw DLM generations, overlong hallucinated phrases are pruned by physical bounds.],
) <fig:run-output>

= Results <sec:results>

The results are a small proof-of-concept rather than a claim of reliable document restoration. The strongest configuration recovered most of the retained short spans, but the sample is small, stochastic, and deliberately limited to cases with known ground truth.

== Initial local LLaDA experiment

We moved the first masked-infill tests from the remote Mercury backend to a local HTTP endpoint running `GSAI-ML/LLaDA-8B-Base` on an NVIDIA RTX 3060 Ti. The client was deliberately changed without modifying the server: it sent one request per candidate, varied the requested token span around the leaked character-range estimate, used 64 diffusion steps and temperature 0.8, cleaned obvious continuation/HTML/replacement-character artifacts, and retained client-side character-range verification. The test set was the six-case India-only `redactions.json` collection; the earlier Pakistan/Israel entry was removed from the active benchmark because its answer depended on external historical context.

The first successful 64-step, six-candidate run (`logs/20260804-010626_llada_smoke.txt` and `logs/20260804-010626_llada_smoke.json`) recovered 3/6 active benchmark cases exactly, improving on the preceding 0/6. These historical logs still contain the later-excluded Pakistan/Israel case, so the denominator here is recomputed after excluding it. It showed that tighter client-side generation lengths substantially reduced the earlier over-generation problem, although 14 candidates were still filtered as artifacts and the complete run took roughly seven minutes.

We then compared two remasking strategies with eight candidates per redaction, 64 diffusion steps, and temperature 0.8:

- `low_confidence` (`logs/20260804-013136_llada_low.txt`, `logs/20260804-013136_llada_low.json`) achieved 5/6 blind exact recovery (83.3%) and candidate recall at K of 5/6 on the retained cases. It recovered `active`, `deterrent`, `northern`, `large`, and `industrial`. The historical seven-case run made 56 requests, returned 36 usable candidates, filtered 18 artifacts, and recorded 3 continuation candidates.
- `random` (`logs/20260804-014304_llada_random.txt`, `logs/20260804-014304_llada_random.json`) achieved 4/6 blind exact recovery (66.7%) while retaining candidate recall at K of 5/6. It recovered `active`, `deterrent`, `northern`, and `industrial`; `large` appeared in the candidate pool but was not selected. This historical seven-case run returned 48 usable candidates and filtered only 7 artifacts, but the extra diversity did not improve blind selection.

For both historical eight-candidate runs there were 56 completed HTTP requests, no truncations, and no empty responses. The removed Pakistan/Israel case is retained only in these archived logs for provenance and is excluded from current scores because its ground truth was an external historical guess not clearly determined by the visible excerpt. These results support retaining `low_confidence` as the current default. They also suggest that candidate selection and artifact filtering, rather than simply increasing model size, are the next limitations to investigate.

#figure(
  placement: auto,
  scope: "parent",
  table(
    columns: (1.35fr, 1fr, 1fr, 0.8fr, 1fr),
    inset: (x: 6pt, y: 7pt),
    stroke: 0.5pt + luma(120),
    fill: (x, y) => if y == 0 { rgb("e0e0e0") } else if calc.even(y) { rgb("f9f9f9") } else { none },
    align: (col, row) => if row == 0 { center + horizon } else { left + top },
    [*Configuration*], [*Blind exact*], [*Recall at K*], [*Usable*], [*Artifacts*],
    [6 candidates, 64 steps], [3/6 (50.0%)], [3/6], [27], [14],
    [8, low-confidence, 64 steps], [5/6 (83.3%)], [5/6], [36], [18],
    [8, random, 64 steps], [4/6 (66.7%)], [5/6], [48], [7],
  ),
  caption: [Blind LLaDA recovery on the retained six-case benchmark. The first row is the successful six-candidate run; the two later rows are recomputed after excluding the archived Pakistan/Israel case.],
) <fig:results>

= Discussion <sec:discussion>

The experiments support a cautious but useful conclusion: the physical box constraint contributes real information, but it is not an exact decoder. Tight client-side token-span requests substantially reduced the long continuations seen in the initial LLaDA trial, and the best run recovered five of six retained spans. At the same time, the constraint is expressed in characters while the model operates in tokens. A token may represent a short word, a long word, punctuation, markup, or an unusual Unicode fragment, so a fixed token span cannot reliably hit a character interval.

The two remasking runs also separate generation from selection. Random remasking returned more usable candidates and fewer filtered artifacts, but it selected fewer exact answers. In the `large` case, the correct answer appeared in the candidate pool but was not chosen. More diversity therefore does not automatically produce better recovery; a genuinely blind reranker and better consensus signals remain open problems.

The redaction taxonomy gives a second boundary. Type 1 cases, where the answer is redundantly stated elsewhere in the document, should be the most favorable setting. The current six-case set is closer to Type 2: grammar and context constrain the answer, but several plausible alternatives may remain. Type 3 cases are different in kind. When the answer is absent from the document, exact recovery requires external information; a text-only system can at most recover a role, class, or constrained candidate set. This is a limit of the information available, not simply a weakness of the chosen model.

= Limitations <sec:limitations>

This study has several important limitations. The active benchmark contains only six retained cases, the model is stochastic, and the reported runs are not repeated enough to estimate variance. The synthetic broad set is useful for controlled experimentation but is not a substitute for simulated redactions drawn from real unredacted documents. The box bounds are approximate, and the token-to-character mismatch means that even a good semantic completion may be rejected or over-generated. Base-model output also requires artifact filtering for markup, continuation text, and unusual Unicode.

Exact match is intentionally strict and can undercount semantically acceptable alternatives, while the contrastive score is an oracle diagnostic because it uses ground truth. Conversely, some labeled ground truths are not uniquely determined by the visible text. The archived Pakistan/Israel case was therefore removed from the active benchmark rather than used to inflate or depress the headline score. These results should be read as evidence that the approach is plausible for short, contextually constrained spans—not as evidence that arbitrary redactions can be restored.

= Conclusion <sec:conclusion>

This project presents a small generate-and-verify pipeline for text-only redaction recovery. A monospace box leaks an approximate character range; a bidirectional diffusion model proposes fills using both sides of the gap; local verification removes candidates that do not fit; and ranking selects among the survivors. The local LLaDA experiment recovered five of six retained short spans in its best configuration, while also exposing the central limitation: character-length bounds are loose relative to token generation.

The most promising scope is short Type 1 and Type 2 redactions with strong surrounding context. Type 3 redactions, whose answers are absent from the document, require external evidence and should be treated as a different task. Future work should focus on better box measurement, realistic simulated-redaction datasets, repeated-run statistics, blind reranking, and semantic evaluation—not on presenting the current numbers as general recovery performance.

= Acknowledgments

This research and codebase were developed with assistance from DeepSeek V4 and Gemini 3.7 Flash.

#bibliography("refs.bib", title: [References])
