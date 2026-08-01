// Unredacting monospace documents using diffusion based LLMs
#import "@preview/tracl:0.8.1": *
#import "@preview/pergamon:0.7.1": *

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

// Single-column override (TRACL journal style)
#show: it => {
  set page(columns: 1)
  it
}

// Placeholder for figures that haven't been created yet: renders a dashed
// box with a TODO note so the layout/structure is visible in the PDF.
#let figstub(note, caption: none) = figure(
  block(
    width: 100%,
    height: 5cm,
    stroke: (dash: "dashed", paint: luma(160)),
    fill: luma(248),
    inset: 1em,
    align(center + horizon, text(fill: luma(120))[#note]),
  ),
  caption: caption,
)

#abstract[
  Redacted documents contain a potentially critical flaw, the use of box redaction removes the text content in a permanent way, however it informs the length of the text contained within the redaction.
  This paper proposes a method of unredaction, the reversal of redacted information.
  The technique involves the use of a diffusion based Large Language Model (LLM), the known size of each character, the surrounding text content, and the size of the redaction box.
  Based on that information, the contents of the redaction can be inferred, given that redaction is a destructive act, the unredaction does not truly restore the original state, but can provide with some degree of accuracy, the original contents.
]

= Introduction
It's important to understand the capabilities of unredact. 
@fig:types describes the different types of redactions and the capability of general purpose programs to unredact information.


#figure(
  // Single-column layout: the wide classification table spans the full
  // text width. (If reverting to the two-column ACL format, re-enable
  // `scope: "parent"` below to make it span both columns.)
  // scope: "parent",
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

#figstub(
  [TODO: pipeline diagram --- document -> locate redactions -> measure box -> derive min/max character bounds -> DLM infill with surrounding context -> rank candidates],
  caption: [The end-to-end unredact pipeline.],
) <fig:pipeline>

= Background <sec:background>
TODO: background on box redaction in declassified government documents (e.g. CIA reading room releases), the physical-size leak, and prior approaches to redaction recovery.

#figstub(
  [TODO: example page from a declassified document (e.g. SNIE 4-1-74, DOC_0001247371.pdf) with redaction boxes highlighted],
  caption: [Example of computer-applied box redaction in a released monospace government document.],
) <fig:example-doc>

= Methods <sec:methods>
unredact uses publicly available text diffusion models to infill redacted segments of text based on their surrounding text, and whatever incidental information we can get access to.
Redactions are primarily done through black bars covering the redacted text, which provides us with more information than may be immediately available.
unredact is solely focused towards monospace documents, where the physical size of characters on the page is uniform for the section being unredacted.
The reason for this is that given a known size redaction, a known character size, a maximum and minimum infill text length can be inferred.
This data gives us access to a subset of infill possibilities, when refined via the grammatical requirements of english, we can further compress the problem space to the following:
Can we come up with a grammatically correct, length approximate text infill that is logical in the situation?

Modern AI development has given us access to tools that logically map onto this problem, primarily Diffusion Language Models (DLMs).
Primarily the feature of DLMs that suits this problem space, is the capability for bidirectional context, or non-autoregressivity.
This describes the capability of a model to "see" all token positions concurrently, and generate based on the preceding and trailing context.
The benefit of this is obvious for our approach, where we have a given context already, and only a small subset must be filled in.

== Box Measurement
TODO: measuring the redaction box, dividing by the monospace glyph advance width to derive min/max character bounds (see box_measure.py).

#figstub(
  [TODO: diagram of a redaction box over monospace text, showing width in points divided by glyph width to yield a character-count range],
  caption: [Deriving character-count bounds from redaction-box geometry in a monospace document.],
) <fig:box-measure>

== Candidate Generation
TODO: filling the box with a diffusion LM (FIM / chat fallback), sampling multiple candidates, filtering to the character range.

== Semantic Ranking
TODO: embedding candidates and ranking by similarity to the surrounding context, with a contrastive baseline; optional keyness prior.

= Evaluation <sec:eval>
TODO: test corpora and ground truth (India / Pakistan passages), metrics (exact / near-miss / off-target, contrastive score).

#figstub(
  [TODO: example run output showing per-redaction candidates, char ranges, and scores],
  caption: [Example unredact run on a redaction with known ground truth.],
) <fig:run-output>

= Results <sec:results>
TODO: report graded recovery (exact / near-miss / off-target counts, mean contrastive score).

#figstub(
  [TODO: bar chart of recovery by category (exact / near-miss / off-target) across the test set],
  caption: [Graded recovery results across the redaction test set.],
) <fig:results>

= Discussion <sec:discussion>
TODO: what the physical-size constraint buys us, the salience trap (Pakistan/Israel), whole-document context vs. local context, the keyness-prior idea.

= Limitations <sec:limitations>
TODO: out-of-document (total) redactions are unrecoverable from text alone; monospace assumption; token cost of chat fallback.

= Conclusion <sec:conclusion>
TODO: summary of the method and results, and future work.

// Bibliography (Pergamon/ACL style). Uncomment once refs.bib has entries:
// #add-bib-resource(read("refs.bib"))
// #print-acl-bibliography()
