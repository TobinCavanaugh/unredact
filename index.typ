#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Unredacting monospace documents using diffusion based LLMs],
  abstract: [
    Redacted documents contain a potentially critical flaw, the use of box redaction removes the text content in a permanent way, however it informs the length of the text contained within the redaction.
    This paper proposes a method of unredaction, the reversal of redacted information. 
    The technique involves the use of a diffusion based Large Language Model (LLM), the known size of each character, the surrounding text content, and the size of the redaction box.
    Based on that information, the contents of the redaction can be inferred, given that redaction is a destructive act, the unredaction does not truly restore the original state, but can provide with some degree of accuracy, the original contents.
  ],
  authors: (
    (
      name: "Tobin Cavanaugh",
    //   department: [],
    //   organization: [],
    //   location: [Seattle, WA],
      email: "tobincavanaugh@gmail.com"
    ),
  ),
  index-terms: ("Scientific writing", "Typesetting", "Document creation", "Syntax"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction
#lorem(45)

Redaction Types:

#table(
  columns: (1fr, 2fr, 2.5fr, 1.5fr),
  fill: (x, y) => if y == 0 { rgb("e0e0e0") } else if calc.even(y) { rgb("f9f9f9") } else { none },
  stroke: 0.5pt + luma(120),
  align: (col, row) => if row == 0 { center + horizon } else { left + top },
  
  // Header
  [*Classification*], [*Theoretical Foundation*], [*Mechanism & Context*], [*Feasibility*],

  // Row 1
  [*Type 1*\ Juxtaposition / Coreference],
  [The semantic meaning or raw information exists elsewhere within the document, but has been selectively redacted in a specific instance to prevent the unification of two contexts.],
  [Acts as a deliberate breaking of coreference. The redaction aims to isolate two ideas, but the underlying data leaks through the broader document context.],
  [ Achievable\ High accuracy via intra-document context.],

  // Row 2
  [*Type 2*\ Total Expungement],
  [The targeted concept, entity, or value is completely purged from the entire document, leaving zero internal traces.],
  [High entropy scenario. Examples include full removal of names or isolated numeric values. Without external data, exact reconstruction is mathematically impossible from the document alone.],
  [ Generally Impossible\ Requires parametric memory or RAG.],

  // Row 3
  [*Type 3*\ Constrained / Situational],
  [The exact answer is absent, but the surrounding contextual clues and domain rules heavily bound the solution space.],
  [Redaction of a country "sharing a border with China." The context rules out non-neighboring nations, leaving a discrete, searchable subset of valid candidates.],
  [ Partial / Conditional\ Solvable as a constraint satisfaction problem.]
)

= Methods <sec:methods>
#lorem(45)

#lorem(80)

// #figure(
//   caption: [The Planets of the Solar System and Their Average Distance from the Sun],
//   placement: top,
//   table(
//     // Table styling is not mandated by the IEEE. Feel free to adjust these
//     // settings and potentially move them into a set rule.
//     columns: (6em, auto),
//     align: (left, right),
//     inset: (x: 8pt, y: 4pt),
//     stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
//     fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0  { rgb("#efefef") },

//     table.header[Planet][Distance (million km)],
//     [Mercury], [57.9],
//     [Venus], [108.2],
//     [Earth], [149.6],
//     [Mars], [227.9],
//     [Jupiter], [778.6],
//     [Saturn], [1,433.5],
//     [Uranus], [2,872.5],
//     [Neptune], [4,495.1],
//   )
// ) <tab:planets>

// In @tab:planets, you see the planets of the solar system and their average distance from the Sun.
// The distances were calculated with @eq:gamma that we presented in @sec:methods.

// #lorem(240)

// #lorem(240)