# Unredact: Recovering Monospace Box Redactions via Diffusion Language Models

[Read the Paper (PDF)](index.pdf)

## Abstract

Redacted documents contain a potentially critical flaw: the use of bounding box redaction permanently removes text pixels, but inadvertently informs the character length of the concealed text. This paper investigates **unredaction**—the probabilistic recovery of redacted spans in monospace declassified documents. By combining non-autoregressive Diffusion Language Models (DLMs) with geometric bounds extracted from monospace font advances and surrounding bidirectional context, candidate fills can be generated, filtered, and ranked. We formalize a taxonomy of redaction recoverability (Type 1 Coreference, Type 2 Expungement, and Type 3 Constrained Situations) and demonstrate on a benchmark evaluation that character-range verification successfully prunes overlength hallucinations, recovering up to 83.3% of short constrained spans.

---

## Core Concept

In monospace documents (e.g., FOIA releases, CIA CREST records, typewriter-era government memoranda), every character occupies a fixed advance width ($w_c$). Consequently, while a solid black rectangular overlay destroys the underlying glyph pixels, the box width $W$ leaks the character length interval $[L_{\min}, L_{\max}]$:

$$L_{\min} = \max\left(1, \left\lfloor \frac{W - 2\delta}{w_c} \right\rfloor\right), \quad L_{\max} = \left\lceil \frac{W + 2\delta}{w_c} \right\rceil$$

```text
Visible Context (Prefix / Suffix) ──┐
                                     ├──> DLM Infilling (LLaDA) ──> Character-Range Pruning ──> Semantic Ranking
Leaked Box Geometry [L_min, L_max] ─┘
```

### Redaction Taxonomy

| Classification | Description | Feasibility |
| :--- | :--- | :--- |
| **Type 1: Juxtaposition / Coreference** | Entity appears elsewhere in the document context. | Achievable (in-document coreference resolution). |
| **Type 2: Total Expungement** | Concept is purged from document, but bound by grammar/syntax. | Grammatically constrained search space. |
| **Type 3: Constrained / Situational** | Entity is entirely absent; requires external domain context. | Partial (role/class recovery or candidate sets). |

---

## Usage Guide

### 1. Installation

```bash
git clone https://github.com/TobinCavanaugh/unredact.git
cd unredact
pip install -r requirements.txt
```

### 2. Offline Dry-Run Verification

Test the generate-and-verify pipeline and dataset validation offline:

```bash
python validate_dataset.py
python unredact.py --dry-run --redactions redactions.json
python unredact.py --dry-run --redactions redactions.json --n-eff
```

### 3. Running with LLaDA (Local Diffusion Server)

To run infilling using local `LLaDA-8B-Base`:

1. **Start the LLaDA Server (GPU machine):**
   ```bash
   pip install -r llada_server_requirements.txt
   python llada_server.py --model GSAI-ML/LLaDA-8B-Base --host 0.0.0.0 --port 8000
   ```

2. **Run the Evaluator:**
   ```bash
   python unredact.py --backend llada --llada-url http://127.0.0.1:8000 \
     --llada-steps 64 --llada-temperature 0.8 \
     --llada-remasking low_confidence --candidates 8 \
     --redactions redactions.json
   ```

### 4. Running with Mercury API

Set your `INCEPTION_API_KEY` in `.env` or environment variables:

```bash
python unredact.py --backend mercury --mode chat --semantic
python unredact.py --backend mercury --mode fim --semantic --prior-weight 0.5
```

### 5. Compiling the Paper

Compile the Typst paper to PDF:

```bash
typst compile index.typ index.pdf
```

---

## Acknowledgments

This project was developed with assistance from DeepSeek V4 and Gemini 3.7 Flash.

