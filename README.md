# HealthBench

A implementation of OpenAI's HealthBench and HealthBench Professional evaluation frameworks, based on [simple-evals](https://github.com/openai/simple-evals). This repository focuses exclusively on HealthBench evaluations, adds support for additional models (Claude, Gemini), and is kept in sync with the upstream scoring and evaluation logic so results are fully reproducible.

## Supported Evaluations

| Eval name | Description | Examples | Data source |
|---|---|---|---|
| `healthbench` | Full HealthBench benchmark | ~5,000 | OpenAI public blob |
| `healthbench_hard` | Difficult subset of HealthBench | ~1,000 | OpenAI public blob |
| `healthbench_consensus` | Consensus subset of HealthBench | ~1,000 | OpenAI public blob |
| `healthbench_meta` | Meta-evaluation (grader quality) | ~500 | OpenAI public blob |
| `healthbench_professional` | HealthBench Professional (clinician chat tasks) | 525 | Bundled locally |

**HealthBench Professional** ([paper](docs/HealthBench-Professional.pdf)) evaluates LLMs on real clinician chat tasks spanning three use cases: care consult, writing and documentation, and medical research. It applies a length adjustment penalty by default (center=2,000 chars, penalty=0.0147 per 500 chars) as described in Section 4.1 of the paper.

## Updates

- **HealthBench Professional support**: `--eval=healthbench_professional` runs the 525-example benchmark with the paper's default length adjustment. Data is bundled at `data/assets/healthbench_professional_eval.jsonl`.
- **Length adjustment**: `--healthbench-length-adjustment-center` and `--healthbench-length-adjustment-penalty-per-500-chars` flags are now available for any HealthBench eval.
- **Custom data path**: `--healthbench-input-path` allows running the `healthbench` eval against any local or remote JSONL file in HealthBench format.
- **GPT-5.4 grader**: `--healthbench-use-gpt-5-4-low-grader` switches the rubric grader to GPT-5.4 at low reasoning effort (the default grader used in the HealthBench Professional paper).
- **Gemini support**: `gemini-2.5-pro`, `gemini-3-pro-preview`, `gemini-2.5-flash`, `gemini-3-flash-preview`.
- **Claude support**: Claude 3 and Claude 4 model families.

## Setup

**Step 1:** Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Step 2:** Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/HealthBench.git
cd HealthBench
uv sync
```

### Environment Variables

Create a `.env` file in the project root with your API keys:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
```

## Usage

Run all commands from inside the HealthBench directory.

**Quick test (10 examples, useful for verifying setup):**

```bash
uv run python -m healthbench \
  --model gpt-5.5-2026-04-23 \
  --eval healthbench_hard \
  --n-threads 4 \
  --examples 10
```

**Run full HealthBench:**

```bash
uv run python -m healthbench \
  --model gpt-4.1 \
  --eval healthbench
```

**Run HealthBench Hard or Consensus:**

```bash
uv run python -m healthbench --model gpt-4.1 --eval healthbench_hard
uv run python -m healthbench --model gpt-4.1 --eval healthbench_consensus
```

**Run HealthBench Professional:**

```bash
uv run python -m healthbench \
  --model gpt-4.1 \
  --eval healthbench_professional
```

This automatically loads the bundled 525-example dataset and applies the paper's default length adjustment. To use the GPT-5.4 grader as in the original paper:

```bash
uv run python -m healthbench \
  --model gpt-4.1 \
  --eval healthbench_professional \
  --healthbench-use-gpt-5-4-low-grader
```

**Run HealthBench with a custom data file and length adjustment** (the manual equivalent of `--healthbench-professional-mode`):

```bash
uv run python -m healthbench \
  --model gpt-4.1 \
  --eval healthbench \
  --healthbench-input-path /path/to/custom_data.jsonl \
  --healthbench-use-gpt-5-4-low-grader \
  --healthbench-length-adjustment-center 2000 \
  --healthbench-length-adjustment-penalty-per-500-chars 0.0147
```

Or use the validation bundle shorthand:

```bash
uv run python -m healthbench \
  --model gpt-4.1 \
  --eval healthbench \
  --healthbench-professional-mode \
  --healthbench-input-path /path/to/data.jsonl \
  --healthbench-use-gpt-5-4-low-grader \
  --healthbench-length-adjustment-center 2000 \
  --healthbench-length-adjustment-penalty-per-500-chars 0.0147
```

### Parameters

- `--model`: Model name (use `--list-models` to see all available models)
- `--eval`: Evaluation type — `healthbench`, `healthbench_hard`, `healthbench_consensus`, `healthbench_meta`, `healthbench_professional`
- `--n-threads`: Number of parallel threads (default: 4)
- `--n-repeats`: Number of evaluation repeats (default: 1)
- `--examples`: Number of examples to run (overrides default)
- `--debug`: Run in debug mode with 10 examples
- `--output-dir`: Directory to write results (default: `results/` in the repo root)
- `--healthbench-input-path`: Custom JSONL data path in HealthBench format (only for `--eval=healthbench`)
- `--healthbench-professional-mode`: Validation bundle requiring `--healthbench-input-path`, `--healthbench-use-gpt-5-4-low-grader`, and both length adjustment flags
- `--healthbench-use-gpt-5-4-low-grader`: Use GPT-5.4 (low reasoning) as the rubric grader
- `--healthbench-length-adjustment-center`: Center character count for length penalty
- `--healthbench-length-adjustment-penalty-per-500-chars`: Score penalty per 500 response characters

## Tips & FAQ

### Managing API Rate Limits

The `--n-threads` parameter controls parallel API requests. The default is **4**, which is safe for local development on any machine. Raise it if you have high-tier API access and want faster runs.

| API tier | Recommended `--n-threads` |
|---|---|
| High-tier / Enterprise | 50–120 |
| Standard | 10–20 |
| Low-tier / Free / Local | 4 (default) |

```bash
uv run python -m healthbench \
  --model gpt-4o \
  --eval healthbench \
  --n-threads 20
```
