# HealthBench

A implementation of OpenAI's HealthBench and HealthBench Professional evaluation frameworks, based on [simple-evals](https://github.com/openai/simple-evals). This repository focuses exclusively on HealthBench evaluations, adds support for additional models (Claude, Gemini), and is kept in sync with the upstream scoring and evaluation logic so results are fully reproducible.

## Supported Evaluations

| Eval name | Description | Examples |
|---|---|---|
| `healthbench` | Full HealthBench benchmark | 5,000 |
| `healthbench_hard` | Difficult subset of HealthBench | 1,000 |
| `healthbench_consensus` | Consensus subset of HealthBench | 3,671 |
| `healthbench_meta` | Meta-evaluation (grader quality) | 29,511 |
| `healthbench_professional` | HealthBench Professional (clinician chat tasks) | 525 |

> **Grader:** `healthbench`, `healthbench_hard`, `healthbench_consensus`, and `healthbench_meta` use `gpt-4.1-2025-04-14` (Chat Completions API) by default. `healthbench_professional` uses `gpt-5.4-2026-03-05` at low reasoning effort (Responses API) per the paper. Override with `--healthbench-grader-model` and `--healthbench-grader-reasoning-effort`.
>
> **Data:** `healthbench`, `healthbench_hard`, and `healthbench_consensus` load from 🤗 [openai/healthbench](https://huggingface.co/datasets/openai/healthbench) with OpenAI public blob as fallback. `healthbench_professional` loads from 🤗 [openai/healthbench-professional](https://huggingface.co/datasets/openai/healthbench-professional) with a bundled local file as fallback. `healthbench_meta` loads from the OpenAI public blob.

**HealthBench Professional** evaluates LLMs on real clinician chat tasks spanning three use cases: care consult, writing and documentation, and medical research. It applies a length adjustment penalty by default (center=2,000 chars, penalty=0.0147 per 500 chars) as described in Section 4.1 of the paper. Data is loaded from HuggingFace automatically, with the bundled local file as a fallback if HuggingFace is unavailable.

## Updates

- **HealthBench Professional support**: `--eval=healthbench_professional` runs the 525-example benchmark. GPT-5.4 grader, length adjustment, and paper defaults are all applied automatically. Data loaded from HuggingFace with local fallback.
- **Explicit grader config**: `--healthbench-grader-model` and `--healthbench-grader-reasoning-effort` control the rubric grader for any eval. All active settings are shown in the args namespace at runtime.
- **Length adjustment**: `--healthbench-length-adjustment-center` and `--healthbench-length-adjustment-penalty-per-500-chars` are available for any HealthBench eval.
- **Custom data path**: `--healthbench-input-path` allows running the `healthbench` eval against any local or remote JSONL file in HealthBench format.
- **GPT-5 models**: `gpt-5.5-2026-04-23`, `gpt-5.4-2026-03-05`, `gpt-5.4-mini-2026-03-17` added via Responses API.
- **Gemini support**: `gemini-3-flash-preview`, `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite`.
- **Claude support**: `claude-opus-4-7`, `claude-opus-4-5-20251101`, `claude-sonnet-4-5-20250929`, `claude-3-7-sonnet-20250219`, `claude-3-haiku-20240307`.

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

This automatically loads the 525-example dataset from HuggingFace, applies the paper's default length adjustment (center=2,000, penalty=0.0147/500 chars), and uses `gpt-5.4-2026-03-05` at low reasoning as the grader — all visible in the printed args namespace at runtime.

**Override the grader for any eval:**

```bash
uv run python -m healthbench \
  --model gpt-4.1 \
  --eval healthbench \
  --healthbench-grader-model gpt-5.4-2026-03-05 \
  --healthbench-grader-reasoning-effort low
```

### Parameters

| Parameter | Description |
|---|---|
| `--model` | Model name (use `--list-models` to see all available models) |
| `--eval` | Evaluation type: `healthbench`, `healthbench_hard`, `healthbench_consensus`, `healthbench_meta`, `healthbench_professional` |
| `--n-threads` | Number of parallel threads (default: `4`) |
| `--n-repeats` | Number of evaluation repeats (default: `1`) |
| `--examples` | Number of examples to run (overrides default) |
| `--debug` | Run in debug mode with 10 examples |
| `--output-dir` | Directory to write results (default: `results/`) |
| `--healthbench-input-path` | Custom JSONL data path in HealthBench format (only for `--eval=healthbench`) |
| `--healthbench-grader-model` | Grader model ID (default: `gpt-4.1-2025-04-14`; auto-set to `gpt-5.4-2026-03-05` for `healthbench_professional`) |
| `--healthbench-grader-reasoning-effort` | Reasoning effort for the grader: `low`, `medium`, `high` (auto-set to `low` for `healthbench_professional`) |
| `--healthbench-length-adjustment-center` | Center character count for length penalty (auto-set to `2000` for `healthbench_professional`) |
| `--healthbench-length-adjustment-penalty-per-500-chars` | Score penalty per 500 response characters (auto-set to `0.0147` for `healthbench_professional`) |
| `--healthbench-professional-mode` | Validation bundle for `--eval=healthbench` with a custom input path — requires `--healthbench-input-path`, `--healthbench-grader-model gpt-5.4-2026-03-05`, `--healthbench-grader-reasoning-effort low`, and both length adjustment flags |

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
