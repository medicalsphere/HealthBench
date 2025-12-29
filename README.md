# 🏥 HealthBench

Implementation of OpenAI's [HealthBench](https://openai.com/index/healthbench/) evaluation framework.

This repository is based on [simple-evals](https://github.com/openai/simple-evals) by OpenAI, which is the official reference implementation for HealthBench and other evaluations. However, the original repository is no longer being actively updated. We have forked and updated it to focus exclusively on the HealthBench evaluation with additional model support.

## 🎉 Updates

- **Gemini Support**: HealthBench now supports Google's Gemini models (gemini-2.5-pro, gemini-3-pro-preview, gemini-2.5-flash, gemini-3-flash-preview)


## 🔧 Setup

**Step 1:** Create and activate a conda environment:

```bash
conda create -n healthbench python=3.11
conda activate healthbench
```

**Step 2:** Install dependencies:

```bash
pip install -r requirements.txt
```

### 🔑 Environment Variables

Create a `.env` file in the project root with your API keys:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
```

## 📖 Usage

Run an evaluation with a specific model:

```bash
python -m HealthBench.simple_evals --model gpt-4.1-nano --eval healthbench_hard --n-threads 2
```

### ⚙️ Parameters

- `--model`: Model name (use `--list-models` to see available models)
- `--eval`: Evaluation type (`healthbench`, `healthbench_hard`, `healthbench_consensus`, `healthbench_meta`)
- `--n-threads`: Number of parallel threads (default: 120)
- `--n-repeats`: Number of evaluation repeats (default: 1)
- `--examples`: Number of examples to use (overrides default)
- `--debug`: Run in debug mode with reduced examples

## 💡 Tips & FAQ

### ⚡ Managing API Rate Limits

The `--n-threads` parameter controls how many parallel API requests are made during evaluation. The default value is **120**, which works well for high-tier API access. However, if you have low-tier access or limited rate limits for model APIs (like OpenAI), you should set this to a lower number to avoid hitting rate limits.

**Example for low-tier access:**

```bash
python -m HealthBench.simple_evals --model gpt-4o --eval healthbench --n-threads 5
```

**Recommended values based on your API tier:**

- **High-tier/Enterprise**: 120+ (default)
- **Standard tier**: 20-50
- **Low-tier/Free**: 1-10