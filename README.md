# HealthBench

Implementation of OpenAI's [HealthBench](https://openai.com/index/healthbench/) evaluation framework.

## Usage

Run an evaluation with a specific model:

```bash
python -m HealthBench.simple_evals --model gpt-4.1-nano --eval healthbench_hard --n-threads 2
```

### Parameters

- `--model`: Model name (use `--list-models` to see available models)
- `--eval`: Evaluation type (`healthbench`, `healthbench_hard`, `healthbench_consensus`, `healthbench_meta`)
- `--n-threads`: Number of parallel threads (default: 120)
- `--n-repeats`: Number of evaluation repeats (default: 1)
- `--examples`: Number of examples to use (overrides default)
- `--debug`: Run in debug mode with reduced examples