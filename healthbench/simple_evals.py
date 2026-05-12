import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from . import common
from .healthbench_eval import (
    HealthBenchEval,
    HealthBenchProfessionalEval,
    PROFESSIONAL_LENGTH_ADJUSTMENT_CENTER,
    PROFESSIONAL_LENGTH_ADJUSTMENT_PENALTY_PER_500_CHARS,
)
from .healthbench_meta_eval import HealthBenchMetaEval
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from .sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS
from .sampler.o_chat_completion_sampler import OChatCompletionSampler
from .sampler.responses_sampler import ResponsesSampler
from .sampler.gemini_sampler import GeminiSampler


# Models that use the Responses API (no temperature, optional reasoning_effort).
_RESPONSES_API_GRADER_MODELS = {
    "gpt-5.5-2026-04-23",
    "gpt-5.4-2026-03-05",
    "gpt-5.4-mini-2026-03-17",
    "o3-2025-04-16",
    "o4-mini-2025-04-16",
    "o1-pro",
}


def _build_healthbench_grader(args, default_grader):
    """Build the HealthBench rubric grader sampler from CLI args.

    Uses --healthbench-grader-model (default: gpt-4.1-2025-04-14) and
    --healthbench-grader-reasoning-effort. GPT-5.x and o-series models
    route to the Responses API; all others use Chat Completions.
    """
    model = args.healthbench_grader_model
    reasoning_effort = args.healthbench_grader_reasoning_effort

    if model in _RESPONSES_API_GRADER_MODELS or reasoning_effort is not None:
        return ResponsesSampler(
            model=model,
            reasoning_model=True,
            reasoning_effort=reasoning_effort,
        )
    return default_grader


def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Select a model by name. Also accepts a comma-separated list of models.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        help="Select an eval by name. Also accepts a comma-separated list of evals.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Number of repeats to run. Only supported for certain evals.",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=4,
        help="Number of threads to run. Only supported for HealthBench and HealthBenchMeta.",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write result files (default: results/ relative to repo root).",
    )
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )
    parser.add_argument(
        "--healthbench-input-path",
        type=str,
        default=None,
        help=(
            "Run the main HealthBench eval from a blobfile-readable JSONL path in "
            "HealthBench format. This only applies to --eval=healthbench."
        ),
    )
    parser.add_argument(
        "--healthbench-professional-mode",
        action="store_true",
        help=(
            "Require the HealthBench Professional option bundle: main HealthBench "
            "custom input path, gpt-5.4 low grader, and length adjustment."
        ),
    )
    parser.add_argument(
        "--healthbench-grader-model",
        type=str,
        default="gpt-4.1-2025-04-14",
        help=(
            "Grader model ID for HealthBench rubric evaluation "
            "(default: gpt-4.1-2025-04-14). "
            "GPT-5.x and o-series models use the Responses API; others use Chat Completions. "
            "For HealthBench Professional use gpt-5.4-2026-03-05 with "
            "--healthbench-grader-reasoning-effort low."
        ),
    )
    parser.add_argument(
        "--healthbench-grader-reasoning-effort",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help=(
            "Reasoning effort for the grader model when using the Responses API "
            "(low, medium, high). Only applies to reasoning models."
        ),
    )
    parser.add_argument(
        "--healthbench-length-adjustment-center",
        type=float,
        default=None,
        help=(
            "Center character count for HealthBench length adjustment. Must be set "
            "together with --healthbench-length-adjustment-penalty-per-500-chars."
        ),
    )
    parser.add_argument(
        "--healthbench-length-adjustment-penalty-per-500-chars",
        type=float,
        default=None,
        help=(
            "Numerical score penalty per 500 response characters for HealthBench "
            "length adjustment. Must be set together with "
            "--healthbench-length-adjustment-center."
        ),
    )

    args = parser.parse_args()

    # When running healthbench_professional, fill in paper defaults so they
    # show up explicitly in the args namespace rather than being silently baked
    # into the eval class.
    evals_requested = set(args.eval.split(",")) if args.eval is not None else set()
    if "healthbench_professional" in evals_requested:
        if args.healthbench_length_adjustment_center is None:
            args.healthbench_length_adjustment_center = PROFESSIONAL_LENGTH_ADJUSTMENT_CENTER
        if args.healthbench_length_adjustment_penalty_per_500_chars is None:
            args.healthbench_length_adjustment_penalty_per_500_chars = PROFESSIONAL_LENGTH_ADJUSTMENT_PENALTY_PER_500_CHARS
        if args.healthbench_grader_model == "gpt-4.1-2025-04-14" and args.healthbench_grader_reasoning_effort is None:
            args.healthbench_grader_model = "gpt-5.4-2026-03-05"
            args.healthbench_grader_reasoning_effort = "low"

    if args.healthbench_professional_mode:
        if evals_requested != {"healthbench"}:
            parser.error(
                "--healthbench-professional-mode requires --eval=healthbench"
            )
        if args.healthbench_input_path is None:
            parser.error(
                "--healthbench-professional-mode requires --healthbench-input-path"
            )
        if args.healthbench_grader_model != "gpt-5.4-2026-03-05" or args.healthbench_grader_reasoning_effort != "low":
            parser.error(
                "--healthbench-professional-mode requires "
                "--healthbench-grader-model gpt-5.4-2026-03-05 "
                "--healthbench-grader-reasoning-effort low"
            )
        if (
            args.healthbench_length_adjustment_center is None
            or args.healthbench_length_adjustment_penalty_per_500_chars is None
        ):
            parser.error(
                "--healthbench-professional-mode requires both HealthBench length "
                "adjustment flags"
            )

    models = {
        # Reasoning Models
        "o3": ResponsesSampler(
            model="o3-2025-04-16",
            reasoning_model=True,
        ),
        "o3-temp-1": ResponsesSampler(
            model="o3-2025-04-16",
            reasoning_model=True,
            temperature=1.0,
        ),
        "o3_high": ResponsesSampler(
            model="o3-2025-04-16",
            reasoning_model=True,
            reasoning_effort="high",
        ),
        "o3_low": ResponsesSampler(
            model="o3-2025-04-16",
            reasoning_model=True,
            reasoning_effort="low",
        ),
        # Default == Medium
        "o4-mini": ResponsesSampler(
            model="o4-mini-2025-04-16",
            reasoning_model=True,
        ),
        "o4-mini_high": ResponsesSampler(
            model="o4-mini-2025-04-16",
            reasoning_model=True,
            reasoning_effort="high",
        ),
        "o4-mini_low": ResponsesSampler(
            model="o4-mini-2025-04-16",
            reasoning_model=True,
            reasoning_effort="low",
        ),
        "o1-pro": ResponsesSampler(
            model="o1-pro",
            reasoning_model=True,
        ),
        "o1": OChatCompletionSampler(
            model="o1",
        ),
        "o1_high": OChatCompletionSampler(
            model="o1",
            reasoning_effort="high",
        ),
        "o1_low": OChatCompletionSampler(
            model="o1",
            reasoning_effort="low",
        ),
        "o1-preview": OChatCompletionSampler(
            model="o1-preview",
        ),
        "o1-mini": OChatCompletionSampler(
            model="o1-mini",
        ),
        # Default == Medium
        "o3-mini": OChatCompletionSampler(
            model="o3-mini",
        ),
        "o3-mini_high": OChatCompletionSampler(
            model="o3-mini",
            reasoning_effort="high",
        ),
        "o3-mini_low": OChatCompletionSampler(
            model="o3-mini",
            reasoning_effort="low",
        ),
        # GPT-5 models
        "gpt-5.5-2026-04-23": ResponsesSampler(
            model="gpt-5.5-2026-04-23",
            reasoning_model=True,
        ),
        "gpt-5.4-2026-03-05": ResponsesSampler(
            model="gpt-5.4-2026-03-05",
            reasoning_model=True,
        ),
        "gpt-5.4-mini-2026-03-17": ResponsesSampler(
            model="gpt-5.4-mini-2026-03-17",
            reasoning_model=True,
        ),
        # GPT-4.1 models
        "gpt-4.1": ChatCompletionSampler(
            model="gpt-4.1-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4.1-temp-1": ChatCompletionSampler(
            model="gpt-4.1-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            temperature=1.0,
        ),
        "gpt-4.1-mini": ChatCompletionSampler(
            model="gpt-4.1-mini-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4.1-nano": ChatCompletionSampler(
            model="gpt-4.1-nano-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        # GPT-4o models
        "gpt-4o": ChatCompletionSampler(
            model="gpt-4o",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4o-2024-11-20": ChatCompletionSampler(
            model="gpt-4o-2024-11-20",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4o-2024-08-06": ChatCompletionSampler(
            model="gpt-4o-2024-08-06",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4o-2024-08-06-temp-1": ChatCompletionSampler(
            model="gpt-4o-2024-08-06",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            temperature=1.0,
        ),
        "gpt-4o-2024-05-13": ChatCompletionSampler(
            model="gpt-4o-2024-05-13",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4o-mini": ChatCompletionSampler(
            model="gpt-4o-mini-2024-07-18",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        # GPT-4.5 model
        "gpt-4.5-preview": ChatCompletionSampler(
            model="gpt-4.5-preview-2025-02-27",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        # GPT-4-turbo model
        "gpt-4-turbo-2024-04-09": ChatCompletionSampler(
            model="gpt-4-turbo-2024-04-09",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
        ),
        # GPT-4 model
        "gpt-4-0613": ChatCompletionSampler(
            model="gpt-4-0613",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
        ),
        # GPT-3.5 Turbo model
        "gpt-3.5-turbo-0125": ChatCompletionSampler(
            model="gpt-3.5-turbo-0125",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
        ),
        "gpt-3.5-turbo-0125-temp-1": ChatCompletionSampler(
            model="gpt-3.5-turbo-0125",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            temperature=1.0,
        ),
        # Chatgpt models:
        "chatgpt-4o-latest": ChatCompletionSampler(
            model="chatgpt-4o-latest",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
            max_tokens=2048,
        ),
        "gpt-4-turbo-2024-04-09_chatgpt": ChatCompletionSampler(
            model="gpt-4-turbo-2024-04-09",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        ),
        # Claude models:
        "claude-sonnet-4-5-20250929": ClaudeCompletionSampler(
            model="claude-sonnet-4-5-20250929",
            system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        ),
        "claude-opus-4-5-20251101": ClaudeCompletionSampler(
            model="claude-opus-4-5-20251101",
            system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        ),
        "claude-3-opus-20240229_empty": ClaudeCompletionSampler(
            model="claude-3-opus-20240229",
            system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        ),
        "claude-3-7-sonnet-20250219": ClaudeCompletionSampler(
            model="claude-3-7-sonnet-20250219",
            system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        ),
        "claude-3-haiku-20240307": ClaudeCompletionSampler(
            model="claude-3-haiku-20240307",
        ),
        # Gemini models:
        "gemini-2.5-pro": GeminiSampler(
            model="gemini-2.5-pro",
        ),
        "gemini-3-pro-preview": GeminiSampler(
            model="gemini-3-pro-preview",
        ),
        "gemini-2.5-flash": GeminiSampler(
            model="gemini-2.5-flash",
        ),
        "gemini-3-flash-preview": GeminiSampler(
            model="gemini-3-flash-preview",
        ),
        "gemini-3.1-pro-preview": GeminiSampler(
            model="gemini-3.1-pro-preview",
        ),
        "gemini-3.1-flash-lite": GeminiSampler(
            model="gemini-3.1-flash-lite",
        )
    }

    if args.list_models:
        print("Available models:")
        for model_name in models.keys():
            print(f" - {model_name}")
        return

    if args.model:
        models_chosen = args.model.split(",")
        for model_name in models_chosen:
            if model_name not in models:
                print(f"Error: Model '{model_name}' not found.")
                return
        models = {model_name: models[model_name] for model_name in models_chosen}

    print(f"Running with args {args}")

    grading_sampler = ChatCompletionSampler(
        model="gpt-4.1-2025-04-14",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
    )
    equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")
    # ^^^ used for fuzzy matching, just for math

    healthbench_grading_sampler = _build_healthbench_grader(args, grading_sampler)

    def get_evals(eval_name, debug_mode):
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "healthbench":
                return HealthBenchEval(
                    grader_model=healthbench_grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                    subset_name=None,
                    input_path=args.healthbench_input_path,
                    length_adjustment_center=args.healthbench_length_adjustment_center,
                    length_adjustment_penalty_per_500_chars=args.healthbench_length_adjustment_penalty_per_500_chars,
                )
            case "healthbench_hard":
                return HealthBenchEval(
                    grader_model=healthbench_grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                    subset_name="hard",
                    length_adjustment_center=args.healthbench_length_adjustment_center,
                    length_adjustment_penalty_per_500_chars=args.healthbench_length_adjustment_penalty_per_500_chars,
                )
            case "healthbench_consensus":
                return HealthBenchEval(
                    grader_model=healthbench_grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                    subset_name="consensus",
                    length_adjustment_center=args.healthbench_length_adjustment_center,
                    length_adjustment_penalty_per_500_chars=args.healthbench_length_adjustment_penalty_per_500_chars,
                )
            case "healthbench_meta":
                return HealthBenchMetaEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                )
            case "healthbench_professional":
                return HealthBenchProfessionalEval(
                    grader_model=healthbench_grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                    length_adjustment_center=args.healthbench_length_adjustment_center,
                    length_adjustment_penalty_per_500_chars=args.healthbench_length_adjustment_penalty_per_500_chars,
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    if args.eval:
        evals_list = args.eval.split(",")
        evals = {}
        for eval_name in evals_list:
            try:
                evals[eval_name] = get_evals(eval_name, args.debug)
            except Exception as e:
                print(f"Error: eval '{eval_name}' not found.")
                print(f"Error message: {e}")
                return
    else:
        evals = {
            eval_name: get_evals(eval_name, args.debug)
            for eval_name in [
                "mmlu",
                "math",
                "gpqa",
                "mgsm",
                "drop",
                "humaneval",
                "simpleqa",
                "browsecomp",
                "healthbench",
                "healthbench_hard",
                "healthbench_consensus",
                "healthbench_meta",
            ]
        }

    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    print(f"Running the following evals: {list(evals.keys())}")
    print(f"Running evals for the following models: {list(models.keys())}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            file_stem = f"{eval_name}_{model_name}_{date_str}"
            report_filename = output_dir / f"{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            report_filename.write_text(common.make_report(result))
            assert result.metrics is not None
            metrics = result.metrics | {"score": result.score}
            metrics = dict(sorted(metrics.items()))
            print(metrics)
            result_filename = output_dir / f"{file_stem}{debug_suffix}.json"
            result_filename.write_text(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")

            full_result_filename = output_dir / f"{file_stem}{debug_suffix}_allresults.json"
            full_result_filename.write_text(json.dumps({
                "score": result.score,
                "metrics": result.metrics,
                "htmls": result.htmls,
                "convos": result.convos,
                "metadata": result.metadata,
            }, indent=2))
            print(f"Writing all results to {full_result_filename}")

            mergekey2resultpath[f"{file_stem}"] = result_filename
    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()