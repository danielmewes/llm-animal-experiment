import argparse
import json
import sys
from typing import Optional

from .experiment import TrialRecord, run_experiment

def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM Animal Guessing Game Experiment: tests whether a model's answers follow the expected 1/3 yes, 2/3 no distribution."
    )
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=10,
        help="Number of experiment trials to run (default: 10)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gemini-3.8-flash",
        help="Model to evaluate; names starting with 'claude' use the Anthropic API (default: gemini-3.8-flash)",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "anthropic"],
        default=None,
        help="API provider (default: inferred from the model name)",
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.95,
        help="Confidence level for the statistical hypothesis test (default: 0.95 for 95%% CI)",
    )
    parser.add_argument(
        "--thinking-level",
        type=str.lower,
        choices=["minimal", "low", "medium", "high"],
        default=None,
        help="Thinking level for the model (optional; Gemini 3 thinking level, or for Claude models adaptive thinking with the matching effort; minimal maps to low)",
    )
    parser.add_argument(
        "--set-expectations",
        action="store_true",
        help="Append a sentence to the initial prompt telling the model we don't expect the user to guess right the first time",
    )
    parser.add_argument(
        "--be-honest",
        action="store_true",
        help='Append " Be honest!" to the initial prompt',
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between successive trials (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for guess generation",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode using simulated responses without calling Google GenAI API",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (defaults to GEMINI_API_KEY or ANTHROPIC_API_KEY from environment or .env, depending on provider)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save detailed JSON trial results",
    )
    return parser.parse_args(args)


def one_line(text: str) -> str:
    return text.replace("\r", "").replace("\n", " ")


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 65)
    print("      LLM ANIMAL GUESSING BEHAVIOR EXPERIMENT")
    print("=" * 65)
    print(f"Model:            {args.model} {'(MOCK MODE)' if args.mock else ''}")
    print(f"Number of Trials: {args.trials}")
    print(f"Confidence Level: {args.confidence * 100:.1f}%")
    if args.thinking_level is not None:
        print(f"Thinking Level:   {args.thinking_level}")
    if args.set_expectations:
        print("Set Expectations: yes")
    if args.be_honest:
        print("Be Honest:        yes")
    if args.seed is not None:
        print(f"Random Seed:      {args.seed}")
    print("-" * 65)
    print("Starting trials...\n")

    def on_trial_complete(record: TrialRecord) -> None:
        if record.error:
            print(f"Trial {record.trial_id:>3}/{args.trials}: ERROR -> {one_line(record.error)}")
            return

        status_tag = f"[{record.classification}]"
        print(f"Trial {record.trial_id:>3}/{args.trials}: Turn 1 Response: \"{one_line(record.turn1_response)}\"")
        if record.turn1_thoughts:
            print(f"   Turn 1 Reasoning: {one_line(record.turn1_thoughts)}")
        print(f"   Guessed '{record.guessed_animal}' -> Response: \"{one_line(record.turn2_response)}\" {status_tag}")
        if record.turn2_thoughts:
            print(f"   Turn 2 Reasoning: {one_line(record.turn2_thoughts)}")
        for fu in record.follow_ups:
            print(f"   Follow-up '{fu.guessed_animal}': \"{one_line(fu.response)}\" [{fu.classification}]")
            if fu.thoughts:
                print(f"   Follow-up '{fu.guessed_animal}' Reasoning: {one_line(fu.thoughts)}")
        if record.invalid_reason:
            print(f"   Marked INVALID: {record.invalid_reason}")

    try:
        result = run_experiment(
            num_trials=args.trials,
            model=args.model,
            confidence_level=args.confidence,
            api_key=args.api_key,
            thinking_level=args.thinking_level,
            set_expectations=args.set_expectations,
            be_honest=args.be_honest,
            mock=args.mock,
            provider=args.provider,
            seed=args.seed,
            delay_between_trials=args.delay,
            on_trial_complete=on_trial_complete,
        )
    except Exception as exc:
        print(f"\n[ERROR] Experiment failed to run: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n" + result.stats.format_summary())

    if args.output:
        try:
            with open(args.output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nSaved detailed experiment results to: {args.output}")
        except Exception as e:
            print(f"\nWarning: Failed to save results to {args.output}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
