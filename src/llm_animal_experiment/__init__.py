"""LLM Animal Guessing Game Experiment package."""

from .classifier import classify_response
from .cli import main
from .experiment import ExperimentResult, TrialRecord, run_experiment
from .stats import StatisticalTestResult, run_statistical_test

__all__ = [
    "main",
    "run_experiment",
    "run_statistical_test",
    "classify_response",
    "ExperimentResult",
    "TrialRecord",
    "StatisticalTestResult",
]
