"""Evaluation and error-analysis utilities."""

from text2sql_rlvr.eval.execution_accuracy import EvalReport, ExampleOutcome, evaluate
from text2sql_rlvr.eval.sampling import SamplingReport, score_samples

__all__ = ["EvalReport", "ExampleOutcome", "SamplingReport", "evaluate", "score_samples"]
