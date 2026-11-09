"""Compatibility shim for lm-format-enforcer with Transformers 4.52."""

from transformers.generation import logits_process


if not hasattr(logits_process, "LogitsWarper"):
    logits_process.LogitsWarper = logits_process.LogitsProcessor
