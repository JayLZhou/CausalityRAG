"""Contribution-flow tools for token-level RAG answer resilience."""

from causalityrag.mixed_cut import (
    RawContributionNetwork,
    build_projected_token_contribution_network,
    remaining_support_flow,
    restrict_group_editable_units,
    solve_bicriteria_flow_interdiction,
)
from causalityrag.reader import (
    ReaderClient,
    answers_exact_match,
    answers_match,
)
from causalityrag.revision import apply_token_replacements

__all__ = [
    "RawContributionNetwork",
    "ReaderClient",
    "apply_token_replacements",
    "answers_exact_match",
    "answers_match",
    "build_projected_token_contribution_network",
    "remaining_support_flow",
    "restrict_group_editable_units",
    "solve_bicriteria_flow_interdiction",
]
