"""Experimental MIRAGE saliency adapted to editable word units."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from causalityrag.arc_jsd import ArcJsdModel, CleanTrajectory


@dataclass(frozen=True)
class MirageDiagnostics:
    cti_scores: list[float]
    cti_threshold: float
    selected_response_indices: list[int]
    response_tokens: list[str]
    foil_tokens: list[str]

    def to_dict(self) -> dict:
        return {
            "cti_scores": self.cti_scores,
            "cti_threshold": self.cti_threshold,
            "selected_response_indices": self.selected_response_indices,
            "response_tokens": self.response_tokens,
            "foil_tokens": self.foil_tokens,
        }


def context_sensitive_indices(scores: Sequence[float], std_threshold: float = 1.0) -> tuple[list[int], float]:
    """Apply MIRAGE's example-level mean-plus-population-std CTI filter."""

    if not scores:
        return [], 0.0
    mean = sum(float(score) for score in scores) / len(scores)
    variance = sum((float(score) - mean) ** 2 for score in scores) / len(scores)
    threshold = mean + std_threshold * math.sqrt(variance)
    selected = [index for index, score in enumerate(scores) if abs(float(score)) >= threshold]
    return selected, threshold


class MirageScorer:
    """Official MIRAGE CTI plus contrastive embedding-gradient saliency.

    MIRAGE scores model subwords. Since this project edits whole surface-word
    units, gradients of all subwords overlapping a word are concatenated before
    taking their L2 norm. Scores are summed across CTI-selected response tokens,
    matching MIRAGE's aggregation across generated tokens.
    """

    def __init__(
        self,
        model: ArcJsdModel,
        *,
        cti_std_threshold: float = 1.0,
        gradient_checkpointing: bool = True,
    ) -> None:
        self.model = model
        self.cti_std_threshold = cti_std_threshold
        self.gradient_checkpointing = gradient_checkpointing

    def score(
        self,
        question: str,
        contexts: Sequence[dict],
        units: Sequence[dict],
        trajectory: CleanTrajectory,
    ) -> tuple[dict[str, float], MirageDiagnostics]:
        torch = self.model.torch
        full_log_probs = trajectory.reference_log_probs.to(self.model.device)
        contextless_prompt_ids = self.model._prompt_ids(question, [])
        contextless_log_probs = self.model._log_probs_for_prompt_ids(
            [contextless_prompt_ids], trajectory.response_ids
        )[0]
        cti = (
            full_log_probs.exp() * (full_log_probs - contextless_log_probs)
        ).sum(dim=-1)
        cti_scores = cti.float().cpu().tolist()
        selected_indices, threshold = context_sensitive_indices(
            cti_scores, self.cti_std_threshold
        )
        foil_ids = contextless_log_probs.argmax(dim=-1)
        response_tokens = self.model.tokenizer.convert_ids_to_tokens(trajectory.response_ids)
        foil_tokens = self.model.tokenizer.convert_ids_to_tokens(foil_ids.tolist())
        diagnostics = MirageDiagnostics(
            cti_scores=cti_scores,
            cti_threshold=threshold,
            selected_response_indices=selected_indices,
            response_tokens=response_tokens,
            foil_tokens=foil_tokens,
        )
        unit_ids = [str(unit["unit_id"]) for unit in units]
        scores = {unit_id: 0.0 for unit_id in unit_ids}
        if not selected_indices or not units:
            return scores, diagnostics

        prompt_text = self.model._prompt_text(question, contexts)
        encoded = self.model.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        prompt_ids = list(encoded["input_ids"])
        positions_by_unit = _unit_model_positions(prompt_text, encoded["offset_mapping"], contexts, units)
        sequence = prompt_ids + list(trajectory.response_ids)
        input_ids = torch.tensor([sequence], dtype=torch.long, device=self.model.device)
        input_embeds = self.model.model.get_input_embeddings()(input_ids).detach()
        input_embeds.requires_grad_(True)
        attention_mask = torch.ones_like(input_ids)

        for parameter in self.model.model.parameters():
            parameter.requires_grad_(False)
        causal_model = self.model.model
        checkpointing = self.gradient_checkpointing
        if checkpointing:
            attention_dropout = float(
                getattr(causal_model.config, "attention_dropout", 0.0)
            )
            if attention_dropout:
                raise RuntimeError(
                    "MIRAGE gradient checkpointing requires zero attention dropout"
                )
            causal_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            causal_model.train()
        try:
            hidden = causal_model.model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            response_positions = torch.tensor(
                [len(prompt_ids) - 1 + index for index in selected_indices],
                dtype=torch.long,
                device=self.model.device,
            )
            logits = causal_model.lm_head(hidden[0, response_positions]).float()
            log_probs = torch.log_softmax(logits, dim=-1)

            for offset, response_index in enumerate(selected_indices):
                actual_id = int(trajectory.response_ids[response_index])
                foil_id = int(foil_ids[response_index])
                objective = log_probs[offset, actual_id]
                if foil_id != actual_id:
                    objective = objective - log_probs[offset, foil_id]
                gradient = torch.autograd.grad(
                    objective,
                    input_embeds,
                    retain_graph=offset + 1 < len(selected_indices),
                )[0][0]
                for unit_id, positions in zip(unit_ids, positions_by_unit):
                    scores[unit_id] += float(
                        torch.linalg.vector_norm(gradient[positions].float()).detach().cpu()
                    )
        finally:
            if checkpointing:
                causal_model.eval()
                causal_model.gradient_checkpointing_disable()
        return scores, diagnostics


def _unit_model_positions(
    prompt_text: str,
    offsets: Sequence[tuple[int, int]],
    contexts: Sequence[dict],
    units: Sequence[dict],
) -> list[list[int]]:
    context_starts = {}
    cursor = 0
    for context in contexts:
        marker = f"[{context.get('chunk_id', '')}] {context.get('text', '')}"
        marker_start = prompt_text.find(marker, cursor)
        if marker_start < 0:
            raise ValueError(f"cannot locate chunk {context.get('chunk_id')} in rendered prompt")
        context_starts[str(context["chunk_id"])] = marker_start + len(
            f"[{context.get('chunk_id', '')}] "
        )
        cursor = marker_start + len(marker)

    positions_by_unit = []
    for unit in units:
        start = context_starts[str(unit["chunk_id"])] + int(unit["chunk_char_start"])
        end = context_starts[str(unit["chunk_id"])] + int(unit["chunk_char_end"])
        positions = [
            index for index, (left, right) in enumerate(offsets)
            if int(right) > start and int(left) < end
        ]
        if not positions:
            raise ValueError(f"no model token overlaps {unit['unit_id']}")
        positions_by_unit.append(positions)
    return positions_by_unit
