"""Replacement-based ARC-JSD attribution over context word tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from causalityrag.io import retrieved_contexts
from causalityrag.reader import READ_SYSTEM, READ_USER, format_passages
from causalityrag.token_units import TOKEN_RE, classify_token


def all_context_word_units(record: dict, *, k: int = 5, nlp=None) -> list[dict]:
    """Return every non-punctuation surface token in the retrieved contexts."""

    units = []
    for context in retrieved_contexts(record)[:k]:
        chunk_id = str(context["chunk_id"])
        annotation = nlp.annotate(context["text"]) if hasattr(nlp, "annotate") else None
        doc = nlp(context["text"]) if nlp is not None and annotation is None else None
        for match in TOKEN_RE.finditer(context["text"]):
            token = match.group(0)
            unit_type = classify_token(token)
            if unit_type == "PUNCT":
                continue
            spacy_token = None
            entity = None
            if annotation is not None:
                spacy_token = next(
                    (
                        item for item in annotation["tokens"]
                        if int(item["start"]) < match.end() and match.start() < int(item["end"])
                    ),
                    None,
                )
                entity = next(
                    (
                        item for item in annotation["entities"]
                        if int(item["start"]) < match.end() and match.start() < int(item["end"])
                    ),
                    None,
                )
                if entity is not None:
                    from causalityrag.rules import ONTO_TYPES
                    if str(entity["label"]).upper() in ONTO_TYPES:
                        unit_type = str(entity["label"]).upper()
            elif doc is not None:
                spacy_token = next(
                    (
                        item for item in doc
                        if item.idx < match.end() and match.start() < item.idx + len(item)
                    ),
                    None,
                )
                entity = next(
                    (
                        ent for ent in doc.ents
                        if ent.start_char < match.end() and match.start() < ent.end_char
                    ),
                    None,
                )
                if entity is not None:
                    from causalityrag.rules import ONTO_TYPES
                    if entity.label_.upper() in ONTO_TYPES:
                        unit_type = entity.label_.upper()
            unit = {
                "unit_id": f"token:{chunk_id}:{match.start()}:{match.end()}",
                "text": token,
                "type": unit_type,
                "chunk_id": chunk_id,
                "chunk_rank": int(context["rank"]),
                "chunk_char_start": match.start(),
                "chunk_char_end": match.end(),
                "cost": 1.0,
                "sources": ["replacement_arc_jsd"],
            }
            if spacy_token is not None and isinstance(spacy_token, dict):
                unit.update({
                    "pos": spacy_token["pos"],
                    "tag": spacy_token["tag"],
                    "lemma": spacy_token["lemma"],
                    "morph": spacy_token["morph"],
                })
            elif spacy_token is not None:
                unit.update({
                    "pos": spacy_token.pos_,
                    "tag": spacy_token.tag_,
                    "lemma": spacy_token.lemma_,
                    "morph": spacy_token.morph.to_dict(),
                })
            if entity is not None and isinstance(entity, dict) and unit_type == str(entity["label"]).upper():
                entity_tokens = entity["tokens"]
                entity_token_index = next(
                    (
                        index for index, item in enumerate(entity_tokens)
                        if int(item["start"]) < match.end() and match.start() < int(item["end"])
                    ),
                    0,
                )
                unit.update({
                    "entity_text": entity["text"],
                    "entity_token_index": entity_token_index,
                    "entity_token_count": len(entity_tokens),
                })
            elif entity is not None and unit_type == entity.label_.upper():
                entity_tokens = [item for item in entity if not item.is_space]
                entity_token_index = next(
                    (
                        index for index, item in enumerate(entity_tokens)
                        if item.idx < match.end() and match.start() < item.idx + len(item)
                    ),
                    0,
                )
                unit.update({
                    "entity_text": entity.text,
                    "entity_token_index": entity_token_index,
                    "entity_token_count": len(entity_tokens),
                })
            units.append(unit)
    return units


def context_sentence_units(
    record: dict,
    *,
    k: int = 5,
    nlp=None,
) -> tuple[list[dict], list[dict]]:
    """Return context word units and punctuation-delimited sentence spans."""

    import re

    units = all_context_word_units(record, k=k, nlp=nlp)
    sentences = []
    sentence_by_chunk: dict[str, list[dict]] = {}
    for context in retrieved_contexts(record)[:k]:
        chunk_id = str(context["chunk_id"])
        chunk_sentences = []
        for index, match in enumerate(re.finditer(r"\S(?:.*?\S)?(?:[.!?]+(?=\s|$)|$)", context["text"], re.S)):
            sentence = {
                "sentence_id": f"sentence:{chunk_id}:{match.start()}:{match.end()}",
                "chunk_id": chunk_id,
                "chunk_rank": int(context["rank"]),
                "sentence_index": index,
                "chunk_char_start": match.start(),
                "chunk_char_end": match.end(),
                "text": match.group(0),
            }
            chunk_sentences.append(sentence)
            sentences.append(sentence)
        sentence_by_chunk[chunk_id] = chunk_sentences
    for unit in units:
        containing = next(
            (
                sentence
                for sentence in sentence_by_chunk.get(str(unit["chunk_id"]), [])
                if int(sentence["chunk_char_start"]) <= int(unit["chunk_char_start"])
                and int(unit["chunk_char_end"]) <= int(sentence["chunk_char_end"])
            ),
            None,
        )
        if containing is None:
            raise ValueError(f"no sentence span contains {unit['unit_id']}")
        unit["sentence_id"] = containing["sentence_id"]
    return units, sentences


def ablate_context_sentence(contexts: Sequence[dict], sentence: dict) -> list[dict]:
    """Remove exactly one sentence span from its retrieved chunk."""

    edited = [dict(context) for context in contexts]
    chunk_id = str(sentence["chunk_id"])
    for context in edited:
        if str(context["chunk_id"]) != chunk_id:
            continue
        start = int(sentence["chunk_char_start"])
        end = int(sentence["chunk_char_end"])
        if str(context["text"])[start:end] != str(sentence["text"]):
            raise ValueError(f"sentence offset mismatch for {sentence['sentence_id']}")
        context["text"] = (str(context["text"])[:start] + " " + str(context["text"])[end:]).strip()
        return edited
    raise ValueError(f"missing chunk for {sentence['sentence_id']}")


def replace_context_units(
    contexts: Sequence[dict],
    units: Sequence[dict],
    replacements: dict[str, dict],
) -> list[dict]:
    """Apply a set of already validated one-word replacements by exact offset."""

    edited = {str(context["chunk_id"]): dict(context) for context in contexts}
    grouped: dict[str, list[dict]] = {}
    for unit in units:
        grouped.setdefault(str(unit["chunk_id"]), []).append(unit)
    for chunk_id, chunk_units in grouped.items():
        text = str(edited[chunk_id]["text"])
        for unit in sorted(chunk_units, key=lambda row: int(row["chunk_char_start"]), reverse=True):
            unit_id = str(unit["unit_id"])
            start = int(unit["chunk_char_start"])
            end = int(unit["chunk_char_end"])
            old = str(unit["text"])
            new = str(replacements[unit_id]["new"])
            if text[start:end] != old:
                raise ValueError(f"offset mismatch for {unit_id}")
            if not new or new.lower() == old.lower() or any(char.isspace() for char in new):
                raise ValueError(f"invalid replacement for {unit_id}: {new!r}")
            text = text[:start] + new + text[end:]
        edited[chunk_id]["text"] = text
    return [edited[str(context["chunk_id"])] for context in contexts]


def complete_pair_graph(
    units: Sequence[dict],
    singleton_scores: dict[str, float],
    pair_scores: dict[tuple[str, str], float],
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Build positive singleton and pair-synergy weights for the ratio solver."""

    unit_ids = {str(unit["unit_id"]) for unit in units}
    nodes = {
        unit_id: max(0.0, float(singleton_scores.get(unit_id, 0.0)))
        for unit_id in unit_ids
    }
    edges = {}
    for (left, right), joint_score in pair_scores.items():
        left, right = sorted((str(left), str(right)))
        if left == right or left not in unit_ids or right not in unit_ids:
            continue
        synergy = float(joint_score) - nodes[left] - nodes[right]
        if synergy > 0:
            edges[(left, right)] = synergy
    return nodes, edges


def fisher_pair_graph(
    units: Sequence[dict],
    singleton_scores: Sequence[float],
    fisher_features: Sequence[Sequence[float]],
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Estimate every positive pair synergy from singleton shift vectors.

    Each feature vector is rescaled so that its squared norm equals the exact
    singleton JSD.  Under a locally additive output-distribution shift, the
    pairwise JSD synergy is twice the feature-vector inner product.
    """

    import numpy as np

    if len(units) != len(singleton_scores) or len(units) != len(fisher_features):
        raise ValueError("units, singleton scores, and features must have equal lengths")
    if not units:
        return {}, {}
    features = np.asarray(fisher_features, dtype=np.float32)
    scores = np.maximum(0.0, np.asarray(singleton_scores, dtype=np.float32))
    norms = np.linalg.norm(features, axis=1)
    scales = np.divide(
        np.sqrt(scores),
        norms,
        out=np.zeros_like(norms),
        where=norms > 1e-12,
    )
    features *= scales[:, None]
    gram = 2.0 * (features @ features.T)
    unit_ids = [str(unit["unit_id"]) for unit in units]
    nodes = {unit_id: float(score) for unit_id, score in zip(unit_ids, scores)}
    edges = {}
    for left_index in range(len(unit_ids)):
        for right_index in range(left_index + 1, len(unit_ids)):
            weight = float(gram[left_index, right_index])
            if weight > 0:
                edges[(unit_ids[left_index], unit_ids[right_index])] = weight
    return nodes, edges


def lift_sentence_features_to_tokens(
    units: Sequence[dict],
    sentences: Sequence[dict],
    sentence_scores: Sequence[float],
    sentence_features: Sequence[Sequence[float]],
    token_response_routing: Sequence[Sequence[float]],
    *,
    response_length: int,
) -> tuple[list[list[float]], dict]:
    """Transport sentence intervention vectors to tokens with conservation."""

    import numpy as np

    if len(sentences) != len(sentence_scores) or len(sentences) != len(sentence_features):
        raise ValueError("sentence score/feature lengths do not match")
    routing = np.asarray(token_response_routing, dtype=np.float32)
    if routing.shape != (len(units), response_length):
        raise ValueError("token routing must have shape [tokens, response_length]")
    # Preserve shift direction through the lift.  Pairwise negative alignment
    # is discarded only when constructing the minimal positive supermodular
    # envelope; taking abs here would make nearly every token pair attractive.
    raw_features = np.asarray(sentence_features, dtype=np.float32)
    if raw_features.shape[1] % response_length:
        raise ValueError("sentence feature dimension must be divisible by response length")
    bins = raw_features.shape[1] // response_length
    sentence_vectors = raw_features.reshape(len(sentences), response_length, bins)
    scores = np.maximum(0.0, np.asarray(sentence_scores, dtype=np.float32))
    norms = np.linalg.norm(sentence_vectors.reshape(len(sentences), -1), axis=1)
    scales = np.divide(np.sqrt(scores), norms, out=np.zeros_like(norms), where=norms > 1e-12)
    sentence_vectors *= scales[:, None, None]

    sentence_index = {str(sentence["sentence_id"]): index for index, sentence in enumerate(sentences)}
    unit_groups: dict[str, list[int]] = {}
    for index, unit in enumerate(units):
        unit_groups.setdefault(str(unit["sentence_id"]), []).append(index)
    normalized_routing = np.zeros_like(routing)
    for sentence_id, indices in unit_groups.items():
        mass = routing[indices].sum(axis=0)
        for index in indices:
            normalized_routing[index] = np.divide(
                routing[index],
                mass,
                out=np.full(response_length, 1.0 / len(indices), dtype=np.float32),
                where=mass > 1e-12,
            )

    token_vectors = np.zeros((len(units), response_length, bins), dtype=np.float32)
    for index, unit in enumerate(units):
        sentence_id = str(unit["sentence_id"])
        token_vectors[index] = (
            normalized_routing[index, :, None]
            * sentence_vectors[sentence_index[sentence_id]]
        )

    max_error = 0.0
    for sentence_id, indices in unit_groups.items():
        lifted = token_vectors[indices].sum(axis=0)
        target = sentence_vectors[sentence_index[sentence_id]]
        max_error = max(max_error, float(np.max(np.abs(lifted - target))))
    return token_vectors.reshape(len(units), -1).tolist(), {
        "max_conservation_error": max_error,
        "response_length": response_length,
        "feature_bins_per_response_token": bins,
    }


def vector_supermodular_graph(
    units: Sequence[dict],
    token_features: Sequence[Sequence[float]],
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Build F(S)=||sum_i z_i||^2 as a positive pairwise graph."""

    import numpy as np

    features = np.asarray(token_features, dtype=np.float32)
    if features.shape[0] != len(units):
        raise ValueError("one token feature vector is required per unit")
    gram = features @ features.T
    unit_ids = [str(unit["unit_id"]) for unit in units]
    nodes = {unit_id: float(gram[index, index]) for index, unit_id in enumerate(unit_ids)}
    edges = {}
    for left in range(len(unit_ids)):
        for right in range(left + 1, len(unit_ids)):
            weight = 2.0 * float(gram[left, right])
            if weight > 0:
                edges[(unit_ids[left], unit_ids[right])] = weight
    return nodes, edges


@dataclass(frozen=True)
class CleanTrajectory:
    response_text: str
    response_ids: list[int]
    reference_log_probs: object


class ArcJsdModel:
    """Local Qwen scorer that exposes full-vocabulary teacher-forced JSD."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
    ) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised on the GPU host
            raise RuntimeError("ArcJsdModel requires torch and transformers") from exc

        self.torch = torch
        self.device = device
        torch_dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Transformers 4.52 validates Qwen's default TP plan even on a single
        # GPU, but torch<2.5 does not populate the supported TP-style table.
        config.base_model_tp_plan = None
        config.base_model_pp_plan = None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        ).to(device)
        self.model.eval()

    def response_to_context_attention_routing(
        self,
        question: str,
        contexts: Sequence[dict],
        units: Sequence[dict],
        response_ids: Sequence[int],
        *,
        last_layers: int = 8,
    ) -> list[list[float]]:
        """Extract upper-layer response-to-context attention for every token."""

        torch = self.torch
        prompt_text = self._prompt_text(question, contexts)
        encoded = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        prompt_ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        context_starts = {}
        cursor = 0
        for context in contexts:
            marker = f"[{context.get('chunk_id', '')}] {context.get('text', '')}"
            marker_start = prompt_text.find(marker, cursor)
            if marker_start < 0:
                raise ValueError(f"cannot locate chunk {context.get('chunk_id')} in rendered prompt")
            text_start = marker_start + len(f"[{context.get('chunk_id', '')}] ")
            context_starts[str(context["chunk_id"])] = text_start
            cursor = marker_start + len(marker)
        positions_by_unit = []
        for unit in units:
            start = context_starts[str(unit["chunk_id"])] + int(unit["chunk_char_start"])
            end = context_starts[str(unit["chunk_id"])] + int(unit["chunk_char_end"])
            positions = [
                index for index, (left, right) in enumerate(offsets)
                if right > start and left < end
            ]
            if not positions:
                raise ValueError(f"no model token overlaps {unit['unit_id']}")
            positions_by_unit.append(positions)

        sequence = prompt_ids + list(response_ids)
        input_ids = torch.tensor([sequence], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        attentions = output.attentions[-max(1, last_layers):]
        query_positions = torch.arange(
            len(prompt_ids) - 1,
            len(prompt_ids) - 1 + len(response_ids),
            device=self.device,
        )
        routing = torch.zeros((len(units), len(response_ids)), device=self.device)
        for attention in attentions:
            averaged = attention[0].mean(dim=0)[query_positions]
            for index, positions in enumerate(positions_by_unit):
                routing[index] += averaged[:, positions].sum(dim=-1)
        routing /= len(attentions)
        return routing.float().cpu().tolist()

    def clean_trajectory(
        self,
        question: str,
        contexts: Sequence[dict],
        *,
        max_new_tokens: int = 96,
    ) -> CleanTrajectory:
        torch = self.torch
        prompt_ids = self._prompt_ids(question, contexts)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        response_ids = generated[0, len(prompt_ids):].tolist()
        if response_ids and response_ids[-1] == self.tokenizer.eos_token_id:
            response_ids = response_ids[:-1]
        if not response_ids:
            raise RuntimeError("model generated an empty clean trajectory")
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        reference = self._log_probs_for_prompt_ids([prompt_ids], response_ids)[0].cpu()
        return CleanTrajectory(response_text, response_ids, reference)

    def score_context_variants(
        self,
        question: str,
        context_variants: Iterable[Sequence[dict]],
        trajectory: CleanTrajectory,
        *,
        batch_size: int = 4,
    ) -> list[float]:
        """Return summed response-token JSD for each modified context."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        scores: list[float] = []
        batch: list[list[int]] = []
        for contexts in context_variants:
            batch.append(self._prompt_ids(question, contexts))
            if len(batch) >= batch_size:
                scores.extend(self._score_prompt_batch(batch, trajectory))
                batch = []
        if batch:
            scores.extend(self._score_prompt_batch(batch, trajectory))
        return scores

    def score_context_variants_with_fisher_features(
        self,
        question: str,
        context_variants: Iterable[Sequence[dict]],
        trajectory: CleanTrajectory,
        *,
        batch_size: int = 4,
        top_k: int = 64,
    ) -> tuple[list[float], list[list[float]]]:
        """Return exact singleton JSDs and compact signed Fisher features.

        At each response position, the feature retains the clean
        distribution's top-k vocabulary entries and one probability-mass
        bucket for the remaining vocabulary.  This preserves the sign of the
        intervention-induced shift without materializing n full-vocabulary
        distributions.
        """

        if batch_size <= 0 or top_k <= 0:
            raise ValueError("batch_size and top_k must be positive")
        scores: list[float] = []
        features: list[list[float]] = []
        batch: list[list[int]] = []
        for contexts in context_variants:
            batch.append(self._prompt_ids(question, contexts))
            if len(batch) >= batch_size:
                batch_scores, batch_features = self._score_prompt_batch_with_features(
                    batch, trajectory, top_k
                )
                scores.extend(batch_scores)
                features.extend(batch_features)
                batch = []
        if batch:
            batch_scores, batch_features = self._score_prompt_batch_with_features(
                batch, trajectory, top_k
            )
            scores.extend(batch_scores)
            features.extend(batch_features)
        return scores, features

    def _score_prompt_batch(
        self,
        prompt_ids: list[list[int]],
        trajectory: CleanTrajectory,
    ) -> list[float]:
        candidate_log_probs = self._log_probs_for_prompt_ids(prompt_ids, trajectory.response_ids)
        reference = trajectory.reference_log_probs.to(candidate_log_probs.device)
        reference = reference.unsqueeze(0).expand(candidate_log_probs.shape[0], -1, -1)
        log_middle = self.torch.logaddexp(reference, candidate_log_probs) - math.log(2.0)
        jsd = 0.5 * (
            (reference.exp() * (reference - log_middle)).sum(dim=-1)
            + (candidate_log_probs.exp() * (candidate_log_probs - log_middle)).sum(dim=-1)
        )
        return jsd.sum(dim=-1).float().cpu().tolist()

    def _score_prompt_batch_with_features(
        self,
        prompt_ids: list[list[int]],
        trajectory: CleanTrajectory,
        top_k: int,
    ) -> tuple[list[float], list[list[float]]]:
        candidate_log_probs = self._log_probs_for_prompt_ids(prompt_ids, trajectory.response_ids)
        reference = trajectory.reference_log_probs.to(candidate_log_probs.device)
        reference_batch = reference.unsqueeze(0).expand(candidate_log_probs.shape[0], -1, -1)
        log_middle = self.torch.logaddexp(reference_batch, candidate_log_probs) - math.log(2.0)
        jsd = 0.5 * (
            (reference_batch.exp() * (reference_batch - log_middle)).sum(dim=-1)
            + (candidate_log_probs.exp() * (candidate_log_probs - log_middle)).sum(dim=-1)
        )

        top_k = min(top_k, reference.shape[-1] - 1)
        clean_probabilities = reference.exp()
        clean_top_probabilities, top_indices = clean_probabilities.topk(top_k, dim=-1)
        gather_indices = top_indices.unsqueeze(0).expand(candidate_log_probs.shape[0], -1, -1)
        candidate_top_probabilities = candidate_log_probs.exp().gather(-1, gather_indices)
        top_delta = candidate_top_probabilities - clean_top_probabilities.unsqueeze(0)
        clean_tail = (1.0 - clean_top_probabilities.sum(dim=-1)).clamp_min(1e-12)
        candidate_tail = (1.0 - candidate_top_probabilities.sum(dim=-1)).clamp_min(0.0)
        tail_delta = candidate_tail - clean_tail.unsqueeze(0)
        top_features = top_delta / (8.0 * clean_top_probabilities).sqrt().unsqueeze(0)
        tail_features = tail_delta / (8.0 * clean_tail).sqrt().unsqueeze(0)
        compact = self.torch.cat((top_features, tail_features.unsqueeze(-1)), dim=-1)
        return (
            jsd.sum(dim=-1).float().cpu().tolist(),
            compact.flatten(1).float().cpu().tolist(),
        )

    def _log_probs_for_prompt_ids(
        self,
        prompt_ids: list[list[int]],
        response_ids: list[int],
    ):
        torch = self.torch
        sequences = [prompt + response_ids for prompt in prompt_ids]
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(sequences):
            input_ids[row, :len(sequence)] = torch.tensor(sequence, device=self.device)
            attention_mask[row, :len(sequence)] = 1
        with torch.inference_mode():
            hidden = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            positions = torch.tensor(
                [len(prompt) - 1 for prompt in prompt_ids],
                dtype=torch.long,
                device=self.device,
            )[:, None] + torch.arange(len(response_ids), device=self.device)[None, :]
            rows = torch.arange(len(sequences), device=self.device)[:, None]
            response_hidden = hidden[rows, positions]
            logits = self.model.lm_head(response_hidden).float()
            return torch.log_softmax(logits, dim=-1)

    def _prompt_ids(self, question: str, contexts: Sequence[dict]) -> list[int]:
        return self.tokenizer(
            self._prompt_text(question, contexts),
            add_special_tokens=False,
        )["input_ids"]

    def _prompt_text(self, question: str, contexts: Sequence[dict]) -> str:
        messages = [
            {"role": "system", "content": READ_SYSTEM},
            {
                "role": "user",
                "content": READ_USER.format(
                    question=question,
                    passages=format_passages(list(contexts)),
                ),
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
