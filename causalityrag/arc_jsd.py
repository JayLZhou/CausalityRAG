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


def ablate_context_sentences(
    contexts: Sequence[dict],
    sentences: Sequence[dict],
) -> list[dict]:
    """Remove multiple sentence spans while preserving original offsets."""

    edited = {str(context["chunk_id"]): dict(context) for context in contexts}
    grouped: dict[str, list[dict]] = {}
    for sentence in sentences:
        grouped.setdefault(str(sentence["chunk_id"]), []).append(sentence)
    for chunk_id, chunk_sentences in grouped.items():
        if chunk_id not in edited:
            raise ValueError(f"missing chunk for {chunk_sentences[0]['sentence_id']}")
        text = str(edited[chunk_id]["text"])
        for sentence in sorted(
            chunk_sentences,
            key=lambda row: int(row["chunk_char_start"]),
            reverse=True,
        ):
            start = int(sentence["chunk_char_start"])
            end = int(sentence["chunk_char_end"])
            if text[start:end] != str(sentence["text"]):
                raise ValueError(f"sentence offset mismatch for {sentence['sentence_id']}")
            text = text[:start] + " " + text[end:]
        edited[chunk_id]["text"] = text.strip()
    return [edited[str(context["chunk_id"])] for context in contexts]


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
        "lift": "attention_routing",
    }


def optimal_transport_sentence_features_to_tokens(
    units: Sequence[dict],
    sentences: Sequence[dict],
    sentence_scores: Sequence[float],
    sentence_features: Sequence[Sequence[float]],
    token_response_routing: Sequence[Sequence[float]],
    *,
    response_length: int,
    epsilon: float = 0.2,
    column_penalty: float = 1.0,
    iterations: int = 50,
) -> tuple[list[list[float]], dict]:
    """Lift sentence shifts with semi-relaxed entropic optimal transport.

    Response-position shift energy is the fixed source marginal. Attention
    gives the transport cost and attention-derived token capacity is a soft
    target marginal, coupling allocations across all response positions.
    """

    import numpy as np

    if epsilon <= 0 or column_penalty <= 0 or iterations <= 0:
        raise ValueError("OT epsilon, column penalty, and iterations must be positive")
    if len(sentences) != len(sentence_scores) or len(sentences) != len(sentence_features):
        raise ValueError("sentence score/feature lengths do not match")
    routing = np.asarray(token_response_routing, dtype=np.float64)
    if routing.shape != (len(units), response_length):
        raise ValueError("token routing must have shape [tokens, response_length]")
    raw_features = np.asarray(sentence_features, dtype=np.float64)
    if raw_features.shape[1] % response_length:
        raise ValueError("sentence feature dimension must be divisible by response length")
    bins = raw_features.shape[1] // response_length
    sentence_vectors = raw_features.reshape(len(sentences), response_length, bins)
    scores = np.maximum(0.0, np.asarray(sentence_scores, dtype=np.float64))
    norms = np.linalg.norm(sentence_vectors.reshape(len(sentences), -1), axis=1)
    scales = np.divide(np.sqrt(scores), norms, out=np.zeros_like(norms), where=norms > 1e-12)
    sentence_vectors *= scales[:, None, None]

    sentence_index = {str(sentence["sentence_id"]): index for index, sentence in enumerate(sentences)}
    unit_groups: dict[str, list[int]] = {}
    for index, unit in enumerate(units):
        unit_groups.setdefault(str(unit["sentence_id"]), []).append(index)

    token_vectors = np.zeros((len(units), response_length, bins), dtype=np.float64)
    max_conservation_error = 0.0
    max_row_error = 0.0
    column_kls = []
    exponent = column_penalty / (column_penalty + epsilon)
    for sentence_id, indices in unit_groups.items():
        sentence_vector = sentence_vectors[sentence_index[sentence_id]]
        energy = np.sum(sentence_vector * sentence_vector, axis=1)
        total_energy = float(energy.sum())
        if total_energy <= 1e-15:
            continue
        source = energy / total_energy
        attention = np.maximum(routing[indices].T, 1e-12)
        # exp(-(-log attention)/epsilon)
        kernel = attention ** (1.0 / epsilon)
        capacity = attention.sum(axis=0)
        capacity /= capacity.sum()
        right = np.ones(len(indices), dtype=np.float64)
        for _ in range(iterations):
            left = source / np.maximum(kernel @ right, 1e-15)
            column_mass = right * (kernel.T @ left)
            right = (capacity / np.maximum(column_mass, 1e-15)) ** exponent
        left = source / np.maximum(kernel @ right, 1e-15)
        plan = left[:, None] * kernel * right[None, :]
        plan *= source[:, None] / np.maximum(plan.sum(axis=1, keepdims=True), 1e-15)
        allocation = plan / np.maximum(source[:, None], 1e-15)
        for local_index, unit_index in enumerate(indices):
            token_vectors[unit_index] = allocation[:, local_index, None] * sentence_vector
        lifted = token_vectors[indices].sum(axis=0)
        max_conservation_error = max(max_conservation_error, float(np.max(np.abs(lifted - sentence_vector))))
        max_row_error = max(max_row_error, float(np.max(np.abs(plan.sum(axis=1) - source))))
        column = plan.sum(axis=0)
        column_kls.append(float(np.sum(column * np.log(np.maximum(column, 1e-15) / capacity))))

    return token_vectors.reshape(len(units), -1).astype(np.float32).tolist(), {
        "max_conservation_error": max_conservation_error,
        "max_source_marginal_error": max_row_error,
        "mean_token_capacity_kl": float(np.mean(column_kls)) if column_kls else 0.0,
        "response_length": response_length,
        "feature_bins_per_response_token": bins,
        "lift": "semi_relaxed_entropic_ot",
        "ot_epsilon": epsilon,
        "ot_column_penalty": column_penalty,
        "ot_iterations": iterations,
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


def fisher_sketch_pair_graph(
    units: Sequence[dict],
    node_weights: dict[str, float],
    signatures: Sequence[Sequence[float]],
    *,
    min_affinity: float = 0.0,
    max_neighbors: int = 0,
    pair_mass_budget: float = 0.0,
) -> tuple[dict[tuple[str, str], float], dict]:
    """Condition JSD unary magnitudes on token-gate Fisher interactions."""

    import numpy as np

    features = np.asarray(signatures, dtype=np.float32)
    if features.shape[0] != len(units):
        raise ValueError("one Fisher signature is required per unit")
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("Fisher signatures must be a nonempty matrix")
    if not 0.0 <= min_affinity <= 1.0:
        raise ValueError("min_affinity must be in [0, 1]")
    if max_neighbors < 0:
        raise ValueError("max_neighbors must be non-negative")
    if pair_mass_budget < 0:
        raise ValueError("pair_mass_budget must be non-negative")
    norms = np.linalg.norm(features, axis=1)
    normalized = np.divide(
        features,
        norms[:, None],
        out=np.zeros_like(features),
        where=norms[:, None] > 1e-12,
    )
    affinity = normalized @ normalized.T
    unit_ids = [str(unit["unit_id"]) for unit in units]
    unary = np.asarray(
        [max(0.0, float(node_weights.get(unit_id, 0.0))) for unit_id in unit_ids],
        dtype=np.float32,
    )
    candidates = []
    for left in range(len(unit_ids)):
        for right in range(left + 1, len(unit_ids)):
            rho = float(affinity[left, right])
            if rho <= min_affinity or unary[left] <= 0 or unary[right] <= 0:
                continue
            candidates.append((rho, left, right))
    retained_indices = set(range(len(candidates)))
    if max_neighbors:
        incident: list[list[int]] = [[] for _ in unit_ids]
        for index, (_, left, right) in enumerate(candidates):
            incident[left].append(index)
            incident[right].append(index)
        retained_indices = set()
        for indices in incident:
            indices.sort(key=lambda index: -candidates[index][0])
            retained_indices.update(indices[:max_neighbors])

    raw_interactions = {}
    positive_affinities = []
    for index in sorted(retained_indices):
        rho, left, right = candidates[index]
        weight = 2.0 * rho * float(np.sqrt(unary[left] * unary[right]))
        if weight > 0:
            raw_interactions[(unit_ids[left], unit_ids[right])] = weight
            positive_affinities.append(rho)

    raw_incident = {unit_id: 0.0 for unit_id in unit_ids}
    for (left, right), weight in raw_interactions.items():
        raw_incident[left] += weight
        raw_incident[right] += weight
    interactions = dict(raw_interactions)
    if pair_mass_budget > 0:
        unary_by_id = dict(zip(unit_ids, unary.tolist()))
        loads = {
            unit_id: raw_incident[unit_id]
            / max(pair_mass_budget * unary_by_id[unit_id], 1e-12)
            for unit_id in unit_ids
        }
        interactions = {
            edge: weight / max(1.0, loads[edge[0]], loads[edge[1]])
            for edge, weight in raw_interactions.items()
        }
    incident = {unit_id: 0.0 for unit_id in unit_ids}
    for (left, right), weight in interactions.items():
        incident[left] += weight
        incident[right] += weight
    unary_by_id = dict(zip(unit_ids, unary.tolist()))
    raw_incident_ratios = [
        raw_incident[unit_id] / unary_by_id[unit_id]
        for unit_id in unit_ids if unary_by_id[unit_id] > 0
    ]
    incident_ratios = [
        incident[unit_id] / unary_by_id[unit_id]
        for unit_id in unit_ids if unary_by_id[unit_id] > 0
    ]
    total_pairs = len(unit_ids) * (len(unit_ids) - 1) // 2
    return interactions, {
        "probes": int(features.shape[1]),
        "nonzero_signature_count": int((norms > 1e-12).sum()),
        "min_affinity": min_affinity,
        "max_neighbors": max_neighbors,
        "pair_mass_budget": pair_mass_budget,
        "positive_pair_candidates": len(candidates),
        "positive_pair_edges": len(interactions),
        "positive_pair_fraction": len(interactions) / total_pairs if total_pairs else 0.0,
        "mean_positive_affinity": (
            sum(positive_affinities) / len(positive_affinities)
            if positive_affinities else 0.0
        ),
        "max_positive_affinity": max(positive_affinities, default=0.0),
        "max_raw_incident_pair_to_unary": max(raw_incident_ratios, default=0.0),
        "max_normalized_incident_pair_to_unary": max(incident_ratios, default=0.0),
        "raw_pair_mass": sum(raw_interactions.values()),
        "normalized_pair_mass": sum(interactions.values()),
        "pair_scale": "2*positive_cosine(fisher_signature_i,fisher_signature_j)*sqrt(a_i*a_j)",
    }


def conservative_sentence_pair_graph(
    units: Sequence[dict],
    sentences: Sequence[dict],
    sentence_scores: Sequence[float],
    token_features: Sequence[Sequence[float]],
    sentence_pair_scores: dict[tuple[str, str], float],
) -> tuple[dict[str, float], dict[tuple[str, str], float], dict]:
    """Lift sentence singleton and joint effects to a token graph.

    Token masses are derived from the information-flow-weighted feature norms.
    Each sentence's singleton JSD and each positive second-order sentence
    interaction are conserved exactly across their token nodes and edges.
    """

    import numpy as np

    if len(sentences) != len(sentence_scores):
        raise ValueError("sentences and sentence scores must have equal lengths")
    features = np.asarray(token_features, dtype=np.float32)
    if features.shape[0] != len(units):
        raise ValueError("one token feature vector is required per unit")

    sentence_score_by_id = {
        str(sentence["sentence_id"]): max(0.0, float(score))
        for sentence, score in zip(sentences, sentence_scores)
    }
    indices_by_sentence: dict[str, list[int]] = {}
    for index, unit in enumerate(units):
        indices_by_sentence.setdefault(str(unit["sentence_id"]), []).append(index)

    token_mass: dict[str, float] = {}
    nodes: dict[str, float] = {}
    unary_errors = []
    for sentence_id, indices in indices_by_sentence.items():
        norms = np.linalg.norm(features[indices], axis=1)
        if float(norms.sum()) <= 1e-12:
            shares = np.full(len(indices), 1.0 / len(indices), dtype=np.float32)
        else:
            shares = norms / norms.sum()
        score = sentence_score_by_id.get(sentence_id, 0.0)
        for index, share in zip(indices, shares):
            unit_id = str(units[index]["unit_id"])
            token_mass[unit_id] = float(share)
            nodes[unit_id] = score * float(share)
        unary_errors.append(abs(sum(nodes[str(units[index]["unit_id"])] for index in indices) - score))

    edges: dict[tuple[str, str], float] = {}
    pair_errors = []
    positive_sentence_pairs = 0
    skipped_sentence_pairs = 0
    sentence_synergies = []
    for pair, joint_score in sentence_pair_scores.items():
        left_sentence, right_sentence = sorted((str(pair[0]), str(pair[1])))
        if left_sentence == right_sentence:
            continue
        synergy = max(
            0.0,
            float(joint_score)
            - sentence_score_by_id.get(left_sentence, 0.0)
            - sentence_score_by_id.get(right_sentence, 0.0),
        )
        sentence_synergies.append({
            "left_sentence_id": left_sentence,
            "right_sentence_id": right_sentence,
            "joint_score": float(joint_score),
            "synergy": synergy,
        })
        if synergy <= 0:
            continue
        left_indices = indices_by_sentence.get(left_sentence, [])
        right_indices = indices_by_sentence.get(right_sentence, [])
        if not left_indices or not right_indices:
            skipped_sentence_pairs += 1
            continue
        positive_sentence_pairs += 1
        lifted_total = 0.0
        for left_index in left_indices:
            left_id = str(units[left_index]["unit_id"])
            for right_index in right_indices:
                right_id = str(units[right_index]["unit_id"])
                edge = tuple(sorted((left_id, right_id)))
                weight = synergy * token_mass[left_id] * token_mass[right_id]
                edges[edge] = edges.get(edge, 0.0) + weight
                lifted_total += weight
        pair_errors.append(abs(lifted_total - synergy))

    return nodes, edges, {
        "max_unary_conservation_error": max(unary_errors, default=0.0),
        "max_pair_conservation_error": max(pair_errors, default=0.0),
        "n_sentence_pairs_scored": len(sentence_pair_scores),
        "n_positive_sentence_pairs": positive_sentence_pairs,
        "n_skipped_positive_sentence_pairs": skipped_sentence_pairs,
        "sentence_pair_synergies": sentence_synergies,
    }


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

    def token_gate_fisher_signatures(
        self,
        question: str,
        contexts: Sequence[dict],
        units: Sequence[dict],
        response_ids: Sequence[int],
        *,
        probes: int = 8,
        seed: int = 0,
    ) -> tuple[list[list[float]], dict]:
        """Sketch the local JSD Hessian between all context-word gates.

        A categorical Fisher probe at every response position is pulled back
        through the complete transformer to all word-level input gates.  The
        resulting signatures approximate the JSD Gauss-Newton Gram matrix
        with ``probes`` reverse passes, independent of the number of pairs.
        """

        if probes <= 0:
            raise ValueError("probes must be positive")
        torch = self.torch
        prompt_ids, positions_by_unit = self._prompt_context_token_positions(
            question, contexts, units
        )
        sequence = prompt_ids + list(response_ids)
        input_ids = torch.tensor([sequence], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        base_embeddings = self.model.get_input_embeddings()(input_ids).detach()

        assignment = torch.zeros(
            (len(sequence), len(units)),
            dtype=base_embeddings.dtype,
            device=self.device,
        )
        for unit_index, positions in enumerate(positions_by_unit):
            assignment[positions, unit_index] = 1.0
        gates = torch.ones(
            len(units),
            dtype=base_embeddings.dtype,
            device=self.device,
            requires_grad=True,
        )
        scales = 1.0 + assignment @ (gates - 1.0)
        embeddings = base_embeddings * scales[None, :, None]

        hidden = self.model.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        positions = (
            torch.arange(len(response_ids), device=self.device)
            + len(prompt_ids) - 1
        )
        logits = self.model.lm_head(hidden[0, positions]).float()
        probabilities = torch.softmax(logits.detach(), dim=-1)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        columns = []
        scale = math.sqrt(8.0)
        for probe_index in range(probes):
            sampled = torch.multinomial(
                probabilities,
                num_samples=1,
                replacement=True,
                generator=generator,
            ).squeeze(-1)
            sampled_logits = logits.gather(-1, sampled[:, None]).squeeze(-1)
            expected_logits = (probabilities * logits).sum(dim=-1)
            probe_objective = ((sampled_logits - expected_logits) / scale).sum()
            gradient = torch.autograd.grad(
                probe_objective,
                gates,
                retain_graph=probe_index + 1 < probes,
                create_graph=False,
            )[0]
            columns.append(gradient.detach().float())
        signatures = torch.stack(columns, dim=1) / math.sqrt(float(probes))
        diagnostics = {
            "probes": probes,
            "seed": int(seed),
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(response_ids),
            "context_units": len(units),
            "model_subtokens_gated": int(assignment.sum().item()),
            "signature_norm_mean": float(signatures.norm(dim=1).mean().item()),
            "signature_norm_max": float(signatures.norm(dim=1).max().item()),
            "probe_covariance": "categorical_fisher/8",
        }
        return signatures.cpu().tolist(), diagnostics

    def score_token_gate_sets(
        self,
        question: str,
        contexts: Sequence[dict],
        units: Sequence[dict],
        gate_sets: Sequence[Sequence[str]],
        trajectory: CleanTrajectory,
        *,
        batch_size: int = 4,
        gate_value: float = 0.0,
    ) -> tuple[list[float], dict]:
        """Measure teacher-forced JSD after jointly gating context tokens.

        Every requested set is evaluated as one intervention: all model
        subtokens covered by its word units have their input embeddings scaled
        by ``gate_value``.  The clean response trajectory is fixed, so this is
        a batched forward-only proposal probe and never edits or matches answer
        strings.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= gate_value <= 1.0:
            raise ValueError("gate_value must be in [0, 1]")
        if not gate_sets:
            return [], {
                "gate_sets": 0,
                "batch_size": batch_size,
                "gate_value": gate_value,
                "forward_batches": 0,
            }

        torch = self.torch
        prompt_ids, positions_by_unit = self._prompt_context_token_positions(
            question, contexts, units
        )
        unit_positions = {
            str(unit["unit_id"]): positions
            for unit, positions in zip(units, positions_by_unit)
        }
        normalized_sets = []
        for raw_gate_set in gate_sets:
            gate_set = tuple(sorted(set(str(unit_id) for unit_id in raw_gate_set)))
            unknown = [unit_id for unit_id in gate_set if unit_id not in unit_positions]
            if unknown:
                raise ValueError(f"unknown token gate units: {unknown[:3]}")
            if not gate_set:
                raise ValueError("token gate sets must be nonempty")
            normalized_sets.append(gate_set)

        sequence = prompt_ids + list(trajectory.response_ids)
        input_ids = torch.tensor([sequence], dtype=torch.long, device=self.device)
        base_embeddings = self.model.get_input_embeddings()(input_ids).detach()
        reference = trajectory.reference_log_probs.to(self.device)
        response_positions = (
            torch.arange(len(trajectory.response_ids), device=self.device)
            + len(prompt_ids) - 1
        )
        scores: list[float] = []
        gated_position_counts = []
        forward_batches = 0
        with torch.inference_mode():
            for start in range(0, len(normalized_sets), batch_size):
                batch_sets = normalized_sets[start:start + batch_size]
                embeddings = base_embeddings.expand(len(batch_sets), -1, -1).clone()
                for row, gate_set in enumerate(batch_sets):
                    gated_positions = sorted({
                        position
                        for unit_id in gate_set
                        for position in unit_positions[unit_id]
                    })
                    gated_position_counts.append(len(gated_positions))
                    if gated_positions:
                        embeddings[row, gated_positions] *= gate_value
                attention_mask = torch.ones(
                    embeddings.shape[:2],
                    dtype=torch.long,
                    device=self.device,
                )
                hidden = self.model.model(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state[:, response_positions]
                candidate = torch.log_softmax(self.model.lm_head(hidden).float(), dim=-1)
                reference_batch = reference.unsqueeze(0).expand(candidate.shape[0], -1, -1)
                log_middle = torch.logaddexp(reference_batch, candidate) - math.log(2.0)
                jsd = 0.5 * (
                    (reference_batch.exp() * (reference_batch - log_middle)).sum(dim=-1)
                    + (candidate.exp() * (candidate - log_middle)).sum(dim=-1)
                )
                scores.extend(jsd.sum(dim=-1).float().cpu().tolist())
                forward_batches += 1

        return scores, {
            "gate_sets": len(normalized_sets),
            "batch_size": batch_size,
            "gate_value": gate_value,
            "forward_batches": forward_batches,
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(trajectory.response_ids),
            "context_units": len(units),
            "mean_gated_model_subtokens": (
                sum(gated_position_counts) / len(gated_position_counts)
                if gated_position_counts else 0.0
            ),
            "max_gated_model_subtokens": max(gated_position_counts, default=0),
            "intervention": "joint_input_embedding_gate",
            "target": "teacher_forced_clean_response_distribution",
        }

    def _prompt_context_token_positions(
        self,
        question: str,
        contexts: Sequence[dict],
        units: Sequence[dict],
    ) -> tuple[list[int], list[list[int]]]:
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
            context_starts[str(context["chunk_id"])] = (
                marker_start + len(f"[{context.get('chunk_id', '')}] ")
            )
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
        return prompt_ids, positions_by_unit

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

    def generate_responses_batch(
        self,
        question: str,
        context_variants: Sequence[Sequence[dict]],
        *,
        max_new_tokens: int = 96,
    ) -> list[str]:
        """Greedily generate several context variants with left padding."""

        if not context_variants:
            return []
        torch = self.torch
        prompts = [self._prompt_ids(question, contexts) for contexts in context_variants]
        max_prompt_length = max(len(prompt) for prompt in prompts)
        input_ids = torch.full(
            (len(prompts), max_prompt_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, prompt in enumerate(prompts):
            start = max_prompt_length - len(prompt)
            input_ids[row, start:] = torch.tensor(prompt, dtype=torch.long, device=self.device)
            attention_mask[row, start:] = 1
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        responses = []
        for sequence in generated[:, max_prompt_length:].tolist():
            if self.tokenizer.eos_token_id in sequence:
                sequence = sequence[:sequence.index(self.tokenizer.eos_token_id)]
            responses.append(
                self.tokenizer.decode(sequence, skip_special_tokens=True).strip()
            )
        return responses

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
