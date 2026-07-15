"""Target-specific token attribution DAGs for RAG answers.

The builders in this module are deliberately labelled by their attribution
mechanism. ``DirectActivationAttributionGraphBuilder`` uses the original
model's attention OV writes and MLP output writes; it does not claim to be the
transcoder graph from the KDD paper.
"""

from __future__ import annotations

from collections import defaultdict
import json
from dataclasses import dataclass
from typing import Any

from causalityrag.io import record_id, retrieved_contexts
from causalityrag.reader import READ_SYSTEM, READ_USER


@dataclass(frozen=True)
class TextSpan:
    region: str
    start: int
    end: int
    chunk_id: str = ""
    chunk_rank: int = 0


class AttentionAttributionGraphBuilder:
    """Build a sparse layer-token DAG targeted at a fixed answer."""

    method = "kdd_style_gradient_attention_approximation"

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_context_tokens: int = 800,
        max_length: int = 1024,
        edge_topk: int = 6,
        max_receivers_per_layer: int = 48,
        max_edges: int = 5000,
        residual_mix: float = 0.5,
        closed_flow: bool = False,
        absorbing_flow: bool = False,
    ) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_context_tokens = max_context_tokens
        self.max_length = max_length
        self.edge_topk = edge_topk
        self.max_receivers_per_layer = max_receivers_per_layer
        self.max_edges = max_edges
        self.residual_mix = residual_mix
        self.closed_flow = closed_flow
        self.absorbing_flow = absorbing_flow
        if self.closed_flow and self.absorbing_flow:
            raise ValueError("closed_flow and absorbing_flow are mutually exclusive")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # transformers 4.52 reads the Qwen tensor-parallel plan even for a
        # single-device load, while the installed torch build cannot use it.
        if hasattr(config, "base_model_tp_plan"):
            config.base_model_tp_plan = None
        config._attn_implementation = "eager"
        torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        self.model.requires_grad_(False)

    def build(self, record: dict, target_answer: str, *, k: int = 5, top_tokens: int = 50) -> dict:
        torch = self.torch
        contexts = self._truncate_contexts(retrieved_contexts(record)[:k])
        text, spans = self._render(record, contexts, target_answer)
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"][0].tolist()]
        token_meta = self._token_metadata(input_ids[0].tolist(), offsets, spans)
        answer_positions = [item["position"] for item in token_meta if item["region"] == "answer"]
        if not answer_positions:
            return self._empty(record, target_answer, "answer_tokens_truncated", len(token_meta))

        with torch.enable_grad():
            embeddings = self.model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
            output = self.model(inputs_embeds=embeddings, output_attentions=True, use_cache=False)
        if not output.attentions or any(attn is None for attn in output.attentions):
            return self._empty(record, target_answer, "model_returned_no_attentions", len(token_meta))

        target_logprob_tensor = self._target_logprob(output.logits, input_ids, answer_positions)
        gradients = torch.autograd.grad(target_logprob_tensor, output.attentions, allow_unused=True)
        target_logprob = float(target_logprob_tensor.detach().item())
        support, edges = self._rollout(output.attentions, gradients, answer_positions)
        support_cpu = support.float().cpu().tolist()
        context_tokens = []
        chunk_token_counts: dict[str, int] = {}
        for item, value in zip(token_meta, support_cpu):
            if item["region"] != "context":
                continue
            chunk_token_counts[item["chunk_id"]] = chunk_token_counts.get(item["chunk_id"], 0) + 1
            context_tokens.append({**item, "support": round(float(value), 10)})
        context_tokens.sort(key=lambda item: (-item["support"], item["position"]))

        kept_edges = sorted(edges, key=lambda edge: -edge["contribution"])[: self.max_edges]
        used_nodes = set()
        for edge in kept_edges:
            used_nodes.add((edge["src_layer"], edge["src_position"]))
            used_nodes.add((edge["dst_layer"], edge["dst_position"]))
        nodes = []
        for layer, position in sorted(used_nodes):
            meta = token_meta[position]
            nodes.append({
                "node_id": f"l{layer}:t{position}",
                "layer": layer,
                **meta,
            })

        return {
            "id": record_id(record),
            "question": str(record.get("question", "")),
            "gold_answer": str(record.get("answer", "")),
            "target_answer": target_answer,
            "status": "ok",
            "method": self.method,
            "target_logprob": round(target_logprob, 8),
            "top_context_tokens": context_tokens[:top_tokens],
            "context_token_supports": context_tokens,
            "chunk_token_counts": chunk_token_counts,
            "graph": {
                "sequence_tokens": len(token_meta),
                "context_tokens": len(context_tokens),
                "token_partitions": self._token_partitions(token_meta),
                "answer_positions": answer_positions,
                "region_edge_mass": self._region_edge_mass(kept_edges, token_meta),
                "layers": len(output.attentions),
                "nodes": nodes,
                "edges": kept_edges,
                "edge_count_before_cap": len(edges),
                "edge_weight_semantics": "positive target-logprob gradient times attention, row-normalized",
                "residual_mix": self.residual_mix,
            },
        }

    def _truncate_contexts(self, contexts: list[dict]) -> list[dict]:
        if not contexts:
            return []
        per_chunk = max(16, self.max_context_tokens // len(contexts))
        out = []
        for context in contexts:
            encoded = self.tokenizer(
                context["text"],
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=True,
                max_length=per_chunk,
            )
            offsets = encoded.get("offset_mapping", [])
            end = int(offsets[-1][1]) if offsets else 0
            out.append({**context, "text": context["text"][:end]})
        return out

    def _render(self, record: dict, contexts: list[dict], target_answer: str) -> tuple[str, list[TextSpan]]:
        passages = "\n\n".join(
            f"[{context['chunk_id']}] {context['text']}" for context in contexts
        )
        question = str(record.get("question", ""))
        user = READ_USER.format(passages=passages, question=question)
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": READ_SYSTEM},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        completion = json.dumps({"answer": target_answer}, ensure_ascii=False)
        text = prompt + completion

        spans: list[TextSpan] = []
        cursor = 0
        for context in contexts:
            start = prompt.find(context["text"], cursor)
            if start >= 0:
                end = start + len(context["text"])
                spans.append(TextSpan("context", start, end, context["chunk_id"], context["rank"]))
                cursor = end
        query_start = prompt.rfind(question)
        if query_start >= 0:
            spans.append(TextSpan("query", query_start, query_start + len(question)))
        answer_start = len(prompt) + completion.find(target_answer)
        spans.append(TextSpan("answer", answer_start, answer_start + len(target_answer)))
        return text, spans

    def _token_metadata(self, token_ids: list[int], offsets: list[tuple[int, int]], spans: list[TextSpan]) -> list[dict]:
        items = []
        for position, (token_id, (start, end)) in enumerate(zip(token_ids, offsets)):
            matched = next((span for span in spans if _overlaps(start, end, span.start, span.end)), None)
            chunk_start = max(0, start - matched.start) if matched and matched.region == "context" else -1
            chunk_end = min(matched.end - matched.start, end - matched.start) if matched and matched.region == "context" else -1
            items.append({
                "position": position,
                "token_id": token_id,
                "token": self.tokenizer.convert_ids_to_tokens(token_id),
                "text": self.tokenizer.decode([token_id]),
                "char_start": start,
                "char_end": end,
                "region": matched.region if matched else "prompt",
                "chunk_id": matched.chunk_id if matched else "",
                "chunk_rank": matched.chunk_rank if matched else 0,
                "chunk_char_start": chunk_start,
                "chunk_char_end": chunk_end,
            })
        return items

    @staticmethod
    def _token_partitions(token_meta: list[dict]) -> dict[str, list[int]]:
        """Keep prompt regions explicit even when pruning hides their layer nodes."""

        partitions: dict[str, list[int]] = {"query": [], "context": [], "answer": [], "prompt": []}
        for token in token_meta:
            region = str(token["region"])
            partitions.setdefault(region, []).append(int(token["position"]))
        return partitions

    @staticmethod
    def _region_edge_mass(edges: list[dict], token_meta: list[dict]) -> dict[str, float]:
        """Aggregate retained attribution edges by their source and target regions."""

        mass: dict[str, float] = {}
        for edge in edges:
            src_position = int(edge["src_position"])
            dst_position = int(edge["dst_position"])
            src_region = (
                str(token_meta[src_position]["region"])
                if 0 <= src_position < len(token_meta)
                else "background"
            )
            dst_region = (
                str(token_meta[dst_position]["region"])
                if 0 <= dst_position < len(token_meta)
                else "target"
            )
            key = f"{src_region}->{dst_region}"
            mass[key] = mass.get(key, 0.0) + float(edge.get("contribution", 0.0))
        return {key: round(value, 10) for key, value in sorted(mass.items())}

    def _target_logprob(self, logits: Any, input_ids: Any, answer_positions: list[int]) -> Any:
        torch = self.torch
        values = []
        for position in answer_positions:
            if position <= 0:
                continue
            log_probs = torch.log_softmax(logits[0, position - 1].float(), dim=-1)
            values.append(log_probs[input_ids[0, position]])
        if not values:
            return torch.tensor(float("nan"), device=logits.device, requires_grad=True)
        return torch.stack(values).mean()

    def _rollout(
        self,
        attentions: tuple[Any, ...],
        gradients: tuple[Any | None, ...],
        answer_positions: list[int],
    ) -> tuple[Any, list[dict]]:
        torch = self.torch
        sequence_length = attentions[0].shape[-1]
        score = torch.zeros(sequence_length, dtype=torch.float32, device=attentions[0].device)
        score[answer_positions] = 1.0 / len(answer_positions)
        edges = []

        for layer in range(len(attentions) - 1, -1, -1):
            raw_attention = attentions[layer][0].detach().float()
            gradient = gradients[layer]
            if gradient is None:
                attention = raw_attention.mean(dim=0)
            else:
                attention = (raw_attention * gradient[0].detach().float()).clamp_min(0).mean(dim=0)
                row_sums = attention.sum(dim=-1, keepdim=True)
                fallback = raw_attention.mean(dim=0)
                attention = torch.where(row_sums > 1e-12, attention / row_sums.clamp_min(1e-12), fallback)
            receiver_count = min(self.max_receivers_per_layer, int((score > 0).sum().item()))
            if receiver_count:
                receivers = torch.topk(score, receiver_count).indices.tolist()
                for receiver in receivers:
                    source_count = min(self.edge_topk, receiver + 1)
                    values, sources = torch.topk(attention[receiver, : receiver + 1], source_count)
                    for value, source in zip(values.tolist(), sources.tolist()):
                        contribution = (1.0 - self.residual_mix) * float(score[receiver]) * float(value)
                        if contribution <= 0:
                            continue
                        edges.append({
                            "src": f"l{layer}:t{source}",
                            "dst": f"l{layer + 1}:t{receiver}",
                            "src_layer": layer,
                            "dst_layer": layer + 1,
                            "src_position": source,
                            "dst_position": receiver,
                            "weight": round(float(value), 10),
                            "contribution": round(contribution, 10),
                            "kind": "attention",
                        })
                    residual_contribution = self.residual_mix * float(score[receiver])
                    if residual_contribution > 0:
                        edges.append({
                            "src": f"l{layer}:t{receiver}",
                            "dst": f"l{layer + 1}:t{receiver}",
                            "src_layer": layer,
                            "dst_layer": layer + 1,
                            "src_position": receiver,
                            "dst_position": receiver,
                            "weight": self.residual_mix,
                            "contribution": round(residual_contribution, 10),
                            "kind": "residual",
                        })
            propagated = torch.matmul(score.unsqueeze(0), attention).squeeze(0)
            score = self.residual_mix * score + (1.0 - self.residual_mix) * propagated
            score = score / score.sum().clamp_min(1e-12)
            del attention, propagated
        return score, edges

    def _empty(self, record: dict, target_answer: str, status: str, sequence_tokens: int = 0) -> dict:
        return {
            "id": record_id(record),
            "question": str(record.get("question", "")),
            "gold_answer": str(record.get("answer", "")),
            "target_answer": target_answer,
            "status": status,
            "method": self.method,
            "top_context_tokens": [],
            "context_token_supports": [],
            "chunk_token_counts": {},
            "graph": {"sequence_tokens": sequence_tokens, "nodes": [], "edges": []},
        }


def _overlaps(start: int, end: int, span_start: int, span_end: int) -> bool:
    return end > start and start < span_end and end > span_start


class _NativeTrace:
    """Forward-pass tensors needed for Qwen's local linear attribution."""

    def __init__(self) -> None:
        self.layer_inputs: dict[int, Any] = {}
        self.layer_outputs: dict[int, Any] = {}
        self.attn_inputs: dict[int, Any] = {}
        self.attn_outputs: dict[int, Any] = {}
        self.mlp_inputs: dict[int, Any] = {}
        self.mlp_outputs: dict[int, Any] = {}
        self.handles: list[Any] = []

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class NativeMLPAttributionGraphBuilder(AttentionAttributionGraphBuilder):
    """KDD-shaped graph using Qwen's native MLP channels as feature units.

    The KDD paper replaces MLPs with learned transcoders and traces a locally
    linear replacement model. Qwen's gated-MLP channels provide an exact local
    feature decomposition: ``SiLU(gate(x)) * up(x)`` decoded by ``down_proj``.
    We freeze attention weights and RMSNorm at their realized forward-pass
    values, then use target-logit gradients to obtain positive token-pair
    contributions. This is substantially closer to the KDD construction than
    attention rollout and needs no external transcoder checkpoints.
    """

    method = "qwen_native_mlp_fixed_forward_local_linear_attribution"

    native_feature_topk = 64

    def build(self, record: dict, target_answer: str, *, k: int = 5, top_tokens: int = 50) -> dict:
        torch = self.torch
        contexts = self._truncate_contexts(retrieved_contexts(record)[:k])
        text, spans = self._render(record, contexts, target_answer)
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"][0].tolist()]
        token_meta = self._token_metadata(input_ids[0].tolist(), offsets, spans)
        answer_positions = [item["position"] for item in token_meta if item["region"] == "answer"]
        if not answer_positions:
            return self._empty(record, target_answer, "answer_tokens_truncated", len(token_meta))

        trace = self._install_trace_hooks()
        try:
            with torch.enable_grad():
                embeddings = self.model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
                output = self.model(inputs_embeds=embeddings, output_attentions=True, use_cache=False)
                if not output.attentions or any(attention is None for attention in output.attentions):
                    return self._empty(record, target_answer, "model_returned_no_attentions", len(token_meta))
                target_logprob_tensor = self._target_logprob(output.logits, input_ids, answer_positions)
                target_logprob_tensor.backward()
            target_logprob = float(target_logprob_tensor.detach().item())
            support, edges = self._native_rollout(output.attentions, trace, answer_positions)
        finally:
            trace.close()

        support_cpu = support.float().cpu().tolist()
        context_tokens = []
        chunk_token_counts: dict[str, int] = {}
        for item, value in zip(token_meta, support_cpu):
            if item["region"] != "context":
                continue
            chunk_token_counts[item["chunk_id"]] = chunk_token_counts.get(item["chunk_id"], 0) + 1
            context_tokens.append({**item, "support": round(float(value), 10)})
        context_tokens.sort(key=lambda item: (-item["support"], item["position"]))

        kept_edges = sorted(edges, key=lambda edge: -edge["contribution"])[: self.max_edges]
        used_nodes = set()
        for edge in kept_edges:
            used_nodes.add((edge["src_layer"], edge["src_position"]))
            used_nodes.add((edge["dst_layer"], edge["dst_position"]))
        nodes = []
        for layer, position in sorted(used_nodes):
            nodes.append({"node_id": f"l{layer}:t{position}", "layer": layer, **token_meta[position]})

        return {
            "id": record_id(record),
            "question": str(record.get("question", "")),
            "gold_answer": str(record.get("answer", "")),
            "target_answer": target_answer,
            "status": "ok",
            "method": self.method,
            "target_logprob": round(target_logprob, 8),
            "top_context_tokens": context_tokens[:top_tokens],
            "context_token_supports": context_tokens,
            "chunk_token_counts": chunk_token_counts,
            "graph": {
                "sequence_tokens": len(token_meta),
                "context_tokens": len(context_tokens),
                "token_partitions": self._token_partitions(token_meta),
                "answer_positions": answer_positions,
                "region_edge_mass": self._region_edge_mass(kept_edges, token_meta),
                "layers": len(output.attentions),
                "nodes": nodes,
                "edges": kept_edges,
                "edge_count_before_cap": len(edges),
                "edge_weight_semantics": (
                    "positive target-logit contribution in a fixed-forward local linearization; "
                    "top native Qwen MLP channels are sparse feature units"
                ),
                "native_feature_topk": self.native_feature_topk,
                "residual_mix": None,
            },
        }

    def _install_trace_hooks(self) -> _NativeTrace:
        trace = _NativeTrace()

        def save_input(store: dict[int, Any], index: int):
            def hook(_module, inputs, kwargs):
                value = inputs[0] if inputs else kwargs.get("hidden_states")
                if value is None:
                    raise RuntimeError("could not capture transformer hidden states")
                store[index] = value
            return hook

        def save_output(store: dict[int, Any], index: int):
            def hook(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                if value.requires_grad:
                    value.retain_grad()
                store[index] = value
            return hook

        for index, layer in enumerate(self.model.model.layers):
            trace.handles.append(layer.register_forward_pre_hook(save_input(trace.layer_inputs, index), with_kwargs=True))
            trace.handles.append(layer.register_forward_hook(save_output(trace.layer_outputs, index)))
            trace.handles.append(layer.self_attn.register_forward_pre_hook(save_input(trace.attn_inputs, index), with_kwargs=True))
            trace.handles.append(layer.self_attn.register_forward_hook(save_output(trace.attn_outputs, index)))
            trace.handles.append(layer.mlp.register_forward_pre_hook(save_input(trace.mlp_inputs, index), with_kwargs=True))
            trace.handles.append(layer.mlp.register_forward_hook(save_output(trace.mlp_outputs, index)))
        return trace

    def _native_rollout(self, attentions: tuple[Any, ...], trace: _NativeTrace, answer_positions: list[int]) -> tuple[Any, list[dict]]:
        torch = self.torch
        sequence_length = attentions[0].shape[-1]
        score = torch.zeros(sequence_length, dtype=torch.float32, device=attentions[0].device)
        score[answer_positions] = 1.0 / len(answer_positions)
        edges = []

        for layer_index in range(len(attentions) - 1, -1, -1):
            matrix, attention_matrix, self_score = self._native_layer_contributions(
                layer_index, attentions[layer_index], trace
            )
            row_sums = matrix.sum(dim=-1, keepdim=True)
            identity = torch.eye(sequence_length, dtype=matrix.dtype, device=matrix.device)
            normalized = torch.where(
                row_sums > 1e-12,
                matrix / row_sums.clamp_min(1e-12),
                identity,
            )
            receiver_count = min(self.max_receivers_per_layer, int((score > 0).sum().item()))
            if receiver_count:
                receivers = torch.topk(score, receiver_count).indices.tolist()
                for receiver in receivers:
                    source_count = min(self.edge_topk, receiver + 1)
                    values, sources = torch.topk(normalized[receiver, : receiver + 1], source_count)
                    for value, source in zip(values.tolist(), sources.tolist()):
                        contribution = float(score[receiver]) * float(value)
                        if contribution <= 0:
                            continue
                        kind = "native_mlp_residual" if source == receiver else "fixed_attention"
                        edges.append({
                            "src": f"l{layer_index}:t{source}",
                            "dst": f"l{layer_index + 1}:t{receiver}",
                            "src_layer": layer_index,
                            "dst_layer": layer_index + 1,
                            "src_position": source,
                            "dst_position": receiver,
                            "weight": round(float(value), 10),
                            "contribution": round(contribution, 10),
                            "kind": kind,
                            "raw_attention_contribution": round(float(attention_matrix[receiver, source]), 10),
                            "raw_native_self_contribution": round(float(self_score[receiver]) if source == receiver else 0.0, 10),
                        })
            score = torch.matmul(score.unsqueeze(0), normalized).squeeze(0)
            score = score / score.sum().clamp_min(1e-12)
            del matrix, attention_matrix, self_score, normalized
        return score, edges

    def _native_layer_contributions(self, index: int, attention: Any, trace: _NativeTrace) -> tuple[Any, Any, Any]:
        """Compute target-logit edge contributions for one frozen-forward layer."""

        torch = self.torch
        layer = self.model.model.layers[index]
        attn_module = layer.self_attn
        attn_input = trace.attn_inputs[index][0].detach()
        attn_output_grad = trace.attn_outputs[index].grad[0].detach().float()
        attn_weights = attention[0].detach().float()
        n_heads = attn_weights.shape[0]
        head_dim = attn_module.head_dim
        n_kv_heads = n_heads // attn_module.num_key_value_groups

        with torch.no_grad():
            values = attn_module.v_proj(attn_input).view(1, -1, n_kv_heads, head_dim)[0].float()
            if n_kv_heads != n_heads:
                values = values.repeat_interleave(n_heads // n_kv_heads, dim=1)
            output_grads = torch.matmul(attn_output_grad, attn_module.o_proj.weight.detach().float())
            output_grads = output_grads.view(-1, n_heads, head_dim)
            dot_products = torch.einsum("thd,shd->hts", output_grads, values)
            attention_matrix = (dot_products * attn_weights).clamp_min(0).sum(dim=0)

            mlp_input = trace.mlp_inputs[index][0].detach()
            gate = layer.mlp.gate_proj(mlp_input).float()
            up = layer.mlp.up_proj(mlp_input).float()
            features = layer.mlp.act_fn(gate) * up
            mlp_grad = trace.mlp_outputs[index].grad[0].detach().float()
            decoder_grad = torch.matmul(mlp_grad, layer.mlp.down_proj.weight.detach().float())
            feature_contribution = (features * decoder_grad).clamp_min(0)
            sparse_contribution = torch.topk(
                feature_contribution,
                k=min(self.native_feature_topk, feature_contribution.shape[-1]),
                dim=-1,
            ).values
            self_score = sparse_contribution.sum(dim=-1)
            matrix = attention_matrix + torch.diag(self_score)
        return matrix, attention_matrix, self_score


class DirectActivationAttributionGraphBuilder(NativeMLPAttributionGraphBuilder):
    """Build a target-logit graph from Qwen's actual component writes.

    Attention edges are source-specific OV writes contracted with the target
    gradient. MLP and residual edges use the actual vectors added to the
    residual stream. The resulting signed weights are local first-order
    contributions in the original model, with no learned feature surrogate.
    """

    method = "qwen_direct_activation_target_logit_attribution"

    def build(self, record: dict, target_answer: str, *, k: int = 5, top_tokens: int = 50) -> dict:
        torch = self.torch
        contexts = self._truncate_contexts(retrieved_contexts(record)[:k])
        text, spans = self._render(record, contexts, target_answer)
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"][0].tolist()]
        token_meta = self._token_metadata(input_ids[0].tolist(), offsets, spans)
        answer_positions = [item["position"] for item in token_meta if item["region"] == "answer"]
        target_positions = [position for position in answer_positions if position > 0]
        if not target_positions:
            return self._empty(record, target_answer, "answer_tokens_truncated", len(token_meta))

        trace = self._install_trace_hooks()
        try:
            with torch.enable_grad():
                embeddings = self.model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
                output = self.model(inputs_embeds=embeddings, output_attentions=True, use_cache=False)
                if not output.attentions or any(attention is None for attention in output.attentions):
                    return self._empty(record, target_answer, "model_returned_no_attentions", len(token_meta))
                target_logits = [
                    output.logits[0, position - 1, input_ids[0, position]].float()
                    for position in target_positions
                ]
                target_logit_tensor = torch.stack(target_logits).mean()
                target_logit_tensor.backward()

            if self.closed_flow or self.absorbing_flow:
                edges, flow_diagnostics = self._closed_flow_edges(
                    output.attentions,
                    trace,
                    token_meta,
                    target_positions,
                )
                if self.absorbing_flow:
                    edges, flow_diagnostics = self._absorbing_flow_subgraph(
                        edges,
                        flow_diagnostics,
                    )
                context_tokens, chunk_token_counts = self._closed_context_support(
                    embeddings,
                    token_meta,
                    edges,
                )
            else:
                edges = self._direct_edges(
                    output.attentions,
                    trace,
                    token_meta,
                    target_positions,
                )
                flow_diagnostics = {}
                context_tokens, chunk_token_counts = self._direct_context_support(
                    embeddings,
                    token_meta,
                    edges,
                    target_positions,
                )
        finally:
            trace.close()

        kept_edges = (
            edges
            if self.closed_flow or self.absorbing_flow
            else self._prune_direct_edges(edges, self.max_edges)
        )
        nodes = self._direct_nodes(kept_edges, token_meta, len(output.attentions), target_positions)
        target_outputs = [
            {
                "answer_position": position,
                "predictor_position": position - 1,
                "token_id": int(input_ids[0, position]),
                "token": token_meta[position]["token"],
                "text": token_meta[position]["text"],
                "logit": round(float(value.detach()), 8),
            }
            for position, value in zip(target_positions, target_logits)
        ]

        return {
            "id": record_id(record),
            "question": str(record.get("question", "")),
            "gold_answer": str(record.get("answer", "")),
            "target_answer": target_answer,
            "status": "ok",
            "method": (
                f"{self.method}_closed_flow"
                if self.closed_flow
                else (
                    f"{self.method}_absorbing_flow"
                    if self.absorbing_flow
                    else self.method
                )
            ),
            "target_logit": round(float(target_logit_tensor.detach()), 8),
            "target_outputs": target_outputs,
            "top_context_tokens": context_tokens[:top_tokens],
            "context_token_supports": context_tokens,
            "chunk_token_counts": chunk_token_counts,
            "graph": {
                "sequence_tokens": len(token_meta),
                "context_tokens": len(context_tokens),
                "token_partitions": self._token_partitions(token_meta),
                "answer_positions": answer_positions,
                "target_positions": target_positions,
                "region_edge_mass": self._region_edge_mass(kept_edges, token_meta),
                "layers": len(output.attentions),
                "stages": 2 * len(output.attentions) + 2,
                "nodes": nodes,
                "edges": kept_edges,
                "edge_count_before_cap": len(edges),
                "edge_weight_semantics": (
                    (
                        "backward-conserved positive flow allocated in proportion to signed "
                        "local target-logit contribution from the actual residual, attention "
                        "OV, or MLP output write"
                    )
                    if self.closed_flow or self.absorbing_flow
                    else (
                        "signed local target-logit contribution: target gradient dotted with "
                        "the actual residual, attention OV, or MLP output write"
                    )
                ),
                "support_capacity_field": "contribution",
                "signed_weight_field": (
                    "raw_signed_contribution"
                    if self.closed_flow or self.absorbing_flow
                    else "signed_contribution"
                ),
                "negative_weight_field": "negative_contribution",
                "target_objective": "mean raw logit over clean-answer tokens",
                "context_support_semantics": (
                    (
                        "closed conserved flow terminating at context input tokens"
                        if self.closed_flow
                        else "retained absorbing flow terminating at context input tokens"
                    )
                    if self.closed_flow or self.absorbing_flow
                    else (
                        "sum of positive context-to-answer-predictor attention OV contributions"
                    )
                ),
                "flow_construction": (
                    "closed_backward_beam_with_explicit_background"
                    if self.closed_flow
                    else (
                        "absorbing_backward_beam_without_background_edges"
                        if self.absorbing_flow
                        else "none"
                    )
                ),
                "flow_diagnostics": flow_diagnostics,
                "uses_transcoder": False,
            },
        }

    def _closed_flow_edges(
        self,
        attentions: tuple[Any, ...],
        trace: _NativeTrace,
        token_meta: list[dict],
        target_positions: list[int],
    ) -> tuple[list[dict], dict]:
        """Trace a sparse conserved positive flow backward from the answer."""

        edges: list[dict] = []
        final_layer = len(attentions) - 1
        final_stage = 2 * len(attentions)
        final_output = trace.layer_outputs[final_layer][0].detach().float()
        final_grad = trace.layer_outputs[final_layer].grad[0].detach().float()
        final_contribution = (final_grad * final_output).sum(dim=-1)
        predictors = sorted({position - 1 for position in target_positions})
        positive_outputs = {
            predictor: max(0.0, float(final_contribution[predictor]))
            for predictor in predictors
        }
        output_total = sum(positive_outputs.values())
        active_post: dict[int, float] = defaultdict(float)
        sink_position = target_positions[-1]
        if output_total <= 1e-15:
            edge = self._closed_background_edge(
                final_stage + 1,
                sink_position,
                1.0,
                final_layer,
                "answer_output",
            )
            edge["dst"] = "answer_target"
            edge["target_positions"] = list(target_positions)
            edges.append(edge)
        else:
            for predictor, raw_value in positive_outputs.items():
                if raw_value <= 0:
                    continue
                flow = raw_value / output_total
                edge = self._closed_flow_edge(
                    final_stage,
                    predictor,
                    final_stage + 1,
                    sink_position,
                    "answer_logit",
                    raw_value,
                    flow,
                    final_layer,
                )
                edge["dst"] = "answer_target"
                edge["target_positions"] = list(target_positions)
                edges.append(edge)
                active_post[predictor] += flow

        for layer_index in range(len(attentions) - 1, -1, -1):
            if not active_post:
                break
            pre_stage = 2 * layer_index
            mid_stage = pre_stage + 1
            post_stage = pre_stage + 2

            mlp_output = trace.mlp_outputs[layer_index][0].detach().float()
            mlp_grad = trace.mlp_outputs[layer_index].grad[0].detach().float()
            mlp_write = (mlp_grad * mlp_output).sum(dim=-1)
            mid_residual = (
                trace.layer_outputs[layer_index][0].detach().float() - mlp_output
            )
            mlp_residual = (mlp_grad * mid_residual).sum(dim=-1)
            active_mid: dict[int, float] = defaultdict(float)
            for receiver, mass in active_post.items():
                values = (
                    ("mlp_residual", max(0.0, float(mlp_residual[receiver]))),
                    ("mlp_output_write", max(0.0, float(mlp_write[receiver]))),
                )
                total = sum(value for _, value in values)
                if total <= 1e-15:
                    edges.append(self._closed_background_edge(
                        post_stage,
                        receiver,
                        mass,
                        layer_index,
                        "mlp",
                    ))
                    continue
                for kind, raw_value in values:
                    if raw_value <= 0:
                        continue
                    flow = mass * raw_value / total
                    edges.append(self._closed_flow_edge(
                        mid_stage,
                        receiver,
                        post_stage,
                        receiver,
                        kind,
                        raw_value,
                        flow,
                        layer_index,
                    ))
                    active_mid[receiver] += flow

            if not active_mid:
                active_post = {}
                continue
            attention_matrix = self._direct_attention_matrix(
                layer_index,
                attentions[layer_index],
                trace.attn_outputs[layer_index].grad,
                trace.attn_inputs[layer_index],
            )
            layer_input = trace.layer_inputs[layer_index][0].detach().float()
            attention_grad = trace.attn_outputs[layer_index].grad[0].detach().float()
            attention_residual = (attention_grad * layer_input).sum(dim=-1)

            allocations: list[dict] = []
            dropped_by_receiver: dict[int, float] = defaultdict(float)
            source_mass: dict[int, float] = defaultdict(float)
            for receiver, mass in active_mid.items():
                positive_row = attention_matrix[receiver, : receiver + 1].clamp_min(0)
                residual_value = max(0.0, float(attention_residual[receiver]))
                total = residual_value + float(positive_row.sum())
                if total <= 1e-15:
                    dropped_by_receiver[receiver] += mass
                    continue

                retained_mass = 0.0
                source_count = min(max(0, self.edge_topk), positive_row.numel())
                if source_count:
                    values, sources = self.torch.topk(positive_row, source_count)
                    for raw_value, source in zip(values.tolist(), sources.tolist()):
                        if raw_value <= 0:
                            continue
                        flow = mass * float(raw_value) / total
                        allocations.append({
                            "receiver": receiver,
                            "source": int(source),
                            "kind": "attention_ov_write",
                            "raw_value": float(raw_value),
                            "flow": flow,
                        })
                        source_mass[int(source)] += flow
                        retained_mass += flow
                if residual_value > 0:
                    flow = mass * residual_value / total
                    allocations.append({
                        "receiver": receiver,
                        "source": receiver,
                        "kind": "attention_residual",
                        "raw_value": residual_value,
                        "flow": flow,
                    })
                    source_mass[receiver] += flow
                    retained_mass += flow
                dropped_by_receiver[receiver] += max(0.0, mass - retained_mass)

            selected_sources = {
                source
                for source, _ in sorted(
                    source_mass.items(),
                    key=lambda item: (-item[1], item[0]),
                )[: max(0, self.max_receivers_per_layer)]
            }
            active_pre: dict[int, float] = defaultdict(float)
            for allocation in allocations:
                receiver = int(allocation["receiver"])
                source = int(allocation["source"])
                flow = float(allocation["flow"])
                if source not in selected_sources:
                    dropped_by_receiver[receiver] += flow
                    continue
                edges.append(self._closed_flow_edge(
                    pre_stage,
                    source,
                    mid_stage,
                    receiver,
                    str(allocation["kind"]),
                    float(allocation["raw_value"]),
                    flow,
                    layer_index,
                ))
                active_pre[source] += flow
            for receiver, flow in dropped_by_receiver.items():
                if flow <= 1e-15:
                    continue
                edges.append(self._closed_background_edge(
                    mid_stage,
                    receiver,
                    flow,
                    layer_index,
                    "attention_beam",
                ))
            active_post = dict(active_pre)

        diagnostics = self._closed_flow_diagnostics(edges, token_meta)
        diagnostics.update({
            "edge_topk": self.edge_topk,
            "receiver_beam": self.max_receivers_per_layer,
            "flow_edges": len(edges),
        })
        return edges, diagnostics

    @staticmethod
    def _closed_flow_edge(
        src_stage: int,
        src_position: int,
        dst_stage: int,
        dst_position: int,
        kind: str,
        raw_value: float,
        flow: float,
        model_layer: int,
    ) -> dict:
        return {
            "src": f"s{src_stage}:t{src_position}",
            "dst": f"s{dst_stage}:t{dst_position}",
            "src_layer": src_stage,
            "dst_layer": dst_stage,
            "src_position": src_position,
            "dst_position": dst_position,
            "model_layer": model_layer,
            "kind": kind,
            "weight": round(raw_value, 10),
            "raw_signed_contribution": round(raw_value, 10),
            "signed_contribution": round(raw_value, 10),
            "contribution": round(max(0.0, flow), 12),
            "negative_contribution": 0.0,
            "relevance": round(max(0.0, flow), 12),
        }

    @staticmethod
    def _closed_background_edge(
        dst_stage: int,
        dst_position: int,
        flow: float,
        model_layer: int,
        reason: str,
    ) -> dict:
        return {
            "src": f"background::{reason}:s{dst_stage}:t{dst_position}",
            "dst": f"s{dst_stage}:t{dst_position}",
            "src_layer": 0,
            "dst_layer": dst_stage,
            "src_position": -1,
            "dst_position": dst_position,
            "model_layer": model_layer,
            "kind": f"background_{reason}",
            "weight": 0.0,
            "raw_signed_contribution": 0.0,
            "signed_contribution": 0.0,
            "contribution": round(max(0.0, flow), 12),
            "negative_contribution": 0.0,
            "relevance": round(max(0.0, flow), 12),
        }

    @staticmethod
    def _absorbing_flow_subgraph(
        edges: list[dict],
        closed_diagnostics: dict,
    ) -> tuple[list[dict], dict]:
        """Drop synthetic background routes and record their mass as absorbed."""

        retained = [
            edge
            for edge in edges
            if not str(edge.get("src", "")).startswith("background::")
        ]
        removed = [
            edge
            for edge in edges
            if str(edge.get("src", "")).startswith("background::")
        ]
        absorbed_mass = sum(
            float(edge.get("contribution", 0.0)) for edge in removed
        )
        return retained, {
            **closed_diagnostics,
            "proposal_background_flow": closed_diagnostics.get(
                "background_flow",
                absorbed_mass,
            ),
            "absorbed_background_mass": absorbed_mass,
            "removed_background_edges": len(removed),
            "retained_flow_edges": len(retained),
            "retained_background_flow": 0.0,
        }

    @staticmethod
    def _closed_flow_diagnostics(
        edges: list[dict],
        token_meta: list[dict],
    ) -> dict:
        incoming: dict[str, float] = defaultdict(float)
        outgoing: dict[str, float] = defaultdict(float)
        source_positions: dict[str, int] = {}
        for edge in edges:
            flow = float(edge.get("contribution", 0.0))
            if flow <= 0:
                continue
            src = str(edge["src"])
            dst = str(edge["dst"])
            outgoing[src] += flow
            incoming[dst] += flow
            source_positions[src] = int(edge.get("src_position", -1))
        internal = set(incoming) & set(outgoing)
        source_nodes = set(outgoing) - set(incoming)
        source_total = sum(outgoing[node] for node in source_nodes)
        background_flow = sum(
            outgoing[node] for node in source_nodes if node.startswith("background::")
        )
        region_flow: dict[str, float] = defaultdict(float)
        for node in source_nodes:
            if node.startswith("background::"):
                continue
            position = source_positions.get(node, -1)
            region = (
                str(token_meta[position]["region"])
                if 0 <= position < len(token_meta)
                else "unknown"
            )
            region_flow[region] += outgoing[node]
        return {
            "sink_inflow": incoming.get("answer_target", 0.0),
            "source_total_flow": source_total,
            "background_flow": background_flow,
            "model_input_flow": source_total - background_flow,
            "input_region_flow": {
                key: round(value, 12) for key, value in sorted(region_flow.items())
            },
            "maximum_internal_conservation_error": max(
                (abs(incoming[node] - outgoing[node]) for node in internal),
                default=0.0,
            ),
            "source_nodes": len(source_nodes),
            "internal_nodes": len(internal),
        }

    def _closed_context_support(
        self,
        embeddings: Any,
        token_meta: list[dict],
        edges: list[dict],
    ) -> tuple[list[dict], dict[str, int]]:
        if embeddings.grad is None:
            raise RuntimeError("input embedding gradients were not retained")
        embedding_signed = (
            embeddings.grad[0].detach().float() * embeddings[0].detach().float()
        ).sum(dim=-1)
        input_flow: dict[int, float] = defaultdict(float)
        for edge in edges:
            position = int(edge.get("src_position", -1))
            if int(edge.get("src_layer", -1)) == 0 and position >= 0:
                input_flow[position] += float(edge.get("contribution", 0.0))
        context_tokens = []
        chunk_token_counts: dict[str, int] = {}
        for item, embedding_value in zip(token_meta, embedding_signed.tolist()):
            if item["region"] != "context":
                continue
            position = int(item["position"])
            flow = input_flow.get(position, 0.0)
            chunk_id = str(item["chunk_id"])
            chunk_token_counts[chunk_id] = chunk_token_counts.get(chunk_id, 0) + 1
            context_tokens.append({
                **item,
                "support": round(flow, 12),
                "signed_support": round(flow, 12),
                "negative_support": 0.0,
                "relevance": round(flow, 12),
                "embedding_support": round(max(0.0, float(embedding_value)), 10),
                "signed_embedding_support": round(float(embedding_value), 10),
            })
        context_tokens.sort(key=lambda item: (-float(item["support"]), item["position"]))
        return context_tokens, chunk_token_counts

    def _direct_context_support(
        self,
        embeddings: Any,
        token_meta: list[dict],
        edges: list[dict],
        target_positions: list[int],
    ) -> tuple[list[dict], dict[str, int]]:
        if embeddings.grad is None:
            raise RuntimeError("input embedding gradients were not retained")
        embedding_signed = (
            embeddings.grad[0].detach().float() * embeddings[0].detach().float()
        ).sum(dim=-1)
        outgoing_positive: dict[int, float] = {}
        outgoing_negative: dict[int, float] = {}
        predictor_positions = {position - 1 for position in target_positions}
        for edge in edges:
            if (
                edge["kind"] != "attention_ov_write"
                or int(edge["dst_position"]) not in predictor_positions
            ):
                continue
            position = int(edge["src_position"])
            outgoing_positive[position] = outgoing_positive.get(position, 0.0) + float(
                edge["contribution"]
            )
            outgoing_negative[position] = outgoing_negative.get(position, 0.0) + float(
                edge["negative_contribution"]
            )
        context_tokens = []
        chunk_token_counts: dict[str, int] = {}
        for item, embedding_value in zip(token_meta, embedding_signed.tolist()):
            if item["region"] != "context":
                continue
            position = int(item["position"])
            positive = outgoing_positive.get(position, 0.0)
            negative = outgoing_negative.get(position, 0.0)
            chunk_id = str(item["chunk_id"])
            chunk_token_counts[chunk_id] = chunk_token_counts.get(chunk_id, 0) + 1
            context_tokens.append({
                **item,
                "support": round(positive, 10),
                "signed_support": round(positive - negative, 10),
                "negative_support": round(negative, 10),
                "relevance": round(positive + negative, 10),
                "embedding_support": round(max(0.0, float(embedding_value)), 10),
                "signed_embedding_support": round(float(embedding_value), 10),
            })
        context_tokens.sort(key=lambda item: (-float(item["support"]), item["position"]))
        return context_tokens, chunk_token_counts

    @staticmethod
    def _prune_direct_edges(edges: list[dict], max_edges: int) -> list[dict]:
        """Keep the answer sink connected while globally pruning by relevance."""

        mandatory = [edge for edge in edges if edge["kind"] == "answer_logit"]
        optional = [edge for edge in edges if edge["kind"] != "answer_logit"]
        optional.sort(key=lambda edge: -float(edge["relevance"]))
        if max_edges <= 0:
            return mandatory
        return mandatory + optional[: max(0, max_edges - len(mandatory))]

    def _direct_edges(
        self,
        attentions: tuple[Any, ...],
        trace: _NativeTrace,
        token_meta: list[dict],
        target_positions: list[int],
    ) -> list[dict]:
        edges = []
        for layer_index, attention in enumerate(attentions):
            attention_matrix = self._direct_attention_matrix(
                layer_index,
                attention,
                trace.attn_outputs[layer_index].grad,
                trace.attn_inputs[layer_index],
            )
            layer_input = trace.layer_inputs[layer_index][0].detach().float()
            attention_grad = trace.attn_outputs[layer_index].grad[0].detach().float()
            attention_residual = (attention_grad * layer_input).sum(dim=-1)

            mlp_output = trace.mlp_outputs[layer_index][0].detach().float()
            mlp_grad = trace.mlp_outputs[layer_index].grad[0].detach().float()
            mlp_write = (mlp_grad * mlp_output).sum(dim=-1)
            mid_residual = (
                trace.layer_outputs[layer_index][0].detach().float() - mlp_output
            )
            mlp_residual = (mlp_grad * mid_residual).sum(dim=-1)

            receiver_relevance = (
                attention_matrix.abs().sum(dim=-1)
                + attention_residual.abs()
                + mlp_write.abs()
                + mlp_residual.abs()
            )
            receiver_count = min(self.max_receivers_per_layer, len(token_meta))
            receivers = self.torch.topk(receiver_relevance, receiver_count).indices.tolist()
            pre_stage = 2 * layer_index
            mid_stage = pre_stage + 1
            post_stage = pre_stage + 2
            for receiver in receivers:
                source_values = attention_matrix[receiver, : receiver + 1]
                source_count = min(self.edge_topk, source_values.numel())
                if source_count:
                    _, sources = self.torch.topk(source_values.abs(), source_count)
                    for source in sources.tolist():
                        value = float(source_values[source])
                        if abs(value) <= 1e-12:
                            continue
                        edges.append(self._direct_edge(
                            pre_stage,
                            source,
                            mid_stage,
                            receiver,
                            "attention_ov_write",
                            value,
                            layer_index,
                        ))
                for kind, value in (
                    ("attention_residual", float(attention_residual[receiver])),
                    ("mlp_residual", float(mlp_residual[receiver])),
                    ("mlp_output_write", float(mlp_write[receiver])),
                ):
                    if abs(value) <= 1e-12:
                        continue
                    src_stage = pre_stage if kind == "attention_residual" else mid_stage
                    dst_stage = mid_stage if kind == "attention_residual" else post_stage
                    edges.append(self._direct_edge(
                        src_stage,
                        receiver,
                        dst_stage,
                        receiver,
                        kind,
                        value,
                        layer_index,
                    ))

        final_layer = len(attentions) - 1
        final_stage = 2 * len(attentions)
        final_output = trace.layer_outputs[final_layer][0].detach().float()
        final_grad = trace.layer_outputs[final_layer].grad[0].detach().float()
        final_contribution = (final_grad * final_output).sum(dim=-1)
        sink_position = target_positions[-1]
        for predictor in sorted({position - 1 for position in target_positions}):
            value = float(final_contribution[predictor])
            if abs(value) <= 1e-12:
                continue
            edge = self._direct_edge(
                final_stage,
                predictor,
                final_stage + 1,
                sink_position,
                "answer_logit",
                value,
                final_layer,
            )
            edge["dst"] = "answer_target"
            edge["target_positions"] = list(target_positions)
            edges.append(edge)
        return edges

    def _direct_attention_matrix(
        self,
        layer_index: int,
        attention: Any,
        attention_output_grad: Any,
        attention_input: Any,
    ) -> Any:
        torch = self.torch
        layer = self.model.model.layers[layer_index]
        module = layer.self_attn
        attn_weights = attention[0].detach().float()
        n_heads = attn_weights.shape[0]
        head_dim = module.head_dim
        n_kv_heads = n_heads // module.num_key_value_groups
        with torch.no_grad():
            values = module.v_proj(attention_input[0].detach()).view(
                -1, n_kv_heads, head_dim
            ).float()
            if n_kv_heads != n_heads:
                values = values.repeat_interleave(n_heads // n_kv_heads, dim=1)
            output_grads = torch.matmul(
                attention_output_grad[0].detach().float(),
                module.o_proj.weight.detach().float(),
            ).view(-1, n_heads, head_dim)
            dot_products = torch.einsum("thd,shd->hts", output_grads, values)
            return (dot_products * attn_weights).sum(dim=0)

    @staticmethod
    def _direct_edge(
        src_stage: int,
        src_position: int,
        dst_stage: int,
        dst_position: int,
        kind: str,
        value: float,
        model_layer: int,
    ) -> dict:
        return {
            "src": f"s{src_stage}:t{src_position}",
            "dst": f"s{dst_stage}:t{dst_position}",
            "src_layer": src_stage,
            "dst_layer": dst_stage,
            "src_position": src_position,
            "dst_position": dst_position,
            "model_layer": model_layer,
            "kind": kind,
            "weight": round(value, 10),
            "signed_contribution": round(value, 10),
            "contribution": round(max(0.0, value), 10),
            "negative_contribution": round(max(0.0, -value), 10),
            "relevance": round(abs(value), 10),
        }

    @staticmethod
    def _direct_nodes(
        edges: list[dict],
        token_meta: list[dict],
        layer_count: int,
        target_positions: list[int],
    ) -> list[dict]:
        sink_stage = 2 * layer_count + 1
        used = set()
        background_nodes: dict[str, dict] = {}
        has_sink = False
        for edge in edges:
            src_position = int(edge["src_position"])
            if src_position >= 0:
                used.add((int(edge["src_layer"]), src_position))
            else:
                background_nodes[str(edge["src"])] = {
                    "node_id": str(edge["src"]),
                    "stage": int(edge["src_layer"]),
                    "stage_name": "background",
                    "model_layer": int(edge.get("model_layer", -1)),
                    "position": -1,
                    "region": "background",
                    "text": "<background>",
                }
            if edge["dst"] == "answer_target":
                has_sink = True
            else:
                dst_position = int(edge["dst_position"])
                if dst_position >= 0:
                    used.add((int(edge["dst_layer"]), dst_position))
        nodes = []
        for stage, position in sorted(used):
            if stage % 2 == 0:
                stage_name = "residual"
                model_layer = stage // 2
            else:
                stage_name = "post_attention"
                model_layer = stage // 2
            nodes.append({
                "node_id": f"s{stage}:t{position}",
                "stage": stage,
                "stage_name": stage_name,
                "model_layer": model_layer,
                **token_meta[position],
            })
        nodes.extend(background_nodes[node] for node in sorted(background_nodes))
        if has_sink:
            nodes.append({
                "node_id": "answer_target",
                "stage": sink_stage,
                "stage_name": "answer_logit",
                "model_layer": layer_count,
                "position": target_positions[-1],
                "region": "answer",
                "target_positions": list(target_positions),
                "text": "<answer-target>",
            })
        return nodes
