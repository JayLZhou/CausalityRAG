"""Run RobustRAG secure decoding on frozen attacked top-10 contexts."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records
from causalityrag.reader import (
    READ_SYSTEM,
    READ_USER,
    answers_match,
    canonicalize_reader_answer,
    format_passages,
    parse_json_object,
)


ATTACKS = ("poisonedrag", "tparag")


class SecureDecoder:
    """Probability aggregation from RobustRAG for short-form QA (eta=0)."""

    def __init__(self, model_path: str, *, device: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(config, "base_model_tp_plan"):
            config.base_model_tp_plan = None
        if hasattr(config, "base_model_pp_plan"):
            config.base_model_pp_plan = None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).to(device)
        self.model.eval()
        self.eos_ids = {
            int(value)
            for value in (
                [self.tokenizer.eos_token_id]
                if isinstance(self.tokenizer.eos_token_id, int)
                else self.tokenizer.eos_token_id or []
            )
        }
        eot = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(eot, int) and eot >= 0:
            self.eos_ids.add(eot)

    def _prompt(self, question: str, context: dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": READ_SYSTEM},
            {
                "role": "user",
                "content": READ_USER.format(
                    question=question,
                    passages=format_passages([context]),
                ),
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def answer(self, question: str, contexts: list[dict[str, Any]]) -> str:
        if not contexts:
            return "I don't know"
        torch = self.torch
        prompts = [self._prompt(question, context) for context in contexts]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        generated: list[int] = []
        model_kwargs = {
            "attention_mask": attention_mask,
            "use_cache": True,
        }
        model_kwargs = self.model._get_initial_cache_position(
            input_ids.shape[1],
            input_ids.device,
            model_kwargs,
        )
        with torch.inference_mode():
            for _ in range(self.max_new_tokens):
                model_inputs = self.model.prepare_inputs_for_generation(
                    input_ids,
                    **model_kwargs,
                )
                outputs = self.model(
                    **model_inputs,
                    return_dict=True,
                )
                probabilities = torch.softmax(
                    outputs.logits[:, -1, :].float(),
                    dim=-1,
                )
                aggregate = probabilities.sum(dim=0)
                if not generated:
                    for surface in ("I", "Please", "/n"):
                        ids = self.tokenizer(
                            surface,
                            add_special_tokens=False,
                        )["input_ids"]
                        if len(ids) == 1:
                            aggregate[int(ids[0])] = 0.0
                token_id = int(torch.argmax(aggregate).item())
                if token_id in self.eos_ids:
                    break
                generated.append(token_id)
                appended = torch.full(
                    (input_ids.shape[0], 1),
                    token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                input_ids = torch.cat((input_ids, appended), dim=1)
                model_kwargs = self.model._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=False,
                )
        raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        parsed = parse_json_object(raw)
        answer = (
            str(parsed.get("answer", "")).strip()
            if isinstance(parsed, dict)
            else raw
        )
        return canonicalize_reader_answer(
            question,
            answer,
            reader_mode="short_answer",
        )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attacks = {}
    for name in ATTACKS:
        values = [int(row["attacks"][name]["correct"]) for row in rows]
        attacks[name] = {
            "attacked_accuracy": statistics.fmean(values),
            "correct": sum(values),
        }
    return {
        "queries": len(rows),
        "attacks": attacks,
        "macro_attacked_accuracy": statistics.fmean(
            int(row["attacks"][name]["correct"])
            for row in rows
            for name in ATTACKS
        ),
        "protocol": {
            "method": "RobustRAG secure probability aggregation",
            "eta": 0.0,
            "aggregation": "sum of isolated-passage next-token probabilities",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attacks", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    attacks = load_records(args.attacks)
    completed = (
        {str(row.get("id", "")): row for row in load_records(args.out)}
        if os.path.exists(args.out)
        else {}
    )
    pending = [row for row in attacks if str(row.get("id", "")) not in completed]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    decoder = SecureDecoder(
        args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    for source in pending:
        started = time.monotonic()
        identifier = str(source.get("id", ""))
        question = str(source.get("question", ""))
        gold = str(source.get("gold_answer", ""))
        result = {"id": identifier, "attacks": {}}
        for name in ATTACKS:
            contexts = [dict(row) for row in source["methods"][name]["attacked_contexts"]]
            answer = decoder.answer(question, contexts)
            result["attacks"][name] = {
                "answer": answer,
                "correct": answers_match(answer, gold),
                "isolated_passages": len(contexts),
            }
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        with open(args.out, "a", encoding="utf-8") as output:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
        completed[identifier] = result
        print(
            f"[robustrag-decoding] {len(completed)}/{len(attacks)} "
            f"seconds={result['elapsed_seconds']}",
            flush=True,
        )

    ordered = [completed[str(row.get("id", ""))] for row in attacks]
    with open(args.out, "w", encoding="utf-8") as output:
        for row in ordered:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = summarize(ordered)
    with open(args.summary_out, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
