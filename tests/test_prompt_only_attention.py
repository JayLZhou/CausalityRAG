from pathlib import Path

from scripts.run_table3_rankers import method_command


def command(method: str) -> list[str]:
    return method_command(
        method,
        repo=Path("/repo"),
        base=Path("/out/dataset"),
        reference=Path("/out/dataset/clean.jsonl"),
        output=Path("/out/result.jsonl"),
        summary=Path("/out/result.summary.json"),
        model_path="/models/reader",
        start=0,
        count=10,
    )


def test_attention_command_cannot_receive_clean_answer_reference() -> None:
    attention = command("attention")

    assert "--clean-reference" not in attention
    assert attention[1].endswith("exp/run_attention.py")


def test_answer_conditioned_baselines_still_receive_reference() -> None:
    for method in ("gradient_x_input", "integrated_gradients", "mirage", "arc_jsd"):
        assert "--clean-reference" in command(method)
