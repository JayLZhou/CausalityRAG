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


def test_arc_jsd_routing_is_not_reused_as_attention() -> None:
    repo = Path(__file__).resolve().parents[1]
    symbol = "_arc_jsd_response_to_context_routing"
    users = []
    for path in repo.rglob("*.py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        if symbol in path.read_text(encoding="utf-8"):
            users.append(path.relative_to(repo).as_posix())

    assert sorted(users) == ["exp/arc_jsd.py", "exp/run_arc_jsd.py"]
