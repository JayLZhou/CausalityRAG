from causalityrag.arc_jsd import (
    all_context_word_units,
    context_sentence_units,
    complete_pair_graph,
    fisher_pair_graph,
    lift_sentence_features_to_tokens,
    replace_context_units,
    vector_supermodular_graph,
)


def test_all_context_word_units_includes_stopwords_but_not_punctuation():
    record = {
        "retrieved": [{"chunk_id": "c1", "text": "Ada was here.", "rank": 1}],
    }
    units = all_context_word_units(record, k=1)
    assert [unit["text"] for unit in units] == ["Ada", "was", "here"]


def test_replace_context_units_applies_two_offset_stable_replacements():
    contexts = [{"chunk_id": "c1", "text": "Ada was here.", "rank": 1}]
    units = [
        {"unit_id": "a", "chunk_id": "c1", "text": "Ada", "chunk_char_start": 0, "chunk_char_end": 3},
        {"unit_id": "b", "chunk_id": "c1", "text": "here", "chunk_char_start": 8, "chunk_char_end": 12},
    ]
    edited = replace_context_units(
        contexts,
        units,
        {"a": {"new": "Eve"}, "b": {"new": "away"}},
    )
    assert edited[0]["text"] == "Eve was away."


def test_complete_pair_graph_keeps_only_positive_joint_synergy():
    units = [{"unit_id": "a"}, {"unit_id": "b"}, {"unit_id": "c"}]
    nodes, edges = complete_pair_graph(
        units,
        {"a": 1.0, "b": 2.0, "c": 0.5},
        {("a", "b"): 4.5, ("a", "c"): 1.0, ("b", "c"): 2.5},
    )
    assert nodes == {"a": 1.0, "b": 2.0, "c": 0.5}
    assert edges == {("a", "b"): 1.5}


def test_fisher_pair_graph_rescales_features_to_exact_singleton_jsd():
    units = [{"unit_id": "a"}, {"unit_id": "b"}, {"unit_id": "c"}]
    nodes, edges = fisher_pair_graph(
        units,
        [4.0, 9.0, 1.0],
        [[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]],
    )
    assert nodes == {"a": 4.0, "b": 9.0, "c": 1.0}
    assert abs(edges[("a", "b")] - 12.0) < 1e-6
    assert ("a", "c") not in edges
    assert ("b", "c") not in edges


def test_sentence_lift_conserves_each_sentence_vector():
    units = [
        {"unit_id": "a", "sentence_id": "s1"},
        {"unit_id": "b", "sentence_id": "s1"},
        {"unit_id": "c", "sentence_id": "s2"},
    ]
    sentences = [{"sentence_id": "s1"}, {"sentence_id": "s2"}]
    token_features, diagnostics = lift_sentence_features_to_tokens(
        units,
        sentences,
        [4.0, 9.0],
        [[1.0, 0.0], [0.0, 2.0]],
        [[3.0], [1.0], [2.0]],
        response_length=1,
    )
    assert diagnostics["max_conservation_error"] < 1e-6
    assert abs(sum(token_features[0][i] + token_features[1][i] for i in range(2)) - 2.0) < 1e-6
    assert abs(sum(value * value for value in token_features[2]) - 9.0) < 1e-6
    nodes, edges = vector_supermodular_graph(units, token_features)
    assert nodes["a"] > nodes["b"]
    assert edges[("a", "b")] > 0


def test_context_sentence_units_assigns_every_word_to_a_sentence():
    record = {
        "retrieved": [{"chunk_id": "c1", "text": "Ada stayed. Bob left!", "rank": 1}],
    }
    units, sentences = context_sentence_units(record, k=1)
    assert len(sentences) == 2
    assert len({unit["sentence_id"] for unit in units}) == 2


def test_contextual_ner_and_pos_annotation_avoid_pool_style_type_guesses():
    import spacy

    nlp = spacy.load("en_core_web_lg")
    record = {
        "retrieved": [{
            "chunk_id": "c1",
            "text": "Anthony Rapp plays Mark Cohen.",
            "rank": 1,
        }],
    }
    units, _ = context_sentence_units(record, k=1, nlp=nlp)
    by_text = {unit["text"]: unit for unit in units}
    assert by_text["Rapp"]["type"] == "PERSON"
    assert by_text["Rapp"]["entity_token_index"] == 1
    assert by_text["Rapp"]["entity_token_count"] == 2
    assert by_text["plays"]["type"] == "CONTENT"
    assert by_text["plays"]["pos"] == "VERB"
