from scripts.render_table3_mean_edited_tokens import render_tex


def test_render_tex_excludes_invalid_clean_queries_from_plotted_mean() -> None:
    payload = {
        "datasets": [
            {
                "label": "HQA",
                "mean_edited_tokens_all_1000": 2.5,
                "mean_edited_tokens_valid_clean": 2.75,
            },
            {
                "label": "PopQA",
                "mean_edited_tokens_all_1000": 4.0,
                "mean_edited_tokens_valid_clean": 4.1,
            },
        ],
        "unweighted_dataset_macro_mean_valid_clean": 3.425,
    }

    rendered = render_tex(payload)

    assert "\\TableThreeMeanEditedTokenCoordinates{(HQA,2.750) (PopQA,4.100)}" in rendered
    assert "\\TableThreeValidCleanMeanEditedTokenCoordinates{(HQA,2.750) (PopQA,4.100)}" in rendered
    assert "\\TableThreeMeanEditedTokenMacro{3.425}" in rendered
