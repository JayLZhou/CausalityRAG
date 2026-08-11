from scripts.render_table3_mean_edited_tokens import render_tex


def test_render_tex_excludes_invalid_clean_queries_from_plotted_mean() -> None:
    payload = {
        "datasets": [
            {
                "dataset": "hotpotqa",
                "label": "HQA",
                "mean_edited_tokens_all_1000": 2.5,
                "mean_edited_tokens_valid_clean": 2.75,
                "min_edited_tokens_valid_clean": 1,
                "q1_edited_tokens_valid_clean": 1.5,
                "median_edited_tokens_valid_clean": 2,
                "q3_edited_tokens_valid_clean": 3.5,
                "max_edited_tokens_valid_clean": 8,
            },
            {
                "dataset": "popqa",
                "label": "PopQA",
                "mean_edited_tokens_all_1000": 4.0,
                "mean_edited_tokens_valid_clean": 4.1,
                "min_edited_tokens_valid_clean": 1,
                "q1_edited_tokens_valid_clean": 2,
                "median_edited_tokens_valid_clean": 4,
                "q3_edited_tokens_valid_clean": 6,
                "max_edited_tokens_valid_clean": 12,
            },
        ],
        "unweighted_dataset_macro_mean_valid_clean": 3.425,
    }

    rendered = render_tex(payload)

    assert "\\TableThreeMeanEditedTokenCoordinates{(HQA,2.750) (PopQA,4.100)}" in rendered
    assert "\\TableThreeValidCleanMeanEditedTokenCoordinates{(HQA,2.750) (PopQA,4.100)}" in rendered
    assert "\\TableThreeMeanEditedTokenMacro{3.425}" in rendered
    assert "\\TableThreeHotpotQAMeanEditedTokens{2.750}" in rendered
    assert "\\TableThreePopQAMeanEditedTokens{4.100}" in rendered
    assert "\\TableThreeHotpotQATokenBoxplot{draw position=1" in rendered
    assert "lower quartile=1.500" in rendered
    assert "upper whisker=12.000" in rendered
