from rag_catalogue.pipeline import _finalise_candidate_collection


def _candidate(**updates):
    base = {
        "rank": 1,
        "reference": "TARGET-1",
        "variant": None,
        "match_level": "closest",
        "reference_mode": "explicit",
        "reference_parts": [],
        "catalogue_pages": [1],
        "evidence": ["TARGET-1"],
        "reason": "Documented candidate",
        "differences": [],
        "mandatory_differences": [],
        "constraint_assessment": [],
    }
    base.update(updates)
    return base


def test_rank_one_does_not_overwrite_conditional_alternative():
    result = _finalise_candidate_collection([
        _candidate(
            match_level="alternative",
            mandatory_differences=["Integrated enclosure instead of panel mounting"],
        )
    ])

    assert result[0]["match_level"] == "alternative_conditionnelle"
    assert result[0]["equivalence_status"] == "alternative_conditionnelle"


def test_function_mismatch_is_non_equivalent_even_when_ranked_first():
    result = _finalise_candidate_collection([
        _candidate(
            constraint_assessment=[{
                "champ": "fonction",
                "attendu": "manual source change-over switch I-0-II",
                "trouve": "switch-disconnector",
                "statut": "different",
                "criticalite": "fonction",
                "provenance": "derived_required",
            }],
        )
    ])

    assert result[0]["equivalence_status"] == "non_equivalent"
    assert result[0]["match_level"] == "non_equivalent"


def test_unproved_mandatory_constraint_is_non_verifiable():
    result = _finalise_candidate_collection([
        _candidate(
            constraint_assessment=[{
                "champ": "mode de commutation",
                "attendu": "I-0-II",
                "trouve": None,
                "statut": "non_prouve",
                "criticalite": "configuration",
                "provenance": "derived_required",
            }],
        )
    ])

    assert result[0]["equivalence_status"] == "non_verifiable"


def test_legacy_closest_without_mandatory_difference_remains_documented_proximity():
    result = _finalise_candidate_collection([_candidate()])

    assert result[0]["match_level"] == "closest"
    assert result[0]["equivalence_status"] == "proximite_documentee"
