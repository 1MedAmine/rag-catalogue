from __future__ import annotations

from rag_catalogue.pipeline import _catalogue_need
from rag_catalogue.provenance_need import merge_structured_provenance_into_need
from rag_catalogue.retrieval import CatalogueChunk, RankedChunk


def _item(champ: str, valeur: str, *, unite: str | None = None, criticalite: str = "secondaire") -> dict:
    return {
        "champ": champ,
        "valeur": valeur,
        "unite": unite,
        "valeur_complete": f"{valeur} {unite}".strip() if unite else valeur,
        "preuve": valeur,
        "source": "https://fabricant.example/fiche",
        "confiance": "elevee",
        "criticalite": criticalite,
    }


def test_merge_v4_maps_each_provenance_category_deterministically() -> None:
    need = {
        "product_type": "commutateur",
        "source_manufacturer": None,
        "source_reference": None,
        "attributes": [],
        "open_questions": [],
        "search_queries": ["commutateur manuel"],
    }
    provenance = {
        "schema_version": "4.0",
        "demande_client": "VendorE REF-DEMO-01 400 A",
        "type_demande": "mixte",
        "identification_produit": {
            "statut": "exacte",
            "reference": "REF-DEMO-01",
            "fabricant": "VendorE",
            "designation": "inverseur de sources manuel",
            "confiance": "elevee",
            "preuves": [],
        },
        "contraintes_explicites": [
            _item("reference_origine", "REF-DEMO-01", criticalite="identite"),
            _item("courant_nominal", "400", unite="A", criticalite="configuration"),
        ],
        "contraintes_derivees_obligatoires": [
            _item("fonction", "inverseur de sources manuel", criticalite="fonction"),
        ],
        "criteres_de_classement": [
            _item("commande", "frontale extérieure"),
        ],
        "capacites_optionnelles": [
            _item("indice_protection_poignee", "IP65"),
        ],
        "informations_a_confirmer": [
            _item("tension_utilisation", "tension exacte"),
        ],
    }

    result = merge_structured_provenance_into_need(need, provenance)

    by_name = {item["name"]: item for item in result["attributes"]}
    assert "reference_origine" not in by_name
    assert by_name["courant_nominal"]["role"] == "constraint"
    assert by_name["courant_nominal"]["provenance"] == "client"
    assert by_name["courant_nominal"]["constraint_scope"] == "hard"
    assert by_name["fonction"]["role"] == "constraint"
    assert by_name["fonction"]["provenance"] == "derived_required"
    assert by_name["fonction"]["constraint_scope"] == "direct_equivalence"
    assert by_name["commande"]["role"] == "selector"
    assert by_name["commande"]["provenance"] == "source_product"
    assert by_name["indice_protection_poignee"]["provenance"] == "optional"
    assert by_name["tension_utilisation"]["provenance"] == "to_confirm"
    assert "tension exacte" in result["open_questions"][0]
    assert result["source_reference"] == "REF-DEMO-01"
    assert result["source_manufacturer"] == "VendorE"
    assert result["product_type"] == "inverseur de sources manuel"


def test_structured_explicit_attribute_overrides_weaker_llm_duplicate() -> None:
    need = {
        "product_type": "appareil",
        "attributes": [{
            "name": "courant_nominal",
            "value": "400",
            "unit": "A",
            "role": "selector",
            "provenance": "source_product",
        }],
        "open_questions": [],
        "search_queries": ["appareil 400 A"],
    }
    provenance = {
        "demande_client": "appareil 400 A",
        "contraintes_explicites": [
            _item("courant_nominal", "400", unite="A", criticalite="configuration"),
        ],
    }

    result = merge_structured_provenance_into_need(need, provenance)

    matching = [item for item in result["attributes"] if item["name"] == "courant_nominal"]
    assert len(matching) == 1
    assert matching[0]["provenance"] == "client"
    assert matching[0]["constraint_scope"] == "hard"


def test_v4_structured_attribute_deduplicates_unique_llm_value_with_translated_name() -> None:
    need = {
        "product_type": "appareil",
        "attributes": [{
            "name": "rated current",
            "value": "400",
            "unit": "A",
            "role": "selector",
            "provenance": "source_product",
        }],
        "open_questions": [],
        "search_queries": ["appareil 400 A"],
    }
    provenance = {
        "schema_version": "4.0",
        "demande_client": "REF-EXEMPLE",
        "contraintes_derivees_obligatoires": [
            _item("courant_nominal", "400", unite="A", criticalite="configuration")
        ],
    }

    result = merge_structured_provenance_into_need(need, provenance)

    matching = [
        item
        for item in result["attributes"]
        if "".join(str(item.get("value", "")).split()) == "400"
        and item.get("unit") == "A"
    ]
    assert len(matching) == 1
    assert matching[0]["name"] == "courant_nominal"
    assert matching[0]["provenance"] == "derived_required"
    assert matching[0]["constraint_scope"] == "direct_equivalence"


def test_legacy_v31_provenance_remains_supported_without_derived_promotion() -> None:
    need = {
        "product_type": "disjoncteur",
        "attributes": [],
        "open_questions": [],
        "search_queries": ["disjoncteur 3 A"],
    }
    provenance = {
        "demande_client": "MCB 3 A",
        "contraintes_client": ["3A"],
        "caracteristiques_produit": ["10 kA"],
        "capacites_optionnelles": [],
        "informations_a_confirmer": [],
    }

    result = merge_structured_provenance_into_need(need, provenance)

    by_value = {item["value"]: item for item in result["attributes"]}
    assert by_value["3A"]["provenance"] == "client"
    assert by_value["10 kA"]["provenance"] == "source_product"
    assert not any(item["provenance"] == "derived_required" for item in result["attributes"])


def test_legacy_v31_merge_deduplicates_values_already_extracted_by_llm() -> None:
    need = {
        "product_type": "disjoncteur",
        "attributes": [
            {
                "name": "number of poles",
                "value": "2P",
                "unit": None,
                "role": "constraint",
                "provenance": "client",
            },
            {
                "name": "rated current",
                "value": "3",
                "unit": "A",
                "role": "constraint",
                "provenance": "client",
            },
            {
                "name": "voltage",
                "value": "125",
                "unit": "V DC",
                "role": "context",
                "provenance": "source_product",
            },
        ],
        "open_questions": ["AC or DC application"],
        "search_queries": ["disjoncteur 2P 3 A"],
    }
    provenance = {
        "demande_client": "MCB 2P 3A",
        "contraintes_client": ["2P", "3A"],
        "caracteristiques_produit": ["125 V DC"],
        "informations_a_confirmer": ["AC ou DC"],
    }

    result = merge_structured_provenance_into_need(need, provenance)

    assert len(result["attributes"]) == 3
    assert [item["name"] for item in result["attributes"]] == [
        "number of poles",
        "rated current",
        "voltage",
    ]
    assert result["open_questions"] == ["AC or DC application"]


def test_catalogue_need_preserves_derived_required_and_constraint_scope() -> None:
    need = {
        "product_type": "inverseur",
        "attributes": [{
            "name": "fonction",
            "value": "inverseur de sources manuel",
            "unit": None,
            "role": "constraint",
            "provenance": "derived_required",
            "constraint_scope": "direct_equivalence",
        }],
        "open_questions": [],
    }

    result = _catalogue_need(need, ["inverseur de sources manuel"])

    assert result["attributes"][0]["provenance"] == "derived_required"
    assert result["attributes"][0]["constraint_scope"] == "direct_equivalence"

from rag_catalogue.llm_client import NvidiaChatClient


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Completion:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Completion(self.responses.pop(0))


class _Chat:
    def __init__(self, responses: list[str]) -> None:
        self.completions = _Completions(responses)


class _Client:
    def __init__(self, responses: list[str]) -> None:
        self.chat = _Chat(responses)


def test_extract_need_preserves_derived_required_scope() -> None:
    fake = _Client([
        '{"product_type":"inverseur","source_manufacturer":"VendorE",'
        '"source_family":null,"source_reference":"REF-DEMO-01",'
        '"attributes":[{"name":"fonction","value":"inverseur de sources manuel",'
        '"unit":null,"role":"constraint","provenance":"derived_required",'
        '"constraint_scope":"direct_equivalence"}],'
        '"open_questions":[],"search_queries":["inverseur de sources manuel"]}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)

    result = client.extract_need("fiche")

    assert result["attributes"][0]["provenance"] == "derived_required"
    assert result["attributes"][0]["constraint_scope"] == "direct_equivalence"


def test_extract_need_preserves_structured_attribute_metadata() -> None:
    fake = _Client([
        '{"product_type":"inverseur","source_manufacturer":"VendorE",'
        '"source_family":null,"source_reference":"REF-DEMO-01",'
        '"attributes":[{"name":"courant_nominal","value":"400","unit":"A",'
        '"role":"constraint","provenance":"client","constraint_scope":"hard",'
        '"source_evidence":"400 A demandé","source_url":"cahier.json",'
        '"confidence":"high","criticality":"performance"}],'
        '"open_questions":[],"search_queries":["inverseur 400 A"]}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)

    result = client.extract_need("fiche")

    attribute = result["attributes"][0]
    assert attribute["constraint_scope"] == "hard"
    assert attribute["source_evidence"] == "400 A demandé"
    assert attribute["source_url"] == "cahier.json"
    assert attribute["confidence"] == "high"
    assert attribute["criticality"] == "performance"


def test_selection_prompt_distinguishes_hard_and_direct_equivalence_constraints() -> None:
    fake = _Client(['{"candidates":[],"explanation":"aucune preuve"}'])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)

    client.select_reference(
        {
            "product_type": "inverseur",
            "attributes": [{
                "name": "fonction",
                "value": "inverseur de sources manuel",
                "unit": None,
                "role": "constraint",
                "provenance": "derived_required",
                "constraint_scope": "direct_equivalence",
            }],
            "open_questions": [],
            "search_queries": ["inverseur de sources manuel"],
        },
        [
            RankedChunk(
                chunk=CatalogueChunk(
                    chunk_id="p1-c1",
                    page_number=1,
                    text="REF-1 inverseur de sources manuel",
                ),
                score=1.0,
                word_score=1.0,
                char_score=1.0,
                exact_score=1.0,
            )
        ],
    )

    system = fake.chat.completions.calls[0]["messages"][0]["content"]
    user = fake.chat.completions.calls[0]["messages"][1]["content"]
    combined = (system + "\n" + user).casefold()
    assert "derived_required" in combined
    assert "equivalent direct" in combined
    assert "alternative" in combined


def test_run_search_v4_merges_provenance_before_catalogue_selection(tmp_path) -> None:
    import json
    from pathlib import Path

    import fitz

    from rag_catalogue.pipeline import run_search

    cahier = Path(tmp_path) / "cahier_reference.txt"
    provenance_path = Path(tmp_path) / "provenance_reference.json"
    catalogue = Path(tmp_path) / "catalogue.pdf"
    cahier.write_text(
        "Cahier complet contenant les dimensions et conditions documentées.",
        encoding="utf-8",
    )
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": "4.0",
                "demande_client": "FABRICANT REF-EXEMPLE-400",
                "type_demande": "reference_exacte",
                "identification_produit": {
                    "statut": "exacte",
                    "reference": "REF-EXEMPLE-400",
                    "fabricant": "FABRICANT",
                    "designation": "commutateur manuel",
                    "confiance": "elevee",
                    "preuves": [],
                },
                "contraintes_explicites": [],
                "contraintes_derivees_obligatoires": [
                    _item("fonction", "commutateur manuel", criticalite="fonction")
                ],
                "criteres_de_classement": [
                    _item("commande", "frontale")
                ],
                "capacites_optionnelles": [],
                "informations_a_confirmer": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(
        (50, 50, 550, 780),
        "Reference TARGET-400. Commutateur manuel à commande frontale.",
        fontsize=10,
    )
    document.save(catalogue)
    document.close()

    class FakeLLM:
        model = "nvidia/fake"

        def __init__(self) -> None:
            self.selection_need = None

        def extract_need(self, _text: str) -> dict:
            return {
                "product_type": "commutateur manuel",
                "source_manufacturer": None,
                "source_reference": None,
                "attributes": [],
                "open_questions": [],
                "search_queries": ["commutateur manuel frontale"],
            }

        def select_reference(self, need: dict, _chunks: list) -> dict:
            self.selection_need = need
            return {
                "reference": "TARGET-400",
                "reference_mode": "explicit",
                "reference_parts": [],
                "catalogue_pages": [1],
                "evidence": ["Reference TARGET-400"],
                "explanation": "référence explicite",
            }

    llm = FakeLLM()
    result = run_search(
        cahier,
        catalogue,
        llm,
        provenance_file=provenance_path,
        top_k=1,
        validate_result=True,
        include_diagnostics=True,
    )

    assert result["reference"] == "TARGET-400"
    assert llm.selection_need is not None
    by_name = {item["name"]: item for item in llm.selection_need["attributes"]}
    assert by_name["fonction"]["provenance"] == "derived_required"
    assert by_name["fonction"]["constraint_scope"] == "direct_equivalence"
    assert by_name["commande"]["role"] == "selector"
    metadata = result["diagnostic"]["retrieval_metadata"]
    assert metadata["structured_provenance_version"] == "4.0"
    assert metadata["request_type"] == "reference_exacte"
