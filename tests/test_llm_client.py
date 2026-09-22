from types import SimpleNamespace

import pytest

from rag_catalogue.llm_client import LLMResponseError, NvidiaChatClient, extract_json_object
from rag_catalogue.pdf_text import CatalogueChunk
from rag_catalogue.retrieval import RankedChunk


def test_extract_json_object_accepts_plain_json() -> None:
    assert extract_json_object('{"reference":"ABC-123"}') == {"reference": "ABC-123"}


def test_extract_json_object_accepts_fenced_json_with_extra_text() -> None:
    text = 'Résultat:\n```json\n{"reference": null, "catalogue_pages": []}\n```\nFin.'
    assert extract_json_object(text) == {"reference": None, "catalogue_pages": []}


def test_extract_json_object_rejects_malformed_response() -> None:
    with pytest.raises(LLMResponseError, match="JSON"):
        extract_json_object("aucun objet ici")


class _FakeCompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response_item = self.responses.pop(0)
        if isinstance(response_item, tuple):
            content, finish_reason = response_item
        else:
            content, finish_reason = response_item, None
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )]
        )


class _FakeOpenAIClient:
    def __init__(self, responses: list[object]) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


def test_client_extracts_generic_need_and_selects_one_reference() -> None:
    fake = _FakeOpenAIClient([
        '{"product_type":"deep groove ball bearing",'
        '"source_reference":"SOURCE-20X52",'
        '"attributes":[{"name":"bore","value":"20","unit":"mm"},'
        '{"name":"outside diameter","value":"52","unit":"mm"},'
        '{"name":"width","value":"15","unit":"mm"},'
        '{"name":"seal","value":"double rubber","unit":null}],'
        '"search_queries":["deep groove ball bearing 20 mm 52 mm 15 mm double rubber seals"]}',
        '{"reference":"6304-2RS","catalogue_pages":[7],'
        '"evidence":["Bore 20 mm; outside diameter 52 mm; width 15 mm; double seals"],'
        '"explanation":"The catalogue line states the requested dimensions and sealing."}',
    ])
    client = NvidiaChatClient(
        api_key="test",
        model="nvidia/test-model",
        client=fake,
    )
    chunk = RankedChunk(
        chunk=CatalogueChunk(
            chunk_id="p7-c1",
            page_number=7,
            text="Reference 6304-2RS. Bore 20 mm. OD 52 mm. Width 15 mm. Double rubber seals.",
        ),
        score=0.9,
        word_score=0.9,
        char_score=0.9,
        exact_score=0.9,
    )

    need = client.extract_need(
        "Deep groove ball bearing, bore 20 mm, outside diameter 52 mm, "
        "width 15 mm, double rubber seals."
    )
    result = client.select_reference(need, [chunk])

    assert need["search_queries"] == [
        "deep groove ball bearing 20 mm 52 mm 15 mm double rubber seals"
    ]
    assert result["reference"] == "6304-2RS"
    assert result["catalogue_pages"] == [7]
    assert all(call["temperature"] == 0 for call in fake.chat.completions.calls)


def test_nemotron_super_uses_vendor_recommended_sampling_defaults() -> None:
    fake = _FakeOpenAIClient([
        '{"product_type":"bearing","source_reference":null,'
        '"attributes":[{"name":"bore","value":"20","unit":"mm"}],'
        '"search_queries":["bearing bore 20 mm"]}'
    ])
    client = NvidiaChatClient(
        api_key="test",
        model="nvidia/nemotron-3-super-120b-a12b",
        client=fake,
    )

    client.extract_need("Bearing bore 20 mm")

    call = fake.chat.completions.calls[0]
    assert call["temperature"] == 1.0
    assert call["top_p"] == 0.95


def test_extract_need_accepts_attribute_mapping_returned_by_model() -> None:
    fake = _FakeOpenAIClient([
        '{"product_type":"miniature circuit breaker",'
        '"source_reference":"A9F77206",'
        '"attributes":{"number of poles":"2","rated current":"6 A",'
        '"curve":"C","breaking capacity":"10 kA"},'
        '"search_queries":["miniature circuit breaker 2 poles 6 A curve C 10 kA"]}'
    ])
    client = NvidiaChatClient(
        api_key="test",
        model="nvidia/nemotron-3-super-120b-a12b",
        client=fake,
    )

    need = client.extract_need("MCB 2P 6A curve C 10kA")

    assert need["attributes"] == [
        {"name": "number of poles", "value": "2", "unit": None},
        {"name": "rated current", "value": "6 A", "unit": None},
        {"name": "curve", "value": "C", "unit": None},
        {"name": "breaking capacity", "value": "10 kA", "unit": None},
    ]


def test_extract_need_accepts_french_aliases_and_keeps_source_excerpt() -> None:
    fake = _FakeOpenAIClient([
        '{"type_produit":"disjoncteur miniature",'
        '"reference_source":"A9F77206",'
        '"caracteristiques":[{"nom":"courant nominal","valeur":"6","unite":"A"}],'
        '"requetes_recherche":["disjoncteur miniature 2 poles 6 A courbe C 10 kA"]}'
    ])
    client = NvidiaChatClient(
        api_key="test",
        model="nvidia/nemotron-3-super-120b-a12b",
        client=fake,
    )

    need = client.extract_need("Fiche complète du produit Schneider 2P 6A")

    assert need["product_type"] == "disjoncteur miniature"
    assert need["source_reference"] == "A9F77206"
    assert need["attributes"] == [
        {"name": "courant nominal", "value": "6", "unit": "A"}
    ]
    assert need["search_queries"] == [
        "disjoncteur miniature 2 poles 6 A courbe C 10 kA"
    ]
    assert "Fiche complète" in need["source_excerpt"]


def test_extract_need_does_not_fail_when_model_omits_attributes_but_has_queries() -> None:
    fake = _FakeOpenAIClient([
        '{"product_type":"miniature circuit breaker",'
        '"source_reference":"A9F77206",'
        '"search_queries":["miniature circuit breaker 2 poles 6 A curve C 10 kA"]}'
    ])
    client = NvidiaChatClient(
        api_key="test",
        model="nvidia/nemotron-3-super-120b-a12b",
        client=fake,
    )

    need = client.extract_need("MCB 2P 6A curve C 10 kA")

    assert need["attributes"] == []
    assert need["search_queries"] == [
        "miniature circuit breaker 2 poles 6 A curve C 10 kA"
    ]
    assert need["source_excerpt"] == "MCB 2P 6A curve C 10 kA"


def test_nemotron_super_uses_normal_reasoning_for_structured_openai_call() -> None:
    fake = _FakeOpenAIClient([
        '{"product_type":"bearing","source_reference":null,'
        '"attributes":[],"search_queries":["bearing 20 mm"]}'
    ])
    client = NvidiaChatClient(
        api_key="test",
        model="nvidia/nemotron-3-super-120b-a12b",
        client=fake,
    )

    client.extract_need("Bearing 20 mm")

    call = fake.chat.completions.calls[0]
    assert call["extra_body"] == {
        "chat_template_kwargs": {
            "enable_thinking": True,
            "low_effort": True,
        },
        "reasoning_budget": 2048,
    }


class _FakeHTTPResponse:
    def __init__(self, body: dict) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


class _FakeHTTPSession:
    def __init__(self, body: dict) -> None:
        self.body = body
        self.calls: list[dict] = []

    def post(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return _FakeHTTPResponse(self.body)


def test_nemotron_super_uses_normal_reasoning_for_raw_http_call() -> None:
    body = {
        "choices": [{
            "message": {
                "content": (
                    '{"product_type":"bearing","source_reference":null,'
                    '"attributes":[],"search_queries":["bearing 20 mm"]}'
                )
            }
        }]
    }
    client = NvidiaChatClient(
        api_key="test",
        model="nvidia/nemotron-3-super-120b-a12b",
    )
    session = _FakeHTTPSession(body)
    client._session = session

    client.extract_need("Bearing 20 mm")

    payload = session.calls[0]["json"]
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": True,
        "low_effort": True,
    }
    assert payload["reasoning_budget"] == 2048


def test_extract_need_normalises_source_manufacturer_and_family() -> None:
    fake = _FakeOpenAIClient([
        '{"product_type":"miniature circuit breaker",'
        '"source_manufacturer":"Schneider Electric",'
        '"source_family":"Acti9 iC60N",'
        '"source_reference":"A9F77206",'
        '"attributes":[],"search_queries":["miniature circuit breaker 2P 6A"]}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)

    need = client.extract_need("Schneider Acti9 A9F77206")

    assert need["source_manufacturer"] == "Schneider Electric"
    assert need["source_family"] == "Acti9 iC60N"


def test_extract_need_preserves_attribute_roles_and_requests_role_schema() -> None:
    fake = _FakeOpenAIClient([
        '{"product_type":"miniature circuit breaker",'
        '"source_reference":"A9F77206",'
        '"attributes":['
        '{"name":"rated current","value":"6","unit":"A","role":"selector"},'
        '{"name":"installation depth","value":"44.5","unit":"mm","role":"context"}],'
        '"search_queries":["miniature circuit breaker 2 poles 6 A C curve 10 kA"]}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)

    need = client.extract_need("MCB 2P 6A C 10kA, depth 44.5 mm")

    assert need["attributes"] == [
        {"name": "rated current", "value": "6", "unit": "A", "role": "selector"},
        {"name": "installation depth", "value": "44.5", "unit": "mm", "role": "context"},
    ]
    prompt = fake.chat.completions.calls[0]["messages"][1]["content"].casefold()
    assert '"role"' in prompt
    assert "selector" in prompt
    assert "context" in prompt


def test_selection_prompt_does_not_make_unclassified_attributes_automatically_mandatory() -> None:
    fake = _FakeOpenAIClient([
        '{"reference":null,"catalogue_pages":[],"evidence":[],"explanation":"not enough evidence"}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)
    chunk = RankedChunk(
        chunk=CatalogueChunk(chunk_id="p1-c1", page_number=1, text="Reference X-1."),
        score=0.5,
        word_score=0.5,
        char_score=0.5,
        exact_score=0.5,
    )

    client.select_reference(
        {
            "product_type": "industrial product",
            "attributes": [{"name": "depth", "value": "44.5", "unit": "mm"}],
            "search_queries": ["industrial product"],
        },
        [chunk],
    )

    call = fake.chat.completions.calls[0]
    combined_prompt = " ".join(
        message["content"] for message in call["messages"]
    ).casefold()
    assert "role absent" in combined_prompt or "sans role" in combined_prompt
    assert "pas automatiquement obligatoire" in combined_prompt


def test_extract_need_prompt_requests_minimal_cross_manufacturer_selectors() -> None:
    fake = _FakeOpenAIClient([
        '{"product_type":"miniature circuit breaker","source_reference":"SRC-1",'
        '"attributes":[{"name":"rated current","value":"6","unit":"A","role":"selector"}],'
        '"search_queries":["miniature circuit breaker 2P 6A C 10kA"]}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)

    client.extract_need("Technical product sheet")

    prompt = fake.chat.completions.calls[0]["messages"][1]["content"].casefold()
    assert "inter-fabricants" in prompt or "cross-manufacturer" in prompt
    assert "au plus 12" in prompt
    assert "valeurs mesurees selon plusieurs normes" in prompt


def test_selection_prompt_requests_closest_corresponding_variant_across_catalogue_conventions() -> None:
    fake = _FakeOpenAIClient([
        '{"reference":"TARGET-6A","catalogue_pages":[1],'
        '"evidence":["TARGET-6A 2P 6A C 10kA"],'
        '"explanation":"closest corresponding variant"}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)
    chunk = RankedChunk(
        chunk=CatalogueChunk(
            chunk_id="p1-c1",
            page_number=1,
            text="TARGET-6A 2P 6A C curve 10 kA AC 240/415 V",
        ),
        score=0.8,
        word_score=0.8,
        char_score=0.8,
        exact_score=0.8,
    )

    client.select_reference(
        {
            "product_type": "miniature circuit breaker",
            "attributes": [
                {"name": "rated current", "value": "6", "unit": "A", "role": "selector"},
                {"name": "number of poles", "value": "2", "unit": None, "role": "selector"},
                {"name": "voltage", "value": "230/400", "unit": "V", "role": "selector"},
            ],
            "search_queries": ["miniature circuit breaker 2P 6A C 10kA"],
        },
        [chunk],
    )

    combined_prompt = " ".join(
        message["content"] for message in fake.chat.completions.calls[0]["messages"]
    ).casefold()
    assert "variante correspondante la plus proche" in combined_prompt
    assert "notation" in combined_prompt
    assert "norme" in combined_prompt


def test_selection_normalises_assembled_alias_and_value_parts() -> None:
    fake = _FakeOpenAIClient([
        '{"reference":"ABC2PC00006","reference_mode":"assembled",'
        '"reference_parts":['
        '{"value":"ABC","page":7},'
        '{"value":"2P","page":7},'
        '{"value":"C","page":7},'
        '{"value":"00006","page":7}],'
        '"catalogue_pages":[7],"evidence":["Ordering Information"],'
        '"explanation":"assembled from the ordering key"}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)
    chunk = RankedChunk(
        chunk=CatalogueChunk(
            chunk_id="p7-c1",
            page_number=7,
            text="Ordering Information. ABC 2P C 00006.",
        ),
        score=0.8,
        word_score=0.8,
        char_score=0.8,
        exact_score=0.8,
    )

    result = client.select_reference(
        {
            "product_type": "industrial protector",
            "attributes": [],
            "search_queries": ["industrial protector 2P 6 A"],
        },
        [chunk],
    )

    assert result["reference_mode"] == "constructed"
    assert result["reference_parts"] == [
        {"position": 1, "code": "ABC", "page": 7, "evidence": ""},
        {"position": 2, "code": "2P", "page": 7, "evidence": ""},
        {"position": 3, "code": "C", "page": 7, "evidence": ""},
        {"position": 4, "code": "00006", "page": 7, "evidence": ""},
    ]

    prompt = fake.chat.completions.calls[0]["messages"][1]["content"]
    assert '"reference_mode": "explicit" ou "constructed" ou null' in prompt
    assert '"position": 1, "code": "segment exact"' in prompt


def test_selection_client_accepts_constructed_reference_mode_used_by_validator() -> None:
    fake = _FakeOpenAIClient([
        '{"reference":"AXR63P2PC00006","reference_mode":"constructed",'
        '"reference_parts":['
        '{"position":1,"code":"AXR","page":2},'
        '{"position":2,"code":"63P","page":2},'
        '{"position":3,"code":"2P","page":2},'
        '{"position":4,"code":"C","page":2},'
        '{"position":5,"code":"00006","page":2}],'
        '"catalogue_pages":[2],"evidence":["Ordering Information"],'
        '"explanation":"assembled from the catalogue key"}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)
    chunk = RankedChunk(
        chunk=CatalogueChunk(
            chunk_id="p2-c1",
            page_number=2,
            text="Ordering Information AXR 63P 2P C 00006",
        ),
        score=1.0,
        word_score=1.0,
        char_score=1.0,
        exact_score=1.0,
    )

    result = client.select_reference(
        {
            "product_type": "industrial circuit protector",
            "attributes": [],
            "search_queries": ["industrial circuit protector"],
        },
        [chunk],
    )

    assert result["reference_mode"] == "constructed"
    assert result["reference_parts"][0]["code"] == "AXR"
    prompt = fake.chat.completions.calls[0]["messages"][1]["content"]
    assert '"reference_mode": "explicit" ou "constructed"' in prompt


def test_selection_prompt_prefers_base_variant_and_same_product_function() -> None:
    fake = _FakeOpenAIClient([
        '{"reference":null,"reference_mode":null,"reference_parts":[],'
        '"catalogue_pages":[],"evidence":[],"explanation":"not enough proof"}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)
    chunks = [
        RankedChunk(
            chunk=CatalogueChunk(
                chunk_id="p1-c1",
                page_number=1,
                text="PAGE CONTEXT: Standard product. Reference BASE-1.",
            ),
            score=1.0,
            word_score=1.0,
            char_score=1.0,
            exact_score=1.0,
        )
    ]

    client.select_reference(
        {
            "product_type": "industrial protection device",
            "attributes": [],
            "search_queries": ["industrial protection device"],
        },
        chunks,
    )

    prompt = " ".join(
        message["content"] for message in fake.chat.completions.calls[0]["messages"]
    ).casefold()
    assert "variante de base" in prompt or "version standard" in prompt
    assert "fonction technique majeure" in prompt
    assert "page context" in prompt


def test_extract_need_prompt_requests_international_catalogue_query_and_common_acronym() -> None:
    fake = _FakeOpenAIClient([
        '{"product_type":"industrial product","source_reference":null,'
        '"attributes":[],"search_queries":["industrial product"]}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)

    client.extract_need("Fiche technique en français")

    prompt = fake.chat.completions.calls[0]["messages"][1]["content"].casefold()
    assert "anglais technique" in prompt or "technical english" in prompt
    assert "abreviation" in prompt or "acronyme" in prompt
    assert "internationale" in prompt or "international" in prompt


def test_selection_prompt_requires_all_declared_order_template_slots() -> None:
    fake = _FakeOpenAIClient([
        '{"reference":null,"reference_mode":null,"reference_parts":[],'
        '"catalogue_pages":[],"evidence":[],"explanation":"not enough proof"}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)
    chunk = RankedChunk(
        chunk=CatalogueChunk(
            chunk_id="p2-table-order",
            page_number=2,
            text=(
                "Ordering Information\nSTRUCTURED ORDER TEMPLATE\n"
                "SLOT COUNT: 4\nEXAMPLE CODES: AXR | 1P | C | 00001"
            ),
        ),
        score=1.0,
        word_score=1.0,
        char_score=1.0,
        exact_score=1.0,
    )

    client.select_reference(
        {"product_type": "protector", "attributes": [], "search_queries": ["protector"]},
        [chunk],
    )

    prompt = fake.chat.completions.calls[0]["messages"][1]["content"]
    assert "SLOT COUNT=N" in prompt
    assert "exactement les N segments" in prompt
    assert "prefixe, la gamme ou le modele de base" in prompt


def test_repair_reference_includes_validation_feedback_and_previous_result() -> None:
    fake = _FakeOpenAIClient([
        '{"reference":"AXR2PMC00006","reference_mode":"constructed",'
        '"reference_parts":['
        '{"position":1,"code":"AXR","page":2},'
        '{"position":2,"code":"2P","page":2},'
        '{"position":3,"code":"MC","page":2},'
        '{"position":4,"code":"00006","page":2}],'
        '"catalogue_pages":[2],"evidence":["Ordering Information"],'
        '"explanation":"complete reference"}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)
    chunk = RankedChunk(
        chunk=CatalogueChunk(
            chunk_id="p2-table-order",
            page_number=2,
            text=(
                "Ordering Information\nSTRUCTURED ORDER TEMPLATE\n"
                "SLOT COUNT: 4\nEXAMPLE CODES: AXR | 1P | C | 00001\n"
                "AXR: product\n2P: two poles\nMC: C curve\n00006: 6 A"
            ),
        ),
        score=1.0,
        word_score=1.0,
        char_score=1.0,
        exact_score=1.0,
    )

    result = client.repair_reference(
        {"product_type": "protector", "attributes": [], "search_queries": ["protector"]},
        [chunk],
        {
            "reference": "AXR",
            "reference_mode": "constructed",
            "reference_parts": [{"position": 1, "code": "AXR", "page": 2}],
            "catalogue_pages": [2],
            "evidence": ["AXR"],
            "explanation": "prefix",
        },
        ["incomplete_constructed_reference"],
    )

    assert result["reference"] == "AXR2PMC00006"
    prompt = fake.chat.completions.calls[0]["messages"][1]["content"]
    assert "CORRECTION APRES VALIDATION DETERMINISTE" in prompt
    assert "incomplete_constructed_reference" in prompt
    assert '"reference": "AXR"' in prompt


def test_select_reference_normalises_numeric_page_strings_and_integer_floats() -> None:
    fake = _FakeOpenAIClient([
        '{"reference":"REF-1","reference_mode":"explicit","reference_parts":[],'
        '"catalogue_pages":["7",8.0],"evidence":["Reference REF-1"],'
        '"explanation":"explicit"}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)
    chunks = [
        RankedChunk(
            chunk=CatalogueChunk("p7-c1", 7, "Reference REF-1"),
            score=1.0,
            word_score=1.0,
            char_score=1.0,
            exact_score=1.0,
        )
    ]

    result = client.select_reference(
        {"product_type": "component", "attributes": [], "search_queries": ["component"]},
        chunks,
    )

    assert result["catalogue_pages"] == [7, 8]


def test_http_error_reports_generation_model_endpoint_and_provider_detail() -> None:
    import requests

    class Response:
        status_code = 404
        text = '{"message":"model not found"}'

        def raise_for_status(self) -> None:
            error = requests.HTTPError("404 error")
            error.response = self
            raise error

        def json(self):
            return {"message": "model not found"}

    class Session:
        def post(self, *_args, **_kwargs):
            return Response()

    client = NvidiaChatClient(api_key="test", model="wrong/model")
    client._session = Session()

    with pytest.raises(LLMResponseError) as captured:
        client.extract_need("industrial component")

    message = str(captured.value)
    assert "wrong/model" in message
    assert "/chat/completions" in message
    assert "model not found" in message


def test_extract_need_preserves_attribute_provenance_and_open_questions() -> None:
    fake = _FakeOpenAIClient([
        '{"product_type":"miniature circuit breaker MCB",'
        '"source_reference":"M9F11203",'
        '"attributes":['
        '{"name":"number of poles","value":"2P","unit":null,"role":"constraint","provenance":"client"},'
        '{"name":"rated voltage","value":"125","unit":"V DC","role":"context","provenance":"source_product"},'
        '{"name":"application","value":"AC or DC","unit":null,"role":"context","provenance":"to_confirm"}],'
        '"open_questions":["application AC or DC"],'
        '"search_queries":["miniature circuit breaker MCB 2P","miniature circuit breaker MCB 2P DC"]}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test-model", client=fake)

    need = client.extract_need("full cahier and structured provenance")

    assert need["attributes"][0]["provenance"] == "client"
    assert need["attributes"][1]["provenance"] == "source_product"
    assert need["attributes"][2]["provenance"] == "to_confirm"
    assert need["open_questions"] == ["application AC or DC"]


def test_extract_need_prompt_explains_that_source_product_attributes_can_select_equivalent() -> None:
    fake = _FakeOpenAIClient([
        '{"product_type":"pump","source_reference":null,"attributes":[],"open_questions":[],"search_queries":["pump"]}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test-model", client=fake)

    client.extract_need("technical cahier")

    user_prompt = fake.chat.completions.calls[0]["messages"][1]["content"]
    assert "source_product" in user_prompt
    assert "necessaire a l'equivalence" in user_prompt.casefold()
    assert "json de provenance" in user_prompt.casefold()


def test_selection_prompt_uses_provenance_without_turning_documentation_into_client_requirement() -> None:
    fake = _FakeOpenAIClient([
        '{"reference":"TARGET","reference_mode":"explicit","reference_parts":[],"catalogue_pages":[1],"evidence":["TARGET"],"explanation":"match"}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test-model", client=fake)
    chunk = RankedChunk(
        chunk=CatalogueChunk(chunk_id="p1-c1", page_number=1, text="Reference TARGET"),
        score=1.0,
        word_score=1.0,
        char_score=1.0,
        exact_score=1.0,
    )

    client.select_reference(
        {
            "product_type": "industrial product",
            "attributes": [
                {"name": "client criterion", "value": "A", "unit": None, "role": "constraint", "provenance": "client"},
                {"name": "source criterion", "value": "B", "unit": None, "role": "selector", "provenance": "source_product"},
                {"name": "ambiguity", "value": "C or D", "unit": None, "role": "context", "provenance": "to_confirm"},
            ],
            "open_questions": ["C or D"],
            "search_queries": ["industrial product A B"],
        },
        [chunk],
    )

    prompt = " ".join(
        message["content"] for message in fake.chat.completions.calls[0]["messages"]
    ).casefold()
    assert "provenance=client" in prompt
    assert "provenance=source_product" in prompt
    assert "provenance=to_confirm" in prompt
    assert "ne doit pas" in prompt and "obligation" in prompt


def test_select_reference_accepts_ranked_candidate_list_and_keeps_best_at_top() -> None:
    fake = _FakeOpenAIClient([
        '{"candidates":['
        '{"rank":2,"reference":"REF-DEMO-032PMCS0000C00003","variant":"Standard",'
        '"match_level":"alternative","reference_mode":"constructed",'
        '"reference_parts":[{"position":1,"code":"REF-DEMO-03","page":42}],'
        '"catalogue_pages":[42],"evidence":["Standard Type"],'
        '"reason":"Compatible alternative."},'
        '{"rank":1,"reference":"HGD63H2PMCS0000C00003","variant":"Deluxe",'
        '"match_level":"closest","reference_mode":"constructed",'
        '"reference_parts":[{"position":1,"code":"HGD63H","page":41}],'
        '"catalogue_pages":[41],"evidence":["Deluxe Type"],'
        '"reason":"Closest documented variant."}],'
        '"explanation":"Two catalogue variants satisfy the selectors."}'
    ])
    client = NvidiaChatClient(
        api_key="test",
        model="nvidia/test-model",
        client=fake,
    )
    chunks = [
        RankedChunk(
            chunk=CatalogueChunk(
                chunk_id="p41-c1",
                page_number=41,
                text="Deluxe Type HGD63H ordering information",
            ),
            score=0.9,
            word_score=0.9,
            char_score=0.9,
            exact_score=0.9,
        ),
        RankedChunk(
            chunk=CatalogueChunk(
                chunk_id="p42-c1",
                page_number=42,
                text="Standard Type REF-DEMO-03 ordering information",
            ),
            score=0.8,
            word_score=0.8,
            char_score=0.8,
            exact_score=0.8,
        ),
    ]

    result = client.select_reference({"product_type": "MCB"}, chunks)

    assert result["reference"] == "HGD63H2PMCS0000C00003"
    assert [item["reference"] for item in result["candidates"]] == [
        "HGD63H2PMCS0000C00003",
        "REF-DEMO-032PMCS0000C00003",
    ]
    assert result["candidates"][0]["rank"] == 1
    assert result["candidates"][1]["match_level"] == "alternative"


def test_select_reference_normalises_structured_explanation_reason_and_differences() -> None:
    fake = _FakeOpenAIClient([
        '{"candidates":[{'
        '"rank":1,"reference":"AXR-100","variant":null,'
        '"match_level":"closest","reference_mode":"explicit",'
        '"reference_parts":[],"catalogue_pages":[7],'
        '"evidence":["Reference AXR-100"],'
        '"reason":["Exact selector match.","Best documented option."],'
        '"differences":"No documented difference."}],'
        '"explanation":{"resume":"One proved candidate was found."}}'
    ])
    client = NvidiaChatClient(
        api_key="test",
        model="nvidia/test-model",
        client=fake,
    )
    chunks = [
        RankedChunk(
            chunk=CatalogueChunk(
                chunk_id="p7-c1",
                page_number=7,
                text="Reference AXR-100",
            ),
            score=1.0,
            word_score=1.0,
            char_score=1.0,
            exact_score=1.0,
        )
    ]

    result = client.select_reference({"product_type": "industrial product"}, chunks)

    assert result["explanation"] == "One proved candidate was found."
    assert result["candidates"][0]["reason"] == (
        "Exact selector match. Best documented option."
    )
    assert result["candidates"][0]["differences"] == [
        "No documented difference."
    ]



def test_nemotron_default_reasoning_profile_is_normal() -> None:
    fake = _FakeOpenAIClient([
        '{"product_type":"bearing","attributes":[],"search_queries":["bearing"]}'
    ])
    client = NvidiaChatClient(
        api_key="test",
        model="nvidia/nemotron-3-super-120b-a12b",
        client=fake,
        reasoning_mode="normal",
    )

    client.extract_need("Bearing")

    call = fake.chat.completions.calls[0]
    assert call["extra_body"] == {
        "chat_template_kwargs": {
            "enable_thinking": True,
            "low_effort": True,
        },
        "reasoning_budget": 2048,
    }
    assert call["max_tokens"] == 3848


def test_nemotron_fast_reasoning_profile_disables_thinking() -> None:
    fake = _FakeOpenAIClient([
        '{"product_type":"bearing","attributes":[],"search_queries":["bearing"]}'
    ])
    client = NvidiaChatClient(
        api_key="test",
        model="nvidia/nemotron-3-super-120b-a12b",
        client=fake,
        reasoning_mode="rapide",
    )

    client.extract_need("Bearing")

    call = fake.chat.completions.calls[0]
    assert call["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert call["max_tokens"] == 1800


def test_nemotron_deep_reasoning_profile_uses_larger_budget() -> None:
    fake = _FakeOpenAIClient([
        '{"product_type":"bearing","attributes":[],"search_queries":["bearing"]}'
    ])
    client = NvidiaChatClient(
        api_key="test",
        model="nvidia/nemotron-3-super-120b-a12b",
        client=fake,
        reasoning_mode="approfondi",
    )

    client.extract_need("Bearing")

    call = fake.chat.completions.calls[0]
    assert call["extra_body"] == {
        "chat_template_kwargs": {
            "enable_thinking": True,
            "low_effort": False,
        },
        "reasoning_budget": 4096,
    }
    assert call["max_tokens"] == 5896


def test_select_reference_accepts_single_page_label_and_extracts_all_page_numbers() -> None:
    fake = _FakeOpenAIClient([
        '{"candidates":[{"rank":1,"reference":"REF-1",'
        '"catalogue_pages":"pages 41 et 42","reason":"match"}],'
        '"explanation":"match"}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)
    chunk = RankedChunk(
        chunk=CatalogueChunk("p41-c1", 41, "Reference REF-1"),
        score=1.0,
        word_score=1.0,
        char_score=1.0,
        exact_score=1.0,
    )

    result = client.select_reference(
        {"product_type": "component", "attributes": [], "search_queries": ["component"]},
        [chunk],
    )

    assert result["catalogue_pages"] == [41, 42]


def test_http_resource_exhausted_is_retried_once(monkeypatch) -> None:
    import requests
    import rag_catalogue.llm_client as module

    class SaturatedResponse:
        status_code = 503
        text = '{"message":"ResourceExhausted: Worker local total request limit reached"}'

        def raise_for_status(self) -> None:
            error = requests.HTTPError("503 error")
            error.response = self
            raise error

        def json(self):
            return {"message": "ResourceExhausted: Worker local total request limit reached"}

    success = _FakeHTTPResponse({
        "choices": [{
            "message": {
                "content": '{"product_type":"bearing","attributes":[],"search_queries":["bearing"]}'
            }
        }]
    })

    class Session:
        def __init__(self) -> None:
            self.responses = [SaturatedResponse(), success]
            self.calls = 0

        def post(self, *_args, **_kwargs):
            self.calls += 1
            return self.responses.pop(0)

    waits: list[float] = []
    monkeypatch.setattr(module.time, "sleep", waits.append)
    client = NvidiaChatClient(api_key="test", model="nvidia/test")
    session = Session()
    client._session = session

    need = client.extract_need("Bearing")

    assert need["product_type"] == "bearing"
    assert session.calls == 2
    assert waits == [5.0]


def test_select_reference_legacy_response_normalises_evidence_string() -> None:
    fake = _FakeOpenAIClient([
        '{"reference":"REF-1","reference_mode":"explicit","reference_parts":[],'
        '"catalogue_pages":[7],"evidence":"Reference REF-1",'
        '"explanation":"Exact catalogue match."}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)
    chunks = [
        RankedChunk(
            chunk=CatalogueChunk("p7-c1", 7, "Reference REF-1"),
            score=1.0,
            word_score=1.0,
            char_score=1.0,
            exact_score=1.0,
        )
    ]

    result = client.select_reference(
        {"product_type": "component", "attributes": [], "search_queries": ["component"]},
        chunks,
    )

    assert result["evidence"] == ["Reference REF-1"]


def test_select_reference_legacy_response_normalises_evidence_object() -> None:
    fake = _FakeOpenAIClient([
        '{"reference":"REF-2","reference_mode":"explicit","reference_parts":[],'
        '"catalogue_pages":[8],'
        '"evidence":{"summary":"Reference REF-2","details":["Rated value 16 A"]},'
        '"explanation":"Exact catalogue match."}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)
    chunks = [
        RankedChunk(
            chunk=CatalogueChunk("p8-c1", 8, "Reference REF-2. Rated value 16 A"),
            score=1.0,
            word_score=1.0,
            char_score=1.0,
            exact_score=1.0,
        )
    ]

    result = client.select_reference(
        {"product_type": "component", "attributes": [], "search_queries": ["component"]},
        chunks,
    )

    assert result["evidence"] == ["Reference REF-2 Rated value 16 A"]


def test_client_repairs_vendorb_parenthetical_json_locally_without_second_call() -> None:
    malformed = (
        '{"candidates":[{"rank":1,"reference":"REF-DEMO-02CC",'
        '"variant":null,"match_level":"closest","reference_mode":"explicit",'
        '"reference_parts":[],"catalogue_pages":[215],'
        '"evidence":["REF-DEMO-02CC: P1061532" (visible in the STRUCTURE)],'
        '"reason":"exact match","differences":[]}],'
        '"explanation":"best match"}'
    )
    fake = _FakeOpenAIClient([malformed])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)
    chunk = RankedChunk(
        chunk=CatalogueChunk(
            chunk_id="p215-c1",
            page_number=215,
            text="REF-DEMO-02CC: P1061532",
        ),
        score=1.0,
        word_score=1.0,
        char_score=1.0,
        exact_score=1.0,
    )

    result = client.select_reference(
        {
            "product_type": "industrial component",
            "attributes": [],
            "search_queries": ["REF-DEMO-02CC"],
        },
        [chunk],
    )

    assert result["reference"] == "REF-DEMO-02CC"
    assert result["evidence"] == [
        "REF-DEMO-02CC: P1061532 (visible in the STRUCTURE)"
    ]
    assert len(fake.chat.completions.calls) == 1
    config = client.json_repair_config
    assert config["local_attempts"] == 1
    assert config["local_successes"] == 1
    assert config["external_attempts"] == 0


def test_repair_json_locally_preserves_all_malformed_vendorb_values() -> None:
    import rag_catalogue.llm_client as module

    malformed = (
        '{"reference":"REF-DEMO-02CC","catalogue_pages":[215],'
        '"evidence":["REF-DEMO-02CC – 15 A" (p215)],'
        '"explanation":"P1061532 reste un identifiant visible",}'
    )

    repaired = module.repair_json_locally(malformed)

    assert repaired == {
        "reference": "REF-DEMO-02CC",
        "catalogue_pages": [215],
        "evidence": ["REF-DEMO-02CC – 15 A (p215)"],
        "explanation": "P1061532 reste un identifiant visible",
    }


def test_truncated_json_uses_no_thinking_external_repair_and_records_finish_reason() -> None:
    truncated = (
        '{"candidates":[{"rank":1,"reference":"TARGET-1",'
        '"variant":null,"match_level":"equivalent","reference_mode":"explicit",'
        '"reference_parts":[],"catalogue_pages":[9],'
        '"evidence":["TARGET-1"],"reason":"Exact documented'
    )
    repaired = (
        '{"candidates":[{"rank":1,"reference":"TARGET-1",'
        '"variant":null,"match_level":"equivalent","reference_mode":"explicit",'
        '"reference_parts":[],"catalogue_pages":[9],'
        '"evidence":["TARGET-1"],"reason":"Exact documented match",'
        '"differences":[],"mandatory_differences":[]}],'
        '"explanation":"Exact documented match"}'
    )
    fake = _FakeOpenAIClient([(truncated, "length"), (repaired, "stop")])
    client = NvidiaChatClient(
        api_key="test",
        model="nvidia/nemotron-3-super-120b-a12b",
        client=fake,
    )
    chunk = RankedChunk(
        chunk=CatalogueChunk("p9-c1", 9, "Reference TARGET-1"),
        score=1.0,
        word_score=1.0,
        char_score=1.0,
        exact_score=1.0,
    )

    result = client.select_reference(
        {"product_type": "component", "attributes": [], "search_queries": ["component"]},
        [chunk],
    )

    assert result["reference"] == "TARGET-1"
    assert len(fake.chat.completions.calls) == 2
    repair_call = fake.chat.completions.calls[1]
    assert repair_call["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    config = client.json_repair_config
    assert config["truncated_responses"] == 1
    assert config["local_attempts"] == 0
    assert config["external_attempts"] == 1
    assert config["external_successes"] == 1


def test_selection_prompt_keeps_near_alternatives_when_source_product_has_no_exact_match() -> None:
    fake = _FakeOpenAIClient([
        '{"candidates":[],"explanation":"none"}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test-model", client=fake)
    chunk = RankedChunk(
        chunk=CatalogueChunk(
            chunk_id="p214-c2",
            page_number=214,
            text="REF-DEMO-02 N1018590 1000 VDC 15 A gPV; maximum breaking capacity 10 kA.",
        ),
        score=1.0,
        word_score=1.0,
        char_score=1.0,
        exact_score=1.0,
    )

    client.select_reference(
        {
            "product_type": "photovoltaic fuse",
            "attributes": [
                {"name": "rated current", "value": "15", "unit": "A", "role": "selector", "provenance": "source_product"},
                {"name": "rated voltage", "value": "1000", "unit": "VDC", "role": "selector", "provenance": "source_product"},
                {"name": "breaking capacity", "value": "30", "unit": "kA", "role": "selector", "provenance": "source_product"},
            ],
            "open_questions": [],
            "search_queries": ["photovoltaic fuse 15 A 1000 VDC 30 kA gPV"],
        },
        [chunk],
    )

    prompt = " ".join(
        message["content"] for message in fake.chat.completions.calls[0]["messages"]
    ).casefold()
    assert "si aucune correspondance exacte" in prompt
    assert "conserve les alternatives les plus proches" in prompt
    assert "provenance=client" in prompt
    assert "provenance=derived_required" in prompt
    assert "equivalence directe" in prompt
    assert "differences" in prompt


def test_extract_need_preserves_derived_required_provenance() -> None:
    fake = _FakeOpenAIClient([
        '{"product_type":"manual transfer switch",'
        '"source_reference":"REF-400",'
        '"attributes":['
        '{"name":"function","value":"manual source transfer","unit":null,"role":"constraint","provenance":"derived_required"},'
        '{"name":"rated current","value":"400","unit":"A","role":"constraint","provenance":"derived_required"},'
        '{"name":"mounting","value":"panel","unit":null,"role":"selector","provenance":"source_product"}],'
        '"search_queries":["manual transfer switch 400 A panel"]}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)

    need = client.extract_need("Reference source and structured V4 provenance")

    assert need["attributes"][0]["provenance"] == "derived_required"
    assert need["attributes"][1]["provenance"] == "derived_required"
    prompt = fake.chat.completions.calls[0]["messages"][1]["content"]
    assert "derived_required" in prompt


def test_select_reference_preserves_mandatory_differences_and_downgrades_equivalent() -> None:
    fake = _FakeOpenAIClient([
        '{"candidates":[{'
        '"rank":1,"reference":"TARGET-400","variant":null,'
        '"match_level":"equivalent","reference_mode":"explicit",'
        '"reference_parts":[],"catalogue_pages":[12],'
        '"evidence":["TARGET-400 is a 400 A four-pole switch"],'
        '"reason":"Same current and pole count.",'
        '"differences":["Simple switch instead of source transfer"],'
        '"mandatory_differences":["Function differs: simple switch instead of source transfer"]'
        '}],"explanation":"One close alternative."}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)
    chunk = RankedChunk(
        chunk=CatalogueChunk(
            chunk_id="p12-c1",
            page_number=12,
            text="TARGET-400 is a 400 A four-pole switch.",
        ),
        score=1.0,
        word_score=1.0,
        char_score=1.0,
        exact_score=1.0,
    )

    result = client.select_reference(
        {
            "product_type": "manual source transfer switch",
            "attributes": [{
                "name": "function",
                "value": "manual source transfer",
                "unit": None,
                "role": "constraint",
                "provenance": "derived_required",
            }],
            "search_queries": ["manual source transfer switch 400 A 4 pole"],
        },
        [chunk],
    )

    candidate = result["candidates"][0]
    assert candidate["mandatory_differences"] == [
        "Function differs: simple switch instead of source transfer"
    ]
    assert candidate["match_level"] == "alternative"
    prompt = fake.chat.completions.calls[0]["messages"][1]["content"]
    assert '"mandatory_differences"' in prompt


class _RoutingStub:
    def __init__(self, model, *, need=None, selection=None, error=None):
        self.model = model
        self.need = need
        self.selection = selection
        self.error = error
        self.calls = []
        self.reasoning_config = {"mode": "normal", "enabled": False, "low_effort": None, "budget": 0}
        self.json_repair_config = {"attempts": 0, "successes": 0}

    def extract_need(self, product_text):
        self.calls.append(("extract_need", product_text))
        if self.error == "extract":
            raise LLMResponseError("primary extraction failed")
        return self.need

    def select_reference(self, need, chunks):
        self.calls.append(("select_reference", need, chunks))
        if self.error == "select":
            raise LLMResponseError("primary selection failed")
        return self.selection


def test_resilient_client_uses_fallback_on_structured_output_error():
    from rag_catalogue.llm_client import ResilientNvidiaChatClient

    primary = _RoutingStub("model-49b", error="extract")
    fallback = _RoutingStub("model-120b", need={"product_type": "switch", "attributes": [], "search_queries": ["switch"]})
    client = ResilientNvidiaChatClient(primary, fallback)

    result = client.extract_need("source product")

    assert result["product_type"] == "switch"
    assert client.model_routing_config["fallback_used"] is True
    assert client.model_routing_config["stage_models"]["extract_need"] == "model-120b"


def test_resilient_client_uses_fallback_when_primary_returns_empty_candidates_with_context():
    from rag_catalogue.llm_client import ResilientNvidiaChatClient

    primary = _RoutingStub("model-49b", selection={"reference": None, "candidates": [], "catalogue_pages": [], "evidence": [], "explanation": "none"})
    fallback = _RoutingStub("model-120b", selection={"reference": "TARGET-1", "candidates": [{"reference": "TARGET-1"}], "catalogue_pages": [1], "evidence": ["TARGET-1"], "explanation": "found"})
    client = ResilientNvidiaChatClient(primary, fallback)
    chunk = RankedChunk(CatalogueChunk("p1-c1", 1, "TARGET-1"), 1.0, 1.0, 1.0, 1.0)

    result = client.select_reference({"product_type": "switch"}, [chunk])

    assert result["reference"] == "TARGET-1"
    assert client.model_routing_config["fallback_used"] is True
    assert client.model_routing_config["stage_models"]["select_reference"] == "model-120b"


def test_resilient_client_keeps_primary_when_result_is_usable():
    from rag_catalogue.llm_client import ResilientNvidiaChatClient

    primary = _RoutingStub("model-49b", selection={"reference": "TARGET-1", "candidates": [{"reference": "TARGET-1"}], "catalogue_pages": [1], "evidence": ["TARGET-1"], "explanation": "found"})
    fallback = _RoutingStub("model-120b", selection={"reference": "TARGET-2", "candidates": [{"reference": "TARGET-2"}]})
    client = ResilientNvidiaChatClient(primary, fallback)
    chunk = RankedChunk(CatalogueChunk("p1-c1", 1, "TARGET-1"), 1.0, 1.0, 1.0, 1.0)

    result = client.select_reference({"product_type": "switch"}, [chunk])

    assert result["reference"] == "TARGET-1"
    assert client.model_routing_config["fallback_used"] is False
    assert fallback.calls == []
