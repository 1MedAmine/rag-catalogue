from pathlib import Path
from types import SimpleNamespace

import fitz

from rag_catalogue.llm_client import NvidiaChatClient
from rag_catalogue.pipeline import retrieve_catalogue_chunks, run_search


def _write_pdf(path: Path, pages: list[str]) -> None:
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        page.insert_textbox((50, 50, 550, 780), text, fontsize=10)
    document.save(path)
    document.close()


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _FakeOpenAIClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


def test_need_extraction_uses_domain_neutral_schema() -> None:
    fake = _FakeOpenAIClient([
        '{"product_type":"roulement a billes",'
        '"source_reference":"SRC-20X52",'
        '"attributes":[{"name":"bore diameter","value":"20","unit":"mm"},'
        '{"name":"outside diameter","value":"52","unit":"mm"},'
        '{"name":"width","value":"15","unit":"mm"},'
        '{"name":"seal","value":"double rubber","unit":null}],'
        '"search_queries":["deep groove ball bearing 20 mm 52 mm 15 mm double rubber seal",'
        '"roulement rigide billes alesage 20 diametre exterieur 52 largeur 15 etancheite 2 faces"]}'
    ])
    client = NvidiaChatClient(api_key="test", model="nvidia/test", client=fake)

    result = client.extract_need(
        "Roulement rigide a billes, alesage 20 mm, diametre 52 mm, largeur 15 mm, "
        "etancheite caoutchouc des deux cotes."
    )

    assert result["source_reference"] == "SRC-20X52"
    assert len(result["attributes"]) == 4
    assert len(result["search_queries"]) == 2
    prompt = fake.chat.completions.calls[0]["messages"][1]["content"].casefold()
    assert "nombre de poles" not in prompt
    assert "pouvoir de coupure" not in prompt
    assert "courbe" not in prompt


def test_retrieval_accepts_multiple_queries_without_target_reference(tmp_path: Path) -> None:
    catalogue = tmp_path / "bearing_catalogue.pdf"
    _write_pdf(
        catalogue,
        [
            "Reference 6204-Z. Deep groove ball bearing. Bore 20 mm. Outside diameter 47 mm. Width 14 mm. One metal shield.",
            "Reference 6304-2RS. Deep groove ball bearing. Bore diameter 20 mm. Outside diameter 52 mm. Width 15 mm. Double rubber seals.",
            "Reference NU204. Cylindrical roller bearing. Bore 20 mm. Outside diameter 47 mm. Width 14 mm.",
        ],
    )

    results = retrieve_catalogue_chunks(
        catalogue,
        [
            "deep groove ball bearing bore 20 mm outside diameter 52 mm width 15 mm double rubber seals",
            "roulement a billes 20 x 52 x 15 etanche deux faces",
        ],
        top_k=1,
    )

    assert results[0].chunk.page_number == 2
    assert "6304-2RS" in results[0].chunk.text


class _GenericFakeLLM:
    model = "nvidia/fake-generic"

    def __init__(self) -> None:
        self.received_chunks = []

    def extract_need(self, product_text: str) -> dict:
        return {
            "product_type": "inductive proximity sensor",
            "source_reference": "SOURCE-SENSOR",
            "attributes": [
                {"name": "housing", "value": "M18", "unit": None},
                {"name": "output", "value": "PNP normally open", "unit": None},
                {"name": "sensing distance", "value": "8", "unit": "mm"},
                {"name": "supply voltage", "value": "10-30", "unit": "V DC"},
                {"name": "connection", "value": "M12", "unit": None},
            ],
            "search_queries": [
                "inductive proximity sensor M18 PNP normally open 8 mm 10-30 V DC M12"
            ],
        }

    def select_reference(self, need: dict, chunks: list) -> dict:
        self.received_chunks = chunks
        return {
            "reference": "SEN-M18-PNP-8-M12",
            "catalogue_pages": [2],
            "evidence": ["M18, PNP NO, 8 mm, 10-30 V DC, M12"],
            "explanation": "All requested characteristics appear on the same catalogue line.",
        }


def test_pipeline_finds_reference_without_putting_target_code_in_query(tmp_path: Path) -> None:
    product = tmp_path / "source_sensor.pdf"
    catalogue = tmp_path / "sensor_catalogue.pdf"
    _write_pdf(product, ["M18 inductive sensor, PNP NO, 8 mm, 10-30 V DC, M12 connector"])
    _write_pdf(
        catalogue,
        [
            "Reference SEN-M18-NPN-8-M12. Inductive proximity sensor M18. NPN normally open. Sensing distance 8 mm. Supply 10-30 V DC. Connector M12.",
            "Reference SEN-M18-PNP-8-M12. Inductive proximity sensor M18. PNP normally open. Sensing distance 8 mm. Supply 10-30 V DC. Connector M12.",
        ],
    )
    llm = _GenericFakeLLM()

    result = run_search(product, catalogue, llm, top_k=1)

    assert result["reference"] == "SEN-M18-PNP-8-M12"
    assert llm.received_chunks[0].chunk.page_number == 2
