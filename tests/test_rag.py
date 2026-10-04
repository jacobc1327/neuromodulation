import json
from pathlib import Path

import pytest

from neuromod.rag.corpus import DEFAULT_CORPUS, load_studies
from neuromod.rag.grounding import check_grounding
from neuromod.rag.pipeline import RTMSResearchAssistant


def test_hybrid_retrieval_finds_relevant(mini_index):
    for mode in ["bm25", "dense", "hybrid"]:
        docs = mini_index.retriever(k=2, mode=mode).invoke("deep TMS for smoking cessation")
        assert docs[0].metadata["study_id"] == "fixture_smoking", mode
    docs = mini_index.retriever(k=4).invoke("PTSD veterans")
    assert docs[0].metadata["study_id"] == "fixture_ptsd"
    assert len({d.metadata["study_id"] for d in docs}) == len(docs)  # one chunk per study


def test_index_roundtrip(mini_index, tmp_path):
    from neuromod.rag.retriever import RTMSIndex

    mini_index.save(tmp_path)
    loaded = RTMSIndex.load(tmp_path)
    q = "alcohol craving rTMS"
    a = [d.metadata["study_id"] for d in mini_index.retriever(k=3).invoke(q)]
    b = [d.metadata["study_id"] for d in loaded.retriever(k=3).invoke(q)]
    assert a == b


def test_grounding_checker():
    sources = ["Active rTMS reduced alcohol craving compared with sham in detoxified patients."]
    good = "Active rTMS reduced alcohol craving versus sham [S1]."
    assert check_grounding(good, sources).grounding_score == 1.0
    assert check_grounding("Active rTMS reduced alcohol craving versus sham.", sources).grounding_score == 0
    assert check_grounding("Active rTMS reduced alcohol craving versus sham [S3].", sources).grounding_score == 0
    bad = "Ketamine infusions cured depression in astronauts permanently [S1]."
    assert check_grounding(bad, sources).grounding_score == 0


@pytest.mark.parametrize("mode", ["ask", "hypotheses", "design"])
def test_offline_assistant_is_grounded(mini_index, mode, monkeypatch):
    monkeypatch.setenv("NEUROMOD_OFFLINE", "1")
    a = RTMSResearchAssistant(index=mini_index, k=3, metrics_path=None)
    ans = a.run("Does rTMS reduce craving in alcohol use disorder?", mode=mode)
    assert ans.backend.startswith("extractive")
    assert ans.grounding["grounding_score"] == 1.0
    assert ans.grounding["citation_coverage"] == 1.0
    assert "[S1]" in ans.answer
    md = ans.to_markdown()
    assert "References" in md and "[S1]" in md


def test_llm_path_uses_langchain_chain(mini_index):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    fake = FakeListChatModel(responses=[
        "Left DLPFC rTMS reduced alcohol craving scores at the end of treatment [S1]. "
        "It also cured every patient forever [S9]."])
    a = RTMSResearchAssistant(index=mini_index, k=3, llm=fake, metrics_path=None)
    ans = a.ask("alcohol craving rTMS")
    assert ans.grounding["n_claims"] == 2
    assert len(ans.grounding["unsupported"]) == 1  # the hallucinated [S9] claim is caught
    strict = RTMSResearchAssistant(index=mini_index, k=3, metrics_path=None,
                                   llm=FakeListChatModel(responses=[fake.responses[0]]))
    s = strict.ask("alcohol craving rTMS", strict=True)
    assert "forever" not in s.answer and s.grounding["grounding_score"] == 1.0


@pytest.mark.skipif(not Path(DEFAULT_CORPUS).exists(), reason="corpus not built")
def test_real_corpus_integrity():
    studies = load_studies()
    assert len(studies) >= 30
    for s in studies:
        assert s.doi or s.pmid, f"{s.id} has no DOI or PMID"
        assert 1990 <= s.year <= 2026
        assert s.link
    evals = [json.loads(x) for x in (Path(DEFAULT_CORPUS).parent / "eval_questions.jsonl")
             .read_text().splitlines() if x.strip()]
    ids = {s.id for s in studies}
    for q in evals:
        assert set(q["relevant"]) <= ids, q
