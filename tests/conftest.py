from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def small_cohort():
    from neuromod.data.synthetic import simulate_cohort

    return simulate_cohort(n=600, seed=11)


@pytest.fixture(scope="session")
def mini_index():
    from neuromod.rag.retriever import RTMSIndex

    return RTMSIndex.from_corpus(FIXTURES / "mini_corpus.jsonl")
