"""Enrich data/corpus/studies.jsonl with full PubMed abstracts via NCBI E-utilities.

The committed corpus carries verified citation metadata and a paraphrased summary of
each study. Run this on a machine with internet access to pull the official
abstracts (stored in the ``abstract`` field, which retrieval then prefers):

    python scripts/fetch_pubmed.py                 # fills abstracts for records with a PMID
    python scripts/fetch_pubmed.py --resolve-doi   # also looks up missing PMIDs by DOI

Set NCBI_API_KEY to raise the rate limit (10 req/s instead of 3). Uses only the
standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus" / "studies.jsonl"


def _get(endpoint: str, **params) -> bytes:
    params.update({"tool": "neuromod", "email": os.getenv("NCBI_EMAIL", "")})
    if os.getenv("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]
    url = f"{EUTILS}/{endpoint}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 - fixed https host
        data = r.read()
    time.sleep(0.12 if os.getenv("NCBI_API_KEY") else 0.35)
    return data


def pmid_for_doi(doi: str) -> str | None:
    data = json.loads(_get("esearch.fcgi", db="pubmed", term=f"{doi}[doi]", retmode="json"))
    ids = data.get("esearchresult", {}).get("idlist", [])
    return ids[0] if ids else None


def fetch_abstracts(pmids: list[str]) -> dict[str, str]:
    out = {}
    for i in range(0, len(pmids), 50):
        root = ET.fromstring(_get("efetch.fcgi", db="pubmed", id=",".join(pmids[i:i + 50]),
                                  retmode="xml"))
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext(".//PMID")
            parts = []
            for node in art.iter("AbstractText"):
                label = node.get("Label")
                text = "".join(node.itertext()).strip()
                parts.append(f"{label}: {text}" if label else text)
            if pmid and parts:
                out[pmid] = " ".join(parts)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--resolve-doi", action="store_true")
    args = ap.parse_args()

    path = Path(args.corpus)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if args.resolve_doi:
        for rec in records:
            if not rec.get("pmid") and rec.get("doi"):
                rec["pmid"] = pmid_for_doi(rec["doi"])
                print(f"{rec['id']}: DOI -> PMID {rec['pmid']}")
    pmids = [r["pmid"] for r in records if r.get("pmid")]
    abstracts = fetch_abstracts(pmids)
    for rec in records:
        if rec.get("pmid") in abstracts:
            rec["abstract"] = abstracts[rec["pmid"]]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    print(f"Added abstracts for {len(abstracts)}/{len(records)} studies -> {path}")


if __name__ == "__main__":
    main()
