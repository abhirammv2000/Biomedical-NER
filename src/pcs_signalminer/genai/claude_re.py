"""LLM-based document-level chemical-induced-disease (CID) relation extraction
with Claude.

Given an abstract and the chemical/disease concepts it mentions, Claude decides
which (chemical, disease) pairs are reported as chemical-INDUCED disease. This is
the GenAI arm of the build-vs-buy study against the fine-tuned encoder.

Uses the Anthropic SDK with structured outputs (output_config.format) so the
response is guaranteed-parseable JSON — no prefill, no temperature (both rejected
on Opus 4.8). A deterministic mock path lets the pipeline run end-to-end without
an API key (set PCS_GENAI_MOCK=1 or pass mock=True).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

MODEL = "claude-opus-4-8"

# Opus 4.8 pricing, USD per 1M tokens
PRICE_IN = 5.0 / 1_000_000
PRICE_OUT = 25.0 / 1_000_000

SYSTEM = (
    "You are a biomedical safety-literature analyst. Given a PubMed abstract and "
    "the chemical and disease concepts it mentions, identify every pair where the "
    "abstract reports that the CHEMICAL INDUCES / CAUSES the DISEASE (an adverse "
    "drug/chemical effect). This is the BioCreative CDR 'chemical-induced disease' "
    "relation, judged at the document level (evidence may span sentences). "
    "Only output pairs drawn from the provided concept ids. Do NOT include a pair "
    "merely because both are mentioned, or because the chemical TREATS the disease."
)

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chemical_mesh": {"type": "string"},
                    "disease_mesh": {"type": "string"},
                },
                "required": ["chemical_mesh", "disease_mesh"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["relations"],
    "additionalProperties": False,
}


@dataclass
class REResult:
    pairs: set[tuple[str, str]]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * PRICE_IN + self.output_tokens * PRICE_OUT


def build_user_prompt(text: str,
                      chemicals: dict[str, list[str]],
                      diseases: dict[str, list[str]]) -> str:
    chem_lines = "\n".join(f"  - {cid}: {', '.join(names)}" for cid, names in chemicals.items())
    dis_lines = "\n".join(f"  - {did}: {', '.join(names)}" for did, names in diseases.items())
    return (
        f"ABSTRACT:\n{text}\n\n"
        f"CHEMICAL CONCEPTS (id: names):\n{chem_lines}\n\n"
        f"DISEASE CONCEPTS (id: names):\n{dis_lines}\n\n"
        "Return the chemical-induced-disease pairs as JSON."
    )


# --------------------------------------------------------------------------- #
# Mock path — deterministic co-occurrence-in-same-sentence heuristic.
# Lets the whole evaluation pipeline run (and be unit-tested) without a key.
# --------------------------------------------------------------------------- #
def _mock_extract(text: str,
                  chemicals: dict[str, list[str]],
                  diseases: dict[str, list[str]]) -> set[tuple[str, str]]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pairs: set[tuple[str, str]] = set()
    low_sents = [s.lower() for s in sentences]
    for cid, cnames in chemicals.items():
        for did, dnames in diseases.items():
            for s in low_sents:
                if any(cn.lower() in s for cn in cnames) and any(dn.lower() in s for dn in dnames):
                    pairs.add((cid, did))
                    break
    return pairs


def extract_cid(text: str,
                chemicals: dict[str, list[str]],
                diseases: dict[str, list[str]],
                client=None,
                model: str = MODEL,
                mock: bool | None = None) -> REResult:
    """Extract CID pairs for one document. Falls back to the mock heuristic when
    mock=True (or PCS_GENAI_MOCK=1, or no client/API key available)."""
    if mock is None:
        mock = os.getenv("PCS_GENAI_MOCK") == "1"

    if mock or (client is None and not os.getenv("ANTHROPIC_API_KEY")):
        return REResult(pairs=_mock_extract(text, chemicals, diseases))

    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_user_prompt(text, chemicals, diseases)}],
        output_config={"format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA},
                       "effort": "low"},
    )
    latency = time.time() - t0
    payload = next((b.text for b in resp.content if b.type == "text"), "{}")
    try:
        data = json.loads(payload)
        pairs = {(r["chemical_mesh"], r["disease_mesh"]) for r in data.get("relations", [])}
    except (json.JSONDecodeError, KeyError, TypeError):
        pairs = set()
    # keep only pairs that reference provided concepts (guard against hallucinated ids)
    pairs = {(c, d) for (c, d) in pairs if c in chemicals and d in diseases}
    return REResult(pairs=pairs,
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                    latency_s=latency)
