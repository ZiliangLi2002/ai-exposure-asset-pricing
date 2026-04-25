#!/usr/bin/env python3
"""
AI Adoption Scoring Pipeline (OpenRouter + Earnings Call Transcript Blocks)

How to run:
    1) Place your input CSV in the working directory (or pass --input path).
    2) Set environment variables:
         - OPENROUTER_API_KEY
         - OPENROUTER_MODEL
    3) Run:
         python ai_adoption_openrouter_pipeline.py \
             --input transcript_2016_2023_finalversion.csv \
             --output-dir outputs

Expected input:
    A block-level earnings transcript CSV with columns including:
      keydevid, transcriptid, headline, mostimportantdateutc, companyid,
      block_num, section, Id, componenttext, pretty_text

Generated output files (in --output-dir):
    - candidate_blocks.csv
    - block_level_ai_annotations.csv
    - call_level_ai_scores.csv
    - company_quarter_ai_scores.csv
    - llm_failures.csv
    - llm_cache.jsonl (cache for repeated runs)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


LOGGER = logging.getLogger("ai_adoption_pipeline")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_VERSION = "v4_dual_adoption_innovation"
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct"
DEFAULT_MAX_TOKENS = 700
DEFAULT_TICKER_MAP_CSV = "companyid_to_ticker_wrds0419.csv"

_REQUEST_GUARD_LOCK = threading.Lock()
_LAST_REQUEST_TS = 0.0

# ---------------------------------------------------------------------------
# Retrieval vocabulary – three tiers
#
# The vocabulary was expanded from a narrow academic prototype into a broader
# enterprise-AI lexicon so that real earnings-call adoption language is captured
# (product names like "Copilot", deployment phrases like "AI-powered workflow",
# agentic / RAG / fine-tuning terminology, etc.).
#
# Tier 1 – STRONG: unambiguously AI-related; triggers candidate status alone.
# Tier 2 – WEAK:   common in AI context but also used outside AI; requires
#                   contextual support (adjacent strong term or contextual phrase).
# Tier 3 – CONTEXTUAL PHRASES: compound phrases where a weak word is paired
#                   with an explicit AI qualifier, confirming AI relevance.
# ---------------------------------------------------------------------------

STRONG_AI_TERMS = [
    r"\bartificial intelligence\b",
    r"\bgenerative ai\b",
    r"\bgen[\s-]?ai\b",
    r"\bmachine learning\b",
    r"\bdeep learning\b",
    r"\blarge language model[s]?\b",
    r"\bllm\b",
    r"\bllms\b",
    r"\bfoundation model[s]?\b",
    r"\bneural network[s]?\b",
    r"\bcomputer vision\b",
    r"\bnatural language processing\b",
    r"\bnlp\b",
    r"\bconversational ai\b",
    r"\bmultimodal\b",
    r"\bautonomous agent[s]?\b",
    r"\bai agent[s]?\b",
    r"\bagentic\b",
    r"\bgenai\b",
    r"\btext[- ]to[- ](?:text|image|video)\b",
    r"\bspeech recognition\b",
    r"\bspeech[- ]to[- ]text\b",
    r"\btext[- ]to[- ]speech\b",
    r"\brecommendation engine\b",
    r"\brecommender system[s]?\b",
    r"\bdocument intelligence\b",
    r"\bintelligent document processing\b",
    r"\bsemantic search\b",
    r"\bvector search\b",
    r"\brag\b",
    r"\bretrieval[- ]augmented generation\b",
    r"\bprompt engineering\b",
    r"\bfine[- ]tuning\b",
    r"\bmodel inference\b",
    r"\bpredictive model(?:ing)?\b",
    r"\bai[- ]powered\b",
    r"\bai[- ]enabled\b",
    r"\bai[- ]driven\b",
    r"\bml[- ]powered\b",
    r"\bmachine[- ]learning[- ]powered\b",
]

WEAK_AI_TERMS = [
    r"\bassistant\b",
    r"\bcopilot\b",
    r"\bcopilots\b",
    r"\bchatbot\b",
    r"\bchatbots\b",
    r"\bvirtual assistant\b",
    r"\bdigital assistant\b",
    r"\bautomation\b",
    r"\bworkflow automation\b",
    r"\bworkflow orchestration\b",
    r"\binference\b",
    r"\bintelligent\b",
    r"\bintelligent systems?\b",
    r"\bpredictive\b",
    r"\bprediction\b",
    r"\bforecasting\b",
    r"\brecommendation\b",
    r"\brecommendations\b",
    r"\bpersonalization\b",
    r"\bpersonalized\b",
    r"\bsummarization\b",
    r"\bsummarize\b",
    r"\bsearch\b",
    r"\bsemantic\b",
    r"\bknowledge\b",
    r"\bdecision(?:ing)?\b",
    r"\bclassification\b",
    r"\btranscription\b",
    r"\bconversation(?:al)?\b",
    r"\bgeneration\b",
    r"\bcontent generation\b",
    r"\bcode generation\b",
    r"\bdocument processing\b",
    r"\bsmart routing\b",
    r"\bcase routing\b",
    r"\bnext[- ]best[- ]action\b",
    r"\bagent assist\b",
    r"\bauto[- ]reply\b",
    r"\bauto[- ]draft\b",
    r"\bknowledge assistant\b",
    r"\bproductivity tool\b",
    r"\bdecision support\b",
]

AI_CONTEXTUAL_PHRASES = [
    r"\bai assistant\b",
    r"\bai[- ]powered assistant\b",
    r"\bvirtual ai assistant\b",
    r"\bdigital ai assistant\b",
    r"\bai copilot\b",
    r"\bsecurity copilot\b",
    r"\bsales copilot\b",
    r"\bservice copilot\b",
    r"\bdeveloper copilot\b",
    r"\bai chatbot\b",
    r"\bconversational assistant\b",
    r"\bconversational search\b",
    r"\bconversational interface\b",
    r"\bml inference\b",
    r"\bmachine learning inference\b",
    r"\bgenerative ai assistant\b",
    r"\bgenerative ai search\b",
    r"\bgenerative search\b",
    r"\bai automation\b",
    r"\bintelligent automation\b",
    r"\bintelligent automation with ai\b",
    r"\bai recommendation engine\b",
    r"\bai[- ]driven recommendation[s]?\b",
    r"\bai[- ]driven personalization\b",
    r"\bai[- ]powered personalization\b",
    r"\bai[- ]powered workflow\b",
    r"\bai[- ]enabled workflow\b",
    r"\bai[- ]driven workflow\b",
    r"\bai[- ]enabled automation\b",
    r"\bai[- ]powered search\b",
    r"\bsemantic search with ai\b",
    r"\bai summarization\b",
    r"\bai[- ]generated summary\b",
    r"\bai[- ]assisted coding\b",
    r"\bai[- ]generated content\b",
    r"\bai[- ]powered routing\b",
    r"\bai[- ]driven decision(?:ing)?\b",
    r"\bdocument intelligence\b",
    r"\bintelligent document processing\b",
    r"\bknowledge assistant powered by ai\b",
    r"\bai agent\b",
    r"\bai agents\b",
    r"\bagentic workflow\b",
    r"\bagentic automation\b",
]

REQUIRED_COLUMNS = [
    "keydevid",
    "transcriptid",
    "headline",
    "mostimportantdateutc",
    "companyid",
    "block_num",
    "section",
    "Id",
    "componenttext",
    "pretty_text",
]

# ---------------------------------------------------------------------------
# Label taxonomy – richer categories better aligned with real enterprise AI
# discussion patterns found in earnings calls.
# ---------------------------------------------------------------------------

DISCUSSION_TYPES = {
    "non_ai",
    "general_market_commentary",
    "strategy_or_positioning",
    "customer_workflow_integration",
    "internal_operations",
    "internal_productivity",
    "customer_support_automation",
    "sales_marketing_enablement",
    "analytics_prediction_decisioning",
    "search_knowledge_management",
    "enterprise_client_solution",
    "other_ai_business_use",
    # Narrow, frontier/supply-side innovation buckets only.
    "r_and_d_or_model_development",
    "infrastructure_compute_data_stack",
    "proprietary_ai_platform",
    "ai_native_product_launch",
    "technical_capability_creation",
    "patent_or_research_activity",
    "commercialization_of_proprietary_ai",
    "frontier_ai_differentiation",
}

ADOPTION_STAGES = {
    "none",
    "awareness",
    "evaluation",
    "pilot",
    "deployment",
    "scaled",
}

ADOPTION_SCOPES = {
    "none",
    "single_team",
    "multi_team",
    "business_unit",
    "enterprise_wide",
    "customer_embedded",
}

ADOPTION_FOCUS_TYPES = {
    "none",
    "internal_productivity",
    "internal_operations",
    "customer_workflow",
    "product_feature",
    "enterprise_solution",
    "other",
}

INNOVATION_TYPES = {
    "none",
    "model_innovation",
    "infrastructure_innovation",
    "platform_innovation",
    "ai_native_product_innovation",
    "technical_capability_innovation",
    "commercialization_of_proprietary_ai",
}

INNOVATION_NOVELTY_TYPES = {
    "none",
    "incremental",
    "meaningful_extension",
    "new_to_firm",
    "new_to_market",
}

COMMERCIALIZATION_STAGES = {
    "none",
    "concept",
    "prototype",
    "pilot_launch",
    "commercial_launch",
    "scaled_revenue",
}

INNOVATION_TECH_FOCUS_TYPES = {
    "none",
    "model",
    "platform",
    "data_asset",
    "infrastructure",
    "other_frontier",
}

BUSINESS_FUNCTIONS = {
    "product",
    "engineering",
    "it",
    "data_analytics",
    "sales",
    "marketing",
    "customer_service",
    "operations",
    "finance",
    "risk_compliance",
    "security",
    "hr",
    "supply_chain",
    "procurement",
    "internal_productivity",
    "search_knowledge_management",
    "enterprise_client_solution",
    "research_and_development",
    "unknown",
}

IMPACT_TYPES = {
    "productivity",
    "cost",
    "revenue",
    "customer_engagement",
    "conversion",
    "retention",
    "speed",
    "quality",
    "accuracy",
    "capacity",
    "time_to_market",
    "risk_reduction",
    "employee_experience",
    "innovation_output",
    "competitive_differentiation",
    "unknown",
}

# Backward-compatible mappings for legacy prompt outputs.
_ADOPTION_STAGE_COMPAT = {
    "exploration": "evaluation",
}
_DISCUSSION_TYPE_COMPAT = {
    "investor_hype_response": "general_market_commentary",
    "product_feature": "other_ai_business_use",
    "product_platform_integration": "enterprise_client_solution",
    "content_or_code_generation": "internal_productivity",
    "risk_compliance_security_use_case": "other_ai_business_use",
    "innovation_process_enablement": "technical_capability_creation",
}
_ADOPTION_SCOPE_COMPAT: Dict[str, str] = {}
_ADOPTION_FOCUS_COMPAT: Dict[str, str] = {}
_INNOVATION_TYPE_COMPAT: Dict[str, str] = {
    "product_innovation": "ai_native_product_innovation",
    "technology_innovation": "model_innovation",
    "process_innovation": "none",
    "business_model_innovation": "none",
    "innovation_process_enablement": "technical_capability_innovation",
}
_INNOVATION_NOVELTY_COMPAT: Dict[str, str] = {}
_COMMERCIALIZATION_STAGE_COMPAT: Dict[str, str] = {}
_INNOVATION_TECH_FOCUS_COMPAT: Dict[str, str] = {}

FRONTIER_DISCUSSION_TYPES = {
    "r_and_d_or_model_development",
    "infrastructure_compute_data_stack",
    "proprietary_ai_platform",
    "ai_native_product_launch",
    "technical_capability_creation",
    "patent_or_research_activity",
    "commercialization_of_proprietary_ai",
    "frontier_ai_differentiation",
}

FRONTIER_INNOVATION_TYPES = {
    "model_innovation",
    "infrastructure_innovation",
    "platform_innovation",
    "ai_native_product_innovation",
    "technical_capability_innovation",
    "commercialization_of_proprietary_ai",
}


def load_data(input_csv: Path) -> pd.DataFrame:
    """Load input CSV and validate core columns."""
    LOGGER.info("Loading CSV: %s", input_csv)
    df = pd.read_csv(input_csv, low_memory=False)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure text fields exist as strings (keep NaN as empty for processing)
    for col in ["componenttext", "pretty_text", "headline", "section", "Id"]:
        df[col] = df[col].fillna("").astype(str)

    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamp and derive year/quarter/fiscal_period."""
    out = df.copy()
    out["parsed_datetime"] = pd.to_datetime(
        out["mostimportantdateutc"], utc=True, errors="coerce"
    )
    invalid_ts = out["parsed_datetime"].isna().sum()
    if invalid_ts > 0:
        LOGGER.warning("Invalid timestamps found: %d", invalid_ts)

    out["year"] = out["parsed_datetime"].dt.year
    out["quarter"] = out["parsed_datetime"].dt.quarter
    out["fiscal_period"] = np.where(
        out["year"].notna() & out["quarter"].notna(),
        out["year"].astype("Int64").astype(str) + "Q" + out["quarter"].astype("Int64").astype(str),
        "",
    )

    # Basic text quality features
    out["componenttext"] = out["componenttext"].fillna("").astype(str)
    out["has_text"] = out["componenttext"].str.strip().ne("")
    out["word_count"] = (
        out["componenttext"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.split(" ")
        .apply(lambda x: 0 if (not isinstance(x, list) or x == [""]) else len(x))
    )
    return out


def _compile_regex_list(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]


STRONG_AI_REGEX = _compile_regex_list(STRONG_AI_TERMS)
WEAK_AI_REGEX = _compile_regex_list(WEAK_AI_TERMS)
AI_CONTEXTUAL_PHRASE_REGEX = _compile_regex_list(AI_CONTEXTUAL_PHRASES)


def contains_strong_ai_term(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(rx.search(t) for rx in STRONG_AI_REGEX)


def contains_weak_ai_term(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(rx.search(t) for rx in WEAK_AI_REGEX)


def _contains_ai_contextual_phrase(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(rx.search(t) for rx in AI_CONTEXTUAL_PHRASE_REGEX)


def has_contextual_ai_support(
    prev_text: str, curr_text: str, next_text: str, headline: str = ""
) -> bool:
    """
    Weak AI terms need support from local context:
    1) weak + strong in same block, or
    2) weak in current block and strong in previous/next block in same local sequence, or
    3) weak appears in an explicit AI phrase, or
    4) headline carries explicit AI signal.
    """
    curr_has_weak = contains_weak_ai_term(curr_text)
    if not curr_has_weak:
        return False
    if contains_strong_ai_term(curr_text) or _contains_ai_contextual_phrase(curr_text):
        return True
    if contains_strong_ai_term(prev_text) or contains_strong_ai_term(next_text):
        return True
    if _contains_ai_contextual_phrase(prev_text) or _contains_ai_contextual_phrase(next_text):
        return True
    if contains_strong_ai_term(headline) or _contains_ai_contextual_phrase(headline):
        return True
    return False


def is_candidate_block(prev_text: str, curr_text: str, next_text: str, headline: str = "") -> bool:
    """
    Shared candidate retrieval logic:
    - Strong terms trigger directly.
    - Weak terms trigger only with contextual AI support.
    This shared layer only decides whether a block is AI-relevant enough
    to send to the LLM. Adoption vs innovation is decided later by the LLM.
    """
    combined_curr = f"{curr_text} {headline}".strip()
    if contains_strong_ai_term(combined_curr):
        return True
    return has_contextual_ai_support(
        prev_text=prev_text,
        curr_text=combined_curr,
        next_text=next_text,
        headline=headline,
    )


def add_local_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add previous/next block text within the same keydevid+transcriptid+section.
    We keep adjacency within section to avoid mixing prepared remarks and Q&A context.
    """
    out = df.copy()
    out = out.sort_values(
        by=["keydevid", "transcriptid", "section", "block_num"], kind="mergesort"
    ).reset_index(drop=True)
    grp = ["keydevid", "transcriptid", "section"]
    out["prev_componenttext"] = out.groupby(grp)["componenttext"].shift(1).fillna("")
    out["next_componenttext"] = out.groupby(grp)["componenttext"].shift(-1).fillna("")
    return out


def _normalize_company_name(name: str) -> str:
    s = (name or "").strip().lower()
    if not s:
        return ""
    s = s.replace("&", " and ")
    s = re.sub(r"\(class [a-z0-9]+\)", " ", s)
    s = re.sub(r"\b(the|incorporated|inc|corp|corporation|company|co|plc|ltd|llc)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_company_name_from_headline(headline: str) -> str:
    h = (headline or "").strip()
    if not h:
        return ""
    m = re.match(r"^(.*?),\s*Q[1-4]\s+\d{4}\s+Earnings Call\b", h, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return h.split(",", 1)[0].strip()


def _mapping_csv_companyid_column(ticker_ref: pd.DataFrame) -> Optional[str]:
    for c in ticker_ref.columns:
        if c.lower() == "companyid":
            return c
    return None


def _ticker_symbol_series_from_mapping_table(ticker_ref: pd.DataFrame) -> Optional[pd.Series]:
    """Resolve Ticker vs duplicate lowercase `ticker` columns (WRDS export style)."""
    has_lower = "ticker" in ticker_ref.columns
    has_upper = "Ticker" in ticker_ref.columns
    if has_lower and has_upper:
        lo = ticker_ref["ticker"]
        up = ticker_ref["Ticker"]
        lo_ok = (
            lo.notna()
            & lo.astype(str).str.strip().ne("")
            & lo.astype(str).str.strip().str.lower().ne("nan")
        )
        return lo.where(lo_ok, up)
    if has_upper:
        return ticker_ref["Ticker"]
    if has_lower:
        return ticker_ref["ticker"]
    return None


def _headline_ticker_lookup_by_companyid(
    out: pd.DataFrame,
    ticker_ref: pd.DataFrame,
) -> pd.DataFrame:
    """companyid -> ticker using CompanyName/Ticker reference + headline-derived names."""
    ref = ticker_ref[["CompanyName", "Ticker"]].copy()
    ref["Ticker"] = ref["Ticker"].fillna("").astype(str).str.strip().str.upper()
    ref["name_norm"] = ref["CompanyName"].fillna("").astype(str).map(_normalize_company_name)
    ref = ref[(ref["name_norm"] != "") & (ref["Ticker"] != "")].copy()

    ref = ref.sort_values(by=["name_norm", "Ticker"], kind="mergesort")
    name_to_ticker = (
        ref.drop_duplicates(subset=["name_norm"], keep="first")
        .set_index("name_norm")["Ticker"]
        .to_dict()
    )

    company_name_df = out[["companyid", "headline"]].copy()
    company_name_df["company_from_headline"] = company_name_df["headline"].map(
        _extract_company_name_from_headline
    )
    company_name_df["name_norm"] = company_name_df["company_from_headline"].map(
        _normalize_company_name
    )
    company_name_df = company_name_df[company_name_df["name_norm"] != ""]

    mode_name = (
        company_name_df.groupby(["companyid", "name_norm"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(
            by=["companyid", "n", "name_norm"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(subset=["companyid"], keep="first")
    )
    mode_name["ticker"] = mode_name["name_norm"].map(name_to_ticker).fillna("")
    return mode_name[["companyid", "ticker"]]


def add_ticker_column(
    df: pd.DataFrame,
    ticker_map_path: Path,
) -> pd.DataFrame:
    """
    Add ticker column from a local CSV map. Supported layouts:
      - companyid + Ticker (and optional lowercase ticker / companyname), or
      - CompanyName + Ticker with headline heuristics keyed by companyid.
    Falls back to the string form of companyid when no symbol is found.
    """
    out = df.copy()
    if "ticker" in out.columns:
        out["ticker"] = out["ticker"].fillna("").astype(str).str.strip()
        if out["ticker"].ne("").all():
            return out

    if not ticker_map_path.exists():
        LOGGER.warning("Ticker map file not found: %s; ticker will fallback.", ticker_map_path)
        out["ticker"] = out.get("companyid", "").astype(str)
        return out

    ticker_ref = pd.read_csv(ticker_map_path, low_memory=False)
    id_col = _mapping_csv_companyid_column(ticker_ref)
    sym_series = _ticker_symbol_series_from_mapping_table(ticker_ref)
    has_legacy_cols = {"CompanyName", "Ticker"}.issubset(set(ticker_ref.columns))

    if id_col is None and not has_legacy_cols:
        LOGGER.warning(
            "Ticker map has no companyid and no CompanyName/Ticker columns: %s",
            ticker_map_path,
        )
        out["ticker"] = out.get("companyid", "").astype(str)
        return out

    out["ticker"] = ""

    if id_col is not None and sym_series is not None:
        map_df = pd.DataFrame(
            {
                "_map_cid": pd.to_numeric(ticker_ref[id_col], errors="coerce"),
                "_map_ticker": sym_series.astype(str).str.strip().str.upper(),
            }
        )
        map_df = map_df.dropna(subset=["_map_cid"])
        map_df = map_df[
            map_df["_map_ticker"].ne("") & map_df["_map_ticker"].ne("NAN")
        ].copy()
        map_df["_map_cid"] = map_df["_map_cid"].astype(np.int64)
        map_df = map_df.sort_values(by=["_map_cid", "_map_ticker"], kind="mergesort").drop_duplicates(
            subset=["_map_cid"], keep="first"
        )
        out["_map_cid"] = pd.to_numeric(out["companyid"], errors="coerce")
        out = out.merge(map_df, on="_map_cid", how="left")
        out["ticker"] = out["_map_ticker"].fillna("").astype(str).str.strip().str.upper()
        out = out.drop(columns=["_map_ticker", "_map_cid"], errors="ignore")
    elif id_col is not None:
        LOGGER.warning(
            "Ticker map has companyid but no Ticker/ticker column: %s",
            ticker_map_path,
        )

    if has_legacy_cols:
        hl_tbl = _headline_ticker_lookup_by_companyid(out, ticker_ref)
        out = out.merge(hl_tbl.rename(columns={"ticker": "_tick_hl"}), on="companyid", how="left")
        fill = out["_tick_hl"].fillna("").astype(str).str.strip().str.upper()
        out["ticker"] = np.where(out["ticker"].eq(""), fill, out["ticker"])
        out = out.drop(columns=["_tick_hl"], errors="ignore")

    missing = int((out["ticker"] == "").sum())
    if missing > 0:
        LOGGER.warning(
            "Ticker mapping missing for %d rows; fallback ticker will use companyid string.",
            missing,
        )
        fallback = out["companyid"].fillna("").astype(str).str.strip()
        out["ticker"] = np.where(out["ticker"] == "", fallback, out["ticker"])

    return out


def flag_candidate_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Flag candidate AI-related blocks with strong/weak contextual retrieval rules."""
    out = df.copy()
    out["is_candidate"] = out.apply(
        lambda r: bool(r.get("has_text", False))
        and is_candidate_block(
            prev_text=str(r.get("prev_componenttext", "")),
            curr_text=str(r.get("componenttext", "")),
            next_text=str(r.get("next_componenttext", "")),
            headline=str(r.get("headline", "")),
        ),
        axis=1,
    )
    out["candidate_keyword_match"] = out["is_candidate"]
    return out


def build_llm_prompts(
    current_block_text: str,
    metadata: Dict[str, Any],
    prev_block_text: str = "",
    next_block_text: str = "",
) -> Tuple[str, str]:
    """Build system + few-shot + current-example prompt for strict JSON annotation."""
    system_prompt = """You are a careful financial text annotation assistant.

Classify ONE earnings-call transcript block on TWO orthogonal dimensions:
1) AI ADOPTION (user-side operational use)
2) AI INNOVATION (frontier/supply-side capability creation)

Strict conceptual definitions:
- ADOPTION = the firm is using/deploying/integrating AI in real business practice
  (operations, workflows, employee productivity, customer support, sales/marketing,
  decision support, customer workflow execution, enterprise rollout).
- INNOVATION = the firm is building/creating/advancing proprietary AI capability
  (model development, platform/infrastructure creation, AI-native product creation,
  technical frontier advancement, patent/R&D activity, commercialization of internally
  developed AI capability).

Critical disambiguation rules:
1) Internal AI assistants/copilots/summarization/routing/automation default to ADOPTION.
2) Customer-facing AI usage also defaults to ADOPTION unless proprietary build is explicit.
3) Product feature mentions are NOT innovation by default.
4) Innovation requires CREATION evidence, not just USE.
5) Market commentary / AI tailwinds / vague strategy are neither adoption nor innovation.
6) If evidence is vague, keep both scores low (0 or 1).
7) Innovation should be materially rarer than adoption.
8) When uncertain, prefer adoption over innovation.
9) Use surrounding context only to interpret the CURRENT block; classify CURRENT block only.

Return strict JSON only with no markdown and no extra text."""

    dt_list = sorted(DISCUSSION_TYPES)
    as_list = sorted(ADOPTION_STAGES)
    aso_list = sorted(ADOPTION_SCOPES)
    af_list = sorted(ADOPTION_FOCUS_TYPES)
    itn_list = sorted(INNOVATION_TYPES)
    inn_list = sorted(INNOVATION_NOVELTY_TYPES)
    cs_list = sorted(COMMERCIALIZATION_STAGES)
    itf_list = sorted(INNOVATION_TECH_FOCUS_TYPES)
    bf_list = sorted(BUSINESS_FUNCTIONS)
    imp_list = sorted(IMPACT_TYPES)

    user_prompt = f"""
Use this strict JSON schema:
{{
  "is_ai_related": 0 or 1,
  "reasoning_short": short explanation in 1-3 sentences,
  "discussion_type": one of {dt_list},
  "measurable_impact": 0 or 1,
  "management_commitment_signal": 0 or 1,
  "business_function": one of {bf_list},
  "impact_type": array of zero or more of {imp_list},
  "developer_focus": 0 or 1,
  "ai_relevance_confidence": integer 0-100,

  "is_ai_adoption_related": 0 or 1,
  "adoption_stage": one of {as_list},
  "operational_integration": 0 or 1,
  "adoption_scope": one of {aso_list},
  "adoption_focus": one of {af_list},
  "final_adoption_score": integer 0-4,

  "is_ai_innovation_related": 0 or 1,
  "innovation_type": one of {itn_list},
  "innovation_novelty": one of {inn_list},
  "commercialization_stage": one of {cs_list},
  "innovation_output_signal": 0 or 1,
  "innovation_technology_focus": one of {itf_list},
  "proprietary_capability_signal": 0 or 1,
  "internal_model_or_platform_signal": 0 or 1,
  "patent_or_r_and_d_signal": 0 or 1,
  "final_innovation_score": integer 0-4
}}

Adoption score rubric:
0 = no adoption evidence
1 = awareness / vague mention only
2 = evaluation or pilot
3 = clear deployment/integration
4 = scaled integration with measurable outcomes

Innovation score rubric (strict frontier interpretation):
0 = no innovation evidence
1 = weak or very early technical exploration signal
2 = meaningful development/prototype signal
3 = clear proprietary AI capability creation / AI-native launch / technical advancement
4 = strong frontier innovation with proprietary development + commercialization/scaling/differentiated IP

Few-shot calibration examples (compact):
1) "Our customer service teams use an AI assistant to summarize tickets and route cases faster."
   -> adoption high, innovation 0, type=customer_support_automation

2) "We rolled out third-party coding copilots across engineering and support teams."
   -> adoption high, innovation 0, type=internal_productivity

3) "Customers use our AI tools in daily workflow execution."
   -> adoption medium/high, innovation 0 unless proprietary build evidence appears

4) "We trained a domain-specific model for financial document understanding."
   -> innovation high, adoption may be low, type=r_and_d_or_model_development

5) "We built our own inference stack and optimized model serving efficiency."
   -> innovation high, type=infrastructure_compute_data_stack

6) "We launched a proprietary AI-native platform built on our internally developed models."
   -> innovation high; adoption only if rollout/use evidence exists

7) "AI is a major trend and we are excited about future opportunities."
   -> adoption low, innovation low, type=general_market_commentary

8) "We embedded AI summarization into an existing workflow using external tooling."
   -> adoption medium/high, innovation 0

9) "Our R&D teams are filing patents around AI model optimization."
   -> innovation high, patent_or_r_and_d_signal=1

10) "We are seeing strong customer demand from AI infrastructure spending."
   -> neither by default unless firm explicitly builds proprietary AI capability

Innovation hard gate:
- To classify innovation-related (>0), require at least one explicit signal:
  proprietary_capability_signal=1 OR internal_model_or_platform_signal=1 OR
  patent_or_r_and_d_signal=1 OR discussion_type in frontier categories.
- Without that, default innovation to 0.

Metadata:
- section: {metadata.get("section", "")}
- companyid: {metadata.get("companyid", "")}
- fiscal_period: {metadata.get("fiscal_period", "")}
- prompt_version: {PROMPT_VERSION}

Now classify the following CURRENT BLOCK.

Previous block:
{prev_block_text.strip() if prev_block_text.strip() else "EMPTY"}

Current block:
{current_block_text.strip() if current_block_text.strip() else "EMPTY"}

Next block:
{next_block_text.strip() if next_block_text.strip() else "EMPTY"}
""".strip()

    return system_prompt, user_prompt


def call_openrouter(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: Optional[int] = 1,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    seed: Optional[int] = None,
    provider_order: Optional[List[str]] = None,
    allow_fallbacks: bool = False,
    require_provider_parameters: bool = True,
    min_request_interval_s: float = 0.0,
    max_retries: int = 5,
    timeout: int = 45,
    backoff_base: float = 2.0,
) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """
    Call OpenRouter chat completion API with retry and basic failure handling.

    Note: OpenRouter routes across external providers, so strict fixed-weight
    reproducibility cannot be guaranteed. These settings aim to maximize output
    stability: deterministic decoding, fixed provider preference, and no fallbacks.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None, "Missing OPENROUTER_API_KEY", {}
    if not model:
        return None, "Missing OPENROUTER_MODEL", {}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
    }
    # Qwen 3.5 Flash may emit a large reasoning channel by default via OpenRouter,
    # which can cause slow responses/timeouts for short JSON classification tasks.
    if str(model).strip().startswith("qwen/qwen3.5-flash"):
        payload["reasoning"] = {"enabled": False}
        payload["include_reasoning"] = False
    if top_k is not None:
        payload["top_k"] = int(top_k)
    if seed is not None:
        payload["seed"] = int(seed)

    provider_cfg: Dict[str, Any] = {
        "allow_fallbacks": bool(allow_fallbacks),
        "require_parameters": bool(require_provider_parameters),
    }
    if provider_order:
        provider_cfg["order"] = [str(p).strip() for p in provider_order if str(p).strip()]
    if provider_cfg.get("order") or provider_cfg.get("allow_fallbacks") is False:
        payload["provider"] = provider_cfg

    top_k_dropped = False
    for attempt in range(1, max_retries + 1):
        try:
            if min_request_interval_s > 0:
                # Cross-thread request spacing to reduce provider-side 429 risk.
                global _LAST_REQUEST_TS
                with _REQUEST_GUARD_LOCK:
                    now = time.time()
                    wait = min_request_interval_s - (now - _LAST_REQUEST_TS)
                    if wait > 0:
                        time.sleep(wait)
                    _LAST_REQUEST_TS = time.time()
            resp = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=timeout
            )
            status = resp.status_code
            if status == 200:
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if not content:
                    return None, "Empty model content", {}
                meta = {
                    "model_requested": model,
                    "model_returned": data.get("model", model),
                    "openrouter_id": data.get("id", ""),
                    "provider": data.get("provider", ""),
                    "system_fingerprint": data.get("system_fingerprint", ""),
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": payload.get("top_k", None),
                    "max_tokens": int(max_tokens),
                    "seed": seed,
                    "provider_config": payload.get("provider", {}),
                }
                return content, None, meta

            if (
                status in (400, 422)
                and ("top_k" in payload)
                and not top_k_dropped
                and ("top_k" in resp.text.lower() or "unsupported" in resp.text.lower())
            ):
                # Some providers/models reject top_k. Drop it once and continue.
                payload.pop("top_k", None)
                top_k_dropped = True
                LOGGER.warning("Dropping top_k due to provider/model incompatibility.")
                continue

            if status in (408, 409, 425, 429, 500, 502, 503, 504):
                wait_s = backoff_base ** (attempt - 1)
                LOGGER.warning(
                    "OpenRouter retryable status %s (attempt %d/%d). Waiting %.1fs",
                    status,
                    attempt,
                    max_retries,
                    wait_s,
                )
                time.sleep(wait_s)
                continue

            return None, f"Non-retryable status {status}: {resp.text[:500]}", {}

        except requests.RequestException as exc:
            wait_s = backoff_base ** (attempt - 1)
            LOGGER.warning(
                "OpenRouter request error (attempt %d/%d): %s. Waiting %.1fs",
                attempt,
                max_retries,
                exc,
                wait_s,
            )
            time.sleep(wait_s)

    return None, "Exceeded max retries", {}


def _extract_json_candidate(raw_text: str) -> str:
    """Extract likely JSON object from model output."""
    text = raw_text.strip()
    # Remove code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # Fast path
    if text.startswith("{") and text.endswith("}"):
        return text

    # Try to slice from first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _to_binary(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return 1 if int(value) != 0 else 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y"}:
            return 1
        if v in {"0", "false", "no", "n"}:
            return 0
    return default


def _to_int_clamped(value: Any, low: int, high: int, default: int) -> int:
    try:
        iv = int(value)
        return int(np.clip(iv, low, high))
    except Exception:
        return default


def _normalize_list(value: Any, allowed: set[str]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # Attempt simple split for malformed outputs
        parts = re.split(r"[,\|;/]+", value)
        vals = [p.strip() for p in parts if p.strip()]
    elif isinstance(value, list):
        vals = [str(v).strip() for v in value if str(v).strip()]
    else:
        vals = []

    out = []
    for v in vals:
        if v in allowed and v not in out:
            out.append(v)
    return out


def _normalize_single(
    value: Any,
    allowed: set[str],
    default: str,
    compat_map: Optional[Dict[str, str]] = None,
) -> str:
    """Normalize a single enum value with optional backward-compat mapping."""
    if value is None:
        return default
    v = str(value).strip()
    if v in allowed:
        return v
    if compat_map and v in compat_map and compat_map[v] in allowed:
        return compat_map[v]
    return default


def parse_llm_json(raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse and normalize LLM JSON output."""
    try:
        json_candidate = _extract_json_candidate(raw_text)
        data = json.loads(json_candidate)
    except Exception as exc:
        return None, f"JSON parse error: {exc}"

    obj: Dict[str, Any] = {}

    # ------------------------------
    # Shared fields
    # ------------------------------
    obj["is_ai_related"] = _to_binary(data.get("is_ai_related", 0), default=0)
    obj["ai_relevance_confidence"] = _to_int_clamped(
        data.get("ai_relevance_confidence", 0), 0, 100, 0
    )
    obj["discussion_type"] = _normalize_single(
        data.get("discussion_type", "non_ai"),
        DISCUSSION_TYPES,
        "non_ai",
        compat_map=_DISCUSSION_TYPE_COMPAT,
    )
    obj["measurable_impact"] = _to_binary(data.get("measurable_impact", 0), default=0)
    obj["management_commitment_signal"] = _to_binary(
        data.get("management_commitment_signal", 0), default=0
    )
    obj["developer_focus"] = _to_binary(data.get("developer_focus", 0), default=0)

    raw_business_function = data.get("business_function", "unknown")
    if isinstance(raw_business_function, list):
        bf_list = _normalize_list(raw_business_function, BUSINESS_FUNCTIONS)
        obj["business_function"] = bf_list[0] if bf_list else "unknown"
    else:
        obj["business_function"] = _normalize_single(
            raw_business_function, BUSINESS_FUNCTIONS, "unknown"
        )
    obj["impact_type"] = _normalize_list(data.get("impact_type", []), IMPACT_TYPES)

    # ------------------------------
    # Adoption fields
    # ------------------------------
    obj["adoption_stage"] = _normalize_single(
        data.get("adoption_stage", "none"),
        ADOPTION_STAGES,
        "none",
        compat_map=_ADOPTION_STAGE_COMPAT,
    )
    obj["operational_integration"] = _to_binary(
        data.get("operational_integration", 0), default=0
    )
    obj["adoption_scope"] = _normalize_single(
        data.get("adoption_scope", "none"),
        ADOPTION_SCOPES,
        "none",
        compat_map=_ADOPTION_SCOPE_COMPAT,
    )
    obj["adoption_focus"] = _normalize_single(
        data.get("adoption_focus", "none"),
        ADOPTION_FOCUS_TYPES,
        "none",
        compat_map=_ADOPTION_FOCUS_COMPAT,
    )
    obj["final_adoption_score"] = _to_int_clamped(
        data.get("final_adoption_score", 0), 0, 4, 0
    )
    obj["is_ai_adoption_related"] = _to_binary(
        data.get("is_ai_adoption_related", 1 if obj["final_adoption_score"] > 0 else 0),
        default=0,
    )
    if (
        obj["is_ai_adoption_related"] == 0
        and (
            obj["operational_integration"] == 1
            or obj["adoption_stage"] in {"pilot", "deployment", "scaled"}
            or obj["adoption_scope"] != "none"
            or obj["adoption_focus"] != "none"
            or obj["final_adoption_score"] > 0
        )
    ):
        obj["is_ai_adoption_related"] = 1

    # ------------------------------
    # Innovation fields
    # ------------------------------
    obj["innovation_type"] = _normalize_single(
        data.get("innovation_type", "none"),
        INNOVATION_TYPES,
        "none",
        compat_map=_INNOVATION_TYPE_COMPAT,
    )
    if obj["innovation_type"] == "none":
        # Conservative fallback: infer only from explicitly frontier discussion types.
        dt = obj["discussion_type"]
        if dt == "r_and_d_or_model_development":
            obj["innovation_type"] = "model_innovation"
        elif dt == "infrastructure_compute_data_stack":
            obj["innovation_type"] = "infrastructure_innovation"
        elif dt == "proprietary_ai_platform":
            obj["innovation_type"] = "platform_innovation"
        elif dt == "ai_native_product_launch":
            obj["innovation_type"] = "ai_native_product_innovation"
        elif dt in {"technical_capability_creation", "frontier_ai_differentiation"}:
            obj["innovation_type"] = "technical_capability_innovation"
        elif dt == "commercialization_of_proprietary_ai":
            obj["innovation_type"] = "commercialization_of_proprietary_ai"
    obj["innovation_novelty"] = _normalize_single(
        data.get("innovation_novelty", "none"),
        INNOVATION_NOVELTY_TYPES,
        "none",
        compat_map=_INNOVATION_NOVELTY_COMPAT,
    )
    obj["commercialization_stage"] = _normalize_single(
        data.get("commercialization_stage", "none"),
        COMMERCIALIZATION_STAGES,
        "none",
        compat_map=_COMMERCIALIZATION_STAGE_COMPAT,
    )
    obj["innovation_output_signal"] = _to_binary(
        data.get("innovation_output_signal", 0), default=0
    )
    obj["innovation_technology_focus"] = _normalize_single(
        data.get("innovation_technology_focus", "none"),
        INNOVATION_TECH_FOCUS_TYPES,
        "none",
        compat_map=_INNOVATION_TECH_FOCUS_COMPAT,
    )
    obj["proprietary_capability_signal"] = _to_binary(
        data.get("proprietary_capability_signal", 0), default=0
    )
    obj["internal_model_or_platform_signal"] = _to_binary(
        data.get("internal_model_or_platform_signal", 0), default=0
    )
    obj["patent_or_r_and_d_signal"] = _to_binary(
        data.get("patent_or_r_and_d_signal", 0), default=0
    )
    obj["final_innovation_score"] = _to_int_clamped(
        data.get("final_innovation_score", 0), 0, 4, 0
    )

    innovation_gate_pass = int(
        obj["proprietary_capability_signal"] == 1
        or obj["internal_model_or_platform_signal"] == 1
        or obj["patent_or_r_and_d_signal"] == 1
        or obj["discussion_type"] in FRONTIER_DISCUSSION_TYPES
        or obj["innovation_type"] in FRONTIER_INNOVATION_TYPES
    )
    obj["innovation_gate_pass"] = innovation_gate_pass

    inferred_innovation_related = 1 if (
        innovation_gate_pass == 1
        and (
            obj["final_innovation_score"] > 0
            or obj["innovation_type"] in FRONTIER_INNOVATION_TYPES
            or obj["discussion_type"] in FRONTIER_DISCUSSION_TYPES
            or obj["innovation_output_signal"] == 1
        )
    ) else 0
    obj["is_ai_innovation_related"] = _to_binary(
        data.get("is_ai_innovation_related", inferred_innovation_related), default=0
    )
    if innovation_gate_pass == 0:
        obj["is_ai_innovation_related"] = 0
        obj["final_innovation_score"] = 0
        obj["innovation_type"] = "none"
        obj["innovation_novelty"] = "none"
        obj["commercialization_stage"] = "none"
        obj["innovation_technology_focus"] = "none"
        obj["innovation_output_signal"] = 0

    # Guardrails to keep fields internally consistent.
    if obj["is_ai_related"] == 0:
        obj["is_ai_adoption_related"] = 0
        obj["is_ai_innovation_related"] = 0
        obj["final_adoption_score"] = 0
        obj["final_innovation_score"] = 0
    else:
        if obj["final_adoption_score"] > 0:
            obj["is_ai_adoption_related"] = 1
        if obj["final_innovation_score"] > 0 and obj["innovation_gate_pass"] == 1:
            obj["is_ai_innovation_related"] = 1
        if obj["innovation_gate_pass"] == 0:
            obj["is_ai_innovation_related"] = 0

    # Backward-compatible aliases expected by some downstream code.
    obj["business_function_list"] = [obj["business_function"]] if obj["business_function"] else []

    reasoning = str(data.get("reasoning_short", "")).strip()
    obj["reasoning_short"] = reasoning[:1000]
    return obj, None


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8", errors="ignore")).hexdigest()


def build_context_hash(
    prev_text: str,
    curr_text: str,
    next_text: str,
    model: str,
    prompt_version: str,
    decoding_config: Optional[Dict[str, Any]] = None,
    provider_config: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "prev_text": (prev_text or "").strip(),
        "curr_text": (curr_text or "").strip(),
        "next_text": (next_text or "").strip(),
        "model": (model or "").strip(),
        "prompt_version": (prompt_version or "").strip(),
        "decoding_config": decoding_config or {},
        "provider_config": provider_config or {},
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


def load_cache(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load cache from jsonl where each line has key, text_hash, parsed, raw_response."""
    cache: Dict[str, Dict[str, Any]] = {}
    if not cache_path.exists():
        return cache

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = str(rec.get("key", "")).strip()
                if key:
                    cache[key] = rec
            except Exception:
                continue
    LOGGER.info("Loaded %d cache records from %s", len(cache), cache_path)
    return cache


def append_cache_record(cache_path: Path, record: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def annotate_blocks_with_cache(
    candidates_df: pd.DataFrame,
    cache_path: Path,
    model: str,
    save_raw_response: bool = True,
    max_workers: int = 6,
    request_timeout: int = 45,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: Optional[int] = 1,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    seed: Optional[int] = None,
    provider_order: Optional[List[str]] = None,
    allow_fallbacks: bool = False,
    require_provider_parameters: bool = True,
    min_request_interval_s: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Classify candidate blocks with OpenRouter using cache + parallel requests."""
    cache = load_cache(cache_path)

    annotations: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    rows = candidates_df.to_dict(orient="records")

    def _annotate_single(row: Dict[str, Any]) -> Dict[str, Any]:
        block_id = str(row.get("Id", "")).strip()
        block_text = str(row.get("componenttext", "")).strip()
        prev_text = str(row.get("prev_componenttext", "")).strip()
        next_text = str(row.get("next_componenttext", "")).strip()
        if not block_text:
            return {
                "ok": False,
                "failure": {
                    "Id": block_id,
                    "keydevid": row.get("keydevid", ""),
                    "transcriptid": row.get("transcriptid", ""),
                    "error": "Empty componenttext",
                    "raw_response": "",
                },
            }

        text_hash = _text_hash(block_text)
        context_hash = build_context_hash(
            prev_text=prev_text,
            curr_text=block_text,
            next_text=next_text,
            model=model,
            prompt_version=PROMPT_VERSION,
            decoding_config={
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_tokens,
                "seed": seed,
            },
            provider_config={
                "order": provider_order or [],
                "allow_fallbacks": allow_fallbacks,
                "require_parameters": require_provider_parameters,
            },
        )
        cache_key = f"{block_id}::{context_hash}"

        # exact cache hit
        if cache_key in cache:
            parsed = cache[cache_key].get("parsed", None)
            raw_response = cache[cache_key].get("raw_response", "")
            ann_row = dict(row)
            ann_row.update(parsed or {})
            ann_row["text_hash"] = text_hash
            ann_row["context_hash"] = context_hash
            ann_row["prompt_version"] = PROMPT_VERSION
            ann_row["llm_model"] = cache[cache_key].get("model", model)
            ann_row["llm_model_returned"] = cache[cache_key].get("model_returned", "")
            ann_row["llm_provider"] = cache[cache_key].get("provider", "")
            ann_row["llm_system_fingerprint"] = cache[cache_key].get("system_fingerprint", "")
            ann_row["llm_seed"] = cache[cache_key].get("seed", seed)
            ann_row["llm_temperature"] = cache[cache_key].get("temperature", temperature)
            ann_row["llm_top_p"] = cache[cache_key].get("top_p", top_p)
            ann_row["llm_top_k"] = cache[cache_key].get("top_k", top_k)
            ann_row["llm_max_tokens"] = cache[cache_key].get("max_tokens", max_tokens)
            ann_row["llm_provider_config"] = json.dumps(
                cache[cache_key].get("provider_config", {}), ensure_ascii=False, sort_keys=True
            )
            ann_row["llm_status"] = "cache_hit"
            ann_row["raw_llm_response"] = raw_response if save_raw_response else ""
            return {"ok": True, "annotation": ann_row, "cache_record": None}

        system_prompt, user_prompt = build_llm_prompts(
            current_block_text=block_text,
            metadata={
                "section": row.get("section", ""),
                "companyid": row.get("companyid", ""),
                "fiscal_period": row.get("fiscal_period", ""),
            },
            prev_block_text=prev_text,
            next_block_text=next_text,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw, call_err, llm_meta = call_openrouter(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=seed,
            provider_order=provider_order,
            allow_fallbacks=allow_fallbacks,
            require_provider_parameters=require_provider_parameters,
            min_request_interval_s=min_request_interval_s,
            timeout=request_timeout,
        )
        if call_err:
            return {
                "ok": False,
                "failure": {
                    "Id": block_id,
                    "keydevid": row.get("keydevid", ""),
                    "transcriptid": row.get("transcriptid", ""),
                    "error": call_err,
                    "raw_response": raw or "",
                },
            }

        raw_response = raw or ""
        parsed, parse_err = parse_llm_json(raw_response)
        if parse_err or parsed is None:
            return {
                "ok": False,
                "failure": {
                    "Id": block_id,
                    "keydevid": row.get("keydevid", ""),
                    "transcriptid": row.get("transcriptid", ""),
                    "error": parse_err or "Unknown parse error",
                    "raw_response": raw_response,
                },
            }

        ann_row = dict(row)
        ann_row.update(parsed)
        ann_row["text_hash"] = text_hash
        ann_row["context_hash"] = context_hash
        ann_row["prompt_version"] = PROMPT_VERSION
        ann_row["llm_model"] = llm_meta.get("model_requested", model)
        ann_row["llm_model_returned"] = llm_meta.get("model_returned", "")
        ann_row["llm_provider"] = llm_meta.get("provider", "")
        ann_row["llm_system_fingerprint"] = llm_meta.get("system_fingerprint", "")
        ann_row["llm_seed"] = llm_meta.get("seed", seed)
        ann_row["llm_temperature"] = llm_meta.get("temperature", temperature)
        ann_row["llm_top_p"] = llm_meta.get("top_p", top_p)
        ann_row["llm_top_k"] = llm_meta.get("top_k", top_k)
        ann_row["llm_max_tokens"] = llm_meta.get("max_tokens", max_tokens)
        ann_row["llm_provider_config"] = json.dumps(
            llm_meta.get("provider_config", {}), ensure_ascii=False, sort_keys=True
        )
        ann_row["llm_status"] = "called"
        ann_row["raw_llm_response"] = raw_response if save_raw_response else ""

        cache_record = {
            "key": cache_key,
            "Id": block_id,
            "text_hash": text_hash,
            "context_hash": context_hash,
            "parsed": parsed,
            "raw_response": raw_response if save_raw_response else "",
            "model": model,
            "model_returned": llm_meta.get("model_returned", ""),
            "provider": llm_meta.get("provider", ""),
            "system_fingerprint": llm_meta.get("system_fingerprint", ""),
            "temperature": llm_meta.get("temperature", temperature),
            "top_p": llm_meta.get("top_p", top_p),
            "top_k": llm_meta.get("top_k", top_k),
            "max_tokens": llm_meta.get("max_tokens", max_tokens),
            "seed": llm_meta.get("seed", seed),
            "provider_config": llm_meta.get("provider_config", {}),
            "prompt_version": PROMPT_VERSION,
            "cached_at_unix": int(time.time()),
        }
        return {"ok": True, "annotation": ann_row, "cache_record": cache_record}

    # Parallel execution for uncached rows; cached rows are also handled in the same path.
    total = len(rows)
    max_workers = max(1, int(max_workers))
    LOGGER.info("Annotating %d candidate blocks with max_workers=%d", total, max_workers)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_annotate_single, row) for row in rows]
        iterator = concurrent.futures.as_completed(futures)
        if tqdm is not None:
            iterator = tqdm(iterator, total=total, desc="Classifying blocks")

        for future in iterator:
            result = future.result()
            if result.get("ok"):
                annotations.append(result["annotation"])
                cache_record = result.get("cache_record")
                if cache_record is not None:
                    append_cache_record(cache_path=cache_path, record=cache_record)
            else:
                failures.append(result["failure"])

    annotations_df = pd.DataFrame(annotations)
    failures_df = pd.DataFrame(failures)
    return annotations_df, failures_df


def compute_adjusted_block_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute dual-dimension adjusted block scores.

    Why two scores:
    - Adoption emphasizes operational integration and deployment depth.
    - Innovation emphasizes novelty, capability creation, and commercialization.
    The same block can score high on one dimension and low on the other.
    """
    out = df.copy()
    section_norm = out["section"].fillna("").astype(str).str.strip().str.lower()
    section_weight = np.where(section_norm.eq("pre"), 1.2, 1.0)

    # Adoption = operational deployment/use. Reward deployment/scope; do not let
    # developer-heavy technical discussion inflate adoption by itself.
    adoption_stage_bonus = out["adoption_stage"].isin(
        ["deployment", "scaled"]
    ).astype(float) * 0.35
    adoption_scope_bonus = out["adoption_scope"].isin(
        ["business_unit", "enterprise_wide", "customer_embedded"]
    ).astype(float) * 0.25
    operational_bonus = out["operational_integration"].fillna(0).astype(float) * 0.35
    adoption_raw = section_weight * (
        out["final_adoption_score"].fillna(0).astype(float)
        + adoption_stage_bonus
        + adoption_scope_bonus
        + operational_bonus
        - 0.65 * out["developer_focus"].fillna(0).astype(float)
    )

    # Innovation = proprietary frontier capability creation. Strongly reward
    # explicit proprietary/model/platform/R&D signals and keep a hard gate.
    novelty_bonus = out["innovation_novelty"].isin(
        ["meaningful_extension", "new_to_firm", "new_to_market"]
    ).astype(float) * 0.25
    commercialization_bonus = out["commercialization_stage"].isin(
        ["commercial_launch", "scaled_revenue"]
    ).astype(float) * 0.45
    innovation_output_bonus = out["innovation_output_signal"].fillna(0).astype(float) * 0.25
    proprietary_bonus = out.get("proprietary_capability_signal", 0)
    proprietary_bonus = pd.Series(proprietary_bonus, index=out.index).fillna(0).astype(float) * 0.8
    model_platform_bonus = out.get("internal_model_or_platform_signal", 0)
    model_platform_bonus = pd.Series(model_platform_bonus, index=out.index).fillna(0).astype(float) * 0.7
    patent_bonus = out.get("patent_or_r_and_d_signal", 0)
    patent_bonus = pd.Series(patent_bonus, index=out.index).fillna(0).astype(float) * 0.6
    innovation_gate = out.get("innovation_gate_pass", 0)
    innovation_gate = pd.Series(innovation_gate, index=out.index).fillna(0).astype(float)

    innovation_raw = section_weight * (
        out["final_innovation_score"].fillna(0).astype(float)
        + novelty_bonus
        + commercialization_bonus
        + innovation_output_bonus
        + proprietary_bonus
        + model_platform_bonus
        + patent_bonus
    )
    innovation_raw = innovation_raw * innovation_gate

    out["section_weight"] = section_weight
    out["adjusted_adoption_block_score"] = np.maximum(adoption_raw, 0.0)
    out["adjusted_innovation_block_score"] = np.maximum(innovation_raw, 0.0)

    # Backward-compat alias for legacy downstream consumers.
    out["adjusted_block_score"] = out["adjusted_adoption_block_score"]

    out["is_high_adoption_block"] = (out["final_adoption_score"] >= 3).astype(int)
    out["is_high_innovation_block"] = (
        (out["final_innovation_score"] >= 3)
        & (out.get("innovation_gate_pass", 0).fillna(0).astype(int) == 1)
    ).astype(int)
    return out


def _aggregate_metrics(
    base_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    group_cols: List[str],
) -> pd.DataFrame:
    """
    Aggregate block annotations to call/company-quarter metrics.

    Shared retrieval is used upstream; this function reports separate adoption
    and innovation aggregates so downstream analysis can compare both dimensions.
    """
    base_group = (
        base_df.groupby(group_cols, dropna=False)
        .agg(
            total_blocks=("Id", "count"),
            total_words=("word_count", "sum"),
            candidate_blocks=("is_candidate", "sum"),
        )
        .reset_index()
    )
    out = base_group.copy()

    if ann_df.empty:
        for c, default in [
            ("ai_related_blocks", 0),
            ("ai_adoption_related_blocks", 0),
            ("ai_innovation_related_blocks", 0),
            ("ai_adoption_score_sum", 0.0),
            ("ai_innovation_score_sum", 0.0),
            ("ai_adoption_score_mean_candidates", 0.0),
            ("ai_innovation_score_mean_candidates", 0.0),
            ("ai_adoption_score_mean_ai_related", 0.0),
            ("ai_innovation_score_mean_ai_related", 0.0),
            ("high_adoption_ai_related_blocks", 0),
            ("high_innovation_ai_related_blocks", 0),
            ("measurable_impact_adoption_blocks", 0),
            ("measurable_impact_innovation_blocks", 0),
            ("product_innovation_blocks", 0),
            ("process_innovation_blocks", 0),
            ("technology_innovation_blocks", 0),
        ]:
            out[c] = default
    else:
        tmp = ann_df.copy()
        tmp["is_ai_related"] = tmp["is_ai_related"].fillna(0).astype(int)
        tmp["is_ai_adoption_related"] = tmp["is_ai_adoption_related"].fillna(0).astype(int)
        tmp["is_ai_innovation_related"] = tmp["is_ai_innovation_related"].fillna(0).astype(int)
        tmp["measurable_impact"] = tmp["measurable_impact"].fillna(0).astype(int)
        tmp["final_adoption_score"] = tmp["final_adoption_score"].fillna(0).astype(float)
        tmp["final_innovation_score"] = tmp["final_innovation_score"].fillna(0).astype(float)
        tmp["adjusted_adoption_block_score"] = tmp["adjusted_adoption_block_score"].fillna(0).astype(float)
        tmp["adjusted_innovation_block_score"] = tmp["adjusted_innovation_block_score"].fillna(0).astype(float)

        tmp["is_high_adoption_ai_related"] = (
            (tmp["is_ai_adoption_related"] == 1) & (tmp["final_adoption_score"] >= 3)
        ).astype(int)
        tmp["is_high_innovation_ai_related"] = (
            (tmp["is_ai_innovation_related"] == 1) & (tmp["final_innovation_score"] >= 3)
        ).astype(int)
        tmp["is_measurable_impact_adoption"] = (
            (tmp["is_ai_adoption_related"] == 1) & (tmp["measurable_impact"] == 1)
        ).astype(int)
        tmp["is_measurable_impact_innovation"] = (
            (tmp["is_ai_innovation_related"] == 1) & (tmp["measurable_impact"] == 1)
        ).astype(int)
        # Keep legacy aggregate column names stable while mapping to stricter
        # frontier innovation categories.
        tmp["is_product_innovation"] = (
            (tmp["is_ai_innovation_related"] == 1)
            & (tmp["innovation_type"] == "ai_native_product_innovation")
        ).astype(int)
        tmp["is_process_innovation"] = (
            (tmp["is_ai_innovation_related"] == 1)
            & (tmp["innovation_type"] == "platform_innovation")
        ).astype(int)
        tmp["is_technology_innovation"] = (
            (tmp["is_ai_innovation_related"] == 1)
            & (
                tmp["innovation_type"].isin(
                    {
                        "model_innovation",
                        "infrastructure_innovation",
                        "technical_capability_innovation",
                        "commercialization_of_proprietary_ai",
                    }
                )
            )
        ).astype(int)

        cand_group = (
            tmp.groupby(group_cols, dropna=False)
            .agg(
                ai_related_blocks=("is_ai_related", "sum"),
                ai_adoption_related_blocks=("is_ai_adoption_related", "sum"),
                ai_innovation_related_blocks=("is_ai_innovation_related", "sum"),
                ai_adoption_score_sum=("adjusted_adoption_block_score", "sum"),
                ai_innovation_score_sum=("adjusted_innovation_block_score", "sum"),
                ai_adoption_score_mean_candidates=("adjusted_adoption_block_score", "mean"),
                ai_innovation_score_mean_candidates=("adjusted_innovation_block_score", "mean"),
                high_adoption_ai_related_blocks=("is_high_adoption_ai_related", "sum"),
                high_innovation_ai_related_blocks=("is_high_innovation_ai_related", "sum"),
                measurable_impact_adoption_blocks=("is_measurable_impact_adoption", "sum"),
                measurable_impact_innovation_blocks=("is_measurable_impact_innovation", "sum"),
                product_innovation_blocks=("is_product_innovation", "sum"),
                process_innovation_blocks=("is_process_innovation", "sum"),
                technology_innovation_blocks=("is_technology_innovation", "sum"),
            )
            .reset_index()
        )

        adopt_rel = tmp[tmp["is_ai_adoption_related"] == 1]
        innov_rel = tmp[tmp["is_ai_innovation_related"] == 1]
        adopt_mean = (
            adopt_rel.groupby(group_cols, dropna=False)["adjusted_adoption_block_score"]
            .mean()
            .reset_index(name="ai_adoption_score_mean_ai_related")
            if not adopt_rel.empty
            else pd.DataFrame(columns=group_cols + ["ai_adoption_score_mean_ai_related"])
        )
        innov_mean = (
            innov_rel.groupby(group_cols, dropna=False)["adjusted_innovation_block_score"]
            .mean()
            .reset_index(name="ai_innovation_score_mean_ai_related")
            if not innov_rel.empty
            else pd.DataFrame(columns=group_cols + ["ai_innovation_score_mean_ai_related"])
        )

        out = out.merge(cand_group, on=group_cols, how="left")
        out = out.merge(adopt_mean, on=group_cols, how="left")
        out = out.merge(innov_mean, on=group_cols, how="left")

        fill_zero_cols = [
            "ai_related_blocks",
            "ai_adoption_related_blocks",
            "ai_innovation_related_blocks",
            "ai_adoption_score_sum",
            "ai_innovation_score_sum",
            "ai_adoption_score_mean_candidates",
            "ai_innovation_score_mean_candidates",
            "ai_adoption_score_mean_ai_related",
            "ai_innovation_score_mean_ai_related",
            "high_adoption_ai_related_blocks",
            "high_innovation_ai_related_blocks",
            "measurable_impact_adoption_blocks",
            "measurable_impact_innovation_blocks",
            "product_innovation_blocks",
            "process_innovation_blocks",
            "technology_innovation_blocks",
        ]
        for c in fill_zero_cols:
            out[c] = out[c].fillna(0)

    # Shared ratios
    out["share_candidate_blocks"] = np.where(
        out["total_blocks"] > 0, out["candidate_blocks"] / out["total_blocks"], 0.0
    )
    out["share_ai_related_blocks"] = np.where(
        out["total_blocks"] > 0, out["ai_related_blocks"] / out["total_blocks"], 0.0
    )

    # Adoption metrics
    out["ai_adoption_score_sum_norm_blocks"] = np.where(
        out["total_blocks"] > 0, out["ai_adoption_score_sum"] / out["total_blocks"], 0.0
    )
    out["ai_adoption_score_sum_norm_words"] = np.where(
        out["total_words"] > 0, out["ai_adoption_score_sum"] / out["total_words"] * 1000, 0.0
    )
    out["share_ai_adoption_related_blocks"] = np.where(
        out["total_blocks"] > 0, out["ai_adoption_related_blocks"] / out["total_blocks"], 0.0
    )
    out["share_high_adoption_blocks"] = np.where(
        out["ai_adoption_related_blocks"] > 0,
        out["high_adoption_ai_related_blocks"] / out["ai_adoption_related_blocks"],
        0.0,
    )
    out["share_measurable_impact_adoption_blocks"] = np.where(
        out["ai_adoption_related_blocks"] > 0,
        out["measurable_impact_adoption_blocks"] / out["ai_adoption_related_blocks"],
        0.0,
    )

    # Innovation metrics
    out["ai_innovation_score_sum_norm_blocks"] = np.where(
        out["total_blocks"] > 0, out["ai_innovation_score_sum"] / out["total_blocks"], 0.0
    )
    out["ai_innovation_score_sum_norm_words"] = np.where(
        out["total_words"] > 0, out["ai_innovation_score_sum"] / out["total_words"] * 1000, 0.0
    )
    out["share_ai_innovation_related_blocks"] = np.where(
        out["total_blocks"] > 0, out["ai_innovation_related_blocks"] / out["total_blocks"], 0.0
    )
    out["share_high_innovation_blocks"] = np.where(
        out["ai_innovation_related_blocks"] > 0,
        out["high_innovation_ai_related_blocks"] / out["ai_innovation_related_blocks"],
        0.0,
    )
    out["share_measurable_impact_innovation_blocks"] = np.where(
        out["ai_innovation_related_blocks"] > 0,
        out["measurable_impact_innovation_blocks"] / out["ai_innovation_related_blocks"],
        0.0,
    )
    out["share_product_innovation_blocks"] = np.where(
        out["ai_innovation_related_blocks"] > 0,
        out["product_innovation_blocks"] / out["ai_innovation_related_blocks"],
        0.0,
    )
    out["share_process_innovation_blocks"] = np.where(
        out["ai_innovation_related_blocks"] > 0,
        out["process_innovation_blocks"] / out["ai_innovation_related_blocks"],
        0.0,
    )
    out["share_technology_innovation_blocks"] = np.where(
        out["ai_innovation_related_blocks"] > 0,
        out["technology_innovation_blocks"] / out["ai_innovation_related_blocks"],
        0.0,
    )

    # Backward-compatible aliases
    out["sum_adjusted_score"] = out["ai_adoption_score_sum"]
    out["ai_adoption_score_mean"] = out["ai_adoption_score_mean_ai_related"]
    out["ai_related_candidate_blocks"] = out["ai_related_blocks"]
    out["high_adoption_candidate_blocks"] = out["high_adoption_ai_related_blocks"]
    out["measurable_impact_candidate_blocks"] = out["measurable_impact_adoption_blocks"]
    return out


def aggregate_call_level(base_df: pd.DataFrame, ann_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate block annotations into transcript/call-level metrics."""
    group_cols = ["keydevid", "transcriptid", "ticker", "year", "quarter", "fiscal_period"]
    return _aggregate_metrics(base_df=base_df, ann_df=ann_df, group_cols=group_cols)


def aggregate_company_quarter(base_df: pd.DataFrame, ann_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate into company-quarter metrics."""
    group_cols = ["ticker", "year", "quarter", "fiscal_period"]
    return _aggregate_metrics(base_df=base_df, ann_df=ann_df, group_cols=group_cols)


def save_outputs(
    output_dir: Path,
    candidate_df: pd.DataFrame,
    block_ann_df: pd.DataFrame,
    call_df: pd.DataFrame,
    company_q_df: pd.DataFrame,
    failures_df: pd.DataFrame,
) -> None:
    """Save all required CSV outputs."""
    def _ticker_first(df: pd.DataFrame) -> pd.DataFrame:
        if "ticker" not in df.columns:
            return df
        cols = ["ticker"] + [c for c in df.columns if c != "ticker"]
        return df[cols].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    _ticker_first(candidate_df).to_csv(output_dir / "candidate_blocks.csv", index=False)
    _ticker_first(block_ann_df).to_csv(output_dir / "block_level_ai_annotations.csv", index=False)
    _ticker_first(call_df).to_csv(output_dir / "call_level_ai_scores.csv", index=False)
    _ticker_first(company_q_df).to_csv(output_dir / "company_quarter_ai_scores.csv", index=False)
    failures_df.to_csv(output_dir / "llm_failures.csv", index=False)

    LOGGER.info("Saved outputs to %s", output_dir)


def _validation_summary(
    full_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    company_q_df: pd.DataFrame,
) -> None:
    total_blocks = len(full_df)
    candidate_blocks = len(candidate_df)
    candidate_rate = candidate_blocks / total_blocks if total_blocks else 0.0
    success_blocks = len(ann_df)
    failure_blocks = len(failures_df)
    unique_cq = len(company_q_df)
    ai_related_blocks = (
        int(ann_df["is_ai_related"].fillna(0).astype(int).sum())
        if ("is_ai_related" in ann_df.columns and not ann_df.empty)
        else 0
    )
    adoption_related_blocks = (
        int(ann_df["is_ai_adoption_related"].fillna(0).astype(int).sum())
        if ("is_ai_adoption_related" in ann_df.columns and not ann_df.empty)
        else 0
    )
    innovation_related_blocks = (
        int(ann_df["is_ai_innovation_related"].fillna(0).astype(int).sum())
        if ("is_ai_innovation_related" in ann_df.columns and not ann_df.empty)
        else 0
    )
    share_adoption = (
        adoption_related_blocks / success_blocks if success_blocks > 0 else 0.0
    )
    share_innovation = (
        innovation_related_blocks / success_blocks if success_blocks > 0 else 0.0
    )
    innovation_to_adoption_ratio = (
        innovation_related_blocks / adoption_related_blocks
        if adoption_related_blocks > 0
        else 0.0
    )
    avg_adoption_score = (
        float(ann_df["final_adoption_score"].fillna(0).astype(float).mean())
        if ("final_adoption_score" in ann_df.columns and not ann_df.empty)
        else 0.0
    )
    avg_innovation_score = (
        float(ann_df["final_innovation_score"].fillna(0).astype(float).mean())
        if ("final_innovation_score" in ann_df.columns and not ann_df.empty)
        else 0.0
    )

    cq_corr = 0.0
    if (
        not company_q_df.empty
        and "ai_adoption_score_sum_norm_blocks" in company_q_df.columns
        and "ai_innovation_score_sum_norm_blocks" in company_q_df.columns
    ):
        corr_df = company_q_df[
            ["ai_adoption_score_sum_norm_blocks", "ai_innovation_score_sum_norm_blocks"]
        ].copy()
        if len(corr_df) >= 2:
            cq_corr = float(
                corr_df["ai_adoption_score_sum_norm_blocks"].corr(
                    corr_df["ai_innovation_score_sum_norm_blocks"]
                )
                or 0.0
            )

    print("\nValidation summary")
    print("------------------")
    print(f"total_blocks: {total_blocks}")
    print(f"candidate_blocks: {candidate_blocks}")
    print(f"candidate_rate: {candidate_rate:.4f}")
    print(f"successfully_classified_blocks: {success_blocks}")
    print(f"ai_related_blocks: {ai_related_blocks}")
    print(f"adoption_related_blocks: {adoption_related_blocks}")
    print(f"innovation_related_blocks: {innovation_related_blocks}")
    print(f"share_adoption_related_in_candidates: {share_adoption:.4f}")
    print(f"share_innovation_related_in_candidates: {share_innovation:.4f}")
    print(f"innovation_to_adoption_ratio: {innovation_to_adoption_ratio:.4f}")
    print(f"avg_final_adoption_score: {avg_adoption_score:.4f}")
    print(f"avg_final_innovation_score: {avg_innovation_score:.4f}")
    print(f"corr_cq_adoption_vs_innovation_norm_blocks: {cq_corr:.4f}")
    print(f"failures: {failure_blocks}")
    print(f"unique_company_quarters_scored: {unique_cq}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI adoption scoring pipeline via OpenRouter.")
    parser.add_argument(
        "--input",
        type=str,
        default="transcript_2016_2023_finalversion.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save output CSV files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
        help="OpenRouter model name",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="outputs/llm_cache.jsonl",
        help="JSONL cache file path for LLM annotations",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--save-raw-response",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save raw LLM response text in annotations/cache (use --no-save-raw-response to disable)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Parallel workers for LLM API calls",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=45,
        help="HTTP timeout (seconds) per OpenRouter request",
    )
    parser.add_argument(
        "--min-request-interval",
        type=float,
        default=float(os.getenv("OPENROUTER_MIN_REQUEST_INTERVAL", "0")),
        help="Minimum seconds between OpenRouter requests (global, cross-thread)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("OPENROUTER_SEED", "42")),
        help="Seed for supported models/providers to improve reproducibility",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=os.getenv("OPENROUTER_PROVIDER", "").strip(),
        help="Preferred OpenRouter provider name (optional)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Fixed max_tokens for JSON classification task",
    )
    parser.add_argument(
        "--allow-fallbacks",
        action="store_true",
        help="Allow provider fallbacks (default off for stability)",
    )
    parser.add_argument(
        "--no-require-provider-parameters",
        action="store_true",
        help="Disable provider parameter requirement (default on for stability)",
    )
    parser.add_argument(
        "--ticker-map-path",
        type=str,
        default=DEFAULT_TICKER_MAP_CSV,
        help="CSV: companyid+Ticker (direct), and/or CompanyName+Ticker (headline fallback)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    input_csv = Path(args.input)
    output_dir = Path(args.output_dir)
    cache_path = Path(args.cache_path)
    ticker_map_path = Path(args.ticker_map_path)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    model = args.model.strip()
    if not model:
        raise EnvironmentError("Model must be provided via --model or OPENROUTER_MODEL.")

    provider_order = [args.provider] if args.provider else []
    require_provider_parameters = not args.no_require_provider_parameters

    df = load_data(input_csv)
    df = extract_time_features(df)
    df = add_ticker_column(df, ticker_map_path=ticker_map_path)
    df = add_local_context_columns(df)
    df = flag_candidate_blocks(df)

    # Candidate subset
    candidate_df = df[df["is_candidate"]].copy()
    if candidate_df.empty:
        LOGGER.warning("No candidate blocks found after keyword filtering.")
        ann_df = pd.DataFrame(columns=df.columns.tolist())
        failures_df = pd.DataFrame(columns=["Id", "keydevid", "transcriptid", "error", "raw_response"])
    else:
        ann_df, failures_df = annotate_blocks_with_cache(
            candidates_df=candidate_df,
            cache_path=cache_path,
            model=model,
            save_raw_response=args.save_raw_response,
            max_workers=args.max_workers,
            request_timeout=args.request_timeout,
            # OpenRouter cannot guarantee strict fixed-weight reproducibility.
            # These settings maximize practical stability for repeated runs.
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            max_tokens=int(args.max_tokens),
            seed=args.seed,
            provider_order=provider_order,
            allow_fallbacks=bool(args.allow_fallbacks),
            require_provider_parameters=require_provider_parameters,
            min_request_interval_s=max(0.0, float(args.min_request_interval)),
        )

    if not ann_df.empty:
        ann_df = compute_adjusted_block_score(ann_df)
    else:
        # ensure expected columns exist when empty
        ann_df = ann_df.copy()
        for c, default in [
            ("section_weight", 1.0),
            ("adjusted_block_score", 0.0),
            ("is_high_adoption_block", 0),
        ]:
            ann_df[c] = default

    call_df = aggregate_call_level(base_df=df, ann_df=ann_df)
    company_q_df = aggregate_company_quarter(base_df=df, ann_df=ann_df)

    if "companyid" in call_df.columns:
        call_df = call_df.drop(columns=["companyid"])
    if "companyid" in company_q_df.columns:
        company_q_df = company_q_df.drop(columns=["companyid"])

    # Keep only valid period rows for final company-quarter score output
    company_q_df = company_q_df[company_q_df["fiscal_period"].astype(str).str.len() > 0].copy()

    save_outputs(
        output_dir=output_dir,
        candidate_df=candidate_df,
        block_ann_df=ann_df,
        call_df=call_df,
        company_q_df=company_q_df,
        failures_df=failures_df,
    )
    _validation_summary(
        full_df=df,
        candidate_df=candidate_df,
        ann_df=ann_df,
        failures_df=failures_df,
        company_q_df=company_q_df,
    )


if __name__ == "__main__":
    main()
