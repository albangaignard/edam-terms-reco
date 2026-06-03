from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Awaitable, Callable

import pandas as pd


RetrieveAllFn = Callable[[str], Awaitable[pd.DataFrame]]
RetrieveClassesFn = Callable[..., Awaitable[pd.DataFrame]]


@dataclass
class StrategyResult:
    """Container for per-field EDAM mappings and their merged view."""

    topics: pd.DataFrame
    operations: pd.DataFrame
    data: pd.DataFrame
    formats: pd.DataFrame
    merged: pd.DataFrame


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["Preferred Label", "Similarity", "child_terms", "Class ID"]
    )


def _merge_dataframes(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty_dataframes = [df for df in dataframes if not df.empty]
    if not non_empty_dataframes:
        return _empty_df()
    return pd.concat(non_empty_dataframes, ignore_index=True)


async def strategy_v1(user_text: str, retrieve_all: RetrieveAllFn) -> StrategyResult:
    """Map user text against all EDAM classes (V1 retriever-only strategy)."""
    merged = await retrieve_all(user_text)
    return StrategyResult(
        topics=_empty_df(),
        operations=_empty_df(),
        data=_empty_df(),
        formats=_empty_df(),
        merged=merged,
    )


async def strategy_v2_or_v3(
    generated_terms: dict[str, Any],
    retrieve_classes: RetrieveClassesFn,
    term_types: dict[str, Any],
) -> StrategyResult:
    """Map LLM-generated terms by EDAM field (shared by V2 and V3)."""
    topics = _empty_df()
    operations = _empty_df()
    data = _empty_df()
    formats = _empty_df()

    if generated_terms.get("topic"):
        topics = await retrieve_classes(
            json.dumps(generated_terms["topic"]), term_types["topic"]
        )

    if generated_terms.get("operation"):
        operations = await retrieve_classes(
            json.dumps(generated_terms["operation"]), term_types["operation"]
        )

    if generated_terms.get("data"):
        data = await retrieve_classes(
            json.dumps(generated_terms["data"]), term_types["data"]
        )

    if generated_terms.get("format"):
        formats = await retrieve_classes(
            json.dumps(generated_terms["format"]), term_types["format"]
        )

    merged = _merge_dataframes([topics, operations, data, formats])

    return StrategyResult(
        topics=topics,
        operations=operations,
        data=data,
        formats=formats,
        merged=merged,
    )


async def strategy_v4(
    generated_terms: dict[str, Any],
    retrieve_classes: RetrieveClassesFn,
    term_types: dict[str, Any],
) -> StrategyResult:
    """Map each generated term independently and keep top-1 hit (V4 strategy)."""

    async def map_field(field_name: str) -> pd.DataFrame:
        bucket = generated_terms.get(field_name, {})
        if not isinstance(bucket, dict) or len(bucket) == 0:
            return _empty_df()

        mapped_rows = _empty_df()
        for generated_term, payload in bucket.items():
            top_class = await retrieve_classes(
                json.dumps(payload), term_types[field_name], k=1
            )
            if top_class.empty:
                continue

            top_class = top_class.copy()
            top_class["Generated term"] = generated_term
            mapped_rows = pd.concat([mapped_rows, top_class], ignore_index=True)

        if not mapped_rows.empty:
            mapped_rows.sort_values(by=["Similarity"], ascending=False, inplace=True)
            mapped_rows = mapped_rows.drop_duplicates(subset=["Class ID"])

        return mapped_rows

    topics = await map_field("topic")
    operations = await map_field("operation")
    data = await map_field("data")
    formats = await map_field("format")

    merged = _merge_dataframes([topics, operations, data, formats])

    return StrategyResult(
        topics=topics,
        operations=operations,
        data=data,
        formats=formats,
        merged=merged,
    )


async def run_edam_mapping(
    chat_profile: str,
    user_text: str,
    generated_terms: dict[str, Any],
    retrieve_all: RetrieveAllFn,
    retrieve_classes: RetrieveClassesFn,
    term_types: dict[str, Any],
) -> StrategyResult:
    """Dispatch EDAM mapping to the strategy matching the active chat profile."""

    if chat_profile == "EDAM retriever V1":
        return await strategy_v1(user_text, retrieve_all)

    if chat_profile in (
        "EDAM generator and retriever V2",
        "EDAM generator and retriever V3",
    ):
        return await strategy_v2_or_v3(generated_terms, retrieve_classes, term_types)

    if chat_profile == "EDAM generator and retriever V4":
        return await strategy_v4(generated_terms, retrieve_classes, term_types)

    raise ValueError(f"Unknown chat profile: {chat_profile}")

