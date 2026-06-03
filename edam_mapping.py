from __future__ import annotations

from typing import Any, Awaitable, Callable

import pandas as pd


RetrieveAllFn = Callable[[str], Awaitable[pd.DataFrame]]
RetrieveClassesFn = Callable[..., Awaitable[pd.DataFrame]]


async def apply_mapping_strategy(
    chat_profile: str,
    user_text: str,
    generated_terms: dict[str, Any] | None,
    retrieve_all: RetrieveAllFn,
    retrieve_classes: RetrieveClassesFn,
) -> pd.DataFrame:
    """Apply an EDAM mapping strategy and return mapped terms as a DataFrame.

    This minimal implementation currently supports only V1 mapping.
    """
    _ = generated_terms
    _ = retrieve_classes

    if chat_profile == "EDAM retriever V1":
        return await retrieve_all(user_text)

    raise ValueError(f"Unknown or unsupported chat profile: {chat_profile}")

