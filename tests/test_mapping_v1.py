import pandas as pd
import pytest
from unittest.mock import AsyncMock

from edam_mapping import apply_mapping_strategy


@pytest.mark.asyncio
async def test_v1_mapping_calls_global_retriever_once() -> None:
    expected_df = pd.DataFrame(
        [
            {
                "Preferred Label": "Image analysis",
                "Similarity": 0.91,
                "child_terms": 4,
                "Class ID": "http://edamontology.org/operation_3434",
            }
        ]
    )

    retrieve_all = AsyncMock(return_value=expected_df)
    retrieve_classes = AsyncMock()

    result = await apply_mapping_strategy(
        chat_profile="EDAM retriever V1",
        user_text="medical image segentation",
        generated_terms=None,
        retrieve_all=retrieve_all,
        retrieve_classes=retrieve_classes,
    )

    retrieve_all.assert_awaited_once_with("medical image segentation")
    retrieve_classes.assert_not_called()
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_df)

