"""Token prices for the production labeling model (Claude Haiku 4.5).

One place for the labeler's stats, the labeling run record and the prompt evaluation's spend
cap. Model-aware pricing is tracked in #54.
"""

INPUT_USD_PER_MILLION = 1.00
OUTPUT_USD_PER_MILLION = 5.00


def llm_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Cost in USD of the given token counts at the production model's prices."""
    return (
        input_tokens / 1_000_000 * INPUT_USD_PER_MILLION
        + output_tokens / 1_000_000 * OUTPUT_USD_PER_MILLION
    )
