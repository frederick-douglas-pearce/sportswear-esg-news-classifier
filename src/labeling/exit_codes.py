"""Exit-code contract between scripts/label_articles.py and the agent runner.

Labeling consumes the rows it selects: an article moves out of `pending` as soon
as it is processed. Retrying a run that already did work therefore cannot repeat
it — the retry finds nothing pending, exits clean, and its empty stats replace
the real ones (issue #81).

So the script has to say which kind of failure it hit:

    EXIT_SUCCESS          every article completed
    EXIT_FAILURE          nothing completed; the cause may be transient
                          (database down, `uv` fetch blip per issue #51) and
                          retrying is the right move
    EXIT_PARTIAL_FAILURE  some articles completed and others errored; the batch
                          is spent and a retry would only erase the results
"""

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_PARTIAL_FAILURE = 2

# Exit codes the agent runner must not retry.
NON_RETRYABLE_EXIT_CODES = frozenset({EXIT_PARTIAL_FAILURE})
