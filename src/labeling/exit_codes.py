"""Exit-code contract between scripts/label_articles.py and the agent runner.

Labeling consumes the rows it selects: an article moves out of `pending` as soon
as it is processed. Retrying a run that already did work therefore cannot repeat
it — the retry finds nothing pending, exits clean, and its empty stats replace
the real ones (issue #81).

So the script has to say which kind of failure it hit:

    EXIT_SUCCESS          every article completed
    EXIT_FAILURE          at least one article remains pending after a
                          transient error; retrying may make progress
    EXIT_PARTIAL_FAILURE  some articles completed and others errored; the batch
                          is spent and a retry would only erase the results
    EXIT_ABORTED          articles remain pending after a known non-transient
                          error; automatic retry cannot fix the cause
"""

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_PARTIAL_FAILURE = 2
EXIT_ABORTED = 4

# Exit codes the agent runner must not retry.
NON_RETRYABLE_EXIT_CODES = frozenset({EXIT_PARTIAL_FAILURE, EXIT_ABORTED})
