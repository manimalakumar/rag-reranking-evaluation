# Experiment 4: Latency Tradeoff

Measure the operational cost of adding reranking to the complete online RAG path. The saved benchmark contains **eight paired questions / sixteen answers**: two questions from each of the four populated answer-quality subgroups, selected with seed 43. The fifth planned subgroup has no eligible cases.

Each arm makes its own query-embedding request, retrieves thirty passages, selects five contexts and generates an answer. The reranked arm also reranks its thirty candidates. **Precomputed document embeddings and initial corpus/index loading are outside the timed path.** Generation is non-streaming, so measurements describe completion time rather than time to first token.

[The notebook](Experiment-4-Latency-Tradeoff.ipynb) preserves **sixteen successful paths, forty hosted stages and forty-one timed attempts**, including one failed attempt and its retry. Total wall time includes attempts, backoff, local work and client overhead; successful-request timing is reported separately. No setup records are excluded and no answers are truncated.

Generation, deterministic answer scoring and both RAGAS metrics are complete. Quality scores refer to the benchmark's own answers, not substituted Experiment 3 answers. Neither IMEI (1053665) nor effervescent (1100870) is in this sample. Human answer review remains pending.

Eight pairs provide a descriptive operational check, not a stable population or production-latency estimate. Network and serving-provider variation affect the observations. The saved plot is unavailable in this snapshot; the numeric comparisons remain readable. Some original captions mention ten questions, but the saved eight-row sample defines the actual scope.

**Read the saved outputs without rerunning.** See the [project README](../../README.md) for setup and deliberate rebuild instructions. A new execution creates new timing observations; it does not reproduce the original network and backend conditions.
