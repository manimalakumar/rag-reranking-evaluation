# Experiment 2: Targeted Reranking

Test whether reranking promotes annotated supporting passages into the five contexts supplied to the generator. The study uses the frozen **152,576-passage corpus** and samples **500 opportunities + 200 controls** before reranking.

An opportunity has **Recall@30 > Recall@5**. A control has no such gap; all 200 selected controls in this run already have all annotated support in the top five. Each question keeps its exact thirty retrieved candidates. Reranking changes their order; it does not retrieve additional passages.

[The notebook](Experiment-2-Targeted-Reranking.ipynb) contains the completed retrieval/reranking comparison. Among the 500 opportunities, Recall@5 improves for **289**, is unchanged for **207**, and declines for **4**. Controls help expose losses of support that embedding retrieval already selected.

The **IMEI walkthrough (1053665)** connects to the baseline. Both annotated supports exist in the larger corpus; reranking promotes one from **rank 6 to rank 3**, increasing Recall@5 from **0.5 to 1.0**. IMEI already belongs to the frozen 700-question cohort, so displaying it adds no observation.

Full-cohort answer generation is optional. A pending answer in an Experiment 2 display does not mean the separate [Experiment 3 answer-quality study](../v3-answer-quality/Experiment-3-Answer-Quality.ipynb) is incomplete. Retrieval gains alone do not establish better answers.

**Read the saved outputs without rerunning.** See the [project README](../../README.md) for setup and deliberate rebuilds. Preserve the selected IDs and candidate lists when comparing paths.
