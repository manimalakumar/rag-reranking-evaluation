# Experiment 1: Baseline

Compare Embedding RAG with Embedding + reranking RAG on **31 seeded questions** using a fixed **10,210-passage corpus**. The corpus combines passages from those questions with passages associated with 1,000 additional distractor questions sampled with seed 42. Both paths retrieve thirty candidates and supply five passages to the generator.

[The notebook](Experiment-1-Baseline.ipynb) preserves the recorded answers, contexts, retrieval scores, answer scores and request records. Reference overlap and evidence support are evaluated separately; human answer review remains pending.

The separate **IMEI diagnostic (1053665)** starts the walkthrough used in Experiments 2 and 3. This smaller corpus contains **neither of its two annotated supports**. Its results do not enter the 31-question averages. The original saved batch also retains the effervescent diagnostic (1100870); its historical records are not relabeled as IMEI.

Next, use the [diagnosis notebook](../../diagnosis-retrieval-opportunity/Diagnosis-Retrieval-Opportunity.ipynb) to locate ranking opportunities in the larger frozen corpus.

**Read the saved outputs without rerunning.** See the [project README](../../README.md) for setup, artifacts and deliberate rebuild instructions. Missing caches are not required to read saved notebook outputs, but are needed for matching offline replay.
