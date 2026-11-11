# Evaluate Retrieval to Generation

Does better retrieval lead to better answers, and what does reranking cost in latency? This tutorial compares embedding RAG with embedding + reranking RAG. **Live is the default:** each new run makes the model calls shown in the notebook cells and saves its own answers, scores and timings.

## Start here

1. **Set up Python and the notebook kernel.** Follow [setup.md](setup.md#1-install-the-environment-and-kernel), then export your own `OPENROUTER_API_KEY`.
2. **Prepare the prerequisites.** Start with the small public-data profile below. Preparation embeds documents once; subsequent notebook runs reuse those document vectors.
3. **Open the warm-up notebook.** Launch Jupyter from the configured shell and select the project kernel.
4. **Restart Kernel → Run All → Save.** Keep `MODE="live"`, `RUN_ID=None`, and `RESUME=False` for a new experiment. Live execution incurs API usage.

```bash
# Inspect the preparation plan: no download, model calls or writes.
.venv/bin/python tools/prepare_tutorial_inputs.py --profile small

# Explicitly prepare 512 passages from the pinned public dataset.
.venv/bin/python tools/prepare_tutorial_inputs.py --profile small --allow-model-calls
export TUTORIAL_INPUTS=.cache/tutorial-inputs/small/tutorial_inputs.json
.venv/bin/python -m jupyterlab
```

Open [RAG with and without reranking](demos/00-rag-with-and-without-reranking/RAG-Retriever-Embedding-Reranker.ipynb). The small profile uses a labeled question covered by its corpus. It supports the warm-up only; its results are a separate exercise, not a reproduction of the larger study.

For **all six notebooks**, prepare the reference inputs instead:

```bash
.venv/bin/python tools/prepare_tutorial_inputs.py --profile reference
.venv/bin/python tools/prepare_tutorial_inputs.py --profile reference --allow-model-calls
export TUTORIAL_INPUTS=.cache/tutorial-inputs/reference/tutorial_inputs.json
.venv/bin/python -m jupyterlab
```

The full route rebuilds **152,576 document embeddings** and the index from the pinned public MS MARCO dataset. A compact committed recipe reconstructs the frozen question membership, candidate ranks, baseline corpus and answer contexts. **No private artifact backup is required.** New embedding responses and subsequent results can differ from the recorded run. Preparation validates the passage IDs/text against the recipe before calling the embedding model. See [setup.md](setup.md#2-prepare-the-prerequisites) for costs, storage, resumption and input checks.

## Notebook sequence

| Step | Notebook | What Live executes |
| --- | --- | --- |
| 0 | [Two RAG paths](demos/00-rag-with-and-without-reranking/RAG-Retriever-Embedding-Reranker.ipynb) | Query embedding, retrieval, reranking, two answers and local scoring |
| 1 | [Baseline](experiments/v1-baseline/Experiment-1-Baseline.ipynb) | Both paths for 31 questions plus the separate IMEI diagnostic |
| 2 | [Diagnosis](diagnosis-retrieval-opportunity/Diagnosis-Retrieval-Opportunity.ipynb) | Query retrieval and opportunity analysis on the frozen eligible subset |
| 3 | [Targeted reranking](experiments/v2-targeted-reranking/Experiment-2-Targeted-Reranking.ipynb) | Rerank 500 opportunities + 200 controls; generate and score two worked-example pairs |
| 4 | [Answer quality](experiments/v3-answer-quality/Experiment-3-Answer-Quality.ipynb) | Generate and score 79 pairs; separately compare three prompts on 12 questions |
| 5 | [Latency](experiments/v4-latency-tradeoff/Experiment-4-Latency-Tradeoff.ipynb) | Time eight paired online paths and score those exact answers |

The reference profile contains a **152,576-passage corpus** and **15,585 fully covered question IDs**, a subset of MS MARCO v2.1 validation. Baseline uses a separate **10,210-passage corpus**. Answer-quality contexts and subgroup assignments are frozen inputs; they are not resampled from a new targeted-reranking run. IMEI **1053665** is excluded from the 31 baseline averages and already included in the targeted and answer-quality samples.

## Models and evaluation

| Role | OpenRouter model ID |
| --- | --- |
| Embedding | `qwen/qwen3-embedding-4b`, 2,560 dimensions |
| Reranking | `qwen/qwen3-reranker-8b` |
| Generation and RAGAS judge | `qwen/qwen3-30b-a3b-instruct-2507` |

The client uses OpenRouter's OpenAI-compatible API. Each answer receives five passages; reranking reorders thirty candidates. Annotated-support Recall@5 measures evidence coverage. SQuAD-style Token F1 and exact match measure reference overlap. RAGAS faithfulness assesses support in the supplied evidence; answer relevancy assesses alignment with the question. Judge estimates do not replace human correctness review.

Main-study requests may use different serving providers. The three-prompt extension pins Nebius with fallbacks disabled. Small, deliberately stratified samples describe these experiments rather than overall traffic or production latency.

## Recorded outputs and reproducibility

Open the notebooks without executing cells to read their **Recorded reference run** outputs. Numerical findings in reference prose belong to that recorded execution; a new Live run's tables are authoritative for its new results. [The execution report](reports/tutorial-live-execution-20261110.json) records completion and the partial extension honestly: its main 79-pair study completed, but its extension stopped at 33/36 answers and has no extension judge results for that run.

A fresh clone supports reading outputs and running offline tests. **Full offline Replay needs the matching run artifacts**, which are excluded from Git. Public input preparation supplies upstream inputs for new Live experiments; it does not recreate earlier model responses. Replay requires an explicit completed run ID; Plan makes no model calls. See [setup.md](setup.md#4-execution-modes-and-request-caps).

Datasets, vectors, indexes and request traces stay under ignored `.cache/`. The committed input recipe is approximately **485 KiB**; preparation does not add gigabytes to Git. Keep prior run folders and the persistent request ledger when resuming.

Older checkpointed rebuild commands and artifact details are documented separately in [Advanced workflows](docs/advanced-workflows.md).
