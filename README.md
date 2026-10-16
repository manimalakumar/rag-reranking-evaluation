# Evaluate Retrieval to Generation

Does better retrieval lead to better answers and what does reranking cost in latency? This project compares embedding-based RAG with embedding + reranking RAG, following evidence from retrieval into generated answers. Four experiments examine the baseline, where reranking can help, whether retrieval gains carry into answer quality, and the response-time tradeoff.

The notebooks contain saved Qwen/OpenRouter executions. **You can read their recorded results without running any cells or making model calls.** See each folder's README and notebook for technical details.

## Start here

| Step | Notebook | Purpose |
| --- | --- | --- |
| Optional demo | [RAG with and without reranking](demos/00-rag-with-and-without-reranking/RAG-Retriever-Embedding-Reranker.ipynb) | Inspect two answers and their exact supporting contexts. |
| 1. Baseline | [Experiment 1](experiments/v1-baseline/Experiment-1-Baseline.ipynb) | Compare both RAG paths on 31 questions and a smaller fixed corpus. |
| 2. Diagnosis | [Retrieval opportunity](diagnosis-retrieval-opportunity/Diagnosis-Retrieval-Opportunity.ipynb) | Find questions with annotated support below the first five results but within the first thirty. |
| 3. Targeted reranking | [Experiment 2](experiments/v2-targeted-reranking/Experiment-2-Targeted-Reranking.ipynb) | Rerank the same thirty candidates for 500 opportunities and 200 controls. |
| 4. Answer quality | [Experiment 3](experiments/v3-answer-quality/Experiment-3-Answer-Quality.ipynb) | Compare 79 saved answer pairs using reference overlap and LLM-based evaluation. |
| 5. Latency | [Experiment 4](experiments/v4-latency-tradeoff/Experiment-4-Latency-Tradeoff.ipynb) | Compare complete online paths on eight question pairs. |

## Study scope

The dataset is MS MARCO v2.1 validation, using the pinned revision and recorded seeds. The larger frozen corpus contains **152,576 passage embeddings**, covering all associated passages for **15,585 question IDs**. Of those questions, **10,242 have annotated support labels**; diagnosis identifies **1,085 reranking opportunities**. This is a subset, not the full validation split. Embedding stopped at this snapshot to limit runtime, cost and storage.

The baseline uses a separate **10,210-passage corpus** built from 31 baseline questions and 1,000 additional distractor questions sampled with seed 42. Experiment 2 samples 500 opportunities and 200 controls from the larger study. Experiment 3 samples up to 25 per resulting subgroup, yielding **79 questions / 158 answers**. Experiment 4 uses **eight questions / sixteen answers**, two per populated subgroup.

The continuous IMEI walkthrough (**1053665**) shows the distinction between corpus coverage and ranking: neither annotated support is in the baseline corpus; both are in the larger corpus, where reranking moves one from rank 6 to rank 3 and increases Recall@5 from 0.5 to 1.0. IMEI is separate from the 31 baseline averages and already included in Experiments 2 and 3. It is not in the latency sample. Original effervescent diagnostic records remain preserved separately.

## Models and evaluation

| Role | OpenRouter model ID |
| --- | --- |
| Embedding | `qwen/qwen3-embedding-4b` — 2,560 dimensions in the saved study |
| Reranking | `qwen/qwen3-reranker-8b` |
| Generation and RAGAS judge | `qwen/qwen3-30b-a3b-instruct-2507` |

Each RAG path supplies five passages to the generator; reranking reorders thirty retrieved candidates rather than searching for new passages.  

We use **MS MARCO v2.1 Dataset** for its questions, passages, support annotations and reference answers. Retrieval metrics measure annotated-support coverage and ranking. **SQuAD-style Token F1 and exact match** provide reproducible reference-overlap scores without additional model calls; these scoring methods can be applied beyond the SQuAD dataset.

**RAGAS** complements reference overlap with LLM-based evaluation: faithfulness assesses whether answer claims are supported by the supplied passages, while answer relevancy assesses how well the answer addresses the question. This study uses the Qwen generator model as the evaluator. Higher Token F1 alone does not establish a supported or correct answer.

Generation and both RAGAS metrics are complete for the saved 79-question answer-quality study and eight-pair latency benchmark. Human answer review remains pending.

Both answer-quality arms requested the same generator model and settings, but OpenRouter could route them to different serving providers. One recorded pair uses SiliconFlow and Nebius. Backend variation is a limitation; a future controlled run can pin a common provider and fallback policy. The deliberately stratified samples do not represent overall question traffic, and the eight-pair latency benchmark is descriptive rather than a production latency guarantee.

## Viewing and reproducing results

**To inspect the study:** open the notebooks and read their saved outputs. No rebuild, API key or notebook execution is required. Saved outputs preserve the recorded execution; a later replay may load matching artifacts instead of making the original calls.

**To execute or rebuild:** install the pinned environment and provide the required data and artifacts. Large caches, indexes, downloads and hosted-run artifacts are excluded from Git. The repository therefore preserves readable notebook results, but a fresh clone does not contain every artifact needed for offline replay. Rebuilding missing artifacts can incur model-call costs and may produce different results.

The recorded environment uses Python **3.14.6**, Haystack **3.4.0**, OpenAI SDK **2.54.0** and RAGAS **0.4.3**. `requirements-lock.txt` records the tested macOS arm64 environment; `requirements-runtime.txt` lists direct runtime requirements. Other platforms may require dependency adjustments. This is a tested configuration, not a claim about the latest available versions.

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python tools/verify_offline.py
```

Select `.venv/bin/python` as the notebook interpreter. For deliberate notebook replay, set `NOTEBOOK_MODE=replay` before starting the kernel; the matching saved artifacts must exist. The notebooks describe their execute-mode action controls. For a deliberate hosted rebuild, use the entry point below after reviewing the configuration and scope.

```sh
# Verification only; no credentials needed.
.venv/bin/python tools/rebuild_results.py

# Set your own key in the shell; do not commit it.
export OPENROUTER_API_KEY='your-key'

# Deliberately rebuild/resume a stage with hosted calls enabled.
.venv/bin/python tools/rebuild_results.py --study baseline --allow-model-calls

# Optional: rebuild/resume the configured stages and explicitly request judging.
.venv/bin/python tools/rebuild_results.py --study all --allow-model-calls --judge
```

`--study` accepts `baseline`, `diagnosis`, `targeted`, `answers`, `latency`, or `all`. Higher stages prepare or resume required parents. **Review corpus scope and configuration before a rebuild; these commands are not needed to view the saved study and do not promise to reproduce its frozen snapshot automatically.** Full-cohort generation in Experiment 2 remains optional; Experiment 3 generates its own sampled answer pairs.

## Configuration and request records

Settings live in `configs/execution.json`, `configs/experiment1.json`, `configs/retrieval_scope.json`, and the experiment-specific `config.json` files. The root `.env.example` documents `OPENROUTER_API_KEY` with a placeholder. Helpers do not automatically load `.env` files. Changing hosting providers may require changing endpoints, credentials and request/response adapters, not only model IDs.

The study uses OpenRouter's embedding, reranking and OpenAI-compatible chat endpoints. Qwen query embeddings use the configured `Instruct: ...\nQuery: ...` format. The saved study uses 2,560 dimensions consistently; changing dimensions or preprocessing requires compatible document and query vectors and new evaluation results.

Hosted artifacts are stored locally under the ignored `.cache/qwen-runs/`. Matching successful requests can be reused. Request fingerprints include model, endpoint, inputs, dimensions and query preprocessing; changed settings require a separate output identity. Saved request and response records support inspection of contexts, provider information, attempts and failures. `tools/live_status.py` reports local run status when those artifacts exist.

## Request and retry caps

`configs/model_limits.json` configures a project-wide cap of **10,000 outbound hosted attempts**, including retries, timeouts and failures. Matching cached successes consume no new attempt. The persistent ledger is `.cache/model-request-budget.sqlite`; retain it when resuming so accounting is not reset. The cap limits request count, not dollars or tokens.

| Stage | Configured attempt cap |
| --- | --- |
| Retrieval embeddings and reranking | 3 lifetime attempts per identical logical request |
| Baseline answer generation | 1 attempt |
| Other answer generation | 1 initial attempt plus at most 1 explicitly requested retry |
| Latency benchmark | 2 lifetime attempts per hosted stage and arm |
| RAGAS | 3 HTTP attempts per logical request, 1 structured-output attempt and 1 metric execution |

The SDK retry layer is disabled. Successful answers remain fixed for matching requests. Offline tests use isolated ledgers and do not consume live allowance. Review the current configuration before enabling paid work.
