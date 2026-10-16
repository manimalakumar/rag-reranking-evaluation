# Evaluate Retrieval to Generation

A Qwen/OpenRouter version of `Evaluate-retrieval-reranker`, preserving the four experiments, dataset revision, baseline corpus and reviewed membership, prompts, candidate/context budgets, scoring definitions and seeds. Targeted reranking now uses 500 opportunities and 200 controls; answer quality takes up to 25 questions per resulting subgroup. The original project retains its NVIDIA results.

| Role | OpenRouter model ID |
| --- | --- |
| Embedding | `qwen/qwen3-embedding-4b` |
| Reranking | `qwen/qwen3-reranker-8b` |
| Generation and RAGAS judge | `qwen/qwen3-30b-a3b-instruct-2507` |

**Hosted results are saved under `.cache/qwen-runs/`.** Inspect the current run with `.venv/bin/python tools/live_status.py`; the background log and status are `.cache/live-rebuild.log` and `.cache/live-rebuild-status.json`. A running state means the full rebuild is not finished. Old embeddings, indexes, retrieved orders, answer checkpoints, scores, latency measurements, human answer judgments and notebook outputs are excluded. Baseline dataset rows, corpus text and reviewed input annotations are retained because they are independent of the model. Higher-stage samples are selected anew from Qwen retrieval outcomes; answer-quality subgroups use `min(25, available questions)`, with no replacement or duplication. Empty subgroups contribute no answer pairs. The targeted sample requires at least 500 eligible opportunities and 200 eligible controls. The existing latency study still requires two questions per configured stratum.

## Environment

Use **Python 3.14.6**, approved as the temporary fallback, and **Haystack 3.4.0** (latest checked October 15, 2026). Python 3.15.0 is newer, but the attempted install required native builds for dependencies including `jiter`; a complete working 3.15 environment was not established. Revisit 3.15 after compatible wheels are available. Haystack's document store, cosine retriever and evaluator cross-check remain usable with Qwen vectors. The tested OpenAI SDK is 2.54.0; it calls OpenRouter's OpenAI-compatible endpoints.

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python tools/verify_offline.py
```

`requirements-runtime.txt` lists the direct runtime requirements; `requirements-lock.txt` records the environment actually tested on macOS arm64. Select `.venv/bin/python` as your notebook interpreter. RAGAS 0.4.3 retains its tested LangChain 0.3 dependencies; newer LangChain removes modules that RAGAS imports. The runtime and lock include the local scoring and indexing dependencies, so the verification runner refuses skipped tests.

## Changed API contracts

- Embeddings: `POST https://openrouter.ai/api/v1/embeddings`, with `model`, `input`, `dimensions` and `encoding_format=float`. Use 2,560 dimensions throughout retrieval, baseline, latency and evaluation. Passage text is unchanged. Queries receive Qwen's `Instruct: ...\nQuery: ...` format; the instruction is configurable as `query_instruction`.
- Reranking: `POST https://openrouter.ai/api/v1/rerank`, with a string `query`, a `documents` list and `top_n` equal to the candidate count. Validate that `results` contains a complete permutation with finite `relevance_score` values. Raw provider responses remain unchanged in traces; the internal study adapter retains the existing score field name for compatibility.
- Generation: `POST https://openrouter.ai/api/v1/chat/completions`. Use the requested model's lowercase OpenRouter slug. Preserve the prompts, temperature, seed and 400-token answer limit. The Instruct-2507 model operates without thinking; NVIDIA-specific `chat_template_kwargs`, `input_type` and `truncate` fields are removed.
- RAGAS uses the same new generator as judge and the same Qwen query preprocessing for relevancy embeddings. Baseline original-question vectors are reused only when their exact new request contract matches. Judge scores and embedding caches are recreated.

Model IDs, base URLs, dimensions and request settings live in `configs/execution.json`, `configs/experiment1.json`, and each experiment's `config.json`. Baseline pipeline settings remain in `experiments/v1-baseline/config.json`. Credentials come only from the environment; `.env.example` contains a placeholder, and helpers do not automatically load `.env` files.

## Verify first, rebuild second

```sh
# No credentials needed; socket connections are blocked during these tests.
.venv/bin/python tools/rebuild_results.py

# Supply your own key in the shell; do not commit it.
export OPENROUTER_API_KEY='your-key'

# Tests run again and must pass before hosted work starts.
.venv/bin/python tools/rebuild_results.py --study baseline --allow-model-calls

# Rebuild all four experiments and the retrieval diagnosis.
.venv/bin/python tools/rebuild_results.py --study all --allow-model-calls

# Explicit additional model judging of the rebuilt answers.
.venv/bin/python tools/rebuild_results.py --study all --allow-model-calls --judge
```

`--study` accepts `baseline`, `diagnosis`, `targeted`, `answers`, `latency`, or `all`. Higher stages prepare and resume their required parents. Targeted reranking rebuilds retrieval/reranking; its optional full-cohort generation stays explicit in the v2 notebook, as in the original study. The answer-quality and latency stages rebuild their own paired answers and deterministic scores. `--judge` evaluates the baseline, answer-quality and latency answers. Fresh human review is always pending.

Model calls default to disabled even if a key exists. The rebuild entry point enables them only after passing all offline tests. For deliberate notebook-based hosted work after verification, explicitly export `ALLOW_MODEL_CALLS=1`; otherwise a missing stage stops with instructions to use the rebuild entry point. Importing helpers and replaying complete artifacts makes no model calls.

All new outputs go under the ignored `.cache/qwen-runs/`. Request fingerprints include model, endpoint, exact inputs, dimensions and query preprocessing. Dataset/index contracts, checksums and request traces support resumption of matching successes. Failed attempts remain recorded; changing inputs/settings requires a separate output identity. The live request ledger records hosted attempts across stages and restarts.

Full validation is substantial: roughly 980,000 passages and 101,000 queries. At 2,560 float32 dimensions the document matrix alone is roughly 10 GB, with another similarly sized index plus temporary training memory. Document/index preparation is outside measured path latency. Start with `--study baseline` before full validation. Hosted inference cost and provider-specific request limits still need live validation with your key.

## Request and retry caps

`configs/model_limits.json` sets a hard project-wide cap of **10,000 outbound hosted attempts**, including initial calls, retries, timeouts and failed responses. Cached successes do not consume allowance. A SQLite ledger at `.cache/model-request-budget.sqlite` reserves each attempt before sending and persists across process restarts. Keep that ledger when resuming; deleting it resets the accounting. Offline tests use isolated temporary ledgers and consume no live allowance.

| Stage | Cap including first attempt |
| --- | --- |
| Retrieval embeddings and reranking | 3 lifetime attempts per identical logical request |
| Baseline answer generation | 1 attempt |
| Other answer generation | 1 initial attempt plus at most 1 explicitly requested retry |
| Latency benchmark | 2 lifetime attempts per hosted stage and arm |
| RAGAS | 3 HTTP attempts per logical request, 1 structured-output attempt and 1 metric execution |

The OpenAI SDK retry layer is disabled. RAGAS structured-output retries no longer multiply the HTTP retry allowance. Successful answers remain fixed. Identical generation or latency requests retain their attempt limits across explicit retries and restarts, as long as the ledger is kept. A crash after reservation still consumes an attempt conservatively. The cap bounds request count; it does not enforce a dollar or token budget. Raising the configured overall cap is an explicit edit; it does not reset per-request limits.

## Notebooks

| Directory | Notebook |
| --- | --- |
| `experiments/v1-baseline/` | `Experiment-1-Baseline.ipynb` |
| `experiments/v2-targeted-reranking/` | `Experiment-2-Targeted-Reranking.ipynb` |
| `experiments/v3-answer-quality/` | `Experiment-3-Answer-Quality.ipynb` |
| `experiments/v4-latency-tradeoff/` | `Experiment-4-Latency-Tradeoff.ipynb` |
| `diagnosis-retrieval-opportunity/` | `Diagnosis-Retrieval-Opportunity.ipynb` |

Rebuild the required stages before viewing their results. Notebook teaching cells now inspect the active Qwen adapters; old hardcoded answer/latency example tables and model-dependent conclusions were removed. Optional historical comparisons are absent. The deterministic SQuAD source remains bundled under `scoring-study/output/squad_metric_sources/`, and new offline integration tests exercise real Haystack, FAISS and local scoring with mocked hosted responses. Historical result-preservation tests stay with the original result bundle.

## API references

- [OpenRouter embeddings](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings)
- [OpenRouter rerank](https://openrouter.ai/docs/api/api-reference/rerank/create-rerank)
- [Qwen embedding instructions and dimensions](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- [Requested generator on OpenRouter](https://openrouter.ai/qwen/qwen3-30b-a3b-instruct-2507)
- [Haystack 3.4.0](https://pypi.org/project/haystack-ai/3.4.0/)
