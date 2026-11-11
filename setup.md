# Setup: live RAG tutorials

The main notebooks use **Live / Replay / Plan**, with **Live as the default**. Opening a notebook does not execute it. Selecting Live and executing its cells authorizes the declared hosted stages; there is no second `NOTEBOOK_ALLOW_MODEL_CALLS` switch in this tutorial path.

The revised code has not yet been executed against hosted models. Original outputs remain in their original cells for review. Cell metadata marks the changed source as unexecuted; these outputs do not validate the revised code. There are no disabled reference sections or parallel notebook implementations.

## Install the environment and kernel

The recorded/tested environment is Python **3.14.6**, Haystack **3.4.0**, OpenAI SDK **2.54.0**, RAGAS **0.4.3**, FAISS, NumPy, pandas and PyArrow. No model, Python or dependency upgrade is required for this tutorial change. `requirements-lock.txt` is the recorded macOS arm64 environment; other platforms may need adjusted compatible packages.

From the project root:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python -m ipykernel install --user --name qwen-rag-tutorial --display-name "Python — Qwen RAG tutorial"
.venv/bin/python tools/verify_tutorial.py
```

Select that interpreter/kernel in your notebook editor. The new hermetic tutorial tests use tiny synthetic inputs, the real SDK with a mocked HTTP transport, and isolated request ledgers. They do not need `.cache/qwen-runs/`, credentials, downloads or model calls. `tools/verify_offline.py` runs the broader offline suite with isolated synthetic fixtures and blocked network access.

## Service, models and credentials

The model service is OpenRouter's OpenAI-compatible API, `https://openrouter.ai/api/v1`; no local model server is needed. Models stay unchanged:

- Embedding: `qwen/qwen3-embedding-4b`, 2,560 dimensions; questions use the recorded instruction prefix.
- Reranking: `qwen/qwen3-reranker-8b`, via `client.post("/rerank", ...)`.
- Generation and RAGAS judge: `qwen/qwen3-30b-a3b-instruct-2507`, via the OpenAI-compatible chat API.

Set your own credential in the environment **before launching the notebook editor/kernel**:

```bash
export OPENROUTER_API_KEY='your-own-key'
```

Do not put credentials in source cells or commit them. `.env.example` contains only the variable name; these helpers do not automatically load `.env`. An already-running kernel may not inherit a later shell export: restart it from the configured environment. The tutorial validates credentials and inputs before any request.

For a browser notebook interface, install JupyterLab in this environment if absent and launch it from the project root after exporting the credential:

```bash
.venv/bin/python -m pip install jupyterlab ipywidgets
.venv/bin/python -m jupyterlab
```

`ipywidgets` is optional for progress display, not model correctness. A VS Code notebook interface can instead use the selected `.venv` interpreter; ensure its kernel receives the environment variable.

## Prepare or restore the declared inputs

`configs/tutorial_inputs.json` explicitly names the frozen upstream inputs. They are **inputs**, not previous answers adopted as current results:

| Notebook | Prepared inputs | What Live executes |
| --- | --- | --- |
| Warm-up | Larger corpus/index; COVID-19 question declared in configuration | Question embedding, retrieval, reranking, two answers, local scoring |
| Baseline | 10,210-passage corpus/matrix; 31 question IDs plus separate IMEI diagnostic | Fresh question embeddings, Haystack cosine retrieval, reranking, 64 answers, scoring/judging |
| Diagnosis | 152,576-passage corpus/index; 15,585 frozen eligible queries | Fresh query embeddings, retrieval and opportunity analysis; no reranking or generation |
| Targeted reranking | Frozen 700 questions and their exact 30 candidates | 700 fresh reranker calls, evidence-coverage comparison, and two worked answer pairs with SQuAD/RAGAS scoring |
| Answer quality | Frozen 79 questions, each arm's five passages; 12-question prompt subset | 158 new main answers plus 36 new prompt-extension answers, local scoring and judging |
| Latency | Frozen 8-question assignments and larger corpus/index | Fresh serial online paths, embeddings, retrieval, reranking where applicable, generation, retries and scoring |

Large original assets are ignored by Git. A fresh clone supports reading recorded notebook outputs and running the hermetic tutorial tests, but does not include every input needed to execute Live or replay a historical cache. Restore the matching upstream files from your local artifact backup, preserving their project-relative structure, or perform a separate deliberate preparation run and update `configs/tutorial_inputs.json` to the compatible corpus, questions and contexts.

**Passage embedding remains stopped.** The tutorial never downloads a dataset, embeds passages, constructs a missing index, runs a missing parent study, or changes the frozen scope implicitly. Missing assets cause a clear error listing the files and referring here. Merely setting an API key does not prepare missing inputs. The legacy checkpointed configuration is not an automatic fresh-clone dataset builder.

## Choose a mode

Each main notebook's first code cell exposes `MODE`, `RUN_ID` and `RESUME`:

```python
MODE = "live"       # new hosted inference; new identity by default
RUN_ID = None
RESUME = False
```

- **Live:** creates a new `.cache/tutorial-runs/<study>/<run-id>/` and produces new responses for the declared stages. Completed older studies never substitute for current answers, scores or timings. Even if a notebook has recorded outputs, a new Live execution is new work.
- **Replay:** set `MODE="replay"` and an explicit completed tutorial `RUN_ID`. It reads only that run’s files and never fills missing stages with model calls. To read the committed outputs, simply open the notebook without executing it; `RUN_ID="reference"` is not an execution mode.
- **Plan:** set `MODE="plan"`; it reports stages, scope, caps and missing inputs without credentials, network access, downloads, inference or loading old answers.
- **Explicit resume:** choose `MODE="live"`, set an existing `RUN_ID`, and set `RESUME=True`. Only successes from that exact run may be reused, with matching inputs/settings. Failed/interrupted requests remain retained for inspection; caps are not reset. A partial latency benchmark cannot use cached HTTP responses as new timings; start a fresh benchmark ID instead. Completed benchmark timings can be replayed unchanged.

Environment equivalents are `NOTEBOOK_MODE`, `TUTORIAL_RUN_ID` and `TUTORIAL_RESUME=1`. Setting `MODE` directly in a notebook overrides its environment default once you edit that assignment.

## Launch and save in the main notebook

Use the main notebook, execute its original sections in order, and save it. Setup only validates prerequisites; each subsequent cell performs the visible experimental stage. The model calls shown are real `client.embeddings.create`, `client.chat.completions.create`, and the client's generic POST for reranking. The callbacks are invoked by the stage loops, not merely displayed as unused examples.

The supplied runner selects the kernel, executes all code cells in section order and saves actual outputs in the **same main notebook**. Live and Replay replace displayed outputs with the selected run’s results. Plan only prints the workflow and leaves the notebook untouched:

```bash
# No model calls or notebook writes; inspect prerequisites and the workflow.
.venv/bin/python tools/run_tutorial.py --study baseline --mode plan

# Live is the default. This spends the configured model requests.
.venv/bin/python tools/run_tutorial.py --study baseline

# Replay or resume a named new tutorial run.
.venv/bin/python tools/run_tutorial.py --study answers --mode replay --run-id YOUR_RUN_ID
.venv/bin/python tools/run_tutorial.py --study answers --mode live --run-id YOUR_RUN_ID --resume
```

`--study` accepts `demo`, `baseline`, `diagnosis`, `targeted`, `answers` or `latency`. The runner never skips a fresh Live run because the notebook was previously completed. No separate executed notebook is created. Manual Run All executes the same active stage cells; Live makes new model calls. Do not execute merely to inspect the existing outputs.

## Scope, scoring and accounting

The 31 baseline averages exclude IMEI. IMEI remains part of the original frozen targeted and 79-question answer-quality cohorts and is not counted twice by its walkthrough. The 12-question generation extension stays separate from the 79-pair aggregates and uses the declared pinned provider/fallback policy. A new run's extension results require new qualitative inspection; old review notes do not judge new answers.

RAGAS belongs to the declared Live baseline, two targeted worked examples, answer-quality and latency workflows. There is no additional generation/judge switch silently leaving those stages pending. Judge metrics may involve multiple chat calls and evaluation-only question embeddings; these do not re-embed the passage corpus. Failed metrics are retained and reported as failures/missing, not as valid zero scores. Current-run reports never load historical scores as replacements.

The project-wide persistent 10,000-attempt ledger remains in force, as do per-request caps: normal question embeddings/reranking up to 3 HTTP attempts, first generation 1, latency stages up to 2, and the existing RAGAS transport/metric limits. OpenAI SDK automatic retries are disabled. Never delete the ledger to resume. Plan counts are workflow sizes, not dollar estimates or upper bounds on all judge HTTP requests.
