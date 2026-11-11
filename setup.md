# Setup: live RAG tutorials

Follow **install → prepare prerequisites → open a notebook → Restart Kernel / Run All / Save**. The small profile is the quickest fresh-clone entry point. No private backup is needed to prepare either profile.

## 1. Install the environment and kernel

The tested environment uses Python **3.14.6**, Haystack **3.4.0**, OpenAI SDK **2.54.0**, RAGAS **0.4.3**, FAISS, NumPy, pandas and PyArrow. The lockfile records macOS arm64 packages; other platforms may need compatible package adjustments.

From the project root:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install jupyterlab ipywidgets
.venv/bin/python -m pip check
.venv/bin/python -m ipykernel install --user --name qwen-rag-tutorial --display-name "Python — Qwen RAG tutorial"
.venv/bin/python tools/verify_offline.py
export OPENROUTER_API_KEY='your-own-key'
```

Set the credential **before launching Jupyter or your editor's kernel**. Do not put it in a notebook or commit it. Helpers do not automatically load `.env`; already-running kernels do not inherit a later shell export. `ipywidgets` supplies notebook progress widgets.

The service is OpenRouter at `https://openrouter.ai/api/v1`; no local model server is needed. Models are `qwen/qwen3-embedding-4b` (2,560 dimensions), `qwen/qwen3-reranker-8b`, and `qwen/qwen3-30b-a3b-instruct-2507` for generation and judging. Calls use the OpenAI SDK: embeddings, generic POST `/rerank`, and chat completions.

## 2. Prepare the prerequisites

### Small warm-up profile

```bash
# Default is a plan: no network, writes or model calls.
.venv/bin/python tools/prepare_tutorial_inputs.py --profile small

# Explicitly download/stream public data and create document vectors + index.
.venv/bin/python tools/prepare_tutorial_inputs.py --profile small --allow-model-calls
export TUTORIAL_INPUTS=.cache/tutorial-inputs/small/tutorial_inputs.json
```

This profile keeps **512 unique passages** in pinned dataset order and chooses a fully covered labeled question. It supports the warm-up notebook only. Use `--passages N` (at least 30) and a new `--output .cache/tutorial-inputs/NAME` for another size. It is a different corpus/question from the recorded larger-corpus warm-up.

At 512 × 2,560 float32 values, the document matrix is about **5 MiB**; index vectors add about another **5 MiB**. Dataset streaming/cache files, metadata and response traces add local storage. Dataset transport can fetch/cache more than the retained passage prefix; the passage limit is not a network-byte limit.

### Reference profile for all six notebooks

```bash
.venv/bin/python tools/prepare_tutorial_inputs.py --profile reference
.venv/bin/python tools/prepare_tutorial_inputs.py --profile reference --allow-model-calls
export TUTORIAL_INPUTS=.cache/tutorial-inputs/reference/tutorial_inputs.json
```

This route streams `microsoft/ms_marco`, configuration `v2.1`, validation split, pinned revision `a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a`. It reconstructs the first **152,576 unique passages** and verifies a canonical ID/text digest and all **15,585 eligible questions' full passage coverage** before sending embedding requests.

The compact `configs/reference_input_recipe.json.gz` fixes question IDs/references, the 10,210-passage baseline membership, the targeted 700 × 30 candidate IDs and orders, the 79-question answer-quality sample, and eight latency assignments. The command joins those IDs to public dataset text. It creates **no generated answers, judge scores or timings**. Baseline vectors are selected from the larger document matrix, without another embedding call. The index uses exhaustive normalized inner-product search (cosine), with one inverted list.

**Full preparation is a paid document-embedding job**, not a small download: matrix and index vectors together are about 2.91 GiB before request traces and dataset files. Review the plan before enabling it. Both profiles run offline tests first and use the existing persistent request budget. Successful embedding batches are checkpointed; rerunning the identical preparation command reuses them. Changed settings require a new output directory. Failed request traces remain for inspection; do not delete the request ledger to reset a cap.

Prepared vectors, indexes, input tables and request traces stay under ignored `.cache/tutorial-inputs/`. Hugging Face may also use its normal user-level metadata cache; neither location enters Git. Neither command overwrites `configs/tutorial_inputs.json`, prior study artifacts, nor notebook outputs. Local readers with an existing prepared snapshot may omit `TUTORIAL_INPUTS` to use the original declared profile. No preparation command has been run against hosted models as part of adding this route.

## 3. Open a notebook and Run All

From the same shell after exporting the key and chosen `TUTORIAL_INPUTS`:

```bash
.venv/bin/python -m jupyterlab
```

Open the [warm-up notebook](demos/00-rag-with-and-without-reranking/RAG-Retriever-Embedding-Reranker.ipynb), select **Python — Qwen RAG tutorial**, and leave the first cell in Live mode with a new run identity:

```python
MODE = os.environ.get("NOTEBOOK_MODE", "live").lower()
RUN_ID = os.environ.get("TUTORIAL_RUN_ID") or None
RESUME = os.environ.get("TUTORIAL_RESUME") == "1"
```

For a fresh experiment, unset `TUTORIAL_RUN_ID` / `TUTORIAL_RESUME`, or set `RUN_ID=None`, `RESUME=False` directly. Choose **Restart Kernel → Run All**, then **Save**. All actual outputs stay in the main notebook. Continue through the [notebook sequence](README.md#notebook-sequence) with the reference profile. The 512-passage profile cannot satisfy the full study's frozen inputs.

Each notebook validates its own prerequisites and makes only its declared calls. It never silently rebuilds missing document embeddings or starts a parent study. Prepared document vectors are reusable inputs; Live question embeddings, reranking, answers, judging and benchmark timings belong to the new run. Recorded numerical prose is labeled **Recorded reference run**; inspect the new results tables for current findings.

The same workflow can run from the terminal, saving in the main notebook:

```bash
.venv/bin/python tools/run_tutorial.py --study demo --mode plan --inputs .cache/tutorial-inputs/small/tutorial_inputs.json
.venv/bin/python tools/run_tutorial.py --study demo --inputs .cache/tutorial-inputs/small/tutorial_inputs.json
```

`--study` accepts `demo`, `baseline`, `diagnosis`, `targeted`, `answers`, `latency`. `--inputs` and `TUTORIAL_INPUTS` select the same declaration file; paths are resolved from the project root, including when the kernel starts in a notebook subfolder.

## 4. Execution modes and request caps

- **Live (default):** a new ID creates fresh inference under `.cache/tutorial-runs/<study>/<run-id>/`. Selecting Live authorizes the declared notebook calls; there is no additional paid-call switch.
- **Replay:** set `NOTEBOOK_MODE=replay` and an explicit completed `TUTORIAL_RUN_ID`. It reads that run's saved questions/results and makes no hosted calls. Matching input/run artifacts are required locally; input preparation alone cannot recreate old answers. To read committed displays, just open the notebook without executing it.
- **Plan:** reports the workflow without credentials, downloads, inference or notebook writes.
- **Resume:** set Live, the existing run ID, and `TUTORIAL_RESUME=1`. Only successes from that exact run are reused. A partial latency benchmark requires a fresh run ID to avoid treating cached HTTP responses as new timings.

The project-wide ledger remains `.cache/model-request-budget.sqlite` with a **10,000-attempt cap**, including failures/retries. Ordinary retrieval embeddings and reranking have up to 3 attempts, initial generation 1, and latency up to 2. Existing RAGAS caps remain in force; failed/missing metrics are reported separately, never as valid zero scores. SDK automatic retries are disabled. Caps count requests, not dollars.

The tutorial SDK transport honors the cooldown for a temporary OpenRouter 402 explicitly marked `in_flight_budget_exhausted`, within the existing attempt cap. Generic insufficient-credit 402 responses are not retried. After inspecting a stopped temporary response, `--resume --retry-in-flight` may use remaining attempts. A separate request-specific retry authorization requires explicit approval and retains the same request identity and counter; it does not reset global accounting.

For checkpointed legacy helpers, raw artifacts and prior-run reproduction, see [Advanced workflows](docs/advanced-workflows.md).
