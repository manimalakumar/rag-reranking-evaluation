# Advanced workflows and earlier checkpointed studies

Start new tutorial runs with [setup → prerequisites → open notebook → Run All](../setup.md). These notes concern existing checkpointed research artifacts; they are not prerequisites for the public small/reference preparation routes.

## Artifact locations and offline replay

- `.cache/tutorial-inputs/`: public preparation outputs selected with `TUTORIAL_INPUTS` or `--inputs`.
- `.cache/tutorial-runs/<study>/<run-id>/`: new Live inference, questions, scores, timed paths and provenance.
- `.cache/qwen-runs/`: earlier checkpointed pipeline artifacts and original upstream inputs.

Preserve each run's directory structure, exact five-passage contexts, configurations and persistent budget ledger. For exact offline Replay, restore the matching run and its upstream inputs from a separately distributed artifact. Do not commit large vectors, indexes, datasets, raw request traces or credentials. A downloadable bundle, if published separately, should carry a manifest and checksum; no hosted bundle URL is claimed here.

Public preparation reconstructs compatible **inputs**, not identical hosted responses. The pinned data recipe preserves passage membership, text and experimental selection. Serving providers, embedding responses, generated answers, metrics and timings can change. New answer-quality inference uses the declared frozen contexts, not a resampling of a new targeted run.

## Earlier rebuild CLI

`tools/rebuild_results.py` is retained for the earlier checkpointed study format. Its implicit parent preparation and notebook-export adapter predate the current Live/Replay runner. It is **not the fresh-clone tutorial launch route** and does not promise that an earlier study's run ID is a Replay ID in `.cache/tutorial-runs/`.

Its verification-only entry point remains:

```bash
.venv/bin/python tools/rebuild_results.py
```

Older usage passed `--allow-model-calls`, `--study baseline|diagnosis|targeted|answers|latency|all`, and optionally `--judge`. Those paths depend on their configuration and checkpoint assets; the current `retrieval_scope.json` names an existing embedding-prefix snapshot. Do not enable them expecting a missing private cache to bootstrap automatically. Their notebook-export adapter needs separate adaptation before treating a legacy rebuild as a current main-notebook execution. Use `tools/prepare_tutorial_inputs.py` and `tools/run_tutorial.py` for the supported fresh-input and notebook paths.

## Recorded execution notes

The committed execution report identifies which reference notebooks completed and which extension work remained incomplete. Earlier compact files under experiment output folders describe their own recorded runs. They are retained for provenance and are not substituted as the results of a new Live run.

The initial frozen-input preparation and study were recorded at approximately US$1.20 in API usage. That is not a quote for a new full preparation or Live rerun. Costs depend on inputs, provider pricing, tokens, retries and judging calls.
