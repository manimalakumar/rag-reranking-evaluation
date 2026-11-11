# Experiment 4: Latency Tradeoff

Use the reference profile. Live times eight question pairs, two per populated subgroup, and scores those exact sixteen answers. Each arm embeds its own query, retrieves thirty passages and generates from five; the reranked arm also reranks. Document/index preparation is outside the timed paths. Failed attempts and backoff remain in wall time. Use the current measurements for retries and latency findings; neither IMEI nor effervescent is in this sample.

1. Follow [setup](../../setup.md#1-install-the-environment-and-kernel).
2. Prepare the required public [input profile](../../setup.md#2-prepare-the-prerequisites) and export `TUTORIAL_INPUTS` and your API key before launching the kernel.
3. Open [the notebook](Experiment-4-Latency-Tradeoff.ipynb), select the project kernel and keep Live mode with a new run ID.
4. **Restart Kernel → Run All → Save.** The main notebook retains its actual execution outputs.

Saved displays and numerical prose labeled **Recorded reference run** describe that execution; a new Live run’s computed tables supply its findings. Opening a notebook without executing it makes no calls. Exact offline Replay requires that selected run’s artifacts; public preparation rebuilds inputs, not prior model responses. See the [project guide](../../README.md) for modes, scope and recorded completion status.
