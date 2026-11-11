# Experiment 2: Targeted Reranking

Use the reference profile. Live reranks the same thirty candidates for 500 opportunities and 200 controls. It then generates and scores the two declared worked-example pairs. The IMEI walkthrough inspects which support moves into the top five; use the current table for its ranks and recall.

1. Follow [setup](../../setup.md#1-install-the-environment-and-kernel).
2. Prepare the required public [input profile](../../setup.md#2-prepare-the-prerequisites) and export `TUTORIAL_INPUTS` and your API key before launching the kernel.
3. Open [the notebook](Experiment-2-Targeted-Reranking.ipynb), select the project kernel and keep Live mode with a new run ID.
4. **Restart Kernel → Run All → Save.** The main notebook retains its actual execution outputs.

Saved displays and numerical prose labeled **Recorded reference run** describe that execution; a new Live run’s computed tables supply its findings. Opening a notebook without executing it makes no calls. Exact offline Replay requires that selected run’s artifacts; public preparation rebuilds inputs, not prior model responses. See the [project guide](../../README.md) for modes, scope and recorded completion status.
