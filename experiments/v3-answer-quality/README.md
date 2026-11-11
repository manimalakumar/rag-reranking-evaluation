# Experiment 3: Answer Quality

Use the reference profile. Live generates and scores 79 pairs on frozen contexts: 25 opportunity gains, 25 unchanged, 4 losses and 25 all-support controls. The 25-question cap and limited/empty groups explain unequal sizes. These assignments come from the frozen inputs, not a new targeted run. IMEI is already included. A separate frozen 12-question, three-prompt extension adds 36 answers under a pinned provider; inspect its current scores and completion status before drawing conclusions.

1. Follow [setup](../../setup.md#1-install-the-environment-and-kernel).
2. Prepare the required public [input profile](../../setup.md#2-prepare-the-prerequisites) and export `TUTORIAL_INPUTS` and your API key before launching the kernel.
3. Open [the notebook](Experiment-3-Answer-Quality.ipynb), select the project kernel and keep Live mode with a new run ID.
4. **Restart Kernel → Run All → Save.** The main notebook retains its actual execution outputs.

Saved displays and numerical prose labeled **Recorded reference run** describe that execution; a new Live run’s computed tables supply its findings. Opening a notebook without executing it makes no calls. Exact offline Replay requires that selected run’s artifacts; public preparation rebuilds inputs, not prior model responses. See the [project guide](../../README.md) for modes, scope and recorded completion status.
