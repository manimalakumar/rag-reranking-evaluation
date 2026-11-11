# Diagnosis: Where could reranking help?

Use the reference profile. Live embeds the frozen eligible questions, retrieves thirty candidates and measures where Recall@30 exceeds Recall@5. It does not rerank or generate answers.

1. Follow [setup](../setup.md#1-install-the-environment-and-kernel).
2. Prepare the required public [input profile](../setup.md#2-prepare-the-prerequisites) and export `TUTORIAL_INPUTS` and your API key before launching the kernel.
3. Open [the notebook](Diagnosis-Retrieval-Opportunity.ipynb), select the project kernel and keep Live mode with a new run ID.
4. **Restart Kernel → Run All → Save.** The main notebook retains its actual execution outputs.

Saved displays and numerical prose labeled **Recorded reference run** describe that execution; a new Live run’s computed tables supply its findings. Opening a notebook without executing it makes no calls. Exact offline Replay requires that selected run’s artifacts; public preparation rebuilds inputs, not prior model responses. See the [project guide](../README.md) for modes, scope and recorded completion status.
