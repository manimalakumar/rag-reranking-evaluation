# RAG with and without reranking

A compact comparison of **“What is the cause of COVID-19?”** using Embedding RAG and Embedding + reranking RAG. Both paths retrieve thirty candidates from the frozen **152,576-passage corpus**, select five contexts and generate an answer with the recorded Qwen settings.

Open [the notebook](RAG-Retriever-Embedding-Reranker.ipynb) to inspect the saved answers, passage citations, exact contexts and request records. **No rerun or model calls are needed to read the outputs.** This demo uses the larger corpus, rather than Experiment 1's smaller baseline corpus.

For environment setup and deliberate execution, see the [project README](../../README.md). Re-execution requires matching artifacts or explicit permission settings for hosted calls; merely opening the notebook does not make calls.
