# Experiment 3: Answer Quality

Does better evidence selection improve generated answers? Compare **79 paired questions / 158 saved answers**, using each path's own frozen five passages from Experiment 2. Sampling occurs after retrieval/reranking grouping and before answer scoring, using seed 42 and a cap of 25 per subgroup.

| Retrieval subgroup | Questions |
| --- | ---: |
| Opportunity gain | 25 |
| Opportunity unchanged | 25 |
| Opportunity loss | 4 |
| Control: all support already present | 25 |
| Control: some support absent from top thirty | 0 |
| **Total** | **79** |

These deliberately unequal subgroup samples are not a representative estimate of overall traffic. Empty groups add no pairs; scarce groups are not duplicated.

[The notebook](Experiment-3-Answer-Quality.ipynb) contains completed generation, SQuAD Token F1/exact match, and both RAGAS metrics. Token F1 measures reference-word overlap, faithfulness assesses support in the supplied contexts, and answer relevancy assesses alignment with the question. A higher overlap score alone does not prove grounding or correctness. Human answer review remains pending.

The **IMEI walkthrough (1053665)** is already one of the 79 sampled questions. It connects the larger-corpus retrieval gain with the saved answers: Token F1 and faithfulness improve in this case, while answer relevancy decreases. The case illustrates one outcome rather than a claim that every metric improves.

A separate **12-question, three-prompt exploratory extension** found small, mixed changes. The evidence-check prompt slightly improved average faithfulness but reduced Token F1 and relevancy while lengthening answers. The concise prompt slightly improved all three metric averages, but did not shorten answers on average and worsened some individual cases. Neither variant consistently improved answer quality in this small sample.

**Interpreting the three-prompt explocratory extesion:** Possible next steps depend on what limits the answer:
-Missing evidence: improve retrieval or corpus coverage.
-Good evidence, poorly used: adjust prompts, context order, or distracting context.
-Model struggles despite clear evidence and instructions: a more capable generator may help more.

**Serving-provider limitation:** both paths request the same generator model and settings, but OpenRouter may route them to different backends. One saved pair uses SiliconFlow for Embedding RAG and Nebius for the reranked arm. This run does not isolate provider variation from context changes. Provider pinning and a common fallback policy are planned for a future controlled run; current outputs remain as recorded.

**Read the saved outputs without rerunning:** Inspect citations alongside their actual passages to assess support. See [Viewing and reproducing results](../../README.md#viewing-and-reproducing-results) in the project README for clone limitations, environment setup and deliberate execution.
