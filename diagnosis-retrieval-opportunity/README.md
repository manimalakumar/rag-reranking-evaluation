# Diagnosis: Where could reranking help?

Identify questions where annotated supporting passages are retrieved in the first thirty results but some are absent from the first five: **Recall@30 > Recall@5**. This separates a ranking opportunity from evidence that was never retrieved.

The saved study covers a frozen **152,576-passage corpus** associated with **15,585 question IDs**. Of these questions, **10,242 have annotated support labels**, and **1,085 meet the opportunity condition**. These counts describe the frozen subset, not the full validation split. Annotations are known support labels, not exhaustive judgments of every relevant passage.

[The notebook](Diagnosis-Retrieval-Opportunity.ipynb) inspects saved coverage, support IDs and candidate ranks. Experiment 2 samples 500 opportunities and 200 no-gap controls before reranking, retaining each question's exact thirty candidates.

**Read the saved outputs without rerunning.** For setup, artifact requirements and deliberate rebuilds, see the [project README](../README.md). Some preserved filenames and output captions say “full validation”; their scope here is the frozen subset.
