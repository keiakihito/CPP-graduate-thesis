01 - The thesis would benefit from including the architecture figures shown during the defense. [x]

02 - Figures and tables should be explicitly referenced and discussed in the body of the text before or near their appearance.

03 - If possible, including qualitative retrieval examples or case studies would strengthen the discussion.

04 - Did you consider incorporating additional song metadata into the retrieval pipeline? It would be interesting to discuss whether metadata could complement the audio embeddings in cold-start settings.

05 - Could you elaborate on the motivation for using Hit@K as a primary evaluation signal? Since Hit@K reduces retrieval success to a binary outcome, it may be too coarse to distinguish meaningful ranking differences between models compared to metrics such as MAP@K or NDCG@K.

06 - The evaluation uses truncated ranking metrics (e.g., NDCG@5). It may help to explicitly clarify that the evaluation focuses on the top-K portion of the ranking rather than the full ranked list and discuss the rationale behind the choice of K=5.

07 - NDCG is typically motivated by graded relevance settings, whereas this work uses binary relevance labels. Could you elaborate on why NDCG@K was prioritized in this setting, and whether MAP@K was considered an alternative evaluation metric?

08 - The character-based relevance definition considers two items relevant if they share at least one label. Given the small label set and potential overlap across many items, this may produce a relatively broad or weak notion of similarity. Did you consider stricter relevance criteria (e.g., requiring multiple shared labels or using similarity thresholds), and how might this affect the observed metric behavior?

09 - Cosine similarity assumes that semantic information is primarily reflected in the direction of embeddings. Since the evaluated models are pretrained under different objectives, could you comment on whether this assumption holds consistently across models and how sensitive the results may be to the choice of similarity metric?

10 - All evaluation metrics used in the experiments should be formally defined, including their mathematical formulations.