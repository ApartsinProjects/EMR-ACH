[School of Software Engineering: Intelligent Systems]{.underline}

Automating Structural Analytics Techniques for Event Forecasting using
Large Language Models

A project report submitted toward the degree of

Master of Science in Intelligent Systems

**Student name: Ben Remez**

**Supervisor:** **Dr Yehudit Aperstein**[]{.paragraph-insertion
author="Ben Remez" date="2025-08-24T21:45:00Z"}

[**Advisor: Dr. Sasha Aperstein**]{.insertion author="Ben Remez"
date="2025-08-24T21:45:00Z"}

**Date: 26.8.25**

# Table of Contents {#table-of-contents .TOC-Heading .unnumbered}

[1. Introduction]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[3]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[2. Related Work]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[4]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[2.1. Introduction to Structural Analytics and Event
Forecasting]{.insertion author="Ben Remez" date="2025-08-26T01:08:00Z"}
[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[4]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[2.2. ACH and Evidence-Based Decision-Making]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"} []{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[4]{.insertion
author="Ben Remez" date="2025-08-26T11:40:00Z"}[]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}

[2.3. Large Language Models (LLMs) in Decision Support and
Forecasting]{.insertion author="Ben Remez" date="2025-08-26T01:08:00Z"}
[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[5]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[2.4. MIRAI Benchmark and Event Forecasting Tasks]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"} []{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[7]{.insertion
author="Ben Remez" date="2025-08-26T11:40:00Z"}[]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}

[2.5. Probabilistic Reasoning and Evidence-Based Forecasting]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"} []{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[8]{.insertion
author="Ben Remez" date="2025-08-26T11:40:00Z"}[]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}

[3. Novelty and Contributions of the Research]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"} []{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[9]{.insertion
author="Ben Remez" date="2025-08-26T11:40:00Z"}[]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}

[4. Technologies and Algorithms Used]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[9]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[4.1. LangChain]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[9]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[4.2. Weaviate]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[10]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[4.3. Maximal Marginal Relevance (MMR)]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[10]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[4.4. Reciprocal Rank Fusion (RRF)]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[11]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[5. Retrieval-Augmented Generation (RAG)]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[11]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[6. Dataset]{.insertion author="Ben Remez" date="2025-08-26T01:08:00Z"}
[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[12]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[7. Event Forecasting Benchmark]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[13]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[8. Methodology]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[14]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[9. Deep Analysis]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[17]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[9.1. Motivation]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[17]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[9.2. Approach]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[17]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[9.3. Impact on the Pipeline]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[18]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[10. Initial Experimentation]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[18]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[11. Experiments]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[21]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[11.1. First Experiment - Baseline Setup]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[22]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[11.1.1. Retrieval Methods]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[22]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[11.1.2. Findings]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[23]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[11.2. Second Experiment -- Improved Indicators & Enhanced
RAG]{.insertion author="Ben Remez" date="2025-08-26T01:08:00Z"}
[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[23]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[11.2.1. Balanced Indicator Generation]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[23]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[11.2.2. Enhanced RAG Retrieval Infrastructure]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"} []{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[24]{.insertion
author="Ben Remez" date="2025-08-26T11:40:00Z"}[]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}

[11.2.3. Findings]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[25]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[11.3. Third Experiment -- Integrating Deep Analysis]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"} []{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[26]{.insertion
author="Ben Remez" date="2025-08-26T11:40:00Z"}[]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}

[11.3.1. Approach]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[26]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[11.3.2. Findings]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[26]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[11.4. Experiments Summary]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[27]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[12. Results]{.insertion author="Ben Remez" date="2025-08-26T01:08:00Z"}
[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[27]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[12.1. Initial Results (Baseline Experiment)]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"} []{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[28]{.insertion
author="Ben Remez" date="2025-08-26T11:40:00Z"}[]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}

[12.2. Results After Fine-Tuned Indicators]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"} []{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[29]{.insertion
author="Ben Remez" date="2025-08-26T11:40:00Z"}[]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}

[12.3. Results After Introducing Deep Analysis]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"} []{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[31]{.insertion
author="Ben Remez" date="2025-08-26T11:40:00Z"}[]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}

[13. Discussion]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[33]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[13.1. Interpretation of Findings]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[33]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[13.2. Trade-offs Between Manual and RAG Pipelines]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"} []{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[33]{.insertion
author="Ben Remez" date="2025-08-26T11:40:00Z"}[]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}

[13.3. Role of Deep Analysis]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[34]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[13.4. Scalability and Practical Implications]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"} []{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[34]{.insertion
author="Ben Remez" date="2025-08-26T11:40:00Z"}[]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}

[13.5. Limitations and Challenges]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[34]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[13.5.1. Abstract-Only Context]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[34]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[13.5.2. Resource Constraints]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[34]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[13.5.3. Outcome Grouping]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[34]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[13.5.4. RAG Limitations vs. Manual Quality]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"} []{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[35]{.insertion
author="Ben Remez" date="2025-08-26T11:40:00Z"}[]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}

[14. Conclusions]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[35]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[15. Future Work]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[36]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[15.1. Benchmark on Larger and Diverse Datasets]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"} []{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[36]{.insertion
author="Ben Remez" date="2025-08-26T11:40:00Z"}[]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}

[15.2. Refine RAG Retrieval]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[36]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[15.3. Explore Multi-Agent LLM Architectures]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"} []{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[36]{.insertion
author="Ben Remez" date="2025-08-26T11:40:00Z"}[]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}

[15.4. Full Article Analysis]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[36]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[15.5. Extend CAMEO Relations]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[36]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[15.6. Optimize Prompt Engineering]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[36]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[16. References]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[37]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[17. Appendix A -- Deep Analysis Prompt]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[40]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[18. Appendix B -- Entropy Analysis of Prediction
Distributions]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[40]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[18.1. Methodology]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[40]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[18.2. Results]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[41]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[18.3. Interpretation]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[41]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[19. Appendix C -- Threshold-Based Deep Analysis Exploration]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"} []{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[41]{.insertion
author="Ben Remez" date="2025-08-26T11:40:00Z"}[]{.insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-26T01:08:00Z"}

[19.1. Motivation]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[41]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[19.2. Methodology]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[41]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[19.3. Results]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[42]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[19.4. Interpretation]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"} []{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[43]{.insertion author="Ben Remez"
date="2025-08-26T11:40:00Z"}[]{.insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T01:08:00Z"}

[1. Introduction 1]{.deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}

[2. Related Work 2]{.deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}

[2.1. Introduction to Structural Analytics and Event Forecasting
2]{.deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}

[2.2. ACH and Evidence-Based Decision-Making 2]{.deletion
author="Ben Remez" date="2025-08-26T00:24:00Z"}[]{.paragraph-deletion
author="Ben Remez" date="2025-08-26T00:24:00Z"}

[2.3. Large Language Models (LLMs) in Decision Support and Forecasting
4]{.deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}

[2.4. MIRAI Benchmark and Event Forecasting Tasks 5]{.deletion
author="Ben Remez" date="2025-08-26T00:24:00Z"}[]{.paragraph-deletion
author="Ben Remez" date="2025-08-26T00:24:00Z"}

[2.5. Probabilistic Reasoning and Evidence-Based Forecasting
6]{.deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}

[3. Novelty and Contributions of the Research 7]{.deletion
author="Ben Remez" date="2025-08-26T00:24:00Z"}[]{.paragraph-deletion
author="Ben Remez" date="2025-08-26T00:24:00Z"}

[4. Dataset]{.deletion author="Ben Remez" date="2025-08-26T00:24:00Z"}
[7]{.deletion author="Ben Remez"
date="2025-08-26T00:23:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}

[5. Event Forecasting Benchmark]{.deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"} [9]{.deletion author="Ben Remez"
date="2025-08-26T00:23:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}

[6. Methodology]{.deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"} [10]{.deletion author="Ben Remez"
date="2025-08-26T00:23:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}

[6. Initial Experimentation]{.deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"} [12]{.deletion author="Ben Remez"
date="2025-08-26T00:23:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}

[7. Test Experiment Design and Execution]{.deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"} [15]{.deletion author="Ben Remez"
date="2025-08-26T00:23:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}

[8. Results]{.deletion author="Ben Remez" date="2025-08-26T00:24:00Z"}
[16]{.deletion author="Ben Remez"
date="2025-08-26T00:23:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}

[9. Current Limitations]{.deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"} [18]{.deletion author="Ben Remez"
date="2025-08-26T00:23:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}

[10. Conclusions and Future Work]{.deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"} [18]{.deletion author="Ben Remez"
date="2025-08-26T00:23:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}

[11. References]{.deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"} [18]{.deletion author="Ben Remez"
date="2025-08-26T00:23:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-26T00:24:00Z"}

# Introduction

Structural analytical techniques, such as Analysis of Competing
Hypotheses (ACH), provide a systematic framework for evaluating complex
problems by organizing information, identifying biases, and assessing
multiple competing explanations. ACH, in particular, encourages analysts
to consider alternative hypotheses, weigh evidence objectively, and
avoid cognitive [pitfalls like]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:54:00Z"}[biases such as]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:54:00Z"} confirmation
bias. By structuring the analysis around competing scenarios, ACH
enhances transparency, reduces uncertainty, and improves
decision-making. This technique is especially valuable in intelligence,
business, and policy analysis, where accurate conclusions are critical.
The benefits of ACH and similar structural techniques include fostering
critical thinking, promoting collaboration, and delivering more robust,
evidence-based outcomes. These methods [help analysts navigate ambiguity
and make well-informed]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:54:00Z"}[enable analysts to navigate ambiguity and
make informed]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:54:00Z"} judgments in dynamic environments.

This research aims to leverage large language models (LLMs) to analyze
diverse information sources and enhance the accuracy of forecasting
future outcomes. The forecasting task is structured as a multiple-choice
question, where each answer choice represents an attribute of a future
event. The study utilizes a dataset of recent news articles to achieve
this, enabling the model to generate informed predictions based on the
[latest available information]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:54:00Z"}[most up-to-date information
available]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:54:00Z"}.

Using large language models (LLMs) in event forecasting offers several
advantages. Firstly, LLMs mitigate the influence of human biases,
ensuring that predictions are based solely on data-driven insights
rather than subjective interpretations. Secondly, they can process and
analyze vast amounts of information efficiently, comprehensively
evaluating multiple sources. Lastly, LLMs leverage extensive common
knowledge and factual information in their training data, enhancing
their ability to generate well-informed and contextually relevant
forecasts.

# Related Work

## Introduction to Structural Analytics and Event Forecasting

Structured analytical techniques provide a systematic framework for
evaluating complex problems by organizing information, mitigating
cognitive biases, and improving decision-making consistency. These
techniques are [particularly useful]{.deletion
author="Alexander Apartsin"
date="2025-08-27T11:23:00Z"}[instrumental]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:23:00Z"} in intelligence
analysis, risk assessment, and forecasting, where uncertainty is high,
and human judgment alone may introduce inconsistencies. One widely used
structural method is the Analysis of Competing Hypotheses (ACH), which
systematically evaluates multiple possible explanations or predictions
by assessing how well available evidence supports or contradicts each
hypothesis. ACH helps analysts minimize cognitive biases, such as
confirmation bias, by requiring them to [consider disconfirming evidence
systematically]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:54:00Z"}[systematically consider disconfirming
evidence]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:54:00Z"}.

However, it remains a manual, time-consuming process prone to human
error and inconsistencies, especially when handling large-scale,
unstructured data. Analysts often deviate from ACH's steps, [increasing
bias and reducing]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:54:00Z"}[which increases bias and
reduces]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:54:00Z"} reliability \[9\]. Cognitive bias
\[4[3]{.deletion author="Ben Remez" date="2025-08-26T09:52:00Z"}\] is
[one of the main issues]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:54:00Z"}[a significant issue]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:54:00Z"} in the
decision-making process. There are [many cognitive factors
that]{.deletion author="Alexander Apartsin" date="2025-08-27T11:23:00Z"}
[might affect the human judgement in decision making, thus, a systematic
framework can assist in reducing the cognitive bias and allow]{.deletion
author="Alexander Apartsin" date="2025-08-27T10:54:00Z"} [analysts to
make more]{.deletion author="Alexander Apartsin"
date="2025-08-27T11:23:00Z"} [numerous cognitive factors that can
influence human judgment in decision-making; therefore, a systematic
framework can help mitigate cognitive bias and enable analysts to make
more informed,]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:23:00Z"} evidence-based decisions. MacLean C.L.
[emphasize]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:54:00Z"} [emphasizes]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:54:00Z"} the importance
of bias mitigation and offers a framework to reduce it [and
support]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:54:00Z"}[, supporting]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:54:00Z"} workplace
investigators in [the development of]{.deletion
author="Alexander Apartsin"
date="2025-08-27T11:23:00Z"}[developing]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:23:00Z"} evidence-based,
consistent [decision making]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:54:00Z"}[decision-making]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:54:00Z"}
\[24[19]{.deletion author="Ben Remez" date="2025-08-26T09:55:00Z"}\].

## ACH and Evidence-Based Decision-Making 

In the late 70ts, [the]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:55:00Z"} CIA developed a methodology for reducing
cognitive bias during the analysis of intelligence information. ACH
helps to [reduce]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:55:00Z"} [minimize]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:55:00Z"} a
[high-stake]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:55:00Z"}[high-stakes]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:55:00Z"} decision into a
series of smaller and, arguably, simpler decisions regarding individual
[information pieces]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:55:00Z"}[pieces of information]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:55:00Z"}. However, the
overall process is manual[, and, thus, time-consuming and still
relies]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:55:00Z"} [and, thus, time-consuming, still
relying]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:55:00Z"} on human judgment. The method is also
problematic when intelligent information is deceptive or unreliable.
Therefore, more formal methodologies based on Bayesian networks
\[16[4]{.deletion author="Ben Remez" date="2025-08-26T09:54:00Z"}\] and
other probabilistic models \[28[3]{.deletion author="Ben Remez"
date="2025-08-26T09:56:00Z"}\] have been introduced for the task.
However, more formal methods require [a translation of]{.deletion
author="Alexander Apartsin"
date="2025-08-27T10:55:00Z"}[translating]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:55:00Z"} mostly
unstructured language data (e.g.[,]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:55:00Z"} news or
reports) into well-defined random variables. The framing of ill-posed
data based on a large corpus of intelligence language and visual data
poses a serious challenge for the application of formal probabilistic
reasoning methods.

Intelligence analysts are required to assess evidence to test
alternative accounts of a current or future situation. [In performing
such a cognitively complex task, analysts may resort to using simple
strategies that can bias their thinking and result in]{.deletion
author="Alexander Apartsin" date="2025-08-27T10:55:00Z"}[When performing
such a cognitively complex task, analysts may resort to using simple
strategies that can bias their thinking and lead to]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:55:00Z"}
[judgement]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:55:00Z"} [judgment]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:55:00Z"} errors. It is
argued that analysts may [suffer from "]{.deletion
author="Alexander Apartsin" date="2025-08-27T10:55:00Z"}[be susceptible
to "]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:55:00Z"}confirmation bias" \[5[4]{.deletion
author="Ben Remez" date="2025-08-26T09:52:00Z"}\]. Dhami M. K. et al.
[performed an experiment]{.deletion author="Alexander Apartsin"
date="2025-08-27T11:23:00Z"}[experimented]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:23:00Z"} [in which they
evaluated the difference between analysts who were trained to use ACH
and analysts]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:55:00Z"}[to evaluate the difference between analysts
who were trained to use ACH and those]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:55:00Z"} who were not in
several categories.

In their experiment, they aimed to compare the accuracy between the
groups[, and to measure the extent of within-individual consistency in
the judgement]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:55:00Z"} [and to measure the extent of
within-individual consistency in the judgment]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:55:00Z"} processes of
the ACH and untrained groups \[13[2]{.deletion author="Ben Remez"
date="2025-08-26T09:54:00Z"}\]. Their results suggest that most analysts
trained (and instructed) to use ACH deviated from one or more of the
steps prescribed by this technique, highlighting the value of an
automated system to perform the steps for [hypotheses evaluation
systematically, avoiding missing steps and causing potential errors in
the final decision making]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:55:00Z"}[hypothesis evaluation systematically,
avoiding missing steps and causing potential errors in the final
decision-making]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:55:00Z"}. As people are more prone to such mistakes,
an automated system with a [well-set pipeline will have substantially
less probability for]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:55:00Z"}[well-defined pipeline will have
substantially less probability of making]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:55:00Z"} mistakes of
this kind.

A few online and downloadable software tools help automate the ACH
process. These programs leave a visual trail of evidence [and allow the
analyst to weigh]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:55:00Z"} [, allowing the analyst to weigh
the]{.insertion author="Alexander Apartsin" date="2025-08-27T10:55:00Z"}
evidence. PARCH ACH 2.0 was developed by [Palo Alto Research Center
(PARC) in collaboration with Richards J. Heuer , Jr. It is a standard
ACH program that allows analysts to ender]{.deletion
author="Alexander Apartsin" date="2025-08-27T10:55:00Z"}[the Palo Alto
Research Center (PARC) in collaboration with Richard J. Heuer, Jr. It is
a standard ACH program that allows analysts to enter]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:55:00Z"} evidence and
rate its credibility and relevance. Analysis of Competing Hypotheses
(ACH) is an open-source [ACH implementation which can be referred
to]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:56:00Z"}[implementation of ACH that can be
accessed]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:56:00Z"} from the project's website at
https://competinghypotheses.org/. DECIDE is another existing
computational implementation of ACH. It was developed by the analytic
research firm SSS Research, Inc. DEDICE not only allows analysts to
manipulate ACH, but it [also]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:56:00Z"} provides multiple visualization products
\[1\].

One of the [foremost challenges, particularly for inexperienced
qualitative researchers, lies in]{.deletion author="Alexander Apartsin"
date="2025-08-27T11:24:00Z"}[primary challenges, particularly for
inexperienced qualitative researchers, is]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:24:00Z"} [how to manage
and analyse]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:56:00Z"}[managing and analyzing]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:56:00Z"} large volumes
of qualitative data collected. The integration of quantitative
techniques into qualitative research enhances analytical rigor, reduces
bias, and improves decision-making across various fields. Patricia
Martyn's study \[26[1]{.deletion author="Ben Remez"
date="2025-08-26T09:55:00Z"}\] on C-Ratios and matrix analysis
demonstrates how structuring qualitative data can systematically
quantify relationships between themes, ensuring completeness and
objectivity. This approach aligns with prior efforts in intelligence
analysis (ACH frameworks), business strategy (risk matrices), and
computational text analytics (NLP models), all of which aim to structure
complex reasoning processes. By bridging the gap between qualitative
depth and quantitative precision, these methods offer a scalable,
bias-resistant approach to analyzing large volumes of data, making them
essential for research, business intelligence, and policy-making.

## Large Language Models (LLMs) in Decision Support and Forecasting

The merging [abilities of Large Language Models motivate]{.deletion
author="Alexander Apartsin" date="2025-08-27T10:56:00Z"}[capabilities
of]{.insertion author="Alexander Apartsin" date="2025-08-27T10:56:00Z"}
[large language models]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:24:00Z"} [have motivated]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:56:00Z"} a surge in
interest in various applications. One of the most promising directions
[tries to combine multiple LLM model states (aka Agents) that are fed
with different prompts and information for achieving]{.deletion
author="Alexander Apartsin" date="2025-08-27T11:24:00Z"}[attempts to
combine multiple LLM model states (also known as Agents) that are fed
different prompts and information to achieve]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:24:00Z"} a common goal
collaboratively \[28[3]{.deletion author="Ben Remez"
date="2025-08-26T09:56:00Z"}\]. Different conceptual and software
frameworks have been proposed for establishing an ensemble of
specialized agents (e.g. [critic, analyzer, summarizer) while coping
with various challenges (task termination, reliability]{.deletion
author="Alexander Apartsin" date="2025-08-27T10:56:00Z"}[, critic,
analyzer, summarizer) while coping with various challenges (task
termination, reliability,]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:56:00Z"} and evaluation).

Large Language Models (LLMs) possess advanced capabilities in processing
and analyzing vast amounts of data by leveraging their scalability,
contextual understanding, and adaptability. As outlined in Large
Language Models: A Survey \[22[7]{.deletion author="Ben Remez"
date="2025-08-26T09:55:00Z"}\], LLMs excel in natural language
comprehension, multilingual processing, reasoning, and tool utilization,
enabling them to synthesize large datasets, extract key insights, and
support complex analytical tasks. Their emergent abilities, such as
in-context learning and self-improvement, allow them to handle dynamic
and unstructured data efficiently, making them valuable for applications
in intelligence analysis, research synthesis, and decision-making
automation. In the rapidly evolving domain of artificial intelligence
(AI), LLMs have emerged as copilot tools in various applications,
notably in forecasting and anomaly detection. However, despite their
growing prominence, a substantial knowledge gap exists regarding their
comprehensive capabilities and limitations in these contexts. The work
of large language models in forecasting can [basically be
divided]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:57:00Z"}[be broadly categorized]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:57:00Z"} into two types.
One method involves applying large language models directly to time
series predictions, focusing on converting time series data into input
data suitable for the models, such as GPT, Llama, and others. The
application of LLMs across various tasks, including forecasting,
involves a spectrum of innovative approaches, each tailored to optimize
performance and accuracy. These approaches include prompt-based,
fine-tuning, zero-shot, one-shot, few-shot[,]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:57:00Z"} and
more[,]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:57:00Z"} as referred [to]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:19:00Z"} by Su, J. et al
\[32[7]{.deletion author="Ben Remez" date="2025-08-26T09:56:00Z"}\]. It
is also [mentioned that there are some challenges for LLMs in the
realm]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:57:00Z"}[noted that LLMs face some challenges in the
realms]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:57:00Z"} of forecasting and anomaly detection. [In
order to perform high quality forecasting using LLMs, an extensive
historical dataset is required]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:57:00Z"}[To perform high-quality forecasting using
LLMs, an extensive historical dataset is necessary]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:57:00Z"}. The reliance
of LLMs on extensive datasets not only necessitates the availability of
vast amounts of data but also raises critical issues regarding the
representativeness, quality[,]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:57:00Z"} and bias inherent in the collected
information. For accurate forecasting with LLMs, access to a
comprehensive historical dataset is essential. These models depend
heavily on large-scale data, making the availability of vast amounts of
information a key requirement. However, this reliance also introduces
significant challenges related to the representativeness, quality, and
potential biases within the collected data, which can impact the
reliability and fairness of the predictions. Hallucinations, instances
where models generate false or misleading information, also [impose a
big threat on]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:57:00Z"}[pose a]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:57:00Z"}
[significant]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:24:00Z"} [threat to]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:57:00Z"} the reliability
of LLMs. These errors can [undermine trust in AI-driven forecasting,
decision-making, and analysis, especially in critical domains
like]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:57:00Z"}[erode trust in AI-driven forecasting,
decision-making, and analysis, particularly in critical domains such
as]{.insertion author="Alexander Apartsin" date="2025-08-27T10:57:00Z"}
finance, healthcare, and intelligence. Addressing hallucinations through
improved training methods, enhanced fact-checking mechanisms, and human
oversight is essential to ensure the accuracy and credibility of
LLM-generated insights.

Another development in the field of Artificial Intelligence (AI),
Retrieval-Augmented Generation (RAG), is a method that enhances LLMs by
retrieving external knowledge before generating responses[,]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:24:00Z"} and can help
make the decision-making process more reliable. There are several
[approached]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:57:00Z"} [approaches]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:57:00Z"} when using RAG.
Naive RAG [which follows a simple retrieve-read framework, Advanced RAG
which optimizes retrieval with indexing and metadata, and Modular RAG
which adds]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:57:00Z"}[, which follows a simple retrieve-read
framework, Advanced RAG which op, which optimizes retrieval with
indexing and metadata, and Modular RAG, which adds]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:57:00Z"} search, memory,
and validation modules for adaptability \[12[0]{.deletion
author="Ben Remez" date="2025-08-26T09:54:00Z"}\]. These approaches
[improve accuracy, transparency, and scalability, 5 reducing
hallucinations and enabling data-driven, structured decision-making in
fields like]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:57:00Z"}[enhance accuracy, transparency, and
scalability, thereby reducing hallucinations and facilitating
data-driven, structured decision-making in fields such as]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:57:00Z"} business,
policy, and risk assessment.

## MIRAI Benchmark and Event Forecasting Tasks

Evaluating large language models on event forecasting tasks is [not easy
and there are many researches]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:58:00Z"}[challenging, and]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:58:00Z"} [extensive
research has been conducted]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:24:00Z"} []{.deletion author="Alexander Apartsin"
date="2025-08-27T11:24:00Z"}in this field. A proposed solution for this
task is the MIRAI benchmark. MIRAI stands for Multi-Information
Fo[r]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:58:00Z"}[R]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:58:00Z"}ecasting Agent Interface. MIRAI is a
benchmarking framework designed to evaluate LLM-based forecasting in the
context of international events. Developed to address the limitations of
current forecasting models, MIRAI provides a structured [agentic
environment where LLMs interact with relational databases and textual
news sources through APIs, allowing them to autonomously collect,
process]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:58:00Z"}[, agentic environment where LLMs interact
with relational databases and textual news sources through APIs,
allowing them]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:58:00Z"} [to autonomously collect, process
autonomously]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:24:00Z"}, and reason over historical data. The
benchmark utilizes a refined version of the GDELT event database and
implements structured prediction tasks across various forecasting
horizons. By evaluating models based on their ability to source critical
information, execute code-based reasoning, and predict future events,
MIRAI establishes a rigorous framework for assessing LLM reliability in
structured decision-making and geopolitical forecasting
\[3[7]{.insertion author="Ben Remez"
date="2025-08-26T11:37:00Z"}[3]{.deletion author="Ben Remez"
date="2025-08-26T11:22:00Z"}\]. The MIRAI benchmark framework utilizes
the Conflict and Mediation Event Observations (CAMEO) ontology. CAMEO is
a framework for coding event data [typically used for events that merit
news coverage and]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:58:00Z"} [, typically used for events that merit
news coverage, and is]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:58:00Z"} generally applied to the study of political
news and violence \[6[5]{.deletion author="Ben Remez"
date="2025-08-26T09:53:00Z"}\].

Several studies have explored event datasets for geopolitical
forecasting, each offering unique solutions. Benjamin, D.M. et al. \[2\]
integrate time-series data with human-machine hybrid models to enhance
prediction accuracy and reduce forecasting errors. Kejriwal, M.
\[17[5]{.deletion author="Ben Remez" date="2025-08-26T09:55:00Z"}\]
applies representation learning algorithms to the Global Terrorism
Database (GTD) to predict event relationships and model [complex event
semantics]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:58:00Z"}[the semantics of complex events]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:58:00Z"}. Mirtaheri, M.
[et al. \[238\] utilize the Integrated Crisis Early Warning System
(ICEWS) dataset and employs supervised and unsupervised machine learning
techniques to improve]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:58:00Z"}[, et al. \[23\] utilize the Integrated
Crisis Early Warning System (ICEWS) dataset and employ both supervised
and unsupervised machine learning techniques to enhance]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:58:00Z"} predictive
accuracy for geopolitical interactions. Hossain, K. T. et al.
\[15[3]{.deletion author="Ben Remez" date="2025-08-26T09:54:00Z"}\]
leverage sequential text datasets and [introduces]{.deletion
author="Alexander Apartsin" date="2025-08-27T10:58:00Z"}
[introduce]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:58:00Z"} an attention-based LSTM model to detect
early warning signals of geopolitical shifts, offering a structured
approach to precursor identification. These studies collectively
highlight the value of structured event datasets in advancing
geopolitical forecasting through data-driven methodologies. However,
evaluating model predictions for international event forecasting remains
a significant challenge due to the complexity, evolving nature, and
diverse sources of geopolitical data. Many models struggle to integrate
structured knowledge graphs and unstructured textual data, leading to
incomplete or biased predictions. Furthermore, the lack of historical
grounding affects the interpretability and reliability of forecasts,
especially for long-term geopolitical shifts. The dynamic nature of
international relations, influenced by alliances, trade, and conflicts,
further complicates forecasting, making it [difficult for models to
anticipate unexpected geopolitical developments with high
accuracy]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:58:00Z"} [challenging for models to anticipate
unexpected geopolitical developments \[37\] accurately]{.insertion
author="Alexander Apartsin" date="2025-08-27T10:58:00Z"}[\[3]{.deletion
author="Alexander Apartsin" date="2025-08-27T10:58:00Z"}[]{.insertion
author="Ben Remez" date="2025-08-26T11:38:00Z"}[1\]]{.deletion
author="Alexander Apartsin" date="2025-08-27T10:58:00Z"}.

## Probabilistic Reasoning and Evidence-Based Forecasting

Mapping qualitative certainty to quantitative probabilities is essential
for effective communication and decision-making across various
disciplines. Two prominent approaches in this domain are the
Intergovernmental Panel on Climate Change (IPCC) uncertainty framework
and Bayesian statistical methods. The IPCC approach to uncertainty
relies on a calibrated language system that translates expert judgments
into confidence levels and likelihood metrics, providing a structured
yet heuristic-based framework for assessing risks. Unlike
decision-theoretic models, which use Bayesian inference and optimization
techniques, the IPCC primarily employs scenario-driven risk assessments,
often without explicit probability distributions. While effective for
communicating uncertainty, this method has limitations in quantifying
deep uncertainty and ambiguity. Integrating [6]{.deletion
author="Alexander Apartsin" date="2025-08-27T11:19:00Z"}
[six]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:19:00Z"} decision-theoretic approaches, such as
robust control and max-min utility frameworks, could enhance the IPCC's
methodology for structured decision-making across various domains beyond
climate science \[3[6]{.insertion author="Ben Remez"
date="2025-08-26T11:38:00Z"}[0]{.deletion author="Ben Remez"
date="2025-08-26T09:56:00Z"}\]. Fairfield and Charman \[11[9]{.deletion
author="Ben Remez" date="2025-08-26T09:54:00Z"}\] propose a Bayesian
approach to process tracing that systematically translates qualitative
evidence into quantitative probabilities [for]{.deletion
author="Alexander Apartsin" date="2025-08-27T11:24:00Z"}[, thereby
facilitating]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:24:00Z"} stronger causal inference. Their framework
involves defining specific rival hypotheses, assigning prior
probabilities based on background knowledge, and using logarithmic
likelihood scales to evaluate new evidence. By applying Bayes\' rule
iteratively, researchers can refine their conclusions in a structured
and transparent manner, reducing subjective biases in qualitative
analysis. While effective, challenges remain in assigning numerical
probabilities to complex qualitative data.

Managing multiple sources of uncertainty is a fundamental challenge in
structured decision models, requiring methodologies that balance
ambiguity, conflicting data, and unknown probabilities. The Evidential
Reasoning Approach \[10[8]{.deletion author="Ben Remez"
date="2025-08-26T09:54:00Z"}\] addresses this by integrating qualitative
and quantitative assessments through a belief structure, enabling a
systematic aggregation of diverse evidence. [Dempster--Shafer Theory
\[86\] extends Bayesian probability, allowing for]{.deletion
author="Alexander Apartsin" date="2025-08-27T10:59:00Z"}[The
Dempster--Shafer Theory \[8\] extends Bayesian probability,
allowing]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:59:00Z"} the combination of multiple evidence
sources to derive degrees of belief, making it particularly effective
for handling incomplete or conflicting information. Meanwhile, Robust
Decision-Making (RDM) \[29[4]{.deletion author="Ben Remez"
date="2025-08-26T10:43:00Z"}\] evaluates strategies across a wide range
of possible futures, ensuring resilience [under]{.deletion
author="Alexander Apartsin" date="2025-08-27T10:59:00Z"} [in the face
of]{.insertion author="Alexander Apartsin" date="2025-08-27T10:59:00Z"}
deep uncertainty. These approaches [provide structured, adaptive
frameworks that enhance the reliability and robustness of
decision-making in complex,]{.deletion author="Alexander Apartsin"
date="2025-08-27T10:59:00Z"}[offer structured, adaptive frameworks that
enhance the reliability and robustness of decision-making in complex
and]{.insertion author="Alexander Apartsin" date="2025-08-27T10:59:00Z"}
uncertain environments. Traditional probabilistic forecasting models,
such as ARIMA and Gaussian Processes, have long been the standard for
time-series prediction, relying on statistical assumptions and
structured historical data. In contrast, LLM-based structured inference
leverages vast datasets and deep learning architectures to capture
complex patterns and dependencies. While some studies suggest that LLMs
struggle to outperform traditional models in computational efficiency
\[33[2]{.deletion author="Ben Remez"
date="2025-08-26T10:44:00Z"}[8]{.deletion author="Ben Remez"
date="2025-08-26T09:56:00Z"}\], others highlight their potential to
zero-shot extrapolate time series and achieve competitive or superior
results without task-specific training \[13[1]{.deletion
author="Ben Remez" date="2025-08-26T09:54:00Z"}\]. This indicates that
while LLMs introduce new possibilities, they must be carefully evaluated
against established models to justify their computational cost and
effectiveness in structured decision-making.

# [N]{.deletion author="Alexander Apartsin" date="2025-08-27T11:25:00Z"}[ovelty]{.deletion author="Alexander Apartsin" date="2025-08-27T11:24:00Z"} []{.deletion author="Alexander Apartsin" date="2025-08-27T11:25:00Z"}[and]{.deletion author="Alexander Apartsin" date="2025-08-27T11:26:00Z"} Contributions of the Research

This research presents the first integration of Large Language Models
(LLMs) with ACH-based structured reasoning for intelligence analysis.
Unlike prior approaches that rely on manual processes or Bayesian
networks, this study introduces an automated, scalable method for
evaluating competing hypotheses using LLM-driven evidence extraction and
probabilistic reasoning.

A key innovation is the [multiagent]{.deletion
author="Alexander Apartsin" date="2025-08-27T10:59:00Z"}
[multi-agent]{.insertion author="Alexander Apartsin"
date="2025-08-27T10:59:00Z"} LLM system, [where specialized agents
handle evidence retrieval, uncertainty estimation, and bias
mitigation,]{.deletion author="Alexander Apartsin"
date="2025-08-27T11:24:00Z"} [which utilizes specialized agents to
handle evidence retrieval, uncertainty estimation, and bias mitigation,
thereby]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:24:00Z"} enhancing both accuracy and transparency.
Additionally, this study bridges the gap between unstructured
intelligence text and structured probabilistic inference, allowing ACH
to be applied dynamically to real-world intelligence streams.

To ensure rigorous evaluation, this research adapts the MIRAI benchmark,
providing the first standardized testing framework for LLM-based ACH
systems in event forecasting.

The key contributions of this research include:

-   Automating ACH for Scalable Intelligence Analysis: Reduces manual
    workload and cognitive biases while enabling real-time, data-driven
    reasoning.

-   Multiagent LLM System for Structured Decision-Making: Introduces a
    specialized LLM-agent framework for evidence extraction, hypothesis
    ranking, and bias mitigation.

-   Probabilistic Evidence Scoring: Maps qualitative intelligence data
    to structured probability assessments, improving reliability and
    transparency.

-   Scalable Evaluation Benchmark: Adapts the MIRAI dataset as a proxy
    for ACH evaluation, addressing the lack of standardized intelligence
    benchmarks.

# [Technologies]{.insertion author="Ben Remez" date="2025-08-24T21:49:00Z"} [and Alg]{.insertion author="Ben Remez" date="2025-08-26T00:08:00Z"}[o]{.insertion author="Ben Remez" date="2025-08-26T00:09:00Z"}[r]{.insertion author="Ben Remez" date="2025-08-26T00:08:00Z"}[ithms]{.insertion author="Ben Remez" date="2025-08-26T00:09:00Z"} [Used]{.insertion author="Ben Remez" date="2025-08-24T21:49:00Z"}

## [LangChain]{.insertion author="Ben Remez" date="2025-08-24T21:50:00Z"}

> []{.insertion author="Ben Remez"
> date="2025-08-24T21:50:00Z"}[LangChain]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:00:00Z"} []{.insertion
> author="Ben Remez" date="2025-08-24T21:50:00Z"}[is a widely adopted
> open-source framework designed for orchestrating large language model
> (LLM) applications. It offers a modular and composable architecture
> that enables seamless chaining of]{.insertion author="Ben Remez"
> date="2025-08-24T21:51:00Z"} [LLMs, retrievers, prompt templates, and
> other utility components. Its structure facilitates rapid prototyping,
> provider-agnostic integration, and structured orchestration of complex
> AI workflows]{.insertion author="Ben Remez"
> date="2025-08-24T21:52:00Z"} [\[]{.insertion author="Ben Remez"
> date="2025-08-24T22:00:00Z"}[19]{.insertion author="Ben Remez"
> date="2025-08-26T11:16:00Z"}[\]]{.insertion author="Ben Remez"
> date="2025-08-24T22:00:00Z"}[.]{.insertion author="Ben Remez"
> date="2025-08-24T21:52:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-24T21:58:00Z"}
>
> [The advantages of]{.insertion author="Ben Remez"
> date="2025-08-24T22:00:00Z"} [LangChain]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:00:00Z"}[:]{.insertion
> author="Ben Remez" date="2025-08-24T22:00:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-24T22:00:00Z"}
>
> [LangChain's primary strength lies in its flexibility and
> mo]{.insertion author="Ben Remez"
> date="2025-08-24T22:00:00Z"}[dularity. Its standardized interfaces
> allow]{.insertion author="Ben Remez" date="2025-08-24T22:01:00Z"} [for
> the easy integration of different LLMs, retrieval systems, and data
> sources without requiring]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:00:00Z"} [significant refactoring. This
> adaptability accelerates the development of
> Retrieval-Augment]{.insertion author="Ben Remez"
> date="2025-08-24T22:01:00Z"}[ed Generation (RAG) systems and supports
> efficient prompt chaining for complex]{.insertion author="Ben Remez"
> date="2025-08-24T22:02:00Z"} [multi-step]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:00:00Z"} [tasks.
> Furthermore, its rich ecosystem of integrations enables developers to
> leverage a wide variety of embedding providers, vector
> databases]{.insertion author="Ben Remez"
> date="2025-08-24T22:02:00Z"}[, and APIs, making it a robust foundation
> for building scalable and maintainable AI pipelines]{.insertion
> author="Ben Remez" date="2025-08-24T22:03:00Z"} [\[]{.insertion
> author="Ben Remez" date="2025-08-26T10:11:00Z"}[20]{.insertion
> author="Ben Remez" date="2025-08-26T11:29:00Z"}[\]]{.insertion
> author="Ben Remez" date="2025-08-26T10:11:00Z"}[.]{.insertion
> author="Ben Remez" date="2025-08-24T22:03:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-24T22:03:00Z"}

## [Weaviate]{.insertion author="Ben Remez" date="2025-08-24T22:03:00Z"}

> [Weaviate is a production-ready, open-source vector database
> optimized]{.insertion author="Ben Remez" date="2025-08-24T22:03:00Z"}
> [for high-dimensional vector searches. It enables efficient storage
> and retrieval of semantic embeddings, making it ideal for powering
> high-throughput RAG pipelines]{.insertion author="Ben Remez"
> date="2025-08-24T22:04:00Z"} [\[3]{.insertion author="Ben Remez"
> date="2025-08-25T11:24:00Z"}[4]{.insertion author="Ben Remez"
> date="2025-08-26T11:38:00Z"}[\]]{.insertion author="Ben Remez"
> date="2025-08-25T11:24:00Z"}[.]{.insertion author="Ben Remez"
> date="2025-08-24T22:04:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-24T22:04:00Z"}
>
> [The advantages of Weaviate:]{.insertion author="Ben Remez"
> date="2025-08-24T22:04:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-24T22:04:00Z"}
>
> [Weaviate]{.insertion author="Ben Remez" date="2025-08-24T22:04:00Z"}
> [is designed for performance and scalability, offering near real-time
> responses even when querying millions of high-dimensional vectors. Its
> support for hybrid search]{.insertion author="Ben Remez"
> date="2025-08-24T22:05:00Z"}[,]{.insertion author="Ben Remez"
> date="2025-08-24T22:06:00Z"} [combining vec]{.insertion
> author="Ben Remez" date="2025-08-24T22:05:00Z"}[tor similarity with
> keyword-based matching, enhances retrieval precision and
> releva]{.insertion author="Ben Remez"
> date="2025-08-24T22:06:00Z"}[nce. Additionally, Weaviate provides
> mature SDKs, excellent documentation, and strong integration
> capabilities with popular LLMs and embedding models, enabling seamless
> adoption in advanced]{.insertion author="Ben Remez"
> date="2025-08-24T22:07:00Z"} [AI workflows. These qualities make it a
> natural fit for large-scale retrieval tasks]{.insertion
> author="Ben Remez" date="2025-08-24T22:08:00Z"}[, such as]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:24:00Z"} [those in
> this project.]{.insertion author="Ben Remez"
> date="2025-08-24T22:08:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-24T22:17:00Z"}

## [Maximal Marginal Relevance]{.insertion author="Ben Remez" date="2025-08-26T00:09:00Z"} [(MMR)]{.insertion author="Ben Remez" date="2025-08-26T00:10:00Z"}

> [Maximal Marginal Relevance (MMR) is a retrieval algorithm that
> iteratively selects documents based on a trade-off between relevance
> to the query and redundancy with previously selected items, as
> described by Carbonell & Goldstein \[3\]. In this project, MMR was
> applied during document retrieval to ensure that the selected articles
> were not only highly relevant to the forecasting query but also
> provided non-redundant perspectives.]{.insertion author="Ben Remez"
> date="2025-08-26T00:10:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T00:18:00Z"}
>
> [Formally, MMR iteratively selects documents that maximize the
> following objective:]{.insertion author="Ben Remez"
> date="2025-08-26T00:18:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T00:11:00Z"}
>
> $$\arg_{_{}\frac{}{}}\left\lbrack \left(_{} \right)()_{_{}}\left(_{}_{} \right) \right\rbrack$$[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-26T00:14:00Z"}
>
> [Where:]{.insertion author="Ben Remez"
> date="2025-08-26T00:14:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T00:14:00Z"}

-   $$ [= candidate documents]{.insertion author="Ben Remez"
    date="2025-08-26T00:15:00Z"}[]{.paragraph-insertion
    author="Ben Remez" date="2025-08-26T00:15:00Z"}

-   $$ [= already selected documents]{.insertion author="Ben Remez"
    date="2025-08-26T00:15:00Z"}[]{.paragraph-insertion
    author="Ben Remez" date="2025-08-26T00:15:00Z"}

-   $_{}$ [= similarity between]{.insertion author="Ben Remez"
    date="2025-08-26T00:15:00Z"} []{.insertion author="Ben Remez"
    date="2025-08-26T00:16:00Z"}[documents]{.insertion
    author="Alexander Apartsin" date="2025-08-27T11:20:00Z"}
    []{.insertion author="Ben Remez" date="2025-08-26T00:16:00Z"}$_{}$
    [and query]{.insertion author="Ben Remez"
    date="2025-08-26T00:16:00Z"} $$[]{.paragraph-insertion
    author="Ben Remez" date="2025-08-26T00:16:00Z"}

-   $\left(_{}_{} \right)$ [= similarity between documents]{.insertion
    author="Ben Remez"
    date="2025-08-26T00:16:00Z"}[]{.paragraph-insertion
    author="Ben Remez" date="2025-08-26T00:16:00Z"}

-   $$ [= balance factor b]{.insertion author="Ben Remez"
    date="2025-08-26T00:16:00Z"}[etween relevance and
    diversity]{.insertion author="Ben Remez"
    date="2025-08-26T00:17:00Z"}[]{.paragraph-insertion
    author="Ben Remez" date="2025-08-26T00:17:00Z"}

> [In this project, MMR was]{.insertion author="Ben Remez"
> date="2025-08-26T00:17:00Z"} [beneficial]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:24:00Z"} [for reducing
> redundancy in the retrieved articles while maintaining a high coverage
> of diverse, relevant evidence for the LLM]{.insertion
> author="Ben Remez" date="2025-08-26T00:17:00Z"}[.]{.insertion
> author="Ben Remez" date="2025-08-26T00:18:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-26T00:10:00Z"}

## [Reciprocal Rank Fusion (RRF)]{.insertion author="Ben Remez" date="2025-08-26T00:10:00Z"}

> [Reciprocal Rank Fusion (RRF) is a rank aggregation method that
> combines multiple ranked lists]{.insertion author="Ben Remez"
> date="2025-08-26T00:11:00Z"}[, favouring documents that consistently
> rank highly across them, as]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:01:00Z"} []{.insertion author="Ben Remez"
> date="2025-08-26T00:11:00Z"}[proposed initially]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:01:00Z"} [by Cormack
> et al. \[]{.insertion author="Ben Remez"
> date="2025-08-26T00:11:00Z"}[7]{.insertion author="Ben Remez"
> date="2025-08-26T10:25:00Z"}[\]. In our case, RRF was used to merge
> results from recent and historical article searches.]{.insertion
> author="Ben Remez" date="2025-08-26T00:11:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-26T00:18:00Z"}
>
> [The RRF score for a document]{.insertion author="Ben Remez"
> date="2025-08-26T00:18:00Z"} $$ []{.insertion author="Ben Remez"
> date="2025-08-26T00:18:00Z"}[Is]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:20:00Z"} [computed
> as:]{.insertion author="Ben Remez"
> date="2025-08-26T00:18:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T00:18:00Z"}
>
> $$()\sum_{_{}}^{}\frac{}{}$$[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T00:19:00Z"}
>
> [Where:]{.insertion author="Ben Remez"
> date="2025-08-26T00:19:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T00:19:00Z"}

-   $_{}$ [= set of ranks assigned to]{.insertion author="Ben Remez"
    date="2025-08-26T00:19:00Z"} [a]{.insertion
    author="Alexander Apartsin" date="2025-08-27T11:01:00Z"}
    [document]{.insertion author="Ben Remez"
    date="2025-08-26T00:19:00Z"} $$ [across different retrieval
    runs]{.insertion author="Ben Remez"
    date="2025-08-26T00:20:00Z"}[]{.paragraph-insertion
    author="Ben Remez" date="2025-08-26T00:20:00Z"}

-   $$ [= rank position of the document in one retrieval
    list]{.insertion author="Ben Remez"
    date="2025-08-26T00:20:00Z"}[]{.paragraph-insertion
    author="Ben Remez" date="2025-08-26T00:20:00Z"}

-   $$ [= constant to reduce the influence of low-ranked
    documents]{.insertion author="Ben Remez"
    date="2025-08-26T00:20:00Z"}[]{.paragraph-insertion
    author="Ben Remez" date="2025-08-26T00:20:00Z"}

> [We combined results from two time windows:]{.insertion
> author="Ben Remez" date="2025-08-26T00:20:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-26T00:20:00Z"}

-   [Recent documents: up to the target forecast date]{.insertion
    author="Ben Remez"
    date="2025-08-26T00:21:00Z"}[]{.paragraph-insertion
    author="Ben Remez" date="2025-08-26T00:21:00Z"}

-   [Historical documents: up to 30 days before the target
    date]{.insertion author="Ben Remez"
    date="2025-08-26T00:21:00Z"}[]{.paragraph-insertion
    author="Ben Remez" date="2025-08-26T00:21:00Z"}

> [To prioritize timeliness, we applied exponential time-decay weighting
> with]{.insertion author="Ben Remez" date="2025-08-26T00:21:00Z"} $$[,
> giving higher importance to recent evidence. This approach improved
> retrieval quality and]{.insertion author="Ben Remez"
> date="2025-08-26T00:22:00Z"} [significantly contributed to the
> enhanced]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:01:00Z"} [performance.]{.insertion
> author="Ben Remez" date="2025-08-26T00:22:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-24T22:17:00Z"}

# [Retrieval-Augmented Generation (RAG)]{.insertion author="Ben Remez" date="2025-08-24T22:17:00Z"}

[Retri]{.insertion author="Ben Remez"
date="2025-08-24T22:18:00Z"}[e]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:01:00Z"}[val-Augmented Generation (RAG) is an
emerging paradigm that enhances large language models (LLMs) by
combining retrieval-based methods with generative reasoning. Instead
of]{.insertion author="Ben Remez" date="2025-08-24T22:18:00Z"} [relying
solely on the model's pre-trained parameters, RAG dynamically retrieves
relevant external knowledge]{.insertion author="Ben Remez"
date="2025-08-24T22:19:00Z"}[. It integrates]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:01:00Z"} [it into the
generation process, enabling LLMs to provide more accurate,
context-aware, and up-to-date responses]{.insertion author="Ben Remez"
date="2025-08-24T22:19:00Z"} [\[]{.insertion author="Ben Remez"
date="2025-08-24T22:26:00Z"}[12]{.insertion author="Ben Remez"
date="2025-08-26T11:13:00Z"}[\]]{.insertion author="Ben Remez"
date="2025-08-24T22:26:00Z"}[.]{.insertion author="Ben Remez"
date="2025-08-24T22:19:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-24T22:20:00Z"}

[In this project, R]{.insertion author="Ben Remez"
date="2025-08-24T22:20:00Z"}[A]{.insertion author="Ben Remez"
date="2025-08-24T22:26:00Z"}[G was implemented by combining LangChain
for orchestrating the retrieval and reasoning pipeline and Weaviate as
the semantic vector database to perform high-dimensional]{.insertion
author="Ben Remez" date="2025-08-24T22:20:00Z"} [similarity searches.
Relevant document embeddings were stored in Weaviate, enabling the
pipeline to retrieve supporting evidence for each hypothesis and feed it
into the LLM for improved hypothesis scoring and evaluation.]{.insertion
author="Ben Remez" date="2025-08-24T22:21:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-24T22:21:00Z"}

[An essential enable]{.insertion author="Ben Remez"
date="2025-08-24T22:21:00Z"}[r]{.insertion author="Ben Remez"
date="2025-08-24T22:22:00Z"} [of RAG's]{.insertion author="Ben Remez"
date="2025-08-24T22:21:00Z"} [performance is the use of vector
embeddings to represent document content in a high-dimensional semantic
space. Weavia]{.insertion author="Ben Remez"
date="2025-08-24T22:22:00Z"}[te's vector search leverages embeddings
derived from modern neural architectures to contextual and semantic
similarity between documents. These concepts are rooted in]{.insertion
author="Ben Remez" date="2025-08-24T22:23:00Z"}
[foundational]{.insertion author="Ben Remez"
date="2025-08-24T22:27:00Z"} [work on word representation learning by
Mikolov et al.]{.insertion author="Ben Remez"
date="2025-08-24T22:23:00Z"} [\[]{.insertion author="Ben Remez"
date="2025-08-26T11:10:00Z"}[21]{.insertion author="Ben Remez"
date="2025-08-26T11:35:00Z"}[\]]{.insertion author="Ben Remez"
date="2025-08-26T11:10:00Z"}[,]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:20:00Z"} [which demonstrated that distributed
representations in vector space allow semantically related words and
concepts to cluster together.]{.insertion author="Ben Remez"
date="2025-08-24T22:24:00Z"} [By extending this principle to
document-level embeddings, RAG efficiently retrieves contextually
relevant documents even from]{.insertion author="Ben Remez"
date="2025-08-24T22:27:00Z"} [massive]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:24:00Z"}
[corpora.]{.insertion author="Ben Remez"
date="2025-08-24T22:27:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-24T22:30:00Z"}

[In this project, RAG was implemented using LangChain and Weaviate to
enable efficient retrieval of relevant article documents from a large
corpus. LangChain orchestrated the retrieval workflow, while Weaviate
provided semantic vector search capabilities to identify the most
contextually relevant documents. These retrieved articles were then
passed to the LLM, which evaluated each article's context against the
generated indicators and used this information to forecast the expected
outcome for a given date between the two countries involved in the
query. Leveraging these tools allowed the pipeline to ground its
predictions in real, contextually relevant evidence, improving the
accuracy, reliability, and scalability of the forecasting
process.]{.insertion author="Ben Remez"
date="2025-08-24T22:39:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-24T21:50:00Z"}

# Dataset

The experiments in this study were conducted using the MIRAI benchmark
dataset, a publicly available and carefully curated resource designed to
evaluate large language models (LLMs) and reasoning agents in the
context of geopolitical event forecasting. The benchmark was introduced
by Ye et al. \[3[7]{.insertion author="Ben Remez"
date="2025-08-26T11:38:00Z"}[1]{.deletion author="Ben Remez"
date="2025-08-26T09:57:00Z"}\] and builds [on top of the GDELT (Global
Database of Events, Language, and Tone) dataset and the CAMEO (Conflict
and Mediation Event Observations) ontology \[65\], both]{.deletion
author="Alexander Apartsin" date="2025-08-27T11:02:00Z"} [upon the GDELT
(Global Database of Events, Language, and Tone) dataset and the CAMEO
(Conflict and Mediation Event Observations) ontology \[6\], both of
which are]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:02:00Z"} widely used in computational political
science and international relations research.

Each instance in the MIRAI benchmark consists of a structured
forecasting query, defined by the following fields:

-   A source and target country.

-   A forecast date, which serves as the reference point for information
    retrieval.

-   A ground truth event label encoded using a three-digit CAMEO code.

-   Optionally, auxiliary context such as summaries and prompts provided
    to the model.

In total, the dataset includes 100 test queries, each mapped to one of
38 CAMEO relation types representing various forms of cooperation or
conflict. These fine-grained relation types span a spectrum of verbal
and material interactions, allowing nuanced evaluation of forecasting
systems.

To enable both fine-grained and coarse-grained evaluations, the MIRAI
benchmark defines additional relation groupings:

-   **Quad categories**: Collapsing 38 relations into four classes ---
    *Verbal Cooperation*, *Material Cooperation*, *Verbal Conflict*, and
    *Material Conflict*.

-   **Binary categories**: Reducing the label space to two classes ---
    *Cooperation* and *Conflict*.

In this project, we [make use of]{.deletion author="Alexander Apartsin"
date="2025-08-27T11:02:00Z"}[utilize]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:02:00Z"} the MIRAI test
subset to evaluate our ACH-based system. For each query, relevant news
documents are retrieved using a RAG-based pipeline from a local corpus
constructed from real-world news data. This corpus is indexed using
Weaviate, an open-source vector database that supports semantic search
and hybrid retrieval [or manually using a list of document IDs provided
in the test files]{.deletion author="Alexander Apartsin"
date="2025-08-27T11:02:00Z"}[. Alternatively, it can be manually
searched utilizing a list of document IDs provided in the test files,
which are specifically related to each query]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:24:00Z"} [and relates to
each query specifically]{.deletion author="Alexander Apartsin"
date="2025-08-27T11:24:00Z"}.

This experimental setup enables alignment with prior evaluations while
offering flexibility for testing new architectures and reasoning
strategies.

# Event Forecasting Benchmark

![A table with black text AI-generated content may be
incorrect.](media/image1.png){width="6.268055555555556in"
height="2.2569444444444446in"}We utilize the recently published MIRAI
benchmark \[3[7]{.insertion author="Ben Remez"
date="2025-08-26T11:38:00Z"}[1]{.deletion author="Ben Remez"
date="2025-08-26T09:57:00Z"}\], which is crafted to evaluate LLM agents
for temporal forecasting in international events. Formally, we represent
an event as 𝑒𝑡 = (𝑡, 𝑠, 𝑟, 𝑜), where 𝑡 is the timestamp, 𝑠, 𝑜 ∈ 𝐶 are
respectively the subject and object countries from the country pool 𝐶,
and 𝑟 ∈ 𝑅 denotes the relation type defined by the Conflict and
Mediation Event Observations (CAMEO) ontology \[6[5]{.deletion
author="Ben Remez" date="2025-08-26T09:53:00Z"}\]. There are two
hierarchical levels from the CAMEO ontology for detailed and
comprehensive analysis of geopolitical dynamics. The first level
includes 20 broad categories, represented by a two-digit code (e.g.,
"01: Public Statement" or "04: Consult"), which are subdivided into
second-level categories identified by a three-digit code that
corresponds to [its]{.deletion author="Alexander Apartsin"
date="2025-08-27T11:19:00Z"} [their]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:19:00Z"} parent
category. For example, "03: Express intent to cooperate" is a
first-level category with 10 different second-level relations, such as
"036: Express intent to meet or negotiate". Subsequently, the quadruple
"(2023-11-03, AUS, 036, CHN)" denotes that on 3 November 2023, the
Australian leader announced a planned visit to China. Figure 1 presents
a sample of relationships extracted from MIRAI test data.

***Figure 1:** 38 Relations used in the test data*

In the implementation of the MIRAI benchmark, the authors simplified the
evaluation by reducing the granularity of the original 38 CAMEO event
codes present in the test set. This was done to create more
interpretable and robust evaluation categories. Specifically, the event
codes were grouped into coarser relation categories under two schemes:

1.  **Binary Classification** **--** The 38 relations were collapsed
    into two broad categories:

-   **Mediation --** Event codes '01' -- '08'

-   **Conflict --** Event codes '09' -- '20'

2.  **Quad Classification --** The same 38 relations were mapped into
    [4]{.deletion author="Alexander Apartsin"
    date="2025-08-27T11:19:00Z"} [four]{.insertion
    author="Alexander Apartsin" date="2025-08-27T11:19:00Z"} classes:

-   **Verbal Cooperation --** Event codes '01' -- '04'

-   **Material Cooperation --** Event codes '05' -- '08'

-   **Verbal Conflict --** Event codes '09' -- '16'

-   **Material Conflict --** Event codes '17' -- '20'

These mappings formed the basis for evaluating system predictions and
comparing them against ground truth labels using metrics such as
precision, recall, F1 score, and KL divergence. The reduction in label
complexity allows for clearer insight into the model's capabilities
across both broad and fine-grained relational understanding.

# Methodology

The original ACH methods include the construction of the evidence
matrix. The matrix is a key tool for systematically evaluating evidence
against multiple hypotheses. It is structured with hypotheses listed
across the top as columns and evidence listed down the side as rows,
creating a grid that allows analysts to visually compare how each piece
of evidence relates to each hypothesis \[1\]. Within the matrix, symbols
such as \"+\" (support), \"-\" (contradict), or \"0\" (neutral) are used
to indicate the relationship between evidence and hypotheses, ensuring a
clear and organized assessment. This structured approach helps analysts
identify diagnostic evidence---information that strongly supports one
hypothesis while undermining others---and highlights inconsistencies or
gaps in the analysis. By providing a transparent and logical framework,
the ACH matrix reduces cognitive biases, fosters objectivity, and
enables analysts to draw more accurate and well-supported conclusions.

Our approach adds [a]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:20:00Z"} [an additional]{.deletion
author="Alexander Apartsin" date="2025-08-27T11:03:00Z"} step where
evidence is extracted from the relevant sources (new articles).
Therefore, the analysis [includes two types of uncertainties:
uncertainty related to the presence of specified evidence (]{.deletion
author="Alexander Apartsin" date="2025-08-27T11:03:00Z"}[encompasses two
types of uncertainties: uncertainty related to the presence of specified
evidence (an]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:03:00Z"} early sign) in a specific news article and
uncertainty related to the degree to which specific evidence impacts
(supports or disproves) a given hypothesis. Our method addresses these
uncertainties by querying LLM using [a]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:03:00Z"} natural
language representation of uncertainties.

To bring ACH fully into the LLM era, we reframe the analyst's "evidence
matrix" as two probabilistic mappings powered by a large language model
(LLM). Instead of manually ticking "+" or "--" in a grid, we (1) use the
LLM to identify latent "early‐sign" factors, (2) map how each factor
drives each hypothesis, (3) retrieve real news manually or via RAG, (4)
estimate whether each article actually mentions each factor, and (5)
aggregate all signals via a Naïve-Bayes--style score.
Concretely:[]{.paragraph-deletion author="Alexander Apartsin"
date="2025-08-27T11:02:00Z"}

1.  **Identify latent evidence signs**

Prompt the LLM (e.g.[,]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:03:00Z"} GPT-4) to list the top *K* "early signs"
that an analyst would watch to forecast the target relation. Example
prompt:

> [*"You are an expert geopolitical analyst assisting with strategic
> forecasting.*]{.insertion author="Ben Remez"
> date="2025-08-25T20:57:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T20:57:00Z"}
>
> [*Given a set of possible future outcomes: '\<list of outcomes\>',
> your task is to generate a set of early warning indicators that would
> help identify which outcome is most likely, with at least one
> indicator specifically designed to assess each outcome independently
> involving country: '\<C1\>' and country: '\<C2\>'.*]{.insertion
> author="Ben Remez" date="2025-08-25T20:57:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T20:57:00Z"}
>
> [*Instructions:*]{.insertion author="Ben Remez"
> date="2025-08-25T20:57:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T20:57:00Z"}
>
> [*For each outcome, generate at least one unique, realistic, and
> observable indicator.*]{.insertion author="Ben Remez"
> date="2025-08-25T20:57:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T20:57:00Z"}
>
> [*Each indicator should be designed to detect early signs or signals
> that support the specific outcome it is associated with.*]{.insertion
> author="Ben Remez" date="2025-08-25T20:57:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T20:57:00Z"}
>
> [*Do not repeat indicators across outcomes.*]{.insertion
> author="Ben Remez" date="2025-08-25T20:57:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T20:57:00Z"}
>
> [*Be*]{.insertion author="Ben Remez" date="2025-08-25T20:57:00Z"}
> [*concise*]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:03:00Z"} [*but informative.*]{.insertion
> author="Ben Remez" date="2025-08-25T20:57:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T20:57:00Z"}

2.  [*Format your output as a list of indicators with each one tied to
    the outcome it assesses."*]{.insertion author="Ben Remez"
    date="2025-08-25T20:57:00Z"}[*"You are an intelligence analyst with
    expertise in international affairs. Your task is to predict a future
    event involving country: Australia and country: China. You need to
    select your prediction from the following list of possible
    outcomes:\'\<list of outcomes\>\'. You monitor news articles for
    early signs and indicators of social or political developments that
    might lead to some future outcomes and decrease the likelihood of
    others. Identify the top 5 significant indicators that you would pay
    attention to for making a prediction*."]{.deletion
    author="Ben Remez"
    date="2025-08-25T20:57:00Z"}[]{.paragraph-deletion
    author="Ben Remez" date="2025-08-25T20:57:00Z"}

3.  **Score each factor's impact on each hypothesis**

For each (factor f, hypothesis h) pair, ask the LLM:

> "Given that \<factor f\> occurs, how likely ('highly likely' /
> 'likely' / 'possible' / 'unlikely' / 'highly unlikely') is
> \<hypothesis h\>? Briefly justify"

The five qualitative results are then mapped into numeric probabilities:

> \"highly likely\" → 0.9, \"likely\" → 0.66, \"possible\" → 0.33,
> \"highly unlikely\" → 0.1, \"unlikely\" → 0.0.
>
> Example Prompt:
>
> *"You are an intelligence analyst specializing in international
> affairs. Your task is to predict a future event involving* [*country:
> Australia and*]{.deletion author="Alexander Apartsin"
> date="2025-08-27T11:03:00Z"} [*the country: Australia, and
> the*]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:03:00Z"} *country: China. There has been a recent
> development described as \'\<factor f\>\' that might indicate the
> future event described by \'\<hypothesis h\>\'. Assess the certainty
> of the future event based on the [observed current development
> above]{.deletion author="Alexander Apartsin"
> date="2025-08-27T11:03:00Z"}[current development observed so
> far]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:03:00Z"}. Use the following qualitative
> descriptions of the certainty: \'highly likely\', \'likely\',
> \'possible\', \'unlikely\', or \'highly unlikely\'. Provide a brief
> justification for your assessment."*

4.  **Retrieve and rank relevant news (manually or using RAG)**

In this [part, we have two cases: Manual retrieval of articles based on
DocIds found in the relation_query.csv file and is related to the
relevant query in the test subset file, or RAG based]{.deletion
author="Alexander Apartsin" date="2025-08-27T11:03:00Z"}[section, we
present two cases: manual retrieval of articles based on DocIds found in
the relation_query.csv file, which is related to the relevant query in
the test subset file, and RAG-based]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:03:00Z"} retrieval using
"Weaviate" as a news-article corpus.

5.  **Estimate factor presence in each article**

For each retrieved article []{.deletion author="Alexander Apartsin"
date="2025-08-27T11:25:00Z"}[']{.deletion author="Alexander Apartsin"
date="2025-08-27T11:20:00Z"}[*a'*]{.deletion author="Alexander Apartsin"
date="2025-08-27T11:03:00Z"} []{.insertion author="Alexander Apartsin"
date="2025-08-27T11:03:00Z"}and each factor '*f'*, prompt the LLM:

> "The article says \<article abstract\>. How certain ('highly
> likely'/\...) is it that \<factor f\> is actually present?"
>
> As in step 2, we map the qualitative results into numeric
> probabilities in the same way.

Example Prompt:

> *"You are an open-source news analyst with expertise in international
> affairs. Your task is to predict a future event involving* [*country:
> Australia and*]{.deletion author="Alexander Apartsin"
> date="2025-08-27T11:20:00Z"} [*the country: Australia, and
> the*]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:20:00Z"} *country: China[.]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:04:00Z"} You are
> examining a news item with the contents being: \<article abstract\>.
> Assess the certainty of the development described as \<factor f\> [is
> happening based on the facts reported at]{.deletion
> author="Alexander Apartsin" date="2025-08-27T11:04:00Z"}[happening
> based on the facts reported in]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:04:00Z"} the news item. Use the following
> qualitative descriptions of the certainty: \'highly likely\',
> \'likely\', \'possible\', \'unlikely\', or \'highly unlikely\'.
> Provide a brief justification for your assessment.*

6.  **Aggregate into hypothesis scores**

In the final step, we compute a score for each hypothesis by aggregating
evidence from all retrieved articles. This is done by first forming two
matrices:

-   The analysis matrix $A \in R^{nxm}$, where each entry $A_{ij}$
    represents the estimated probability that [the]{.insertion
    author="Alexander Apartsin" date="2025-08-27T11:20:00Z"} article $i$
    contains indicator $j$, denoted as
    $\Pr{\left( f_{i}\  \right|\ a_{i})}$.

-   The influence matrix $I\  \in \ R^{mxh}$, where each entry $I_{jh}$
    represents the probability that [the]{.insertion
    author="Alexander Apartsin" date="2025-08-27T11:20:00Z"} hypothesis
    $h$ is true given indicator $j$, denoted as $\Pr(h\ |\ f_{j})$.

> These matrices are multiplied to obtain a prediction
> matrix[.]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:20:00Z"} $P \in R^{nxh}$:
>
> $$P = A \cdot I$$
>
> Each entry $P_{ih}$ reflects the contribution of [the]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:20:00Z"} article $i$
> to hypothesis $h$, aggregating over all factors.
>
> To compute the final score of each hypothesis, we take the mean across
> all articles:
>
> $$Score(h) = \ \frac{1}{n}\sum_{i = 1}^{n}P_{ih} = \ \frac{1}{n}\sum_{i = 1}^{n}{\sum_{j = 1}^{m}{A_{ij} \cdot I_{jh}}}$$
>
> This results in a single score per hypothesis, reflecting how well the
> collected news evidence supports each potential outcome.

This matrix-based aggregation approach allows us to scale probabilistic
ACH-style reasoning across large document sets. It retains the
interpretability of manual ACH while enabling automation,
reproducibility, and quantitative evaluation of complex geopolitical
forecasts.

The diagram below illustrates the flow of forecasting for each query:

![](media/image2.png){width="6.268055555555556in"
height="0.6541666666666667in"}

***Figure 2:** The forecasting flow per query*

# [Deep Analysis]{.insertion author="Ben Remez" date="2025-08-25T11:27:00Z"}

## [Motivation]{.insertion author="Ben Remez" date="2025-08-25T11:28:00Z"}

> [During the experiments, we observed a recurring issue where the model
> struggled to distinguish between "Material" and "Verbal"
> categ]{.insertion author="Ben Remez"
> date="2025-08-25T11:28:00Z"}[ories when evaluating the context of the
> retrieved articles. For example, articles discussing diplomatic
> negotiations]{.insertion author="Ben Remez"
> date="2025-08-25T11:29:00Z"} [(verbal)]{.insertion author="Ben Remez"
> date="2025-08-25T11:30:00Z"} []{.insertion author="Ben Remez"
> date="2025-08-25T11:29:00Z"}[were sometimes classified as material
> events, and vice versa. This ambiguity reduced the reliability
> of]{.insertion author="Ben Remez" date="2025-08-25T11:30:00Z"}
> [the]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:05:00Z"} [pipeline's forec]{.insertion
> author="Ben Remez" date="2025-08-25T11:30:00Z"}[asts, especially when
> articles contained mixed signals or lacked explicit descriptions of
> event types.]{.insertion author="Ben Remez"
> date="2025-08-25T11:31:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T11:31:00Z"}
>
> [To address this limitation, we introduced a Deep Analysis step as an
> additional]{.insertion author="Ben Remez" date="2025-08-25T11:31:00Z"}
> [second-stage reasoning layer with the pipeline.]{.insertion
> author="Ben Remez" date="2025-08-25T11:32:00Z"} [The prompt of the
> Deep Analysis step can be found under Appendix A.]{.insertion
> author="Ben Remez" date="2025-08-26T00:28:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T11:32:00Z"}

## [Approach]{.insertion author="Ben Remez" date="2025-08-25T11:32:00Z"}

> [In the Deep Analysis step, the LLM receives the full
> context]{.insertion author="Ben Remez" date="2025-08-25T11:32:00Z"}
> [of each retrieved article]{.insertion author="Ben Remez"
> date="2025-08-25T11:33:00Z"}[[[]{.insertion author="Ben Remez"
> date="2025-08-25T11:34:00Z"}]{.insertion author="Ben Remez"
> date="2025-08-25T11:37:00Z"}]{.insertion author="Ben Remez"
> date="2025-08-25T11:38:00Z"}[, along with the names of the countries
> involved in the query, a base description of cooperation or conflict,
> depending on the top-ranked result, and base descriptions of verbal
> and material elements]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:33:00Z"}[.]{.insertion author="Ben Remez"
> date="2025-08-25T11:38:00Z"} [The LLM is tasked with]{.insertion
> author="Ben Remez" date="2025-08-25T11:39:00Z"} [classifying each
> article for one of three categories: "Verbal", "Material"]{.insertion
> author="Ben Remez" date="2025-08-25T11:40:00Z"}[,]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:20:00Z"} [or
> "Uncertain" in cases where it fails to assess]{.insertion
> author="Ben Remez" date="2025-08-25T11:40:00Z"} [if the article is
> defined closer to "Verbal" or "Material". The LLM is also required to
> specify the certainty level for]{.insertion author="Ben Remez"
> date="2025-08-25T11:41:00Z"} ["Verbal" or "Material" with certainty
> levels of either "High" or "Low"]{.insertion author="Ben Remez"
> date="2025-08-25T11:42:00Z"} [and]{.insertion author="Ben Remez"
> date="2025-08-25T11:47:00Z"} [a short explanation for the LLM's
> choice.]{.insertion author="Ben Remez"
> date="2025-08-25T11:48:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T11:42:00Z"}
>
> [After receiving the results from the LLM,]{.insertion
> author="Ben Remez" date="2025-08-25T11:42:00Z"} [we score each
> category, "Verbal" or "Material", by]{.insertion author="Ben Remez"
> date="2025-08-25T11:43:00Z"} [adding a score of 1 for each category
> chosen with a certainty level of "High" and a score of 0.5 for each
> category]{.insertion author="Ben Remez" date="2025-08-25T11:44:00Z"}
> [selected]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:05:00Z"} [with a certainty level of
> "Low"]{.insertion author="Ben Remez" date="2025-08-25T11:44:00Z"}[. We
> ignore articles categorized as "Uncertain". After scoring each
> category, the category with the highest score is chosen as the final
> decision. In]{.insertion author="Ben Remez"
> date="2025-08-25T11:45:00Z"} []{.insertion author="Ben Remez"
> date="2025-08-25T11:46:00Z"}[cases where scores are equal, we
> retain]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:05:00Z"} [the original category that was initially
> ranked as the top decision in the original pipeline.]{.insertion
> author="Ben Remez" date="2025-08-25T11:46:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T11:58:00Z"}
>
> [For example, the scores of the categories, computed according to the
> LLM's response as shown in]{.insertion author="Ben Remez"
> date="2025-08-25T11:58:00Z"} [Figure]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:05:00Z"} [3, would be
> calculated as follows:]{.insertion author="Ben Remez"
> date="2025-08-25T11:58:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T11:58:00Z"}
>
> ["]{.insertion author="Ben Remez"
> date="2025-08-25T11:59:00Z"}[Material]{.insertion author="Ben Remez"
> date="2025-08-25T11:58:00Z"}["]{.insertion author="Ben Remez"
> date="2025-08-25T11:59:00Z"} []{.insertion author="Ben Remez"
> date="2025-08-25T11:58:00Z"}[Score = 1 + 1 = 2]{.insertion
> author="Ben Remez" date="2025-08-25T11:59:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T11:59:00Z"}
>
> ["Verbal" Score = 1 + 1 + 1 + 1 = 4]{.insertion author="Ben Remez"
> date="2025-08-25T11:59:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T11:59:00Z"}
>
> [For this case, the chosen category shall be "Verbal]{.insertion
> author="Ben Remez" date="2025-08-25T11:59:00Z"}[]{.insertion
> author="Ben Remez" date="2025-08-25T12:00:00Z"}[", and the final
> result will be Verbal Conflict/Cooperation (based on the
> top-ranked]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:05:00Z"} [decision from the original
> pipeline).]{.insertion author="Ben Remez"
> date="2025-08-25T12:00:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T11:46:00Z"}
>
> [![](media/image3.png){width="2.968695319335083in"
> height="1.3952865266841645in"}]{.insertion author="Ben Remez"
> date="2025-08-25T11:55:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T11:55:00Z"}
>
> [***Figure 3:***]{.insertion author="Ben Remez"
> date="2025-08-25T11:56:00Z"} [*An example of a saved response file
> from*]{.insertion author="Ben Remez" date="2025-08-25T11:57:00Z"}
> [*the*]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:20:00Z"} [*Deep Analysis step*]{.insertion
> author="Ben Remez" date="2025-08-25T11:57:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T12:00:00Z"}

## [Impact on the Pipeline]{.insertion author="Ben Remez" date="2025-08-25T12:01:00Z"}

> [Introducing]{.insertion author="Ben Remez"
> date="2025-08-25T12:01:00Z"} [the]{.insertion author="Ben Remez"
> date="2025-08-25T12:09:00Z"} [Deep Analysis]{.insertion
> author="Ben Remez" date="2025-08-25T12:01:00Z"} [step]{.insertion
> author="Ben Remez" date="2025-08-25T12:09:00Z"} [significantly
> improved the]{.insertion author="Ben Remez"
> date="2025-08-25T12:01:00Z"} [reliability]{.insertion
> author="Ben Remez" date="2025-08-25T12:12:00Z"} [of the pipeline's
> predictions by leveraging the full context of each retrieved article.
> Instead of solely relying on top-ranked]{.insertion author="Ben Remez"
> date="2025-08-25T12:09:00Z"} [decisions from the initial
> classification, the LLM now evaluates each article individually,
> assigns a certainty score for "Verbal" or]{.insertion
> author="Ben Remez" date="2025-08-25T12:10:00Z"} ["Material", and
> aggregates these scores to determine the]{.insertion
> author="Ben Remez" date="2025-08-25T12:11:00Z"} [outcome]{.insertion
> author="Ben Remez" date="2025-08-25T12:12:00Z"}[. Articles labelled
> "Uncertain" are ignored, reducing noise from ambiguous data. This
> refinement resulted in more consistent and accurate]{.insertion
> author="Ben Remez" date="2025-08-25T12:11:00Z"} [categorizations,
> particularly in scenarios where the original pipeline struggled to
> distinguish between "Verbal" and "Material" categories.]{.insertion
> author="Ben Remez" date="2025-08-25T12:12:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T11:56:00Z"}

# Initial Experimentation

**[Forecasting Query]{.underline}: "2023-11-04, Israel, Accuse,
Palestine"**

**[The news articles]{.underline}** (Only titles are shown)

![](media/image4.png){width="6.268055555555556in"
height="5.164583333333334in"}

**[Underlying latent evidence space (only 10 are shown)]{.underline}**

-   Statements from Israeli officials regarding peace negotiations with
    Palestine

-   Increased military presence or operations in Palestinian territories

-   Reports of civilian casualties or mass killings in conflict zones

-   International diplomatic efforts or visits aimed at mediating the
    Israel-Palestine conflict

-   Public demonstrations or rallies in support of Palestinian rights

-   Changes in U.S. or EU foreign policy towards Israel and Palestine

-   Incidents of violence or terrorism attributed to militant groups in
    Palestine

-   Economic aid packages or sanctions proposed by international bodies

-   Official recognition or support for Palestinian statehood by other
    nations

-   Joint initiatives or agreements between Israel and Palestine on
    humanitarian issues

**[Influence of the evidence on the outcomes (a sample)]{.underline}**

![A white and black document with black text AI-generated content may be
incorrect.](media/image5.png){width="6.268055555555556in"
height="3.527083333333333in"}

**[News Analysis (a sample)]{.underline}**

+------------+---------+--------+-----------------------+------------+
| **         | **Evi   | *      | **Justification**     | **Prob**   |
| Abstract** | dence** | *Certa |                       |            |
|            |         | inty** |                       |            |
+============+=========+========+=======================+============+
| Indo-US    | Sta     | Po     | The news item         | 0.33       |
| 2+2 to     | tements | ssible | indicates that the    |            |
| take up    | from    |        | Israel-Hamas conflict |            |
| Israel,    | Israeli |        | and human rights      |            |
| rights     | of      |        | issues will be        |            |
| issue The  | ficials |        | discussed at the      |            |
| Is         | re      |        | upcoming Indo-US 2+2  |            |
| rael-Hamas | garding |        | meeting. However, it  |            |
| conflict,  | peace   |        | does not provide      |            |
| besides    | negot   |        | direct evidence of    |            |
| human      | iations |        | ongoing peace         |            |
| rights and | with    |        | negotiations between  |            |
| democracy, | Pa      |        | Israel and Palestine. |            |
| will be on | lestine |        | While discussions at  |            |
| the table  |         |        | such high-level       |            |
| at the     |         |        | meetings could        |            |
| f          |         |        | influence future      |            |
| orthcoming |         |        | negotiations, the     |            |
| 2+2        |         |        | actual statements     |            |
| meeting of |         |        | from Israeli          |            |
| the        |         |        | officials regarding   |            |
| Foreign    |         |        | peace negotiations    |            |
| and        |         |        | are not confirmed in  |            |
| Defence    |         |        | this report, making   |            |
| Ministers  |         |        | the likelihood of     |            |
| of the US  |         |        | such statements       |            |
| and India  |         |        | occurring             |            |
| on         |         |        | \'possible\' but not  |            |
| November   |         |        | certain.              |            |
| 10.        |         |        |                       |            |
+------------+---------+--------+-----------------------+------------+
| Indo-US    | In      | po     | The news item         | 0.33       |
| 2+2 to     | creased | ssible | indicates that the    |            |
| take up    | m       |        | Israel-Hamas conflict |            |
| Israel,    | ilitary |        | and human rights      |            |
| rights     | p       |        | issues will be        |            |
| issue The  | resence |        | discussed at the      |            |
| Is         | or      |        | upcoming Indo-US 2+2  |            |
| rael-Hamas | ope     |        | meeting. While this   |            |
| conflict,  | rations |        | suggests a focus on   |            |
| besides    | in      |        | the situation in      |            |
| human      | Pale    |        | Israel and Palestine, |            |
| rights and | stinian |        | it does not provide   |            |
| democracy, | terr    |        | direct evidence of an |            |
| will be on | itories |        | imminent increase in  |            |
| the table  |         |        | military presence or  |            |
| at the     |         |        | operations in         |            |
| f          |         |        | Palestinian           |            |
| orthcoming |         |        | territories. The      |            |
| 2+2        |         |        | discussions may lead  |            |
| meeting of |         |        | to diplomatic actions |            |
| the        |         |        | or statements rather  |            |
| Foreign    |         |        | than immediate        |            |
| and        |         |        | military changes,     |            |
| Defence    |         |        | making the likelihood |            |
| Ministers  |         |        | of increased military |            |
| of the US  |         |        | operations possible   |            |
|            |         |        | but not certain.      |            |
| and India  |         |        |                       |            |
| on         |         |        |                       |            |
| November   |         |        |                       |            |
| 10.        |         |        |                       |            |
+------------+---------+--------+-----------------------+------------+
| Indo-US    | Reports | likely | The news item         | 0.66       |
| 2+2 to     | of      |        | indicates that the    |            |
| take up    | c       |        | Israel-Hamas          |            |
| Israel,    | ivilian |        | conflict, along       |            |
| rights     | cas     |        |                       |            |
| issue[.]{  | ualties |        | with human rights     |            |
| .insertion | or mass |        | issues, will be       |            |
| author=    | k       |        | discussed at an       |            |
| "Alexander | illings |        | upcoming              |            |
|  Apartsin" | in      |        |                       |            |
| date="20   | c       |        | international         |            |
| 25-08-27T1 | onflict |        | meeting. Given the    |            |
| 1:23:00Z"} | zones   |        | ongoing nature of the |            |
| The        |         |        | conflict              |            |
| Is         |         |        |                       |            |
| rael-Hamas |         |        | and previous reports  |            |
| conflict,  |         |        | of civilian           |            |
| besides    |         |        | casualties, it is     |            |
| human      |         |        | reasonable to         |            |
| rights and |         |        |                       |            |
| democracy, |         |        | assess that there are |            |
| will be on |         |        | likely ongoing        |            |
| the table  |         |        | reports of civilian   |            |
| at the     |         |        | casualties            |            |
| f          |         |        |                       |            |
| orthcoming |         |        | or mass killings in   |            |
| 2+2        |         |        | the conflict zones.   |            |
| meeting of |         |        | However, without      |            |
| the        |         |        | specific              |            |
| Foreign    |         |        |                       |            |
| and        |         |        | recent data or        |            |
| Defence    |         |        | reports confirming    |            |
| Ministers  |         |        | these incidents, the  |            |
| of the US  |         |        | certainty             |            |
| and India  |         |        |                       |            |
| on         |         |        | is not at the highest |            |
| November   |         |        | level.                |            |
| 10.        |         |        |                       |            |
+------------+---------+--------+-----------------------+------------+

**[Resulting Scores (correct relation is in red and has rank 4 out of
19)]{.underline}**

![A table of military personnel AI-generated content may be
incorrect.](media/image6.png){width="6.268055555555556in"
height="5.550694444444445in"}

# [Test Experiment Design and Execution]{.deletion author="Ben Remez" date="2025-08-25T14:39:00Z"}[Experiments]{.insertion author="Ben Remez" date="2025-08-25T14:39:00Z"}

To evaluate the performance of our proposed ACH-based inference
framework, we conducted experiments using the MIRAI benchmark,
introduced by Ye et al. \[3[7]{.insertion author="Ben Remez"
date="2025-08-26T11:39:00Z"}[1]{.deletion author="Ben Remez"
date="2025-08-26T09:57:00Z"}\], which is designed to assess LLM agents
in the context of geopolitical event forecasting. Each MIRAI test query
defines a structured prediction task, including a source and target
country, a forecast date, and a ground truth event label encoded using
the CAMEO schema \[6[5]{.deletion author="Ben Remez"
date="2025-08-26T09:53:00Z"}\].

Our experimental pipeline mirrors the high-level structure of the MIRAI
evaluation framework. For each forecasting query, the system generates
early indicators (factors) using an LLM, infers their influence on a
predefined hypothesis space (CAMEO relations), retrieves relevant
articles from a local news corpus using a retrieval-augmented generation
(RAG) approach or manually using the document Ids (given in the test
subset files), and computes final hypothesis scores using a matrix-based
probabilistic aggregation method.

In our evaluation, we focused specifically on the first-level relation
classification, which involves mapping the predicted and ground truth
CAMEO codes to their corresponding two-digit top-level relation
categories. This [allows us to measure model performance at a coarser
but still meaningful semantic resolution, in line]{.deletion
author="Alexander Apartsin" date="2025-08-27T11:22:00Z"}[enables us to
evaluate model performance at a coarser yet still meaningful semantic
resolution, aligning]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:22:00Z"} with prior work.

To further align with the MIRAI framework, we replicated the benchmark's
"quad" and "binary" relation grouping schemes. However, at this stage,
we have only conducted experiments on the quad classification setting,
which reduces the 38 original CAMEO classes into four high-level
categories: Verbal Cooperation, Material Cooperation, Verbal Conflict,
and Material Conflict. This grouping enables us to evaluate the model's
capacity to capture the nature of inter-state interactions while
reducing sparsity and class imbalance.

Performance was assessed using standard evaluation metrics: precision,
recall, F1 score, and Kullback--Leibler (KL) divergence. These metrics
were computed for both our RAG and manual document retrieval pipelines.
The results are compared against the MIRAI benchmark's reported
performance, enabling a direct evaluation of the proposed method's
strengths and limitations.[]{.paragraph-insertion author="Ben Remez"
date="2025-08-25T14:37:00Z"}

[This section presents the experimental design and execution process
used to evaluate and refine our ACH-based]{.insertion author="Ben Remez"
date="2025-08-25T14:37:00Z"} [forecasting pipeline. The experiments were
designed iteratively, with each stage introducing targeted improvements
based on insights from previous results. The evaluation
compa]{.insertion author="Ben Remez" date="2025-08-25T14:38:00Z"}[red
two retrieval strategies]{.insertion author="Ben Remez"
date="2025-08-25T14:39:00Z"} [---]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:06:00Z"} [manual
extraction and RAG-based retrieval ---]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:22:00Z"} [across three
experiments.]{.insertion author="Ben Remez"
date="2025-08-25T14:39:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-25T23:24:00Z"}

[These experiments were conducted using the MIRAI benchmark
\[3]{.insertion author="Ben Remez"
date="2025-08-25T23:24:00Z"}[7]{.insertion author="Ben Remez"
date="2025-08-26T11:39:00Z"}[\],]{.insertion author="Ben Remez"
date="2025-08-25T23:24:00Z"} [which ensures a standardized evaluation
framework and enables]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:06:00Z"} [direct comparison with MIRAI's reported
results.]{.insertion author="Ben Remez"
date="2025-08-25T23:24:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-25T14:39:00Z"}

## [First Experiment - Baseline Setup]{.insertion author="Ben Remez" date="2025-08-25T14:40:00Z"}

> [The first experiment aimed to establish a baseline performance
> benchmark for the pipeline.]{.insertion author="Ben Remez"
> date="2025-08-25T14:40:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T14:40:00Z"}

### [Retrieval Methods]{.insertion author="Ben Remez" date="2025-08-25T14:41:00Z"}

-   [**Manual Extraction:** Used manually provided Document IDs (docids)
    for each query, ensuring a fixed set of documents]{.insertion
    author="Ben Remez" date="2025-08-25T14:41:00Z"} [per
    event.]{.insertion author="Ben Remez"
    date="2025-08-25T14:43:00Z"}[]{.paragraph-insertion
    author="Ben Remez" date="2025-08-25T14:43:00Z"}

-   [**RAG Retrieval (Basic Setup):** Im]{.insertion author="Ben Remez"
    date="2025-08-25T14:44:00Z"}[plemented directly via Weaviate
    interface without LangChain orchestration. Retrieval queries
    consisted only of the two country names (e.g., {C1} {C2}]{.insertion
    author="Ben Remez" date="2025-08-25T14:46:00Z"}[), and the system
    returned the top 10 most semantically similar documents based on
    vector similarity, without metadata]{.insertion author="Ben Remez"
    date="2025-08-25T14:47:00Z"} [filtering or advanced
    strategies.]{.insertion author="Ben Remez"
    date="2025-08-25T14:49:00Z"}[]{.paragraph-insertion
    author="Ben Remez" date="2025-08-25T14:51:00Z"}

### [Findings]{.insertion author="Ben Remez" date="2025-08-25T14:52:00Z"}

> [While manual extraction provided consistent input, the RAG setup
> suffered from limited coverage and contextual relevance. The retrieved
> documents were often]{.insertion author="Ben Remez"
> date="2025-08-25T14:52:00Z"} [irrelevant]{.insertion
> author="Ben Remez" date="2025-08-25T21:17:00Z"} [or]{.insertion
> author="Ben Remez" date="2025-08-25T14:52:00Z"} [loosely related to
> the forecasting period, leading to unstable rankings and lower overall
> accuracy.]{.insertion author="Ben Remez"
> date="2025-08-25T14:53:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T23:25:00Z"}
>
> [Compared to MIRAI's reported metrics \[3]{.insertion
> author="Ben Remez" date="2025-08-25T23:25:00Z"}[7]{.insertion
> author="Ben Remez" date="2025-08-26T11:39:00Z"}[\], our baseline
> results un]{.insertion author="Ben Remez"
> date="2025-08-25T23:25:00Z"}[derperformed, highlighting the need for
> refinements in indicator generation and retrieval
> infrastructure.]{.insertion author="Ben Remez"
> date="2025-08-25T23:26:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T14:53:00Z"}

## [Second Experiment -- Improved Indicators & Enhanced RAG]{.insertion author="Ben Remez" date="2025-08-25T14:54:00Z"}

> [Building on insights from the first experiment, the second experiment
> focused]{.insertion author="Ben Remez" date="2025-08-25T14:54:00Z"}
> [on improving both indicator generation and retrieval
> quality.]{.insertion author="Ben Remez"
> date="2025-08-25T14:55:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T14:55:00Z"}

### [Balanced Indicator Generation]{.insertion author="Ben Remez" date="2025-08-25T14:55:00Z"}

> [In the baseline experiment, the LLM-generated indicators showed a
> noticeable imbalance across outcome categories, with a bias
> toward]{.insertion author="Ben Remez" date="2025-08-25T15:00:00Z"}
> [different outcomes each time. This imbalance negatively impacted
> hypothesis scoring, as some categories were
> overrepresented]{.insertion author="Ben Remez"
> date="2025-08-25T15:01:00Z"} [while others lacked sufficient
> supporting evidence.]{.insertion author="Ben Remez"
> date="2025-08-25T15:02:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T15:02:00Z"}
>
> [To address this, we refined the prompt used for indicator generation.
> The updated prompt explicitly instructed the LLM to generate a
> balanced set of indicators distributed]{.insertion author="Ben Remez"
> date="2025-08-25T15:02:00Z"} [evenly across the four high-level CAMEO
> categories (Verbal Cooperation, Material Cooperation, Verbal Conflict,
> Material Conflict).]{.insertion author="Ben Remez"
> date="2025-08-25T15:03:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T15:03:00Z"}
>
> [Initial Prompt for Indicators Generation:]{.insertion
> author="Ben Remez" date="2025-08-25T15:04:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T15:04:00Z"}
>
> [*"You are an intelligence analyst with expertise in international
> affairs. Your task is to predict a future event involving
> country:*]{.insertion author="Ben Remez" date="2025-08-25T15:06:00Z"}
> [*\<C1\>*]{.insertion author="Ben Remez" date="2025-08-25T20:55:00Z"}
> [*and country:*]{.insertion author="Ben Remez"
> date="2025-08-25T15:06:00Z"} *[\<C2\>]{.insertion author="Ben Remez"
> date="2025-08-25T20:55:00Z"}[. You need to select your prediction from
> the following list of possible outcomes:]{.insertion
> author="Ben Remez" date="2025-08-25T15:06:00Z"}* []{.insertion
> author="Ben Remez" date="2025-08-25T20:56:00Z"}[*\'\<list of
> outcomes\>\'. You monitor news articles for early signs and indicators
> of social or political developments that might lead to some future
> outcomes and decrease the likelihood of others. Identify the
> top*]{.insertion author="Ben Remez" date="2025-08-25T15:06:00Z"}
> [*'\<k\>'*]{.insertion author="Ben Remez" date="2025-08-25T20:56:00Z"}
> [*significant indicators that you would pay attention to for making a
> prediction*."]{.insertion author="Ben Remez"
> date="2025-08-25T15:06:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T15:06:00Z"}
>
> [Fine-Tuned Prompt for Indicators Generation:]{.insertion
> author="Ben Remez" date="2025-08-25T15:06:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T15:06:00Z"}

[*"You are an expert geopolitical analyst assisting with strategic
forecasting.*]{.insertion author="Ben Remez"
date="2025-08-25T15:45:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-25T15:46:00Z"}

> [*Given a set of possible future outcomes:*]{.insertion
> author="Ben Remez" date="2025-08-25T15:45:00Z"} *['\<list of
> outcomes\>']{.insertion author="Ben Remez"
> date="2025-08-25T20:55:00Z"}[, your task is to generate a set of
> early]{.insertion author="Ben Remez" date="2025-08-25T15:45:00Z"}*
> []{.insertion author="Ben Remez" date="2025-08-25T20:55:00Z"}[*warning
> indicators that would help identify which outcome is most likely, with
> at*]{.insertion author="Ben Remez" date="2025-08-25T15:45:00Z"}
> []{.insertion author="Ben Remez"
> date="2025-08-25T15:46:00Z"}[*least*]{.insertion author="Ben Remez"
> date="2025-08-25T15:45:00Z"} []{.insertion author="Ben Remez"
> date="2025-08-25T15:47:00Z"}[*one indicator specifically designed
> to*]{.insertion author="Ben Remez" date="2025-08-25T15:45:00Z"}
> [*assess*]{.insertion author="Ben Remez" date="2025-08-25T15:49:00Z"}
> [*each outcome independently involving country:*]{.insertion
> author="Ben Remez" date="2025-08-25T15:45:00Z"}
> [*'\<C1\>'*]{.insertion author="Ben Remez"
> date="2025-08-25T20:56:00Z"} [*and country:*]{.insertion
> author="Ben Remez" date="2025-08-25T15:45:00Z"} *['\<C2\>']{.insertion
> author="Ben Remez" date="2025-08-25T20:56:00Z"}[.]{.insertion
> author="Ben Remez"
> date="2025-08-25T15:45:00Z"}*[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T15:45:00Z"}

[*Instructions:*]{.insertion author="Ben Remez"
date="2025-08-25T15:45:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-25T15:45:00Z"}

> [*For each outcome, generate at least one unique, realistic, and
> observable indicator.*]{.insertion author="Ben Remez"
> date="2025-08-25T15:45:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T15:45:00Z"}
>
> [*Each indicator should be designed to detect early signs or signals
> that support the specific outcome it is associated with.*]{.insertion
> author="Ben Remez" date="2025-08-25T15:45:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T15:45:00Z"}

[*Do not repeat indicators across outcomes.*]{.insertion
author="Ben Remez" date="2025-08-25T15:45:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-25T15:45:00Z"}

[*Be*]{.insertion author="Ben Remez" date="2025-08-25T15:45:00Z"}
[*concise*]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:06:00Z"} [*but informative.*]{.insertion
author="Ben Remez" date="2025-08-25T15:45:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-25T15:45:00Z"}

> *[Format your output as a list of indicators with each one tied to the
> outcome it assesses.]{.insertion author="Ben Remez"
> date="2025-08-25T15:45:00Z"}["]{.insertion author="Ben Remez"
> date="2025-08-25T15:47:00Z"}*[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T15:49:00Z"}
>
> [By enforcing balance in the prompt]{.insertion author="Ben Remez"
> date="2025-08-25T15:49:00Z"}[, the generated indicators became more
> representative and unbiased, which significantly improved
> classification accuracy in both manual and RAG-based pipelines during
> the second experiment.]{.insertion author="Ben Remez"
> date="2025-08-25T15:50:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T16:26:00Z"}

### [Enhanced RAG Retrieval Infrastructure]{.insertion author="Ben Remez" date="2025-08-25T16:27:00Z"}

> [In this experiment, LangChain was introduced to orchestrate the
> retrieval pipeline, making it modular and more]{.insertion
> author="Ben Remez" date="2025-08-25T16:27:00Z"} [adaptable. Several
> Key enhancements were applied:]{.insertion author="Ben Remez"
> date="2025-08-25T16:28:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T16:28:00Z"}

#### [Metadata-Aware]{.insertion author="Ben Remez" date="2025-08-25T16:28:00Z"} [Retrieval]{.insertion author="Ben Remez" date="2025-08-25T16:29:00Z"}

> [Articles stored in Weaviate were enriched with a list of
> countries]{.insertion author="Ben Remez" date="2025-08-25T16:29:00Z"}
> [supported by the current]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:06:00Z"} [article\'s metadata, including the
> date]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:22:00Z"}[. This allowed queries to
> preci]{.insertion author="Ben Remez" date="2025-08-25T16:29:00Z"}[sely
> filter documents relevant to specific countries involved and limit
> retrieval to articles around the forecasting period.]{.insertion
> author="Ben Remez" date="2025-08-25T16:30:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T16:31:00Z"}

#### [Recent vs. Old Documents]{.insertion author="Ben Remez" date="2025-08-25T16:31:00Z"}

> [We divided retrieval into two distinct buckets:]{.insertion
> author="Ben Remez" date="2025-08-25T16:31:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T16:31:00Z"}

-   [Recent Documents:]{.insertion author="Ben Remez"
    date="2025-08-25T16:31:00Z"} [Articles published on the target date
    (D).]{.insertion author="Ben Remez"
    date="2025-08-25T16:32:00Z"}[]{.paragraph-insertion
    author="Ben Remez" date="2025-08-25T16:32:00Z"}

-   [Old Documents: Articles published up to 30 days before the target
    date.]{.insertion author="Ben Remez"
    date="2025-08-25T16:32:00Z"}[]{.paragraph-insertion
    author="Ben Remez" date="2025-08-25T16:32:00Z"}

> [This structure reflected the assumption that the most recent month of
> news is the most relevant]{.insertion author="Ben Remez"
> date="2025-08-25T16:33:00Z"} [for]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:06:00Z"} [predicting
> short-term geopolitical outcomes.]{.insertion author="Ben Remez"
> date="2025-08-25T16:34:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T16:34:00Z"}

#### [Exponential Time Decay Weighting]{.insertion author="Ben Remez" date="2025-08-25T16:34:00Z"}

> [To prioritize newer documents without discarding]{.insertion
> author="Ben Remez" date="2025-08-25T16:34:00Z"} [potentially relevant
> historical context, we applied an exponential decay
> function:]{.insertion author="Ben Remez"
> date="2025-08-25T16:35:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T16:36:00Z"}
>
> $$^{}$$[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T16:36:00Z"}
>
> [where]{.insertion author="Ben Remez" date="2025-08-25T16:36:00Z"}
> $$[.]{.insertion author="Ben Remez"
> date="2025-08-25T16:37:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T16:37:00Z"}
>
> [This choice of]{.insertion author="Ben Remez"
> date="2025-08-25T16:37:00Z"} $$ []{.insertion author="Ben Remez"
> date="2025-08-25T16:37:00Z"}[Ensured]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:07:00Z"} [that
> documents from]{.insertion author="Ben Remez"
> date="2025-08-25T16:37:00Z"} [3]{.insertion author="Ben Remez"
> date="2025-08-25T16:38:00Z"}[0 days]{.insertion author="Ben Remez"
> date="2025-08-25T16:37:00Z"} [before retained approximately
> 63%]{.insertion author="Ben Remez" date="2025-08-25T16:38:00Z"}
> []{.insertion author="Ben Remez" date="2025-08-25T16:42:00Z"}[of their
> weight]{.insertion author="Ben Remez" date="2025-08-25T16:39:00Z"}
> [(as shown in]{.insertion author="Ben Remez"
> date="2025-08-25T16:42:00Z"} [Figure]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:07:00Z"}
> [4)]{.insertion author="Ben Remez" date="2025-08-25T16:42:00Z"}[,
> while articles older than a month were naturally
> deprioritized.]{.insertion author="Ben Remez"
> date="2025-08-25T16:39:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T16:40:00Z"}
>
> [![A graph of different colored lines AI-generated content may be
> incorrect.](media/image7.png){width="6.261805555555555in"
> height="3.652083333333333in"}]{.insertion author="Ben Remez"
> date="2025-08-25T16:40:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T16:40:00Z"}
>
> [***Figure 4:** A graph representing*]{.insertion author="Ben Remez"
> date="2025-08-25T16:40:00Z"} *[differe]{.insertion author="Ben Remez"
> date="2025-08-25T16:48:00Z"}[n]{.insertion author="Ben Remez"
> date="2025-08-25T17:16:00Z"}[t]{.insertion author="Ben Remez"
> date="2025-08-25T16:48:00Z"}* []{.insertion author="Ben Remez"
> date="2025-08-25T16:40:00Z"}$$ []{.insertion author="Ben Remez"
> date="2025-08-25T16:41:00Z"}[*Values*]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:07:00Z"} [*to days
> from*]{.insertion author="Ben Remez" date="2025-08-25T16:41:00Z"}
> [*the*]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:07:00Z"} [*target date. The dashed line shows the
> value for*]{.insertion author="Ben Remez" date="2025-08-25T16:41:00Z"}
> $$[*.*]{.insertion author="Ben Remez"
> date="2025-08-25T16:41:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T16:41:00Z"}

#### [MMR (Maximal Marginal Relevance)]{.insertion author="Ben Remez" date="2025-08-25T16:49:00Z"}

> [MMR was applied to reduce redundancy by selecting documents that are
> both highly relevant to the query and diverse]{.insertion
> author="Ben Remez" date="2025-08-25T16:49:00Z"} [in content. This
> ensured the]{.insertion author="Ben Remez"
> date="2025-08-25T16:57:00Z"} [retrieved evidence set
> covered]{.insertion author="Ben Remez" date="2025-08-25T16:58:00Z"}
> [multiple perspectives rather than repeating similar
> information.]{.insertion author="Ben Remez"
> date="2025-08-25T17:09:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T17:09:00Z"}

#### [RRF (Reciprocal Rank Fusion)]{.insertion author="Ben Remez" date="2025-08-25T17:09:00Z"}

> [RRF was used to merge]{.insertion author="Ben Remez"
> date="2025-08-25T17:09:00Z"} [multiple ranked lists -- for example,
> combining results from recent and old document retrieval into a single
> ranking. By leveraging complementary rankings, RRF increased recall
> and improved evidence]{.insertion author="Ben Remez"
> date="2025-08-25T17:10:00Z"} [completeness]{.insertion
> author="Ben Remez" date="2025-08-25T20:54:00Z"}[.]{.insertion
> author="Ben Remez" date="2025-08-25T17:11:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T17:11:00Z"}

### [Findings]{.insertion author="Ben Remez" date="2025-08-25T17:11:00Z"}

> [The combination of balanced indicators, metadata-aware retrieval,
> advanced ranking strategies, and time-sensitive weighting]{.insertion
> author="Ben Remez" date="2025-08-25T17:11:00Z"} [significantly
> improved RAG retrieval quality. Articles were more contextually
> relevant, coverage was broader, and classification accuracy
> increased.]{.insertion author="Ben Remez" date="2025-08-25T17:12:00Z"}
> [In this experiment]{.insertion author="Ben Remez"
> date="2025-08-25T17:14:00Z"}[,]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:07:00Z"} [RAG slightly outperformed]{.insertion
> author="Ben Remez" date="2025-08-25T17:14:00Z"} [m]{.insertion
> author="Ben Remez" date="2025-08-25T17:15:00Z"}[anual]{.insertion
> author="Ben Remez" date="2025-08-25T17:14:00Z"} [extraction. This
> experiment]{.insertion author="Ben Remez" date="2025-08-25T17:15:00Z"}
> [significantly reduced the performance gap and laid the
> groundwork]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:07:00Z"} [for further improvement]{.insertion
> author="Ben Remez" date="2025-08-25T17:15:00Z"}[s.]{.insertion
> author="Ben Remez" date="2025-08-25T17:16:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T23:26:00Z"}
>
> [These improvements enabled the]{.insertion author="Ben Remez"
> date="2025-08-25T23:26:00Z"} [enhanced RAG pipeline to approach and,
> in several metrics, exceed MIRAI's baseline performance, showing
> significant gains in retrieval relevance and classification
> accuracy.]{.insertion author="Ben Remez"
> date="2025-08-25T23:27:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T16:39:00Z"}

## [Third Experiment -- Integrating Deep Analysis]{.insertion author="Ben Remez" date="2025-08-25T17:19:00Z"}

> [The third experiment aimed to evaluate the impact of incorporating a
> Deep Analysis step into the pipeline to improve classification
> consistency and prediction quality. In earlier experiments, we
> observed that the LLM frequently struggled to differentiate between
> Verbal and Material categories, especially when articles contained
> ambiguous signals or lacked explicit descriptions of event types. To
> address this issue, we introduced Deep Analysis as a second-stage
> reasoning layer on top of the original ranked results.]{.insertion
> author="Ben Remez" date="2025-08-25T17:34:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T17:19:00Z"}

### [Approach]{.insertion author="Ben Remez" date="2025-08-25T17:26:00Z"}

> [In the Deep Analysis step, we focused on the ranked results generated
> by the original pipeline. For each query, we took the top-ranked
> articles]{.insertion author="Ben Remez" date="2025-08-25T17:31:00Z"}[.
> We]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:08:00Z"} [provided the LLM with their full
> context]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:35:00Z"}[, including the names of the involved
> countries, base descriptions of cooperation or conflict, and relevant
> text excerpts from the retrieved documents. The LLM was instructed to
> classify each article into one of three categories: Verbal, Material,
> or Uncertain when the information was insufficient for a confident
> decision.]{.insertion author="Ben Remez"
> date="2025-08-25T17:31:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T17:31:00Z"}
>
> [For articles classified as \"Verbal\" or \"Material,\" the LLM also
> provided a certainty level, either High or Low, along with a short
> explanation justifying its choice. To aggregate these outputs, we
> assigned 1 point for each \"High\" classification and 0.5 points for
> each \"Low\" classification, while ignoring any articles marked as
> \"Uncertain.\" The cumulative scores for the Verbal and Material
> categories were then compared, and the category with the higher total
> score was selected as the final classification. In cases where the
> scores were tied, we preserved the original top-ranked result from the
> main pipeline.]{.insertion author="Ben Remez"
> date="2025-08-25T17:32:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T17:32:00Z"}
>
> [This additional reasoning layer addressed ambiguities in the initial
> predictions,]{.insertion author="Ben Remez"
> date="2025-08-25T17:32:00Z"} [particularly when retrieved articles
> contained mixed signals or lacked explicit descriptions, leading
> to]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:08:00Z"} [improved classification consistency and
> overall forecast accuracy.]{.insertion author="Ben Remez"
> date="2025-08-25T17:32:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T17:32:00Z"}

### [Findings]{.insertion author="Ben Remez" date="2025-08-25T17:37:00Z"}

> [The integration of the Deep Analysis step had a significant impact on
> improving classification consistency and overall predictive
> performance. By introducing a second reasoning layer, the
> pipeline]{.insertion author="Ben Remez" date="2025-08-25T17:38:00Z"}
> [improved its ability to resolve ambiguities between Verbal and
> Material event classifications, particularly]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:08:00Z"} [in cases
> where articles contained mixed or weak contextual signals.]{.insertion
> author="Ben Remez" date="2025-08-25T17:38:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T17:38:00Z"}
>
> [The improvement was most pronounced in the manual extraction
> pipeline, where access to highly relevant, curated documents allowed
> Deep Analysis to make more accurate and confident classifications. In
> this setting, we observed substantial gains in precision, recall, and
> F1 scores, alongside a notable reduction in KL divergence, indicating
> more stable and reliable forecasts.]{.insertion author="Ben Remez"
> date="2025-08-25T17:39:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T17:39:00Z"}
>
> [For the RAG-based pipeline, the gains were more modest but still
> measurable. Since the retrieved articles via RAG occasionally included
> noisy or less relevant content, the second-stage reasoning helped
> reduce misclassifications]{.insertion author="Ben Remez"
> date="2025-08-25T17:39:00Z"}[. Still, its]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:08:00Z"} [overall
> impact was limited compared to the manual extraction
> setup.]{.insertion author="Ben Remez"
> date="2025-08-25T17:39:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T17:39:00Z"}
>
> [However, while manual extraction currently offers slightly higher
> accuracy due to the quality of curated documents, it comes at the cost
> of being complex and time-consuming. In contrast, RAG]{.insertion
> author="Ben Remez" date="2025-08-25T17:42:00Z"} [offers a scalable
> solution by automating the retrieval process, enabling]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:35:00Z"} [rapid
> adaptation to new queries and datasets without requiring]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:08:00Z"} [manual
> intervention. This makes RAG more suitable for real-world forecasting
> scenarios where efficiency and coverage are critical.]{.insertion
> author="Ben Remez" date="2025-08-25T17:42:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T23:28:00Z"}
>
> [Deep Analysis further reduced misclassification errors, pushing both
> manual extraction and RAG pipelines beyond MIRAI's reported benchmarks
> on precision]{.insertion author="Ben Remez"
> date="2025-08-25T23:28:00Z"} [and F1 score, while maintaining
> comparability in KL divergence and accuracy.]{.insertion
> author="Ben Remez" date="2025-08-25T23:29:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T20:37:00Z"}

## [Experiments Summary]{.insertion author="Ben Remez" date="2025-08-25T20:37:00Z"}

> [Across the three experiments, we gradually refined the pipeline to
> improve both retrieval quality and classification accuracy. In the
> first experiment, we established a baseline using predefined document
> IDs and a basic RAG setup, but performance was limited due to
> imbalanced indicators and restricted retrieval relevance. The second
> experiment introduced a fine-tuned indicator generation prompt
> alongside a significantly enhanced RAG infrastructure powered by
> LangChain and Weaviate, enriched with metadata filtering, MMR, RRF,
> and exponential time decay weighting. These improvements substantially
> boosted RAG performance, enabling it]{.insertion author="Ben Remez"
> date="2025-08-25T20:40:00Z"} [to outperform manual extraction when
> using the refined indicators slightly]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:08:00Z"}[. In the
> third experiment, we integrated Deep Analysis, a second-stage
> reasoning layer designed to resolve ambiguities between Verbal and
> Material classifications. While manual extraction combined with Deep
> Analysis achieved the highest overall accuracy due to its reliance on
> highly relevant, curated documents, RAG with Deep Analysis also
> improved considerably and approached manual performance.
> Despite]{.insertion author="Ben Remez" date="2025-08-25T20:40:00Z"}
> [the slight accuracy advantage of manual extraction]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:08:00Z"}[, its
> complexity and time-consuming nature make it less practical for
> large-scale forecasting. In contrast, the RAG-based
> pipeline]{.insertion author="Ben Remez" date="2025-08-25T20:40:00Z"}
> [offers a scalable and automated solution that can adapt]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:09:00Z"} [efficiently
> to new queries and datasets, making it more suitable for real-world
> applications.]{.insertion author="Ben Remez"
> date="2025-08-25T20:40:00Z"}

# Results

To evaluate the effectiveness of our proposed pipeline for first-level
relation classification, we compared our system's performance with the
MIRAI benchmark. As outlined earlier, our experiment focused on the quad
setting of relation classification, grouping the original 38 CAMEO codes
into four coarse-grained classes: verbal cooperation, material
cooperation, verbal conflict, and material conflict.

We evaluated two variations of document retrieval for hypothesis
formation:

-   **Manual Document Extraction:** where relevant documents were
    manually selected based on known relevance.

-   **RAG-Based Retrieval:** using Retrieval-Augmented Generation [to
    automatically retrieve relevant documents for hypothesis
    scoring]{.deletion author="Alexander Apartsin"
    date="2025-08-27T11:09:00Z"}[to retrieve relevant documents for
    hypothesis scoring automatically]{.insertion
    author="Alexander Apartsin" date="2025-08-27T11:09:00Z"}.

## [We computed standard evaluation metrics, including]{.deletion author="Ben Remez" date="2025-08-25T21:07:00Z"}[System performance was measured]{.insertion author="Ben Remez" date="2025-08-25T21:07:00Z"} [precision]{.deletion author="Alexander Apartsin" date="2025-08-27T11:09:00Z"}[using precision]{.insertion author="Alexander Apartsin" date="2025-08-27T11:09:00Z"}, recall, F1 score[, accuracy]{.insertion author="Ben Remez" date="2025-08-25T20:49:00Z"}, and Kullback-Leibler (KL) divergence[,]{.deletion author="Alexander Apartsin" date="2025-08-27T11:09:00Z"} to assess the alignment of predicted distributions with ground truth. [These metrics enable a direct and meaningful comparison with MIRAI's results while ensuring alignment with the ground-truth distributions.]{.insertion author="Ben Remez" date="2025-08-25T21:08:00Z"}

## [Initial Results (Baseline Experiment)]{.insertion author="Ben Remez" date="2025-08-25T21:10:00Z"}

> [In the first experiment, we established a baseline by
> evaluating]{.insertion author="Ben Remez" date="2025-08-25T21:11:00Z"}
> [the pipeline using two retrieval strategies: manual document
> extraction and a basic RAG-based retrieval. For the manu]{.insertion
> author="Ben Remez" date="2025-08-25T21:14:00Z"}[al approach, we relied
> on document]{.insertion author="Ben Remez"
> date="2025-08-25T21:15:00Z"} [IDs]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:22:00Z"} [provided
> within the MIRAI dataset, ensuring that the selected articles were
> highly relevant to each forecasting query.]{.insertion
> author="Ben Remez" date="2025-08-25T21:15:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T21:15:00Z"}
>
> [In contrast, the RAG-based retrieval]{.insertion author="Ben Remez"
> date="2025-08-25T21:15:00Z"} []{.insertion author="Ben Remez"
> date="2025-08-25T21:16:00Z"}[employed a simple implementation through
> the Weaviate interface, without utilizing]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:09:00Z"} [LangChain
> orchestration or advanced ranking strategies. Queries consisted solely
> of the two country names involved, and no temporal constraints or
> contextual filters were applied.]{.insertion author="Ben Remez"
> date="2025-08-25T21:16:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T21:16:00Z"}
>
> [The results highlighted a]{.insertion author="Ben Remez"
> date="2025-08-25T21:16:00Z"} [slight]{.insertion author="Ben Remez"
> date="2025-08-25T21:17:00Z"} [gap between]{.insertion
> author="Ben Remez" date="2025-08-25T21:16:00Z"} [the two methods, with
> manual extraction being more consistent with the]{.insertion
> author="Ben Remez" date="2025-08-25T21:18:00Z"} [outcomes.]{.insertion
> author="Ben Remez" date="2025-08-25T21:19:00Z"} [As shown in the
> results]{.insertion author="Ben Remez" date="2025-08-25T21:21:00Z"}
> [in Table 1, m]{.insertion author="Ben Remez"
> date="2025-08-25T21:22:00Z"}[anual extraction outperformed the basic
> RAG across all metrics due to its inherently higher-quality and more
> relevant documents]{.insertion author="Ben Remez"
> date="2025-08-25T21:19:00Z"}[, with]{.insertion author="Ben Remez"
> date="2025-08-25T21:51:00Z"} [a]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:09:00Z"}
> [precision]{.insertion author="Ben Remez" date="2025-08-25T21:51:00Z"}
> [of 23.2%, recall of 38.8%,]{.insertion author="Ben Remez"
> date="2025-08-25T21:52:00Z"} [and an]{.insertion author="Ben Remez"
> date="2025-08-25T21:53:00Z"} [F1 score of 28.8%]{.insertion
> author="Ben Remez" date="2025-08-25T21:52:00Z"} []{.insertion
> author="Ben Remez" date="2025-08-25T21:51:00Z"}[for manual extraction
> vs precision]{.insertion author="Ben Remez"
> date="2025-08-25T21:53:00Z"} [of 22.1]{.insertion author="Ben Remez"
> date="2025-08-25T21:54:00Z"}[%, recall of 3]{.insertion
> author="Ben Remez" date="2025-08-25T21:53:00Z"}[7% and an F1 score of
> 27.5% for RAG.]{.insertion author="Ben Remez"
> date="2025-08-25T21:54:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T21:54:00Z"}
>
> [Meanwhile]{.insertion author="Ben Remez"
> date="2025-08-25T21:19:00Z"}[, the RAG pipeline underperformed because
> the simplistic querying method failed to capture sufficient context,
> resulting in less accurate retrievals and weaker forecasting
> performance.]{.insertion author="Ben Remez"
> date="2025-08-25T21:20:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T21:20:00Z"}
>
> [This baseline established a clear motivation for refining both the
> indi]{.insertion author="Ben Remez" date="2025-08-25T21:20:00Z"}[cator
> generation process and the retrieval infrastructure in subsequent
> experiments.]{.insertion author="Ben Remez"
> date="2025-08-25T21:21:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T23:29:00Z"}
>
> [When compared to M]{.insertion author="Ben Remez"
> date="2025-08-25T23:29:00Z"}[IRAI's results, both manual extraction
> and early RAG retrieval underperformed substantially, motivating
> further iterations to close the performance gap.]{.insertion
> author="Ben Remez" date="2025-08-25T23:30:00Z"}

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  [**Metric**]{.deletion author="Ben Remez"            [**Manual**]{.deletion author="Ben Remez"            [**RAG**]{.deletion author="Ben Remez"               [**MIRAI**]{.deletion author="Ben Remez"
  date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion   date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion   date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion   date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion
  author="Ben Remez" date="2025-08-25T21:22:00Z"}      author="Ben Remez" date="2025-08-25T21:22:00Z"}      author="Ben Remez" date="2025-08-25T21:22:00Z"}      author="Ben Remez" date="2025-08-25T21:22:00Z"}
  ---------------------------------------------------- ---------------------------------------------------- ---------------------------------------------------- ----------------------------------------------------
  [Precision (↑)]{.deletion author="Ben Remez"         [23.2 (%)]{.deletion author="Ben Remez"              [22.1 (%)]{.deletion author="Ben Remez"              [47.6 (%)]{.deletion author="Ben Remez"
  date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion   date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion   date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion   date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion
  author="Ben Remez" date="2025-08-25T21:22:00Z"}      author="Ben Remez" date="2025-08-25T21:22:00Z"}      author="Ben Remez" date="2025-08-25T21:22:00Z"}      author="Ben Remez" date="2025-08-25T21:22:00Z"}

  [Recall (↑)]{.deletion author="Ben Remez"            [38.8 (%)]{.deletion author="Ben Remez"              [37 (%)]{.deletion author="Ben Remez"                [58.3 (%)]{.deletion author="Ben Remez"
  date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion   date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion   date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion   date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion
  author="Ben Remez" date="2025-08-25T21:22:00Z"}      author="Ben Remez" date="2025-08-25T21:22:00Z"}      author="Ben Remez" date="2025-08-25T21:22:00Z"}      author="Ben Remez" date="2025-08-25T21:22:00Z"}

  [F1 Score (↑)]{.deletion author="Ben Remez"          [28.8 (%)]{.deletion author="Ben Remez"              [27.5 (%)]{.deletion author="Ben Remez"              [44.2 (%)]{.deletion author="Ben Remez"
  date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion   date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion   date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion   date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion
  author="Ben Remez" date="2025-08-25T21:22:00Z"}      author="Ben Remez" date="2025-08-25T21:22:00Z"}      author="Ben Remez" date="2025-08-25T21:22:00Z"}      author="Ben Remez" date="2025-08-25T21:22:00Z"}

  [KL Divergence (↓)]{.deletion author="Ben Remez"     [10.645]{.deletion author="Ben Remez"                [10.643]{.deletion author="Ben Remez"                [5.9]{.deletion author="Ben Remez"
  date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion   date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion   date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion   date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion
  author="Ben Remez" date="2025-08-25T21:22:00Z"}      author="Ben Remez" date="2025-08-25T21:22:00Z"}      author="Ben Remez" date="2025-08-25T21:22:00Z"}      author="Ben Remez" date="2025-08-25T21:22:00Z"}
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[***Table 1:** A table displaying the results of the experiment vs the
results from MIRAI \[31\]*]{.deletion author="Ben Remez"
date="2025-08-25T21:22:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-25T21:22:00Z"}

## []{.insertion author="Ben Remez" date="2025-08-25T21:23:00Z"}[Results]{.insertion author="Alexander Apartsin" date="2025-08-27T11:22:00Z"} [After Fine-Tuned Indicators]{.insertion author="Ben Remez" date="2025-08-25T21:23:00Z"}

> [The introduction of the fine-tuned indicator generation prompt marked
> a significant improvement over the initial experiment. In the baseline
> setup, both manual extraction and the early, basic RAG retrieval
> struggled with limited relevance and inconsistent coverage. By
> refining the indicator generation process, the system produced more
> balanced and contextually aligned signals, enabling more accurate
> classification of relationships between countries.]{.insertion
> author="Ben Remez" date="2025-08-25T21:37:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T21:37:00Z"}
>
> [Compared to the initial experiment, the RAG retrieval pipeline showed
> the]{.insertion author="Ben Remez" date="2025-08-25T21:38:00Z"} [most
> significant]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:22:00Z"} [gains. Early RAG, which relied solely on
> simple keyword-based country queries, delivered a precision of 22.1%
> and an F1 score of 27.5%. After introducing the improved
> infrastructure with metadata-driven retrieval, MMR re-ranking, RRF
> fusion, and time-decay weighting (α = 0.015 over a 30-day window), RAG
> Improved achieved a precision of 51.4% and an F1 score of 34.7%,
> nearly doubling performance in key metrics]{.insertion
> author="Ben Remez" date="2025-08-25T21:38:00Z"} [while also showing
> increased accuracy (45% vs. 53% after enhancements)]{.insertion
> author="Ben Remez" date="2025-08-25T21:39:00Z"}[.]{.insertion
> author="Ben Remez" date="2025-08-25T21:38:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T21:39:00Z"}
>
> [Similarly, manual extraction with fine-tuned indicators (Manual FT
> in]{.insertion author="Ben Remez" date="2025-08-25T21:40:00Z"}
> [Table]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:11:00Z"} [1) also benefited from the refined
> signals, improving to a precision of 50.4% and an F1 score of
> 33.3%.]{.insertion author="Ben Remez" date="2025-08-25T21:40:00Z"}
> [Overall]{.insertion author="Ben Remez" date="2025-08-25T21:42:00Z"}[,
> when comparing the two, 'RAG Improved' slightly outperformed 'Manual
> FT' across most metrics]{.insertion author="Ben Remez"
> date="2025-08-25T21:40:00Z"}[.]{.insertion author="Ben Remez"
> date="2025-08-25T21:41:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T22:08:00Z"}
>
> [To better illustrate these improvements, Figures 5 and 6 compare the
> correct answer rank distributions between the initial experiment and
> the fine-tuned setup]{.insertion author="Ben Remez"
> date="2025-08-25T22:08:00Z"}[.]{.insertion author="Ben Remez"
> date="2025-08-25T22:10:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T22:10:00Z"}
>
> [We can observe that the most significant change]{.insertion
> author="Ben Remez" date="2025-08-25T22:10:00Z"} []{.insertion
> author="Ben Remez" date="2025-08-25T22:11:00Z"}[occurred in queries
> where the correct answer was initially]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:11:00Z"} [ranked in
> 3rd place. We c]{.insertion author="Ben Remez"
> date="2025-08-25T22:11:00Z"}[an see that the rank 2 bar for RAG did
> not change noticeably, indicating that most new correctly ranked
> queries were]{.insertion author="Ben Remez"
> date="2025-08-25T22:12:00Z"} []{.insertion author="Ben Remez"
> date="2025-08-25T22:13:00Z"}[previously ranked in 3rd place and were
> moved]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:11:00Z"} [to rank 1, showcasing the value of the
> enhanced retrieval pipeline. Meanwhile]{.insertion author="Ben Remez"
> date="2025-08-25T22:13:00Z"}[,]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:11:00Z"} [we can observe that]{.insertion
> author="Ben Remez" date="2025-08-25T22:13:00Z"} [fine-tuning the
> indicators prompt also had a]{.insertion author="Ben Remez"
> date="2025-08-25T22:14:00Z"} [substantial]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:11:00Z"} [effect on
> the queries ranked in rank 3 for the manual extraction as well. In
> this case]{.insertion author="Ben Remez"
> date="2025-08-25T22:14:00Z"}[]{.insertion author="Ben Remez"
> date="2025-08-25T22:15:00Z"}[, there is a noticeable change in rank
> two compared to the initial experiment, which indicates that the
> fine-tuned prompt has significantly improved the
> distributions]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:11:00Z"}[]{.insertion author="Ben Remez"
> date="2025-08-25T22:15:00Z"}[. Yet, it]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:11:00Z"} [still didn't
> manage to reach higher accuracy levels]{.insertion author="Ben Remez"
> date="2025-08-25T22:15:00Z"} [than]{.insertion author="Ben Remez"
> date="2025-08-25T22:17:00Z"} [the]{.insertion author="Ben Remez"
> date="2025-08-25T22:15:00Z"} [']{.insertion author="Ben Remez"
> date="2025-08-25T22:16:00Z"}[RAG Imp]{.insertion author="Ben Remez"
> date="2025-08-25T22:15:00Z"}[roved' method.]{.insertion
> author="Ben Remez" date="2025-08-25T22:16:00Z"} []{.insertion
> author="Ben Remez" date="2025-08-25T22:12:00Z"}[These results
> demonstrate that the refined]{.insertion author="Ben Remez"
> date="2025-08-25T22:16:00Z"} [indicators generation prompt and
> the]{.insertion author="Ben Remez" date="2025-08-25T22:17:00Z"}
> [enhanced]{.insertion author="Ben Remez" date="2025-08-25T22:18:00Z"}
> [RAG setup improved retrieval precision and the overall quality of
> top-ranked evidence, leading to more reliable hypothesis scoring
> without compromising scalability.]{.insertion author="Ben Remez"
> date="2025-08-25T22:16:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T21:42:00Z"}
>
> [These results demonstrate a clear evolution: while manual
> extraction]{.insertion author="Ben Remez" date="2025-08-25T21:42:00Z"}
> [has historically provided more controlled, high-quality input, the
> upgraded RAG pipeline not only closes the gap but also
> surpasses]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:11:00Z"} [manual performance in several aspects.
> Moreover, RAG offers a scalable and automated alternative to manual
> selection, which is inherently time-consuming and resource-intensive,
> making it a more practical solution for large-scale forecasting
> tasks.]{.insertion author="Ben Remez"
> date="2025-08-25T21:42:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T23:30:00Z"}
>
> [Following these refinements,]{.insertion author="Ben Remez"
> date="2025-08-25T23:30:00Z"} [']{.insertion author="Ben Remez"
> date="2025-08-25T23:31:00Z"}[RAG improved]{.insertion
> author="Ben Remez" date="2025-08-25T23:30:00Z"}[']{.insertion
> author="Ben Remez" date="2025-08-25T23:31:00Z"} [not only closed the
> gap bu]{.insertion author="Ben Remez" date="2025-08-25T23:30:00Z"}[t
> exceeded MIRAI's reported precision and F1 score, while
> 'Manual]{.insertion author="Ben Remez" date="2025-08-25T23:31:00Z"}
> [FT' also achieved competitive performance.]{.insertion
> author="Ben Remez" date="2025-08-25T23:32:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-26T00:36:00Z"}
>
> [In addition to the primary evaluation metrics, we also examined the
> entropy of the output distributions to assess the]{.insertion
> author="Ben Remez" date="2025-08-26T00:36:00Z"} [model\'s uncertainty
> in]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:11:00Z"} [predictions. This analysis was
> performed]{.insertion author="Ben Remez" date="2025-08-26T00:36:00Z"}
> [on both the MIRAI benchmark results and]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:11:00Z"} [our pipeline
> after incorporating the fine-tuned indicators and enhanced RAG
> retrieval. A detailed explanation and numerical results]{.insertion
> author="Ben Remez" date="2025-08-26T00:36:00Z"} [are
> provided]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:11:00Z"} [in Appendix B.]{.insertion
> author="Ben Remez" date="2025-08-26T00:36:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T22:06:00Z"}

[![A graph with blue and orange bars AI-generated content may be
incorrect.](media/image8.png){width="6.268055555555556in"
height="3.127083333333333in"}]{.insertion author="Ben Remez"
date="2025-08-25T22:06:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-25T22:06:00Z"}

[***Figure 5:** The rank position of the correct label as predicted by
the system vs*]{.insertion author="Ben Remez"
date="2025-08-25T22:06:00Z"} [*the*]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:12:00Z"} [*number of
queries*]{.insertion author="Ben Remez" date="2025-08-25T22:06:00Z"}
*[for the initial (baseline) experiment]{.insertion author="Ben Remez"
date="2025-08-25T22:07:00Z"}[.]{.insertion author="Ben Remez"
date="2025-08-25T22:06:00Z"}*[]{.paragraph-insertion author="Ben Remez"
date="2025-08-25T22:07:00Z"}

[![A graph with blue and orange squares AI-generated content may be
incorrect.](media/image9.png){width="6.268055555555556in"
height="3.1145833333333335in"}]{.insertion author="Ben Remez"
date="2025-08-25T22:07:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-25T22:07:00Z"}

[***Figure 6:** The rank position of the correct label as predicted by
the system vs*]{.insertion author="Ben Remez"
date="2025-08-25T22:07:00Z"} [*the*]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:11:00Z"} *[number of
queries for the second experiment (fine-tuned indicators]{.insertion
author="Ben Remez" date="2025-08-25T22:07:00Z"}[, prompt,]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:12:00Z"}* [*and enhanced
RAG).*]{.insertion author="Ben Remez"
date="2025-08-25T22:07:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-25T21:43:00Z"}

## [Results After Introducing Deep Analysis]{.insertion author="Ben Remez" date="2025-08-25T21:44:00Z"}

> [The introduction of the Deep A]{.insertion author="Ben Remez"
> date="2025-08-25T21:44:00Z"}[nalysis step further refined the
> pipeline's performance by addressing a]{.insertion author="Ben Remez"
> date="2025-08-25T21:45:00Z"} [recurring]{.insertion author="Ben Remez"
> date="2025-08-25T21:46:00Z"} [limitation observed in previous
> experiments. Earlier, the LLM struggled to distinguish between
> "Verbal" and "Material" interaction when evaluating]{.insertion
> author="Ben Remez" date="2025-08-25T21:45:00Z"} [article context. This
> often led to misclassifications, particularly in ambiguous cases such
> as diplomatic negotiations being categorized as material
> events.]{.insertion author="Ben Remez"
> date="2025-08-25T21:46:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T21:47:00Z"}
>
> [By incorporating a second-stage reasoning layer, Deep Analysis
> leveraged the full context of each retrieved article along with
> certainty-based scoring. For each top-ranked result, the
> LLM]{.insertion author="Ben Remez" date="2025-08-25T21:47:00Z"} [was
> tasked with]{.insertion author="Ben Remez"
> date="2025-08-25T21:48:00Z"} [assess]{.insertion author="Ben Remez"
> date="2025-08-25T21:47:00Z"}[ing]{.insertion author="Ben Remez"
> date="2025-08-25T21:48:00Z"} [whether the article indicated Verbal or
> Material cooperation or conflict]{.insertion author="Ben Remez"
> date="2025-08-25T21:47:00Z"}[. Then we]{.insertion author="Ben Remez"
> date="2025-08-25T21:48:00Z"} [assigned scores based on the certainty
> of its predictions. This structured reasoning process]{.insertion
> author="Ben Remez" date="2025-08-25T21:47:00Z"} [enabled the pipeline
> to resolve conflicting signals across multiple articles and make more
> consistent, evidence-based]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:12:00Z"} [decisions.]{.insertion
> author="Ben Remez" date="2025-08-25T21:47:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T21:48:00Z"}
>
> [The impact of this step was especially significant for the manual
> extraction pipeline. With Deep Analysis, 'Manual DA' achieved a
> precision of 66.7%,]{.insertion author="Ben Remez"
> date="2025-08-25T21:48:00Z"} [a]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:12:00Z"} [recall of
> 54.2%, an F1 score of 49.5%,]{.insertion author="Ben Remez"
> date="2025-08-25T21:48:00Z"} [and an accuracy score of
> 56%,]{.insertion author="Ben Remez" date="2025-08-25T21:49:00Z"}
> [outperforming all other configurations. In contrast,]{.insertion
> author="Ben Remez" date="2025-08-25T21:48:00Z"} [']{.insertion
> author="Ben Remez" date="2025-08-25T21:49:00Z"}[RAG Improved
> DA]{.insertion author="Ben Remez"
> date="2025-08-25T21:48:00Z"}[']{.insertion author="Ben Remez"
> date="2025-08-25T21:49:00Z"} [also benefited from the enhanced
> reasoning layer but achieved slightly lower gains, reaching a
> precision of 50.4%,]{.insertion author="Ben Remez"
> date="2025-08-25T21:48:00Z"} [a]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:12:00Z"} [recall of
> 51.1%]{.insertion author="Ben Remez"
> date="2025-08-25T21:48:00Z"}[,]{.insertion author="Ben Remez"
> date="2025-08-25T21:49:00Z"} [an F1 score of 50.5%]{.insertion
> author="Ben Remez" date="2025-08-25T21:48:00Z"}[, and an accuracy
> score of]{.insertion author="Ben Remez" date="2025-08-25T21:49:00Z"}
> [54%]{.insertion author="Ben Remez"
> date="2025-08-25T21:50:00Z"}[.]{.insertion author="Ben Remez"
> date="2025-08-25T21:48:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T21:50:00Z"}
>
> [While manual extraction with Deep Analysis currently leads in
> predictive accuracy, 'RAG Improved DA' closed much of the gap and
> remains the more scalable and automated solution. Manual document
> selection inherently provides higher-quality, domain-relevant context,
> but it is also time-consuming and resource-intensive. RAG, combined
> with Deep Analysis, delivers comparable performance without requiring
> manual curation, positioning it as a practical alternative for
> large-scale, real-world forecasting tasks.]{.insertion
> author="Ben Remez" date="2025-08-25T21:50:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T23:32:00Z"}
>
> [With Deep Analysis, 'Manual DA' and 'RAG Improved DA' both surpassed
> MIRAI's baseline scores across several key]{.insertion
> author="Ben Remez" date="2025-08-25T23:32:00Z"} [metrics,
> demonstrating that the proposed pipeline offers competitive or
> superior performance.]{.insertion author="Ben Remez"
> date="2025-08-25T23:33:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T00:57:00Z"}
>
> []{.insertion author="Ben Remez"
> date="2025-08-26T00:57:00Z"}[Additionally, we investigated]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:12:00Z"} [whether a
> threshold could be identified for selectively applying the Deep
> Analysis step. Specifically, we examined the delta between the top two
> ranked hypotheses,]{.insertion author="Ben Remez"
> date="2025-08-26T00:57:00Z"} [assuming that smaller deltas (indicating
> higher]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:12:00Z"} [ambiguity) would benefit most from
> further analysis. However, contrary to expectations, the results
> showed that lower deltas correlated with higher accuracy rather than
> lower, and thus no effective threshold could be established. Further
> details and supporting visualizations are provided in Appendix
> C.]{.insertion author="Ben Remez"
> date="2025-08-26T00:57:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T00:58:00Z"}

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  [**Metric**]{.insertion author="Ben Remez"            [**RAG Improved DA**]{.insertion author="Ben Remez"   [**RAG Improved**]{.insertion author="Ben Remez"      [**RAG**]{.insertion author="Ben Remez"               [**Manual DA**]{.insertion author="Ben Remez"         [**Manual FT**]{.insertion author="Ben Remez"         [**Manual**]{.insertion author="Ben Remez"            [**MIRAI**]{.insertion author="Ben Remez"
  date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion
  author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}
  ----------------------------------------------------- ----------------------------------------------------- ----------------------------------------------------- ----------------------------------------------------- ----------------------------------------------------- ----------------------------------------------------- ----------------------------------------------------- -----------------------------------------------------
  [Precision (↑)]{.insertion author="Ben Remez"         [50.4%]{.insertion author="Ben Remez"                 [51.4%]{.insertion author="Ben Remez"                 [22.1%]{.insertion author="Ben Remez"                 [66.7%]{.insertion author="Ben Remez"                 [50.4%]{.insertion author="Ben Remez"                 [23.2%]{.insertion author="Ben Remez"                 [47.6%]{.insertion author="Ben Remez"
  date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion
  author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}

  [Recall (↑)]{.insertion author="Ben Remez"            [51.1%]{.insertion author="Ben Remez"                 [43.2%]{.insertion author="Ben Remez"                 [37%]{.insertion author="Ben Remez"                   [54.2%]{.insertion author="Ben Remez"                 [41.5%]{.insertion author="Ben Remez"                 [38.8%]{.insertion author="Ben Remez"                 [58.3%]{.insertion author="Ben Remez"
  date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion
  author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}

  [F1 Score (↑)]{.insertion author="Ben Remez"          [50.5%]{.insertion author="Ben Remez"                 [34.7%]{.insertion author="Ben Remez"                 [27.5%]{.insertion author="Ben Remez"                 [49.5%]{.insertion author="Ben Remez"                 [33.3%]{.insertion author="Ben Remez"                 [28.8%]{.insertion author="Ben Remez"                 [44.2%]{.insertion author="Ben Remez"
  date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion
  author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}

  [KL Divergence (↓)]{.insertion author="Ben Remez"     [0.014]{.insertion author="Ben Remez"                 [5.209]{.insertion author="Ben Remez"                 [10.643]{.insertion author="Ben Remez"                [0.312]{.insertion author="Ben Remez"                 [0.88]{.insertion author="Ben Remez"                  [10.546]{.insertion author="Ben Remez"                [5.9]{.insertion author="Ben Remez"
  date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion
  author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}

  [Accuracy (↑)]{.insertion author="Ben Remez"          [54%]{.insertion author="Ben Remez"                   [53%]{.insertion author="Ben Remez"                   [43%]{.insertion author="Ben Remez"                   [56%]{.insertion author="Ben Remez"                   [50%]{.insertion author="Ben Remez"                   [45%]{.insertion author="Ben Remez"                   [42%]{.insertion author="Ben Remez"
  date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion   date="2025-08-25T21:57:00Z"}[]{.paragraph-insertion
  author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}       author="Ben Remez" date="2025-08-25T21:56:00Z"}
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[***Table 1:** The final results for*]{.insertion author="Ben Remez"
date="2025-08-25T21:59:00Z"} [*the*]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:12:00Z"} [*table,
showing*]{.insertion author="Ben Remez" date="2025-08-25T21:59:00Z"}
*[the results for all methods]{.insertion author="Ben Remez"
date="2025-08-25T22:00:00Z"}[.]{.insertion author="Ben Remez"
date="2025-08-25T21:59:00Z"}*[]{.paragraph-insertion author="Ben Remez"
date="2025-08-25T21:22:00Z"}

[These results demonstrate that both manual and RAG-based approaches
achieve comparable performance, with the manual extraction yielding
slightly higher precision, recall, and F1 score. However, the KL
divergence between predicted and actual distributions is nearly
identical in both cases, suggesting that RAG retrieval can approximate
ground truth distributions effectively, albeit with minor trade-offs in
fine-grained classification accuracy.]{.deletion author="Ben Remez"
date="2025-08-25T22:00:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-25T22:00:00Z"}

[In comparison with MIRAI's best-performing first-level relation
classification models (e.g., ReAct with Single Function and All API
access: Precision = 47.6%, Recall = 58.3%, F1 = 44.2%, Quad KL = 5.9),
our system performs at a lower level. This is expected given the
relatively early stage of our system and the limited scope and
resources. However, these initial results are promising and provide a
solid baseline for continued refinement.]{.deletion author="Ben Remez"
date="2025-08-25T22:00:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-25T22:00:00Z"}

[Figure 3 presents the distribution of correct answer ranks for each
query, comparing the manual and RAG-based document extraction
strategies. As shown, both approaches most frequently rank the correct
hypothesis in the top position (rank 1), with manual extraction
achieving this slightly more often. The performance gradually declines
for lower ranks, indicating that both methods are capable of identifying
relevant hypotheses, albeit with some variance in ranking accuracy.
Notably, the RAG approach shows a relatively stronger presence at rank
3, suggesting occasional misranking that may be attributed to less
targeted retrieval. These rank distributions reinforce the metric-based
findings, showing that while manual extraction currently holds a small
advantage, the RAG-based pipeline remains a viable and scalable
alternative.]{.deletion author="Ben Remez"
date="2025-08-25T22:00:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-25T22:00:00Z"}

[![](media/image8.png){width="6.268055555555556in"
height="3.127083333333333in"}]{.deletion author="Ben Remez"
date="2025-08-25T22:06:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-25T22:06:00Z"}

[***Figure***]{.deletion author="Ben Remez" date="2025-08-25T22:06:00Z"}
*[**3**]{.deletion author="Ben Remez" date="2025-08-25T11:57:00Z"}[**:**
The rank position of the correct label as predicted by the system vs
number of queries.]{.deletion author="Ben Remez"
date="2025-08-25T22:06:00Z"}*[]{.paragraph-deletion author="Ben Remez"
date="2025-08-25T22:06:00Z"}

# [Current Limitations]{.deletion author="Ben Remez" date="2025-08-25T22:34:00Z"}

-   [Currently we still use abstract as the overhead of adding the full
    article to the prompt is high and resource consuming both
    financially and computationally.]{.deletion author="Ben Remez"
    date="2025-08-25T22:34:00Z"}[]{.paragraph-deletion
    author="Ben Remez" date="2025-08-25T22:34:00Z"}

-   [Running an experiment using the full list of outcomes (38 outcomes
    in total) takes too long and is resource consuming while we have a
    current limit usage of the Azure Student OpenAI API.]{.deletion
    author="Ben Remez"
    date="2025-08-25T22:34:00Z"}[]{.paragraph-deletion
    author="Ben Remez" date="2025-08-25T22:34:00Z"}

-   [Since we are using the grouping method as in MIRAI we can't
    evaluate the results on second-level relations.]{.deletion
    author="Ben Remez"
    date="2025-08-25T22:34:00Z"}[]{.paragraph-deletion
    author="Ben Remez" date="2025-08-25T22:34:00Z"}

# [Discussion]{.insertion author="Ben Remez" date="2025-08-25T22:21:00Z"}

## [Interpretation of Findings]{.insertion author="Ben Remez" date="2025-08-25T22:21:00Z"}

> [The experiments demonstrated that the refined RAG-based retrieval
> pipeline, combined with fine-tuned indicator generation, significantly
> improved performance compared to the initial baseline. While manual
> extraction maintained slightly higher document relevance, the improved
> RAG approach achieved comparable accuracy]{.insertion
> author="Ben Remez" date="2025-08-25T22:22:00Z"}[. It
> outperformed]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:21:00Z"} [manual selection in several metrics
> (e.g., precision, recall,]{.insertion author="Ben Remez"
> date="2025-08-25T22:22:00Z"} [F1]{.insertion author="Ben Remez"
> date="2025-08-25T22:23:00Z"} [score, and accuracy), while
> also]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:21:00Z"} []{.insertion author="Ben Remez"
> date="2025-08-25T22:23:00Z"}[enabling greater scalability.]{.insertion
> author="Ben Remez" date="2025-08-25T22:22:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T22:23:00Z"}

## [Trade-offs Between Manual and RAG Pipelines]{.insertion author="Ben Remez" date="2025-08-25T22:23:00Z"}

> [Manual extraction leverages human judgment to guarantee high-quality
> documents, but it introduces scalability and efficiency challenges,
> especially when working with large corpora. In contrast, RAG retrieval
> automates the process, reducing reliance on manual labor and enabling
> real-time adaptability]{.insertion author="Ben Remez"
> date="2025-08-25T22:23:00Z"}[. However,]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:12:00Z"} [its
> effectiveness depends heavily on the quality of embeddings, prompt
> design, and filtering mechanisms.]{.insertion author="Ben Remez"
> date="2025-08-25T22:23:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T22:24:00Z"}

## [Role of Deep Analysis]{.insertion author="Ben Remez" date="2025-08-25T22:25:00Z"}

> [The introduction of the Deep Analysis layer significantly improved
> classification between verbal and material categories, especially for
> manually retrieved documents. While the RAG pipeline]{.insertion
> author="Ben Remez" date="2025-08-25T22:25:00Z"} [also benefited,
> manual extraction]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:13:00Z"} [gained more from this step due to its
> higher baseline document quality. This highlights an
> opportunity]{.insertion author="Ben Remez"
> date="2025-08-25T22:25:00Z"} [to optimize RAG retrieval
> further]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:13:00Z"} [to close the performance
> gap.]{.insertion author="Ben Remez"
> date="2025-08-25T22:25:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T22:25:00Z"}

## [Scalability and Practical Implications]{.insertion author="Ben Remez" date="2025-08-25T22:26:00Z"}

> [The improved RAG pipeline demonstrates strong potential for scalable,
> real-world deployment. By leveraging LangChain and Weaviate, the
> system can efficiently handle growing corpora and rapidly adapt to new
> data, enabling broader applicability in geopolitical forecasting and
> similar domains where fast retrieval and contextual reasoning are
> critical.]{.insertion author="Ben Remez"
> date="2025-08-25T22:26:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T22:26:00Z"}

## [Limitations and]{.insertion author="Ben Remez" date="2025-08-25T22:26:00Z"} [Challenges]{.insertion author="Ben Remez" date="2025-08-25T22:27:00Z"}

> [Despite the significant improvements achieved throughout this
> project, several limitations and challenges remain that should be
> considered when interpreting the results:]{.insertion
> author="Ben Remez" date="2025-08-25T22:29:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T22:30:00Z"}

### [Abstract-Only Context]{.insertion author="Ben Remez" date="2025-08-25T22:30:00Z"}

> [Currently, the pipeline]{.insertion author="Ben Remez"
> date="2025-08-25T22:31:00Z"} [utilizes only article abstracts instead
> of full texts due to the high computational and financial costs
> associated with]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:13:00Z"} [embedding and processing complete
> documents. While abstracts often contain sufficient information, this
> limitation may cause the system to miss nuanced contextual signals
> that could improve classification accuracy.]{.insertion
> author="Ben Remez" date="2025-08-25T22:31:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T22:30:00Z"}

### [Resource Constraints]{.insertion author="Ben Remez" date="2025-08-25T22:30:00Z"}

> [Running experiments at scale, especially when testing the complete
> list of 38 potential CAMEO outcomes, remains highly]{.insertion
> author="Ben Remez" date="2025-08-25T22:31:00Z"} []{.insertion
> author="Ben Remez"
> date="2025-08-25T22:32:00Z"}[resource-intensive]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:13:00Z"}[. Due to the
> usage limits of the Azure Student OpenAI API, we focused our
> evaluations on a subset of queries. While sufficient for benchmarking
> purposes, broader-scale testing would provide a more
> robust]{.insertion author="Ben Remez" date="2025-08-25T22:31:00Z"}
> [assessment]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:13:00Z"} [of the pipeline's
> capabilities.]{.insertion author="Ben Remez"
> date="2025-08-25T22:31:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T22:30:00Z"}

### [Outcome Grouping]{.insertion author="Ben Remez" date="2025-08-25T22:30:00Z"}

> [Similar to the MIRAI benchmark, our experiments evaluate predictions
> using the quad-class grouping of CAMEO codes. While this reduces
> sparsity and facilitates comparisons, it also prevents evaluation at
> the second-level relation granularity. As a result, finer distinctions
> between related event types remain unexplored.]{.insertion
> author="Ben Remez" date="2025-08-25T22:32:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T22:30:00Z"}

### [RAG Limitations vs. Manual Quality]{.insertion author="Ben Remez" date="2025-08-25T22:30:00Z"} 

> [Although the improved RAG-based retrieval pipeline outperformed
> manual extraction in several metrics, its effectiveness]{.insertion
> author="Ben Remez" date="2025-08-25T22:33:00Z"} [remains highly
> dependent on the quality of the embedding]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:13:00Z"} [and Weaviate
> indexing. In specific scenarios, manual extraction of documents still
> produces higher-quality context, suggesting potential gains from
> hybrid approaches.]{.insertion author="Ben Remez"
> date="2025-08-25T22:33:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T22:21:00Z"}

# [Conclusions]{.insertion author="Ben Remez" date="2025-08-25T23:04:00Z"}

[This project evaluated an ACH-based inference framework for
geopolitical event forecasting, using the MIRAI benchmark
\[3]{.insertion author="Ben Remez"
date="2025-08-26T10:05:00Z"}[7]{.insertion author="Ben Remez"
date="2025-08-26T11:39:00Z"}[\] as the primary baseline for comparison.
MIRAI provides a structured framework for evaluating LLM-based agents,
offering ground truth data, task formulations, and reference results.
Leveraging this benchmark allowed us to assess our pipeline's
performance meaningfully and ensure comparability with prior
work.]{.insertion author="Ben Remez"
date="2025-08-26T10:05:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T10:05:00Z"}

[By leveraging the MIRAI benchmark as the evaluation ba]{.insertion
author="Ben Remez" date="2025-08-25T23:33:00Z"}[seline, our approach
demonstrated iterative improvements that gradually closed the
performance gap and ultimately exceeded MIRAI's reported results across
several key metrics.]{.insertion author="Ben Remez"
date="2025-08-25T23:34:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-25T23:34:00Z"}

[Over three iterative experiments, we refined both the indicator
generation and retrieval processes, achieving a robust and scalable
forecasting pipeline. Starting from MIRAI's baseline, our initial
results underperformed its benchmarks, but through iterative
improvements, our system gradually closed the performance gap and even
surpassed MIRAI on several key metrics. Specifically, incorporating
balanced indicator prompts, upgrading RAG retrieval with MMR re-ranking,
RRF fusion, and time-decay weighting, and introducing a Deep Analysis
layer to better distinguish between verbal and material classifications
all contributed to significant gains.]{.insertion author="Ben Remez"
date="2025-08-25T23:10:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-25T23:10:00Z"}

[Although manual extraction often yielded more relevant and
higher-quality documents, the enhanced RAG pipeline]{.insertion
author="Ben Remez" date="2025-08-25T23:10:00Z"} [with Deep Analysis and
the Manual Extraction with Deep Analysis]{.insertion author="Ben Remez"
date="2025-08-25T23:12:00Z"} [ultimately achieved better overall
performance across precision, F1 score, and accuracy, while maintaining
full scalability. Compared to MIRAI's reported performance, our final
pipeline demonstrates competitive or superior results while also
introducing design improvements that enhance adaptability to broader
forecasting contexts.]{.insertion author="Ben Remez"
date="2025-08-25T23:10:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-25T23:13:00Z"}

[Overall, this work demonstrates the value of combining retrieval-aware
architectures with LLM-based reasoning and highlights the potential to
outperform existing baselines]{.insertion author="Ben Remez"
date="2025-08-25T23:13:00Z"}[, such as MIRAI,]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:14:00Z"} [through
iterative optimization of retrieval relevance and reasoning fidelity.
By]{.insertion author="Ben Remez" date="2025-08-25T23:13:00Z"}
[enhancing document coverage and automating key stages of the pipeline,
we take a significant]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:14:00Z"} [step toward]{.insertion author="Ben Remez"
date="2025-08-25T23:13:00Z"} [achieving scalable and]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:21:00Z"} [accurate
geopolitical forecasting.]{.insertion author="Ben Remez"
date="2025-08-25T23:13:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-25T23:04:00Z"}

# [Conclusions and]{.deletion author="Ben Remez" date="2025-08-25T23:04:00Z"} [Future]{.deletion author="Alexander Apartsin" date="2025-08-27T11:14:00Z"}[Future]{.insertion author="Alexander Apartsin" date="2025-08-27T11:14:00Z"} Work

[Looking ahead,]{.insertion author="Ben Remez"
date="2025-08-25T23:36:00Z"} [there are several directions in which this
research can be extended]{.insertion author="Ben Remez"
date="2025-08-25T23:37:00Z"} [to improve performance, scalability, and
applicability further]{.insertion author="Alexander Apartsin"
date="2025-08-27T11:14:00Z"}[:]{.insertion author="Ben Remez"
date="2025-08-25T23:37:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-25T23:37:00Z"}

## [Benchmark on Larger and Diverse Datasets]{.insertion author="Ben Remez" date="2025-08-25T23:37:00Z"}

> [Future evaluations should test the generalizability of the pipeline
> across multiple domains and larger datasets. While the MIRAI dataset
> provided a strong benchmark for geopolitical event forecasting,
> broader testing would better assess the system's robustness and
> transferability to new contexts.]{.insertion author="Ben Remez"
> date="2025-08-25T23:38:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-25T23:38:00Z"}

## [Refine RAG Retrieval]{.insertion author="Ben Remez" date="2025-08-25T23:38:00Z"}

> [Improving the Retrieval-Augmented Generation (RAG) component remains
> a priority. Advanced query optimization strategies, improved
> similarity metrics, and adaptive retrieval thresholds can enhance
> recall while maintaining high precision. This could lead to a more
> comprehensive evidence base for the model's reasoning.]{.insertion
> author="Ben Remez" date="2025-08-25T23:38:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T23:38:00Z"}

## [Explore Multi-Agent LLM Architectures]{.insertion author="Ben Remez" date="2025-08-25T23:39:00Z"}

> [Introducing a multi-agent setup, where specialized language models
> handle distinct subtasks (e.g., retrieval, reasoning, classification),
> could improve robustness and reduce errors. This modular design could
> enhance both interpretability and flexibility.]{.insertion
> author="Ben Remez" date="2025-08-25T23:39:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T23:39:00Z"}

## [Ful]{.insertion author="Ben Remez" date="2025-08-25T23:39:00Z"}[l Article Analysis]{.insertion author="Ben Remez" date="2025-08-25T23:40:00Z"}

> [Currently, the system relies on article abstracts due to cost and
> processing constraints. Extending the framework to handle full-text
> analysis would enable richer contextual understanding, improving
> forecast accuracy. This requires balancing computational costs with
> the added benefits of deeper content coverage.]{.insertion
> author="Ben Remez" date="2025-08-25T23:40:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T23:40:00Z"}

## [Extend CAMEO Relations]{.insertion author="Ben Remez" date="2025-08-25T23:40:00Z"}

> [Future experiments could leverage the full granularity of CAMEO's 38
> relation types rather than aggregating into four broad classes.
> Second-level relation forecasting would capture more nuanced
> distinctions in inter-state dynamics but would require careful
> handling of sparsity and class imbalance.]{.insertion
> author="Ben Remez" date="2025-08-25T23:40:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T23:41:00Z"}

## [Optimize Prompt Engineering]{.insertion author="Ben Remez" date="2025-08-25T23:41:00Z"}

> [Exploring reinforcement learning or human-in-the-loop prompt
> refinement strategies could improve indicator generation and
> classification accuracy. Optimized prompting would reduce uncertainty
> and enhance the quality of downstream decisions.]{.insertion
> author="Ben Remez" date="2025-08-25T23:41:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-25T23:37:00Z"}

[This study presented a novel approach for applying Analysis of
Competing Hypotheses (ACH) methodology to the domain of event
forecasting, leveraging LLM-based pipelines and comparing manual versus
retrieval-augmented generation (RAG) document selection mechanisms. Our
evaluation, conducted using the MIRAI benchmark framework \[31\] and
focusing specifically on first-level relation classification under the
quad setting, demonstrated that both manual and RAG-based pipelines are
capable of producing reasonable predictions. However, the manually
curated document selection consistently outperformed the RAG-based
pipeline across all key metrics---including precision, recall, F1-score,
and KL divergence---albeit by a relatively narrow margin.]{.deletion
author="Ben Remez" date="2025-08-25T23:36:00Z"}[]{.paragraph-deletion
author="Ben Remez" date="2025-08-25T23:36:00Z"}

[The comparative analysis of hypothesis rank distributions further
revealed that both methods frequently ranked the correct hypothesis in
the top position, with manual extraction maintaining a slight edge in
rank-1 accuracy. These findings underscore the effectiveness of
structured manual retrieval in scenarios requiring high interpretability
and precision. Nevertheless, the RAG-based pipeline\'s proximity in
performance suggests its potential for scalable and automated deployment
in larger systems.]{.deletion author="Ben Remez"
date="2025-08-25T23:36:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-25T23:36:00Z"}

[For future work, we aim to enhance the RAG component by adjusting the
retrieval query so that it better aligns with the hypothesis evaluation
task. We also intend to fine-tune the prompts in an attempt to make them
clearer and more task precise. Finally, exploring more advanced
LLM-agent reasoning strategies and multi-step tool use may offer further
improvements in both interpretability and predictive
accuracy.]{.deletion author="Ben Remez"
date="2025-08-25T23:36:00Z"}[]{.paragraph-deletion author="Ben Remez"
date="2025-08-25T23:36:00Z"}

# References

> \[1\] Analysis of Competing Hypotheses,
> https://en.wikipedia.org/wiki/Analysis_of_competing_hypotheses
>
> \[2\] Benjamin, D. M., Morstatter, F., Abbas, A. E., Abeliuk, A.,
> Atanasov, P., Bennett, S., \... & Galstyan, A. (2023). Hybrid
> forecasting of geopolitical events. AI Magazine, 44(1),
> 112-128.[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}
>
> [\[3\] Carbonell, J., & Goldstein, J. (1998, August). The use of MMR,
> diversity-based reranking for reordering documents and producing
> summaries]{.insertion author="Ben Remez" date="2025-08-26T09:50:00Z"}
> [in]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:15:00Z"}[ *Proceedings of the 21st annual
> international ACM SIGIR conference on Research and development in
> information retrieval* (pp. 335-336).]{.insertion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}

\[4[3]{.deletion author="Ben Remez" date="2025-08-26T09:50:00Z"}\]
Cognitive bias, <https://en.wikipedia.org/wiki/Cognitive_bias>

\[5[4]{.deletion author="Ben Remez" date="2025-08-26T09:50:00Z"}\]
Confirmation bias, <https://en.wikipedia.org/wiki/Confirmation_bias>

> \[6[5]{.deletion author="Ben Remez" date="2025-08-26T09:50:00Z"}\]
> Conflict and Mediation Event Observations,
> <https://en.wikipedia.org/wiki/Conflict_and_Mediation_Event_Observations>[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-26T10:12:00Z"}
>
> [\[]{.insertion author="Ben Remez"
> date="2025-08-26T10:12:00Z"}[7]{.insertion author="Ben Remez"
> date="2025-08-26T10:18:00Z"}[\] Cormack, G. V., Clarke, C. L., &
> Buettcher, S. (2009, July). Reciprocal rank fusion
> outperforms]{.insertion author="Ben Remez"
> date="2025-08-26T10:12:00Z"} [Condorcet]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:15:00Z"} [and
> individual rank learning methods]{.insertion author="Ben Remez"
> date="2025-08-26T10:12:00Z"} [in]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:15:00Z"}[ *Proceedings
> of the 32nd*]{.insertion author="Ben Remez"
> date="2025-08-26T10:12:00Z"} [*International ACM SIGIR Conference on
> Research and Development in Information Retrieval*]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:21:00Z"}[ (pp.
> 758-759).]{.insertion author="Ben Remez" date="2025-08-26T10:12:00Z"}
>
> \[[8]{.insertion author="Ben Remez"
> date="2025-08-26T10:19:00Z"}[6]{.deletion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}\] Dempster-Shafer Theory,
> <https://en.wikipedia.org/wiki/Dempster%E2%80%93Shafer_theory>
>
> \[[9]{.insertion author="Ben Remez"
> date="2025-08-26T10:19:00Z"}[7]{.deletion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}\] Dhami, M. K., Belton, I. K., & Mandel,
> D. R. (2019). The "analysis of competing hypotheses" in intelligence
> analysis. Applied Cognitive Psychology, 33(6), 1080-1090.
>
> \[[10]{.insertion author="Ben Remez"
> date="2025-08-26T10:19:00Z"}[8]{.deletion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}\] Evidential Reasoning Approach,
> <https://en.wikipedia.org/wiki/Evidential_reasoning_approach>
>
> \[[11]{.insertion author="Ben Remez"
> date="2025-08-26T10:19:00Z"}[9]{.deletion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}\] Fairfield, T., & Charman, A. E. (2017).
> Explicit Bayesian analysis for process tracing: Guidelines,
> opportunities, and caveats. Political Analysis, 25(3), 363-380.
>
> \[1[2]{.insertion author="Ben Remez"
> date="2025-08-26T10:20:00Z"}[0]{.deletion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}\] Gao, Y., Xiong, Y., Gao, X., Jia, K.,
> Pan, J., Bi, Y., \... [&]{.deletion author="Alexander Apartsin"
> date="2025-08-27T11:21:00Z"} [Y]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:21:00Z"} Wang, H.
> (2023). Retrieval-augmented generation for large language models: A
> survey. arXiv preprint arXiv:2312.10997, 2.
>
> \[1[3]{.insertion author="Ben Remez"
> date="2025-08-26T10:20:00Z"}[1]{.deletion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}\] Gruver, N., Finzi, M., Qiu, S., &
> Wilson, A. G. (2023). Large language models are zero-shot time series
> forecasters. Advances in Neural Information Processing Systems, 36,
> 19622-19635.
>
> \[1[4]{.insertion author="Ben Remez"
> date="2025-08-26T10:20:00Z"}[2]{.deletion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}\] Heuer, Richards J. Psychology of
> Intelligence Analysis. Center for the Study of Intelligence, 1999
>
> \[1[5]{.insertion author="Ben Remez"
> date="2025-08-26T10:20:00Z"}[3]{.deletion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}\] Hossain, K. T., Harutyunyan, H., Ning,
> Y., Kennedy, B., Ramakrishnan, N., & Galstyan, A. (2022). Identifying
> [geopolitical event precursors using attention-based]{.deletion
> author="Alexander Apartsin" date="2025-08-27T11:15:00Z"}[Geopolitical
> Event Precursors Using Attention-Based]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:15:00Z"} LSTMs.
> Frontiers in Artificial Intelligence, 5, 893875.
>
> \[1[6]{.insertion author="Ben Remez"
> date="2025-08-26T10:21:00Z"}[4]{.deletion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}\] Karvetski, C.W., Olson, K.C., Gantz,
> D.T. et al. Structuring and analysing competing hypotheses with
> Bayesian networks for intelligence analysis. EURO J Decis Process 1,
> 205--231 (2013)
>
> \[1[7]{.insertion author="Ben Remez"
> date="2025-08-26T10:21:00Z"}[5]{.deletion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}\] Kejriwal, M. (2021). Link [prediction
> between structured geopolitical events: models and
> experiments]{.deletion author="Alexander Apartsin"
> date="2025-08-27T11:15:00Z"}[Prediction between Structured
> Geopolitical Events: Models and Experiments]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:15:00Z"}. Frontiers in
> [big]{.deletion author="Alexander Apartsin"
> date="2025-08-27T11:21:00Z"} [Big]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:21:00Z"} Data, 4,
> 779792. 15
>
> \[1[8]{.insertion author="Ben Remez"
> date="2025-08-26T10:21:00Z"}[6]{.deletion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}\] Kunkler, K. S., & Roy, T. (2023).
> Reducing the impact of cognitive bias in decision making: Practical
> actions for forensic science practitioners. Forensic Science
> International: Synergy, 7, 100341.[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-26T11:15:00Z"}
>
> [\[19\] LangChain,]{.insertion author="Ben Remez"
> date="2025-08-26T11:15:00Z"} [[]{.insertion author="Ben Remez"
> date="2025-08-26T11:25:00Z"}https://js.langchain.com/docs/concepts/why_langchain/]{.insertion
> author="Ben Remez" date="2025-08-26T11:15:00Z"}[]{.insertion
> author="Ben Remez" date="2025-08-26T11:25:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-26T11:25:00Z"}
>
> [\[20\] LangChain and Its Role in Building Robust Rag Systems,
> https://blog.gopenai.com/langchain-and-its-role-in-building-robust-rag-systems-2808b7e5ddb9]{.insertion
> author="Ben Remez" date="2025-08-26T11:25:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-26T11:31:00Z"}
>
> [\[21\] Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013).
> Efficient estimation of word representations in vector space. *arXiv
> preprint arXiv:1301.3781*.]{.insertion author="Ben Remez"
> date="2025-08-26T11:31:00Z"}
>
> \[[2]{.insertion author="Ben Remez"
> date="2025-08-26T11:15:00Z"}[2]{.insertion author="Ben Remez"
> date="2025-08-26T11:32:00Z"}[1]{.deletion author="Ben Remez"
> date="2025-08-26T11:15:00Z"}[7]{.deletion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}\] Minaee, S., Mikolov, T., Nikzad, N.,
> Chenaghlu, M., Socher, R., Amatriain, X., & Gao, J. (2024). Large
> Language Models: A Survey. arXiv preprint arXiv:2402.06196
>
> \[[2]{.insertion author="Ben Remez"
> date="2025-08-26T10:21:00Z"}[3]{.insertion author="Ben Remez"
> date="2025-08-26T11:32:00Z"}[8]{.deletion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}\] Mirtaheri, M., Abu-El-Haija, S., &
> Hossain, T. (2019). Tensor-based method for []{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:16:00Z"}temporal
> geopolitical event forecasting. In [the]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:16:00Z"} ICML Workshop
> on Learning and Reasoning with [GraphStructured]{.deletion
> author="Alexander Apartsin" date="2025-08-27T11:16:00Z"}
> [Graph-Structured]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:16:00Z"} Data.
>
> \[2[4]{.insertion author="Ben Remez"
> date="2025-08-26T11:32:00Z"}[19]{.deletion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}\] Lidén, M., Thiblin, I., & Dror, I. E.
> (2023). The role of alternative hypotheses in reducing bias in
> forensic medical experts' [decision making]{.deletion
> author="Alexander Apartsin"
> date="2025-08-27T11:21:00Z"}[decision-making]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:21:00Z"}. Science &
> Justice, 63(5), 581-587.
>
> \[2[5]{.insertion author="Ben Remez"
> date="2025-08-26T11:32:00Z"}[0]{.deletion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}\] MacLean, C. L. (2022). Cognitive bias
> in workplace investigation: Problems, perspectives and proposed
> solutions. Applied Ergonomics, 105, 103860.
>
> \[2[6]{.insertion author="Ben Remez"
> date="2025-08-26T11:32:00Z"}[1]{.deletion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}\] Martyn, P. (2021). Using Quantitative
> Analytical Methods to Support Qualitative Data Analysis: Lessons
> Learnt During a PhD Study. Accounting, Finance & Governance Review,
> 27, 70-80.
>
> \[2[7]{.insertion author="Ben Remez"
> date="2025-08-26T11:32:00Z"}[2]{.deletion author="Ben Remez"
> date="2025-08-26T09:50:00Z"}\] Meerveld, H. W., Lindelauf, R. H. A.,
> Postma, E. O., & Postma, M. (2023). The [irresponsibility of not using
> AI in the military]{.deletion author="Alexander Apartsin"
> date="2025-08-27T11:21:00Z"}[Irresponsibility of Not Utilizing AI in
> the Military]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:21:00Z"}. Ethics and Information Technology,
> 25(1), 14.
>
> \[2[8]{.insertion author="Ben Remez"
> date="2025-08-26T11:32:00Z"}[3]{.deletion author="Ben Remez"
> date="2025-08-26T09:51:00Z"}\] Multi-Agent LLM Applications \| A
> Review of Current Research, Tools, and Challenges,
> <https://newsletter.victordibia.com/p/multi-agent-llm-applications-a-review>

\[2[9]{.insertion author="Ben Remez"
date="2025-08-26T11:32:00Z"}[4]{.deletion author="Ben Remez"
date="2025-08-26T09:51:00Z"}\] Robust Decision-Making,
<https://en.wikipedia.org/wiki/Robust_decision-making>

\[[30]{.insertion author="Ben Remez"
date="2025-08-26T11:32:00Z"}[2]{.deletion author="Ben Remez"
date="2025-08-26T11:32:00Z"}[5]{.deletion author="Ben Remez"
date="2025-08-26T09:51:00Z"}\] RootClaim, <https://www.rootclaim.com/>

> \[[3]{.insertion author="Ben Remez"
> date="2025-08-26T11:26:00Z"}[1]{.insertion author="Ben Remez"
> date="2025-08-26T11:32:00Z"}[2]{.deletion author="Ben Remez"
> date="2025-08-26T11:26:00Z"}[6]{.deletion author="Ben Remez"
> date="2025-08-26T09:51:00Z"}\] Roy, D., Srivastava, R., Jat, M., &
> Karaca, M. S. (2022). A complete overview of analytics techniques:
> descriptive, predictive, and prescriptive. Decision intelligence
> analytics and the implementation of strategic business management,
> 15-30.
>
> \[[3]{.insertion author="Ben Remez"
> date="2025-08-26T11:16:00Z"}[2]{.insertion author="Ben Remez"
> date="2025-08-26T11:32:00Z"}[2]{.deletion author="Ben Remez"
> date="2025-08-26T11:16:00Z"}[7]{.deletion author="Ben Remez"
> date="2025-08-26T09:51:00Z"}\] Su, J., Jiang, C., Jin, X., Qiao, Y.,
> Xiao, T., Ma, H., Wei, R., Jing, Z., Xu, J., & Lin, J. (2024). Large
> language models for forecasting and anomaly detection: A systematic
> literature review. arXiv preprint arXiv:2402.10350.
>
> \[[3]{.insertion author="Ben Remez"
> date="2025-08-26T10:23:00Z"}[3]{.insertion author="Ben Remez"
> date="2025-08-26T11:32:00Z"}[8]{.deletion author="Ben Remez"
> date="2025-08-26T09:51:00Z"}\] Tan, M., Merrill, M., Gupta, V.,
> Althoff, T., & Hartvigsen, T. (2025). Are language models actually
> useful for time series [forecasting?.]{.deletion
> author="Alexander Apartsin"
> date="2025-08-27T11:17:00Z"}[forecasting?]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:17:00Z"} Advances in
> Neural Information Processing Systems, 37, 60162-
> 60191.[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T11:36:00Z"}
>
> [\[3]{.insertion author="Ben Remez"
> date="2025-08-26T11:36:00Z"}[4]{.insertion author="Ben Remez"
> date="2025-08-26T11:37:00Z"}[\] Top 7 Best Vector Databases, DataCamp,
> https://www.datacamp.com/blog/the-top-5-vector-databases]{.insertion
> author="Ben Remez" date="2025-08-26T11:36:00Z"}
>
> \[3[5]{.insertion author="Ben Remez"
> date="2025-08-26T11:37:00Z"}[29]{.deletion author="Ben Remez"
> date="2025-08-26T09:51:00Z"}\] Whitesmith, Martha. Cognitive Bias in
> Intelligence Analysis: Testing the Analysis of Competing Hypotheses
> Method. Edinburgh University Press, 2020
>
> \[3[6]{.insertion author="Ben Remez"
> date="2025-08-26T11:37:00Z"}[0]{.deletion author="Ben Remez"
> date="2025-08-26T09:51:00Z"}\] Xepapadeas, A. (2024). Uncertainty and
> climate change: The IPCC approach vs decision theory. Journal of
> [Behavioral]{.deletion author="Alexander Apartsin"
> date="2025-08-27T11:17:00Z"}[Behavioural]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:17:00Z"} and
> Experimental Economics, 109, 102188.
>
> \[3[7]{.insertion author="Ben Remez"
> date="2025-08-26T11:37:00Z"}[1]{.deletion author="Ben Remez"
> date="2025-08-26T09:51:00Z"}\] Ye, C., Hu, Z., Deng, Y., Huang, Z.,
> Ma, M. D., Zhu, Y., & Wang, W. (2024). Mirai: Evaluating
> [llm]{.deletion author="Alexander Apartsin"
> date="2025-08-27T11:17:00Z"} [LLM]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:17:00Z"} agents for
> event forecasting. arXiv preprint
> arXiv:2407.01231.[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-24T21:54:00Z"}

# [Appendix A]{.insertion author="Ben Remez" date="2025-08-26T00:28:00Z"} [-- Deep Analysis Prompt]{.insertion author="Ben Remez" date="2025-08-26T00:29:00Z"}

[*"*]{.insertion author="Ben Remez" date="2025-08-26T00:30:00Z"}[*You
are a geopolitical analyst. Your task is to assess international news
articles and forecast the nature of future relations between two
countries, based on*]{.insertion author="Ben Remez"
date="2025-08-26T00:29:00Z"} [*the*]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:21:00Z"} [*textual
context.*]{.insertion author="Ben Remez"
date="2025-08-26T00:29:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T00:29:00Z"}

[*We are analyzing the likely future relationship between {} and {},
using the news article provided below. It is already established that
the relationship will be classified as*]{.insertion author="Ben Remez"
date="2025-08-26T00:29:00Z"} [*'\<Cooperation/Conflict \>'*]{.insertion
author="Ben Remez" date="2025-08-26T00:30:00Z"}[*, meaning:*]{.insertion
author="Ben Remez" date="2025-08-26T00:29:00Z"} *[']{.insertion
author="Ben Remez"
date="2025-08-26T00:31:00Z"}[\<Cooperation/]{.insertion
author="Ben Remez" date="2025-08-26T00:30:00Z"}[Conflict
description\>']{.insertion author="Ben Remez"
date="2025-08-26T00:31:00Z"}[.]{.insertion author="Ben Remez"
date="2025-08-26T00:29:00Z"}*[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T00:29:00Z"}

[*Your task is to determine the type of this relationship based on the
news article:*]{.insertion author="Ben Remez"
date="2025-08-26T00:29:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T00:29:00Z"}

[*- Material*]{.insertion author="Ben Remez"
date="2025-08-26T00:29:00Z"} [*'\<Cooperation/Conflict \>'*]{.insertion
author="Ben Remez" date="2025-08-26T00:31:00Z"} [*- typically
involves:*]{.insertion author="Ben Remez" date="2025-08-26T00:29:00Z"}
[*'\<Material Cooperation/Conflict description\>'*]{.insertion
author="Ben Remez" date="2025-08-26T00:31:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-26T00:29:00Z"}

[*- Verbal*]{.insertion author="Ben Remez" date="2025-08-26T00:29:00Z"}
[*'\<Cooperation/Conflict \>'*]{.insertion author="Ben Remez"
date="2025-08-26T00:31:00Z"} [*- typically involves:*]{.insertion
author="Ben Remez" date="2025-08-26T00:29:00Z"} [*'\<Verbal
Cooperation/Conflict description\>'*]{.insertion author="Ben Remez"
date="2025-08-26T00:32:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T00:29:00Z"}

[*Please provide the following:*]{.insertion author="Ben Remez"
date="2025-08-26T00:29:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T00:29:00Z"}

*[- Predicted relation type: verbal, material]{.insertion
author="Ben Remez" date="2025-08-26T00:29:00Z"}[,]{.insertion
author="Alexander Apartsin" date="2025-08-27T11:21:00Z"}* [*or
uncertain*]{.insertion author="Ben Remez"
date="2025-08-26T00:29:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T00:29:00Z"}

[*- Certainty level: high or low (if verbal or material)*]{.insertion
author="Ben Remez" date="2025-08-26T00:29:00Z"}[]{.paragraph-insertion
author="Ben Remez" date="2025-08-26T00:29:00Z"}

[*- Explanation: A concise justification based on evidence from the
article*]{.insertion author="Ben Remez"
date="2025-08-26T00:29:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T00:29:00Z"}

[*News Article:*]{.insertion author="Ben Remez"
date="2025-08-26T00:29:00Z"}[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T00:29:00Z"}

*['\<Article Context\>']{.insertion author="Ben Remez"
date="2025-08-26T00:32:00Z"}["]{.insertion author="Ben Remez"
date="2025-08-26T00:30:00Z"}*[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T00:28:00Z"}

# [Appendix B -- Entropy Analysis of Prediction Distributions]{.insertion author="Ben Remez" date="2025-08-26T00:38:00Z"}

> [In addition to evaluating standard performance metrics such as
> precision, recall, F1 score, accuracy, and KL divergence, we also
> examined the entropy of the output distributions. Entropy provides a
> measure of the uncertainty associated with the model's predictions.
> Lower entropy indicates that the model assigns higher confidence to
> its preferred classification, while higher entropy reflects greater
> uncertainty or a more uniform distribution of probabilities across
> categories.]{.insertion author="Ben Remez"
> date="2025-08-26T00:42:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T00:42:00Z"}

## [Methodology]{.insertion author="Ben Remez" date="2025-08-26T00:43:00Z"}

> [Entropy was computed using the Shannon entropy formula over the
> probability distribution of the four quad-class categories (Verbal
> Cooperation, Material Cooperation, Verbal Conflict, Material
> Conflict). For each query, the model produced probability scores for
> the four possible outcomes, and the entropy was calculated
> as:]{.insertion author="Ben Remez"
> date="2025-08-26T00:44:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T00:44:00Z"}
>
> $$()\sum_{}^{}{_{}}{_{}}$$[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T00:45:00Z"}
>
> [Where]{.insertion author="Ben Remez" date="2025-08-26T00:45:00Z"}
> $_{}$ [represents the predicted probability for]{.insertion
> author="Ben Remez" date="2025-08-26T00:46:00Z"} [the]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:19:00Z"}
> [class]{.insertion author="Ben Remez" date="2025-08-26T00:46:00Z"}
> $$[. The results presented here are the average entropy across all
> queries for each system.]{.insertion author="Ben Remez"
> date="2025-08-26T00:46:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T00:46:00Z"}

## [Results]{.insertion author="Ben Remez" date="2025-08-26T00:47:00Z"}

> [The average entropy values for the MIRAI benchmark and our pipelines
> are shown below:]{.insertion author="Ben Remez"
> date="2025-08-26T00:47:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T00:47:00Z"}

  -----------------------------------------------------------------------------------------------------------
  [**System**]{.insertion author="Ben Remez"            [**Average Entropy**]{.insertion author="Ben Remez"
  date="2025-08-26T00:47:00Z"}[]{.paragraph-insertion   date="2025-08-26T00:47:00Z"}[]{.paragraph-insertion
  author="Ben Remez" date="2025-08-26T00:47:00Z"}       author="Ben Remez" date="2025-08-26T00:47:00Z"}
  ----------------------------------------------------- -----------------------------------------------------
  [Manual FT]{.insertion author="Ben Remez"             [0.1444]{.insertion author="Ben Remez"
  date="2025-08-26T00:47:00Z"}[]{.paragraph-insertion   date="2025-08-26T00:47:00Z"}[]{.paragraph-insertion
  author="Ben Remez" date="2025-08-26T00:47:00Z"}       author="Ben Remez" date="2025-08-26T00:47:00Z"}

  [RAG Improved]{.insertion author="Ben Remez"          [0.1421]{.insertion author="Ben Remez"
  date="2025-08-26T00:47:00Z"}[]{.paragraph-insertion   date="2025-08-26T00:47:00Z"}[]{.paragraph-insertion
  author="Ben Remez" date="2025-08-26T00:47:00Z"}       author="Ben Remez" date="2025-08-26T00:47:00Z"}

  [MI]{.insertion author="Ben Remez"                    [0.6745]{.insertion author="Ben Remez"
  date="2025-08-26T00:47:00Z"}[RAI]{.insertion          date="2025-08-26T00:47:00Z"}[]{.paragraph-insertion
  author="Ben Remez"                                    author="Ben Remez" date="2025-08-26T00:47:00Z"}
  date="2025-08-26T00:48:00Z"}[]{.paragraph-insertion   
  author="Ben Remez" date="2025-08-26T00:47:00Z"}       
  -----------------------------------------------------------------------------------------------------------

[***Table***]{.insertion author="Ben Remez" date="2025-08-26T01:03:00Z"}
*[**B.1**]{.insertion author="Ben Remez"
date="2025-08-26T01:06:00Z"}[**:** The calculated average entropy per
system.]{.insertion author="Ben Remez"
date="2025-08-26T01:03:00Z"}*[]{.paragraph-insertion author="Ben Remez"
date="2025-08-26T00:48:00Z"}

## [Interpretation]{.insertion author="Ben Remez" date="2025-08-26T00:48:00Z"}

> [The results demonstrate a substantial reduction in entropy for both
> our pipelines compared to the MIRAI benchmark. While MIRAI's outputs
> exhibited an average entropy of 0.6745, indicating relatively high
> uncertainty in its predictions, our pipelines achieved much lower
> values: 0.1444 for Manual FT and 0.1421 for RAG Improved. This
> suggests that our approaches not only improved classification
> performance but also produced more confident predictions.]{.insertion
> author="Ben Remez" date="2025-08-26T00:48:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-26T00:48:00Z"}
>
> [Notably, the RAG Improved pipeline yielded the lowest entropy,
> reflecting both the effectiveness of the refined indicator prompts and
> the enhanced retrieval infrastructure. These results reinforce the
> observation that our system consistently produces more decisive
> outputs than MIRAI, while maintaining scalability and competitive
> accuracy.]{.insertion author="Ben Remez"
> date="2025-08-26T00:48:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T00:49:00Z"}

# [Appendix C]{.insertion author="Ben Remez" date="2025-08-26T00:56:00Z"} [--]{.insertion author="Ben Remez" date="2025-08-26T00:59:00Z"} []{.insertion author="Ben Remez" date="2025-08-26T00:56:00Z"}[Threshold-Based Deep Analysis Exploration]{.insertion author="Ben Remez" date="2025-08-26T00:59:00Z"}

## [Motivation]{.insertion author="Ben Remez" date="2025-08-26T00:59:00Z"}

> [After implementing the Deep Analysis step, we sought to investigate
> whether its application could be optimized by introducing a threshold
> mechanism. The underlying hypothesis was that when the confidence gap
> between the top two ranked hypotheses was small, the system's decision
> was more ambiguous]{.insertion author="Ben Remez"
> date="2025-08-26T00:59:00Z"}[. Thus]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:21:00Z"}[,]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:18:00Z"} [additional
> Deep Analysis might yield improved accuracy. Conversely, if the
> confidence gap was large, the top-ranked hypothesis could be accepted
> directly without the added computational overhead of Deep
> Analysis.]{.insertion author="Ben Remez"
> date="2025-08-26T00:59:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T01:00:00Z"}

## [Methodology]{.insertion author="Ben Remez" date="2025-08-26T01:00:00Z"}

> [To test this hypothesis, we calculated the delta between the scores
> of the top]{.insertion author="Ben Remez" date="2025-08-26T01:00:00Z"}
> [two-ranked]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:18:00Z"} [hypotheses for each query. We then
> measured the accuracy of the system when applying Deep Analysis only
> to queries falling below different delta thresholds. This analysis was
> performed on queries where the correct answer appeared in either
> rank]{.insertion author="Ben Remez" date="2025-08-26T01:00:00Z"}
> [one]{.insertion author="Alexander Apartsin"
> date="2025-08-27T11:18:00Z"} [or rank]{.insertion author="Ben Remez"
> date="2025-08-26T01:00:00Z"} [two]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:18:00Z"}[, as these
> cases best capture ambiguity between competing hypotheses.]{.insertion
> author="Ben Remez" date="2025-08-26T01:00:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-26T01:01:00Z"}
>
> [Two plots were generated:]{.insertion author="Ben Remez"
> date="2025-08-26T01:02:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T01:02:00Z"}

-   [A raw accuracy vs. delta threshold curve (]{.insertion
    author="Ben Remez" date="2025-08-26T01:02:00Z"}[Figure]{.insertion
    author="Alexander Apartsin" date="2025-08-27T11:18:00Z"}
    []{.insertion author="Ben Remez"
    date="2025-08-26T01:02:00Z"}[7)]{.insertion author="Ben Remez"
    date="2025-08-26T01:03:00Z"}[, showing the direct relationship
    between delta values and accuracy.]{.insertion author="Ben Remez"
    date="2025-08-26T01:02:00Z"}[]{.paragraph-insertion
    author="Ben Remez" date="2025-08-26T01:02:00Z"}

-   [A smoothed version of the curve]{.insertion author="Ben Remez"
    date="2025-08-26T01:02:00Z"} [(figure 8)]{.insertion
    author="Ben Remez" date="2025-08-26T01:03:00Z"}[, applying a
    quantile-based smoothing technique]{.insertion author="Ben Remez"
    date="2025-08-26T01:02:00Z"} [to highlight overall trends
    better]{.insertion author="Alexander Apartsin"
    date="2025-08-27T11:18:00Z"} [and minimize noise.]{.insertion
    author="Ben Remez"
    date="2025-08-26T01:02:00Z"}[]{.paragraph-insertion
    author="Ben Remez" date="2025-08-26T01:04:00Z"}

## [Results]{.insertion author="Ben Remez" date="2025-08-26T01:04:00Z"}

> [Contrary to our expectations, the results revealed that smaller
> deltas]{.insertion author="Ben Remez" date="2025-08-26T01:04:00Z"}[,
> which were assumed to reflect greater ambiguity,]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:21:00Z"} [actually
> correlated with higher accuracy. This trend was consistent in both the
> raw and smoothed curves. As delta values increased, accuracy
> decreased, suggesting that the system was most reliable in cases of
> close competition between hypotheses.]{.insertion author="Ben Remez"
> date="2025-08-26T01:04:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T01:04:00Z"}
>
> [Figures C.1 and C.2 illustrate these findings.]{.insertion
> author="Ben Remez" date="2025-08-26T01:05:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-26T01:05:00Z"}
>
> [![A graph with a line AI-generated content may be
> incorrect.](media/image10.png){width="6.268055555555556in"
> height="4.029166666666667in"}]{.insertion author="Ben Remez"
> date="2025-08-26T01:05:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T01:05:00Z"}
>
> [**Figure C**]{.insertion author="Ben Remez"
> date="2025-08-26T01:05:00Z"}[**.1:** Raw accuracy vs. delta threshold
> (per-query deltas)]{.insertion author="Ben Remez"
> date="2025-08-26T01:06:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T01:06:00Z"}
>
> [![A graph of a line AI-generated content may be
> incorrect.](media/image11.png){width="6.268055555555556in"
> height="4.004166666666666in"}]{.insertion author="Ben Remez"
> date="2025-08-26T01:06:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T01:06:00Z"}
>
> [**Figure C.2:**]{.insertion author="Ben Remez"
> date="2025-08-26T01:06:00Z"} [Smoothed accuracy vs. delta threshold
> (quantile analysis).]{.insertion author="Ben Remez"
> date="2025-08-26T01:07:00Z"}[]{.paragraph-insertion author="Ben Remez"
> date="2025-08-26T01:07:00Z"}

## [Interpretation]{.insertion author="Ben Remez" date="2025-08-26T01:07:00Z"}

> [The observed trend implies that introducing a simple thresholding
> mechanism to trigger Deep Analysis is not effective. Instead of
> improving decision quality in ambiguous cases, the analysis showed
> that the system already performed well when deltas were small. As
> deltas grew larger, the accuracy deteriorated, likely due to weaker
> overall evidence rather than decision ambiguity.]{.insertion
> author="Ben Remez" date="2025-08-26T01:07:00Z"}[]{.paragraph-insertion
> author="Ben Remez" date="2025-08-26T01:07:00Z"}
>
> [This finding highlights a potential limitation of threshold-based
> heuristics in pipeline orchestration. Future work may consider
> alternative strategies, such as dynamic confidence calibration or
> probabilistic uncertainty estimation, to more effectively]{.insertion
> author="Ben Remez" date="2025-08-26T01:07:00Z"} [determine when
> additional reasoning steps, like Deep Analysis,]{.insertion
> author="Alexander Apartsin" date="2025-08-27T11:19:00Z"} [should be
> applied.]{.insertion author="Ben Remez" date="2025-08-26T01:07:00Z"}
