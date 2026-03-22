# Literature Review: Simulated Sensory World

## Research Area Overview

This review examines whether large language models (LLMs) encode internal representations corresponding to human sensory modalities (sight, sound, smell, taste, touch), even though they are trained only on text. The hypothesis is that when sensory-related objects appear in text (e.g., flowers → smell, thunder → sound), LLMs activate latent sensory representations rather than merely producing associations post-hoc. Furthermore, the five classic senses may occupy similar representational subspaces, with non-traditional senses encoded nearby.

This question lies at the intersection of NLP, cognitive science, and philosophy of mind — touching on the symbol grounding problem and embodied cognition debates.

---

## Key Papers

### 1. Wang et al. (2025) — "Words That Make Language Models Perceive"
- **Authors**: Sophie L. Wang, Phillip Isola, Brian Cheung (MIT)
- **Source**: arXiv:2510.02425
- **Key Contribution**: Demonstrates that sensory prompts ("imagine seeing/hearing") can steer text-only LLM representations toward alignment with vision (DINOv2) and audio (BEATs) encoders.
- **Methodology**:
  - Introduces "generative representations" — averaged hidden states across autoregressive generation steps
  - Uses mutual-kNN alignment to compare LLM kernel structure with sensory encoder kernels
  - Tests Qwen3 models (0.6B–32B) on WiT and AudioCaps datasets
- **Key Findings**:
  - SEE cues increase vision alignment, HEAR cues increase audio alignment — modality-specific steering works
  - Alignment increases with generation length (more tokens = more sensory elaboration)
  - Larger models show stronger alignment and clearer modality separation
  - Sensory-word ablation drops alignment; random sensory words don't help — *scene-appropriate* detail matters
  - Visual prompting improves text-only VQA on MME benchmark (64.78 → 67.14)
- **Code**: github.com/sophicle/sensory
- **Relevance**: **Most directly relevant paper.** Provides methodology (generative representations + kernel alignment) and evidence that LLMs maintain latent sensory structure that can be controllably activated.

### 2. Hicke, Hamilton & Mimno (2025) — "The Zero Body Problem: Probing LLM Use of Sensory Language"
- **Authors**: Cornell University
- **Source**: arXiv:2504.06393
- **Key Contribution**: Large-scale comparison of sensory language use between 19 LLMs and humans across 12 sensory axes.
- **Methodology**:
  - Extends WritingPrompts dataset with 18,000 model-generated stories
  - Uses Lancaster Sensorimotor Norms + Brysbaert Concreteness to score texts on 12 axes
  - IDF-weighted scoring to measure sensory language density
  - Linear probes on BERT, RoBERTa, T5, GPT-2, Qwen to test internal sensory representation
- **Key Findings**:
  - All 19 models differ significantly from humans in sensory language use
  - Gemini uses MORE sensory language; most others use LESS
  - Linear probes show models CAN identify sensory language (R² up to 0.85 for concreteness)
  - RLHF training may discourage sensory language use — strong correlation with Anthropic RLHF dataset
  - "The better a model understands a sensory axis, the less likely it is to use it"
- **Datasets**: WritingPrompts, Lancaster Sensorimotor Norms, Brysbaert Concreteness
- **Relevance**: Provides key lexicons and methodology for measuring sensory content. The probing results directly support our hypothesis that LLMs have internal sensory representations.

### 3. Marjieh et al. (2023) — "Large language models predict human sensory judgments across six modalities"
- **Authors**: Raja Marjieh, Ilia Sucholutsky, Pol van Rijn, Nori Jacoby, Thomas L. Griffiths (Princeton/MPI)
- **Source**: arXiv:2302.01308
- **Key Contribution**: Shows GPT models can predict human pairwise similarity judgments across pitch, loudness, colors, consonants, taste, and timbre.
- **Methodology**:
  - Elicits pairwise similarity judgments from GPT-3/3.5/4 via prompt engineering
  - Correlates with human psychophysical data across six modalities
  - Uses MDS to recover known perceptual structures (color wheel, pitch spiral)
- **Key Findings**:
  - Significant correlations with human data across ALL six modalities
  - GPT-4 achieves human inter-rater reliability on pitch (r=.92) and consonants
  - GPT-4's multimodal training does NOT specifically help visual modality — improvements are across all modalities
  - MDS recovers color wheel, pitch spiral, consonant articulation space
  - GPT-4 replicates cross-linguistic differences in color naming (English vs Russian)
- **Relevance**: Strong evidence that LLMs encode perceptual structure from text alone. Provides methodology for probing sensory representations via similarity judgments.

### 4. Xu et al. (2025) — "Large language models without grounding recover non-sensorimotor but not sensorimotor features"
- **Authors**: Qihui Xu et al. (Nature Human Behaviour)
- **Key Contribution**: Systematic comparison of ~4,442 concepts between humans and LLMs across sensorimotor domains.
- **Key Findings**:
  - Similarity decreases from non-sensorimotor → sensory → motor domains
  - Visual training enhances alignment in visual dimensions
  - Highlights the "embodiment gap" — text-only models lag in sensory/motor domains
- **Relevance**: Establishes the baseline limitation our research aims to investigate further.

### 5. Lee & Lim (2024) — "Language Models Don't Learn the Physical Manifestation of Language"
- **Source**: arXiv:2402.11718
- **Key Contribution**: H-Test benchmark for visual-auditory properties of language.
- **Key Findings**: LLMs perform near random chance on physical manifestation tasks; chain-of-thought and scaling don't help
- **Relevance**: Provides a counterpoint — argues for fundamental limitations in text-only sensory understanding.

### 6. Ngo & Kim (2024) — "What Do Language Models Hear?"
- **Key Contribution**: Linear probes that retrieve audio representations from text LM representations.
- **Methodology**: Contrastive loss aligning language and audio model representations; tests generalization to unseen objects
- **Key Findings**: Above-chance probe generalization indicates text-only LMs encode grounded knowledge of sounds
- **Relevance**: Direct evidence of auditory representations in text-only LMs.

### 7. Lee et al. (2025) — "Exploring Multimodal Perception in LLMs Through Perceptual Strength Ratings"
- **Source**: arXiv:2503.06980
- **Key Contribution**: Evaluates 21 models on Lancaster Sensorimotor Norms ratings for 3,611 words.
- **Key Findings**: Top models achieve 85-90% accuracy, 0.58-0.65 correlations with humans; distributional factors show minimal impact
- **Relevance**: Comprehensive benchmark of LLM sensory perception capabilities.

### 8. Wu et al. (2026) — "How does fine-tuning improve sensorimotor representations in LLMs?"
- **Source**: arXiv:2603.03313
- **Key Contribution**: Uses RSA and dimension-specific metrics to show fine-tuning can bridge the embodiment gap.
- **Key Findings**: Sensorimotor improvements generalize across languages and related dimensions but are sensitive to learning objective
- **Relevance**: Shows representations can be steered toward embodied patterns.

### 9. Siedenburg & Saitis (2023) — "The language of sounds unheard"
- **Source**: arXiv:2304.07830
- **Key Contribution**: Tests ChatGPT's ratings of musical instrument sounds on 20 semantic scales.
- **Key Findings**: Partial correlation with human ratings; robust agreement on brightness and pitch height; comparable internal variability to humans
- **Relevance**: Evidence of auditory-semantic structure in LLMs.

### 10. Merullo et al. (2025) — "I Have No Mouth, and I Must Rhyme"
- **Source**: arXiv:2508.02527
- **Key Contribution**: Discovers internal phonetic representations in LLaMA, including a "phoneme mover head" and vowel chart similar to IPA.
- **Relevance**: Shows LLMs learn rich sensory-adjacent representations (phonemes) without any auditory grounding.

---

## Common Methodologies

1. **Probing classifiers / linear probes**: Train simple models on LLM hidden states to predict sensory properties (Hicke 2025, Ngo 2024, Merullo 2025)
2. **Representational Similarity Analysis (RSA)**: Compare LLM representation geometry with human/sensory encoder geometry (Wu 2026, Jones 2024)
3. **Mutual-kNN alignment**: Compare kernel structures between LLM and sensory encoders (Wang 2025)
4. **Psychophysical similarity judgments**: Elicit pairwise similarity ratings via prompts (Marjieh 2023)
5. **Lexicon-based scoring**: Use sensorimotor norms to measure sensory content in generated text (Hicke 2025)

## Standard Baselines
- **DINOv2** (vision encoder) and **BEATs** (audio encoder) for cross-modal alignment (Wang 2025)
- **Lancaster Sensorimotor Norms** as ground truth for sensory ratings
- Human inter-rater reliability as performance ceiling
- Random/shuffled controls for probing experiments

## Evaluation Metrics
- **Mutual-kNN alignment score**: Fraction of shared k-nearest neighbors between two kernel matrices
- **Pearson/Spearman correlation**: Between model and human sensory ratings
- **R² of linear probes**: How well LLM hidden states predict sensory properties
- **RSA correlation**: Representational geometry similarity
- **Classification accuracy**: On sensory property tasks

## Datasets in the Literature
- **Lancaster Sensorimotor Norms** (Lynott et al., 2020): Used in Hicke 2025, Lee 2025, Wu 2026, Gupta 2026
- **Brysbaert Concreteness** (Brysbaert et al., 2014): Used in Hicke 2025
- **WritingPrompts** (Fan et al., 2018): Used in Hicke 2025 for story generation
- **WiT** (Srinivasan et al., 2021): Used in Wang 2025 for vision alignment
- **AudioCaps** (Kim et al., 2019): Used in Wang 2025 for audio alignment
- **Glasgow Norms** (Scott et al., 2019): Used in Xu 2025

## Gaps and Opportunities

1. **Smell and taste are understudied**: Most work focuses on vision and audio. Our hypothesis about smell (flowers → olfactory) is novel and directly testable.
2. **Story/narrative context**: The hypothesis that sensory representations activate during story comprehension (not just explicit prompting) is largely untested.
3. **Cross-sensory subspace structure**: Whether the five senses occupy similar subspaces and whether non-traditional senses (proprioception, interoception) are encoded nearby is unexplored.
4. **Layer-wise analysis**: Where in the network sensory representations emerge and whether they follow a consistent pattern across senses.
5. **Implicit vs. explicit activation**: Wang 2025 uses explicit cues; testing whether implicit sensory objects (flowers, garbage) activate representations without explicit prompting is the core of our hypothesis.

## Recommendations for Our Experiment

- **Primary dataset**: Lancaster Sensorimotor Norms (comprehensive sensory word ratings)
- **Methodology**: Combination of (1) probing hidden states for sensory activation when processing sensory-laden text, and (2) representational similarity analysis comparing sensory subspaces
- **Models**: Qwen3 family (following Wang 2025) or LLaMA family (open weights, well-studied)
- **Key baselines**: Random probes, shuffled text controls, non-sensory word controls
- **Metrics**: Linear probe accuracy, RSA correlation, mutual-kNN alignment
- **Code to reuse**: Wang 2025's generative representation extraction (github.com/sophicle/sensory)
