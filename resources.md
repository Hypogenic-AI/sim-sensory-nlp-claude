# Resources Catalog

## Summary
This document catalogs all resources gathered for the "Simulated Sensory World" research project investigating whether LLMs activate internal sensory representations when processing sensory-laden text.

## Papers
Total papers downloaded: 20

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Words That Make Language Models Perceive | Wang, Isola, Cheung | 2025 | papers/wang2025_words_make_llms_perceive.pdf | Sensory prompting steers LLM representations; CODE AVAILABLE |
| 2 | The Zero Body Problem | Hicke, Hamilton, Mimno | 2025 | papers/hicke2025_zero_body_problem.pdf | 19 LLMs vs human sensory language; linear probes; RLHF effects |
| 3 | LLMs predict human sensory judgments across six modalities | Marjieh et al. | 2023 | papers/marjieh2023_llms_predict_sensory_judgments.pdf | GPT recovers color wheel, pitch spiral from text |
| 4 | LLMs without grounding recover non-sensorimotor features | Xu et al. | 2025 | papers/xu2023_conceptual_representation_embodiment.pdf | Embodiment gap: text-only LLMs lag in sensory/motor domains |
| 5 | Language Models Don't Learn Physical Manifestation | Lee & Lim | 2024 | papers/lee2024_lms_dont_learn_physical.pdf | H-Test: LLMs near random on physical properties |
| 6 | Multimodal LMs Show Embodied Simulation | Jones & Trott | 2024 | papers/jones2024_embodied_simulation.pdf | MLLMs sensitive to implicit visual features |
| 7 | Perceptual Structure Without Grounding (Color) | Abdou et al. | 2021 | papers/abdou2021_perceptual_structure_color.pdf | Color embeddings align with perceptual space |
| 8 | Experience Grounds Language | Bisk et al. | 2020 | papers/bisk2020_experience_grounds_language.pdf | Foundational grounding paper |
| 9 | Does Thought Require Sensory Grounding? | Chalmers | 2024 | papers/chalmers2024_thought_sensory_grounding.pdf | Philosophical analysis of pure thinkers |
| 10 | Enriching LMs with Sensorimotor Norms | Kennington | 2021 | papers/kennington2021_enriching_lm_sensorimotor.pdf | Adding Lancaster norms to ELECTRA improves tasks |
| 11 | Contextualised Semantic Features from BERT | Derby et al. | 2020 | papers/derby2020_contextualised_semantic_features_bert.pdf | Extracting sensory features from BERT |
| 12 | Multimodal Perception via Perceptual Strength | Lee et al. | 2025 | papers/lee2025_multimodal_perception_perceptual_strength.pdf | 21 models on Lancaster norms; 85-90% accuracy |
| 13 | Fine-tuning Sensorimotor Representations | Wu et al. | 2026 | papers/wu2026_finetuning_sensorimotor.pdf | RSA shows fine-tuning bridges embodiment gap |
| 14 | Language of Sounds Unheard | Siedenburg & Saitis | 2023 | papers/siedenburg2023_language_sounds_unheard.pdf | ChatGPT on musical timbre semantics |
| 15 | Phonetic Representations in LLaMA | Merullo et al. | 2025 | papers/merullo2025_phonetic_representations_llama.pdf | Internal phoneme representations; vowel chart |
| 16 | Visual Grounding and Embodied Knowledge | Yang et al. | 2025 | papers/yang2025_visual_grounding_embodied.pdf | VLMs don't outperform text-only on sensory tasks |
| 17 | Words that make SENSE | Gupta et al. | 2026 | papers/gupta2026_words_that_make_sense.pdf | Sensorimotor norms from embeddings |
| 18 | Can Transformers Smell Like Humans? | Le et al. | 2024 | papers/le2024_can_transformers_smell.pdf | Olfactory representations in transformers |
| 19 | Sniff AI | Kang et al. | 2024 | papers/kang2024_sniff_ai.pdf | LLM alignment with human smell experiences |
| 20 | Does Thought Require Sensory Grounding? | Chalmers | 2024 | papers/chalmers2024_thought_sensory_grounding.pdf | Philosophical framework |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 2 (+ 1 sampled, 2 documented for download)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Lancaster Sensorimotor Norms | OSF (Lynott et al., 2020) | 39,707 words | Sensory ratings (11 dims) | datasets/lancaster_sensorimotor_norms/ | PRIMARY DATASET |
| Brysbaert Concreteness | OSF (Brysbaert et al., 2014) | 39,954 words | Concreteness ratings | datasets/brysbaert_concreteness/ | Supplementary |
| WritingPrompts | HuggingFace | 303K+ stories | Creative writing | datasets/writing_prompts/ (sample) | For sensory language analysis |
| WiT | Google/HuggingFace | 37M+ pairs | Image-text alignment | (download as needed) | For cross-modal experiments |
| AudioCaps | AudioSet-derived | 50K pairs | Audio-text alignment | (download as needed) | For cross-modal experiments |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 1

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| sensory-prompting | github.com/sophicle/sensory | Generative representations + cross-modal alignment | code/sensory-prompting/ | Key codebase for experiments |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Used paper-finder service with 3 queries: "sensory representations in LLMs", "probing internal representations senses smell taste LLM", "embodied cognition grounding language models sensorimotor norms"
- Focused on papers from 2020-2026 with direct relevance to sensory representations in text-only LLMs
- Prioritized papers with code/data availability and those using probing methodologies

### Selection Criteria
- Direct relevance to the hypothesis (LLMs activate sensory representations)
- Methodological utility (probing techniques, alignment metrics, sensory lexicons)
- Recency (prioritized 2023-2026 work)
- Code availability for reproducibility

### Challenges Encountered
- Semantic Scholar rate limiting required multiple download attempts
- Some papers (SENSE-LM, Representations of Smells) not available on arXiv
- Ngo & Kim 2024 "What Do Language Models Hear?" — correct arXiv ID could not be confirmed; the paper is known from Semantic Scholar but the PDF was not downloadable

### Gaps
- No single existing dataset pairs stories with sensory activation labels
- Olfactory and gustatory modalities are understudied compared to vision/audio
- No existing work tests implicit sensory activation during narrative comprehension (vs. explicit prompting)

## Recommendations for Experiment Design

1. **Primary dataset**: Lancaster Sensorimotor Norms — comprehensive ground truth for word-level sensory associations across all 5+ senses
2. **Experimental approach**:
   - Extract hidden states from LLMs processing sensory-laden text (stories mentioning flowers, garbage, music, etc.)
   - Train linear probes to predict which sensory modality is implicitly activated
   - Use RSA to compare sensory subspace geometry across the five senses
   - Test whether sensory representations cluster by modality in a shared subspace
3. **Baseline methods**: Random probes, shuffled text, non-sensory control words
4. **Evaluation metrics**: Linear probe accuracy/R², RSA correlation, mutual-kNN alignment
5. **Code to adapt**: Wang 2025's generative representation extraction pipeline (code/sensory-prompting/) — extend prompt templates to smell/taste/touch cues and adapt alignment measurement
6. **Models**: Qwen3 family (0.6B-32B) for consistency with Wang 2025, or LLaMA family for broader community adoption
