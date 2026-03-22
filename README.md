# Simulated Sensory World

**Do LLMs spontaneously activate sensory representations when processing sensory-laden text?**

This project investigates whether large language models encode and activate internal sensory representations (smell, sight, sound, taste, touch, interoception) when processing objects with strong sensory associations — even without explicit sensory language.

## Key Findings

- **Sensory modalities are linearly decodable** from LLM hidden states at 87-90% accuracy (chance = 16.7%, p = 0.005)
- **Implicit activation is real**: "The perfume drifted through the window" activates smell representations almost as strongly as "The perfume smelled sweet" — this is NOT just a post-hoc effect
- **All six senses form distinct directions** in representation space with structured negative correlations (mean cosine = -0.20 vs 0.00 for random)
- **Interoception (non-traditional sense)** is encoded in the same geometric framework as the five classic senses
- **Olfactory representations are the strongest** in continuous prediction (R² = 0.57, Spearman ρ = 0.73)
- **Sensory encoding emerges by layer 1** and is present even in word embeddings for auditory concepts

## How to Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add torch transformers accelerate scikit-learn matplotlib seaborn pandas numpy scipy

# Run pipeline
python src/prepare_stimuli.py      # Construct sensory stimuli from Lancaster norms
python src/extract_hidden_states.py # Extract hidden states from Qwen2.5-7B (requires GPU)
python src/run_experiments.py       # Run all 5 experiments and generate plots
```

Requires: NVIDIA GPU with 16GB+ VRAM (tested on RTX A6000).

## File Structure

```
.
├── REPORT.md                          # Full research report with results
├── README.md                          # This file
├── planning.md                        # Research plan and motivation
├── src/
│   ├── prepare_stimuli.py            # Stimulus construction from Lancaster norms
│   ├── extract_hidden_states.py      # Hidden state extraction from Qwen2.5-7B
│   └── run_experiments.py            # Linear probing, geometry analysis, visualization
├── results/
│   ├── stimuli.json                  # Generated stimuli
│   ├── hidden_states.npy             # Extracted hidden states (450 x 29 x 3584)
│   ├── metadata.json                 # Stimulus metadata
│   ├── experiment_results.json       # All numerical results
│   └── plots/                        # 9 visualization plots
├── datasets/
│   ├── lancaster_sensorimotor_norms/ # Primary dataset (39,707 words)
│   └── brysbaert_concreteness/       # Supplementary concreteness ratings
├── papers/                           # 20 downloaded research papers
├── code/sensory-prompting/           # Wang et al. 2025 reference code
└── literature_review.md              # Comprehensive literature review
```

## Method

1. Select 150 words (25 per sense) from Lancaster Sensorimotor Norms with high sensory dominance
2. Construct sentences in 3 conditions: implicit (narrative), explicit (sensory verbs), control (neutral)
3. Extract hidden states from Qwen2.5-7B at target word positions across all 29 layers
4. Train linear probes (logistic regression with PCA to 100 dims) to classify dominant sensory modality
5. Analyze subspace geometry via cosine similarity of learned sensory direction vectors

See [REPORT.md](REPORT.md) for full details and analysis.
