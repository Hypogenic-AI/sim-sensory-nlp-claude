# Research Plan: Simulated Sensory World

## Motivation & Novelty Assessment

### Why This Research Matters
When humans read "the garden was full of roses," we involuntarily activate olfactory imagery — we "smell" the roses even without being asked. If LLMs similarly activate sensory representations during narrative comprehension (not just when explicitly prompted about smell), it would fundamentally change our understanding of what these models learn from text alone. This has implications for AI safety (do models have proto-experiences?), cognitive science (is embodied grounding necessary for sensory concepts?), and practical NLP (can we leverage implicit sensory representations for better multimodal alignment?).

### Gap in Existing Work
- Wang et al. (2025) showed explicit sensory cues ("imagine seeing") steer LLM representations, but did not test **implicit** activation from narrative objects.
- Hicke et al. (2025) measured sensory language usage and probed for sensory classification ability, but didn't test whether sensory representations activate during story comprehension.
- No prior work systematically compares **all five senses + non-traditional senses** in terms of subspace geometry within a single model.
- Smell and taste are drastically understudied relative to vision and audition.

### Our Novel Contribution
1. Test whether sensory representations activate **implicitly** when LLMs process sensory-laden objects in narrative context (without explicit sensory prompts).
2. Compare implicit vs. explicit activation to determine if sensory representations are "always on" or only produced post-hoc.
3. Map the subspace geometry of all five classic senses plus non-traditional senses (interoception, proprioception).
4. Use Lancaster Sensorimotor Norms as ground truth to select stimuli and validate findings.

### Experiment Justification
- **Experiment 1 (Linear Probing)**: Can we decode which sensory modality is implicitly activated from hidden states when the model processes sensory-laden words in context? This directly tests the core hypothesis.
- **Experiment 2 (Implicit vs. Explicit)**: Compare hidden states when processing "the roses bloomed" (implicit smell) vs. "the roses smelled sweet" (explicit smell) vs. "describe the smell of roses" (prompted). Tests whether activation is spontaneous or post-hoc.
- **Experiment 3 (Subspace Geometry)**: Use PCA and cosine similarity to map how the five senses relate in representation space, and whether non-traditional senses cluster nearby.

## Research Question
Do LLMs spontaneously activate sensory-specific internal representations when processing objects with strong sensory associations (e.g., flowers → smell, thunder → sound), or are these associations only produced post-hoc when the model is explicitly asked about a sense? Do the five classic senses occupy a shared subspace, and are non-traditional senses encoded nearby?

## Hypothesis Decomposition
- H1: Linear probes can decode the dominant sensory modality from LLM hidden states when processing sensory-laden words in narrative context (above chance).
- H2: Sensory activation occurs implicitly (from object mention alone), not only when explicitly prompted about the relevant sense.
- H3: The five classic senses (visual, auditory, olfactory, gustatory, haptic) occupy geometrically related subspaces (high inter-sense cosine similarity of probe directions).
- H4: Non-traditional senses (interoception, proprioception) are encoded in nearby but distinct subspaces.

## Proposed Methodology

### Approach
Extract hidden-state representations from an open-weight LLM (LLaMA-3.1-8B or Qwen2.5-7B) processing carefully constructed stimuli. Use Lancaster Sensorimotor Norms to select high-association words for each modality. Train linear probes on hidden states to classify dominant sense. Compare implicit vs. explicit conditions. Analyze subspace geometry via PCA and cosine similarity.

### Experimental Steps
1. Load Lancaster Sensorimotor Norms; select top-N words per sensory modality (high dominant-sense score, low cross-modal scores).
2. Construct stimulus sentences in three conditions: (a) implicit narrative ("The garden was full of [roses]"), (b) explicit sensory ("The [roses] smelled sweet"), (c) control (neutral context with the same word).
3. Load LLM and extract hidden states at the target word position across all layers.
4. Train linear probes (logistic regression) to classify dominant sensory modality from hidden states.
5. Compare probe accuracy across conditions and layers.
6. Extract learned probe weight vectors as "sensory directions" and compute pairwise cosine similarities.
7. Visualize sensory subspace with PCA/t-SNE.
8. Extend to non-traditional senses (interoceptive words from Lancaster norms).

### Baselines
- Random baseline (1/6 for 6 senses)
- Shuffled labels (permutation test)
- Bag-of-words baseline (does a simple word embedding probe work as well?)

### Evaluation Metrics
- Linear probe accuracy (per-sense and overall)
- Probe R² for continuous sensory strength prediction
- Cosine similarity between sensory direction vectors
- PCA variance explained by top components
- Statistical significance via permutation tests

### Statistical Analysis Plan
- Permutation tests (1000 permutations) for probe accuracy significance
- Bootstrap confidence intervals for accuracy and cosine similarities
- Bonferroni correction for multiple comparisons across senses

## Expected Outcomes
- H1 supported: Probe accuracy significantly above chance (>30% for 6-way, chance=16.7%)
- H2 supported: Probe accuracy in implicit condition comparable to explicit condition
- H3 supported: Cosine similarity between sense directions higher than random directions
- H4 supported: Non-traditional senses have moderate cosine similarity with classic senses

## Timeline and Milestones
1. Data preparation & stimulus construction: 15 min
2. Model loading & hidden state extraction: 30 min
3. Linear probing experiments: 30 min
4. Implicit vs. explicit comparison: 20 min
5. Subspace geometry analysis: 20 min
6. Visualization & documentation: 30 min

## Potential Challenges
- Model loading time/memory: Use 8B model on single GPU with float16
- Word tokenization: Multi-token words may require aggregation strategy
- Confounds: Word frequency, sentence position, semantic similarity may correlate with sensory modality
- Small effect sizes: Sensory information may be subtle in hidden states

## Success Criteria
- At least one experiment produces statistically significant results
- Clear visualizations showing sensory subspace structure
- Meaningful comparison between implicit and explicit conditions
- Complete, reproducible analysis pipeline
