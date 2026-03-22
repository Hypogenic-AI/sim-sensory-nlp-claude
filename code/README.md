# Cloned Repositories

## Repo 1: sensory-prompting (Wang et al. 2025)
- **URL**: https://github.com/sophicle/sensory
- **Purpose**: Code for "Words That Make Language Models Perceive" — extracting generative representations from LLMs and measuring cross-modal alignment with vision/audio encoders.
- **Location**: code/sensory-prompting/
- **Key files**:
  - `run_embed.py` — Main embedding pipeline (Qwen3 + DINOv2/BEATs)
  - `run_align.py` — Compute mutual-kNN alignment scores
  - `src/models/` — Model loading and embedding extraction
  - `src/datasets/` — Dataset loaders (WiT, AudioCaps)
  - `src/utils/align.py` — Alignment computation
  - `environment.yml` — Conda environment spec
- **Dependencies**: PyTorch, transformers, DINOv2, BEATs
- **How to use for our research**:
  - Adapt the generative representation extraction for our sensory probing experiments
  - The mutual-kNN alignment metric can be reused directly
  - The prompt template structure (SEE/HEAR cues) can be extended to SMELL/TASTE/TOUCH cues
