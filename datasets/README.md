# Downloaded Datasets

Data files are NOT committed to git due to size. Follow download instructions below.

## Dataset 1: Lancaster Sensorimotor Norms (Lynott et al., 2020)

### Overview
- **Source**: https://osf.io/7emr6/ (Open Science Framework)
- **Size**: 39,707 words, ~5 MB CSV
- **Format**: CSV with header row
- **Task**: Word-level sensory/motor ratings across 11 dimensions
- **License**: CC-BY

### Dimensions
- **6 Perceptual modalities**: Auditory, Gustatory, Haptic, Interoceptive, Olfactory, Visual
- **5 Action effectors**: Foot/leg, Hand/arm, Head, Mouth, Torso
- Each word rated 0-5 by native English speakers (N=3,500)
- Additional columns: standard deviations, max strength, dominant modality, exclusivity scores

### Download Instructions
```bash
mkdir -p datasets/lancaster_sensorimotor_norms
curl -L "https://osf.io/7emr6/download" -o datasets/lancaster_sensorimotor_norms/lancaster_sensorimotor_norms.csv
```

### Loading
```python
import pandas as pd
df = pd.read_csv("datasets/lancaster_sensorimotor_norms/lancaster_sensorimotor_norms.csv")
# Key columns: Word, Auditory.mean, Gustatory.mean, Haptic.mean, etc.
```

### Notes
- Used extensively in Hicke et al. (2025), Lee et al. (2025), Wu et al. (2026), Gupta et al. (2026)
- Central resource for measuring sensory content of text
- The IDF-weighted scoring method from Hicke et al. is recommended for measuring sensory language in stories

---

## Dataset 2: Brysbaert Concreteness Ratings (Brysbaert et al., 2014)

### Overview
- **Source**: https://osf.io/j7czp/ (Open Science Framework)
- **Size**: 39,954 words, ~1.5 MB CSV
- **Format**: CSV with header row
- **Task**: Word-level concreteness ratings (1-5 scale)
- **License**: CC-BY

### Download Instructions
```bash
mkdir -p datasets/brysbaert_concreteness
curl -L "https://osf.io/j7czp/download" -o datasets/brysbaert_concreteness/brysbaert_concreteness.csv
```

### Loading
```python
import pandas as pd
df = pd.read_csv("datasets/brysbaert_concreteness/brysbaert_concreteness.csv")
# Key columns: Word, Conc.M (mean concreteness 1-5), Conc.SD, Dom_Pos
```

### Notes
- Concreteness is closely tied to sensory grounding
- Used in Hicke et al. (2025) alongside Lancaster norms

---

## Dataset 3: WritingPrompts (Fan et al., 2018) / GPT-WritingPrompts (Huang et al., 2024)

### Overview
- **Source**: HuggingFace `euclaise/writingprompts`
- **Size**: 303K+ human stories + 206K GPT-3.5 stories
- **Format**: JSON/parquet with prompt-story pairs
- **Task**: Creative writing generation and analysis
- **License**: Research use

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("euclaise/writingprompts")
ds.save_to_disk("datasets/writing_prompts/full")
```

### Notes
- Extended by Hicke et al. (2025) with 18 additional model generations
- Key dataset for comparing sensory language between humans and LLMs
- Hicke's extended dataset available at their paper's release link

---

## Dataset 4: WiT - Wikipedia Image-Text (Srinivasan et al., 2021)

### Overview
- **Source**: HuggingFace `google/wit`
- **Size**: Large (37M+ image-text pairs; experiments use 1,024 subset)
- **Format**: Image-caption pairs
- **Task**: Cross-modal alignment evaluation

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("google/wit", split="train", streaming=True)
# Take first 1024 for experiments as in Wang et al. (2025)
```

### Notes
- Used in Wang et al. (2025) for measuring LLM-vision alignment
- Only need ~1,024 pairs for the experiments

---

## Dataset 5: AudioCaps (Kim et al., 2019)

### Overview
- **Source**: HuggingFace or AudioSet-derived
- **Size**: ~50K audio-caption pairs (975 used in experiments)
- **Format**: Audio-caption pairs
- **Task**: Cross-modal alignment evaluation (audio)

### Notes
- Used in Wang et al. (2025) for measuring LLM-audio alignment
- Requires downloading audio from YouTube (may have availability issues)
