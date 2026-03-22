"""
Extract hidden states from an LLM for sensory-laden stimuli.

Loads a pretrained model, processes stimulus sentences, and extracts
hidden-state vectors at the target word position across all layers.
"""

import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use Qwen2.5-7B following Wang 2025 methodology (Qwen family)
MODEL_NAME = "Qwen/Qwen2.5-7B"


def load_model(model_name=MODEL_NAME):
    """Load model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    print(f"  Model loaded on {DEVICE}")
    print(f"  Num layers: {model.config.num_hidden_layers}")
    print(f"  Hidden dim: {model.config.hidden_size}")
    return model, tokenizer


def find_target_word_positions(tokenizer, sentence, word):
    """
    Find the token positions corresponding to the target word in the sentence.
    Returns list of token indices for the word.
    """
    # Tokenize full sentence
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Find word tokens by tokenizing word alone and searching for match
    word_tokens = tokenizer.tokenize(word)

    # Search for contiguous match in sentence tokens
    positions = []
    for i in range(len(tokens) - len(word_tokens) + 1):
        # Check if tokens match (handling BPE prefixes)
        match = True
        for j, wt in enumerate(word_tokens):
            sent_tok = tokens[i + j].lower().replace("Ġ", "").replace("▁", "").replace("##", "")
            word_tok = wt.lower().replace("Ġ", "").replace("▁", "").replace("##", "")
            if sent_tok != word_tok:
                match = False
                break
        if match:
            positions = list(range(i, i + len(word_tokens)))
            break

    # Fallback: search for word substring in tokens
    if not positions:
        word_lower = word.lower()
        for i, tok in enumerate(tokens):
            cleaned = tok.lower().replace("Ġ", "").replace("▁", "").replace("##", "")
            if cleaned == word_lower or word_lower.startswith(cleaned):
                positions.append(i)
                # Check subsequent tokens
                remaining = word_lower[len(cleaned):]
                j = i + 1
                while remaining and j < len(tokens):
                    next_cleaned = tokens[j].lower().replace("Ġ", "").replace("▁", "").replace("##", "")
                    if remaining.startswith(next_cleaned):
                        positions.append(j)
                        remaining = remaining[len(next_cleaned):]
                        j += 1
                    else:
                        break
                if not remaining:
                    break
                positions = []

    return positions


def extract_hidden_states(model, tokenizer, stimuli, batch_size=8):
    """
    Extract hidden states at target word positions for all stimuli.

    Returns dict mapping stimulus index to:
    - hidden_states: (num_layers, hidden_dim) mean-pooled over word tokens
    - metadata: word, sentence, condition, sense
    """
    results = []
    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer

    for i in range(0, len(stimuli), batch_size):
        batch = stimuli[i:i+batch_size]
        sentences = [s["sentence"] for s in batch]

        # Tokenize batch
        inputs = tokenizer(sentences, return_tensors="pt", padding=True,
                          truncation=True, max_length=128)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # outputs.hidden_states: tuple of (batch, seq_len, hidden_dim) per layer
        hidden_states = outputs.hidden_states  # tuple of num_layers tensors

        for j, stim in enumerate(batch):
            # Find target word positions in this specific sentence
            positions = find_target_word_positions(tokenizer, stim["sentence"], stim["word"])

            if not positions:
                # Fallback: use last non-padding token
                attn_mask = inputs["attention_mask"][j]
                last_pos = attn_mask.sum().item() - 1
                positions = [last_pos]

            # Extract and mean-pool hidden states at word positions
            layer_vecs = []
            for layer_idx in range(num_layers):
                hs = hidden_states[layer_idx][j]  # (seq_len, hidden_dim)
                word_hs = hs[positions].mean(dim=0)  # (hidden_dim,)
                layer_vecs.append(word_hs.cpu().float().numpy())

            results.append({
                "hidden_states": np.stack(layer_vecs),  # (num_layers, hidden_dim)
                "word": stim["word"],
                "sentence": stim["sentence"],
                "condition": stim["condition"],
                "dominant_sense": stim["dominant_sense"],
                "sense_scores": stim["sense_scores"],
                "word_positions": positions,
            })

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {i + len(batch)}/{len(stimuli)} stimuli")

    return results


def main():
    # Load stimuli
    with open("results/stimuli.json") as f:
        stimuli = json.load(f)
    print(f"Loaded {len(stimuli)} stimuli")

    # Load model
    model, tokenizer = load_model()

    # Extract hidden states
    print("\nExtracting hidden states...")
    results = extract_hidden_states(model, tokenizer, stimuli)
    print(f"  Extracted {len(results)} hidden state vectors")

    # Save results
    # Save hidden states as numpy arrays (separate from metadata)
    hidden_states_array = np.array([r["hidden_states"] for r in results])
    metadata = [{k: v for k, v in r.items() if k != "hidden_states"} for r in results]

    os.makedirs("results", exist_ok=True)
    np.save("results/hidden_states.npy", hidden_states_array)
    with open("results/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Hidden states shape: {hidden_states_array.shape}")
    print(f"  Saved to results/hidden_states.npy and results/metadata.json")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    main()
