"""
Stimulus preparation for sensory probing experiments.

Uses Lancaster Sensorimotor Norms to select words strongly associated with each
sensory modality and constructs sentences in implicit, explicit, and control conditions.
"""

import pandas as pd
import numpy as np
import json
import os

SENSES = ["Auditory", "Gustatory", "Haptic", "Interoceptive", "Olfactory", "Visual"]
CLASSIC_SENSES = ["Auditory", "Gustatory", "Haptic", "Olfactory", "Visual"]
SENSE_LABELS = {s: i for i, s in enumerate(SENSES)}

# Manually curated high-quality sensory words (nouns/concrete objects)
# Selected from Lancaster norms for high dominance + natural sentence fit
CURATED_WORDS = {
    "Auditory": [
        "thunder", "siren", "whistle", "bell", "drum", "alarm", "horn",
        "buzzer", "chime", "gong", "gunshot", "scream", "clap", "echo",
        "hiss", "roar", "howl", "bang", "chirp", "beep", "ringtone",
        "explosion", "murmur", "crackle", "squeak",
    ],
    "Gustatory": [
        "chocolate", "lemon", "vinegar", "honey", "cinnamon", "pepper",
        "garlic", "ginger", "mint", "mustard", "vanilla", "curry",
        "wasabi", "licorice", "syrup", "caramel", "molasses", "chili",
        "nutmeg", "paprika", "horseradish", "turmeric", "clove",
        "saffron", "cardamom",
    ],
    "Haptic": [
        "sandpaper", "silk", "velvet", "gravel", "ice", "wool", "feather",
        "thorn", "needle", "blade", "sponge", "cotton", "rope", "chain",
        "brick", "marble", "clay", "mud", "foam", "steel", "glass",
        "leather", "rubber", "wax", "cork",
    ],
    "Interoceptive": [
        "headache", "nausea", "dizziness", "hunger", "thirst", "fatigue",
        "heartbeat", "cramp", "itch", "fever", "chill", "pain",
        "anxiety", "exhaustion", "drowsiness", "soreness", "numbness",
        "tension", "shivers", "palpitation", "ache", "breathlessness",
        "vertigo", "faintness", "tingling",
    ],
    "Olfactory": [
        "perfume", "smoke", "garbage", "roses", "incense", "coffee",
        "skunk", "lavender", "gasoline", "cedar", "bacon", "jasmine",
        "ammonia", "cinnamon", "campfire", "mildew", "eucalyptus",
        "chlorine", "pine", "sulfur", "sage", "onion", "basil",
        "oregano", "mint",
    ],
    "Visual": [
        "rainbow", "sunset", "lightning", "fireworks", "diamond", "mirror",
        "crystal", "flame", "spotlight", "lantern", "aurora", "eclipse",
        "prism", "sparkle", "shadow", "silhouette", "beacon", "glitter",
        "neon", "kaleidoscope", "star", "moon", "candle", "laser", "torch",
    ],
}

# Templates where the word slot is naturally a noun
IMPLICIT_TEMPLATES = {
    "Auditory": [
        "The {word} filled the air as they walked through the city.",
        "A distant {word} broke the silence of the night.",
        "They noticed the {word} coming from down the street.",
    ],
    "Gustatory": [
        "There was {word} on the kitchen counter.",
        "The {word} had been sitting on the table all morning.",
        "Someone had left {word} next to the stove.",
    ],
    "Haptic": [
        "The {word} sat on the shelf beside the door.",
        "There was {word} scattered across the workbench.",
        "The {word} lay in a pile on the floor.",
    ],
    "Interoceptive": [
        "A wave of {word} came over them suddenly.",
        "The {word} grew stronger as the hours passed.",
        "They couldn't shake the {word} all afternoon.",
    ],
    "Olfactory": [
        "The {word} drifted through the open window.",
        "There was {word} everywhere in the old house.",
        "The {word} lingered in the hallway near the entrance.",
    ],
    "Visual": [
        "The {word} appeared just above the horizon.",
        "A {word} caught their attention from across the field.",
        "The {word} was visible even from miles away.",
    ],
}

EXPLICIT_TEMPLATES = {
    "Auditory": [
        "The {word} sounded incredibly loud and startling.",
        "You could hear the {word} echoing from far away.",
        "The sound of the {word} was completely unmistakable.",
    ],
    "Gustatory": [
        "The {word} tasted absolutely wonderful and rich.",
        "The flavor of the {word} was complex and unforgettable.",
        "The taste of the {word} lingered delightfully on their tongue.",
    ],
    "Haptic": [
        "The {word} felt rough and uneven against their fingers.",
        "Touching the {word} produced a surprising sensation.",
        "The texture of the {word} was unlike anything they had felt.",
    ],
    "Interoceptive": [
        "The {word} made their whole body feel strange.",
        "They could feel the {word} deep in their core.",
        "The sensation of {word} spread through their body.",
    ],
    "Olfactory": [
        "The {word} smelled incredibly strong and pungent.",
        "The scent of the {word} was impossible to ignore.",
        "You could smell the {word} from across the entire room.",
    ],
    "Visual": [
        "The {word} looked stunningly beautiful in the light.",
        "The sight of the {word} was absolutely breathtaking.",
        "The {word} appeared bright and vivid against the sky.",
    ],
}

NEUTRAL_TEMPLATES = [
    "The {word} was mentioned briefly in the report.",
    "They discussed the {word} at the afternoon meeting.",
    "The {word} was listed in the official document.",
]


def load_lancaster_norms(path="datasets/lancaster_sensorimotor_norms/lancaster_sensorimotor_norms.csv"):
    """Load Lancaster Sensorimotor Norms dataset."""
    return pd.read_csv(path)


def get_word_scores(df, words):
    """Look up Lancaster norms scores for a list of words."""
    df_lower = df.copy()
    df_lower["Word_lower"] = df_lower["Word"].str.lower()
    results = []
    for word in words:
        match = df_lower[df_lower["Word_lower"] == word.lower()]
        if len(match) > 0:
            row = match.iloc[0]
            scores = {s: float(row[f"{s}.mean"]) for s in SENSES}
            results.append({"word": word, "scores": scores, "in_norms": True})
        else:
            results.append({"word": word, "scores": {s: 0.0 for s in SENSES}, "in_norms": False})
    return results


def construct_stimuli(df):
    """Construct stimulus sentences in three conditions."""
    stimuli = []
    word_info = {}

    for sense in SENSES:
        words = CURATED_WORDS[sense]
        scores_list = get_word_scores(df, words)

        for ws in scores_list:
            word = ws["word"]
            scores = ws["scores"]
            word_info[word] = {"sense": sense, "scores": scores, "in_norms": ws["in_norms"]}

            # Implicit condition
            templates = IMPLICIT_TEMPLATES[sense]
            template = templates[hash(word) % len(templates)]
            stimuli.append({
                "word": word,
                "sentence": template.format(word=word),
                "condition": "implicit",
                "dominant_sense": sense,
                "sense_scores": scores,
                "in_norms": ws["in_norms"],
            })

            # Explicit condition
            templates = EXPLICIT_TEMPLATES[sense]
            template = templates[hash(word) % len(templates)]
            stimuli.append({
                "word": word,
                "sentence": template.format(word=word),
                "condition": "explicit",
                "dominant_sense": sense,
                "sense_scores": scores,
                "in_norms": ws["in_norms"],
            })

            # Control condition
            template = NEUTRAL_TEMPLATES[hash(word) % len(NEUTRAL_TEMPLATES)]
            stimuli.append({
                "word": word,
                "sentence": template.format(word=word),
                "condition": "control",
                "dominant_sense": sense,
                "sense_scores": scores,
                "in_norms": ws["in_norms"],
            })

    return stimuli


def main():
    print("Loading Lancaster Sensorimotor Norms...")
    df = load_lancaster_norms()
    print(f"  Loaded {len(df)} words")

    print("\nChecking curated words against norms...")
    for sense in SENSES:
        words = CURATED_WORDS[sense]
        scores = get_word_scores(df, words)
        in_norms = sum(1 for s in scores if s["in_norms"])
        print(f"  {sense}: {in_norms}/{len(words)} in norms")
        # Show top words with their dominant-sense score
        scored = [(s["word"], s["scores"].get(sense, 0)) for s in scores if s["in_norms"]]
        scored.sort(key=lambda x: -x[1])
        if scored:
            print(f"    Top: {', '.join(f'{w}({s:.1f})' for w, s in scored[:5])}")

    print("\nConstructing stimuli...")
    stimuli = construct_stimuli(df)
    print(f"  Total stimuli: {len(stimuli)}")

    os.makedirs("results", exist_ok=True)
    with open("results/stimuli.json", "w") as f:
        json.dump(stimuli, f, indent=2)
    print("Saved to results/stimuli.json")

    # Print examples
    print("\n=== Example Stimuli ===")
    for sense in SENSES:
        sense_stim = [s for s in stimuli if s["dominant_sense"] == sense]
        print(f"\n{sense}:")
        for cond in ["implicit", "explicit", "control"]:
            example = [s for s in sense_stim if s["condition"] == cond][0]
            print(f"  [{cond:8s}] {example['sentence']}")

    return stimuli


if __name__ == "__main__":
    main()
