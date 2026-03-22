"""
Main experiment script: linear probing, implicit vs explicit comparison,
and sensory subspace geometry analysis.

Experiments:
1. Linear probes to decode dominant sensory modality from hidden states
2. Compare probe accuracy across implicit/explicit/control conditions
3. Analyze sensory subspace geometry (cosine similarity, PCA)
4. Test non-traditional senses (interoception)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
from scipy.stats import permutation_test, spearmanr
import warnings
warnings.filterwarnings("ignore")

SENSES = ["Auditory", "Gustatory", "Haptic", "Interoceptive", "Olfactory", "Visual"]
CLASSIC_SENSES = ["Auditory", "Gustatory", "Haptic", "Olfactory", "Visual"]
SENSE_COLORS = {
    "Auditory": "#e41a1c",
    "Gustatory": "#ff7f00",
    "Haptic": "#4daf4a",
    "Interoceptive": "#984ea3",
    "Olfactory": "#a65628",
    "Visual": "#377eb8",
}

np.random.seed(42)

# Reduce dimensionality for faster probing
PCA_DIM = 100  # Reduce 3584 -> 100 dims (must be <= min(n_samples, n_features))


def reduce_dims(X, n_components=PCA_DIM):
    """PCA reduction for tractable probing."""
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X), pca


def load_data():
    """Load hidden states and metadata."""
    hidden_states = np.load("results/hidden_states.npy")  # (N, num_layers, hidden_dim)
    with open("results/metadata.json") as f:
        metadata = json.load(f)
    return hidden_states, metadata


# ============================================================================
# EXPERIMENT 1: Linear Probing per Layer
# ============================================================================

def experiment1_linear_probing(hidden_states, metadata, results_dir):
    """
    Train linear probes to decode dominant sensory modality from hidden states.
    Test across all layers to find where sensory information is encoded.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Linear Probing for Sensory Modality")
    print("="*70)

    # Prepare labels
    labels = np.array([SENSES.index(m["dominant_sense"]) for m in metadata])
    conditions = np.array([m["condition"] for m in metadata])
    num_layers = hidden_states.shape[1]

    # Run probing for each layer, each condition, and all combined
    all_results = {}

    for condition in ["implicit", "explicit", "control", "all"]:
        print(f"\n--- Condition: {condition} ---")
        if condition == "all":
            mask = np.ones(len(metadata), dtype=bool)
        else:
            mask = conditions == condition

        X_all = hidden_states[mask]
        y = labels[mask]
        n_samples = len(y)
        chance = 1.0 / len(SENSES)

        layer_accuracies = []
        layer_reports = []

        for layer in range(num_layers):
            X = X_all[:, layer, :]
            X_reduced, _ = reduce_dims(X)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_reduced)

            # 5-fold stratified CV
            clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                                     random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
            acc = accuracy_score(y, y_pred)
            layer_accuracies.append(acc)

            if layer == num_layers - 1:  # last layer
                report = classification_report(y, y_pred,
                    target_names=SENSES, output_dict=True)
                layer_reports.append(report)
                print(f"  Layer {layer}: accuracy={acc:.3f} (chance={chance:.3f})")
                print(classification_report(y, y_pred, target_names=SENSES))

        all_results[condition] = {
            "layer_accuracies": layer_accuracies,
            "n_samples": n_samples,
            "chance": chance,
        }

        best_layer = np.argmax(layer_accuracies)
        print(f"  Best layer: {best_layer}, accuracy: {layer_accuracies[best_layer]:.3f}")

    # Permutation test for significance at best layer
    print("\n--- Permutation Test (all conditions, best layer) ---")
    mask = np.ones(len(metadata), dtype=bool)
    X_all = hidden_states[mask]
    y = labels[mask]
    best_layer = np.argmax(all_results["all"]["layer_accuracies"])
    X = X_all[:, best_layer, :]
    X_reduced, _ = reduce_dims(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)

    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                             random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
    real_acc = accuracy_score(y, y_pred)

    # Permutation test (reduced for speed)
    n_perms = 200
    perm_accs = []
    for p in range(n_perms):
        y_perm = np.random.permutation(y)
        y_pred_perm = cross_val_predict(clf, X_scaled, y_perm, cv=cv)
        perm_accs.append(accuracy_score(y_perm, y_pred_perm))

    p_value = (np.sum(np.array(perm_accs) >= real_acc) + 1) / (n_perms + 1)
    print(f"  Real accuracy: {real_acc:.3f}")
    print(f"  Permutation mean: {np.mean(perm_accs):.3f} ± {np.std(perm_accs):.3f}")
    print(f"  p-value: {p_value:.4f}")
    all_results["permutation_test"] = {
        "real_accuracy": float(real_acc),
        "perm_mean": float(np.mean(perm_accs)),
        "perm_std": float(np.std(perm_accs)),
        "p_value": float(p_value),
        "n_perms": n_perms,
        "best_layer": int(best_layer),
    }

    # Confusion matrix at best layer (all conditions)
    X = X_all[:, best_layer, :]
    X_reduced, _ = reduce_dims(X)
    X_scaled = StandardScaler().fit_transform(X_reduced)
    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                             random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
    cm = confusion_matrix(y, y_pred)
    all_results["confusion_matrix"] = cm.tolist()

    # ---- PLOTS ----

    # 1. Layer-wise accuracy curves
    fig, ax = plt.subplots(figsize=(10, 6))
    for cond in ["implicit", "explicit", "control"]:
        accs = all_results[cond]["layer_accuracies"]
        ax.plot(range(len(accs)), accs, label=cond, linewidth=2)
    ax.axhline(y=1/len(SENSES), color="gray", linestyle="--", label="chance")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Probe Accuracy (5-fold CV)", fontsize=12)
    ax.set_title("Sensory Modality Decoding Accuracy Across Layers", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/layer_accuracy_curves.png", dpi=150)
    plt.close()

    # 2. Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=SENSES,
                yticklabels=SENSES, cmap="YlOrRd", ax=ax)
    ax.set_xlabel("Predicted Sense", fontsize=12)
    ax.set_ylabel("True Sense", fontsize=12)
    ax.set_title(f"Confusion Matrix (Layer {best_layer}, All Conditions)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/confusion_matrix.png", dpi=150)
    plt.close()

    # 3. Condition comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    conditions_list = ["implicit", "explicit", "control"]
    best_accs = [max(all_results[c]["layer_accuracies"]) for c in conditions_list]
    bars = ax.bar(conditions_list, best_accs, color=["#2196F3", "#4CAF50", "#9E9E9E"])
    ax.axhline(y=1/len(SENSES), color="red", linestyle="--", label="chance")
    ax.set_ylabel("Best Layer Accuracy", fontsize=12)
    ax.set_title("Peak Probe Accuracy by Condition", fontsize=14)
    for bar, acc in zip(bars, best_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", fontsize=11)
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/condition_comparison.png", dpi=150)
    plt.close()

    return all_results


# ============================================================================
# EXPERIMENT 2: Implicit vs Explicit Activation (Paired Analysis)
# ============================================================================

def experiment2_implicit_vs_explicit(hidden_states, metadata, results_dir):
    """
    Paired comparison: for the same word, how similar are implicit vs explicit
    hidden states? Do both conditions activate the same sensory subspace?
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Implicit vs. Explicit Sensory Activation")
    print("="*70)

    # Group by word
    word_groups = {}
    for i, m in enumerate(metadata):
        word = m["word"]
        if word not in word_groups:
            word_groups[word] = {}
        word_groups[word][m["condition"]] = i

    # For each word, compute cosine similarity between conditions at each layer
    num_layers = hidden_states.shape[1]
    similarities = {pair: [] for pair in
                    [("implicit", "explicit"), ("implicit", "control"), ("explicit", "control")]}

    for word, indices in word_groups.items():
        if not all(c in indices for c in ["implicit", "explicit", "control"]):
            continue
        for pair_name in similarities:
            c1, c2 = pair_name
            layer_sims = []
            for layer in range(num_layers):
                v1 = hidden_states[indices[c1], layer, :]
                v2 = hidden_states[indices[c2], layer, :]
                sim = 1 - cosine(v1, v2)
                layer_sims.append(sim)
            similarities[pair_name].append(layer_sims)

    results = {}
    for pair_name, sims_list in similarities.items():
        sims_array = np.array(sims_list)  # (n_words, num_layers)
        mean_sims = sims_array.mean(axis=0)
        std_sims = sims_array.std(axis=0)
        results[f"{pair_name[0]}_vs_{pair_name[1]}"] = {
            "mean_per_layer": mean_sims.tolist(),
            "std_per_layer": std_sims.tolist(),
            "overall_mean": float(mean_sims.mean()),
        }
        print(f"  {pair_name[0]} vs {pair_name[1]}: mean cosine sim = {mean_sims.mean():.4f}")

    # Plot: cosine similarity across layers for each pair
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3", "#FF9800", "#9E9E9E"]
    for (pair_name, sims_list), color in zip(similarities.items(), colors):
        sims_array = np.array(sims_list)
        mean_sims = sims_array.mean(axis=0)
        std_sims = sims_array.std(axis=0)
        label = f"{pair_name[0]} vs {pair_name[1]}"
        ax.plot(range(num_layers), mean_sims, label=label, color=color, linewidth=2)
        ax.fill_between(range(num_layers),
                        mean_sims - std_sims, mean_sims + std_sims,
                        alpha=0.15, color=color)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title("Hidden State Similarity: Implicit vs Explicit vs Control", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/implicit_vs_explicit_similarity.png", dpi=150)
    plt.close()

    # Statistical test: is implicit-explicit more similar than implicit-control?
    imp_exp = np.array(similarities[("implicit", "explicit")])  # (n_words, layers)
    imp_ctrl = np.array(similarities[("implicit", "control")])

    # Average across layers for each word, then paired test
    imp_exp_mean = imp_exp.mean(axis=1)  # (n_words,)
    imp_ctrl_mean = imp_ctrl.mean(axis=1)
    diff = imp_exp_mean - imp_ctrl_mean

    from scipy.stats import ttest_rel, wilcoxon
    t_stat, t_pval = ttest_rel(imp_exp_mean, imp_ctrl_mean)
    print(f"\n  Paired t-test (implicit-explicit vs implicit-control):")
    print(f"    mean diff = {diff.mean():.4f}, t = {t_stat:.3f}, p = {t_pval:.4e}")

    results["paired_test"] = {
        "mean_diff": float(diff.mean()),
        "t_stat": float(t_stat),
        "p_value": float(t_pval),
    }

    return results


# ============================================================================
# EXPERIMENT 3: Sensory Subspace Geometry
# ============================================================================

def experiment3_subspace_geometry(hidden_states, metadata, results_dir):
    """
    Analyze the geometry of sensory subspaces:
    - Extract probe weight vectors as "sensory directions"
    - Compute pairwise cosine similarity between sense directions
    - PCA visualization of sensory representations
    - Test whether non-traditional senses (interoception) cluster nearby
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Sensory Subspace Geometry")
    print("="*70)

    labels = np.array([SENSES.index(m["dominant_sense"]) for m in metadata])
    num_layers = hidden_states.shape[1]

    # Use best layer from experiment 1 (avoid recomputing)
    # Quick recompute on just a few layers to save time
    candidate_layers = list(range(0, num_layers, 2)) + [num_layers - 1]
    accs = {}
    for layer in candidate_layers:
        X = hidden_states[:, layer, :]
        X_reduced, _ = reduce_dims(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reduced)
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                                 random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(clf, X_scaled, labels, cv=cv)
        accs[layer] = accuracy_score(labels, y_pred)
    best_layer = max(accs, key=accs.get)
    print(f"  Using best layer: {best_layer} (acc={accs[best_layer]:.3f})")

    # Train full probe on best layer to get weight vectors
    X = hidden_states[:, best_layer, :]
    X_reduced, pca_model = reduce_dims(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)
    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                             random_state=42)
    clf.fit(X_scaled, labels)

    # Weight vectors: each row is a "sensory direction" in hidden state space
    sense_directions = clf.coef_  # (n_senses, hidden_dim)

    # Pairwise cosine similarity between sense directions
    n_senses = len(SENSES)
    cos_sim_matrix = np.zeros((n_senses, n_senses))
    for i in range(n_senses):
        for j in range(n_senses):
            cos_sim_matrix[i, j] = 1 - cosine(sense_directions[i], sense_directions[j])

    print("\n  Pairwise cosine similarity of sense directions:")
    print(f"  {'':15s}", end="")
    for s in SENSES:
        print(f"{s[:5]:>8s}", end="")
    print()
    for i, s in enumerate(SENSES):
        print(f"  {s:15s}", end="")
        for j in range(n_senses):
            print(f"{cos_sim_matrix[i,j]:8.3f}", end="")
        print()

    # Classic vs non-traditional sense similarity
    classic_indices = [SENSES.index(s) for s in CLASSIC_SENSES]
    intro_idx = SENSES.index("Interoceptive")

    classic_sims = []
    for i in classic_indices:
        for j in classic_indices:
            if i < j:
                classic_sims.append(cos_sim_matrix[i, j])

    intro_classic_sims = [cos_sim_matrix[intro_idx, i] for i in classic_indices]

    print(f"\n  Classic inter-sense similarity: mean={np.mean(classic_sims):.4f} ± {np.std(classic_sims):.4f}")
    print(f"  Interoceptive-to-classic similarity: mean={np.mean(intro_classic_sims):.4f} ± {np.std(intro_classic_sims):.4f}")

    # Random direction baseline
    n_random = 1000
    random_sims = []
    for _ in range(n_random):
        v1 = np.random.randn(sense_directions.shape[1])
        v2 = np.random.randn(sense_directions.shape[1])
        random_sims.append(1 - cosine(v1, v2))
    print(f"  Random direction similarity: mean={np.mean(random_sims):.4f} ± {np.std(random_sims):.4f}")

    results = {
        "best_layer": int(best_layer),
        "cosine_similarity_matrix": cos_sim_matrix.tolist(),
        "classic_inter_sense_sim": {
            "mean": float(np.mean(classic_sims)),
            "std": float(np.std(classic_sims)),
            "values": [float(x) for x in classic_sims],
        },
        "interoceptive_to_classic_sim": {
            "mean": float(np.mean(intro_classic_sims)),
            "std": float(np.std(intro_classic_sims)),
            "values": [float(x) for x in intro_classic_sims],
        },
        "random_baseline_sim": {
            "mean": float(np.mean(random_sims)),
            "std": float(np.std(random_sims)),
        },
    }

    # ---- PLOTS ----

    # 1. Cosine similarity heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.eye(n_senses, dtype=bool)
    sns.heatmap(cos_sim_matrix, annot=True, fmt=".3f", xticklabels=SENSES,
                yticklabels=SENSES, cmap="RdBu_r", center=0, ax=ax,
                vmin=-0.4, vmax=0.4)
    ax.set_title("Cosine Similarity of Sensory Direction Vectors", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/sense_direction_similarity.png", dpi=150)
    plt.close()

    # 2. PCA visualization of hidden states colored by sense
    X_best = hidden_states[:, best_layer, :]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_best)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax_idx, condition in enumerate(["implicit", "explicit", "control"]):
        ax = axes[ax_idx]
        for sense_idx, sense in enumerate(SENSES):
            mask = np.array([
                (m["dominant_sense"] == sense and m["condition"] == condition)
                for m in metadata
            ])
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=SENSE_COLORS[sense], label=sense, s=40, alpha=0.7)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
        ax.set_title(f"{condition.capitalize()} Condition", fontsize=13)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.2)
    plt.suptitle(f"PCA of Hidden States (Layer {best_layer})", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/pca_by_condition.png", dpi=150)
    plt.close()

    # 3. PCA of sense direction vectors themselves
    pca_dirs = PCA(n_components=2)
    dirs_2d = pca_dirs.fit_transform(sense_directions)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, sense in enumerate(SENSES):
        ax.scatter(dirs_2d[i, 0], dirs_2d[i, 1], c=SENSE_COLORS[sense],
                  s=200, zorder=5, edgecolors="black", linewidth=1.5)
        ax.annotate(sense, (dirs_2d[i, 0], dirs_2d[i, 1]),
                   textcoords="offset points", xytext=(10, 10), fontsize=12,
                   fontweight="bold")
    # Draw lines from interoceptive to each classic sense
    intro_pos = dirs_2d[SENSES.index("Interoceptive")]
    for s in CLASSIC_SENSES:
        s_pos = dirs_2d[SENSES.index(s)]
        ax.plot([intro_pos[0], s_pos[0]], [intro_pos[1], s_pos[1]],
                "k--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel(f"PC1 ({pca_dirs.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca_dirs.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    ax.set_title("Sensory Direction Vectors in PCA Space", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/sense_directions_pca.png", dpi=150)
    plt.close()

    return results


# ============================================================================
# EXPERIMENT 4: Continuous Sensory Strength Prediction
# ============================================================================

def experiment4_continuous_prediction(hidden_states, metadata, results_dir):
    """
    Instead of classification, predict continuous sensory strength scores
    from Lancaster norms using Ridge regression. Tests whether LLMs encode
    graded sensory information.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Continuous Sensory Strength Prediction")
    print("="*70)

    num_layers = hidden_states.shape[1]

    # Extract continuous targets
    targets = {}
    for sense in SENSES:
        targets[sense] = np.array([m["sense_scores"].get(sense, 0.0) for m in metadata])

    # Use middle-to-late layers (best for probing typically)
    best_layer = num_layers * 3 // 4  # ~layer 21 for 29-layer model

    X = hidden_states[:, best_layer, :]
    X_reduced, _ = reduce_dims(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)

    results = {}
    print(f"\n  Ridge regression R² at layer {best_layer}:")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, sense in enumerate(SENSES):
        y = targets[sense]
        ridge = Ridge(alpha=1.0)
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(ridge, X_scaled, y, cv=cv)

        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
        rho, p_val = spearmanr(y, y_pred)

        print(f"    {sense:15s}: R²={r2:.3f}, Spearman ρ={rho:.3f} (p={p_val:.2e})")

        results[sense] = {
            "r2": float(r2),
            "spearman_rho": float(rho),
            "spearman_p": float(p_val),
        }

        # Scatter plot
        ax = axes[idx]
        ax.scatter(y, y_pred, alpha=0.5, s=20, c=SENSE_COLORS[sense])
        ax.set_xlabel(f"True {sense} Score", fontsize=10)
        ax.set_ylabel(f"Predicted {sense} Score", fontsize=10)
        ax.set_title(f"{sense}\nR²={r2:.3f}, ρ={rho:.3f}", fontsize=11)

        # Add diagonal
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", alpha=0.3)
        ax.grid(True, alpha=0.2)

    plt.suptitle(f"Continuous Sensory Strength Prediction (Layer {best_layer})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/continuous_prediction.png", dpi=150)
    plt.close()

    return results


# ============================================================================
# EXPERIMENT 5: Layer-wise Emergence of Sensory Representations
# ============================================================================

def experiment5_layerwise_analysis(hidden_states, metadata, results_dir):
    """
    Analyze how sensory representations emerge and evolve across layers.
    Compute per-sense probe accuracy and inter-sense separability at each layer.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Layer-wise Emergence Analysis")
    print("="*70)

    labels = np.array([SENSES.index(m["dominant_sense"]) for m in metadata])
    num_layers = hidden_states.shape[1]

    # Per-sense accuracy at each layer
    per_sense_accuracy = {s: [] for s in SENSES}

    for layer in range(num_layers):
        X = hidden_states[:, layer, :]
        X_reduced, _ = reduce_dims(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reduced)
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                                 random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(clf, X_scaled, labels, cv=cv)

        for sense_idx, sense in enumerate(SENSES):
            sense_mask = labels == sense_idx
            if sense_mask.sum() > 0:
                sense_acc = accuracy_score(labels[sense_mask], y_pred[sense_mask])
                per_sense_accuracy[sense].append(sense_acc)

    # Plot per-sense accuracy curves
    fig, ax = plt.subplots(figsize=(12, 6))
    for sense in SENSES:
        ax.plot(range(num_layers), per_sense_accuracy[sense],
                label=sense, color=SENSE_COLORS[sense], linewidth=2)
    ax.axhline(y=1/len(SENSES), color="gray", linestyle="--", label="chance", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Per-Sense Accuracy", fontsize=12)
    ax.set_title("Layer-wise Emergence of Sensory Representations", fontsize=14)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/layerwise_per_sense.png", dpi=150)
    plt.close()

    # Find emergence patterns
    results = {}
    for sense in SENSES:
        accs = per_sense_accuracy[sense]
        peak_layer = np.argmax(accs)
        peak_acc = accs[peak_layer]
        # "Emergence" layer: first layer where accuracy exceeds 2x chance
        emergence = None
        for l, a in enumerate(accs):
            if a > 2 / len(SENSES):
                emergence = l
                break
        results[sense] = {
            "peak_layer": int(peak_layer),
            "peak_accuracy": float(peak_acc),
            "emergence_layer": int(emergence) if emergence is not None else None,
            "layer_accuracies": [float(a) for a in accs],
        }
        print(f"  {sense:15s}: peak at layer {peak_layer} ({peak_acc:.3f}), "
              f"emerges at layer {emergence}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Loading data...")
    hidden_states, metadata = load_data()
    print(f"  Hidden states: {hidden_states.shape}")
    print(f"  Stimuli: {len(metadata)}")

    results_dir = "results"
    os.makedirs(f"{results_dir}/plots", exist_ok=True)

    all_results = {}

    # Run all experiments
    all_results["exp1_probing"] = experiment1_linear_probing(
        hidden_states, metadata, results_dir)

    all_results["exp2_implicit_explicit"] = experiment2_implicit_vs_explicit(
        hidden_states, metadata, results_dir)

    all_results["exp3_subspace"] = experiment3_subspace_geometry(
        hidden_states, metadata, results_dir)

    all_results["exp4_continuous"] = experiment4_continuous_prediction(
        hidden_states, metadata, results_dir)

    all_results["exp5_layerwise"] = experiment5_layerwise_analysis(
        hidden_states, metadata, results_dir)

    # Save all results
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    with open(f"{results_dir}/experiment_results.json", "w") as f:
        json.dump(convert_numpy(all_results), f, indent=2)

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Results saved to {results_dir}/experiment_results.json")
    print(f"Plots saved to {results_dir}/plots/")

    return all_results


if __name__ == "__main__":
    main()
