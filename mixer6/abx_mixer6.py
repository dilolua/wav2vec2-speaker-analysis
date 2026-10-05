"""
abx_mixer6.py

A simplified ABX-style evaluation for gender/dialect, adapted from the
core logic of Sun, McIntosh, Choi, Yeo, Saito & Minematsu (2026),
"Prosodic ABX: A Language-Agnostic Method for Measuring Prosodic Contrast
in Speech Representations."

WHAT'S KEPT FROM THE ORIGINAL METHOD:
  - The core triplet logic: given a reference point X, is it closer to a
    same-class point A, or a different-class point B?
  - Training-free: no classifier is trained (unlike probing), and no
    optimization over cluster assignments happens (unlike K-means
    clustering) -- this is a third, genuinely different kind of technique.
  - Scoring: 1 if d(X,A) < d(X,B), 0.5 if tied, 0 otherwise, averaged over
    many sampled triplets, then reported as an ERROR RATE (1 - mean score),
    matching the original paper's convention.

WHAT'S SIMPLIFIED FROM THE ORIGINAL METHOD (and why):
  - The original method compares FULL per-timestep sequences via Dynamic
    Time Warping (DTW), which requires saving every timestep's hidden
    state per clip (not just the mean-pooled summary). This would need a
    complete re-extraction with ~100-200x more storage per clip -- not
    practical given the disk-space/iCloud issues already encountered in
    this project. This script instead uses the mean-pooled 768-dim
    embeddings already extracted, with plain Euclidean or cosine distance
    instead of DTW. The core "same-class closer than different-class"
    logic is preserved; the sequence-alignment aspect is not.
  - The original method uses phonetic/prosodic MINIMAL PAIRS (same words,
    different prosody). Since this project's dimension is a stable
    SPEAKER-level property (gender, dialect), not a moment-to-moment
    linguistic contrast, triplets are instead sampled across different
    SPEAKERS -- X and A share the target label, B does not, and X/A/B are
    always three DIFFERENT speakers, so the test measures whether the
    property is prominent in overall representation geometry, not just
    decodable via a trained classifier.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_distance


MODEL_CHOICE = "wav2vec2"
#options: "wav2vec2", "hubert", or "wavlm"

MODEL_CONFIGS = {
    "wav2vec2": {"name": "facebook/wav2vec2-base", "dir": "embeddings_mixer6"},
    "hubert": {"name": "facebook/hubert-base-ls960", "dir": "embeddings_mixer6_hubert"},
    "wavlm": {"name": "microsoft/wavlm-base", "dir": "embeddings_mixer6_wavlm"},
}

MODEL_NAME = MODEL_CONFIGS[MODEL_CHOICE]["name"]
EMBEDDINGS_DIR = Path(MODEL_CONFIGS[MODEL_CHOICE]["dir"])
METADATA_FILE = EMBEDDINGS_DIR / "metadata_processed.csv"

LAYERS = ["first", "middle", "final"]

TASK = "gender"
#options: "gender" or "dialect_region"

DISTANCE_METRIC = "euclidean"
#options: "cosine" or "euclidean"

N_TRIPLETS = 5000
#number of random (X, A, B) triplets to sample for the ABX score

AGGREGATE_BY_SPEAKER = True
#sample triplets at the SPEAKER level (one embedding
#per speaker, averaged -- same as the successful clustering approach),

RANDOM_SEED = 123

OUTPUT_DIR = Path("abx_results_mixer6")
OUTPUT_FILE = OUTPUT_DIR / f"{TASK}_abx_summary_{MODEL_NAME.replace('/', '_')}.csv"


#load embeddings + metadata (same pattern as other scripts)

def load_layer_embeddings(layer_name: str) -> np.ndarray:
    safe_model_name = MODEL_NAME.replace("/", "_")
    path = EMBEDDINGS_DIR / f"embeddings_{safe_model_name}_{layer_name}.npy"
    return np.load(path)


def load_metadata() -> pd.DataFrame:
    return pd.read_csv(METADATA_FILE)


def drop_missing_labels(metadata: pd.DataFrame, task: str) -> np.ndarray:
    valid_mask = ~metadata[task].astype(str).str.contains(
        "Unknown", case=False, na=True
    )
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"Dropping {n_dropped} rows with missing/unknown '{task}' labels")
    return valid_mask.to_numpy()


def aggregate_by_speaker(embeddings: np.ndarray, metadata: pd.DataFrame,
                          task: str) -> tuple[np.ndarray, pd.Series, np.ndarray]:
    """Average each speaker's embeddings into one row. Returns
    (speaker_embeddings, speaker_labels, speaker_ids), all aligned."""
    df = pd.DataFrame({"speaker_id": metadata["speaker_id"].to_numpy()})
    df["_row_idx"] = np.arange(len(df))

    speaker_labels = metadata.groupby("speaker_id")[task].first()
    speaker_ids_ordered = speaker_labels.index.to_numpy()

    speaker_embeddings = np.vstack([
        embeddings[df.loc[df["speaker_id"] == sid, "_row_idx"].to_numpy()].mean(axis=0)
        for sid in speaker_ids_ordered
    ])

    return speaker_embeddings, speaker_labels.reset_index(drop=True), speaker_ids_ordered


#distance function

def compute_distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        return cosine_distance(a, b)
    elif metric == "euclidean":
        return np.linalg.norm(a - b)
    raise ValueError(f"Unknown distance metric: {metric}")


#sample triplets and compute ABX score

def run_abx(embeddings: np.ndarray, labels: pd.Series, n_triplets: int,
            distance_metric: str, random_seed: int) -> dict:
    """For n_triplets random (X, A, B) triplets -- where X and A share a
    label, B has a different label, and X/A/B are three different
    speakers -- score 1 if d(X,A) < d(X,B), 0.5 if tied, 0 otherwise.
    Returns the mean score and the resulting ABX error rate (1 - mean)."""
    rng = np.random.default_rng(random_seed)
    labels_arr = labels.to_numpy()
    unique_labels = np.unique(labels_arr)

    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 classes to run ABX")

    label_to_indices = {lbl: np.where(labels_arr == lbl)[0] for lbl in unique_labels}

    scores = []
    attempts = 0
    max_attempts = n_triplets * 20  # safety valve in case some classes are tiny

    while len(scores) < n_triplets and attempts < max_attempts:
        attempts += 1

        x_label = rng.choice(unique_labels)
        other_labels = [l for l in unique_labels if l != x_label]
        b_label = rng.choice(other_labels)

        same_class_pool = label_to_indices[x_label]
        diff_class_pool = label_to_indices[b_label]

        if len(same_class_pool) < 2 or len(diff_class_pool) < 1:
            continue  # not enough speakers in this class to form a triplet

        x_idx, a_idx = rng.choice(same_class_pool, size=2, replace=False)
        b_idx = rng.choice(diff_class_pool)

        d_xa = compute_distance(embeddings[x_idx], embeddings[a_idx], distance_metric)
        d_xb = compute_distance(embeddings[x_idx], embeddings[b_idx], distance_metric)

        if d_xa < d_xb:
            scores.append(1.0)
        elif d_xa > d_xb:
            scores.append(0.0)
        else:
            scores.append(0.5)

    mean_score = np.mean(scores)
    error_rate = 1.0 - mean_score

    return {
        "mean_abx_score": mean_score,
        "abx_error_rate": error_rate,
        "n_triplets_used": len(scores),
    }



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Task: {TASK}")
    print(f"Model: {MODEL_NAME}")
    print(f"Distance metric: {DISTANCE_METRIC}")
    print(f"Aggregate by speaker: {AGGREGATE_BY_SPEAKER}\n")

    metadata_full = load_metadata()
    valid_mask = drop_missing_labels(metadata_full, TASK)
    metadata = metadata_full[valid_mask].reset_index(drop=True)

    summary_rows = []

    for layer_name in LAYERS:
        print(f"--- Layer: {layer_name} ---")
        embeddings_full = load_layer_embeddings(layer_name)
        embeddings = embeddings_full[valid_mask]
        labels = metadata[TASK]

        if AGGREGATE_BY_SPEAKER:
            embeddings, labels, _ = aggregate_by_speaker(embeddings, metadata, TASK)
            print(f"  Aggregated to {len(embeddings)} speaker-level embeddings")

        result = run_abx(embeddings, labels, N_TRIPLETS, DISTANCE_METRIC, RANDOM_SEED)

        print(f"  ABX error rate: {result['abx_error_rate']:.4f} "
              f"(0 = perfect, 0.5 = chance, 1 = always wrong)")
        print(f"  Mean ABX score: {result['mean_abx_score']:.4f}")
        print(f"  Triplets used: {result['n_triplets_used']}\n")

        summary_rows.append({
            "task": TASK,
            "model": MODEL_NAME,
            "layer": layer_name,
            "distance_metric": DISTANCE_METRIC,
            "aggregate_by_speaker": AGGREGATE_BY_SPEAKER,
            "abx_error_rate": result["abx_error_rate"],
            "mean_abx_score": result["mean_abx_score"],
            "n_triplets_used": result["n_triplets_used"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_FILE, index=False)
    print(summary_df.to_string())
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
