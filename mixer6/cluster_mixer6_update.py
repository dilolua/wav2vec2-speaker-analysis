from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
)


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

RANDOM_SEED = 42

#two methodological refinements, each independently toggleable for testing purposes

STANDARDIZE = True
#Standardizes each of the 768 embedding dimensions to mean 0, variance 1
#before clustering. Without this, K-means (which uses raw Euclidean
#distance) is dominated by whichever dimensions happen to have the largest
#variance -- which may have nothing to do with the property you're testing
#for. This is standard practice for K-means
# it fixes a real methodological gap in using raw,
#unscaled embeddings for distance-based clustering.

AGGREGATE_BY_SPEAKER = True
#If True, averages all of a speaker's sentence-level embeddings into a
#single "speaker profile" BEFORE clustering, so you cluster ~196 speaker
#profiles instead of ~140,000 individual sentences. This removes
#utterance-level noise (differing sentence content, recording conditions)
#that has nothing to do with speaker-level properties like gender/dialect.

OUTPUT_DIR = Path("cluster_results_mixer6/cluster_results_mixer6_new")
_suffix = ""
if STANDARDIZE:
    _suffix += "_standardized"
if AGGREGATE_BY_SPEAKER:
    _suffix += "_speakerlevel"
OUTPUT_FILE = OUTPUT_DIR / f"{TASK}_cluster_summary_{MODEL_NAME.replace('/', '_')}{_suffix}.csv"
PER_CLUSTER_OUTPUT_FILE = OUTPUT_DIR / f"{TASK}_cluster_details_{MODEL_NAME.replace('/', '_')}{_suffix}.csv"
#since the number of clusters differs by task (2 for gender, 5 for
#dialect_region), the per-cluster dominant-class/size/purity breakdown is
#saved as a SEPARATE file, one row per (layer, cluster_id), rather than
#trying to cram a variable number of clusters into the single-row-per-layer
#summary file above.


# ------------------------------------------------------------------------
# STEP 1 -- Load embeddings + metadata (same pattern as probe_mixer6.py)
# ------------------------------------------------------------------------

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


# ------------------------------------------------------------------------
# STEP 1b -- Optional: aggregate to one embedding per speaker
# ------------------------------------------------------------------------

def aggregate_by_speaker(embeddings: np.ndarray, metadata: pd.DataFrame,
                          task: str) -> tuple[np.ndarray, pd.Series]:
    """Average all of a speaker's embeddings into one row, and take their
    (single, unique) label. Returns (speaker_embeddings, speaker_labels),
    both ordered the same way."""
    df = pd.DataFrame({"speaker_id": metadata["speaker_id"].to_numpy()})
    df["_row_idx"] = np.arange(len(df))

    speaker_labels = metadata.groupby("speaker_id")[task].first()
    speaker_ids_ordered = speaker_labels.index.to_numpy()

    speaker_embeddings = np.vstack([
        embeddings[df.loc[df["speaker_id"] == sid, "_row_idx"].to_numpy()].mean(axis=0)
        for sid in speaker_ids_ordered
    ])

    return speaker_embeddings, speaker_labels.reset_index(drop=True)


# ------------------------------------------------------------------------
# STEP 1c -- Cluster purity (dominant class proportion per cluster)
# ------------------------------------------------------------------------

def compute_cluster_purity(true_labels: pd.Series, cluster_assignments: np.ndarray) -> dict:
    """For each cluster, find the dominant (most common) true label and
    what fraction of that cluster's members belong to it:
        purity of one cluster = (count of dominant class in that cluster)
                                 / (total observations in that cluster)
    Also computes an overall purity across all clusters, weighted by
    cluster size. This is simpler and more directly interpretable than
    ARI/AMI, at the cost of being less sensitive to some kinds of
    clustering errors (e.g. it doesn't penalize a real label being split
    across multiple clusters, the way completeness does)."""
    true_labels_arr = np.asarray(true_labels)
    per_cluster_purity = {}
    per_cluster_dominant_class = {}
    per_cluster_size = {}

    for cluster_id in np.unique(cluster_assignments):
        members_mask = cluster_assignments == cluster_id
        members_labels = true_labels_arr[members_mask]
        counts = pd.Series(members_labels).value_counts()
        dominant_class = counts.index[0]
        dominant_count = counts.iloc[0]
        cluster_size = len(members_labels)

        per_cluster_purity[cluster_id] = dominant_count / cluster_size
        per_cluster_dominant_class[cluster_id] = dominant_class
        per_cluster_size[cluster_id] = cluster_size

    total_correct = sum(per_cluster_purity[c] * per_cluster_size[c]
                         for c in per_cluster_purity)
    total_n = sum(per_cluster_size.values())
    overall_purity = total_correct / total_n

    return {
        "overall_purity": overall_purity,
        "per_cluster_purity": per_cluster_purity,
        "per_cluster_dominant_class": per_cluster_dominant_class,
        "per_cluster_size": per_cluster_size,
    }


# ------------------------------------------------------------------------
# STEP 2 -- Run K-means and evaluate against true labels
# ------------------------------------------------------------------------

def run_clustering(embeddings: np.ndarray, true_labels: pd.Series,
                    n_clusters: int, random_seed: int,
                    standardize: bool = STANDARDIZE) -> dict:
    if standardize:
        embeddings = StandardScaler().fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    cluster_assignments = kmeans.fit_predict(embeddings)

    ari = adjusted_rand_score(true_labels, cluster_assignments)
    ami = adjusted_mutual_info_score(true_labels, cluster_assignments)
    homogeneity = homogeneity_score(true_labels, cluster_assignments)
    completeness = completeness_score(true_labels, cluster_assignments)
    purity_results = compute_cluster_purity(true_labels, cluster_assignments)

    return {
        "n_clusters": n_clusters,
        "ari": ari,
        "ami": ami,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "overall_purity": purity_results["overall_purity"],
        "per_cluster_purity": purity_results["per_cluster_purity"],
        "per_cluster_dominant_class": purity_results["per_cluster_dominant_class"],
        "per_cluster_size": purity_results["per_cluster_size"],
        "cluster_assignments": cluster_assignments,
    }



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Task: {TASK}")
    print(f"Model: {MODEL_NAME}")
    print(f"Standardize features: {STANDARDIZE}")
    print(f"Aggregate by speaker: {AGGREGATE_BY_SPEAKER}\n")

    metadata_full = load_metadata()
    valid_mask = drop_missing_labels(metadata_full, TASK)
    metadata = metadata_full[valid_mask].reset_index(drop=True)

    true_labels = metadata[TASK]
    n_true_classes = true_labels.nunique()
    print(f"Number of true classes for '{TASK}': {n_true_classes}")
    print(f"Class distribution:\n{true_labels.value_counts()}\n")

    #ask K-means for the SAME number of clusters as true classes.
    #This is the natural choice when directly comparing "do the clusters
    #match the labels"
    n_clusters = n_true_classes

    summary_rows = []
    per_cluster_rows = []  # NEW: one row per (layer, cluster_id)

    for layer_name in LAYERS:
        print(f"--- Layer: {layer_name} ---")
        embeddings_full = load_layer_embeddings(layer_name)
        embeddings = embeddings_full[valid_mask]
        labels_for_clustering = true_labels

        if AGGREGATE_BY_SPEAKER:
            embeddings, labels_for_clustering = aggregate_by_speaker(
                embeddings, metadata, TASK
            )
            print(f"  Aggregated to {len(embeddings)} speaker-level embeddings")

        result = run_clustering(embeddings, labels_for_clustering, n_clusters, RANDOM_SEED)

        print(f"  ARI:          {result['ari']:.4f}")
        print(f"  AMI:          {result['ami']:.4f}")
        print(f"  Homogeneity:  {result['homogeneity']:.4f}")
        print(f"  Completeness: {result['completeness']:.4f}")
        print(f"  Overall purity: {result['overall_purity']:.4f} "
              f"(fraction of all points in their cluster's dominant class)")
        for cid in sorted(result["per_cluster_dominant_class"]):
            print(f"    Cluster {cid}: size={result['per_cluster_size'][cid]}, "
                  f"dominant class={result['per_cluster_dominant_class'][cid]}, "
                  f"purity={result['per_cluster_purity'][cid]:.4f}")


            per_cluster_rows.append({
                "task": TASK,
                "model": MODEL_NAME,
                "layer": layer_name,
                "cluster_id": cid,
                "cluster_size": result["per_cluster_size"][cid],
                "dominant_class": result["per_cluster_dominant_class"][cid],
                "cluster_purity": result["per_cluster_purity"][cid],
            })
        print()

        summary_rows.append({
            "task": TASK,
            "model": MODEL_NAME,
            "layer": layer_name,
            "n_clusters": result["n_clusters"],
            "ari": result["ari"],
            "ami": result["ami"],
            "homogeneity": result["homogeneity"],
            "completeness": result["completeness"],
            "overall_purity": result["overall_purity"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_FILE, index=False)
    print(summary_df.to_string())
    print(f"\nSaved: {OUTPUT_FILE}")

    per_cluster_df = pd.DataFrame(per_cluster_rows)
    per_cluster_df.to_csv(PER_CLUSTER_OUTPUT_FILE, index=False)
    print(f"Saved: {PER_CLUSTER_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
