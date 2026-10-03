import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


TASK = "dialect"
#options: "gender" or "dialect"

MODEL_NAME = "microsoft/wavlm-base"
##METADATA_FILE = "embeddings_mixer6_hubert/metadata_processed.csv"

EMBEDDINGS_DIR = "embeddings_mixer6_wavlm"  # change just this one line when switching models

EMBEDDING_FILES = {
    "first": f"{EMBEDDINGS_DIR}/embeddings_{MODEL_NAME.replace('/', '_')}_first.npy",
    "middle": f"{EMBEDDINGS_DIR}/embeddings_{MODEL_NAME.replace('/', '_')}_middle.npy",
    "final": f"{EMBEDDINGS_DIR}/embeddings_{MODEL_NAME.replace('/', '_')}_final.npy",
}
METADATA_FILE = "embeddings_mixer6_wavlm/metadata_processed.csv"

TEST_SIZE = 0.2

RANDOM_SEED = 42

OUTPUT_FILE = f"{TASK}_probe_summary_mixer6_{MODEL_NAME.replace('/', '_')}.csv"


def load_metadata(metadata_file):
    """Load metadata containing labels."""
    return pd.read_csv(metadata_file)


def get_labels(metadata, task):
    """Return the target labels for the selected probing task."""
    if task == "gender":
        # Female = 0, Male = 1
        return (metadata["gender"] == "M").astype(int)

    if task == "dialect":
        return metadata["dialect_region"]

    raise ValueError("TASK must be 'gender' or 'dialect'")


def drop_missing_labels(metadata, task):
    """CHANGED from TIMIT: Mixer 6 has a small number of rows with unknown
    labels (e.g. the 1 speaker not matched to demographics), which TIMIT's
    fully-labeled metadata never had. Drop those rows here so they don't
    corrupt the probe, and return a boolean mask so embeddings can be
    filtered the same way."""
    label_col = "gender" if task == "gender" else "dialect_region"
    valid_mask = ~metadata[label_col].astype(str).str.contains(
        "Unknown", case=False, na=True
    )
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"Dropping {n_dropped} rows with missing/unknown '{label_col}' labels")
    return valid_mask.to_numpy()


def get_train_test_indices(metadata, test_size, random_seed):
    """CHANGED from TIMIT: Mixer 6 has no official TRAIN/TEST split column
    (unlike TIMIT). More importantly, Mixer 6 has around 650 sentences per
    speaker, vs. TIMIT's ~1 sentence per file -- so a plain random
    row-level split would put the same speaker's sentences in both train
    and test, letting the classifier learn speaker-specific quirks
    instead of generalizable gender/dialect patterns. This splits by
    speaker_id first, so every sentence from a given speaker stays
    entirely in train or entirely in test."""
    unique_speakers = metadata["speaker_id"].unique()

    train_speakers, test_speakers = train_test_split(
        unique_speakers, test_size=test_size, random_state=random_seed,
    )

    train_idx = metadata.index[metadata["speaker_id"].isin(train_speakers)].to_numpy()
    test_idx = metadata.index[metadata["speaker_id"].isin(test_speakers)].to_numpy()

    print(f"Speakers: {len(train_speakers)} train / {len(test_speakers)} test "
          f"({len(unique_speakers)} total)")

    return train_idx, test_idx


def print_dataset_info(metadata, y, train_idx, test_idx):
    """Print dataset size and label distribution."""
    print("Total examples:", len(metadata))
    print("Train examples:", len(train_idx))
    print("Test examples :", len(test_idx))

    print("\nLabel counts in TRAIN:")
    print(y.iloc[train_idx].value_counts())

    print("\nLabel counts in TEST:")
    print(y.iloc[test_idx].value_counts())


def run_probe(X, y, train_idx, test_idx, task):
    """Train logistic regression and compute evaluation metrics."""
    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    mistakes = y_pred != y_test
    n_mistakes = mistakes.sum()

    #NEW: majority-class baseline -- what accuracy/macro F1 would you get
    #by always predicting the single most common label in TRAIN? This
    #makes the accuracy-vs-macro-F1 gap concrete: a probe that barely
    #beats this baseline on accuracy, while scoring far worse on macro F1,
    #is mostly just learning to predict the majority class.
    majority_label = y_train.value_counts().idxmax()
    y_baseline_pred = pd.Series([majority_label] * len(y_test), index=y_test.index)
    baseline_acc = accuracy_score(y_test, y_baseline_pred)
    baseline_macro_f1 = f1_score(y_test, y_baseline_pred, average="macro")

    results = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "mistakes": n_mistakes,
        "confusion_matrix": cm,
        "majority_label": majority_label,
        "baseline_accuracy": baseline_acc,
        "baseline_macro_f1": baseline_macro_f1,
    }
    #additional confidence metrics for binary gender classification
    if task == "gender":
        #probability that each test item is male
        y_prob_male = clf.predict_proba(X_test)[:, 1]

        #probability assigned to predicted class
        confidence = np.maximum(y_prob_male, 1 - y_prob_male)

        results["mean_confidence"] = confidence.mean()
        results["mean_prob_male_for_true_F"] = y_prob_male[y_test == 0].mean()
        results["mean_prob_male_for_true_M"] = y_prob_male[y_test == 1].mean()

    return results


#main

metadata_full = load_metadata(METADATA_FILE)

valid_mask = drop_missing_labels(metadata_full, TASK)
metadata = metadata_full[valid_mask].reset_index(drop=True)

y = get_labels(metadata, TASK)

train_idx, test_idx = get_train_test_indices(metadata, TEST_SIZE, RANDOM_SEED)

print_dataset_info(metadata, y, train_idx, test_idx)

summary_rows = []

for layer, filename in EMBEDDING_FILES.items():

    X_full = np.load(filename)
    X = X_full[valid_mask]

    results = run_probe(
        X=X,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        task=TASK,
    )

    print(f"\n{layer.upper()} LAYER")
    print("Accuracy:", results["accuracy"])
    print("Macro F1:", results["macro_f1"])
    print("Number of mistakes:", results["mistakes"])

    #print baseline right next to the real result, so the gap is
    #immediately visible
    print(f"\nMajority-class baseline (always predict '{results['majority_label']}'):")
    print("  Baseline accuracy:", results["baseline_accuracy"])
    print("  Baseline macro F1:", results["baseline_macro_f1"])
    print(f"  Accuracy gain over baseline: {results['accuracy'] - results['baseline_accuracy']:+.4f}")
    print(f"  Macro F1 gain over baseline: {results['macro_f1'] - results['baseline_macro_f1']:+.4f}")

    print("\nConfusion matrix:")
    print(results["confusion_matrix"])

    row = {
        "task": TASK,
        "model": MODEL_NAME,
        "layer": layer,
        "accuracy": results["accuracy"],
        "macro_f1": results["macro_f1"],
        "mistakes": results["mistakes"],
        "majority_label": results["majority_label"],
        "baseline_accuracy": results["baseline_accuracy"],
        "baseline_macro_f1": results["baseline_macro_f1"],
    }

    if TASK == "gender":
        print("\nMean confidence:", results["mean_confidence"])

        print("\nMean probability of MALE:")
        print("True female examples:", results["mean_prob_male_for_true_F"])
        print("True male examples  :", results["mean_prob_male_for_true_M"])

        row["mean_confidence"] = results["mean_confidence"]
        row["mean_prob_male_for_true_F"] = results["mean_prob_male_for_true_F"]
        row["mean_prob_male_for_true_M"] = results["mean_prob_male_for_true_M"]

    summary_rows.append(row)


summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_FILE, index=False)

print(summary)
print(f"\nSaved: {OUTPUT_FILE}")
