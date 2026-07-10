import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


TASK = "gender"
#options: "gender" or "dialect"

SENTENCE = "SA2"
#options: "SA1" or "SA2"

METADATA_FILE = f"timit_sa1_sa2/{SENTENCE}/metadata_{SENTENCE.lower()}.csv"

EMBEDDING_FILES = {
    "first": "embeddings_facebook_wav2vec2-base_first2.npy",
    "middle": "embeddings_facebook_wav2vec2-base_middle2.npy",
    "final": "embeddings_facebook_wav2vec2-base_final2.npy",
}

OUTPUT_FILE = f"{TASK}_probe_summary_{SENTENCE.lower()}.csv"


def load_metadata(metadata_file):
    """Load metadata containing labels and train/test split."""
    return pd.read_csv(metadata_file)


def get_labels(metadata, task):
    """Return the target labels for the selected probing task."""
    if task == "gender":
        # Female = 0, Male = 1
        return (metadata["gender"] == "M").astype(int)

    if task == "dialect":
        #dialects: DR1, DR2, ..., DR8
        return metadata["dialect"]

    raise ValueError("TASK must be 'gender' or 'dialect'")


def get_train_test_indices(metadata):
    """Use the official TIMIT TRAIN/TEST split."""
    train_idx = metadata.index[metadata["split"] == "TRAIN"].to_numpy()
    test_idx = metadata.index[metadata["split"] == "TEST"].to_numpy()
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

    results = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "mistakes": n_mistakes,
        "confusion_matrix": cm,
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

metadata = load_metadata(METADATA_FILE)
y = get_labels(metadata, TASK)

train_idx, test_idx = get_train_test_indices(metadata)

print_dataset_info(metadata, y, train_idx, test_idx)

summary_rows = []

for layer, filename in EMBEDDING_FILES.items():

    X = np.load(filename)

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

    print("\nConfusion matrix:")
    print(results["confusion_matrix"])

    row = {
        "task": TASK,
        "sentence": SENTENCE,
        "layer": layer,
        "accuracy": results["accuracy"],
        "macro_f1": results["macro_f1"],
        "mistakes": results["mistakes"],
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