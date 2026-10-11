"""
linear_probing_per_sentence.py

Per-sentence linear probing, matching the TIMIT control: one probe is
trained PER SENTENCE, so the linguistic content is identical across every
training and test example of that probe. Only the speaker changes.

LogisticRegression(max_iter=5000), accuracy,
macro F1, majority-class baseline, layers first/middle/final, mean-pooled
embeddings, "Unknown" labels dropped.

ONE speaker-disjoint 80/20 split (random_state=42) is made
once over all speakers and reused for every sentence, so all probes see
the same train/test speakers and results are comparable.

A sentence gets a probe only if it has enough speakers in train and test
and at least MIN_SPEAKERS_PER_CLASS_* speakers of every class on both sides.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

TASK = "dialect_region"
#options: "gender" or "dialect_region"

MODEL_CHOICE = "hubert"
#options: "wav2vec2", "hubert", or "wavlm"

MODEL_CONFIGS = {
    "wav2vec2": {"name": "facebook/wav2vec2-base", "dir": "embeddings_mixer6"},
    "hubert": {"name": "facebook/hubert-base-ls960", "dir": "embeddings_mixer6_hubert"},
    "wavlm": {"name": "microsoft/wavlm-base", "dir": "embeddings_mixer6_wavlm"},
}

MODEL_NAME = MODEL_CONFIGS[MODEL_CHOICE]["name"]
EMBEDDINGS_DIR = MODEL_CONFIGS[MODEL_CHOICE]["dir"]
METADATA_FILE = f"{EMBEDDINGS_DIR}/metadata_processed.csv"
LAYERS = ["first", "middle", "final"]

TEST_SIZE = 0.2
RANDOM_SEED = 42

N_SENTENCES = 20
#probe N_SENTENCES eligible sentences 

MIN_SPEAKERS_PER_CLASS_TRAIN = 5
MIN_SPEAKERS_PER_CLASS_TEST = 2

safe = MODEL_NAME.replace("/", "_")
PER_SENTENCE_FILE = f"{TASK}_probe_per_sentence_{safe}.csv"
SUMMARY_FILE = f"{TASK}_probe_per_sentence_summary_{safe}.csv"


def get_labels(metadata, task):
    if task == "gender":
        return (metadata["gender"] == "M").astype(int)   # F=0, M=1
    if task == "dialect_region":
        return metadata["dialect_region"]
    raise ValueError("TASK must be 'gender' or 'dialect_region'")


def main():
    meta_full = pd.read_csv(METADATA_FILE)
    valid = ~meta_full[TASK].astype(str).str.contains("Unknown", case=False, na=True)
    print(f"Dropping {(~valid).sum()} rows with unknown '{TASK}'")
    meta = meta_full[valid].reset_index(drop=True)
    meta["sentence"] = meta["sentence_text"].astype(str).str.strip().str.upper()
    y = get_labels(meta, TASK)

    speakers = meta["speaker_id"].unique()
    train_spk, test_spk = train_test_split(speakers, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_set, test_set = set(train_spk), set(test_spk)
    print(f"Speakers: {len(train_spk)} train / {len(test_spk)} test")

    # speaker -> class (one label per speaker)
    spk_class = meta.groupby("speaker_id")[TASK].first()
    if TASK == "gender":
        spk_class = (spk_class == "M").astype(int)

    eligible = []
    for sent, g in meta.groupby("sentence"):
        spks = g["speaker_id"].unique()
        tr = [s for s in spks if s in train_set]
        te = [s for s in spks if s in test_set]
        tr_counts = spk_class.loc[tr].value_counts()
        te_counts = spk_class.loc[te].value_counts()
        n_cls = spk_class.nunique()
        if (len(tr_counts) == n_cls and tr_counts.min() >= MIN_SPEAKERS_PER_CLASS_TRAIN
                and len(te_counts) == n_cls and te_counts.min() >= MIN_SPEAKERS_PER_CLASS_TEST):
            eligible.append((sent, len(spks)))

    np.random.default_rng(RANDOM_SEED).shuffle(eligible)
    eligible.sort(key=lambda t: -t[1])  # stable: keeps the shuffled order within ties
    print(f"Eligible sentences: {len(eligible)}")
    chosen = eligible if N_SENTENCES is None else eligible[:N_SENTENCES]
    print(f"Probing {len(chosen)} sentence(s)\n")

    rows = []
    for layer in LAYERS:
        X_all = np.load(f"{EMBEDDINGS_DIR}/embeddings_{safe}_{layer}.npy")[valid]
        for sent, n_spk in chosen:
            m = (meta["sentence"] == sent).to_numpy()
            tr_m = m & meta["speaker_id"].isin(train_set).to_numpy()
            te_m = m & meta["speaker_id"].isin(test_set).to_numpy()
            clf = LogisticRegression(max_iter=5000).fit(X_all[tr_m], y[tr_m])
            pred = clf.predict(X_all[te_m])
            majority = y[tr_m].value_counts().idxmax()
            base = np.full(te_m.sum(), majority)
            rows.append({
                "task": TASK, "model": MODEL_NAME, "layer": layer, "sentence": sent,
                "n_speakers": n_spk, "n_train": int(tr_m.sum()), "n_test": int(te_m.sum()),
                "accuracy": accuracy_score(y[te_m], pred),
                "macro_f1": f1_score(y[te_m], pred, average="macro"),
                "baseline_accuracy": accuracy_score(y[te_m], base),
                "baseline_macro_f1": f1_score(y[te_m], base, average="macro"),
            })
        print(f"{layer}: done")

    per = pd.DataFrame(rows)
    per.to_csv(PER_SENTENCE_FILE, index=False)
    summary = (per.groupby("layer", sort=False)
               [["accuracy", "macro_f1", "baseline_accuracy", "baseline_macro_f1"]]
               .agg(["mean", "std"]))
    summary.to_csv(SUMMARY_FILE)
    print("\n", summary.to_string())
    print(f"\nSaved: {PER_SENTENCE_FILE}\nSaved: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
