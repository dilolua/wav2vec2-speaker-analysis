"""
abx_mixer6_sentence_matched.py

ABX for GENDER with the trial structure:
  - A is always a MALE speaker, B is always a FEMALE speaker.
  - X is male or female, 50/50 (alternating per trial).
  - A, B and X are three DIFFERENT speakers saying the SAME SENTENCE
    (same sentence_text), so content is held constant and only the
    speaker differs.
  - Score: X male   -> correct if d(X,A) < d(X,B)
           X female -> correct if d(X,B) < d(X,A)
    ties = 0.5. Error rate = 1 - mean score (0 = perfect, 0.5 = chance).

Why sample instead of enumerating every male/female combination: all
3-way combinations within a sentence would give X's gender whatever mix
the combinations happen to have (more males -> more male-X trials). Here
X's gender alternates male/female, so it is 50/50 by construction, and
the script prints the achieved split as a check.

Input is SENTENCE-level embeddings (no speaker averaging). If a speaker
said the same sentence more than once, their embeddings for it are
averaged, so each (sentence, speaker) pair is one point.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_distance



MODEL_CHOICE = "wavlm"
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

DISTANCE_METRIC = "cosine"
#options: "cosine" or "euclidean"

N_TRIPLETS = 5000
RANDOM_SEED = 42

MIN_SPEAKERS_PER_SENTENCE = 3
#a sentence is usable only if >=3 speakers said it (needed for A, B, X)
#AND it has at least one male and one female speaker (see below).

OUTPUT_DIR = Path("abx_results_mixer6")
OUTPUT_FILE = OUTPUT_DIR / f"gender_abx_sentence_matched_{MODEL_NAME.replace('/', '_')}_{DISTANCE_METRIC}.csv"


#Load

def load_layer_embeddings(layer_name: str) -> np.ndarray:
    safe_model_name = MODEL_NAME.replace("/", "_")
    return np.load(EMBEDDINGS_DIR / f"embeddings_{safe_model_name}_{layer_name}.npy")


def compute_distance(a, b, metric):
    if metric == "cosine":
        return cosine_distance(a, b)
    if metric == "euclidean":
        return float(np.linalg.norm(a - b))
    raise ValueError(f"Unknown distance metric: {metric}")


# Build the (sentence, speaker) table

def build_sentence_speaker_table(embeddings, metadata):
    """Returns (points, table). points[i] = mean embedding of one speaker
    saying one sentence; table has sentence, speaker_id, gender per row."""
    df = metadata[["speaker_id", "sentence_text", "gender"]].copy()
    df = df[df["gender"].isin(["M", "F"])]
    df["sentence"] = df["sentence_text"].astype(str).str.strip().str.upper()
    df["_row"] = df.index  # row index into embeddings (metadata is aligned)

    rows, points = [], []
    for (sent, spk), g in df.groupby(["sentence", "speaker_id"]):
        rows.append({"sentence": sent, "speaker_id": spk, "gender": g["gender"].iloc[0]})
        points.append(embeddings[g["_row"].to_numpy()].mean(axis=0))
    return np.vstack(points), pd.DataFrame(rows)


def eligible_sentences(table):
    """Sentences where a trial can exist for BOTH X genders:
    X male needs >=2 males + >=1 female; X female needs >=1 male + >=2
    females. Requiring both keeps the 50/50 X split achievable everywhere."""
    out = {}
    for sent, g in table.groupby("sentence"):
        m = g.index[g["gender"] == "M"].to_numpy()
        f = g.index[g["gender"] == "F"].to_numpy()
        if len(g) >= MIN_SPEAKERS_PER_SENTENCE and len(m) >= 2 and len(f) >= 2:
            out[sent] = (m, f)
    return out


#Sample trials and score

def run_abx(points, table, n_triplets, metric, seed):
    rng = np.random.default_rng(seed)
    sents = eligible_sentences(table)
    if not sents:
        raise ValueError("No sentence has >=2 male and >=2 female speakers")
    sent_list = list(sents)

    scores, x_gender = [], []
    for i in range(n_triplets):
        m_idx, f_idx = sents[sent_list[rng.integers(len(sent_list))]]
        x_is_male = (i % 2 == 0)                  # exactly 50/50 X gender by design

        if x_is_male:
            x, a = rng.choice(m_idx, size=2, replace=False)  # X, A both male (different speakers)
            b = rng.choice(f_idx)
        else:
            x, b = rng.choice(f_idx, size=2, replace=False)  # X, B both female
            a = rng.choice(m_idx)

        d_xa = compute_distance(points[x], points[a], metric)
        d_xb = compute_distance(points[x], points[b], metric)
        correct_if_closer_to = "A" if x_is_male else "B"
        if d_xa == d_xb:
            s = 0.5
        else:
            closer = "A" if d_xa < d_xb else "B"
            s = 1.0 if closer == correct_if_closer_to else 0.0
        scores.append(s)
        x_gender.append("M" if x_is_male else "F")

    scores, x_gender = np.array(scores), np.array(x_gender)
    return {
        "abx_error_rate": 1.0 - scores.mean(),
        "error_rate_x_male": 1.0 - scores[x_gender == "M"].mean(),
        "error_rate_x_female": 1.0 - scores[x_gender == "F"].mean(),
        "share_x_male": float((x_gender == "M").mean()),
        "n_triplets_used": len(scores),
        "n_eligible_sentences": len(sent_list),
    }



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model: {MODEL_NAME} | metric: {DISTANCE_METRIC}\n")

    metadata = pd.read_csv(METADATA_FILE)
    rows = []
    for layer in LAYERS:
        print(f"--- Layer: {layer} ---")
        points, table = build_sentence_speaker_table(load_layer_embeddings(layer), metadata)
        r = run_abx(points, table, N_TRIPLETS, DISTANCE_METRIC, RANDOM_SEED)
        print(f"  Eligible sentences: {r['n_eligible_sentences']}")
        print(f"  X gender split: {r['share_x_male']:.3f} male (target 0.500)")
        print(f"  ABX error: {r['abx_error_rate']:.4f}  "
              f"(X male {r['error_rate_x_male']:.4f}, X female {r['error_rate_x_female']:.4f})\n")
        rows.append({"model": MODEL_NAME, "layer": layer, "distance_metric": DISTANCE_METRIC, **r})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(df.to_string())
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
