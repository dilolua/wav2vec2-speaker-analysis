import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULT_FILES = {
    "gender_SA1": "gender_probe_summary_sa1.csv",
    "gender_SA2": "gender_probe_summary_sa2.csv",
    "dialect_SA1": "dialect_probe_summary_sa1.csv",
    "dialect_SA2": "dialect_probe_summary_sa2.csv",
}

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

LAYER_ORDER = ["first", "middle", "final"]


def load_results(file_path, sentence):
    df = pd.read_csv(file_path)

    df["sentence"] = sentence
    df["layer"] = pd.Categorical(
        df["layer"],
        categories=LAYER_ORDER,
        ordered=True
    )

    return df.sort_values("layer")


def plot_accuracy(task, chance_level):
    sa1 = load_results(RESULT_FILES[f"{task}_SA1"], "SA1")
    sa2 = load_results(RESULT_FILES[f"{task}_SA2"], "SA2")

    plt.figure(figsize=(7, 5))

    plt.plot(
        sa1["layer"].astype(str),
        sa1["accuracy"],
        marker="o",
        label="SA1"
    )

    plt.plot(
        sa2["layer"].astype(str),
        sa2["accuracy"],
        marker="o",
        label="SA2"
    )

    plt.axhline(
        chance_level,
        linestyle="--",
        label="Chance level"
    )

    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(f"{task.capitalize()} classification accuracy across layers")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{task}_accuracy.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


def plot_gender_confidence():
    sa1 = load_results(RESULT_FILES["gender_SA1"], "SA1")
    sa2 = load_results(RESULT_FILES["gender_SA2"], "SA2")

    plt.figure(figsize=(7, 5))

    plt.plot(
        sa1["layer"].astype(str),
        sa1["mean_confidence"],
        marker="o",
        label="SA1"
    )

    plt.plot(
        sa2["layer"].astype(str),
        sa2["mean_confidence"],
        marker="o",
        label="SA2"
    )

    plt.xlabel("Layer")
    plt.ylabel("Mean confidence")
    plt.title("Gender classifier confidence across layers")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "gender_confidence.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


plot_accuracy(task="gender", chance_level=0.5)
plot_accuracy(task="dialect", chance_level=1 / 8)
plot_gender_confidence()