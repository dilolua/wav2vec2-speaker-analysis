# wav2vec2-speaker-analysis

Analysis of gender and dialect information encoded in wav2vec2 representations using internal probing.

---

# Project Overview

This project investigates whether a pretrained **wav2vec2** model encodes information about a speaker's **gender** and **dialect (accent)**.

Rather than directly comparing speech embeddings, the project uses an **internal probing** approach. A simple Logistic Regression classifier is trained on embeddings extracted from different layers of wav2vec2. If the classifier performs well, this indicates that the corresponding speaker information is encoded in the learned representations.

The analysis was performed separately for the two common TIMIT sentences (**SA1** and **SA2**) in order to compare the results across different utterances while keeping the spoken content identical.

---

# Motivation

The goal of this project is to investigate:

- Is gender information encoded in wav2vec2 representations?
- Is dialect (accent) information encoded in wav2vec2 representations?
- How does this information change across different layers of the model?

---

# Dataset

The experiments use the **TIMIT Acoustic-Phonetic Continuous Speech Corpus**.

Only the two sentences spoken by every speaker were selected:

- SA1
- SA2

Each sentence was analyzed independently.

The official **TRAIN** and **TEST** split provided by TIMIT was preserved throughout all experiments.

The metadata files contain:

- audio path
- train/test split
- speaker ID
- gender
- dialect region
- sentence


---

# Model

Pretrained model:

```
facebook/wav2vec2-base
```

The model was loaded using the Hugging Face Transformers library.

---

# Method

## 1. Embedding Extraction

For each audio file:

1. Load the waveform.
2. Resample the audio to **16 kHz** (if necessary).
3. Pass the audio through the pretrained wav2vec2 model.
4. Extract hidden representations from three transformer layers:
   - first layer
   - middle layer
   - final layer
5. Apply **mean pooling** over the time dimension to obtain a single **768-dimensional embedding** for each utterance.
6. Save the embeddings as NumPy (`.npy`) files.

---

## 2. Internal Probing

The extracted embeddings are used to train a simple **Logistic Regression** classifier.

Two probing experiments were performed.

### Gender Classification

Classes:

- Female
- Male

Metrics:

- Accuracy
- Macro F1-score
- Confusion Matrix
- Mean confidence
- Mean probability assigned to the male class

---

### Dialect Classification

Classes:

- TIMIT dialect regions (DR1–DR8)

Metrics:

- Accuracy
- Macro F1-score
- Confusion Matrix

---

# Results

## Gender

The classifier achieved approximately **98–99% accuracy** for both SA1 and SA2, indicating that gender information is strongly encoded in the wav2vec2 embeddings across all examined layers.

Although the classification accuracy was very similar across layers, the **first layer** produced slightly higher confidence scores. This suggests that low-level acoustic characteristics, such as pitch and vocal tract properties, are represented most strongly in the early layers of the model. These acoustic cues are highly informative for distinguishing between male and female speakers.

---

## Dialect

Dialect classification was considerably more challenging than gender classification, with accuracies ranging from approximately **20% to 36%**. However, these results are still well above the chance level of **12.5%** (8 dialect classes), indicating that dialect information is also encoded in the learned representations.

Unlike gender, the **middle layer** consistently achieved the highest accuracy for both SA1 and SA2. A possible explanation is that intermediate layers capture richer phonetic information, which is more useful for distinguishing regional pronunciation differences than the lower-level acoustic features found in the first layer.

---

# Repository Structure

```
wav2vec2-speaker-analysis/

├── extract_embeddings.py
├── probe.py
├── plot_results.py
├── README.md
├── requirements.txt
│
├── figures/
│   ├── gender_accuracy.png
│   ├── gender_confidence.png
│   └── dialect_accuracy.png
│
├── results/
│   ├── gender_probe_summary_sa1.csv
│   ├── gender_probe_summary_sa2.csv
│   ├── dialect_probe_summary_sa1.csv
│   └── dialect_probe_summary_sa2.csv
│
└── metadata/
    ├── metadata_sa1.csv
    └── metadata_sa2.csv
```

---

# How to Run

## 1. Install the required packages

```bash
pip install -r requirements.txt
```

---

## 2. Download the TIMIT dataset

Download the TIMIT dataset and organize the files according to the paths specified in the metadata files.

---

## 3. Extract embeddings

Run:

```bash
python extract_embeddings.py
```

This script extracts mean-pooled embeddings from the selected wav2vec2 layers.

---

## 4. Run the probing experiment

Run:

```bash
python probe.py
```

Choose the probing task by setting:

```python
TASK = "gender"
```

or

```python
TASK = "dialect"
```

The script automatically loads the appropriate labels, trains the Logistic Regression classifier, evaluates the results, and saves a summary CSV file.

---

## 5. Generate the figures

Run:

```bash
python plot_results.py
```

The generated figures are saved in the `figures/` folder.
