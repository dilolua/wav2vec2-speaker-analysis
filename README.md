# wav2vec2-speaker-analysis

Analysis of gender and dialect information encoded in **wav2vec2** representations using internal probing.

# Introduction / Motivation

Self-supervised speech models such as wav2vec 2.0 [1] learn speech representations directly from raw audio, without requiring manual transcriptions during pretraining. Despite achieving strong performance on downstream tasks such as automatic speech recognition and speaker-related tasks, it remains unclear what kind of information these models encode internally.

**Probing** has become the standard methodology for investigating this question. In a probing experiment, the pretrained model is kept fixed, and a lightweight classifier — typically logistic regression — is trained on the extracted embeddings to predict a property of interest, such as phoneme identity, speaker gender, or language. Strong probe performance is taken as evidence that the corresponding information is encoded in the representations. Across the literature, probing studies converge on a consistent finding: self-supervised speech models encode substantially more than the phonetic content required for transcription.

Pasad et al. (2021) conducted a layer-wise analysis of wav2vec2 using canonical correlation analysis, showing that different types of information are concentrated at different depths of the network. Early layers align most closely with low-level acoustic features, while phonetic information, word identity, and word meaning become increasingly prominent in later and central layers, with the most contextual information peaking around layers 7–8 of wav2vec2-base [2]. This acoustic-to-semantic hierarchy motivates the comparison of early, middle, and final layers in probing studies, rather than assuming a single layer to be uniformly optimal.

De Seyssel et al. (2022) probed Contrastive Predictive Coding (CPC) representations, a closely related family of self-supervised speech models, for phone class, gender, and language, comparing monolingual and bilingual models. They found that gender and phonetic class information were present in both model types, whereas language identity was strongly decodable only in the bilingual model, with a comparatively weaker and more diffuse signal in the monolingual models [3]. This constituted one of the earliest demonstrations that self-supervised speech representations encode speaker-level information, such as gender, despite not being explicitly trained to do so.

Glazer et al. (2025) extended this line of inquiry to modern automatic speech recognition and audio-language models, namely Whisper and Qwen2-Audio, applying linear probing alongside interpretability techniques such as logit lens and activation patching. Their results show that internal representations encode speaker gender (with the clearest linear signal in deeper encoder layers), accent, acoustic recording conditions such as clean versus noisy audio, and semantic information — none of which is required for, or directly visible in, the final transcription [4]. These findings support the view that models trained solely for transcription nonetheless represent much richer speaker- and context-level information internally.

Most gender and dialect probing studies, including [3] and [4], draw probe training data from utterances with varying lexical content - different sentences and different speakers. This leaves open the possibility that probe performance partly reflects correlations with spoken content rather than purely speaker-related characteristics.

This project addresses this limitation by controlling spoken content directly, using the two TIMIT "SA" sentences (SA1 and SA2). Because the text is identical across speakers, any observed difference in probe performance can be attributed to how a given sentence was spoken, and by whom, rather than to what was said. SA1 and SA2 are analyzed separately, allowing an additional check on whether observed patterns are sentence-specific or replicate across two distinct, fixed texts.

By combining this content control with the layer-wise probing methodology established in prior work, this project aims to provide a clearer account of how and where facebook/wav2vec2-base represents speaker gender and dialect, disentangled as far as possible from the influence of lexical content.


# Project Overview

This project investigates whether a pretrained wav2vec2 model encodes information about a speaker's gender and dialect (accent).

Rather than directly comparing speech embeddings, the project uses an internal probing approach. A simple Logistic Regression classifier is trained on embeddings extracted from different layers of wav2vec2. If the classifier performs well, this indicates that the corresponding speaker information is encoded in the learned representations.

The analysis was performed separately for the two common TIMIT sentences (SA1 and SA2) in order to compare the results across different utterances while keeping the spoken content identical.

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
- Number of mistakes
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
- Number of mistakes

---

# Results / Discussion

## Gender

The classifier reached **99.4% accuracy** for SA1 at all three layers, and for SA2 reached **99.4% at the first layer** and **98.2% at the middle and final layers**. This shows that gender information is strongly present in wav2vec2 embeddings, across all layers tested, with only a small drop in the deeper layers for SA2. This is consistent with De Seyssel et al. (2022), who found gender information reliably decodable from CPC representations regardless of whether the model was monolingual or bilingual [3]. The result extends this observation to wav2vec2 and to a content-controlled setting, where gender remains just as strongly decodable even when spoken content is held fixed.

The **first layer** gave both the highest (or equal-highest) accuracy and the highest confidence scores for both sentences (mean confidence: 0.963 for SA1, 0.957 for SA2, vs. around 0.93 at middle and final layers). This fits the idea that low-level acoustic cues (like pitch and vocal tract properties), which are useful for telling gender apart, are stronger in early layers. This is worth noting in contrast to Glazer et al. (2025), who found gender to be most strongly decodable in *deeper* layers of the Whisper encoder, peaking at layer 25 of 32 with 94.6% accuracy [4]. The difference may come from the two models being trained differently: wav2vec2-base is trained purely self-supervised on raw audio, while Whisper's encoder is trained with supervision from transcriptions, which may push acoustic speaker cues like gender further into later layers as the model builds toward the transcription task. This is a plausible explanation rather than a confirmed one, and would need direct comparison between models to verify.

## Dialect

Dialect classification was much harder, with accuracy ranging from **19% to 36%** across layers and sentences (SA1: 23.2%–35.7%; SA2: 19.0%–29.8%). This is well above the **12.5%** chance level for 8 equally likely classes, but far below gender-level accuracy. This suggests dialect information is present, but less strongly and less "linearly" separable than gender in this setup. Interestingly, this contrasts with Glazer et al. (2025), who found accent to be *more* strongly decodable than gender in Whisper (97.0% peak accuracy for a 4-way accent classifier, vs. 94.6% for gender) [4]. A plausible reason for the difference is task granularity: Glazer et al. classified four broad, geographically distant accent groups (New Zealand, Welsh Valleys, South African, Indian), while this project classifies eight finer-grained US regional dialects from TIMIT, using a single fixed sentence per probe rather than full natural utterances. Finer-grained, content-controlled dialect classification may simply be a harder linear decoding problem than broad accent classification from varied speech. This is a reasonable interpretation given the setup differences, but it is not something directly tested here.

It is also worth noting that dialect accuracy was consistently higher for SA1 than SA2 at every layer (e.g. middle layer: 35.7% vs. 29.8%). Since both sentences use the same model, layers, and method, this indicates that even fixed-content probing is not fully independent of which specific sentence is used — a relevant observation given this project's original motivation of isolating content effects.

The **middle layer** gave the best accuracy for both SA1 and SA2. This may be because middle layers capture more detailed phonetic information, which is useful for telling regional pronunciation differences apart. This lines up with Pasad et al. (2021), who found phonetic information peaking in the central layers of wav2vec2-base [2] — since dialect is expressed largely through pronunciation, a phonetically-driven property, it is plausible that its clearest signal would appear where phonetic information is most concentrated.


---

# Repository Structure

```
wav2vec2-speaker-analysis/

├── embenddings_extractor.py
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
python embenddings_extractor.py
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

# References

[1] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. 2020. *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.* NeurIPS 2020. [arXiv:2006.11477](https://arxiv.org/abs/2006.11477)

[2] Ankita Pasad, Ju-Chieh Chou, and Karen Livescu. 2021. *Layer-wise Analysis of a Self-supervised Speech Representation Model.* IEEE ASRU 2021. [arXiv:2107.04734](https://arxiv.org/abs/2107.04734)

[3] Maureen de Seyssel, Marvin Lavechin, Yossi Adi, Emmanuel Dupoux, and Guillaume Wisniewski. 2022. *Probing phoneme, language and speaker information in unsupervised speech representations.* Interspeech 2022. [arXiv:2203.16193](https://arxiv.org/abs/2203.16193)

[4] Neta Glazer, Yael Segal-Feldman, Hilit Segev, Aviv Shamsian, Asaf Buchnick, Gill Hetz, Ethan Fetaya, Joseph Keshet, and Aviv Navon. 2025. *Beyond Transcription: Mechanistic Interpretability in ASR.* [arXiv:2508.15882](https://arxiv.org/abs/2508.15882)

**Note on TIMIT SA sentences:** SA1 and SA2 are TIMIT's two special "dialect" sentences. Every speaker in the corpus reads these two sentences, and they were chosen specifically to show dialect differences. 

