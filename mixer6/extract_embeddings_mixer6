import torch
import torchaudio
import numpy as np
import pandas as pd
import time

from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModel


class EmbeddingExtractor:

    def __init__(
        self,
        model_name,
        metadata_file,
        output_dir,
        layers_to_use,
        path_column="clip_path",
        target_sample_rate=16000,
        n_test_samples=None,
        checkpoint_every=500,
        random_seed=42,
    ):
        self.model_name = model_name
        self.metadata_file = Path(metadata_file)
        self.output_dir = Path(output_dir)
        self.layers_to_use = layers_to_use
        self.path_column = path_column
        self.target_sample_rate = target_sample_rate
        self.checkpoint_every = checkpoint_every

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.metadata = pd.read_csv(self.metadata_file)

        if n_test_samples is not None:
            self.metadata = self.metadata.sample(
                n=min(n_test_samples, len(self.metadata)),
                random_state=random_seed,
            ).reset_index(drop=True)
            print(f"TEST MODE: using {len(self.metadata)} randomly sampled clips")
        else:
            print(f"Full run: {len(self.metadata)} clips")

        self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            output_hidden_states=True,
        )
        self.model.eval()
        self.model.to(self.device)

        # Keep metadata columns (speaker_id, gender, dialect_region, etc.)
        # aligned with the embeddings, saved alongside them so there's
        # never a row-order mismatch when loading embeddings + labels

        self.embeddings = {layer_name: [] for layer_name in self.layers_to_use}
        self.processed_rows = []  # metadata rows successfully processed, in order
        self.failed_rows = []     # rows that errored out, logged but not fatal

    def load_audio(self, wav_path):
        waveform, sr = torchaudio.load(str(wav_path))
        if sr != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.target_sample_rate,
            )
        return waveform.squeeze().numpy()

    def extract_one_file(self, wav_path):
        audio = self.load_audio(wav_path)

        # Guard against empty/near-empty clips, which would otherwise
        # crash the processor or produce a meaningless embedding.
        if audio.ndim == 0 or audio.shape[-1] < int(0.05 * self.target_sample_rate):
            raise ValueError(f"Clip too short to process: {wav_path}")

        inputs = self.processor(
            audio,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        pooled_embeddings = {}
        for layer_name, layer_index in self.layers_to_use.items():
            hidden = outputs.hidden_states[layer_index]
            pooled = hidden.mean(dim=1)
            pooled_embeddings[layer_name] = pooled.squeeze().cpu().numpy()

        return pooled_embeddings

    def run(self):
        total = len(self.metadata)
        start_time = time.time()

        for i, row in self.metadata.iterrows():
            wav_path = Path(row[self.path_column])

            if not wav_path.exists():
                print(f"  [SKIP] File not found: {wav_path}")
                self.failed_rows.append({**row.to_dict(), "error": "file_not_found"})
                continue

            try:
                pooled_embeddings = self.extract_one_file(wav_path)
            except Exception as e:
                #don't crash the whole run over one bad file -- log it
                print(f"  [FAIL] {wav_path}: {e}")
                self.failed_rows.append({**row.to_dict(), "error": str(e)})
                continue

            for layer_name, embedding in pooled_embeddings.items():
                self.embeddings[layer_name].append(embedding)
            self.processed_rows.append(row.to_dict())

            if (i + 1) % 100 == 0 or (i + 1) == total:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (total - (i + 1)) / rate if rate > 0 else float("inf")
                print(f"  [{i+1}/{total}] {rate:.1f} clips/sec, "
                      f"~{remaining/60:.1f} min remaining")

            if (i + 1) % self.checkpoint_every == 0:
                self.save_embeddings(checkpoint=True)

        self.save_embeddings(checkpoint=False)
        self.save_failure_log()

    def save_embeddings(self, checkpoint=False):
        safe_model_name = self.model_name.replace("/", "_")
        suffix = "_checkpoint" if checkpoint else ""

        for layer_name, vectors in self.embeddings.items():
            if len(vectors) == 0:
                continue
            X = np.vstack(vectors)
            output_file = (
                self.output_dir
                / f"embeddings_{safe_model_name}_{layer_name}{suffix}.npy"
            )
            np.save(output_file, X)
            if not checkpoint:
                print(f"  {layer_name}: {X.shape} saved to {output_file}")

        if self.processed_rows:
            processed_df = pd.DataFrame(self.processed_rows)
            processed_path = self.output_dir / f"metadata_processed{suffix}.csv"
            processed_df.to_csv(processed_path, index=False)
            if not checkpoint:
                print(f"  Processed metadata ({len(processed_df)} rows) "
                      f"saved to {processed_path}")

    def save_failure_log(self):
        if self.failed_rows:
            failed_df = pd.DataFrame(self.failed_rows)
            failed_path = self.output_dir / "failed_clips.csv"
            failed_df.to_csv(failed_path, index=False)
            print(f"\n  {len(failed_df)} clips failed -- logged to {failed_path}")
        else:
            print("\n  No failures.")



MODEL_CHOICE = "wav2vec2"
#options: "wav2vec2", "hubert", or "wavlm"

MODEL_CONFIGS = {
    "wav2vec2": {"name": "facebook/wav2vec2-base", "dir": "embeddings_mixer6"},
    "hubert": {"name": "facebook/hubert-base-ls960", "dir": "embeddings_mixer6_hubert"},
    "wavlm": {"name": "microsoft/wavlm-base", "dir": "embeddings_mixer6_wavlm"},
}

MODEL_NAME = MODEL_CONFIGS[MODEL_CHOICE]["name"]
OUTPUT_DIR = MODEL_CONFIGS[MODEL_CHOICE]["dir"]

METADATA_FILE = "metadata_mixer6_sentences_with_clips.csv"

LAYERS_TO_USE = {
    "first": 1,
    "middle": 6,
    "final": 12,
}

# Set to a small number (e.g. 50) for a quick test run first 
N_TEST_SAMPLES = 50


if __name__ == "__main__":
    extractor = EmbeddingExtractor(
        model_name=MODEL_NAME,
        metadata_file=METADATA_FILE,
        output_dir=OUTPUT_DIR,
        layers_to_use=LAYERS_TO_USE,
        n_test_samples=N_TEST_SAMPLES,
    )
    extractor.run()
