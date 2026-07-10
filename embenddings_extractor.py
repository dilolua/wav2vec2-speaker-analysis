import torch
import torchaudio
import numpy as np
import pandas as pd

from pathlib import Path
from transformers import AutoProcessor, AutoModel


class EmbeddingExtractor:

    def __init__(
        self,
        model_name,
        metadata_file,
        data_dir,
        output_dir,
        layers_to_use,
        target_sample_rate=16000,
    ):
        #store configuration
        self.model_name = model_name
        self.metadata_file = Path(metadata_file)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.layers_to_use = layers_to_use
        self.target_sample_rate = target_sample_rate

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        #load metadata and pretrained model
        self.metadata = pd.read_csv(self.metadata_file)

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.model = AutoModel.from_pretrained(
            self.model_name,
            output_hidden_states=True,
        )

        self.model.eval()
        self.model.to(self.device)

        #store embeddings for each selected layer
        self.embeddings = {
            layer_name: [] for layer_name in self.layers_to_use
        }

    def load_audio(self, wav_path):
        """Load audio and resample to the target sampling rate if necessary."""

        waveform, sr = torchaudio.load(str(wav_path))

        if sr != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                sr,
                self.target_sample_rate,
            )

        return waveform.squeeze().numpy()

    def extract_one_file(self, wav_path):
        """Extract mean-pooled embeddings from the selected layers."""
        audio = self.load_audio(wav_path)

        inputs = self.processor(
            audio,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        pooled_embeddings = {}

        for layer_name, layer_index in self.layers_to_use.items():
            hidden = outputs.hidden_states[layer_index]
            #mean pooling over the time dimension
            pooled = hidden.mean(dim=1)

            pooled_embeddings[layer_name] = (
                pooled.squeeze().cpu().numpy()
            )

        return pooled_embeddings

    def run(self):
        """Extract embeddings for all audio files."""

        for _, row in self.metadata.iterrows():
            wav_path = self.data_dir / row["path"]

            if not wav_path.exists():
                raise FileNotFoundError(wav_path)

            pooled_embeddings = self.extract_one_file(wav_path)

            for layer_name, embedding in pooled_embeddings.items():
                self.embeddings[layer_name].append(embedding)

            print("done:", wav_path)

        self.save_embeddings()

    def save_embeddings(self):
        """Save embeddings for each layer as NumPy files."""

        safe_model_name = self.model_name.replace("/", "_")

        for layer_name, vectors in self.embeddings.items():
            X = np.vstack(vectors)

            output_file = (
                self.output_dir
                / f"embeddings_{safe_model_name}_{layer_name}.npy"
            )

            np.save(output_file, X)

            print(layer_name, X.shape, "saved to", output_file)


MODEL_NAME = "facebook/wav2vec2-base"
SENTENCE = "SA1"
#options: "SA1" or "SA2"
METADATA_FILE = f"timit_sa1_sa2/{SENTENCE}/metadata_{SENTENCE.lower()}.csv"
DATA_DIR = f"timit_sa1_sa2/{SENTENCE}"
OUTPUT_DIR ="embeddings"

#layers used for probing
LAYERS_TO_USE = {
    "first": 1,
    "middle": 6,
    "final": 12,
}


extractor = EmbeddingExtractor(
    model_name=MODEL_NAME,
    metadata_file=METADATA_FILE,
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    layers_to_use=LAYERS_TO_USE,
)

extractor.run()
