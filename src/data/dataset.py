import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Callable, Optional
from abc import ABC, abstractmethod
import os
import json


class Processor(ABC):
    def __init__(self, keys: list[str] = ["audio"]):
        self.keys = keys

    def __call__(self, x):
        for key in self.keys:
            x.update(**self._process(x, key))
        return x

    def _process(self, x, key):
        pass


class ProcessorChain:
    def __init__(self, processors: list[Callable]):
        self.processors = processors

    def __call__(self, x):
        for processor in self.processors:
            x = processor(x)
        return x


class TimeFrequencyMask(Processor):
    def __init__(
        self,
        keys: list[str] = ["audio"],
        time_mask_param: int = 100,
        time_p=0.8,
        freq_mask_param: int = 100,
    ):
        super().__init__(keys=keys)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param, p=time_p)

    def _process(self, x, key="audio"):
        audio = x.get(key)
        out_ = self.time_mask(audio)
        out_ = self.freq_mask(out_)
        return {key: out_}


class Truncate(Processor):
    def __init__(self, keys: list[str] = ["audio"], n_samples: int = 48000):
        super().__init__(keys=keys)
        self.n_samples = n_samples

    def _process(self, x, key="audio"):
        audio = x.get(key)
        return {key: audio[: self.n_samples]}


class MelSpectrogram(Processor):

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: float = None,
        win_length: int = None,
        power: float = 2.0,
        keys: list[str] = ["audio"],
    ):
        super().__init__(keys=keys)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.win_length = win_length
        self.power = power

        # Create the torchaudio MelSpectrogram transform
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            win_length=win_length,
            power=power,
        )

    def _process(self, x, key="audio"):
        """Apply mel spectrogram transformation to input audio."""

        views = x.get(key)
        # Ensure input is a tensor
        if not isinstance(views, torch.Tensor):
            views = torch.tensor(views, dtype=torch.float32)

        # Apply mel spectrogram transform
        mel_spec = self.mel(views)

        # Convert to log scale (common practice for mel spectrograms)
        mel_spec = torch.log(mel_spec + 1e-8)

        return {key: mel_spec}


class MultiView(Processor):
    def __init__(
        self, view_samples, strategy: str = "same_view", keys: list[str] = ["audio"]
    ):
        super().__init__(keys=keys)
        self.view_samples = view_samples
        self.strategy = strategy

    def _process(self, x, key):
        audio = x.get(key)
        if self.strategy == "same_view":
            view_1, view_2 = self._get_same_view(audio)
        elif self.strategy == "adjacent_view":
            view_1, view_2 = self._get_adjacent_view(audio)
        elif self.strategy == "random_view":
            view_1, view_2 = self._get_random_view(audio)

        # check for nans
        if torch.isnan(view_1).any() or torch.isnan(view_2).any():
            raise ValueError("NaNs found in views")

        return {"view_1": view_1, "view_2": view_2}

    def _get_same_view(self, audio):
        start_idx = torch.randint(0, audio.shape[0] - self.view_samples, (1,))
        view_1 = audio[start_idx : start_idx + self.view_samples]
        return view_1, view_1

    def _get_adjacent_view(self, audio):
        start_idx = torch.randint(0, audio.shape[0] - self.view_samples, (1,))
        view_1 = audio[start_idx : start_idx + self.view_samples]
        view_2 = audio[
            start_idx + self.view_samples : start_idx + self.view_samples * 2
        ]
        return view_1, view_2

    def _get_random_view(self, audio):
        start_idx_1 = torch.randint(0, audio.shape[0] - self.view_samples, (1,))
        start_idx_2 = torch.randint(0, audio.shape[0] - self.view_samples, (1,))
        view_1 = audio[start_idx_1 : start_idx_1 + self.view_samples]
        view_2 = audio[start_idx_2 : start_idx_2 + self.view_samples]
        return view_1, view_2


class BaseDataset(Dataset):
    def __init__(
        self,
        audio_dir: Optional[str] = None,
        metadata_path: Optional[str] = None,
        sample_rate: int = 16000,
        mono: bool = True,
        processors: list[Callable] = [
            MultiView(view_samples=48000, strategy="same_view"),
        ],
        labels_: bool = False,
        n_samples: int = 48000,
    ):
        self.audio_dir = audio_dir
        self.metadata_path = metadata_path
        self.sample_rate = sample_rate
        self.mono = mono
        self.processors = processors
        self.labels_ = labels_
        # Load metadata (you'll need to implement this based on your data format)
        self.metadata = self._load_metadata()
        self.n_samples = n_samples

        # Create processor chain
        self.processor_chain = ProcessorChain(processors)

    def _load_metadata(self):
        """Load metadata from the specified path. Implement based on your data format."""

        if self.metadata_path is None or not os.path.exists(self.metadata_path):
            return self._ssl_load_metadata()
        with open(self.metadata_path, "r") as f:
            return json.load(f)

    def _ssl_load_metadata(self):
        audio_files = [f for f in os.listdir(self.audio_dir)]
        metadata = []
        for audio_file in audio_files:
            metadata.append(
                {
                    "file_path": os.path.join(self.audio_dir, audio_file),
                }
            )
        return metadata

    def __len__(self):
        return len(self.metadata)


class Giantsteps(BaseDataset):
    def __init__(
        self,
        audio_dir: Optional[str] = "data/giantsteps/audio",
        metadata_path: Optional[str] = "data/giantsteps/metadata.json",
        sample_rate: int = 16000,
        n_samples: int = 48000,
        mono: bool = True,
        processors: list[Callable] = [
            MultiView(view_samples=48000, strategy="same_view", keys=["audio"]),
        ],
        labels_: bool = False,
    ):
        super().__init__(
            audio_dir=audio_dir,
            metadata_path=metadata_path,
            sample_rate=sample_rate,
            mono=mono,
            processors=processors,
            labels_=labels_,
            n_samples=n_samples,
        )
        self.labels_ = labels_

    def __getitem__(self, idx):
        item = self.metadata[idx]
        audio_path = item["file_path"]
        label_ = item.get("label_id")
        key = item.get("key")
        audio, sr = torchaudio.load(audio_path)
        if self.mono:
            audio = audio.mean(dim=0)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        sample = {
            "audio": audio,
        }

        if self.labels_:
            sample["label_"] = label_
            sample["name"] = key

        sample = self.processor_chain(sample)

        # random truncate audio to n_samples
        start_idx = torch.randint(0, audio.shape[0] - self.n_samples, (1,))
        sample["audio"] = sample["audio"][start_idx : start_idx + self.n_samples]

        return sample
