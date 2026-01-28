from pathlib import Path
from typing import Any, Optional


import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T


class SongFormer:
    """
    Wrapper around ASLP-lab/SongFormer.

    It:
      - loads the model once
      - loads audio from path
      - runs segmentation
      - writes chunk .wav files into out_dir
      - returns list of chunk paths
    """

    def __init__(self, model_id: str = "ASLP-lab/SongFormer"):
        print(f"Loading SongFormer model: {model_id}")
        self.model_id = model_id
        self.model = self._load_model(model_id)

        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Model expects 24kHz waveform input
        self.target_sr = 24000
        self.resampler: Optional[T.Resample] = None

    def chunk_to_dir(
        self,
        audio_path: str | Path,
        out_dir: str | Path,
    ) -> list[Path]:
        """
        Splits audio with SongFormer and saves chunks to out_dir.

        :param audio_path: Path to the input audio.
        :param out_dir: Directory to write chunks into.
        :param prefix: Chunk filename prefix.
        :return: (chunk_paths, segments, original_sr)
        """
        audio_path = Path(audio_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        audio, sr = sf.read(audio_path)  # (T,) or (T, C)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float32, copy=False)

        segments = self._infer_segments(audio=audio, sr=sr)

        chunk_paths: list[Path] = []
        for i, seg in enumerate(segments):
            start_sec = float(seg["start"])
            end_sec = float(seg["end"])

            start_i = int(round(start_sec * sr))
            end_i = int(round(end_sec * sr))

            start_i = max(0, min(audio.shape[0], start_i))
            end_i = max(0, min(audio.shape[0], end_i))

            if end_i <= start_i:
                continue

            chunk = audio[start_i:end_i].astype(np.float32, copy=False)

            out_path = out_dir / f"{i:03d}_{seg['label']}.wav"
            sf.write(out_path, chunk, sr)
            print(f"Saved chunk: {out_path}")
            chunk_paths.append(out_path)

        # Fallback: if SongFormer produced nothing, write the whole file as chunk_000
        if not chunk_paths:
            out_path = out_dir / "000_full.wav"
            sf.write(out_path, audio, sr)
            chunk_paths = [out_path]
            dur_sec = float(audio.shape[0]) / float(sr)
            segments = [{"start": 0.0, "end": dur_sec, "label": "full"}]

        return chunk_paths

    # -------------------------
    # Internals
    # -------------------------

    def _load_model(self, model_id: str):
        try:
            import sys
            import importlib.util
            import importlib.machinery
            from pathlib import Path
            from transformers import AutoModel

            import torch

            torch.set_default_device("cpu")

            SONGFORMER_DIR = Path(
                "/home/max/.cache/huggingface/hub/"
                "models--ASLP-lab--SongFormer/snapshots/"
                "5ac5227fccf286519464fdf211e15b606898408e"
            )

            # 1. Make snapshot visible
            sys.path.insert(0, str(SONGFORMER_DIR))

            # 2. Register namespace packages (dirs WITHOUT __init__.py)
            def register_namespace(pkg_name: str, pkg_dir: Path):
                if pkg_name in sys.modules:
                    return

                spec = importlib.machinery.ModuleSpec(
                    name=pkg_name,
                    loader=None,
                    is_package=True,
                )
                module = importlib.util.module_from_spec(spec)
                module.__path__ = [str(pkg_dir)]
                sys.modules[pkg_name] = module

            register_namespace("dataset", SONGFORMER_DIR / "dataset")
            register_namespace("postprocessing", SONGFORMER_DIR / "postprocessing")
            register_namespace("musicfm", SONGFORMER_DIR / "musicfm")

            # 3. Preload required top-level modules
            preload_files = [
                "configuration_songformer.py",
                "model_config.py",
                "model.py",
                "modeling_songformer.py",
            ]

            for fname in preload_files:
                name = fname.replace(".py", "")
                if name in sys.modules:
                    continue

                path = SONGFORMER_DIR / fname
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[name] = module

            # 4. Load model via LOCAL PATH
            return AutoModel.from_pretrained(
                SONGFORMER_DIR,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
                device_map=None,
            )
        except Exception:
            raise
            # from huggingface_hub import snapshot_download
            # import os
            # import sys

            # local_dir = snapshot_download(
            #     repo_id=model_id,
            #     repo_type="model",
            #     local_dir_use_symlinks=False,
            #     resume_download=True,
            #     allow_patterns="*",
            #     ignore_patterns=["SongFormer.pt", "SongFormer.safetensors"],
            # )
            # sys.path.append(local_dir)
            # os.environ["SONGFORMER_LOCAL_DIR"] = local_dir
            # print(f"Loaded SongFormer model from local directory: {local_dir}")

            # return AutoModel.from_pretrained(
            #     local_dir, trust_remote_code=True, low_cpu_mem_usage=False
            # )

    def _infer_segments(self, *, audio: np.ndarray, sr: int) -> list[dict[str, Any]]:
        """
        Resamples to 24kHz, runs model, returns sorted segments:
          [{"start": float_sec, "end": float_sec, "label": str}, ...]
        """
        audio_24k = self._resample_to_target(audio=audio, sr=sr)

        wav = torch.from_numpy(audio_24k).to(self.device)

        with torch.inference_mode():
            result = self.model(wav)

        if not isinstance(result, list) or (result and not isinstance(result[0], dict)):
            raise RuntimeError(f"Unexpected SongFormer output format: {type(result)}")

        segments: list[dict[str, Any]] = []
        for s in result:
            if "start" not in s or "end" not in s:
                continue
            segments.append(
                {
                    "start": float(s["start"]),
                    "end": float(s["end"]),
                    "label": str(s.get("label", "seg")),
                }
            )

        segments.sort(key=lambda x: (x["start"], x["end"]))

        if not segments:
            dur_sec = float(audio.shape[0]) / float(sr)
            segments = [{"start": 0.0, "end": dur_sec, "label": "full"}]

        return segments

    def _resample_to_target(self, *, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == self.target_sr:
            return audio.astype(np.float32, copy=False)

        audio_t = torch.from_numpy(audio).float()

        if self.resampler is None or self.resampler.orig_freq != sr:
            self.resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)

        audio_rs = self.resampler(audio_t)
        return audio_rs.cpu().numpy().astype(np.float32, copy=False)
