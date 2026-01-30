# vibesimatcher
System to detect plagiarism in music

## melody_plag pipeline

The repo now includes a deterministic melody-token similarity pipeline under
`src/melody_plag`. You can run it via:

```bash
python -m melody_plag path/to/audio.wav --out /tmp/melody_plag.json
```

Notes:
- The pipeline expects Demucs to be installed (Python package or CLI `demucs`).
- RMVPE weights are required via `RMVPE_MODEL_PATH` and a compatible RMVPE
  implementation must be importable.
