# What this fork changes

Based on upstream's `v5.6.0_roformer_add+directml` branch (Roformer support).
Everything below keeps upstream's output unless the section says otherwise.
`tools/` has the scripts used to check that.

## Faster

**Half precision (Settings → General → "Half Precision (Faster GPU)").**
Runs Roformer, MDX23C, Demucs and VR models with 16-bit autocast on NVIDIA
(CUDA) and Apple (MPS) GPUs. Off by default because the output changes very
slightly. If a model produces inf/NaN or fails in half precision, that call is
redone in full precision and the rest of the song stays in full precision.

How much the output changes, measured with CPU float16 emulation (difference
between half and full precision output, relative to the stem):

| Model | Difference |
|---|---|
| Mel-Roformer (viperx 1143) | 61 dB below the stem |
| MDX-Net HQ_3 (PyTorch path) | 60 dB below |
| VR 1_HP-UVR | 79 dB below |

For scale, that Roformer's SDR of 11.4 dB means its own separation error is
about 11 dB below the stem, so half precision adds an error roughly 50 dB
quieter than the one the model already makes. `tools/benchmark.py` measures
the real numbers on your GPU.

MDX-Net models run through ONNX Runtime (the default segment size on
NVIDIA) aren't affected by this setting.

**FlashAttention for Roformer.** Upstream disabled PyTorch's FlashAttention
kernel on every GPU except the A100 (a leftover from PyTorch 2.0). PyTorch now
picks the fastest kernel the GPU supports. Full precision output is unchanged;
the gain shows up with half precision on RTX 30-series and newer.

**Models stay loaded between songs.** Model files were read, built and moved to
the GPU for every song and every pass (main model, vocal splitter,
denoise/deverb models). The last two models are kept in RAM; only the one in
use sits on the GPU, so peak VRAM is unchanged. When processing finishes, the
models move back to RAM so the GPU is free while UVR is idle. For a 1 GB
Roformer this saves about 1.7 s per song with the file already in the OS
cache (building the network 1.1 s, reading the checkpoint 0.6 s), more when it
has to come off a slow disk.

**Decoded audio is reused.** Re-running a song with another model, the vocal
splitter and ensembles each decoded (and, for non-44.1 kHz files, resampled)
the input again. The last two decodes of an unchanged file are kept.

**Batch size settings work.**
- MDX-Net: the "Batch Size" option did nothing (every chunk was sent alone).
  Chunks are now batched. Output is bit-identical.
- Roformer / MDX23C: the GUI batch size was ignored in favor of the model
  config's value (usually 1). A GUI value above 1 now takes effect.

## Better output

**VR models: random noise near 22 kHz.** `cmb_spectrogram_to_wave` built each
band's spectrogram with `np.ndarray`, which leaves memory uninitialized. Bins
outside the band's crop range (the top 3 bins of the 4-band models' upper band,
right at Nyquist) kept whatever bytes were in memory and went into the inverse
STFT. The result changed from run to run and was sometimes huge: one upstream
test run produced samples near 1e9, which clip to full-scale noise in 16-bit
output. Those bins are now zero.

**Roformer / MDX23C crossfade with batch size > 1.** Every chunk in the final
batch skipped its fade-out, not just the last chunk. With batch size 1 (the
config default) this changes nothing.

## Smaller fixes

- `torch_directml` is optional, so NVIDIA/CPU installs don't need it.
- Removed unused `tkinter` imports from the Demucs code, so separation can run
  without a GUI (used by `tools/headless_separate.py`).
- Removed a Demucs v4 model that was built and immediately thrown away on
  every load.

## Requirements

Half precision works with any PyTorch 2.x. Roformer attention is fastest with
PyTorch 2.2 or newer (which ships FlashAttention-2) on an RTX 30-series
(Ampere) or newer GPU; other setups still get the half precision speedup
through PyTorch's other attention kernels. The upstream README's Windows
instructions install a CUDA 11.7 build (PyTorch 2.0), so for the full gain
install a newer one, for example:

```
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
```

## Checking it yourself

```
python tools/benchmark.py --upstream-dir <upstream checkout> --arch roformer \
    --model <model.ckpt> --config <model.yaml> --input song.flac --device cuda --batch 4
```

prints time, peak VRAM and the difference from upstream's output for
upstream, the fork, the fork with half precision, and half precision with a
bigger batch.
