# Jetson Environment Setup

Per-platform recipes for reproducing the software environment the corpus
was measured with. Reviewers using the provided boards can skip this page
— the environment there is pre-provisioned and activates on login.

## Validated configurations

| Board | JetPack | Python | PyTorch | Wheel source |
|---|---|---|---|---|
| AGX Xavier | 5.1 | 3.8 | `2.1.0a0+41361538.nv23.06` | [NVIDIA JetPack redist](https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/) |
| AGX Orin | 6.2 | 3.10 | `2.3.0` | [jetson-ai-lab index](https://pypi.jetson-ai-lab.io/jp6/cu126) |
| AGX Thor | 7.1 | 3.12 | `2.11.0` | [jetson-ai-lab index](https://pypi.jetson-ai-lab.io/sbsa/cu130) |

The exact wheels used for the corpus are also archived as assets on this
repository's GitHub Releases page (see `docs/WHEELS_NOTICE.md` for
provenance and licensing). Verify downloads against these checksums:

```
5112e2ef5051f1003ae2ffb545ae596377df66979d62a954d774f18009064dbe  torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl  (Xavier)
5feee5f5d1a75229eb5290c6fcdc534658dd8b99e55e4a49e8f7c509c313d52d  torch-2.3.0-cp310-cp310-linux_aarch64.whl                   (Orin)
e859c4c1a6c0d9661262c5c940f7495ba5abaf15243a94cb23f98e27746a4869  torch-2.11.0-cp312-cp312-linux_aarch64.whl                  (Thor)
```

## Setup steps (any board)

```bash
# 1. Create a CLOSED virtualenv - do not use --system-site-packages.
#    (System-wide scipy/numpy/pandas from apt can leak into the venv and
#    conflict with the pinned versions.)
python3 -m venv ~/hydra-venv
source ~/hydra-venv/bin/activate
pip install -U pip

# 2. Install PyTorch FIRST, from the platform wheel for your board
#    (table above). Never install torch from PyPI on Jetson - the PyPI
#    aarch64 wheel is CPU-only.
pip install /path/to/torch-<version>-<abi>-linux_aarch64.whl
# or, from an index:  pip install torch==<version> --index-url <index-url>

# 3. Install the remaining pinned dependencies for your platform:
pip install -r scripts/requirements_xavier.txt   # or _orin / _thor

# 4. Verify:
python3 -c "import torch, transformers, pandas; \
  print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
# expect: <torch version>  True  <pinned transformers version>
```

## Board configuration

All corpus measurements were taken with the board in **MAXN power mode**
(`sudo nvpmodel -m 0`), with DVFS left active (no `jetson_clocks`). Set
the same mode before comparing measurements against the corpus.

## Notes

- The llama.cpp profiler is independent of this Python environment; see
  the README's build section (`scripts/build_llamacpp_profiler.sh`).
- `tegrastats` requires sudo; grant it narrowly, e.g. a sudoers entry
  limited to `/usr/bin/tegrastats`.
