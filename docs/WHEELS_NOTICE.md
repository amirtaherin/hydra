# Archived PyTorch Wheels — Provenance and Licensing

The GitHub Release assets of this repository include the three PyTorch
wheels used to produce the released corpus, archived so the validated
environment remains reproducible.

| Wheel | Board | Obtained from |
|---|---|---|
| `torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl` | AGX Xavier (JetPack 5) | NVIDIA JetPack redist (`developer.download.nvidia.com/compute/redist/jp/v512/pytorch/`) |
| `torch-2.3.0-cp310-cp310-linux_aarch64.whl` | AGX Orin (JetPack 6) | NVIDIA-hosted build announced on the NVIDIA Developer Forums ("PyTorch for Jetson") |
| `torch-2.11.0-cp312-cp312-linux_aarch64.whl` | AGX Thor (JetPack 7) | jetson-ai-lab index (`pypi.jetson-ai-lab.io/sbsa/cu130`) |

The wheels are redistributed **unmodified**. PyTorch is licensed under the
BSD-3-Clause-style license; each wheel contains its own `LICENSE` and
`NOTICE` files under `torch-<version>.dist-info/`, which apply to the
respective wheel. No CUDA toolkit or driver components are bundled; the
wheels link against the CUDA runtime installed by JetPack on the target
board.

SHA-256 checksums:

```
5112e2ef5051f1003ae2ffb545ae596377df66979d62a954d774f18009064dbe  torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
5feee5f5d1a75229eb5290c6fcdc534658dd8b99e55e4a49e8f7c509c313d52d  torch-2.3.0-cp310-cp310-linux_aarch64.whl
e859c4c1a6c0d9661262c5c940f7495ba5abaf15243a94cb23f98e27746a4869  torch-2.11.0-cp312-cp312-linux_aarch64.whl
```
