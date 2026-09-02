# Notice and attribution

This repository is a derivative modernization of **FaceOcc: A Diverse, High-quality Face Occlusion Dataset for Human Face Extraction** and its reference implementation.

## Original work

The FaceOcc dataset, original face-extraction method, baseline architecture and training objective originate from the original authors:

- Xiangnan Yin and Liming Chen, *FaceOcc: A Diverse, High-quality Face Occlusion Dataset for Human Face Extraction*, TAIMA 2022 / arXiv:2201.08425.
- Original implementation: https://github.com/face3d0725/FaceExtraction

The original software copyright notice is preserved verbatim in `LICENSE`:

> Copyright (c) 2024 Xiangnan Yin (ECL Liris), Di Huang (Beihang University), Liming Chen (ECL Liris)

## Modernization

Modernization and additional implementation work in this derivative repository:

> Copyright (c) 2026 Mert Akın (@mertakinstd)

These modifications are distributed under the same MIT License as the upstream software unless a file states otherwise.

The modernization includes current runtime/environment support, reproducibility controls, automated preparation and training entrypoints, safe checkpoint serialization, expanded diagnostics, and corrections to legacy implementation semantics. These changes do not transfer ownership of the FaceOcc dataset, the original method, or third-party datasets and artifacts.

## Third-party data and software

Third-party datasets, pretrained artifacts, and external projects remain subject to their respective licenses, terms, and citation requirements. In particular, the MIT License in this repository does not supersede the terms of FaceOcc, CelebAMask-HQ, 3DDFA_V2, or any other separately distributed resource.
