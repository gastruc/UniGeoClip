# UniGeoCLIP: Unified Geospatial Contrastive Learning

**EarthVision Workshop @ CVPR 2026**

**Authors:** [Guillaume Astruc](https://github.com/) · [Eduard Trulls](https://github.com/) · [Jan Hosang](https://github.com/) · [Loïc Landrieu](https://github.com/) · [Paul-Edouard Sarlin](https://github.com/)

*LASTIG, Univ Gustave Eiffel · IGN · ENSG · CNES · LIGM, École des Ponts ParisTech · Google Switzerland*

[![Paper](https://img.shields.io/badge/paper-coming%20soon-red)](#paper) [![Project Page](https://img.shields.io/badge/project-page-gastruc.github.io%2Funigeoclip-blue)](https://gastruc.github.io/unigeoclip) [![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## Overview

UniGeoCLIP is a multimodal contrastive framework that jointly aligns five geospatial modalities — aerial imagery, street-level imagery, Digital Surface Models (DSM), text descriptions, and geographic coordinates — into a single unified embedding space.

**This repository releases the location encoder**, a key architectural contribution of the paper.

---

## Location Encoder

The location encoder maps raw latitude/longitude coordinates to a rich D-dimensional embedding by combining multi-scale Random Fourier Features with self-attention.

**Key design choices:**
- K Fourier projections with increasing spectral bandwidths are computed and treated as **tokens**
- A transformer with register tokens processes them jointly, enabling **cross-scale interactions**
- The parameter count is **independent of the number of frequency scales**

This contrasts with GeoCLIP's design, where each scale is processed by a separate MLP and aggregated by averaging — a design that misses inter-scale dependencies and grows in parameters with K.

### Usage

To use the location encoder, import and instantiate the `LocationEncoder` from `location_encoder.py`. This module maps raw latitude/longitude coordinates (in degrees) to a fixed-dimensional embedding suitable for downstream models.

#### Example

```python
import torch
from location_encoder import LocationEncoder

# Example: batch of 3 locations [latitude, longitude] in degrees
locations = torch.tensor([
    [48.8584, 2.2945],   # Paris
    [40.6892, -74.0445], # New York
    [35.6586, 139.7454], # Tokyo
])

# Instantiate the encoder
encoder = LocationEncoder(
    sigma=[2**0, 2**4, 2**8, 2**12],  # frequency scales (default)
    embed_dim=768,                    # embedding dimension (default)
    n_registers=4,                    # number of transformer register tokens
    num_heads=12,                     # transformer heads
    mlp_ratio=4,                      # dimensionality multiplier for MLP layers
    depth=12                          # number of transformer blocks
)

# Forward pass
# Output: [batch_size, embedding_dim] tensor
embeddings = encoder(locations)
print(embeddings.shape)  # torch.Size([3, 768])
```

- Input tensor shape: `[batch_size, 2]` (`[latitude, longitude]` in degrees)
- Output tensor shape: `[batch_size, embed_dim]` (default: 768)

#### Notes
- Coordinates are internally projected using the Equal Earth map projection before encoding.
- Parameters can be adjusted for model size/speed tradeoffs.


### Pretrained weights

> Pretrained weights are not available in this release.

---

## Results

UniGeoCLIP's location encoder achieves a mean **R² = 57.0** across **27** downstream regression tasks (health, socio-economic, and environmental indicators), outperforming all contrastive baselines:

| Model | Mean R² |
|---|---:|
| SatCLIP | 30.1 |
| GeoCLIP | 49.8 |
| † GeoCLIP (retrained) | 51.6 |
| **UniGeoCLIP (ours)** | **57.0** |

† retrained on our dataset under identical conditions.

Increasing encoder depth yields consistent gains across all retrieval tasks:

| Depth | SV → GPS | Multimodal Ens. | OOD |
|---|---:|---:|---:|
| 0 (fixed RFF) | 55.0 | 10.2 | 13.6 |
| 4 | 73.1 | 40.7 | 30.0 |
| 8 | 73.1 | 44.0 | 27.8 |
| **12** | **74.4** | **47.0** | **29.2** |

See the [paper](#) and [project page](https://gastruc.github.io/unigeoclip) for full results across all tasks and modalities.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{astruc2026unigeoclip,
  title     = {UniGeoCLIP: Unified Geospatial Contrastive Learning},
  author    = {Astruc, Guillaume and Trulls, Eduard and Hosang, Jan
               and Landrieu, Lo{\"i}c and Sarlin, Paul-Edouard},
  booktitle = {EarthVision Workshop, CVPR},
  year      = {2026}
}
```

---

## License

The code in this repository is released under the [MIT License](LICENSE).

---

## Acknowledgements

This code build on top of prior work and code from [Geoclip](https://github.com/VicenteVivan/geo-clip/tree/main)

