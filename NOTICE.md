# Third-party notices

MLX Creator is MIT-licensed (see [LICENSE](LICENSE)). It vendors patched copies
of the following projects under `vendor/`, each of which is also MIT-licensed.
Patches are limited to keeping their **MLX inference paths torch-free** and
minor MLX-version compatibility; the original LICENSE file is retained in each
subtree.

| Project | Path | License | Upstream |
|---|---|---|---|
| MLX Examples (Flux) | `vendor/mlx-examples/` | MIT | https://github.com/ml-explore/mlx-examples |
| DiffusionKit (SD3/3.5) | `vendor/DiffusionKit/` | MIT | https://github.com/argmaxinc/DiffusionKit |
| mflux (Qwen-Image) | `vendor/mflux/` | MIT | https://github.com/filipstrand/mflux |

Runtime model engines also build on:

- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — music/TTS (ACE-Step)
- [mlx-video](https://github.com/Blaizzy/mlx-video) — video (Wan 2.2)
- [MLX](https://github.com/ml-explore/mlx) — the underlying array framework

Model **weights** are downloaded by the user at runtime and are governed by
their own licenses (e.g. FLUX.1, Stable Diffusion 3.5, Qwen-Image, ACE-Step,
Wan 2.2). MLX Creator does not redistribute any weights.
