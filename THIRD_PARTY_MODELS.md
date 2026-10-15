# Third-party models

Models are never vendored or redistributed. Users download weights themselves; the registry
(`models/registry.py`) logs each license on first use (`model_license_shown`), pins each model to
a full commit hash and to the size and sha256 of its `.safetensors` weights and of `config.json`,
and refuses a model whose license is outside `DATACOPILOT_ALLOWED_LICENSES` before anything is
downloaded.

| Model | Source | License | Commercial use | Notes |
|---|---|---|---|---|
| `dinov2_small` (DINOv2 ViT-S/14) | [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) @ `ed25f3a31f01632728cabb09d1542f84ab7b0056` | Apache-2.0 | Yes | Embedder, 384 dimensions. Weights are downloaded by the user (not gated), verified against the pinned sha256, loaded as safetensors without `trust_remote_code`. |
| `dinov2_base` (DINOv2 ViT-B/14) | [facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base) @ `f9e44c814b77203eaa57a6bdbbd535f21ede1415` | Apache-2.0 | Yes | Embedder, 768 dimensions. Same handling as above. |
| `siglip2_base` (SigLIP 2 B/16, 224 px) | [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) @ `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | Apache-2.0 | Yes | Embedder (image tower only is used), 768 dimensions; the checkpoint also contains the text tower and is about 1.5 GB. Same handling as above. |

License text above is the `license` field of each model card at the pinned revision; the
Apache-2.0 terms apply to the weights as published there. If an upstream card changes its
license, the pinned revision keeps the terms it had when it was reviewed.

Reminders: SAM weights use the SAM License and DINOv3 the DINOv3 License (gated Hugging Face
repos, user accepts terms). Ultralytics/YOLOE and Cleanlab are AGPL: separate plugin or optional extra only.
