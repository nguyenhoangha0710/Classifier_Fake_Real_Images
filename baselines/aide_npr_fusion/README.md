# AIDE + NPR fusion

Đây là mô hình thí nghiệm **thay nhánh forensic DCT/SRM của AIDE bằng NPR-ResNet18**.
Nó không phải AIDE gốc.

## Hai cấu hình riêng biệt

| File | Semantic | NPR | Vector fusion |
|---|---:|---:|---:|
| `train_aide_npr_fusion_256_2048_modal.py` | 256-D | 2048-D | 2304-D |
| `train_aide_npr_fusion_256_256_modal.py` | 256-D | projection 256-D | 512-D |

Artifact cũng được tách theo đúng cấu hình:

- `artifacts/runs/fusion_256_2048/aide_npr_fusion_256_2048_v1/`
- `artifacts/runs/fusion_256_256/aide_npr_fusion_256_256_v1/`

Checkpoint `fusion_256_256/.../checkpoints/best/model.pt` có SHA256
`570db2fd43af50fa1cd289e08915c36420ddfe689ccd17f0517c97be3072a51f`.
Nó thuộc AIDE + NPR cân bằng, không được dùng làm checkpoint AIDE gốc.
