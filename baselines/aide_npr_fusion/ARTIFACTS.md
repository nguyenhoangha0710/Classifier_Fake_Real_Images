# AIDE + NPR fusion artifacts

```text
artifacts/
|-- runs/
|   |-- fusion_256_2048/aide_npr_fusion_256_2048_v1/
|   |   |-- checkpoints/
|   |   |-- metrics/
|   |   |-- predictions/
|   |   `-- provenance/
|   `-- fusion_256_256/aide_npr_fusion_256_256_v1/
|       `-- checkpoints/best/model.pt
`-- download_attempts/
```

`fusion_256_2048` ghép semantic 256-D với NPR 2048-D.
`fusion_256_256` cân bằng hai nhánh 256-D và dùng LayerNorm cho từng nhánh.

Run `fusion_256_256` đã train xong nhưng test bị dừng vì workspace cũ đạt
spend limit; checkpoint best đã được tải về. Đây không phải checkpoint của
AIDE gốc.

Tên `run_id` nằm bên trong một số metadata vẫn là ID lịch sử do Modal tạo ra;
tên thư mục cục bộ phía trên là tên rõ nghĩa dùng để tránh nhầm mô hình.
