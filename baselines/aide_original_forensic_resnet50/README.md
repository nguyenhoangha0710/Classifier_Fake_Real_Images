# AIDE original — Forensic-only ResNet50

Đây là **nhánh forensic của AIDE gốc chạy độc lập**:

- chọn 2 patch low-frequency và 2 patch high-frequency bằng DCT;
- áp dụng SRM filter;
- phân loại bằng ResNet50.

Mô hình này không có nhánh semantic OpenCLIP và không dùng NPR.

## File chính

- `train_aide_original_forensic_resnet50_modal.py`: train/test trên Modal.
- `artifacts/checkpoints/model.pt`: checkpoint canonical của forensic-only
  (`SHA256 57024d5c0256869f8855a081abf27e6e882f5bb6de61659c5ef6a28afe8d5b39`).
