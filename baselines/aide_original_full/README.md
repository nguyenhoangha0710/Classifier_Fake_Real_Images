# AIDE original — Full model

Đây là **AIDE gốc đầy đủ**, gồm hai nhánh:

- semantic: frozen OpenCLIP ConvNeXt-XXLarge;
- forensic: chọn patch bằng DCT, SRM filter và CNN/ResNet50.

Không có NPR trong mô hình này.

## File chính

- `train_aide_original_full_tiny_commfor_kaggle.py`: train trên Kaggle.
- `test_aide_original_full_commfor_exact_kaggle.py`: test cohort CommFor cố định trên Kaggle.
- `test_aide_original_full_commfor_unseen_modal.py`: test CommFor trên Modal.
- `artifacts/checkpoints/model_trainable.pt`: checkpoint canonical của AIDE gốc Full
  (`SHA256 62848895b44255d6a0567754c503a05807ed0a29b8046f3c6956a049844e47c3`).

Các output cũ đã chạy nằm trong `artifacts/`; chúng không phải source code để chạy mới.
