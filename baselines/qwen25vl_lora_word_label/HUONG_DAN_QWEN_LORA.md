# Hướng Dẫn Inference Qwen2.5-VL + LoRA

File dùng để inference:

```text
baselines/qwen25vl_lora_word_label/infer_qwen25vl_lora_commfor_modal.py
baselines/qwen25vl_lora_word_label/infer_qwen25vl_lora_commfor_real_only_modal.py
```

## 1. Folder LoRA Cần Có

Folder LoRA mẫu:

```text
20260910_003026/
  adapter/
    adapter_model.safetensors
    adapter_config.json
    tokenizer.json
    tokenizer_config.json
    preprocessor_config.json
    ...
```

Hai file quan trọng nhất:

```text
adapter_model.safetensors  -> trọng số LoRA đã fine-tune
adapter_config.json        -> cấu hình để gắn LoRA vào Qwen
```

## 2. LoRA Được Gắn Vào Đâu

Base model:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

LoRA được gắn vào các module:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Trong đó:

```text
q_proj, k_proj, v_proj, o_proj -> attention
gate_proj, up_proj, down_proj  -> MLP/feed-forward
```

## 3. Gắn LoRA Như Thế Nào

Code load base model rồi gắn LoRA:

```python
import torch
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

adapter_dir = "path/to/20260910_003026/adapter"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(adapter_dir)

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
model.eval()
```

## 4. Cách Inference

Prompt dùng cho mỗi ảnh:

```text
Look at the image and decide whether it is real or AI-generated. Answer with exactly one word: real or fake.
```

Model không generate tự do. Thay vào đó, code sẽ tính điểm cho hai đáp án:

```text
"real"
"fake"
```

Sau đó lấy softmax:

```text
fake_probability = P(fake) / (P(real) + P(fake))
```

Quy tắc dự đoán:

```text
fake_probability >= 0.5 -> fake
fake_probability < 0.5  -> real
```

## 5. Chạy Inference Trên Modal

Nếu LoRA đã nằm trong Modal Volume:

```text
qwen25vl-lora-outputs:/qwen25vl_lora_word_label/20260910_003026/adapter
```

chạy:

```powershell
$env:PYTHONIOENCODING='utf-8'

C:\Users\namth\AppData\Local\Programs\Python\Python312\python.exe -m modal run baselines/qwen25vl_lora_word_label/infer_qwen25vl_lora_commfor_modal.py --adapter-run-id 20260910_003026 --max-samples 1000
```

Chạy 2000 ảnh random CommunityForensics-Eval:

```powershell
$env:PYTHONIOENCODING='utf-8'

C:\Users\namth\AppData\Local\Programs\Python\Python312\python.exe -m modal run baselines/qwen25vl_lora_word_label/infer_qwen25vl_lora_commfor_modal.py --adapter-run-id 20260910_003026 --max-samples 2000 --sample-strategy streaming_shuffle --shuffle-buffer-size 1000
```

Chạy 1000 ảnh real-only để đo false positive:

```powershell
$env:PYTHONIOENCODING='utf-8'

C:\Users\namth\AppData\Local\Programs\Python\Python312\python.exe -m modal run baselines/qwen25vl_lora_word_label/infer_qwen25vl_lora_commfor_real_only_modal.py --adapter-run-id 20260910_003026 --max-real-samples 1000 --shuffle-buffer-size 1000
```

Nếu muốn dùng checkpoint adapter:

```powershell
$env:PYTHONIOENCODING='utf-8'

C:\Users\namth\AppData\Local\Programs\Python\Python312\python.exe -m modal run baselines/qwen25vl_lora_word_label/infer_qwen25vl_lora_commfor_modal.py --adapter-run-id 20260910_003026 --adapter-subdir checkpoints/latest/adapter --max-samples 1000
```

## 6. Output

Kết quả inference lưu ở:

```text
qwen25vl-lora-outputs:/qwen25vl_lora_word_label/<run_id>/inference_commfor_<inference_id>/
```

Các file chính:

```text
metrics/community_forensics_eval_overall_metrics.json
metrics/community_forensics_eval_generator_counts.csv
metrics/community_forensics_eval_generator_label_counts.csv
predictions/community_forensics_eval_predictions.csv
```

Với real-only inference, output nằm trong:

```text
qwen25vl-lora-outputs:/qwen25vl_lora_word_label/<run_id>/real_only_commfor_<inference_id>/
```

File chính:

```text
metrics/community_forensics_real_only_metrics.json
metrics/community_forensics_real_only_threshold_sweep.csv
metrics/community_forensics_real_only_generator_counts.csv
metrics/community_forensics_real_only_generator_metrics.csv
predictions/community_forensics_real_only_predictions.csv
```

Tải output về local:

```powershell
$env:PYTHONIOENCODING='utf-8'

C:\Users\namth\AppData\Local\Programs\Python\Python312\python.exe -m modal volume get --force qwen25vl-lora-outputs /qwen25vl_lora_word_label/20260910_003026/inference_commfor_20260910_165220 C:\tmp
```
