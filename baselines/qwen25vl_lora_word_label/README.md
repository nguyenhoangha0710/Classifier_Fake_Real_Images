# Qwen2.5-VL-7B LoRA Word-Label

Thu muc nay chua pipeline fine-tune **Qwen2.5-VL-7B-Instruct** cho bai toan phan loai anh `real/fake`.

File chinh:

```text
train_qwen25vl_lora_word_label_kaggle.ipynb
train_qwen25vl_lora_word_label_modal.py
```

## Protocol

```text
Train: Tiny-GenImage combined train split
Early stopping: Tiny-GenImage train_inner/val_inner
Internal sanity test: Tiny-GenImage validation split
External benchmark: 1000 random streaming samples from CommunityForensics-Eval / CompEval
```

CommunityForensics-Eval chi dung de test cuoi, khong dung de train, early stopping, threshold tuning hoac model selection.

## Y Tuong

Khong ep `real` hoac `fake` la mot token duy nhat.

Thay vao do:

```text
Prompt ep model tra loi dung mot word: real hoac fake.
Ground truth la text answer: "real" hoac "fake".
Training dung SFT loss tren phan assistant answer.
Evaluation so sanh score chuoi P("real") va P("fake").
```

Prompt mac dinh:

```text
Look at the image and decide whether it is real or AI-generated. Answer with exactly one word: real or fake.
```

## Model

Base model:

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

Fine-tuning:

```text
QLoRA 4-bit
LoRA target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

Pipeline khong luu full Qwen2.5-VL-7B. Sau khi train, chi luu LoRA adapter.

## Config Mac Dinh

```python
MAX_TRAIN_SAMPLES = 5000
MAX_TINY_INTERNAL_TEST_SAMPLES = 1000
MAX_VAL_SAMPLES = 1000
MAX_COMMFOR_EVAL_SAMPLES = 1000
COMMFOR_SHUFFLE_BUFFER_SIZE = 100

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
MAX_EPOCHS = 2
PATIENCE = 1
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
LOAD_IN_4BIT = True
USE_GRADIENT_CHECKPOINTING = True
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 128 * 28 * 28
CHECKPOINT_EVERY_OPTIMIZER_STEPS = 25
```

Qwen2.5-VL-7B rat nang, nen mac dinh chi train 5000 anh de test pipeline truoc.

Khi muon train full Tiny-GenImage combined:

```python
MAX_TRAIN_SAMPLES = None
```

Trong file Modal, dung flag:

```bash
modal run baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_modal.py --full-train
```

## Chay Tren Modal

File Modal moi:

```text
train_qwen25vl_lora_word_label_modal.py
```

Mac dinh file nay dung:

```python
GPU_TYPE = "A100-40GB"
TINY_VOLUME_NAME = "tiny-genimage-data"
OUTPUT_VOLUME_NAME = "qwen25vl-lora-outputs"
HF_CACHE_VOLUME_NAME = "hf-cache"
REMOTE_TINY_ROOT = "/data/tiny-genimage"
REMOTE_OUTPUT_ROOT = "/outputs/qwen25vl_lora_word_label"
KAGGLE_DATASET_SLUG = "yangsangtai/tiny-genimage"
```

Cau truc Tiny-GenImage trong Modal can dung:

```text
/data/tiny-genimage/
  imagenet_ai_0419_biggan/
    train/ai/
    train/nature/
    val/ai/
    val/nature/
  imagenet_ai_0419_vqdm/
  imagenet_ai_0424_sdv5/
  imagenet_ai_0424_wukong/
  imagenet_ai_0508_adm/
  imagenet_glide/
  imagenet_midjourney/
```

Chuan bi Modal local:

```bash
pip install modal
modal setup
```

### Kaggle API Key

Vao Kaggle:

```text
Account -> Settings -> API -> Create New Token
```

Kaggle se tai ve file:

```text
kaggle.json
```

Tao Modal Secret tu file nay:

```bash
modal secret create kaggle-secret --from-json /path/to/kaggle.json
```

Tren may Windows cua minh, nen chay bang Python 3.12:

```powershell
$env:PYTHONIOENCODING='utf-8'
C:\Users\namth\AppData\Local\Programs\Python\Python312\python.exe -m modal secret create kaggle-secret --from-json "C:\path\to\kaggle.json"
```

Khong commit `kaggle.json` vao Git.

### Tai Tiny-GenImage Tu Kaggle Vao Modal Volume

Script Modal co the tu tai dataset tu Kaggle vao Modal Volume theo flow:

```text
Modal container -> Kaggle API -> download Tiny-GenImage -> /data/tiny-genimage -> train
```

Tao volume:

```bash
modal volume create tiny-genimage-data
```

Tai dataset ve volume ma chua train:

```bash
modal run baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_modal.py --download-only
```

Lenh tren se tai Kaggle dataset:

```text
https://www.kaggle.com/datasets/yangsangtai/tiny-genimage
```

vao path:

```text
tiny-genimage-data:/tiny-genimage
```

Kiem tra volume:

```bash
modal volume ls tiny-genimage-data /
```

Chay nhanh de kiem tra pipeline:

```bash
modal run baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_modal.py --max-train-samples 1000 --max-epochs 1 --max-val-samples 300 --max-commfor-eval-samples 300
```

Chay config mac dinh:

```bash
modal run baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_modal.py
```

Chay full Tiny-GenImage combined:

```bash
modal run baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_modal.py --full-train --max-epochs 2
```

### Checkpoint Va Resume

Modal script luu checkpoint dinh ky vao:

```text
qwen25vl-lora-outputs:/qwen25vl_lora_word_label/<run_id>/checkpoints/latest/
```

File chinh:

```text
training_state.pt
lora_state_dict.pt
checkpoint_info.json
adapter/
```

Mac dinh checkpoint duoc ghi moi:

```text
25 optimizer steps
```

Voi:

```text
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
```

thi checkpoint se duoc luu khoang moi 400 anh train.

Chay tiep tu checkpoint moi nhat:

```bash
modal run baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_modal.py --full-train --max-epochs 2
```

Chay tiep tu mot run cu the:

```bash
modal run baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_modal.py --resume-run-id 20260909_184050 --full-train --max-epochs 2
```

Tat resume tu dong va tao run moi:

```bash
modal run baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_modal.py --no-resume --full-train --max-epochs 2
```

Doi tan suat checkpoint:

```bash
modal run baselines/qwen25vl_lora_word_label/train_qwen25vl_lora_word_label_modal.py --full-train --checkpoint-every-optimizer-steps 10
```

## Output

Kaggle notebook luu tai:

```text
/kaggle/working/outputs/qwen25vl_lora_word_label/<run_id>/
```

Modal script luu tai Modal Volume:

```text
qwen25vl-lora-outputs:/qwen25vl_lora_word_label/<run_id>/
```

File chinh trong moi run:

```text
adapter/
  adapter_config.json
  adapter_model.safetensors
  tokenizer/processor files

metrics/
  config.json
  history.csv
  tiny_genimage_validation_overall_metrics.json
  community_forensics_eval_overall_metrics.json
  summary.json

predictions/
  tiny_genimage_validation_predictions.csv
  community_forensics_eval_predictions.csv
```

Manifest tong:

```text
summary_qwen25vl_lora_word_label_<run_id>.csv
adapter_manifest_<run_id>.csv
```

Lay ket qua Modal ve local:

```bash
modal volume get qwen25vl-lora-outputs / ./modal_qwen_outputs
```

## Load Lai Adapter

```python
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    device_map="auto",
    quantization_config=quantization_config,
)
model = PeftModel.from_pretrained(base_model, "/path/to/adapter")
processor = AutoProcessor.from_pretrained("/path/to/adapter")
```

## Ghi Chu Bao Cao

Co the mo ta trong bao cao:

```text
We fine-tune Qwen2.5-VL-7B-Instruct using QLoRA for binary real/fake image classification. Each sample is formatted as a vision-language instruction where the model is asked to answer with exactly one word: real or fake. The training target is the textual label itself, and the loss is applied only to the assistant answer tokens. At evaluation time, we do not rely on free-form generation. Instead, we score the candidate answer sequences "real" and "fake" and convert their scores into a fake probability.
```
