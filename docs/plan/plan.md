# Kế hoạch triển khai dự án Explainable Fake Image Detection

## 1. Mục tiêu nghiên cứu

Dự án hướng tới xây dựng một hệ thống phát hiện ảnh thật/ảnh do AI tạo ra có khả năng giải thích được. Hệ thống không chỉ dự đoán nhãn `real/fake`, mà còn cần đưa ra lời giải thích dựa trên bằng chứng thị giác/forensic.

Luận điểm chính:

> Một hệ thống explainable fake image detection nên tách riêng ba năng lực: forensic perception, token-space alignment và explanation reasoning. Việc tách này giúp model học tín hiệu ảnh giả tốt hơn, chuyển được tín hiệu đó sang không gian MLLM, rồi mới sinh lời giải thích bằng ngôn ngữ tự nhiên.

Các câu hỏi cần chứng minh:

1. MLLM gốc có yếu ở fake image detection và forensic reasoning không?
2. Visual expert ở Stage 1 có phân loại tốt và generalize tốt hơn CLIP/MLLM thuần không?
3. Stage 2 có thật sự chuyển được fused forensic embedding sang token space của MLLM không?
4. Stage 3 có cải thiện explanation quality mà không làm giảm classification accuracy không?
5. Full pipeline có tốt hơn các baseline đơn giản như Base MLLM, Stage 3-only hoặc Stage 1 + prompt không?

## 2. Nguyên tắc triển khai

Không chạy tất cả experiment cùng lúc. Mỗi experiment phải trả lời một câu hỏi cụ thể. Nếu một stage thất bại, cần dừng lại phân tích trước khi train stage tiếp theo.

Thứ tự ưu tiên:

1. Chuẩn hóa dataset và evaluation protocol.
2. Chạy Base MLLM baseline.
3. Train và ablation Stage 1.
4. Kiểm tra Stage 1 + LLM/prompt.
5. Train Stage 2 alignment.
6. Kiểm tra Stage 1 + Stage 2 + prompt.
7. Train Stage 3-only.
8. Train full pipeline Stage 1 -> Stage 2 -> Stage 3.
9. So sánh với biến thể `(Stage 1 + Stage 2) + Stage 3`.

## 3. Phase 0: Chuẩn hóa dataset và evaluation

### Mục tiêu

Tạo nền tảng dữ liệu và metric thống nhất để mọi experiment có thể so sánh công bằng.

### Dataset format đề xuất

Mỗi sample nên có metadata dạng:

```json
{
  "image_path": "path/to/image.jpg",
  "label": "real",
  "dataset_source": "GenImage",
  "generator": "stable-diffusion-v1-5",
  "split": "train",
  "explanation": "optional natural language explanation",
  "forensic_attributes": ["optional", "artifact", "tags"]
}
```

Các trường bắt buộc:

- `image_path`
- `label`
- `dataset_source`
- `generator`
- `split`

Các trường nên có nếu dataset hỗ trợ:

- `explanation`
- `forensic_attributes`
- `manipulation_type`
- `resolution`
- `compression_level`

### Split evaluation

Cần tách rõ:

- `in-domain`: train/test cùng dataset hoặc cùng generator distribution.
- `cross-generator`: test trên generator chưa thấy khi train.
- `cross-dataset`: train trên dataset này, test trên dataset khác.
- `robustness`: test sau JPEG compression, resize, crop, blur, screenshot/re-upload nếu có thể.

### Metrics

Classification:

- Accuracy
- F1-score
- AUROC
- EER nếu cần

Explanation:

- Human/GPT-based explanation score
- Evidence consistency
- Faithfulness score
- Error type analysis

Robustness:

- Accuracy drop sau perturbation
- F1 drop sau perturbation
- Cross-generator performance gap

## 4. Phase 1: Base MLLM baseline

### Experiment

`Base MLLM`

### Mục đích

Kiểm tra khả năng gốc của MLLM trong việc:

- nhìn ảnh và phân loại `real/fake`,
- sinh explanation,
- phát hiện forensic cues.

### Thiết lập

Không train model. Chỉ dùng MLLM có sẵn với prompt cố định.

Prompt mẫu:

```text
Is this image real or AI-generated?
Answer with one label: real or fake.
Then briefly explain the visual evidence.
```

### Kết quả cần lưu

- Label prediction
- Confidence nếu model có sinh
- Explanation
- Lỗi thường gặp

### Câu hỏi cần trả lời

- MLLM gốc có phân loại tốt không?
- Explanation có cụ thể hay chỉ nói chung chung?
- Model dựa vào semantic hay forensic cues?
- Model có hallucinate bằng chứng không?

### Tiêu chí qua phase

Hoàn thành phase này khi có baseline table và qualitative error analysis.

## 5. Phase 2: Stage 1 - Classification Expert Training

### Experiment

`Stage 1`

### Mục đích

Train visual expert chuyên phân biệt ảnh thật và ảnh AI bằng hai loại tín hiệu:

- semantic cues từ CLIP,
- low-level forensic cues từ NPR-ResNet.

### Kiến trúc

Ảnh đầu vào đi qua hai nhánh:

1. CLIP Vision Encoder: lấy semantic/general visual representation.
2. NPR + ResNet: lấy forensic representation như texture, noise, edge artifact, oversmoothing.

Sau đó fusion feature để train classifier `real/fake`.

### Ablation bắt buộc

| Model | Mục đích |
| --- | --- |
| CLIP-only classifier | Kiểm tra semantic branch có đủ mạnh không |
| NPR-ResNet-only | Kiểm tra forensic branch có đủ mạnh không |
| CLIP + NPR fusion | Kiểm tra fusion có tốt hơn từng nhánh riêng không |

### Evaluation

- In-domain test
- Cross-generator test
- Cross-dataset test
- JPEG compression robustness
- Resize/crop robustness

### Câu hỏi cần trả lời

- Fusion có tốt hơn CLIP-only và NPR-only không?
- NPR branch có giúp unseen generator không?
- CLIP branch có giúp giảm false positive trên ảnh real không?
- Model có học shortcut từ dataset không?

### Tiêu chí qua phase

Stage 1 chỉ được xem là ổn nếu:

- fusion tốt hơn hoặc ít nhất ổn định hơn từng branch riêng,
- performance trên unseen generator không sụp mạnh,
- robustness không quá kém khi ảnh bị nén hoặc resize.

Nếu fusion không hơn CLIP-only, cần xem lại NPR transform, fusion strategy hoặc dataset bias.

## 6. Phase 3: Stage 1 + LLM

### Experiment

`Stage1 + LLM`

### Mục đích

Kiểm tra output từ Stage 1 có giúp LLM sinh câu trả lời tốt hơn Base MLLM không.

### Thiết lập

Stage 1 đã train xong. LLM có thể nhận thông tin từ Stage 1 theo dạng text/prompt.

Mức đơn giản:

```json
{
  "prediction": "fake",
  "confidence": 0.87
}
```

Mức tốt hơn:

```json
{
  "prediction": "fake",
  "confidence": 0.87,
  "evidence": ["texture inconsistency", "unnatural edge", "oversmoothing"]
}
```

### Câu hỏi cần trả lời

- LLM có viết explanation tốt hơn Base MLLM không?
- Explanation có bám vào output của Stage 1 không?
- Nếu Stage 1 sai, LLM có bị kéo sai theo không?
- Prompt-based method có đủ tốt để không cần Stage 2 không?

### Tiêu chí qua phase

Hoàn thành khi có so sánh:

- Base MLLM
- Stage 1 + label prompt
- Stage 1 + label + evidence prompt

## 7. Phase 4: Stage 2 - Token-Space Forensic Perception Alignment

### Experiment

`Stage2 + LLM`

### Mục đích

Train projector để chuyển fused visual embedding từ Stage 1 sang hidden/token space mà MLLM đóng băng có thể sử dụng.

### Thiết lập

- Freeze Stage 1.
- Freeze MLLM.
- Chỉ train projector.
- Objective ban đầu: MLLM predict đúng token `real/fake`.

### Objective chính

Input:

- fused embedding từ Stage 1,
- prompt ngắn yêu cầu trả lời `real/fake`.

Output:

- token `real` hoặc `fake`.

### Objective phụ nên cân nhắc

Không nên chỉ train `real/fake`, vì projector có thể trở thành classifier head trá hình. Nên thêm forensic attribute tokens nếu có dữ liệu hoặc có thể pseudo-label.

Ví dụ attribute tokens:

- `noise_inconsistent`
- `unnatural_edges`
- `oversmoothing`
- `texture_artifact`
- `lighting_inconsistency`
- `semantic_inconsistency`

### Evaluation

- Real/fake token accuracy
- Attribute token accuracy nếu có
- So sánh với Stage 1 classifier gốc
- So sánh với Stage 1 + prompt

### Câu hỏi cần trả lời

- Projector có giữ được năng lực phân loại từ Stage 1 không?
- MLLM đóng băng có đọc được visual tokens không?
- Stage 2 có cải thiện so với prompt-based Stage 1 + LLM không?
- Alignment có giúp explanation về sau không?

### Tiêu chí qua phase

Stage 2 đạt yêu cầu nếu:

- token accuracy gần với Stage 1 classifier accuracy,
- performance không giảm mạnh trên cross-generator,
- visual tokens giúp MLLM trả lời ổn định hơn prompt-only.

## 8. Phase 5: Stage 1 + Stage 2 + Prompt

### Experiment

`Stage1+Stage2 + Prompt`

### Mục đích

Kiểm tra sau khi alignment, chỉ cần prompt nhẹ thì MLLM có thể trả lời và giải thích tốt chưa.

### Thiết lập

- Stage 1 frozen.
- Stage 2 projector frozen.
- MLLM chưa fine-tune explanation.
- Prompt yêu cầu output `real/fake` và explanation ngắn.

### Câu hỏi cần trả lời

- Label accuracy có tốt không?
- Explanation có tốt hơn Base MLLM không?
- Nếu label đúng nhưng explanation tệ, Stage 3 là cần thiết.
- Nếu label tệ, Stage 2 alignment chưa ổn.

### Tiêu chí qua phase

Phase này là diagnostic checkpoint. Không cần đạt explanation tốt nhất, nhưng phải cho thấy visual-token alignment có ích.

## 9. Phase 6: Stage 3 - Explanation Reasoning Training

### Experiment

`Stage 3`

### Mục đích

Kiểm tra riêng khả năng MLLM sinh explanation sau fine-tuning trên explanation dataset.

### Thiết lập

- Fine-tune MLLM hoặc LoRA trên dataset có label + explanation.
- Có thể chưa nối Stage 1/2.
- Input là ảnh + instruction.
- Output là label + explanation.

### Câu hỏi cần trả lời

- Stage 3-only có cải thiện explanation fluency không?
- Label accuracy có tăng hay giảm?
- Model có học template explanation không?
- Explanation có faithful với ảnh không?

### Rủi ro

Stage 3-only có thể viết rất hay nhưng không thật sự nhìn forensic evidence. Vì vậy không được chỉ đánh giá bằng độ mượt của câu trả lời.

### Tiêu chí qua phase

Hoàn thành khi có:

- classification metrics,
- explanation quality score,
- qualitative error analysis,
- so sánh với Base MLLM.

## 10. Phase 7: Full sequential pipeline

### Experiment

`Stage 1 -> Stage 2 -> Stage 3`

### Mục đích

Kiểm tra chất lượng full pipeline khi train theo đúng thứ tự:

1. Train Stage 1 visual expert.
2. Freeze Stage 1.
3. Train Stage 2 projector.
4. Freeze Stage 1 + Stage 2.
5. Train Stage 3 explanation/reasoning.

### Câu hỏi cần trả lời

- Full pipeline có tốt hơn Base MLLM không?
- Full pipeline có tốt hơn Stage 3-only không?
- Full pipeline có tốt hơn Stage 1 + prompt không?
- Stage 3 có làm giảm classification accuracy không?
- Explanation có bám vào evidence từ Stage 1/2 không?

### Tiêu chí thành công

Full pipeline được xem là thành công nếu:

- classification accuracy tốt hơn Base MLLM và Stage 3-only,
- cross-generator performance tốt,
- explanation cụ thể hơn Base MLLM,
- faithfulness tốt hơn Stage 3-only,
- không làm mất năng lực phân loại của Stage 1.

## 11. Phase 8: Independent combination

### Experiment

`(Stage1+2) + Stage3`

### Mục đích

Kiểm tra giả thuyết rằng forensic alignment và explanation reasoning có thể nên được học độc lập.

Lý do:

- Stage 1 + Stage 2 học forensic technique và token alignment.
- Stage 3 học semantic/explanation style.
- Hai phần này có thể không cùng supervision hoặc không cùng evidence.

### Thiết lập

Train riêng:

- Module A: Stage 1 + Stage 2 để học forensic perception và token-space alignment.
- Module B: Stage 3 để học explanation generation.

Sau đó ghép lại khi inference.

### So sánh cần có

| Setup | Ý nghĩa |
| --- | --- |
| Stage 1 -> Stage 2 -> Stage 3 | Pipeline tuần tự |
| `(Stage 1 + Stage 2) + Stage 3` | Forensic alignment và explanation học độc lập |
| Stage 3-only | Chỉ fine-tune MLLM |
| Stage 1 + prompt | Classifier-assisted explanation |

### Câu hỏi cần trả lời

- Independent combination có ổn định hơn sequential training không?
- Explanation có ít hallucination hơn không?
- Classification accuracy có giữ tốt hơn không?
- Hai loại supervision forensic và semantic có xung đột không?

## 12. Bảng experiment tổng hợp

| Experiment | Mục đích | Output chính |
| --- | --- | --- |
| Base MLLM | Kiểm khả năng nhìn, phân loại và sinh explanation của MLLM gốc | Baseline accuracy + explanation quality |
| Stage 1 | Kiểm tra visual expert và fusion semantic/forensic | Classifier metrics + ablation |
| Stage1 + LLM | Kiểm tra Stage 1 có giúp LLM sinh câu trả lời tốt hơn không | Prompt-based answer quality |
| Stage2 + LLM | Kiểm tra token-space alignment | Token accuracy + alignment quality |
| Stage1+Stage2 + Prompt | Kiểm tra classifier, answer quality và chuyển đổi space trước Stage 3 | Diagnostic result |
| Stage 3 | Kiểm tra chất lượng riêng MLLM sau fine-tuning explanation | Explanation quality + label accuracy |
| Stage 1 -> 2 -> 3 | Kiểm tra full sequential pipeline | Main result |
| `(Stage1+2) + Stage3` | Kiểm tra kết hợp độc lập giữa forensic alignment và explanation reasoning | Comparison với full sequential |

## 13. Main result table đề xuất

| Method | Acc | F1 | AUROC | Cross-Gen Acc | Cross-Dataset Acc | Robust Acc | Explanation Score | Faithfulness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base MLLM | | | | | | | | |
| Stage 3-only | | | | | | | | |
| Stage 1 + prompt | | | | | | | | |
| Stage 2 + prompt | | | | | | | | |
| Stage 1 -> 2 -> 3 | | | | | | | | |
| `(Stage 1 + 2) + Stage 3` | | | | | | | | |

## 14. Ablation table đề xuất cho Stage 1

| Model | In-domain Acc | Cross-Gen Acc | Cross-Dataset Acc | JPEG Robust Acc | Resize Robust Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLIP-only | | | | | |
| NPR-ResNet-only | | | | | |
| CLIP + NPR Fusion | | | | | |

## 15. Error analysis bắt buộc

Sau mỗi phase lớn cần lưu lại ví dụ lỗi:

- fake nhưng model dự đoán real,
- real nhưng model dự đoán fake,
- label đúng nhưng explanation sai,
- explanation đúng hướng nhưng quá chung chung,
- model hallucinate evidence không có trong ảnh,
- model dựa vào semantic shortcut,
- model bị ảnh hưởng bởi compression/resolution.

Mỗi lỗi nên lưu:

```json
{
  "image_path": "path/to/image.jpg",
  "ground_truth": "fake",
  "prediction": "real",
  "confidence": 0.62,
  "explanation": "model output",
  "error_type": "false_negative",
  "notes": "possible compression artifact or weak forensic signal"
}
```

## 16. Rủi ro chính và cách kiểm tra

### Rủi ro 1: Model học shortcut dataset

Cách kiểm tra:

- cross-dataset evaluation,
- cross-generator evaluation,
- kiểm tra metadata như resolution, compression, watermark.

### Rủi ro 2: Explanation nghe hợp lý nhưng không faithful

Cách kiểm tra:

- so sánh explanation với evidence/attribute,
- perturb vùng nghi ngờ và xem prediction/explanation có đổi không,
- dùng human review cho một subset.

### Rủi ro 3: Stage 2 chỉ là classifier head trá hình

Cách kiểm tra:

- thêm forensic attribute prediction,
- so sánh token-space visual embedding với text forensic concepts,
- đánh giá explanation sau Stage 2 + prompt.

### Rủi ro 4: Stage 3 làm giảm classification accuracy

Cách kiểm tra:

- freeze Stage 1 và Stage 2 khi train Stage 3,
- so sánh accuracy trước và sau Stage 3,
- theo dõi performance trên cross-generator.

## 17. Milestone thực hiện

### Milestone 1: Data and baseline

Hoàn thành:

- dataset loader thống nhất,
- metadata schema,
- Base MLLM inference script,
- baseline result table.

### Milestone 2: Stage 1

Hoàn thành:

- CLIP-only classifier,
- NPR-ResNet-only classifier,
- CLIP + NPR fusion classifier,
- Stage 1 ablation table.

### Milestone 3: Stage 1 + LLM

Hoàn thành:

- prompt template,
- inference pipeline từ Stage 1 sang LLM,
- qualitative comparison với Base MLLM.

### Milestone 4: Stage 2

Hoàn thành:

- projector module,
- frozen Stage 1 + frozen MLLM training loop,
- token `real/fake` accuracy,
- optional forensic attribute prediction.

### Milestone 5: Stage 3

Hoàn thành:

- explanation dataset loader,
- LoRA/fine-tuning setup,
- Stage 3-only result.

### Milestone 6: Full comparison

Hoàn thành:

- Stage 1 -> 2 -> 3 full pipeline,
- `(Stage 1 + 2) + Stage 3` independent combination,
- final main table,
- error analysis.

## 18. Quyết định tiếp theo

Bước nên làm ngay:

1. Tạo schema dữ liệu chuẩn.
2. Viết script kiểm tra dataset statistics.
3. Chạy Base MLLM trên một subset nhỏ.
4. Train Stage 1 ablation trên subset nhỏ để kiểm tra pipeline chạy đúng.
5. Sau khi code ổn mới scale lên full dataset.

Không nên bắt đầu bằng full training Stage 1 -> 2 -> 3 ngay. Làm vậy rất khó biết lỗi nằm ở dataset, visual expert, projector hay LLM.
