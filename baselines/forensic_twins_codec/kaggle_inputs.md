# Kaggle inputs và manifest reuse — Twins–GMM / Twins–GSD–GMM

## 1. Những gì đã kiểm tra

Đối chiếu CHECKPOINTS.md, notebook/build scripts và lịch sử hai chat “Đọc dự án và baseline”, “Đọc dự án và baseline (2)”. Bản summary đã tải của evaluation-twins-bfree Version 2 xác nhận full run, không phải pilot.
Web công khai không đọc được các dataset trong lần kiểm tra này; URL được xác nhận từ lịch sử/code và artifact local, chưa xác nhận lại quyền truy cập hay version hiện tại trên server.

## 2. Các URL đã dùng

| Input/artifact | URL | Vai trò |
|---|---|---|
| Native encoder/FRE + manifest200K | https://www.kaggle.com/datasets/hoangha071005/twins-gmm-200k-checkpoint | Warm-start Twins FRE; manifest gốc dùng audit cả hai model |
| GMM native + fit/calibration manifests | https://www.kaggle.com/datasets/hoangha071005/fit-gmm-checkpoint-200k | Baseline cũ và split160K/40K; không reuse GMM cho encoder mới |
| Full augmented-GMM evaluation, Version 2 trong lần chạy đã lưu | https://www.kaggle.com/datasets/hoangha071005/evaluation-twins-bfree | Ưu tiên tìm split base và B-Free clean manifest đã lưu |
| Output slug trong notebook augmented GMM | https://www.kaggle.com/datasets/hoangha071005/forensic-twins-augmented-gmms | Nguồn output/resume khác; không mặc định đồng nhất nội dung/version với evaluation-twins-bfree |
| COCO ảnh gốc | https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset | Pixel training/GMM-fit |
| ImageNet ảnh gốc | https://www.kaggle.com/c/imagenet-object-localization-challenge/data | Pixel training/GMM-fit của cohort200K |
| B-Free ảnh | https://www.kaggle.com/datasets/baopv051/b-free-training-data | Evaluation real + sáu fake variants |
| Tiny-GenImage | https://www.kaggle.com/datasets/yangsangtai/tiny-genimage | Benchmark bổ sung đã có trong notebook GMM |

Slug manifest-only “hoangha071005/forensic-twins-bfree-manifests” chỉ là ví dụ trong notebook local, MANIFEST_DATASET_SLUG đang trống; chưa xác nhận dataset riêng này tồn tại.
Chưa tìm thấy manifest COCO-only riêng đã được xác nhận; cohort200K thực tế gồm cả COCO và ImageNet. Nếu muốn COCO-only, tạo profile riêng lọc source=coco2017 từ split đã lưu: 80K fit/20K calibration trước eligibility; không gọi đó là run200K.

## 3. Manifest/artifact paths cần resolve

Các đường dẫn dưới đây tương đối với root dataset, không hardcode toàn bộ mount:

| Dataset | Relative path |
|---|---|
| twins-gmm-200k-checkpoint | forensic_twins_v2_200k/manifests/train_real_200k.csv |
| twins-gmm-200k-checkpoint | forensic_twins_v2_200k/artifacts/checkpoints/last.ckpt |
| twins-gmm-200k-checkpoint | forensic_twins_v2_200k/cache/fre.pt |
| fit-gmm-checkpoint-200k | forensic_twins_200k_gmm_eval/manifests/gmm_fit_real.csv |
| fit-gmm-checkpoint-200k | forensic_twins_200k_gmm_eval/manifests/gmm_calibration_real.csv |
| evaluation-twins-bfree V2 | forensic_twins_augmented_gmm_full/manifests/augmented_gmm_fit_base.csv |
| evaluation-twins-bfree V2 | forensic_twins_augmented_gmm_full/manifests/augmented_gmm_calibration_base.csv |
| evaluation-twins-bfree V2 | forensic_twins_augmented_gmm_full/manifests/bfree_valid_clean_eval.csv |
| evaluation-twins-bfree V2 | forensic_twins_augmented_gmm_full/manifests/augmented_gmm_prepare_summary.json |
| evaluation-twins-bfree V2 | forensic_twins_augmented_gmm_full/manifests/bfree_manifest_summary.json |

Các path full-run được xác nhận từ summary local; phải kiểm tra file thực sự có trong dataset/version được mount. Nếu ZIP packaging, giải nén manifests.zip/artifacts.zip an toàn vào thư mục run riêng dưới /kaggle/working; input read-only. Tránh pilot directory forensic_twins_augmented_gmm.

Hashes ghi trong summary full-run:

- train_real_200k.csv: 56023995de848527cecad2298c2ddfa074bcd06dacd7f314b6f6bf9554066f90
- augmented_gmm_fit_base.csv: 47d6adc1f82d4a25778893eb6a7c0565f25d758cbc397ab2351689d887712591
- augmented_gmm_calibration_base.csv: 246cfac49ba4c86f29cada43263ffd48a0b4f46c5d0e8da444b89e34ec11d949
- bfree_valid_clean_eval.csv: 30d0dd7c2eb3063549141df8512c44e33d7fc7719703c8a1473c316390e9ab29

Giữ hash file gốc trước rebase. Sau rebase, byte hash có thể đổi; kiểm tra membership bằng sample IDs/source IDs và lưu parent hash + derived hash. Không đòi raw hash giống nhau cho hai schema CSV khác nhau.

## 4. Split và B-Free cohort phải giữ

- Full run: 160K fit = 80K COCO + 80K ImageNet; 40K calibration = 20K mỗi nguồn; seed42; split trước sinh variant.
- Quyết định hiện hành: SSL/fine-tune dùng toàn bộ parent train_real_200k.csv; với GSD lọc eligible hai crop224 từ toàn bộ parent200K và báo count thực tế. Split160K/40K chỉ dành cho PCA/GMM-fit/calibration; không tự thu SSL xuống160K.
- Calibration40K disjoint khỏi density-fit nhưng thuộc SSL pool của encoder (cả native và run mới); ghi rõ giới hạn này, không gọi encoder-held-out hay suy ra FPR5% ngoài miền.
- Reuse B-Free valid_clean: 1.557 source IDs, 1.557 real, 9.342 fake, 10.899 ảnh; sáu variant, không invalid group theo summary cũ.
- Cohort đã loại 8.747/10.304 valid IDs overlap với toàn bộ COCO train100K; giữ loại cả real và sáu fake theo source ID, kể cả khi run mới dùng subset nhỏ hơn.
- Recheck overlap với mọi nguồn train mới; không tái thêm IDs trước đây bị loại để làm đẹp số lượng. GSD224 và FRE256 dùng cùng image IDs; nếu có sample invalid phải báo missing và common intersection.
- Report từng fake variant trên cặp real/fake cân bằng, rồi macro-average; không diễn giải precision của pooled1:6 như pooled balanced.
- B-Free valid_clean đã được phân tích nhiều lần: benchmark hồi quy, không xem là test mù hoàn toàn mới và không tune threshold từ nhãn của nó.

## 5. Inputs theo stage

| Stage | Cần mount |
|---|---|
| Fine-tune Twins FRE + fit GMM mới | Native checkpoint/manifest + COCO + ImageNet + saved split manifests |
| Train Twins GSD + fit PCA/GMM | CLIP pretrained weights + COCO + ImageNet + saved split manifests; native manifest để audit, không dùng native encoder |
| Test baseline native cũ trên B-Free | Native encoder/FRE + fitted native GMM + B-Free + clean eval manifest |
| Test encoder mới trên B-Free | Bundle encoder mới (+Es/CLIP cho GSD), GMM mới/PCA nếu có, threshold + B-Free + clean eval manifest |

Test-only không cần mount COCO/ImageNet khi model, density và manifests đã đủ.
Manifest không chứa pixel; train/fit lại embeddings vẫn cần ảnh COCO/ImageNet gốc.
CLIP weights cần được tải/cache trước nếu notebook Internet off; model ID/hash phải nằm trong config.

## 6. Resolver và preflight bắt buộc

1. Support /kaggle/input/<slug> và /kaggle/input/datasets/<owner>/<slug>; competition ImageNet dùng root competition. Cho phép override root.
2. Reuse saved manifests trước; không quét recursive toàn B-Free hoặc rebuild dataset membership mỗi lần. Rebase relative_path/source+ID sang mount hiện tại; reject ambiguous matches.
3. Verify file/schema/labels/source counts/hash, fit∩calibration=∅; kiểm tra đủ pixel paths. Không dùng .tmp_kaggle_gmm_analysis cohort50K thay cho full200K.
4. Filter crop eligibility thành derived manifests, không thay parent split; log exclusions.
5. Validate B-Free groups 1real+6fake và source-ID non-overlap trước GPU.
6. Log resolved roots, dataset version (nếu biết), manifests, run profile, checkpoint epoch/step. Artifact đã kiểm tra có optimizer_step117250; không mặc định file tên last.ckpt là epoch80.
7. Output /kaggle/working/<run_id>; resume chỉ khi encoder/preprocess/manifest identity tương thích. Không reuse embedding/GMM cũ khi backbone hoặc GSD thay đổi.

## 7. Bằng chứng local

- Source/code: ../forensic_twins_original/CHECKPOINTS.md, bfree_evaluation_workflow.md, augmented_gmm_workflow.md và tools/build_*_notebook.py.
- Full-run summary đã đọc: C:/tmp/evaluation_twins_bfree_v2/forensic_twins_augmented_gmm_full__manifests__augmented_gmm_prepare_summary.json và ...__bfree_manifest_summary.json.
- Native GMM metadata đã đọc: workspace .tmp_fit_gmm_200k_report/real_k1_full.json.
- Cache .tmp_kaggle_gmm_analysis/forensic_twins_gmm_eval/manifests/gmm_split_summary.json là run50K (40K fit/10K calibration); tuyệt đối không nhầm với split200K.

## 8. Truy nguồn code train và URL manifest đời trước

Lần kiểm tra sâu bổ sung tìm được trong lịch sử yêu cầu trực tiếp: “Ngay sau Manifest ready, upload hai file manifest lên hoangha071005/forensic-twins-checkpoints”. Vì vậy URL lịch sử lưu manifest là https://www.kaggle.com/datasets/hoangha071005/forensic-twins-checkpoints . Chưa kiểm tra lại danh sách file/version hiện tại trên server.

Các URL lịch sử khác, không tự đánh đồng nội dung:

- https://www.kaggle.com/datasets/hoangha071005/twins-gmm-checkpoint : log run50K xác nhận file forensic_twins_original/manifests/train_real_50k.csv và encoder/FRE cùng bundle.
- https://www.kaggle.com/datasets/hoangha071005/check-point-gmm-of-twins : fitted GMM đời50K, không dùng nhầm với checkpoint200K.
- https://www.kaggle.com/datasets/hoangha071005/twins-gmm-200k-checkpoint : log200K và metadata xác nhận train_real_200k.csv như mục3.
- B-Free manifest đã được upload trong manifests.zip của forensic-twins-augmented-gmms (lịch sử log upload thành công); full-run summary của evaluation-twins-bfree V2 xác nhận bfree_valid_clean_eval.csv. Slug manifest-only trong notebook vẫn chưa xác nhận tồn tại.

Luồng code chính xác:

1. train_forensic_twins_kaggle.py:282–302: manifest_path = --manifest nếu có; mặc định OUTPUT_ROOT/manifests/train_real_<size>.csv. restore_manifest_artifacts chạy trước build_real_50k_manifest (tên hàm giữ từ bản50K nhưng count lấy từ config).
2. src/checkpointing.py:253: restore chỉ copy manifest + summary đã có trong checkpoint_directory local vào destination. Notebook tải bundle về checkpoint directory trước; bản thân hàm restore không tự tìm dataset manifest tùy ý trên toàn Kaggle Input.
3. src/data.py:398: nếu CSV đã có, kiểm tra count/label/provenance và reuse; nếu thiếu thì tạo từ pixel inputs ImageNet/COCO. COCO index ưu tiên captions_train2017.json, fallback instances_train2017.json; đây là annotation index của awsaf49/coco-2017-dataset, không phải manifest người dùng đã xuất.
4. train script:305–347 và src/checkpointing.py:226: --build-manifest-only có checkpoint.kaggle_dataset_slug thì publish CSV/summary ngay vào cùng checkpoint dataset. URL phụ thuộc slug được điền khi chạy, không hardcode COCO-only dataset.
5. Notebook train200K local vẫn để CHECKPOINT_DATASET_SLUG trống; dòng forensic-twins-v2-200k-checkpoints chỉ là gợi ý. Ưu tiên URL đã chứng minh trong run logs, không coi comment gợi ý là dataset đã tạo.

Khi code notebook mới, expose TRAIN_MANIFEST_PATH và BFREE_EVAL_MANIFEST_PATH để trỏ thẳng tới manifest đã mount. Không dựng manifest lại nếu đã có đúng CSV/provenance; vẫn phải mount ảnh gốc để train.
