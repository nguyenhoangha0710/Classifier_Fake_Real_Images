# Implementation plan hiện hành — Twins–GMM

Baseline native/paper reproduction và checkpoint hiện có giữ nguyên. Kế hoạch triển khai extension được thống nhất trong cuộc thảo luận hiện tại:

- [Plan Twins–GMM codec augmentation](../forensic_twins_codec/plan.md)
- [Workflow/tensor contracts](../forensic_twins_codec/workflow.md)
- [Plan native trước cập nhật](archive/plan_before_codec_gsd_update.md)
- [Workflow native trước cập nhật](archive/workflow_before_codec_gsd_update.md)

Không đổi mặc định train native khi triển khai extension. Code codec mới ở thư mục riêng, reuse core native.
