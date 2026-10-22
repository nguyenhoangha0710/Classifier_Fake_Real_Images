"""Static checks for the self-contained augmented-GMM Kaggle notebook."""

from __future__ import annotations

import base64
import io
import json
import re
import zipfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK = BASE_DIR / "forensic_twins_augmented_gmm_bfree_kaggle.ipynb"


def main() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    for index, cell in enumerate(code_cells):
        compile("".join(cell.get("source", [])), f"{NOTEBOOK}:cell-{index}", "exec")

    extraction = next(
        "".join(cell.get("source", []))
        for cell in code_cells
        if "SOURCE_ARCHIVE_B64 =" in "".join(cell.get("source", []))
    )
    match = re.search(r"SOURCE_ARCHIVE_B64 = \(\n(.*?)\n\)", extraction, re.DOTALL)
    assert match is not None
    encoded = "".join(re.findall(r'"([A-Za-z0-9+/=]+)"', match.group(1)))
    archive_bytes = base64.b64decode(encoded)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = set(archive.namelist())
        for required in (
            "fit_augmented_gmms_bfree.py",
            "test_bfree_gmm_kaggle.py",
            "fit_gmm_and_evaluate.py",
            "configs/kaggle_2xt4_200k.json",
            "src/core.py",
            "tests/test_augmented_gmm.py",
        ):
            assert required in names
        evaluator = archive.read("fit_augmented_gmms_bfree.py").decode("utf-8")
        for required in (
            'choices=("prepare", "extract", "fit", "evaluate", "all")',
            "resize512_png",
            "mixed_k{components}",
            "calibration_fpr.csv",
            "augmented_evaluation_complete.json",
        ):
            assert required in evaluator

    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    for required in (
        "imagenet-object-localization-challenge",
        "coco-2017-dataset",
        "twins-gmm-200k-checkpoint",
        "b-free-training-data",
        "RESULT_DATASET_SLUG",
        "RESUME_OUTPUT_DATASET_ROOT",
        'run_stage("prepare")',
        'run_stage("fit")',
        'run_stage("evaluate")',
        '"-r", "zip"',
        "Pipeline will continue",
        "MAX_GMM_FIT_EMBEDDINGS = None",
    ):
        assert required in source
    assert '"--private"' not in source
    print(
        f"Augmented GMM notebook tests passed; cells={len(payload['cells'])}; "
        f"archive_bytes={len(archive_bytes)}"
    )


if __name__ == "__main__":
    main()
