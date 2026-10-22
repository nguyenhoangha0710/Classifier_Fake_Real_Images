"""Static integrity checks for the self-contained B-Free Kaggle notebook."""

from __future__ import annotations

import base64
import io
import json
import re
import zipfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK = BASE_DIR / "forensic_twins_bfree_eval_kaggle.ipynb"


def main() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    for index, cell in enumerate(code_cells):
        compile("".join(cell.get("source", [])), f"{NOTEBOOK}:cell-{index}", "exec")

    extraction_cell = next(
        "".join(cell.get("source", []))
        for cell in code_cells
        if "SOURCE_ARCHIVE_B64 =" in "".join(cell.get("source", []))
    )
    match = re.search(r"SOURCE_ARCHIVE_B64 = \(\n(.*?)\n\)", extraction_cell, re.DOTALL)
    assert match is not None
    encoded = "".join(re.findall(r'"([A-Za-z0-9+/=]+)"', match.group(1)))
    archive_bytes = base64.b64decode(encoded)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = set(archive.namelist())
        for required in (
            "test_bfree_gmm_kaggle.py",
            "fit_gmm_and_evaluate.py",
            "configs/kaggle_2xt4_200k.json",
            "src/core.py",
            "src/checkpointing.py",
            "tests/test_bfree_evaluation.py",
        ):
            assert required in names
        evaluator = archive.read("test_bfree_gmm_kaggle.py").decode("utf-8")
        for required in (
            "bfree_valid_overlap_excluded.csv",
            "bfree_valid_clean_ids.csv",
            "bfree_manifest_summary.json",
            "embedding-checkpoint-every",
            "pooled_balanced_auroc",
            "bootstrap_ci95.json",
        ):
            assert required in evaluator

    notebook_source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    for required in (
        "b-free-training-data",
        "twins-gmm-200k-checkpoint",
        "fit-gmm-checkpoint-200k",
        "REUSE_MANIFEST_PATH",
        "MANIFEST_DATASET_SLUG",
        "RESULT_DATASET_SLUG",
        "--build-manifest-only",
        "--eval-only",
    ):
        assert required in notebook_source
    print(
        f"B-Free notebook tests passed; cells={len(payload['cells'])}; "
        f"archive_bytes={len(archive_bytes)}"
    )


if __name__ == "__main__":
    main()
