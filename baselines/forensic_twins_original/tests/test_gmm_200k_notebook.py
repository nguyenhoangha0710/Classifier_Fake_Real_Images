"""Static integrity checks for the self-contained 200K GMM notebook."""

from __future__ import annotations

import base64
import io
import json
import re
import zipfile
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK = BASE_DIR / "forensic_twins_200k_gmm_eval_kaggle.ipynb"


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
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        names = set(archive.namelist())
        assert "fit_gmm_and_evaluate.py" in names
        assert "configs/kaggle_2xt4_200k.json" in names
        evaluator = archive.read("fit_gmm_and_evaluate.py").decode("utf-8")
        for required in (
            "native_raw",
            "fake_codec_matched",
            "build_common_jpeg_rows",
            "tiny_eligibility_audit.csv",
            "protocol_comparison.csv",
        ):
            assert required in evaluator

    notebook_source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    for required in (
        "twins-gmm-200k-checkpoint",
        "train_real_200k.csv",
        "JPEG_QUALITIES = [95, 80, 65, 50, 35]",
        "--fit-only",
        "--eval-only",
        "protocol_comparison.csv",
    ):
        assert required in notebook_source
    print(
        f"200K GMM notebook tests passed; cells={len(payload['cells'])}; "
        f"archive_bytes={len(base64.b64decode(encoded))}"
    )


if __name__ == "__main__":
    main()
