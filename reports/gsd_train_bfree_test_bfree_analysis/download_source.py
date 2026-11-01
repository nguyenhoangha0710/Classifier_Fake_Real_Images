"""Download only public result tables needed to audit the B-Free evaluation."""
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

HANDLE = "hoangha071005/test-gsd-train-bfree-test-bfree"
API = "https://www.kaggle.com/api/v1/datasets/"
ROOT = Path(__file__).resolve().parent / "source"
RUN = "gsd_bfree_samecat_origbg_best_bfree_valid_v1"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "BFree-result-audit/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    metadata = fetch(API + "view/" + HANDLE)
    version = json.loads(metadata)["currentVersionNumber"]
    (ROOT / "dataset_metadata.json").write_bytes(metadata)

    files, token = [], ""
    while True:
        query = urllib.parse.urlencode({"pageSize": 1000, "pageToken": token})
        page = json.loads(fetch(API + "list/" + HANDLE + "?" + query))
        files.extend(page["datasetFiles"])
        token = page.get("nextPageToken")
        if not token:
            break
    (ROOT / "dataset_file_inventory.json").write_text(
        json.dumps(files, indent=2), encoding="utf-8"
    )

    wanted = {
        "bfree_manifest_summary.json",
        "checkpoint_inspection.json",
        "completion.json",
        "evaluation_config.json",
        "evaluation_state.json",
        "macro_metrics.csv",
        "manifest_identity.json",
        "metrics.csv",
        "partial_metrics.csv",
        "predictions.csv",
        "predictions_SD2.1_inpainted_diffcat_origBG.partial.csv",
    }
    selected = [
        item["name"]
        for item in files
        if item["name"].startswith(RUN + "/")
        and "/progress/" not in item["name"]
        and Path(item["name"]).name in wanted
    ]

    downloaded = []
    for name in selected:
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = (
            API
            + "download/"
            + HANDLE
            + "/"
            + urllib.parse.quote(name, safe="")
            + "?datasetVersionNumber="
            + str(version)
        )
        data = fetch(url)
        target.write_bytes(data)
        downloaded.append(
            {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
        print(f"downloaded {name}: {len(data):,} bytes", flush=True)

    provenance = {
        "dataset": HANDLE,
        "version": version,
        "listed_files": len(files),
        "downloads": downloaded,
    }
    (ROOT / "download_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
