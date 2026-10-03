"""Bind an evaluation to its actual database, dataset, and model."""

import hashlib
import json
import pathlib

from text2sql.config import ANALYSIS_MODEL, WORKFLOW_VARIANT

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_DIR / "data" / "global_sales_data.sqlite"
MANIFEST_PATH = PROJECT_DIR / "data" / "eval_manifest.json"


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def verify_dataset(dataset):
    manifest = json.loads(MANIFEST_PATH.read_text())
    if (digest(DB_PATH) != manifest["database_sha256"]
            or pathlib.Path(dataset).name != manifest["dataset"]
            or digest(dataset) != manifest["dataset_sha256"]):
        raise ValueError("Database/eval dataset fingerprint mismatch. Regenerate ground truth and its manifest before evaluating.")


def run_metadata(dataset):
    return {"workflow": WORKFLOW_VARIANT, "model": ANALYSIS_MODEL,
            "database_sha256": digest(DB_PATH), "dataset": pathlib.Path(dataset).name,
            "dataset_sha256": digest(dataset)}
