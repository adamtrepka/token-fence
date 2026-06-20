from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
from joblib import dump
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

from shell_features import build_model_text


DEFAULT_INPUT_DIR = Path("dataset-shell")
DEFAULT_OUTPUT_DIR = Path("model-shell")
WARN_THRESHOLD_FLOOR = 0.05


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an ONNX-exportable output-risk model from OpenCode tool-call data."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing train.jsonl, val.jsonl and test.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the trained model and reports.",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=1,
        help="Minimum token n-gram size.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Maximum token n-gram size.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency for token n-grams.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=30000,
        help="Maximum number of TF-IDF features.",
    )
    parser.add_argument(
        "--select-k",
        type=int,
        default=None,
        help="Keep only the top-k TF-IDF features by chi-square score.",
    )
    parser.add_argument(
        "--C",
        dest="C",
        type=float,
        default=2.0,
        help="Inverse regularization strength for logistic regression.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum logistic regression iterations.",
    )
    parser.add_argument(
        "--threshold-beta",
        type=float,
        default=2.0,
        help="Beta for fallback F-beta threshold selection on the validation set.",
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.60,
        help="Target validation precision for block-threshold selection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the classifier.",
    )
    return parser


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def load_split(input_dir: Path, name: str) -> list[dict[str, object]]:
    path = input_dir / f"{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}")
    return read_jsonl(path)


def load_splits(input_dir: Path) -> dict[str, list[dict[str, object]]]:
    return {
        "train": load_split(input_dir, "train"),
        "val": load_split(input_dir, "val"),
        "test": load_split(input_dir, "test"),
    }


def get_input(row: dict[str, object]) -> dict[str, object]:
    value = row.get("input")
    return value if isinstance(value, dict) else {}


def get_workdir(row: dict[str, object]) -> str:
    input_data = get_input(row)
    workdir = input_data.get("workdir")
    if isinstance(workdir, str) and workdir.strip():
        return workdir.strip()
    features = row.get("features")
    if isinstance(features, dict):
        maybe_workdir = features.get("workdir")
        if isinstance(maybe_workdir, str) and maybe_workdir.strip():
            return maybe_workdir.strip()
    return ""


def get_command(row: dict[str, object]) -> str:
    input_data = get_input(row)
    command = input_data.get("command")
    if isinstance(command, str) and command.strip():
        return command.strip()
    command = row.get("command")
    if isinstance(command, str) and command.strip():
        return command.strip()
    return ""


def get_family(row: dict[str, object]) -> str:
    family = row.get("tool_family")
    if isinstance(family, str) and family.strip():
        return family.strip()
    return "unknown"


def get_feature_tokens(row: dict[str, object]) -> list[str]:
    features = row.get("features")
    if not isinstance(features, dict):
        return []
    tokens = features.get("tokens")
    if not isinstance(tokens, list):
        return []
    return [str(token).strip() for token in tokens if str(token).strip()]


def build_text(row: dict[str, object]) -> str:
    if get_family(row) == "shell":
        return "family_shell " + build_model_text(get_command(row), get_workdir(row))

    tokens = get_feature_tokens(row)
    if tokens:
        return " ".join(tokens)
    return "family_native input_missing_tokens"


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    by_family: dict[str, dict[str, int]] = {}
    positive = 0
    for row in rows:
        blocked = bool(row.get("label", {}).get("blocked")) if isinstance(row.get("label"), dict) else False
        positive += int(blocked)
        family = get_family(row)
        if family not in by_family:
            by_family[family] = {"rows": 0, "positive": 0, "negative": 0}
        by_family[family]["rows"] += 1
        if blocked:
            by_family[family]["positive"] += 1
        else:
            by_family[family]["negative"] += 1
    return {
        "rows": len(rows),
        "positive": positive,
        "negative": len(rows) - positive,
        "by_family": by_family,
    }


def infer_native_tool_identity(split_rows: dict[str, list[dict[str, object]]]) -> str:
    modes: set[str] = set()
    for rows in split_rows.values():
        for row in rows:
            if get_family(row) != "native":
                continue
            features = row.get("features")
            if not isinstance(features, dict):
                continue
            mode = features.get("tool_identity_mode")
            if isinstance(mode, str) and mode in {"hash", "raw", "none"}:
                modes.add(mode)
    if len(modes) == 1:
        return next(iter(modes))
    return "hash"


def extract_label(row: dict[str, object]) -> int:
    label = row.get("label")
    if isinstance(label, dict):
        return int(bool(label.get("blocked")))
    return 0


def make_xy(rows: Iterable[dict[str, object]]) -> tuple[list[str], np.ndarray]:
    texts = [build_text(row) for row in rows]
    labels = np.array([extract_label(row) for row in rows], dtype=np.int64)
    return texts, labels


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    steps: list[tuple[str, object]] = [
        (
            "vectorizer",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(args.ngram_min, args.ngram_max),
                min_df=args.min_df,
                max_features=args.max_features,
                lowercase=False,
                sublinear_tf=True,
            ),
        ),
    ]
    if args.select_k is not None:
        steps.append(("selector", SelectKBest(score_func=chi2, k=args.select_k)))
    steps.append(
        (
            "classifier",
            LogisticRegression(
                C=args.C,
                class_weight="balanced",
                max_iter=args.max_iter,
                random_state=args.seed,
                solver="liblinear",
            ),
        )
    )
    return Pipeline(steps=steps)


def positive_proba(model: Pipeline, texts: list[str]) -> np.ndarray:
    proba = model.predict_proba(texts)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise RuntimeError("Expected binary probability output from the classifier.")
    return proba[:, 1]


def warn_threshold(block_threshold: float) -> float:
    return max(WARN_THRESHOLD_FLOOR, block_threshold * 0.5)


def best_threshold(y_true: np.ndarray, y_score: np.ndarray, beta: float) -> tuple[float, dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, {"precision": 0.0, "recall": 0.0, "f_beta": 0.0}

    beta2 = beta * beta
    eps = 1e-12
    precision = precision[:-1]
    recall = recall[:-1]
    f_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)

    best_index = 0
    best_key = (-1.0, -1.0, -1.0)
    for index, (score, prec, thr) in enumerate(zip(f_beta, precision, thresholds)):
        key = (float(score), float(prec), float(thr))
        if key > best_key:
            best_key = key
            best_index = index

    return float(thresholds[best_index]), {
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f_beta": float(f_beta[best_index]),
    }


def target_precision_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_precision: float,
    beta: float,
) -> tuple[float, dict[str, object]]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        threshold, stats = best_threshold(y_true, y_score, beta)
        return threshold, {
            "method": "fbeta_fallback",
            "target_precision": float(target_precision),
            "fallback_used": True,
            "fallback_reason": "validation_split_has_no_thresholds",
            "selection_beta": float(beta),
            "selection": stats,
        }

    precision = precision[:-1]
    recall = recall[:-1]
    qualifying = np.where(precision >= target_precision)[0]
    if len(qualifying) > 0:
        best_index = max(
            qualifying,
            key=lambda index: (float(recall[index]), -float(thresholds[index])),
        )
        return float(thresholds[best_index]), {
            "method": "target_precision",
            "target_precision": float(target_precision),
            "fallback_used": False,
            "fallback_reason": None,
            "selection_beta": float(beta),
            "selection": {
                "precision": float(precision[best_index]),
                "recall": float(recall[best_index]),
            },
        }

    threshold, stats = best_threshold(y_true, y_score, beta)
    return threshold, {
        "method": "fbeta_fallback",
        "target_precision": float(target_precision),
        "fallback_used": True,
        "fallback_reason": "no_validation_threshold_meets_target_precision",
        "selection_beta": float(beta),
        "selection": stats,
    }


def evaluate(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, object]:
    y_pred = (y_score >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = None
    try:
        avg_precision = average_precision_score(y_true, y_score)
    except ValueError:
        avg_precision = None

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "average_precision": None if avg_precision is None else float(avg_precision),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
    }


def evaluate_by_family(
    rows: list[dict[str, object]],
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, object]]:
    indexes_by_family: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        indexes_by_family.setdefault(get_family(row), []).append(index)

    metrics: dict[str, dict[str, object]] = {}
    for family, indexes in sorted(indexes_by_family.items()):
        index_array = np.asarray(indexes, dtype=np.int64)
        metrics[family] = evaluate(y_true[index_array], y_score[index_array], threshold)
    return metrics


def onnx_positive_proba(session: ort.InferenceSession, texts: list[str]) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    feed = {input_name: np.asarray(texts, dtype=object).reshape(-1, 1)}
    outputs = session.run(None, feed)
    output_names = [output.name.lower() for output in session.get_outputs()]

    chosen = None
    for index, name in enumerate(output_names):
        if "prob" in name:
            chosen = outputs[index]
            break
    if chosen is None:
        chosen = outputs[-1]

    array = np.asarray(chosen)
    if array.dtype == object:
        probs: list[float] = []
        for item in array:
            if isinstance(item, dict):
                if 1 in item:
                    probs.append(float(item[1]))
                elif "1" in item:
                    probs.append(float(item["1"]))
                else:
                    probs.append(float(max(item.values())))
            else:
                probs.append(float(item))
        return np.asarray(probs, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] >= 2:
        return array[:, 1].astype(np.float64)
    if array.ndim == 1:
        return array.astype(np.float64)
    raise RuntimeError(f"Unsupported ONNX probability output shape: {array.shape}")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_onnx_model(pipeline: Pipeline, onnx_path: Path) -> None:
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=[("text", StringTensorType([None, 1]))],
        options={id(pipeline.named_steps["classifier"]): {"zipmap": False}},
        target_opset=17,
    )
    onnx.save_model(onnx_model, onnx_path)


def build_manifest(
    args: argparse.Namespace,
    split_rows: dict[str, list[dict[str, object]]],
    threshold: float,
    threshold_stats: dict[str, float],
    val_metrics: dict[str, object],
    val_family_metrics: dict[str, dict[str, object]],
    sklearn_test_metrics: dict[str, object],
    sklearn_test_family_metrics: dict[str, dict[str, object]],
    onnx_metrics: dict[str, object],
    onnx_family_metrics: dict[str, dict[str, object]],
    parity_max_abs_diff: float,
    sklearn_path: Path,
    onnx_path: Path,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": 1,
        "model_type": "tfidf-word-logreg",
        "input_dir": str(args.input_dir),
        "text_template": "space-separated token stream derived from shell command features or structured tool input features",
        "feature_extraction": {
            "shell_prefix": "family_shell",
            "native_tokens": "features.tokens",
            "native_tool_identity": infer_native_tool_identity(split_rows),
        },
        "vectorizer": {
            "analyzer": "word",
            "ngram_range": [args.ngram_min, args.ngram_max],
            "min_df": args.min_df,
            "max_features": args.max_features,
            "lowercase": False,
            "sublinear_tf": True,
        },
        "classifier": {
            "name": "LogisticRegression",
            "C": args.C,
            "class_weight": "balanced",
            "max_iter": args.max_iter,
            "solver": "liblinear",
            "random_state": args.seed,
        },
        "threshold": {
            "block_threshold": threshold,
            "warn_threshold": warn_threshold(threshold),
            "selection_method": threshold_stats["method"],
            "target_precision": threshold_stats["target_precision"],
            "fallback_used": threshold_stats["fallback_used"],
            "fallback_reason": threshold_stats["fallback_reason"],
            "selection_beta": threshold_stats["selection_beta"],
            "selection": threshold_stats["selection"],
            "validation": threshold_stats,
        },
        "dataset": {name: summarize_rows(rows) for name, rows in split_rows.items()},
        "metrics": {
            "val": val_metrics,
            "val_by_family": val_family_metrics,
            "test_sklearn": sklearn_test_metrics,
            "test_sklearn_by_family": sklearn_test_family_metrics,
            "test_onnx": onnx_metrics,
            "test_onnx_by_family": onnx_family_metrics,
            "onnx_parity_max_abs_diff": parity_max_abs_diff,
        },
        "artifacts": {
            "sklearn": str(sklearn_path),
            "onnx": str(onnx_path),
            "threshold": str(args.output_dir / "threshold.json"),
        },
    }

    if args.select_k is not None:
        manifest["selector"] = {
            "name": "SelectKBest",
            "score_func": "chi2",
            "k": args.select_k,
        }

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    split_rows = load_splits(args.input_dir)

    train_texts, y_train = make_xy(split_rows["train"])
    val_texts, y_val = make_xy(split_rows["val"])
    test_texts, y_test = make_xy(split_rows["test"])

    pipeline = build_pipeline(args)
    pipeline.fit(train_texts, y_train)

    val_score = positive_proba(pipeline, val_texts)
    threshold, threshold_stats = target_precision_threshold(
        y_val,
        val_score,
        args.target_precision,
        args.threshold_beta,
    )

    val_metrics = evaluate(y_val, val_score, threshold)
    val_family_metrics = evaluate_by_family(split_rows["val"], y_val, val_score, threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sklearn_path = args.output_dir / "model.joblib"
    onnx_path = args.output_dir / "model.onnx"
    write_json(
        args.output_dir / "threshold.json",
        {"block_threshold": threshold, "warn_threshold": warn_threshold(threshold)},
    )

    dump(pipeline, sklearn_path)
    export_onnx_model(pipeline, onnx_path)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_score = onnx_positive_proba(session, test_texts)
    onnx_metrics = evaluate(y_test, onnx_score, threshold)
    onnx_family_metrics = evaluate_by_family(split_rows["test"], y_test, onnx_score, threshold)
    sklearn_test_score = positive_proba(pipeline, test_texts)
    sklearn_test_metrics = evaluate(y_test, sklearn_test_score, threshold)
    sklearn_test_family_metrics = evaluate_by_family(split_rows["test"], y_test, sklearn_test_score, threshold)
    parity_max_abs_diff = float(np.max(np.abs(onnx_score - sklearn_test_score)))

    manifest = build_manifest(
        args,
        split_rows,
        threshold,
        threshold_stats,
        val_metrics,
        val_family_metrics,
        sklearn_test_metrics,
        sklearn_test_family_metrics,
        onnx_metrics,
        onnx_family_metrics,
        parity_max_abs_diff,
        sklearn_path,
        onnx_path,
    )
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
