# heros_eval.py
from __future__ import annotations

import time
import pickle
from pathlib import Path
import numpy as np

# HEROS import (supports two possible package layouts)
try:
    from src.skheros.heros import HEROS  # type: ignore
except ImportError:
    try:
        from skheros.heros import HEROS  # type: ignore
    except ImportError as e:
        HEROS = None  # type: ignore
        _HEROS_IMPORT_ERROR = e
    else:
        _HEROS_IMPORT_ERROR = None
else:
    _HEROS_IMPORT_ERROR = None


def _require_heros():
    if HEROS is None:
        raise ImportError(
            "HEROS could not be imported. Ensure it is installed and importable.\n"
            f"Original error: {_HEROS_IMPORT_ERROR}"
        )


def train_heros(
    X_train: np.ndarray,
    y_train: np.ndarray,
    row_id: np.ndarray,
    cat_feat_indexes: list[int],
    *,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[Any, float]:
    """
    Train HEROS and return (model, training_time_seconds).
    """
    _require_heros()

    heros = HEROS(
        outcome_type="class",
        iterations=50000,
        pop_size=500,
        model_iterations=100,
        model_pop_size=100,
        nu=1,
        beta=0.2,
        theta_sel=0.5,
        cross_prob=0.8,
        mut_prob=0.04,
        merge_prob=0.1,
        subsumption="both",
        compaction="sub",
        random_state=random_state,
        track_performance=1000,
        model_tracking=True,
        verbose=verbose,
    )

    start = time.time()
    heros.fit(X_train, y_train, row_id, cat_feat_indexes=cat_feat_indexes)
    return heros, (time.time() - start)


def evaluate_heros(
    heros_model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate HEROS and return key metrics.
    """
    best_model_idx = heros_model.auto_select_top_model(X_test, y_test, verbose=False)
    preds_test = heros_model.predict(X_test, whole_rule_pop=False, target_model=best_model_idx)
    preds_train = heros_model.predict(X_train, whole_rule_pop=False, target_model=best_model_idx)
    rules = heros_model.get_model_rules(best_model_idx)

    return {
        "test_accuracy": float(np.mean(preds_test == y_test)),
        "train_accuracy": float(np.mean(preds_train == y_train)),
        "num_rules": int(len(rules)),
        "best_model_idx": int(best_model_idx),
    }


def run_heros_for_dataset(
    dataset_name: str,
    dataset_dir: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    row_id: np.ndarray,
    *,
    random_state: int = 42,
    verbose: bool = True,
    save_pickle: bool = True,
) -> Dict[str, Any]:
    """
    Train + evaluate HEROS for one dataset. Optionally save pickle to:
      <dataset_dir>/heros_model.pickle
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)

    cat_feat_indexes = list(range(X_train.shape[1]))
    model, train_time = train_heros(
        X_train, y_train, row_id, cat_feat_indexes, random_state=random_state, verbose=verbose
    )
    metrics = evaluate_heros(model, X_test, y_test, X_train, y_train)
    metrics["training_time"] = float(train_time)

    if save_pickle:
        with open(dataset_dir / "heros_model.pickle", "wb") as f:
            pickle.dump(model, f)

    return metrics
