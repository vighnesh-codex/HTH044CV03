"""
Quality Anomaly Detection Module
Provides statistical cohort outlier detection using scikit-learn IsolationForest.
Gracefully transitions between 'Baseline CV mode' and 'Historical anomaly mode'.
"""

from typing import Dict, Any, List, Optional
import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


MIN_HISTORICAL_SAMPLES = 8


def extract_anomaly_features(
    defects: Dict[str, Any],
    multiscale_info: Dict[str, Any] = None,
    structure_info: Dict[str, Any] = None
) -> List[float]:
    """
    Extracts standardized 8-dimensional feature vector for cohort modeling.
    """
    blur_score = float(defects["blur"]["score"])
    contrast_score = float(defects["contrast"]["score"])
    noise_score = float(defects["noise"]["score"])
    skew_angle = abs(float(defects["skew"].get("angle", 0.0)))
    comp_score = float(defects.get("missing_section", {}).get("score", 100.0))

    if multiscale_info:
        reg_var = float(multiscale_info.get("regional_variance", 0.0))
        # Mean edge density across tiles
        tiles = multiscale_info.get("tile_features", [])
        edge_densities = [t.get("edge_density", 0.0) for t in tiles]
        edge_density = float(np.mean(edge_densities)) if edge_densities else 0.05
        bright_vals = [t.get("brightness", 128.0) for t in tiles]
        brightness = float(np.mean(bright_vals)) if bright_vals else 128.0
    else:
        reg_var = 0.0
        edge_density = 0.05
        brightness = 128.0

    return [
        blur_score,
        contrast_score,
        noise_score,
        skew_angle,
        brightness,
        edge_density,
        reg_var,
        comp_score
    ]


def evaluate_anomaly(
    feature_vector: List[float],
    historical_matrix: Optional[List[List[float]]] = None
) -> Dict[str, Any]:
    """
    Evaluates whether document quality signature is an outlier within intake history.
    """
    sample_count = len(historical_matrix) if historical_matrix is not None else 0

    if not SKLEARN_AVAILABLE or historical_matrix is None or sample_count < MIN_HISTORICAL_SAMPLES:
        return {
            "mode": "Baseline CV mode",
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "confidence": 1.0,
            "sample_count": sample_count,
            "min_required": MIN_HISTORICAL_SAMPLES,
            "explanation": (
                f"Operating in Baseline CV mode. Historical sample size ({sample_count}/{MIN_HISTORICAL_SAMPLES}) "
                "is below threshold for empirical cohort anomaly modeling."
            )
        }

    try:
        X = np.array(historical_matrix, dtype=np.float32)
        current_x = np.array([feature_vector], dtype=np.float32)

        # Isolation Forest with 10% contamination expectation
        iso = IsolationForest(
            n_estimators=50,
            contamination=0.10,
            random_state=42
        )
        iso.fit(X)

        pred = int(iso.predict(current_x)[0])  # -1 for anomaly, 1 for inlier
        raw_score = float(-iso.score_samples(current_x)[0])  # Higher = more anomalous

        is_anomaly = (pred == -1)
        explanation = (
            "Document quality profile deviates significantly from intake distribution."
            if is_anomaly else
            "Document quality profile is consistent with historical intake baseline."
        )

        return {
            "mode": "Historical anomaly mode",
            "is_anomaly": is_anomaly,
            "anomaly_score": round(raw_score, 3),
            "confidence": 0.88,
            "sample_count": sample_count,
            "min_required": MIN_HISTORICAL_SAMPLES,
            "explanation": explanation
        }
    except Exception as e:
        return {
            "mode": "Baseline CV mode",
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "confidence": 1.0,
            "sample_count": sample_count,
            "min_required": MIN_HISTORICAL_SAMPLES,
            "explanation": f"Fell back to Baseline CV mode due to modeling error: {str(e)}"
        }
