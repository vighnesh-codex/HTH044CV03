"""
Persistent Analysis History & Analytics Database
Lightweight SQLite storage for document intake quality records,
audit fingerprints, correction metrics, and cohort anomaly training features.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


DB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "document_quality.db")


def get_db_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            quality_index REAL NOT NULL,
            decision TEXT NOT NULL,
            dominant_risk TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            blur_score REAL,
            contrast_score REAL,
            noise_score REAL,
            skew_angle REAL,
            handwriting_detected INTEGER,
            handwriting_confidence REAL,
            completeness_score REAL,
            regional_variance REAL,
            correction_attempted INTEGER DEFAULT 0,
            before_score REAL,
            after_score REAL,
            best_correction TEXT,
            metrics_json TEXT,
            features_json TEXT
        );
    """)
    conn.commit()
    conn.close()


def save_analysis_record(
    filename: str,
    analysis_result: Dict[str, Any],
    correction_result: Optional[Dict[str, Any]] = None,
    document_id: Optional[str] = None
) -> str:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    if not document_id:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        document_id = f"DOC_{timestamp_str}"

    now_iso = datetime.now().isoformat()

    qi = float(analysis_result.get("quality_index", 0.0))
    decision = analysis_result.get("decision", "NEEDS REVIEW")
    dom_risk = analysis_result.get("risk_vector", {}).get("primary_risk", {}).get("label", "Unknown")
    fp = analysis_result.get("fingerprint", {}).get("quality_fingerprint", "N/A")

    defects = analysis_result.get("defects", {})
    blur_score = float(defects.get("blur", {}).get("score", 0))
    contrast_score = float(defects.get("contrast", {}).get("score", 0))
    noise_score = float(defects.get("noise", {}).get("score", 0))
    skew_angle = float(defects.get("skew", {}).get("angle", 0.0))

    hw = defects.get("handwriting", {})
    hw_detected = 1 if hw.get("detected", False) else 0
    hw_conf = float(hw.get("confidence", 0.0))

    comp_score = float(defects.get("missing_section", {}).get("score", 100))
    reg_var = float(analysis_result.get("multiscale", {}).get("regional_variance", 0.0))

    correction_attempted = 0
    before_score = qi
    after_score = qi
    best_corr_label = "None"

    if correction_result:
        correction_attempted = 1
        before_score = float(correction_result.get("original_score", qi))
        best_cand = correction_result.get("best_candidate")
        if best_cand:
            after_score = float(best_cand.get("new_score", qi))
            best_corr_label = str(best_cand.get("label", "None"))

    # Feature vector for anomaly model
    features = [
        blur_score,
        contrast_score,
        noise_score,
        abs(skew_angle),
        float(analysis_result.get("multiscale", {}).get("tile_features", [{}])[0].get("brightness", 128.0) if analysis_result.get("multiscale", {}).get("tile_features") else 128.0),
        0.05,
        reg_var,
        comp_score
    ]

    # Metrics summary for lightweight JSON
    metrics_summary = {
        "subscores": analysis_result.get("subscores", {}),
        "reasons": analysis_result.get("reasons", []),
        "suggestions": analysis_result.get("suggestions", []),
        "decision_trigger": analysis_result.get("decision_trigger", "")
    }

    cursor.execute("""
        INSERT OR REPLACE INTO document_history (
            document_id, filename, timestamp, quality_index, decision, dominant_risk,
            fingerprint, blur_score, contrast_score, noise_score, skew_angle,
            handwriting_detected, handwriting_confidence, completeness_score,
            regional_variance, correction_attempted, before_score, after_score,
            best_correction, metrics_json, features_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        document_id, filename, now_iso, qi, decision, dom_risk,
        fp, blur_score, contrast_score, noise_score, skew_angle,
        hw_detected, hw_conf, comp_score, reg_var,
        correction_attempted, before_score, after_score,
        best_corr_label, json.dumps(metrics_summary), json.dumps(features)
    ))

    conn.commit()
    conn.close()
    return document_id


def get_history(limit: int = 100) -> List[Dict[str, Any]]:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM document_history
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_record_by_id(document_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM document_history WHERE document_id = ?", (document_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_feature_vectors() -> List[List[float]]:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT features_json FROM document_history WHERE features_json IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    vectors = []
    for r in rows:
        try:
            vec = json.loads(r["features_json"])
            if isinstance(vec, list) and len(vec) >= 8:
                vectors.append(vec)
        except Exception:
            continue
    return vectors


def get_analytics_summary() -> Dict[str, Any]:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total, AVG(quality_index) as avg_qi FROM document_history")
    total_row = cursor.fetchone()
    total = total_row["total"] if total_row else 0
    avg_qi = float(total_row["avg_qi"]) if total_row and total_row["avg_qi"] is not None else 0.0

    if total == 0:
        conn.close()
        return {
            "total_documents": 0,
            "avg_quality": 0.0,
            "pass_count": 0, "pass_pct": 0.0,
            "review_count": 0, "review_pct": 0.0,
            "reject_count": 0, "reject_pct": 0.0,
            "dominant_risks": {},
            "correction_attempts": 0,
            "correction_successes": 0,
            "correction_success_rate": 0.0,
            "avg_before_score": 0.0,
            "avg_after_score": 0.0
        }

    # Decision counts
    cursor.execute("SELECT decision, COUNT(*) as c FROM document_history GROUP BY decision")
    dec_counts = {r["decision"]: r["c"] for r in cursor.fetchall()}
    pass_cnt = dec_counts.get("PASS", 0)
    review_cnt = dec_counts.get("NEEDS REVIEW", 0)
    reject_cnt = dec_counts.get("REJECT", 0)

    # Dominant risks
    cursor.execute("SELECT dominant_risk, COUNT(*) as c FROM document_history GROUP BY dominant_risk ORDER BY c DESC")
    dom_risks = {r["dominant_risk"]: r["c"] for r in cursor.fetchall()}

    # Corrections
    cursor.execute("""
        SELECT COUNT(*) as attempts,
               SUM(CASE WHEN after_score > before_score THEN 1 ELSE 0 END) as successes,
               AVG(before_score) as avg_before,
               AVG(after_score) as avg_after
        FROM document_history
        WHERE correction_attempted = 1
    """)
    corr_row = cursor.fetchone()
    attempts = corr_row["attempts"] if corr_row and corr_row["attempts"] else 0
    successes = corr_row["successes"] if corr_row and corr_row["successes"] else 0
    avg_before = float(corr_row["avg_before"]) if corr_row and corr_row["avg_before"] else 0.0
    avg_after = float(corr_row["avg_after"]) if corr_row and corr_row["avg_after"] else 0.0
    corr_rate = round((successes / attempts * 100.0), 1) if attempts > 0 else 0.0

    conn.close()

    return {
        "total_documents": total,
        "avg_quality": round(avg_qi, 1),
        "pass_count": pass_cnt,
        "pass_pct": round(pass_cnt / total * 100.0, 1),
        "review_count": review_cnt,
        "review_pct": round(review_cnt / total * 100.0, 1),
        "reject_count": reject_cnt,
        "reject_pct": round(reject_cnt / total * 100.0, 1),
        "dominant_risks": dom_risks,
        "correction_attempts": attempts,
        "correction_successes": successes,
        "correction_success_rate": corr_rate,
        "avg_before_score": round(avg_before, 1),
        "avg_after_score": round(avg_after, 1)
    }
