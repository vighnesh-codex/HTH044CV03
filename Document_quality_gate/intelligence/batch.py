"""
Batch Quality Analysis Module
Processes multiple intake documents in batch, tracks execution throughput,
and compiles aggregate defect epidemiology and routing statistics.
"""

from typing import Dict, Any, List, Tuple, Callable, Optional
import numpy as np
from intelligence.engine import analyze_document_quality
from storage.database import save_analysis_record


def process_document_batch(
    document_items: List[Tuple[str, np.ndarray]],
    save_to_db: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    Executes batch analysis over multiple document images.

    Args:
        document_items: List of tuples (filename, numpy_image)
        save_to_db: Whether to persist each analysis to SQLite history
        progress_callback: Optional callback(current_idx, total_count, filename)

    Returns:
        Dict with total, pass/review/reject counts and percentages,
        average_quality, most_common_defect, second_most_common_defect,
        and per-document itemized records.
    """
    total = len(document_items)
    if total == 0:
        return {
            "total": 0,
            "pass_count": 0, "pass_pct": 0.0,
            "review_count": 0, "review_pct": 0.0,
            "reject_count": 0, "reject_pct": 0.0,
            "average_quality": 0.0,
            "most_common_defect": "None",
            "second_most_common_defect": "None",
            "defect_counts": {},
            "records": []
        }

    records: List[Dict[str, Any]] = []
    pass_cnt = 0
    review_cnt = 0
    reject_cnt = 0
    quality_sum = 0.0

    defect_frequencies = {
        "Blur / Softness": 0,
        "Low Contrast": 0,
        "Noise / Grain": 0,
        "Document Skew": 0,
        "Handwriting Interference": 0,
        "Missing Sections": 0
    }

    for idx, (filename, image) in enumerate(document_items):
        if progress_callback:
            progress_callback(idx + 1, total, filename)

        try:
            analysis = analyze_document_quality(image)
            qi = analysis["quality_index"]
            dec = analysis["decision"]
            defects = analysis["defects"]

            quality_sum += qi
            if dec == "PASS":
                pass_cnt += 1
            elif dec == "NEEDS REVIEW":
                review_cnt += 1
            else:
                reject_cnt += 1

            # Count defect occurrences
            if defects["blur"]["score"] < 60:
                defect_frequencies["Blur / Softness"] += 1
            if defects["contrast"]["score"] < 60:
                defect_frequencies["Low Contrast"] += 1
            if defects["noise"]["score"] < 65:
                defect_frequencies["Noise / Grain"] += 1
            if abs(defects["skew"]["angle"]) > 2.0:
                defect_frequencies["Document Skew"] += 1
            if defects["handwriting"].get("detected", False):
                defect_frequencies["Handwriting Interference"] += 1
            if defects.get("missing_sections"):
                defect_frequencies["Missing Sections"] += 1

            doc_id = None
            if save_to_db:
                doc_id = save_analysis_record(filename, analysis)

            records.append({
                "filename": filename,
                "document_id": doc_id,
                "quality_index": qi,
                "decision": dec,
                "dominant_risk": analysis["risk_vector"]["primary_risk"]["label"],
                "blur_score": defects["blur"]["score"],
                "contrast_score": defects["contrast"]["score"],
                "skew_angle": defects["skew"]["angle"],
                "noise_score": defects["noise"]["score"],
                "handwriting_detected": defects["handwriting"].get("detected", False),
                "missing_sections": ", ".join(defects.get("missing_sections", [])),
                "fingerprint": analysis["fingerprint"]["quality_fingerprint"],
                "full_analysis": analysis
            })
        except Exception as e:
            records.append({
                "filename": filename,
                "document_id": None,
                "quality_index": 0.0,
                "decision": "ERROR",
                "dominant_risk": f"Processing Error: {str(e)}",
                "blur_score": 0, "contrast_score": 0, "skew_angle": 0.0, "noise_score": 0,
                "handwriting_detected": False, "missing_sections": "Error",
                "fingerprint": "N/A", "full_analysis": None
            })

    # Rank most common defects
    sorted_defects = sorted(defect_frequencies.items(), key=lambda kv: kv[1], reverse=True)
    most_common = sorted_defects[0][0] if sorted_defects and sorted_defects[0][1] > 0 else "None"
    second_common = sorted_defects[1][0] if len(sorted_defects) > 1 and sorted_defects[1][1] > 0 else "None"

    return {
        "total": total,
        "pass_count": pass_cnt,
        "pass_pct": round(pass_cnt / total * 100.0, 1),
        "review_count": review_cnt,
        "review_pct": round(review_cnt / total * 100.0, 1),
        "reject_count": reject_cnt,
        "reject_pct": round(reject_cnt / total * 100.0, 1),
        "average_quality": round(quality_sum / total, 1),
        "most_common_defect": most_common,
        "second_most_common_defect": second_common,
        "defect_counts": defect_frequencies,
        "records": records
    }
