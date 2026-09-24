"""
Document-to-Document Comparison Module
Computes multidimensional delta metrics, defect regressions, and improvements
between two analyzed document versions.
"""

from typing import Dict, Any, List


def compare_document_analyses(
    analysis_1: Dict[str, Any],
    analysis_2: Dict[str, Any],
    label_1: str = "Version 1",
    label_2: str = "Version 2"
) -> Dict[str, Any]:
    """
    Compares two quality analyses and produces structured metric deltas.
    """
    qi_1 = float(analysis_1.get("quality_index", 0.0))
    qi_2 = float(analysis_2.get("quality_index", 0.0))
    total_delta = round(qi_2 - qi_1, 1)

    def_1 = analysis_1.get("defects", {})
    def_2 = analysis_2.get("defects", {})

    b1, b2 = float(def_1.get("blur", {}).get("score", 0)), float(def_2.get("blur", {}).get("score", 0))
    c1, c2 = float(def_1.get("contrast", {}).get("score", 0)), float(def_2.get("contrast", {}).get("score", 0))
    n1, n2 = float(def_1.get("noise", {}).get("score", 0)), float(def_2.get("noise", {}).get("score", 0))
    s1, s2 = abs(float(def_1.get("skew", {}).get("angle", 0))), abs(float(def_2.get("skew", {}).get("angle", 0)))
    comp1, comp2 = float(def_1.get("missing_section", {}).get("score", 100)), float(def_2.get("missing_section", {}).get("score", 100))

    metric_comparisons: List[Dict[str, Any]] = [
        {
            "metric": "Blur / Sharpness",
            "val_1": b1,
            "val_2": b2,
            "delta": round(b2 - b1, 1),
            "improved": (b2 > b1),
            "unit": "pts"
        },
        {
            "metric": "Contrast",
            "val_1": c1,
            "val_2": c2,
            "delta": round(c2 - c1, 1),
            "improved": (c2 > c1),
            "unit": "pts"
        },
        {
            "metric": "Noise Score",
            "val_1": n1,
            "val_2": n2,
            "delta": round(n2 - n1, 1),
            "improved": (n2 > n1),
            "unit": "pts"
        },
        {
            "metric": "Skew Angle",
            "val_1": s1,
            "val_2": s2,
            "delta": round(s1 - s2, 1),  # Less skew angle is better
            "improved": (s2 < s1),
            "unit": "deg"
        },
        {
            "metric": "Completeness",
            "val_1": comp1,
            "val_2": comp2,
            "delta": round(comp2 - comp1, 1),
            "improved": (comp2 > comp1),
            "unit": "%"
        }
    ]

    sub_1 = analysis_1.get("subscores", {})
    sub_2 = analysis_2.get("subscores", {})

    subscore_comparisons = {
        "visual_quality": {"v1": sub_1.get("visual_quality", 0), "v2": sub_2.get("visual_quality", 0)},
        "structural_quality": {"v1": sub_1.get("structural_quality", 0), "v2": sub_2.get("structural_quality", 0)},
        "readability_risk": {"v1": sub_1.get("readability_risk", 0), "v2": sub_2.get("readability_risk", 0)},
        "completeness": {"v1": sub_1.get("completeness", 0), "v2": sub_2.get("completeness", 0)},
        "interference_risk": {"v1": sub_1.get("interference_risk", 0), "v2": sub_2.get("interference_risk", 0)}
    }

    if total_delta > 0:
        verdict = f"{label_2} demonstrates significant technical quality improvement (+{total_delta} points)."
    elif total_delta < 0:
        verdict = f"{label_2} shows quality degradation ({total_delta} points relative to {label_1})."
    else:
        verdict = f"{label_1} and {label_2} have identical quality indices."

    return {
        "label_1": label_1,
        "label_2": label_2,
        "score_1": qi_1,
        "score_2": qi_2,
        "total_delta": total_delta,
        "metrics": metric_comparisons,
        "subscores": subscore_comparisons,
        "verdict": verdict,
        "decision_1": analysis_1.get("decision", "N/A"),
        "decision_2": analysis_2.get("decision", "N/A")
    }
