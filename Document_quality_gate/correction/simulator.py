"""
Correction Simulation & Validation Loop Engine
Executes controlled 'What-If' quality enhancement simulations and re-analyzes
the candidate images through the full CV pipeline to empirically validate or reject improvements.
"""

from typing import Dict, Any, Callable, List
import numpy as np
from correction.enhancement import apply_contrast_enhancement
from correction.denoise import apply_denoising
from correction.deskew import apply_deskew
from correction.illumination import apply_illumination_normalization


def simulate_corrections(
    image: np.ndarray,
    baseline_analysis: Dict[str, Any],
    analyzer_func: Callable[[np.ndarray], Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Simulates quality repairs without modifying the original document.

    Validation Loop:
    Original -> Candidate Correction -> Re-analysis -> Validation
    If post-correction quality improves (delta > 0), the correction is marked as VALIDATED.
    If post-correction quality degrades or stays flat (delta <= 0), it is REJECTED.
    """
    baseline_score = float(baseline_analysis.get("quality_index", 0.0))
    skew_angle = float(baseline_analysis.get("defects", {}).get("skew", {}).get("angle", 0.0))

    simulations: List[Dict[str, Any]] = []

    correction_candidates = [
        ("contrast", "Contrast Enhancement (CLAHE)", lambda img: apply_contrast_enhancement(img)),
        ("denoise", "Image Denoising (Bilateral)", lambda img: apply_denoising(img)),
        ("deskew", f"Document Deskew ({skew_angle:+.1f} deg)", lambda img: apply_deskew(img, skew_angle)),
        ("illumination", "Illumination Normalization", lambda img: apply_illumination_normalization(img)),
    ]

    for corr_id, label, corr_fn in correction_candidates:
        try:
            corr_image = corr_fn(image)
            reanalysis = analyzer_func(corr_image)
            new_score = float(reanalysis.get("quality_index", 0.0))
            improvement = round(new_score - baseline_score, 1)

            # Strict validation check
            is_validated = improvement > 0.0
            status_text = "VALIDATED IMPROVEMENT" if is_validated else "REJECTED (DEGRADATION / NO GAIN)"

            # Metric changes
            b_def = baseline_analysis.get("defects", {})
            r_def = reanalysis.get("defects", {})
            changed_metrics = {
                "blur": r_def.get("blur", {}).get("score", 0) - b_def.get("blur", {}).get("score", 0),
                "contrast": r_def.get("contrast", {}).get("score", 0) - b_def.get("contrast", {}).get("score", 0),
                "noise": r_def.get("noise", {}).get("score", 0) - b_def.get("noise", {}).get("score", 0),
                "skew": abs(b_def.get("skew", {}).get("angle", 0.0)) - abs(r_def.get("skew", {}).get("angle", 0.0))
            }

            simulations.append({
                "id": corr_id,
                "label": label,
                "corrected_image": corr_image,
                "new_score": new_score,
                "improvement": improvement,
                "is_validated": is_validated,
                "validation_status": status_text,
                "changed_metrics": changed_metrics,
                "reanalysis": reanalysis
            })
        except Exception as e:
            simulations.append({
                "id": corr_id,
                "label": label,
                "corrected_image": None,
                "new_score": baseline_score,
                "improvement": 0.0,
                "is_validated": False,
                "validation_status": f"FAILED ({str(e)})",
                "changed_metrics": {},
                "reanalysis": None
            })

    # Sort candidates by validated improvement descending
    validated_candidates = [s for s in simulations if s["is_validated"]]
    validated_candidates.sort(key=lambda s: s["improvement"], reverse=True)

    if validated_candidates:
        best_candidate = validated_candidates[0]
        recommendation = (
            f"Simulated {best_candidate['label']} achieved highest validated improvement "
            f"({best_candidate['improvement']:+0.1f} points: {baseline_score:.1f} -> {best_candidate['new_score']:.1f})."
        )
    else:
        best_candidate = None
        recommendation = (
            "No simulated correction produced an empirical quality improvement. "
            "Original document quality is optimal or defect requires hardware rescan."
        )

    return {
        "original_score": baseline_score,
        "simulations": simulations,
        "best_candidate": best_candidate,
        "recommendation": recommendation
    }
