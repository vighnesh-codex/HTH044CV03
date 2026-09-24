"""
Document Quality Fingerprint Module
Generates compact, deterministic quality profiles and hashes for document quality
identification, duplicate comparison, and audit trail verification.
(Not intended for cryptographic security claims).
"""

import hashlib
from typing import Dict, Any


def generate_quality_fingerprint(
    quality_index: float,
    defects: Dict[str, Any],
    decision: str,
    dominant_risk: str,
    multiscale_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Constructs a deterministic fingerprint from quantized CV quality descriptors.
    """
    blur_score = round(float(defects["blur"]["score"]), 1)
    contrast_score = round(float(defects["contrast"]["score"]), 1)
    noise_score = round(float(defects["noise"]["score"]), 1)
    skew_angle = round(float(defects["skew"].get("angle", 0.0)), 1)
    hw_detected = defects["handwriting"].get("detected", False)
    hw_conf = round(float(defects["handwriting"].get("confidence", 0.0)), 2)
    comp_score = round(float(defects.get("missing_section", {}).get("score", 100.0)), 1)
    qi = round(float(quality_index), 1)

    reg_var = 0.0
    if multiscale_info:
        reg_var = round(float(multiscale_info.get("regional_variance", 0.0)), 1)

    # Canonical representation string for deterministic hashing
    canonical_repr = (
        f"QI={qi}|BLUR={blur_score}|CONT={contrast_score}|NOISE={noise_score}|"
        f"SKEW={skew_angle}|HW={1 if hw_detected else 0}:{hw_conf}|"
        f"COMP={comp_score}|RVAR={reg_var}|DEC={decision}|DOM={dominant_risk}"
    )

    hasher = hashlib.sha256()
    hasher.update(canonical_repr.encode("utf-8"))
    full_hash = hasher.hexdigest()

    # Human-readable compact token
    tag = decision.replace(" ", "_")
    compact_fp = f"DQF-{full_hash[:8].upper()}-{tag}-{int(round(qi))}"

    return {
        "quality_fingerprint": compact_fp,
        "fingerprint_hash": full_hash,
        "quality_index": qi,
        "dominant_risk": dominant_risk,
        "decision": decision,
        "profiles": {
            "blur_profile": blur_score,
            "contrast_profile": contrast_score,
            "noise_profile": noise_score,
            "skew_profile": skew_angle,
            "handwriting_risk": hw_conf if hw_detected else 0.0,
            "completeness": comp_score,
            "regional_variance": reg_var
        },
        "canonical_string": canonical_repr
    }
