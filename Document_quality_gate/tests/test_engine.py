"""
Automated Test Suite for Document Quality Intelligence Engine
Covers all 14 required hackathon test scenarios plus regression/backward compatibility verification.
"""

import os
import unittest
import numpy as np
import cv2

from cv.preprocessing import validate_and_preprocess
from cv.blur import analyze_blur
from cv.contrast import analyze_contrast
from cv.skew import analyze_skew
from cv.noise import analyze_noise
from cv.handwriting import analyze_handwriting
from cv.completeness import analyze_completeness
from cv.structure import analyze_structure

from intelligence.engine import analyze_document_quality
from intelligence.sensitivity import analyze_quality_sensitivity
from intelligence.fingerprint import generate_quality_fingerprint
from intelligence.batch import process_document_batch
from intelligence.comparison import compare_document_analyses
from correction.simulator import simulate_corrections
from correction.enhancement import apply_contrast_enhancement
from correction.deskew import apply_deskew

from modules.analyzer import analyze_document
from backend_modules.scoring import calculate_quality_score
from backend_modules.routing import route_document
from backend_modules.suggestion import generate_suggestions


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Datas")


def create_synthetic_clean_document() -> np.ndarray:
    """Generates a high-resolution, high-contrast, perfectly aligned synthetic text document."""
    img = np.full((800, 600, 3), 255, dtype=np.uint8)
    cv2.putText(img, "STANDARD INTAKE SPECIFICATION", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Section 1: General Quality Requirements", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    for y in range(140, 720, 24):
        cv2.line(img, (50, y), (550, y), (0, 0, 0), 2)
    return img


class TestDocumentQualityIntelligenceEngine(unittest.TestCase):

    def setUp(self):
        self.clean_doc = create_synthetic_clean_document()

    # 1. Excellent document
    def test_01_excellent_document(self):
        result = analyze_document_quality(self.clean_doc)
        self.assertGreaterEqual(result["quality_index"], 85.0)
        self.assertEqual(result["decision"], "PASS")
        self.assertTrue(result["intake_accepted"])
        self.assertEqual(result["defects"]["blur"]["status"], "good")
        self.assertEqual(result["defects"]["contrast"]["status"], "good")

    # 2. Blurry document
    def test_02_blurry_document(self):
        path = os.path.join(DATA_DIR, "blur.jpeg")
        if os.path.exists(path):
            img = cv2.imread(path)
            res = analyze_document_quality(img)
            self.assertEqual(res["defects"]["blur"]["status"], "poor")
            self.assertIn(res["decision"], ["REJECT", "NEEDS REVIEW"])
        else:
            blurred = cv2.GaussianBlur(self.clean_doc, (35, 35), 0)
            res = analyze_document_quality(blurred)
            self.assertLess(res["defects"]["blur"]["score"], 40)
            self.assertGreater(res["risk_vector"]["vector"]["blur_risk"], 0.60)

    # 3. Low contrast document
    def test_03_low_contrast_document(self):
        path = os.path.join(DATA_DIR, "contrast.jpg")
        if os.path.exists(path):
            img = cv2.imread(path)
            res = analyze_document_quality(img)
            self.assertLess(res["defects"]["contrast"]["score"], 60)
            self.assertGreater(res["risk_vector"]["vector"]["contrast_risk"], 0.40)
        else:
            gray_flat = np.full((600, 600, 3), 128, dtype=np.uint8)
            cv2.putText(gray_flat, "Faded ink", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (135, 135, 135), 1)
            res = analyze_document_quality(gray_flat)
            self.assertLess(res["defects"]["contrast"]["score"], 40)

    # 4. Noisy document
    def test_04_noisy_document(self):
        path = os.path.join(DATA_DIR, "noise.png")
        if os.path.exists(path):
            img = cv2.imread(path)
            res = analyze_document_quality(img)
            self.assertLess(res["defects"]["noise"]["score"], 70)
            self.assertGreater(res["risk_vector"]["vector"]["noise_risk"], 0.30)
        else:
            noise = np.random.normal(0, 40, self.clean_doc.shape).astype(np.float32)
            noisy = np.clip(self.clean_doc.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            res = analyze_document_quality(noisy)
            self.assertLess(res["defects"]["noise"]["score"], 50)

    # 5. Skewed document
    def test_05_skewed_document(self):
        path = os.path.join(DATA_DIR, "skew.jpeg")
        if os.path.exists(path):
            img = cv2.imread(path)
            res = analyze_document_quality(img)
            self.assertGreater(abs(res["defects"]["skew"]["angle"]), 2.0)
            self.assertGreater(res["risk_vector"]["vector"]["skew_risk"], 0.30)
        else:
            rotated = apply_deskew(self.clean_doc, -8.0)
            res = analyze_document_quality(rotated)
            self.assertGreater(abs(res["defects"]["skew"]["angle"]), 4.0)

    # 6. Handwriting document
    def test_06_handwriting_document(self):
        path = os.path.join(DATA_DIR, "handwrit.png")
        if os.path.exists(path):
            img = cv2.imread(path)
            res = analyze_document_quality(img)
            self.assertTrue(res["defects"]["handwriting"]["detected"])
            self.assertGreaterEqual(res["defects"]["handwriting"]["confidence"], 0.5)
            self.assertEqual(res["decision"], "NEEDS REVIEW")

    # 6b. Non-handwriting documents must not produce false positives
    def test_06b_no_handwriting_in_printed_or_noisy_documents(self):
        # 1. Clean synthetic document
        res_clean = analyze_handwriting(self.clean_doc)
        self.assertFalse(res_clean["detected"])
        self.assertEqual(res_clean["legibility_status"], "not_present")
        self.assertEqual(res_clean["legibility_verdict"], "PASS")

        # 2. Printed documents and other defect types in Datas/
        for fname in ["blur.jpeg", "contrast.jpg", "skew.jpeg", "noise.png", "miss_bottom.png"]:
            path = os.path.join(DATA_DIR, fname)
            if os.path.exists(path):
                img = cv2.imread(path)
                res = analyze_handwriting(img)
                self.assertFalse(res["detected"], f"False positive handwriting in {fname}")

    # 7. Cropped / Missing document
    def test_07_cropped_document(self):
        path = os.path.join(DATA_DIR, "miss_bottom.png")
        if os.path.exists(path):
            img = cv2.imread(path)
            res = analyze_document_quality(img)
            self.assertIn("bottom", res["defects"]["missing_sections"])
            self.assertLess(res["defects"]["missing_section"]["score"], 100)

    # 8. Multiple simultaneous defects
    def test_08_multiple_simultaneous_defects(self):
        # Generate document with optical blur + low contrast + skew simultaneously
        bad_doc = apply_deskew(self.clean_doc, -8.0)
        noise = np.random.normal(0, 20, bad_doc.shape).astype(np.float32)
        bad_doc = np.clip(bad_doc.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        bad_doc = cv2.GaussianBlur(bad_doc, (35, 35), 0)
        bad_doc = np.clip(bad_doc * 0.4 + 90, 0, 255).astype(np.uint8)

        res = analyze_document_quality(bad_doc)
        self.assertEqual(res["decision"], "REJECT")
        self.assertLess(res["quality_index"], 65.0)
        self.assertGreater(len(res["reasons"]), 0)

    # 9. Correction improves quality
    def test_09_correction_improves_quality(self):
        path = os.path.join(DATA_DIR, "contrast.jpg")
        if os.path.exists(path):
            img = cv2.imread(path)
            baseline = analyze_document_quality(img)
            sim_res = simulate_corrections(img, baseline, analyze_document_quality)
            self.assertIsNotNone(sim_res["best_candidate"])
            self.assertGreater(sim_res["best_candidate"]["improvement"], 0.0)
            self.assertTrue(sim_res["best_candidate"]["is_validated"])

    # 10. Correction makes quality worse (Validation loop rejects)
    def test_10_correction_validation_rejects_degradation(self):
        baseline = analyze_document_quality(self.clean_doc)
        sim_res = simulate_corrections(self.clean_doc, baseline, analyze_document_quality)
        for cand in sim_res["simulations"]:
            if cand["improvement"] <= 0.0:
                self.assertFalse(cand["is_validated"])
                self.assertIn("REJECTED", cand["validation_status"])

    # 11. Batch analysis
    def test_11_batch_analysis(self):
        items = [
            ("clean_doc.png", self.clean_doc),
            ("clean_doc_copy.png", self.clean_doc)
        ]
        batch_res = process_document_batch(items, save_to_db=False)
        self.assertEqual(batch_res["total"], 2)
        self.assertEqual(batch_res["pass_count"], 2)
        self.assertGreaterEqual(batch_res["average_quality"], 85.0)

    # 12. Sensitivity analysis
    def test_12_sensitivity_analysis(self):
        path = os.path.join(DATA_DIR, "contrast.jpg")
        if os.path.exists(path):
            img = cv2.imread(path)
            analysis = analyze_document_quality(img)
            sens = analyze_quality_sensitivity(analysis["defects"], analysis["defects"]["structure"])
            self.assertIsNotNone(sens["dominant_driver"])
            self.assertGreaterEqual(len(sens["sensitivities"]), 5)
            self.assertLessEqual(sens["sensitivities"][0]["impact"], 0.0)

    # 13. Decision trace
    def test_13_decision_trace(self):
        res = analyze_document_quality(self.clean_doc)
        self.assertIn("checklist", res)
        self.assertGreaterEqual(len(res["checklist"]), 6)
        self.assertIn("decision_trigger", res)
        self.assertIn("supporting_evidence", res)

    # 14. Fingerprint consistency
    def test_14_fingerprint_consistency(self):
        res1 = analyze_document_quality(self.clean_doc)
        res2 = analyze_document_quality(self.clean_doc)
        fp1 = res1["fingerprint"]["quality_fingerprint"]
        fp2 = res2["fingerprint"]["quality_fingerprint"]
        self.assertEqual(fp1, fp2)
        self.assertEqual(res1["fingerprint"]["fingerprint_hash"], res2["fingerprint"]["fingerprint_hash"])

    # 15. Backward compatibility
    def test_15_legacy_backward_compatibility(self):
        res = analyze_document(self.clean_doc)
        self.assertIn("blur", res)
        self.assertIn("contrast", res)
        self.assertIn("skew", res)
        self.assertIn("noise", res)
        self.assertIn("handwriting", res)
        self.assertIn("missing_section", res)

        score = calculate_quality_score(res)
        self.assertIsInstance(score, (int, float))

        routing = route_document(score, res)
        self.assertIn("decision", routing)
        self.assertIn("reasons", routing)
        self.assertIn("intake_accepted", routing)

        suggestions = generate_suggestions(res)
        self.assertIsInstance(suggestions, list)


from cv.forensics import analyze_document_forensics
from evidence.radar import generate_quality_radar_chart
from intelligence.passport import generate_html_audit_passport

class TestAdvancedFeatures(unittest.TestCase):
    def setUp(self):
        self.clean_doc = create_synthetic_clean_document()

    def test_16_radar_chart_generation(self):
        analysis = analyze_document_quality(self.clean_doc)
        radar = generate_quality_radar_chart(analysis["defects"], analysis["quality_index"], size=480)
        self.assertIsInstance(radar, np.ndarray)
        self.assertEqual(radar.shape, (480, 480, 3))

    def test_17_forensic_ela_analysis(self):
        forensic = analyze_document_forensics(self.clean_doc)
        self.assertIn("tamper_risk", forensic)
        self.assertIn("uniformity_score", forensic)
        self.assertIn("ela_overlay", forensic)
        self.assertIsInstance(forensic["ela_overlay"], np.ndarray)

    def test_18_html_audit_passport(self):
        analysis = analyze_document_quality(self.clean_doc)
        html = generate_html_audit_passport("test_doc.png", analysis)
        self.assertIn("DOCUMENT INTAKE AUDIT PASSPORT", html)
        self.assertIn(analysis["fingerprint"]["quality_fingerprint"], html)

    def test_19_handwriting_legibility_scoring(self):
        path = os.path.join(DATA_DIR, "handwrit.png")
        if os.path.exists(path):
            img = cv2.imread(path)
            res = analyze_handwriting(img)
            self.assertTrue(res["detected"])
            self.assertIn("score", res)
            self.assertIn("legibility_status", res)
            self.assertIn("legibility_verdict", res)
            self.assertIn(res["legibility_verdict"], ["PASS", "NEEDS REVIEW", "REJECT"])

    def test_20_illegible_handwriting_routing(self):
        # Create synthetic illegible scribble: fragmented faint ink with high width irregularity
        scribble_doc = np.full((600, 600, 3), 245, dtype=np.uint8)
        # Draw erratic intersecting scribble lines with faint gray ink
        for _ in range(35):
            pts = np.random.randint(50, 550, (6, 2))
            cv2.polylines(scribble_doc, [pts], False, (190, 190, 190), np.random.randint(1, 8))
        res = analyze_handwriting(scribble_doc)
        if res["detected"]:
            # Faint chaotic scribbles should have low legibility score
            self.assertLess(res["score"], 80)
            self.assertIn(res["legibility_verdict"], ["NEEDS REVIEW", "REJECT"])

    # 21. PDF Loader Verification
    def test_21_pdf_loader_single_and_multipage(self):
        from cv.pdf_loader import is_pdf, load_pdf_pages, load_single_pdf_page, get_pdf_page_count
        pdf_path = os.path.join(DATA_DIR, "sample_document.pdf")
        if os.path.exists(pdf_path):
            self.assertTrue(is_pdf(pdf_path))
            count = get_pdf_page_count(pdf_path)
            self.assertEqual(count, 2)

            pages = load_pdf_pages(pdf_path, scale=1.5)
            self.assertEqual(len(pages), 2)
            p_num, bgr, meta = pages[0]
            self.assertEqual(p_num, 1)
            self.assertIsInstance(bgr, np.ndarray)
            self.assertEqual(bgr.ndim, 3)
            self.assertEqual(bgr.shape[2], 3)

            # Single page loader
            single_img, single_meta = load_single_pdf_page(pdf_path, page_number=2, scale=1.5)
            self.assertIsInstance(single_img, np.ndarray)
            self.assertEqual(single_meta["page_number"], 2)

    # 22. PDF Multi-Page Quality Intelligence Analysis
    def test_22_analyze_pdf_quality(self):
        from intelligence.engine import analyze_pdf_quality
        pdf_path = os.path.join(DATA_DIR, "sample_document.pdf")
        if os.path.exists(pdf_path):
            res = analyze_pdf_quality(pdf_path)
            self.assertEqual(res["document_type"], "PDF")
            self.assertEqual(res["total_pages"], 2)
            self.assertIn("quality_index", res)
            self.assertIn("mean_quality_index", res)
            self.assertIn("bottleneck_quality_index", res)
            self.assertIn("bottleneck_page", res)
            self.assertIn("decision", res)
            self.assertIn(res["decision"], ["PASS", "NEEDS REVIEW", "REJECT"])
            self.assertEqual(len(res["summary_table"]), 2)
            self.assertEqual(len(res["page_results"]), 2)

    # 23. Corrupted PDF Error Handling
    def test_23_corrupted_pdf_handling(self):
        from cv.pdf_loader import is_pdf, load_pdf_pages
        corrupted_bytes = b"CORRUPTED_NON_PDF_DATA_STREAM"
        self.assertFalse(is_pdf(corrupted_bytes))
        with self.assertRaises(ValueError):
            load_pdf_pages(corrupted_bytes)


if __name__ == "__main__":
    unittest.main()
