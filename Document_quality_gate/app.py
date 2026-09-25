"""
Document Quality Intelligence Platform - Streamlit Interface
Enterprise Computer Vision Intake Quality Gate, Multi-Scale Diagnostics & Restoration Studio.
"""

import os
import io
import json
import numpy as np
import cv2
import pandas as pd
import streamlit as st

from cv.preprocessing import validate_and_preprocess
from cv.forensics import analyze_document_forensics
from cv.pdf_loader import is_pdf, load_pdf_pages, load_single_pdf_page
from intelligence.engine import analyze_document_quality, analyze_pdf_quality
from intelligence.sensitivity import analyze_quality_sensitivity
from intelligence.comparison import compare_document_analyses
from intelligence.batch import process_document_batch
from intelligence.passport import generate_html_audit_passport
from evidence.heatmaps import (
    generate_blur_heatmap,
    generate_noise_heatmap,
    generate_contrast_heatmap,
    generate_quality_heatmap,
    render_regional_tile_overlay
)
from evidence.overlays import generate_defect_overlay
from evidence.radar import generate_quality_radar_chart
from correction.simulator import simulate_corrections
from correction.enhancement import apply_contrast_enhancement
from correction.denoise import apply_denoising
from correction.deskew import apply_deskew
from correction.illumination import apply_illumination_normalization
from storage.database import (
    init_db,
    save_analysis_record,
    get_history,
    get_analytics_summary
)
from cv.ocr import (
    is_tesseract_available,
    extract_sample_ocr,
    generate_ocr_word_overlay,
    compare_ocr_impact
)


# Initialize Database
init_db()

# Page Setup
st.set_page_config(
    page_title="Document Quality Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism Dark Theme Styling
st.markdown("""
<style>
    /* Dark Theme Obsidian Canvas */
    .stApp {
        background-color: #0B0F17;
        color: #E2E8F0;
    }
    
    /* Modern Glassmorphic Cards */
    .glass-card {
        background: rgba(22, 27, 34, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.35);
    }
    
    /* Verdict Glow Badges */
    .verdict-pass {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.3) 100%);
        border: 1px solid #10B981;
        color: #34D399;
        font-weight: 800;
        font-size: 1.5rem;
        padding: 10px 24px;
        border-radius: 10px;
        display: inline-block;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.25);
    }
    
    .verdict-review {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(217, 119, 6, 0.3) 100%);
        border: 1px solid #F59E0B;
        color: #FBBF24;
        font-weight: 800;
        font-size: 1.5rem;
        padding: 10px 24px;
        border-radius: 10px;
        display: inline-block;
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.25);
    }
    
    .verdict-reject {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.3) 100%);
        border: 1px solid #EF4444;
        color: #F87171;
        font-weight: 800;
        font-size: 1.5rem;
        padding: 10px 24px;
        border-radius: 10px;
        display: inline-block;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.25);
    }

    /* Tab header polish */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 600;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
    }
</style>
""", unsafe_allow_html=True)


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datas")


def load_sample_image(filename: str):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        if filename.lower().endswith(".pdf") or is_pdf(path):
            img, _ = load_single_pdf_page(path, page_number=1)
            return img
        return cv2.imread(path)
    return None


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Sidebar Architecture
st.sidebar.markdown("## 🛡️ Document Intelligence")
st.sidebar.caption("Advanced Computer Vision Intake Gate")

nav_mode = st.sidebar.radio(
    "Intake Gate Workspaces",
    [
        "Single Document Analysis",
        "Document Comparison (V1 vs V2)",
        "Batch Intake & Epidemiology",
        "Persistent History & Analytics",
        "Judge Demo Mode (Offline Scenarios)"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### 🔬 CV Capabilities")
st.sidebar.markdown("✓ Multi-Scale Spatial Grid (3x3)")
st.sidebar.markdown("✓ Authentic Laplacian & Residual Heatmaps")
st.sidebar.markdown("✓ Polar Quality Radar Profile")
st.sidebar.markdown("✓ Error Level Forensics (ELA)")
st.sidebar.markdown("✓ Counterfactual Sensitivity Analysis")
st.sidebar.markdown("✓ Closed-Loop Validation Simulator")
st.sidebar.markdown("✓ Interactive Restoration Studio")
st.sidebar.markdown("✓ Printable Audit Passports")
st.sidebar.caption("Pure Computer Vision • OCR-Free Platform")


# -------------------------------------------------------------
# MODE 1: SINGLE DOCUMENT ANALYSIS
# -------------------------------------------------------------
if nav_mode == "Single Document Analysis":
    st.title("📄 Document Quality Intelligence Gate")
    st.write("Examine technical quality, localize optical defects, inspect forensic compression, and generate verified intake audit passports.")

    # Document Input (Images or PDFs)
    col_input1, col_input2 = st.columns([2, 1])
    with col_input1:
        uploaded_file = st.file_uploader(
            "Upload Document (Image or PDF) for Intake Inspection",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "pdf"]
        )

    with col_input2:
        sample_choice = st.selectbox(
            "Or select benchmark sample:",
            [
                "(None)",
                "sample_document.pdf",
                "blur.jpeg",
                "contrast.jpg",
                "skew.jpeg",
                "noise.png",
                "handwrit.png",
                "miss_bottom.png",
                "miss_midbot.png",
                "miss_top.png"
            ]
        )

    image_to_process = None
    doc_name = "uploaded_document"
    is_pdf_doc = False
    pdf_source = None
    pdf_pages = []
    total_pdf_pages = 0
    show_multipage_summary = False

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        doc_name = uploaded_file.name
        if uploaded_file.name.lower().endswith(".pdf") or is_pdf(file_bytes):
            is_pdf_doc = True
            pdf_source = file_bytes
            pdf_pages = load_pdf_pages(file_bytes, scale=2.0)
            total_pdf_pages = len(pdf_pages)
        else:
            nparr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
            image_to_process = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif sample_choice != "(None)":
        doc_name = sample_choice
        sample_path = os.path.join(DATA_DIR, sample_choice)
        if sample_choice.lower().endswith(".pdf") or is_pdf(sample_path):
            is_pdf_doc = True
            pdf_source = sample_path
            pdf_pages = load_pdf_pages(sample_path, scale=2.0)
            total_pdf_pages = len(pdf_pages)
        else:
            image_to_process = load_sample_image(sample_choice)

    if is_pdf_doc and total_pdf_pages > 1:
        st.info(f"📑 Multi-page PDF detected: **{doc_name}** ({total_pdf_pages} Total Pages)")
        pdf_mode = st.radio(
            "PDF Intake View Option:",
            ["Full Multi-Page Intake Audit Summary", "Inspect Specific Page Diagnostics"],
            horizontal=True
        )
        if pdf_mode == "Full Multi-Page Intake Audit Summary":
            show_multipage_summary = True
        else:
            page_num = st.slider("Select Page to Inspect:", 1, total_pdf_pages, 1)
            image_to_process = pdf_pages[page_num - 1][1]
            doc_name = f"{doc_name} (Page {page_num})"
    elif is_pdf_doc and total_pdf_pages == 1:
        image_to_process = pdf_pages[0][1]
        doc_name = f"{doc_name} (Page 1)"

    if show_multipage_summary and pdf_source is not None:
        with st.spinner(f"Analyzing all {total_pdf_pages} pages across multi-scale CV layers..."):
            pdf_analysis = analyze_pdf_quality(pdf_source)

        st.divider()
        b_col1, b_col2, b_col3, b_col4 = st.columns([1.3, 1, 1, 1])
        with b_col1:
            dec = pdf_analysis["decision"]
            if dec == "PASS":
                st.markdown(f'<div class="verdict-pass">VERDICT: {dec}</div>', unsafe_allow_html=True)
            elif dec == "NEEDS REVIEW":
                st.markdown(f'<div class="verdict-review">VERDICT: {dec}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-reject">VERDICT: {dec}</div>', unsafe_allow_html=True)
            st.caption(f"Governance Trigger: {pdf_analysis['decision_trigger']}")

        with b_col2:
            st.metric("Mean Quality Index", f"{pdf_analysis['mean_quality_index']:.1f} / 100")
        with b_col3:
            st.metric("Bottleneck Page", f"Page {pdf_analysis['bottleneck_page']}", f"{pdf_analysis['bottleneck_quality_index']:.1f} / 100")
        with b_col4:
            st.metric("Total Pages Analyzed", pdf_analysis["total_pages"])

        st.subheader("📑 Page-by-Page Quality Scorecard")
        summary_df = pd.DataFrame(pdf_analysis["summary_table"])
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("📈 Multi-Page Quality Trajectory")
        chart_data = pd.DataFrame({
            "Page": [f"P{p['page']}" for p in pdf_analysis["summary_table"]],
            "Quality Index": [p["quality_index"] for p in pdf_analysis["summary_table"]]
        }).set_index("Page")
        st.bar_chart(chart_data)

        st.subheader("🖼️ Page Previews & Quality Badges")
        preview_cols = st.columns(min(4, total_pdf_pages))
        for idx, (p_num, bgr_img, _) in enumerate(pdf_pages[:8]):
            with preview_cols[idx % len(preview_cols)]:
                p_qi = pdf_analysis["page_results"][idx]["quality_index"]
                p_dec = pdf_analysis["page_results"][idx]["decision"]
                st.image(bgr_to_rgb(bgr_img), caption=f"Page {p_num}: {p_dec} ({p_qi:.1f})", use_container_width=True)

        st.subheader("📋 Decision Trace & Governance Reasons")
        for reason in pdf_analysis["reasons"]:
            st.markdown(f"- {reason}")

    elif image_to_process is not None:
        with st.spinner("Analyzing document across multi-scale CV layers..."):
            analysis = analyze_document_quality(image_to_process)
            doc_id = save_analysis_record(doc_name, analysis)

        qi = analysis["quality_index"]
        decision = analysis["decision"]
        defects = analysis["defects"]
        risk_info = analysis["risk_vector"]
        multiscale = analysis["multiscale"]
        sensitivity = analysis["sensitivity"]
        fp = analysis["fingerprint"]
        subscores = analysis["subscores"]

        # Main Verdict Banner
        st.divider()
        b_col1, b_col2, b_col3, b_col4 = st.columns([1.3, 1, 1, 1])

        with b_col1:
            if decision == "PASS":
                st.markdown(f'<div class="verdict-pass">VERDICT: {decision}</div>', unsafe_allow_html=True)
            elif decision == "NEEDS REVIEW":
                st.markdown(f'<div class="verdict-review">VERDICT: {decision}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-reject">VERDICT: {decision}</div>', unsafe_allow_html=True)
            st.caption(f"Rule Trigger: {analysis['decision_trigger']}")

        with b_col2:
            st.metric("Quality Index", f"{qi:.1f} / 100", delta=round(qi - 60.0, 1))

        with b_col3:
            st.metric("Legacy Score", f"{analysis['legacy_score']:.1f} / 100")

        with b_col4:
            st.metric("Primary Defect Driver", risk_info["primary_risk"]["label"])

        # Tabbed Intelligence Views
        tab_summary, tab_defects, tab_evidence, tab_forensics, tab_trace, tab_sim, tab_studio, tab_passport, tab_ocr = st.tabs([
            "📊 Executive Summary & Radar",
            "🔍 Defect Intelligence Cards",
            "🗺️ Multi-Scale Evidence & Heatmaps",
            "🕵️ Forensic ELA & Integrity",
            "🧠 Decision Trace & Sensitivity",
            "🧪 What-If Correction Simulator",
            "🎛️ Interactive Restoration Studio",
            "📜 Audit Passport & Certificate",
            "📝 Sample OCR & Downstream Impact"
        ])

        # TAB 1: SUMMARY & RADAR PROFILE
        with tab_summary:
            s_col1, s_col2 = st.columns([1.1, 1])
            with s_col1:
                st.subheader("Document Preview & Metadata")
                st.image(bgr_to_rgb(image_to_process), use_container_width=True)
                st.caption(f"Fingerprint: `{fp['quality_fingerprint']}` | Dimensions: {analysis['image_metadata']['width']}x{analysis['image_metadata']['height']}")

            with s_col2:
                st.subheader("Polar Quality Radar Profile")
                radar_img = generate_quality_radar_chart(defects, qi, size=480)
                st.image(bgr_to_rgb(radar_img), use_container_width=True)
                st.caption("Hexagonal Quality Polygon: Identifies deficiency axes relative to the 100% outer boundary.")

            st.divider()
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.subheader("Ensemble 5-Dimension Subscore Breakdown")
                sub_df = pd.DataFrame([
                    {"Dimension": "Visual Quality (Optics/Noise)", "Score": subscores["visual_quality"], "Target": ">= 75"},
                    {"Dimension": "Structural Quality (Geometry)", "Score": subscores["structural_quality"], "Target": ">= 75"},
                    {"Dimension": "Readability Risk (Defocus/Noise)", "Score": subscores["readability_risk"], "Target": "< 30"},
                    {"Dimension": "Completeness (Zones)", "Score": subscores["completeness"], "Target": "100%"},
                    {"Dimension": "Interference Risk (Ink/Stamps)", "Score": subscores["interference_risk"], "Target": "< 25"}
                ])
                st.dataframe(sub_df, hide_index=True, use_container_width=True)

            with sub_col2:
                st.subheader("Intake Recommendations & Directives")
                for s in analysis["suggestions"]:
                    st.info(f"💡 {s}")

                if multiscale["is_localized_defect"]:
                    st.warning(f"⚠️ Spatial Anomaly: {multiscale['diagnosis']}")

        # TAB 2: DEFECT CARDS
        with tab_defects:
            st.subheader("Multi-Dimensional Defect Measurements")
            d1, d2, d3 = st.columns(3)

            with d1:
                st.markdown("### 🔍 Blur / Sharpness")
                b_ev = defects["blur"]["evidence"]
                st.write(f"**Score:** {defects['blur']['score']} / 100")
                st.write(f"**Status:** `{defects['blur']['status']}`")
                st.write(f"**Severity:** {defects['blur']['severity']} / 100")
                st.write(f"**Confidence:** {defects['blur']['confidence']:.2f}")
                st.caption(f"Laplacian Var: {b_ev['laplacian_variance']} | Tenengrad: {b_ev['tenengrad_energy']}")

                st.divider()
                st.markdown("### 📐 Skew & Geometry")
                sk_ev = defects["skew"]["evidence"]
                st.write(f"**Angle:** {defects['skew']['angle']:+.2f}°")
                st.write(f"**Status:** `{defects['skew']['status']}`")
                st.write(f"**Severity:** {defects['skew']['severity']} / 100")
                st.write(f"**Confidence:** {defects['skew']['confidence']:.2f}")
                st.caption(f"Lines Detected: {sk_ev['lines_detected']} | Dispersion Var: {sk_ev['angle_variance']}")

            with d2:
                st.markdown("### 🌓 Tonal Contrast")
                c_ev = defects["contrast"]["evidence"]
                st.write(f"**Score:** {defects['contrast']['score']} / 100")
                st.write(f"**Status:** `{defects['contrast']['status']}`")
                st.write(f"**Severity:** {defects['contrast']['severity']} / 100")
                st.write(f"**Confidence:** {defects['contrast']['confidence']:.2f}")
                st.caption(f"Std Dev: {c_ev['contrast_std']} | P95-P5 Spread: {c_ev['dynamic_spread']}")

                st.divider()
                st.markdown("### ✍️ Handwriting Legibility & Readability")
                hw = defects["handwriting"]
                hw_ev = hw["evidence"]
                st.write(f"**Detected:** `{'Yes' if hw['detected'] else 'None (Clean Machine Print)'}`")
                st.write(f"**Legibility Score:** {hw.get('score', 100.0):.1f} / 100")
                hw_v = hw.get("legibility_verdict", "PASS")
                if hw_v == "PASS":
                    st.success(f"Verdict: **{hw_v}** ({hw.get('legibility_status', 'clear').title()})")
                elif hw_v == "NEEDS REVIEW":
                    st.warning(f"Verdict: **{hw_v}** ({hw.get('legibility_status', 'marginal').title()} Legibility)")
                else:
                    st.error(f"Verdict: **{hw_v}** ({hw.get('legibility_status', 'illegible').title()} Scribble)")
                st.write(f"**Stroke Contrast:** {hw_ev.get('stroke_contrast', 1.0):.2f} | **Width CV:** {hw_ev.get('stroke_uniformity', 0.5):.2f}")
                st.caption(f"Strokes: {hw_ev.get('candidate_regions', 0)} | Spacing: {hw_ev.get('stroke_spacing', 1.0):.2f} | Pen: {'Color' if hw_ev.get('color_ink_detected') else 'Monochrome'}")

            with d3:
                st.markdown("### 📡 Sensor Noise / Grain")
                n_ev = defects["noise"]["evidence"]
                st.write(f"**Score:** {defects['noise']['score']} / 100")
                st.write(f"**Status:** `{defects['noise']['status']}`")
                st.write(f"**Severity:** {defects['noise']['severity']} / 100")
                st.write(f"**Confidence:** {defects['noise']['confidence']:.2f}")
                st.caption(f"Residual: {n_ev['noise_level']} | Flat Patch SNR: {n_ev['estimated_snr_db']} dB")

                st.divider()
                st.markdown("### 🧩 Completeness & Layout")
                comp_ev = defects["missing_section"]["evidence"]
                st.write(f"**Completeness:** {defects['missing_section']['score']}%")
                st.write(f"**Missing:** `{', '.join(defects.get('missing_sections', [])) or 'None'}`")
                st.write(f"**Structure Score:** {defects['structure']['score']}/100")
                st.caption(f"Orientation: {defects['structure']['evidence']['orientation']} | Aspect: {defects['structure']['evidence']['aspect_ratio']}")

            st.divider()
            st.subheader("Normalized Quality Risk Vector")
            for r_key, r_val in risk_info["ranking"]:
                r_col1, r_col2 = st.columns([1, 3])
                with r_col1:
                    st.write(f"**{r_key.replace('_', ' ').title()}**")
                with r_col2:
                    st.progress(float(r_val), text=f"Risk: {r_val:.2f}")

        # TAB 3: VISUAL EVIDENCE & HEATMAPS
        with tab_evidence:
            st.subheader("Authentic Computer Vision Defect Evidence")
            vis_mode = st.radio(
                "Select Visualization Layer",
                [
                    "Composite Quality Heatmap",
                    "Defect Overlay (Skew/Handwriting/Missing)",
                    "Blur / Sharpness Defect Heatmap",
                    "Noise Grain Heatmap",
                    "Local Contrast Heatmap",
                    "Regional Tile Grid (3x3 Scores)"
                ],
                horizontal=True
            )

            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.markdown("**Original Document**")
                st.image(bgr_to_rgb(image_to_process), use_container_width=True)

            with v_col2:
                st.markdown(f"**{vis_mode}**")
                if vis_mode == "Composite Quality Heatmap":
                    vis_img = generate_quality_heatmap(image_to_process, multiscale)
                elif vis_mode == "Defect Overlay (Skew/Handwriting/Missing)":
                    vis_img = generate_defect_overlay(image_to_process, defects, defects.get("structure"))
                elif vis_mode == "Blur / Sharpness Defect Heatmap":
                    vis_img = generate_blur_heatmap(image_to_process)
                elif vis_mode == "Noise Grain Heatmap":
                    vis_img = generate_noise_heatmap(image_to_process)
                elif vis_mode == "Local Contrast Heatmap":
                    vis_img = generate_contrast_heatmap(image_to_process)
                else:
                    vis_img = render_regional_tile_overlay(image_to_process, multiscale)

                st.image(bgr_to_rgb(vis_img), use_container_width=True)

            st.divider()
            st.subheader("Regional Quality Map Inspection (3x3 Grid)")
            tile_df = pd.DataFrame(multiscale["tile_features"])[["region", "score", "laplacian_var", "contrast", "brightness", "noise", "edge_density"]]
            st.dataframe(tile_df, hide_index=True, use_container_width=True)

        # TAB 4: FORENSIC ELA & INTEGRITY
        with tab_forensics:
            st.subheader("🕵️ Forensic Document Integrity & Error Level Analysis (ELA)")
            st.write("Inspects compression homogeneity across the document canvas to detect digital tampering, spliced signatures, or inserted text.")

            forensic_res = analyze_document_forensics(image_to_process)

            f_c1, f_c2 = st.columns(2)
            with f_c1:
                st.metric("Tamper Risk Index", f"{forensic_res['tamper_risk']:.1f} / 100", f"Status: {forensic_res['status']}")
                st.info(f"**Diagnosis:** {forensic_res['diagnosis']}")
                st.caption(f"Spatial Error Dispersion Std Dev: {forensic_res['evidence'].get('spatial_dispersion_std', 0.0)} | Scale: {forensic_res['evidence'].get('scale_factor_applied', 15.0)}x")
                st.markdown("**Original Document**")
                st.image(bgr_to_rgb(image_to_process), use_container_width=True)

            with f_c2:
                st.markdown("**Error Level Analysis (ELA) Heatmap**")
                st.image(bgr_to_rgb(forensic_res["ela_overlay"]), use_container_width=True)
                st.caption("Bright glowing regions highlight compression anomalies where digital content was inserted or modified at different compression ratios.")

        # TAB 5: DECISION TRACE & SENSITIVITY
        with tab_trace:
            tr_col1, tr_col2 = st.columns(2)

            with tr_col1:
                st.subheader("Explainable Decision Trace")
                st.markdown(f"**Final Verdict:** `{decision}`")
                st.markdown(f"**Primary Reason:** {analysis['reasons'][0]}")
                st.markdown(f"**Trigger:** `{analysis['decision_trigger']}`")

                st.divider()
                st.markdown("**Dynamic Checklist Trace:**")
                for item in analysis["checklist"]:
                    if item["state"] == "pass":
                        st.markdown(f"✅ **{item['msg']}**")
                    elif item["state"] == "warn":
                        st.markdown(f"⚠️ **{item['msg']}**")
                    else:
                        st.markdown(f"❌ **{item['msg']}**")

                st.divider()
                st.markdown("**Supporting Mathematical Evidence:**")
                for ev in analysis["supporting_evidence"]:
                    st.write(f"- {ev}")

            with tr_col2:
                st.subheader("Quality Sensitivity Analysis")
                st.write("Measures counterfactual impact: *What defect is penalizing the score most?*")
                st.info(sensitivity["summary"])

                sens_rows = []
                for s in sensitivity["sensitivities"]:
                    sens_rows.append({
                        "Factor": s["label"],
                        "Current Drag": f"{s['impact']:+.1f} pts",
                        "Potential Score if Repaired": f"{s['potential_score']:.1f}"
                    })
                st.table(pd.DataFrame(sens_rows))

                st.divider()
                st.subheader("Document Quality Fingerprint")
                st.code(fp["quality_fingerprint"], language="text")
                st.caption(f"Deterministic Hash: {fp['fingerprint_hash']}")

                st.divider()
                st.subheader("Cohort Anomaly Detection")
                ano = analysis["anomaly"]
                st.markdown(f"**Mode:** `{ano['mode']}`")
                st.markdown(f"**Anomaly Detected:** `{ano['is_anomaly']}`")
                st.caption(ano["explanation"])

        # TAB 6: WHAT-IF CORRECTION SIMULATION
        with tab_sim:
            st.subheader("Automatic Quality Correction Simulation")
            st.write("Non-destructive simulation running candidate image restorations through the quality validation loop.")

            if st.button("🚀 Run Controlled Correction Simulation"):
                with st.spinner("Simulating and validating repair candidates..."):
                    sim_result = simulate_corrections(image_to_process, analysis, analyze_document_quality)

                st.info(sim_result["recommendation"])

                sim_table = []
                for cand in sim_result["simulations"]:
                    sim_table.append({
                        "Correction Candidate": cand["label"],
                        "Before Quality": f"{sim_result['original_score']:.1f}",
                        "After Quality": f"{cand['new_score']:.1f}",
                        "Net Improvement": f"{cand['improvement']:+.1f} pts",
                        "Validation Status": cand["validation_status"]
                    })
                st.table(pd.DataFrame(sim_table))

                if sim_result["best_candidate"] and sim_result["best_candidate"]["corrected_image"] is not None:
                    st.divider()
                    st.subheader(f"Side-by-Side: Original vs Simulated {sim_result['best_candidate']['label']}")
                    c_b1, c_b2 = st.columns(2)
                    with c_b1:
                        st.markdown(f"**Original Document (Score: {analysis['quality_index']:.1f})**")
                        st.image(bgr_to_rgb(image_to_process), use_container_width=True)
                    with c_b2:
                        st.markdown(f"**Validated Corrected Image (Score: {sim_result['best_candidate']['new_score']:.1f})**")
                        st.image(bgr_to_rgb(sim_result["best_candidate"]["corrected_image"]), use_container_width=True)
                        save_analysis_record(f"corrected_{doc_name}", sim_result["best_candidate"]["reanalysis"], correction_result=sim_result)
                        st.success("Correction validated and logged to database.")

        # TAB 7: INTERACTIVE RESTORATION STUDIO
        with tab_studio:
            st.subheader("🎛️ Interactive Quality Restoration Studio")
            st.write("Fine-tune restoration filters interactively and re-score the enhanced document in real-time.")

            st_c1, st_c2 = st.columns([1, 1.2])
            with st_c1:
                st.markdown("#### Filter Controls")
                enable_clahe = st.checkbox("Enable Adaptive Contrast (CLAHE)", value=(defects["contrast"]["score"] < 65))
                clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 5.0, 2.5, 0.2)

                enable_denoise = st.checkbox("Enable Bilateral Denoising", value=(defects["noise"]["score"] < 65))
                denoise_sigma = st.slider("Denoise Strength (Sigma)", 15.0, 75.0, 45.0, 5.0)

                detected_angle = float(defects["skew"]["angle"])
                enable_deskew = st.checkbox("Enable Geometric Deskewing", value=(abs(detected_angle) > 1.0))
                manual_angle = st.slider("Deskew Angle (°)", -15.0, 15.0, detected_angle, 0.2)

                enable_illum = st.checkbox("Enable Illumination Uniformity Normalization", value=False)

                apply_btn = st.button("⚡ Render Custom Enhanced Document")

            with st_c2:
                # Execute custom pipeline
                restored = image_to_process.copy()
                if enable_deskew and abs(manual_angle) > 0.1:
                    restored = apply_deskew(restored, manual_angle)
                if enable_illum:
                    restored = apply_illumination_normalization(restored)
                if enable_clahe:
                    restored = apply_contrast_enhancement(restored, clip_limit=clahe_clip)
                if enable_denoise:
                    restored = apply_denoising(restored, sigma_color=denoise_sigma, sigma_space=denoise_sigma)

                r_analysis = analyze_document_quality(restored)
                r_qi = r_analysis["quality_index"]
                delta_qi = round(r_qi - qi, 1)

                st.metric(
                    "Restored Quality Index",
                    f"{r_qi:.1f} / 100",
                    delta=f"{delta_qi:+.1f} pts vs Original"
                )

                st.image(bgr_to_rgb(restored), caption="Restored Document Preview", use_container_width=True)

                # Download restored image
                _, encoded_img = cv2.imencode(".png", restored)
                st.download_button(
                    "📥 Download Restored Document (PNG)",
                    data=encoded_img.tobytes(),
                    file_name=f"restored_{doc_name}.png",
                    mime="image/png"
                )

                if enable_clahe or enable_denoise or enable_deskew or enable_illum:
                    with st.expander("📝 Compare Downstream OCR Impact (Raw vs Restored)"):
                        if is_tesseract_available():
                            if st.button("⚡ Run OCR Yield Comparison", key="btn_ocr_cmp"):
                                with st.spinner("Running downstream Tesseract comparison..."):
                                    cmp_ocr = compare_ocr_impact(image_to_process, restored)
                                    oc1, oc2, oc3 = st.columns(3)
                                    oc1.metric("Raw Word Yield", f"{cmp_ocr['raw_word_count']} words")
                                    oc2.metric("Restored Word Yield", f"{cmp_ocr['restored_word_count']} words", delta=f"{cmp_ocr['word_delta']:+d}")
                                    oc3.metric("Restored Confidence", f"{cmp_ocr['restored_confidence']:.1f}%", delta=f"{cmp_ocr['confidence_delta']:+.1f}%")
                                    st.info(f"💡 {cmp_ocr['impact_summary']}")
                        else:
                            st.info("Tesseract OCR is not installed or available.")

        # TAB 8: AUDIT PASSPORT & CERTIFICATE
        with tab_passport:
            st.subheader("📜 Document Intake Quality Audit Passport")
            st.write("Generate and download an official, cryptographic-styled HTML compliance certificate.")

            passport_html = generate_html_audit_passport(doc_name, analysis)

            st.download_button(
                "📥 Download Official Quality Passport (HTML)",
                data=passport_html,
                file_name=f"Intake_Passport_{doc_name}.html",
                mime="text/html"
            )

            st.markdown("#### Live Passport Preview")
            st.components.v1.html(passport_html, height=720, scrolling=True)

        # TAB 9: SAMPLE OCR & DOWNSTREAM IMPACT
        with tab_ocr:
            st.subheader("📝 Downstream Tesseract OCR Sample Output")
            st.write(
                "Demonstrates downstream text extraction yield and word-level recognition confidence. "
                "Intake optical defects (blur, low contrast, severe skew, noise) directly degrade OCR accuracy, "
                "validating why the Document Quality Gate is essential before automated intake."
            )

            if not is_tesseract_available():
                st.warning("⚠️ Tesseract OCR engine is not detected on the system PATH. Install `tesseract-ocr` to enable downstream sample output.")
            else:
                ocr_cache_key = f"ocr_{doc_name}_{analysis.get('fingerprint', {}).get('quality_fingerprint', '')}"

                col_btn, col_info = st.columns([1, 2])
                with col_btn:
                    run_ocr = st.button("🚀 Run Tesseract OCR Sample Extraction", key="btn_ocr_run", use_container_width=True)
                with col_info:
                    st.caption("Executes Tesseract 5 engine with automatic page segmentation (PSM 3) to evaluate document machine-readability.")

                if run_ocr:
                    with st.spinner("Extracting text and calculating word-level recognition confidence..."):
                        st.session_state[ocr_cache_key] = extract_sample_ocr(image_to_process)

                ocr_result = st.session_state.get(ocr_cache_key, None)
                if ocr_result is not None:
                    st.divider()

                    # Metric row
                    m1, m2, m3, m4 = st.columns(4)
                    readiness = ocr_result["ocr_readiness"]
                    if readiness == "OPTIMAL":
                        r_badge = "🟢 OPTIMAL"
                    elif readiness == "ACCEPTABLE":
                        r_badge = "🟡 ACCEPTABLE"
                    elif readiness == "DEGRADED":
                        r_badge = "🟠 DEGRADED"
                    else:
                        r_badge = "🔴 FAILED"

                    m1.metric("Downstream OCR Readiness", r_badge)
                    m2.metric("Average Word Confidence", f"{ocr_result['average_confidence']:.1f}%")
                    m3.metric("Extracted Word Count", f"{ocr_result['word_count']} words")
                    m4.metric("Character Count", f"{ocr_result['character_count']} chars")

                    # Confidence distribution breakdown
                    cdist = ocr_result["confidence_distribution"]
                    st.markdown(
                        f"**Word Confidence Distribution:** "
                        f"🟢 **High (≥ 80%):** `{cdist['high']}` words &nbsp;|&nbsp; "
                        f"🟡 **Medium (50-79%):** `{cdist['medium']}` words &nbsp;|&nbsp; "
                        f"🔴 **Low (< 50%):** `{cdist['low']}` words"
                    )

                    # Quality Gate Correlation Callout
                    q_score = analysis.get("quality_index", analysis.get("overall_score", 0))
                    decision = analysis.get("decision", "PASS")
                    if decision == "REJECT":
                        st.error(
                            f"🛡️ **Quality Gate Protection Active (Decision: {decision}, Score: {q_score:.1f}/100):** "
                            f"Downstream OCR recognition achieved only {ocr_result['word_count']} words with "
                            f"{ocr_result['average_confidence']:.1f}% average confidence. "
                            f"Rejecting this defective document at intake successfully prevented corrupted text and garbage data "
                            f"from entering downstream enterprise indexing or automated parsing!"
                        )
                    elif decision == "NEEDS REVIEW":
                        st.warning(
                            f"⚠️ **Operator Review Justified (Decision: {decision}, Score: {q_score:.1f}/100):** "
                            f"Downstream OCR average word confidence is {ocr_result['average_confidence']:.1f}% with "
                            f"{cdist['low']} low-confidence tokens. Human verification or guided rescan is warranted."
                        )
                    else:
                        st.success(
                            f"✅ **High-Fidelity Machine Intake (Decision: {decision}, Score: {q_score:.1f}/100):** "
                            f"Document passes quality gates cleanly. Downstream OCR achieved {ocr_result['average_confidence']:.1f}% "
                            f"confidence across {ocr_result['word_count']} words with minimal recognition errors."
                        )

                    st.markdown("---")

                    # View switcher: Text snippet vs Spatial Bounding Box overlay
                    ocr_view = st.radio(
                        "Downstream Output Display Mode:",
                        ["🔤 Extracted Text Stream", "🎯 Spatial Word Confidence Bounding Boxes"],
                        horizontal=True
                    )

                    if ocr_view == "🔤 Extracted Text Stream":
                        if ocr_result["extracted_text"]:
                            st.markdown("##### Extracted OCR Text Content:")
                            st.code(ocr_result["extracted_text"], language="text")
                        else:
                            st.info("No readable text was detected by Tesseract OCR due to document degradation.")
                    else:
                        st.markdown("##### Word Confidence Spatial Overlay (Color-Coded):")
                        st.caption("🟢 Green: High Confidence (≥ 80%) | 🟠 Orange: Medium (50-79%) | 🔴 Red: Low (< 50%)")
                        overlay_img = generate_ocr_word_overlay(image_to_process, ocr_result)
                        st.image(bgr_to_rgb(overlay_img), use_container_width=True)
                else:
                    st.info("💡 Click **'Run Tesseract OCR Sample Extraction'** above to test downstream text recognition on this document.")


# -------------------------------------------------------------
# MODE 2: DOCUMENT COMPARISON (V1 vs V2)
# -------------------------------------------------------------
elif nav_mode == "Document Comparison (V1 vs V2)":
    st.title("⚖️ Document-to-Document Quality Comparison")
    st.write("Compare two versions or captures of a document to evaluate resolution improvement or defect regression.")

    cmp_col1, cmp_col2 = st.columns(2)
    with cmp_col1:
        st.subheader("Version 1 (Baseline)")
        v1_choice = st.selectbox("Select Version 1 sample:", ["contrast.jpg", "blur.jpeg", "skew.jpeg", "noise.png", "miss_bottom.png"], index=0)
        img1 = load_sample_image(v1_choice)
        if img1 is not None:
            st.image(bgr_to_rgb(img1), use_container_width=True)

    with cmp_col2:
        st.subheader("Version 2 (Target / Improved)")
        v2_choice = st.selectbox("Select Version 2 sample:", ["handwrit.png", "miss_full.png", "contrast.jpg", "blur.jpeg"], index=0)
        img2 = load_sample_image(v2_choice)
        if img2 is not None:
            st.image(bgr_to_rgb(img2), use_container_width=True)

    if img1 is not None and img2 is not None:
        if st.button("⚡ Compare Document Versions"):
            with st.spinner("Analyzing both versions..."):
                a1 = analyze_document_quality(img1)
                a2 = analyze_document_quality(img2)
                cmp_res = compare_document_analyses(a1, a2, label_1=v1_choice, label_2=v2_choice)

            st.divider()
            c_score1, c_score2, c_delta = st.columns(3)
            c_score1.metric(v1_choice, f"{cmp_res['score_1']:.1f}")
            c_score2.metric(v2_choice, f"{cmp_res['score_2']:.1f}")
            c_delta.metric("Total Quality Improvement", f"{cmp_res['total_delta']:+.1f} pts")

            st.info(cmp_res["verdict"])

            st.subheader("Metric-by-Metric Comparison")
            metrics_table = []
            for m in cmp_res["metrics"]:
                metrics_table.append({
                    "Quality Metric": m["metric"],
                    f"{v1_choice}": f"{m['val_1']:.1f}",
                    f"{v2_choice}": f"{m['val_2']:.1f}",
                    "Delta Change": f"{m['delta']:+.1f} {m['unit']}",
                    "Status": "Improved" if m["improved"] else "Regressed / Same"
                })
            st.table(pd.DataFrame(metrics_table))


# -------------------------------------------------------------
# MODE 3: BATCH INTAKE & EPIDEMIOLOGY
# -------------------------------------------------------------
elif nav_mode == "Batch Intake & Epidemiology":
    st.title("📦 Batch Document Quality Intake")
    st.write("Process document cohorts at scale, analyze defect distributions, and route batches automatically.")

    batch_files = st.file_uploader(
        "Upload Cohort Documents (Images or PDFs)",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "pdf"],
        accept_multiple_files=True
    )

    use_samples = st.checkbox("Or process all built-in benchmark dataset documents in batch", value=False)

    batch_items = []
    if batch_files:
        for f in batch_files:
            file_bytes = f.read()
            if f.name.lower().endswith(".pdf") or is_pdf(file_bytes):
                pages = load_pdf_pages(file_bytes)
                for p_num, p_img, _ in pages:
                    batch_items.append((f"{f.name} [Page {p_num}]", p_img))
            else:
                b_bytes = np.asarray(bytearray(file_bytes), dtype=np.uint8)
                b_img = cv2.imdecode(b_bytes, cv2.IMREAD_COLOR)
                if b_img is not None:
                    batch_items.append((f.name, b_img))
    elif use_samples:
        for fname in sorted(os.listdir(DATA_DIR)):
            p = os.path.join(DATA_DIR, fname)
            if fname.lower().endswith(".pdf") or is_pdf(p):
                pages = load_pdf_pages(p)
                for p_num, p_img, _ in pages:
                    batch_items.append((f"{fname} [Page {p_num}]", p_img))
            elif fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                b_img = cv2.imread(p)
                if b_img is not None:
                    batch_items.append((fname, b_img))

    if batch_items:
        st.write(f"Ready to process **{len(batch_items)} documents**.")
        if st.button("▶️ Execute Batch Analysis"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def p_callback(cur, tot, fn):
                progress_bar.progress(cur / tot)
                status_text.text(f"Processing ({cur}/{tot}): {fn}")

            batch_result = process_document_batch(batch_items, progress_callback=p_callback)
            progress_bar.empty()
            status_text.empty()

            st.divider()
            st.subheader("Batch Intake Statistics")
            b_m1, b_m2, b_m3, b_m4, b_m5 = st.columns(5)
            b_m1.metric("Total Documents", batch_result["total"])
            b_m2.metric("PASS Rate", f"{batch_result['pass_pct']}%", f"{batch_result['pass_count']} docs")
            b_m3.metric("REVIEW Rate", f"{batch_result['review_pct']}%", f"{batch_result['review_count']} docs")
            b_m4.metric("REJECT Rate", f"{batch_result['reject_pct']}%", f"{batch_result['reject_count']} docs")
            b_m5.metric("Cohort Avg Quality", f"{batch_result['average_quality']:.1f}")

            st.divider()
            b_d1, b_d2 = st.columns(2)
            with b_d1:
                st.subheader("Common Quality Failure Modes")
                st.markdown(f"**Primary Defect:** {batch_result['most_common_defect']}")
                st.markdown(f"**Secondary Defect:** {batch_result['second_most_common_defect']}")
                chart_data = pd.DataFrame(list(batch_result["defect_counts"].items()), columns=["Defect", "Incidence Count"])
                st.bar_chart(chart_data.set_index("Defect"))

            with b_d2:
                st.subheader("Itemized Intake Records")
                rec_rows = []
                for r in batch_result["records"]:
                    rec_rows.append({
                        "Filename": r["filename"],
                        "Quality Index": r["quality_index"],
                        "Verdict": r["decision"],
                        "Primary Risk": r["dominant_risk"],
                        "Blur": r["blur_score"],
                        "Contrast": r["contrast_score"],
                        "Skew": f"{r['skew_angle']:+.1f}°"
                    })
                rec_df = pd.DataFrame(rec_rows)
                st.dataframe(rec_df, hide_index=True, use_container_width=True)

                # CSV Export
                csv_buffer = io.StringIO()
                rec_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 Export Batch Report (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="batch_quality_report.csv",
                    mime="text/csv"
                )


# -------------------------------------------------------------
# MODE 4: PERSISTENT HISTORY & ANALYTICS
# -------------------------------------------------------------
elif nav_mode == "Persistent History & Analytics":
    st.title("📊 Document Intake History & Analytics")
    st.write("Audit history and empirical quality statistics stored in local SQLite database.")

    summary = get_analytics_summary()

    h_m1, h_m2, h_m3, h_m4 = st.columns(4)
    h_m1.metric("Historical Documents", summary["total_documents"])
    h_m2.metric("Mean Quality Index", f"{summary['avg_quality']:.1f}")
    h_m3.metric("Pass Rate", f"{summary['pass_pct']}%")
    h_m4.metric("Correction Success Rate", f"{summary['correction_success_rate']}%")

    st.divider()
    st.subheader("Intake Audit Log")
    records = get_history(limit=50)

    if records:
        log_rows = []
        for r in records:
            log_rows.append({
                "Document ID": r["document_id"],
                "Filename": r["filename"],
                "Timestamp": r["timestamp"][:19],
                "Quality Index": r["quality_index"],
                "Verdict": r["decision"],
                "Dominant Risk": r["dominant_risk"],
                "Fingerprint": r["fingerprint"]
            })
        st.dataframe(pd.DataFrame(log_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No records in history yet. Analyze documents to populate history.")


# -------------------------------------------------------------
# MODE 5: JUDGE DEMO MODE (Offline Scenarios)
# -------------------------------------------------------------
elif nav_mode == "Judge Demo Mode (Offline Scenarios)":
    st.title("🎯 Live Hackathon Judge Demo Mode")
    st.write("Demonstrate the complete Document Quality Intelligence capabilities in 6 offline, zero-dependency scenarios.")

    st.markdown("""
    Select a test scenario below to immediately trigger the computer vision engine:
    """)

    d_col1, d_col2, d_col3 = st.columns(3)
    d_col4, d_col5, d_col6 = st.columns(3)

    scenario = None
    with d_col1:
        if st.button("🟢 Scenario A: Clean Document → PASS"):
            scenario = "A"
    with d_col2:
        if st.button("🟡 Scenario B: Handwriting → REVIEW"):
            scenario = "B"
    with d_col3:
        if st.button("🔴 Scenario C: Blurry Document → REJECT"):
            scenario = "C"
    with d_col4:
        if st.button("🗺️ Scenario D: Localized Defect Heatmap"):
            scenario = "D"
    with d_col5:
        if st.button("🧪 Scenario E: Correction Simulation"):
            scenario = "E"
    with d_col6:
        if st.button("⚖️ Scenario F: V1 vs V2 Comparison"):
            scenario = "F"

    if scenario:
        st.divider()
        if scenario == "A":
            st.subheader("Scenario A: Excellent Document → PASS")
            clean_img = np.full((800, 600, 3), 255, dtype=np.uint8)
            cv2.putText(clean_img, "OFFICIAL INTAKE SPECIFICATION", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(clean_img, "Technical Compliance Verified", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            for y in range(140, 720, 24):
                cv2.line(clean_img, (50, y), (550, y), (0, 0, 0), 2)

            res = analyze_document_quality(clean_img)
            st.success(f"Verdict: **{res['decision']}** (Quality Index: {res['quality_index']}/100)")
            st.write(f"Trigger: `{res['decision_trigger']}`")
            st.image(bgr_to_rgb(clean_img), caption="Clean Synthetic Input Document", width=400)

        elif scenario == "B":
            st.subheader("Scenario B: Handwriting Interference → NEEDS REVIEW")
            img = load_sample_image("handwrit.png")
            res = analyze_document_quality(img)
            st.warning(f"Verdict: **{res['decision']}** (Quality Index: {res['quality_index']}/100)")
            st.write(f"Trigger: `{res['decision_trigger']}`")
            st.write(f"Handwriting Confidence: {res['defects']['handwriting']['confidence']}")
            over = generate_defect_overlay(img, res["defects"])
            st.image(bgr_to_rgb(over), caption="Handwriting Detected (Magenta Outlines)", width=450)

        elif scenario == "C":
            st.subheader("Scenario C: Severe Optical Blur → REJECT")
            img = load_sample_image("blur.jpeg")
            res = analyze_document_quality(img)
            st.error(f"Verdict: **{res['decision']}** (Quality Index: {res['quality_index']}/100)")
            st.write(f"Trigger: `{res['decision_trigger']}`")
            heat = generate_blur_heatmap(img)
            st.image(bgr_to_rgb(heat), caption="Blur Defect Heatmap (Red = High Defocus)", width=450)

        elif scenario == "D":
            st.subheader("Scenario D: Localized Quality Defect & Spatial Map")
            img = load_sample_image("miss_midbot.png")
            res = analyze_document_quality(img)
            st.info(f"Spatial Diagnosis: **{res['multiscale']['diagnosis']}**")
            tile_map = render_regional_tile_overlay(img, res["multiscale"])
            st.image(bgr_to_rgb(tile_map), caption="Regional Tile Grid (3x3 Scores & Status)", width=450)

        elif scenario == "E":
            st.subheader("Scenario E: Simulated Correction & Validation Loop")
            img = load_sample_image("contrast.jpg")
            res = analyze_document_quality(img)
            sim = simulate_corrections(img, res, analyze_document_quality)
            st.info(sim["recommendation"])
            if sim["best_candidate"]:
                c1, c2 = st.columns(2)
                c1.image(bgr_to_rgb(img), caption=f"Original (Score: {res['quality_index']:.1f})")
                c2.image(bgr_to_rgb(sim["best_candidate"]["corrected_image"]), caption=f"Corrected: {sim['best_candidate']['label']} (Score: {sim['best_candidate']['new_score']:.1f})")

        elif scenario == "F":
            st.subheader("Scenario F: Document-to-Document Comparison")
            img1 = load_sample_image("contrast.jpg")
            img2 = load_sample_image("handwrit.png")
            a1 = analyze_document_quality(img1)
            a2 = analyze_document_quality(img2)
            cmp = compare_document_analyses(a1, a2, "contrast.jpg", "handwrit.png")
            st.metric("Quality Improvement", f"{cmp['total_delta']:+.1f} pts", f"{cmp['score_1']} -> {cmp['score_2']}")
            st.info(cmp["verdict"])
