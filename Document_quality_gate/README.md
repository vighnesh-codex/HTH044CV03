# 🛡️ Document Quality Intelligence Engine
### High-Performance Computer Vision Intake Gate, Forensic Diagnostics & Interactive Studio

A modular, explainable, multi-scale Computer Vision engine designed to inspect, evaluate, and verify technical document intake quality.

---

## 🌟 Advanced CV Capabilities
1. **Multi-Scale Spatial Quality Decomposition**: Evaluates documents globally and across a configurable 3x3 / 4x4 spatial grid. Detects localized regional defects (e.g. *"Overall quality is 82, but lower-right region has severe quality degradation"*).
2. **Authentic Computer Vision Defect Heatmaps**: Generates dense spatial heatmaps for optical blur (local Laplacian energy), noise grain (high-frequency Gaussian residuals), and contrast (local standard deviation).
3. **Polar Quality Radar / Spider Profile**: High-resolution multi-axial polar diagram showing the 6 deficiency axes (Sharpness, Contrast, Alignment, Cleanliness, Clean Ink, Completeness) against the optimal boundary.
4. **Forensic Document Tamper & Error Level Analysis (ELA)**: Recompresses in-memory at 90% JPEG quality to expose digital cut-and-paste tampering, spliced signatures, and compression rate inconsistencies.
5. **Interactive Restoration Studio**: Fine-tune CLAHE contrast clip limits, bilateral denoising strengths, and manual deskew angles with live re-scoring and instant PNG download.
6. **Printable Document Quality Passport (HTML/PDF)**: Official cryptographic-styled intake audit certificate with holographic security layout, fingerprint verification stamp, and checklist trail.
7. **Geometric Defect Overlays**: Renders probabilistic Hough line skew baselines, handwriting contour bounds, missing section bands, and document boundary frames.
8. **Ensemble 5-Dimension Quality Model**:
   - Visual Quality Score (Optics, Dynamic Range, Sensor Noise)
   - Structural Quality Score (Skew, Boundary, Margin Symmetry)
   - Readability Risk Score (Defocus, Low Contrast, Noise Risk)
   - Completeness Score (Functional Header/Body/Footer Occupancy)
   - Interference Risk Score (Handwriting Strokes, Stamps, Noise)
9. **Quality Risk Vector & Dynamic Decision Trace**: Ranks primary and secondary risks and generates explainable step-by-step checklist traces with trigger provenance.
10. **Counterfactual Quality Sensitivity Analysis**: Determines *"What Matters Most?"* by measuring the exact marginal point drag of each defect on the final Quality Index.
11. **Deterministic Document Quality Fingerprints**: Quantized canonical profile hash (`DQF-<HASH>-<DECISION>-<SCORE>`) for document comparison and audit trails.
12. **Automatic Quality Correction Simulation & Validation Loop**: Simulates CLAHE contrast enhancement, bilateral denoising, affine deskewing, and illumination normalization, strictly validating post-repair quality gains before recommending.
13. **Multi-Page PDF Intake Support**: Direct, high-speed rasterization of single-page and multi-page PDFs using `pypdfium2` (PDFium engine) with multi-page scorecard, bottleneck page routing, and batch cohort processing without OCR dependencies.
14. **Document-to-Document Comparison**: Metric-by-metric delta comparison between two versions of a document.
15. **Batch Intake & Epidemiology**: Processes cohorts of documents at scale with defect frequency distribution and CSV export.
16. **Persistent SQLite History & Analytics**: Audited quality records and historical statistics.
17. **Pure Computer Vision**: Free of OCR dependencies or artificial AI claims.

---

## 📂 Project Structure

```
Document_Quality_Intelligence_Upgrade/
├── .venv/                            # Self-contained local virtual environment
├── cv/                               # Core Computer Vision defect analyzers
│   ├── preprocessing.py              # Multi-channel normalization & resizing
│   ├── pdf_loader.py                 # High-performance PDF rasterizer (pypdfium2 / pdf2image)
│   ├── blur.py                       # Laplacian variance & Tenengrad energy
│   ├── contrast.py                   # RMS contrast & dynamic percentile spread
│   ├── skew.py                       # HoughLinesP & angle median dispersion
│   ├── noise.py                      # Gaussian residual & flat-patch SNR
│   ├── handwriting.py                # Morphological line suppression & tortuosity
│   ├── completeness.py               # Functional zone density analysis
│   ├── structure.py                  # Quadrant illumination & border truncation
│   └── forensics.py                  # Error Level Analysis (ELA) & Tamper detection
├── intelligence/                     # Intelligence & decision modeling
│   ├── engine.py                     # Master orchestrator
│   ├── multiscale.py                 # 3x3 spatial grid quality decomposition
│   ├── quality_model.py              # Ensemble 5-dimension scoring & legacy fallback
│   ├── risk_engine.py                # Risk vector & explainable checklist trace
│   ├── sensitivity.py                # Counterfactual 'What Matters Most?' analysis
│   ├── fingerprint.py                # Deterministic canonical fingerprinting
│   ├── anomaly.py                    # IsolationForest cohort outlier detection
│   ├── comparison.py                 # Document-to-document version comparison
│   ├── passport.py                   # Official HTML Quality Audit Passport generator
│   └── batch.py                      # Cohort batch processing & epidemiology
├── evidence/                         # Authentic CV visualizations
│   ├── heatmaps.py                   # Blur, noise, contrast, and composite heatmaps
│   ├── overlays.py                   # Skew, handwriting, and missing section overlays
│   └── radar.py                      # High-resolution Polar Quality Radar chart
├── correction/                       # Repair simulations & validation
│   ├── enhancement.py                # CLAHE adaptive contrast optimization
│   ├── denoise.py                    # Bilateral edge-preserving filter
│   ├── deskew.py                     # Affine warp rotation
│   ├── illumination.py               # Morphological shading normalization
│   └── simulator.py                  # Closed-loop validation engine
├── storage/                          # Local audit database
│   └── database.py                   # SQLite history & KPI aggregation
├── backend_modules/                  # 100% backward-compatible wrappers
│   ├── scoring.py
│   ├── routing.py
│   └── suggestion.py
├── modules/                          # 100% backward-compatible wrappers
│   ├── analyzer.py
│   ├── blur.py
│   ├── contrast.py
│   ├── skew.py
│   ├── noise.py
│   ├── handwriting.py
│   ├── missing_section.py
│   └── preprocessing.py
├── tests/                            # Automated test suite
│   └── test_engine.py                # 18 comprehensive unit & integration tests
├── data/                             # SQLite database storage directory
├── Datas/                            # Sample hackathon test dataset
├── Main.py                           # CLI test runner
├── test_analyzer.py                  # Legacy analyzer test runner
├── app.py                            # Streamlit Dashboard (5 modes, 8 tabs)
└── requirements.txt                  # Python dependencies
```

---

## 🚀 Quickstart Commands

All commands are run directly inside `~/Documents/Document_Quality_Intelligence_Upgrade`:

```bash
cd ~/Documents/Document_Quality_Intelligence_Upgrade
source .venv/bin/activate
```

### 1. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

### 2. Run the Full Test Suite (18 Tests)
```bash
python3 -m unittest tests/test_engine.py
```

### 3. Run the CLI Intake Engine
```bash
python3 Main.py
```

### 4. Run Legacy Verification
```bash
python3 test_analyzer.py
```
