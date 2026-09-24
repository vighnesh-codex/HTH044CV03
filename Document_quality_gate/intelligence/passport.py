"""
Document Intake Quality Audit Passport Module
Generates cryptographic-style HTML intake certificates with tamper-resistant
fingerprint verification stamps, metric audits, and printable styling.
"""

from typing import Dict, Any
from datetime import datetime


def generate_html_audit_passport(
    filename: str,
    analysis: Dict[str, Any]
) -> str:
    """
    Renders an official, beautifully styled HTML Document Quality Passport.
    """
    qi = analysis.get("quality_index", 0.0)
    dec = analysis.get("decision", "NEEDS REVIEW")
    fp = analysis.get("fingerprint", {}).get("quality_fingerprint", "N/A")
    fp_hash = analysis.get("fingerprint", {}).get("fingerprint_hash", "N/A")
    sub = analysis.get("subscores", {})
    defects = analysis.get("defects", {})
    checklist = analysis.get("checklist", [])
    suggestions = analysis.get("suggestions", [])
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Status theme colors
    if dec == "PASS":
        badge_bg = "#10B981"
        badge_text = "#FFFFFF"
        badge_border = "#059669"
    elif dec == "NEEDS REVIEW":
        badge_bg = "#F59E0B"
        badge_text = "#FFFFFF"
        badge_border = "#D97706"
    else:
        badge_bg = "#EF4444"
        badge_text = "#FFFFFF"
        badge_border = "#DC2626"

    checklist_html = ""
    for item in checklist:
        icon = "✓" if item["state"] == "pass" else ("⚠" if item["state"] == "warn" else "✗")
        clr = "#10B981" if item["state"] == "pass" else ("#F59E0B" if item["state"] == "warn" else "#EF4444")
        checklist_html += f"""
        <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #E5E7EB;">
            <span style="font-weight:600; color:#374151;">{item['name']}</span>
            <span style="color:{clr}; font-weight:700;">{icon} {item['msg']}</span>
        </div>
        """

    suggestions_html = ""
    for s in suggestions:
        suggestions_html += f"<li style='margin-bottom:6px; color:#4B5563;'>{s}</li>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Intake Quality Passport - {filename}</title>
<style>
    @media print {{
        body {{ margin: 0; padding: 20px; }}
        .no-print {{ display: none; }}
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background: #F3F4F6;
        color: #1F2937;
        margin: 0;
        padding: 40px 20px;
    }}
    .passport-container {{
        max-width: 820px;
        margin: 0 auto;
        background: #FFFFFF;
        border-radius: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        border: 2px solid #E5E7EB;
        overflow: hidden;
    }}
    .passport-header {{
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: #FFFFFF;
        padding: 32px 40px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 4px solid {badge_bg};
    }}
    .passport-body {{
        padding: 40px;
    }}
    .score-banner {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 32px;
    }}
    .badge {{
        display: inline-block;
        padding: 8px 24px;
        border-radius: 9999px;
        font-size: 20px;
        font-weight: 800;
        letter-spacing: 0.05em;
        background: {badge_bg};
        color: {badge_text};
        border: 2px solid {badge_border};
    }}
    .subscores-grid {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 12px;
        margin-bottom: 32px;
    }}
    .subscore-box {{
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }}
    .subscore-val {{
        font-size: 22px;
        font-weight: 700;
        color: #1E293B;
    }}
    .subscore-lbl {{
        font-size: 11px;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 600;
        margin-top: 4px;
    }}
    .section-title {{
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 700;
        margin-bottom: 12px;
        border-bottom: 2px solid #F1F5F9;
        padding-bottom: 6px;
    }}
    .hash-box {{
        background: #1E293B;
        color: #94A3B8;
        font-family: monospace;
        font-size: 12px;
        padding: 12px;
        border-radius: 8px;
        word-break: break-all;
    }}
    .print-btn {{
        background: #2563EB;
        color: #FFFFFF;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
    }}
</style>
</head>
<body>

<div class="passport-container">
    <div class="passport-header">
        <div>
            <h1 style="margin:0; font-size:24px; font-weight:800; letter-spacing:-0.02em;">DOCUMENT INTAKE AUDIT PASSPORT</h1>
            <p style="margin:4px 0 0 0; color:#94A3B8; font-size:13px;">Official Computer Vision Intake Gate Certification</p>
        </div>
        <button class="print-btn no-print" onclick="window.print()">🖨️ Print / Save PDF</button>
    </div>

    <div class="passport-body">
        <div class="score-banner">
            <div>
                <span class="badge">{dec}</span>
                <p style="margin:8px 0 0 0; font-size:13px; color:#64748B;">{analysis.get('decision_trigger', '')}</p>
            </div>
            <div style="text-align:right;">
                <div style="font-size:48px; font-weight:900; color:#1E293B; line-height:1;">{qi:.1f}<span style="font-size:20px; color:#94A3B8;">/100</span></div>
                <div style="font-size:12px; color:#64748B; font-weight:600; text-transform:uppercase; margin-top:4px;">Ensemble Quality Index</div>
            </div>
        </div>

        <div class="section-title">Ensemble 5-Dimension Subscores</div>
        <div class="subscores-grid">
            <div class="subscore-box">
                <div class="subscore-val">{sub.get('visual_quality', 0):.1f}</div>
                <div class="subscore-lbl">Visual Quality</div>
            </div>
            <div class="subscore-box">
                <div class="subscore-val">{sub.get('structural_quality', 0):.1f}</div>
                <div class="subscore-lbl">Structure</div>
            </div>
            <div class="subscore-box">
                <div class="subscore-val">{sub.get('readability_risk', 0):.1f}</div>
                <div class="subscore-lbl">Readability Risk</div>
            </div>
            <div class="subscore-box">
                <div class="subscore-val">{sub.get('completeness', 0):.1f}%</div>
                <div class="subscore-lbl">Completeness</div>
            </div>
            <div class="subscore-box">
                <div class="subscore-val">{sub.get('interference_risk', 0):.1f}</div>
                <div class="subscore-lbl">Interference Risk</div>
            </div>
        </div>

        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:24px; margin-bottom:32px;">
            <div>
                <div class="section-title">Defect Verification Checklist</div>
                {checklist_html}
            </div>
            <div>
                <div class="section-title">Intake Action Directives</div>
                <ul style="padding-left:20px; margin:0;">
                    {suggestions_html}
                </ul>
            </div>
        </div>

        <div class="section-title">Cryptographic Audit Signature & Fingerprint</div>
        <table style="width:100%; font-size:13px; margin-bottom:12px;">
            <tr>
                <td style="color:#64748B; padding:4px 0; width:140px;">Target Document:</td>
                <td style="font-weight:600;">{filename}</td>
            </tr>
            <tr>
                <td style="color:#64748B; padding:4px 0;">Audit Timestamp:</td>
                <td>{now_str}</td>
            </tr>
            <tr>
                <td style="color:#64748B; padding:4px 0;">Quality Fingerprint:</td>
                <td><strong style="color:#2563EB;">{fp}</strong></td>
            </tr>
        </table>
        <div class="hash-box">
            SHA256: {fp_hash}
        </div>
    </div>
</div>

</body>
</html>
"""
    return html
