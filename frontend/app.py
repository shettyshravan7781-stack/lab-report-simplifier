import os
import re
import requests
import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:8000/api/analyze-report"
HEALTH_CHECK_URL = "http://127.0.0.1:8000/"

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & REBRANDING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Lab Report Simplifier • AI Medical Diagnostic Audit",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #1E293B;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0369A1 100%);
        border-radius: 14px;
        padding: 1.75rem 2rem;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.2);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .hero-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 0.35rem;
        font-weight: 400;
    }

    /* Live Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.025em;
    }

    .status-online {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-offline {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .status-pulse-green {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.35);
    }

    .status-pulse-red {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #EF4444;
        box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.35);
    }

    /* Patient Meta Banner */
    .patient-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.25rem;
    }

    .patient-prop-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }

    .patient-prop-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    .badge-panel {
        background-color: #E0F2FE;
        color: #0369A1;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Organ Risk Card Grid */
    .risk-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .risk-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        position: relative;
        overflow: hidden;
    }

    .risk-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.06);
    }

    .risk-card.high {
        border-top: 4px solid #EF4444;
    }

    .risk-card.moderate {
        border-top: 4px solid #F59E0B;
    }

    .risk-card.low {
        border-top: 4px solid #10B981;
    }

    .risk-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .risk-organ-name {
        font-size: 0.85rem;
        font-weight: 700;
        color: #334155;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .risk-pct-value {
        font-size: 1.75rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.4rem;
    }

    .pct-high { color: #DC2626; }
    .pct-mod  { color: #D97706; }
    .pct-low  { color: #059669; }

    /* Custom Meter Bar */
    .meter-track {
        width: 100%;
        height: 6px;
        background-color: #F1F5F9;
        border-radius: 9999px;
        overflow: hidden;
        margin-bottom: 0.6rem;
    }

    .meter-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.4s ease;
    }

    .fill-high { background-color: #EF4444; }
    .fill-mod  { background-color: #F59E0B; }
    .fill-low  { background-color: #10B981; }

    .risk-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .badge-high-risk {
        background-color: #FEE2E2;
        color: #991B1B;
    }

    .badge-low-risk {
        background-color: #DCFCE7;
        color: #166534;
    }

    /* Narrative Cards */
    .narrative-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        margin-bottom: 1.25rem;
    }

    .narrative-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 0.5rem;
    }

    .narrative-body {
        font-size: 0.92rem;
        line-height: 1.65;
        color: #334155;
        white-space: pre-wrap;
    }

    /* Button Primary */
    .stButton>button {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.25rem !important;
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(2, 132, 199, 0.3) !important;
        transition: transform 0.1s ease, box-shadow 0.1s ease !important;
    }

    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 12px -2px rgba(2, 132, 199, 0.4) !important;
    }

    /* Empty State */
    .empty-state-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=5)
def check_backend_health():
    """Checks whether the FastAPI backend is running and reachable."""
    try:
        r = requests.get(HEALTH_CHECK_URL, timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def eval_is_out_of_range(val_str, ref_str):
    """
    Evaluates whether an extracted biomarker value falls outside the reference range.
    Returns: (is_abnormal: bool, deviation_type: 'NORMAL' | 'HIGH' | 'LOW')
    """
    try:
        val_clean = str(val_str).replace(",", "").strip()
        m_val = re.search(r"[-+]?\d*\.?\d+", val_clean)
        if not m_val:
            return False, "NORMAL"
        val = float(m_val.group(0))

        ref = str(ref_str).strip()
        if not ref or ref.upper() in ["N/A", "NONE", "NORMAL", "NEGATIVE", "NIL"]:
            return False, "NORMAL"

        # Pattern: low - high (e.g. 13 - 17, 81.0 - 101.0, 4.0 to 10.0)
        m_range = re.search(r"(\d*\.?\d+)\s*(?:-|–|to)\s*(\d*\.?\d+)", ref, re.IGNORECASE)
        if m_range:
            low = float(m_range.group(1))
            high = float(m_range.group(2))
            if val < low:
                return True, "LOW"
            elif val > high:
                return True, "HIGH"
            return False, "NORMAL"

        # Pattern: < limit
        m_lt = re.search(r"(?:<|<=|less than)\s*(\d*\.?\d+)", ref, re.IGNORECASE)
        if m_lt:
            limit = float(m_lt.group(1))
            if val > limit:
                return True, "HIGH"
            return False, "NORMAL"

        # Pattern: > limit
        m_gt = re.search(r"(?:>|>=|greater than)\s*(\d*\.?\d+)", ref, re.IGNORECASE)
        if m_gt:
            limit = float(m_gt.group(1))
            if val < limit:
                return True, "LOW"
            return False, "NORMAL"

    except Exception:
        pass

    return False, "NORMAL"


def parse_narrative_sections(raw_text: str):
    """
    Splits the 4-section XAI clinical narrative into structured cards.
    """
    sections = {
        "Executive Summary": "",
        "Biomarker Analysis": "",
        "Health Threats & Risks": "",
        "Recommended Action Plan": ""
    }
    if not raw_text or not isinstance(raw_text, str):
        return sections

    patterns = [
        ("Executive Summary", r"EXECUTIVE SUMMARY\s*-*\s*"),
        ("Biomarker Analysis", r"BIOMARKER ANALYSIS\s*-*\s*"),
        ("Health Threats & Risks", r"HEALTH THREATS & POTENTIAL RISKS\s*-*\s*"),
        ("Recommended Action Plan", r"RECOMMENDED ACTION PLAN\s*-*\s*"),
    ]

    positions = []
    for title, pat in patterns:
        m = re.search(pat, raw_text, re.IGNORECASE)
        if m:
            positions.append((m.start(), m.end(), title))

    if not positions:
        sections["Executive Summary"] = raw_text.strip()
        return sections

    positions.sort(key=lambda x: x[0])
    for i, (start, end, title) in enumerate(positions):
        next_start = positions[i + 1][0] if i + 1 < len(positions) else len(raw_text)
        content = raw_text[end:next_start].strip()
        sections[title] = content

    return sections


ORGAN_METADATA = {
    "Target_Anemia_Risk": {"label": "Anemia Risk", "icon": "🩸"},
    "Target_Diabetes_Risk": {"label": "Diabetes Risk", "icon": "🍬"},
    "Target_Kidney_Risk": {"label": "Kidney Function", "icon": "🫘"},
    "Target_Lipid_Risk": {"label": "Lipid Profile", "icon": "🫀"},
    "Target_Liver_Risk": {"label": "Liver Function", "icon": "🧪"},
    "Target_Overall_Health_Risk": {"label": "Overall Health", "icon": "🛡️"}
}


# -----------------------------------------------------------------------------
# HERO HEADER & CONNECTIVITY (CLEAN MINIFIED HTML — NO STRAY TAGS)
# -----------------------------------------------------------------------------
is_backend_up = check_backend_health()
pill_class = "status-online" if is_backend_up else "status-offline"
pulse_class = "status-pulse-green" if is_backend_up else "status-pulse-red"
status_text = "Backend Online (127.0.0.1:8000)" if is_backend_up else "Backend Offline — Start main.py"

st.markdown(f'''<div class="hero-container"><div><h1 class="hero-title">🧪 Lab Report Simplifier</h1><div class="hero-subtitle">Layout-Aware PDF Extraction • XGBoost Organ Risk Scoring • Clear Diagnostic Explanations</div></div><div><div class="status-pill {pill_class}"><div class="{pulse_class}"></div><span>{status_text}</span></div></div></div>''', unsafe_allow_html=True)

if not is_backend_up:
    st.warning("⚠️ **Backend API Unreachable:** Ensure FastAPI is running on `http://127.0.0.1:8000` via `python main.py` or `uvicorn main:app --reload`.")


# -----------------------------------------------------------------------------
# PDF / IMAGE REPORT UPLOAD & PIPELINE EXECUTION
# -----------------------------------------------------------------------------
col_upload, col_btn = st.columns([3, 1], vertical_alignment="bottom")

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Diagnostic Lab Report (PDF / Image)",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Upload a clinical laboratory report (PDF or PNG/JPG image)."
    )

with col_btn:
    analyze_clicked = st.button("🚀 Simplify & Analyze Report", use_container_width=True, disabled=(uploaded_file is None or not is_backend_up))

if analyze_clicked and uploaded_file is not None:
    with st.spinner("Processing report with layout-aware spatial extraction & evaluating predictive models..."):
        try:
            mime_type = "application/pdf" if uploaded_file.name.lower().endswith(".pdf") else f"image/{uploaded_file.name.split('.')[-1].lower().replace('jpg', 'jpeg')}"
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), mime_type)}
            res = requests.post(API_URL, files=files, timeout=45)

            if res.status_code == 200:
                resp_json = res.json()
                st.session_state["last_analysis"] = resp_json
                if resp_json.get("parsed_data", {}).get("status") == "error":
                    st.warning(f"⚠️ Document validation notice for **{uploaded_file.name}**.")
                else:
                    st.success(f"✅ Diagnostic audit completed successfully for **{uploaded_file.name}**.")
            else:
                error_detail = res.json().get("detail", res.text) if res.headers.get("content-type") == "application/json" else res.text
                st.error(f"❌ Diagnostic Pipeline Error ({res.status_code}): {error_detail}")
        except requests.exceptions.Timeout:
            st.error("⏳ Diagnostic processing timed out after 45 seconds.")
        except Exception as e:
            st.error(f"🔌 Connection error contacting diagnostic backend: {str(e)}")


# -----------------------------------------------------------------------------
# CLINICAL DASHBOARD VIEW
# -----------------------------------------------------------------------------
if "last_analysis" in st.session_state:
    payload = st.session_state["last_analysis"]
    parsed_data = payload.get("parsed_data", {})

    if parsed_data.get("status") == "error":
        err_msg = parsed_data.get("error") or parsed_data.get("message") or "Document validation failed."
        st.error(f"❌ **Document Validation Error:** {err_msg}")
        st.info("💡 **Upload Guidance:** Please upload a valid, digital clinical laboratory report (e.g., Complete Blood Count (CBC), Kidney Function (KFT), Liver Function (LFT), Lipid Profile, Glycemic / Diabetes, or Thyroid Panel PDF) containing diagnostic biomarker results.")
    else:

        patient_meta = parsed_data.get("patient", {})
        patient_name = patient_meta.get("name", "N/A")
        patient_age = patient_meta.get("age", "N/A")
        patient_gender = patient_meta.get("gender", "N/A")
        panel_type = parsed_data.get("panel_type", "General Diagnostic Laboratory")
        organ_risks = parsed_data.get("organ_risks", {})
        raw_tests = parsed_data.get("tests", [])
        narrative_str = parsed_data.get("clinical_narrative", "")

        # 1. PATIENT META BANNER
        st.markdown(f'''<div class="patient-card"><div><div class="patient-prop-label">Patient Name</div><div class="patient-prop-val">👤 {patient_name}</div></div><div><div class="patient-prop-label">Age / Gender</div><div class="patient-prop-val">🎂 {patient_age} Y / {patient_gender}</div></div><div><div class="patient-prop-label">Primary Diagnostic Panel</div><div class="patient-prop-val"><span class="badge-panel">🔬 {panel_type}</span></div></div><div><div class="patient-prop-label">Audit Pipeline Status</div><div class="patient-prop-val" style="color: #059669;"> Verified Deterministic</div></div></div>''', unsafe_allow_html=True)

        # 2. TOP PRIORITY VIEW: ORGAN RISK AUDIT
        st.markdown("### 🧬 Machine Learning Multi-Organ Risk Audit")
        st.caption("Calibrated XGBoost predictive models evaluating subclinical organ dysfunction and systemic compromise.")

        if organ_risks:
            cols = st.columns(len(organ_risks))
            for idx, (target_key, risk_info) in enumerate(organ_risks.items()):
                meta = ORGAN_METADATA.get(target_key, {"label": target_key.replace("Target_", "").replace("_Risk", ""), "icon": "🩺"})
                pct = risk_info.get("risk_percentage", 0.0) if isinstance(risk_info, dict) else float(risk_info)
                status_text = risk_info.get("status", "LOW RISK") if isinstance(risk_info, dict) else "LOW RISK"
                is_high = "HIGH" in str(status_text).upper() or pct > 50.0
                is_mod = 10.0 <= pct <= 50.0

                card_class = "high" if is_high else ("moderate" if is_mod else "low")
                pct_class = "pct-high" if is_high else ("pct-mod" if is_mod else "pct-low")
                fill_class = "fill-high" if is_high else ("fill-mod" if is_mod else "fill-low")
                badge_class = "badge-high-risk" if is_high else "badge-low-risk"

                with cols[idx]:
                    st.markdown(f'''<div class="risk-card {card_class}"><div class="risk-header"><div class="risk-organ-name">{meta['icon']} {meta['label']}</div></div><div class="risk-pct-value {pct_class}">{pct:.1f}%</div><div class="meter-track"><div class="meter-fill {fill_class}" style="width: {min(max(pct, 4.0), 100.0)}%;"></div></div><span class="risk-badge {badge_class}">{status_text}</span></div>''', unsafe_allow_html=True)
        else:
            st.info("No organ risk profile returned for this report.")

        st.write("")

        # 3. INTERACTIVE BIOMARKER MATRIX & CLEAR CLINICAL EXPLANATION
        tab_matrix, tab_narrative = st.tabs([
            "📋 Biomarker Analysis Matrix",
            "📄 Clear Clinical Explanation"
        ])

        # TAB 1: BIOMARKERS TABLE
        with tab_matrix:
            if raw_tests:
                # Process & augment tests with range evaluation
                processed_tests = []
                abnormal_count = 0

                for t in raw_tests:
                    t_name = str(t.get("test", "N/A"))
                    t_val = str(t.get("value", "N/A"))
                    t_unit = str(t.get("unit", "N/A"))
                    t_ref = str(t.get("reference", "N/A"))

                    is_abnormal, flag_type = eval_is_out_of_range(t_val, t_ref)
                    if is_abnormal:
                        abnormal_count += 1
                        status_col = f"⚠️ OUT OF RANGE ({flag_type})"
                    else:
                        status_col = " NORMAL"

                    processed_tests.append({
                        "Biomarker Parameter": t_name,
                        "Extracted Value": t_val,
                        "Unit": t_unit,
                        "Reference Interval": t_ref,
                        "Clinical Status": status_col,
                        "is_abnormal": is_abnormal
                    })

                df_all = pd.DataFrame(processed_tests)

                # Filter Controls & Summary Metrics
                c1, c2, c3, c4 = st.columns([1.5, 1.2, 1.2, 2.5])
                with c1:
                    st.metric("Total Parameters", len(df_all))
                with c2:
                    st.metric("Physiological Baseline", len(df_all) - abnormal_count)
                with c3:
                    st.metric("Out-of-Range Flags", abnormal_count, delta=f"{abnormal_count} Critical" if abnormal_count > 0 else "Optimal", delta_color="inverse")
                with c4:
                    filter_choice = st.radio(
                        "Biomarker View Filter:",
                        options=["All Parameters", "Out-of-Range Flags Only"],
                        horizontal=True,
                        label_visibility="visible"
                    )

                # Apply Filter
                if filter_choice == "Out-of-Range Flags Only":
                    df_display = df_all[df_all["is_abnormal"]].copy()
                else:
                    df_display = df_all.copy()

                # Render Table
                display_columns = ["Biomarker Parameter", "Extracted Value", "Unit", "Reference Interval", "Clinical Status"]
            
                if not df_display.empty:
                    st.dataframe(
                        df_display[display_columns],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Biomarker Parameter": st.column_config.TextColumn("Biomarker Parameter", width="medium"),
                            "Extracted Value": st.column_config.TextColumn("Extracted Value", width="small"),
                            "Unit": st.column_config.TextColumn("Unit", width="small"),
                            "Reference Interval": st.column_config.TextColumn("Reference Interval", width="medium"),
                            "Clinical Status": st.column_config.TextColumn("Clinical Status", width="medium"),
                        }
                    )
                else:
                    st.success("🎉 All extracted parameters fall strictly within normal physiological reference ranges.")
            else:
                st.warning("No biomarker parameters extracted from this report.")

        # TAB 2: STRUCTURED NARRATIVE VIEWER
        with tab_narrative:
            st.markdown("### 📑 Clear Clinical Diagnostic Explanation")
            st.caption("Explainable AI (XAI) clinical synthesis generated deterministically with zero remote network calls.")

            parsed_sections = parse_narrative_sections(narrative_str)

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown(f'''<div class="narrative-card"><div class="narrative-title">📋 Executive Summary</div><div class="narrative-body">{parsed_sections.get("Executive Summary", "No summary synthesized.")}</div></div>''', unsafe_allow_html=True)
                st.markdown(f'''<div class="narrative-card"><div class="narrative-title">🔬 Biomarker Analysis & Pathophysiology</div><div class="narrative-body">{parsed_sections.get("Biomarker Analysis", "No biomarker analysis synthesized.")}</div></div>''', unsafe_allow_html=True)

            with col_right:
                st.markdown(f'''<div class="narrative-card"><div class="narrative-title">⚠️ Health Threats & Potential Risks</div><div class="narrative-body">{parsed_sections.get("Health Threats & Risks", "No clinical threats identified.")}</div></div>''', unsafe_allow_html=True)
                st.markdown(f'''<div class="narrative-card"><div class="narrative-title">🎯 Recommended Action Plan</div><div class="narrative-body">{parsed_sections.get("Recommended Action Plan", "Standard routine screening recommended.")}</div></div>''', unsafe_allow_html=True)

else:
    # -------------------------------------------------------------------------
    # EMPTY STATE PLACEHOLDER
    # -------------------------------------------------------------------------
    st.markdown('''<div class="empty-state-box"><div style="font-size: 3.5rem; margin-bottom: 1rem;">🧪</div><h2 style="font-size: 1.5rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem;">No Diagnostic Report Loaded</h2><p style="color: #64748B; max-width: 580px; margin: 0 auto 1.5rem auto; font-size: 0.95rem; line-height: 1.6;">Upload a patient laboratory report in PDF or image format (.pdf, .png, .jpg, .jpeg) above and click <strong>🚀 Simplify & Analyze Report</strong>. The pipeline will perform layout-aware spatial extraction, evaluate multi-organ pathology risk, and provide clear clinical explanations.</p><div style="display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap; color: #475569; font-size: 0.85rem; font-weight: 600;"><span>🩸 Complete Blood Count (CBC)</span><span>🫘 Kidney Function (KFT)</span><span>🧪 Liver Function (LFT)</span><span>🫀 Lipid Profile</span><span>🍬 Glycemic & Diabetes</span><span>🦋 Thyroid Profile</span></div></div>''', unsafe_allow_html=True)