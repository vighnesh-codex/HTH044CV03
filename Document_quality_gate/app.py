import streamlit as st
import cv2
import numpy as np

from backend_modules.scoring import calculate_quality_score
from backend_modules.routing import route_document
from backend_modules.suggestion import generate_suggestions

from modules.analyzer import analyze_document



# Page Setup
st.set_page_config(
    page_title="Document Quality Gate",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Quality Gate")
st.write(
    "Upload a document to check its quality before OCR processing."
)



# Upload Document
uploaded_file = st.file_uploader(
    "Upload Document",
    type=["png", "jpg", "jpeg"]
)

# Process Document
if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error("Could not read the uploaded image.")

    else:

        # CV Analysis
        analysis = analyze_document(image)

        # Prepare Defects
        defects = {

            "blur": analysis["blur"],

            "contrast": analysis["contrast"],

            "skew": analysis["skew"],

            "noise": analysis["noise"],

            "handwriting": {
                "detected": analysis["handwriting"]["detected"],
                "confidence": analysis["handwriting"]["confidence"]
            },

            "missing_section": {
                "score": analysis["missing_section"]["score"],
                "status": (
                    "good"
                    if analysis["missing_section"]["score"] >= 80
                    else "poor"
                )
            },

            "missing_sections": analysis["missing_section"]["missing"]
        }


        # Quality Score
        score = calculate_quality_score(defects)


       
        # Routing Decision
        decision = route_document(
            score,
            defects
        )

        # Suggestions
        suggestions = generate_suggestions(
            defects
        )


        # Display Document
        st.subheader("Uploaded Document")

        st.image(
            image,
            channels="BGR",
            use_container_width=True
        )


        st.divider()


        # Quality Result
        st.subheader("Quality Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Quality Score",
                f"{score}/100"
            )

        with col2:
            st.metric(
                "Decision",
                decision["decision"]
            )


        st.divider()

        # Quality Checks
        st.subheader("Quality Checks")

        col1, col2, col3 = st.columns(3)


        with col1:

            st.write("### Blur")

            st.write(
                f"Score: {defects['blur']['score']}"
            )

            st.write(
                f"Status: {defects['blur']['status']}"
            )


            st.write("### Contrast")

            st.write(
                f"Score: {defects['contrast']['score']}"
            )

            st.write(
                f"Status: {defects['contrast']['status']}"
            )


        with col2:

            st.write("### Skew")

            st.write(
                f"Angle: {defects['skew']['angle']}°"
            )

            st.write(
                f"Status: {defects['skew']['status']}"
            )


            st.write("### Noise")

            st.write(
                f"Score: {defects['noise']['score']}"
            )

            st.write(
                f"Status: {defects['noise']['status']}"
            )


        with col3:

            st.write("### Handwriting")

            st.write(
                f"Detected: "
                f"{defects['handwriting']['detected']}"
            )

            st.write(
                f"Confidence: "
                f"{defects['handwriting']['confidence']}"
            )


            st.write("### missing_section")

            st.write(
                f"Score: "
                f"{defects['missing_section']['score']}"
            )


        st.divider()

        # Reasons
        st.subheader("Detected Issues")

        if decision["reasons"]:

            for reason in decision["reasons"]:
                st.warning(reason)

        else:

            st.success(
                "No major issues detected."
            )


        # Missing Sections
        if defects["missing_sections"]:

            st.subheader("Missing Sections")

            for section in defects["missing_sections"]:
                st.warning(section)


        # Suggestions
        st.subheader("Corrective Suggestions")

        for suggestion in suggestions:

            st.info(suggestion)