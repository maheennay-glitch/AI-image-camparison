"""
AI Image Comparison
--------------------
Compares two images (e.g. an original photo vs. an AI-generated or
edited version) and highlights where they differ.

Core technique: Structural Similarity Index (SSIM), a standard method
used in digital forensics / tamper-detection to measure how similar
two images are, pixel-region by pixel-region.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import io

# ---------- Page setup ----------
st.set_page_config(
    page_title="AI Image Comparison",
    page_icon="🔍",
    layout="wide",
)

# Custom styling: purple + pink + white palette, Poppins/Inter typography
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Inter:wght@400;500&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #6B2FA0;
    }
    .stApp {
        background: linear-gradient(180deg, #FFFFFF 0%, #FBEFFB 100%);
    }
    div.stButton > button {
        background-color: #A63ACB;
        color: white;
        border-radius: 8px;
        border: none;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #E85DA0;
        color: white;
    }
    .metric-box {
        background-color: #F7E6F7;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 AI Image Comparison")
st.write(
    "Upload two images to detect differences, measure similarity, "
    "and visualize where an image has been altered or AI-generated "
    "content has been introduced."
)

# ---------- Helper functions ----------

def load_image(uploaded_file):
    """Read an uploaded file into an OpenCV (BGR) image."""
    image = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def resize_to_match(img1, img2):
    """Resize the second image to match the first, so they can be compared."""
    h, w = img1.shape[:2]
    return cv2.resize(img2, (w, h))


def compare_images(img1, img2):
    """
    Returns:
        score       - float, 0 to 1 (1 = identical)
        diff_image  - grayscale difference map
        boxed_image - copy of img2 with bounding boxes around differing regions
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    # Threshold the diff map to find regions that changed significantly
    thresh = cv2.threshold(
        diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxed_image = img2.copy()
    min_area = 40  # ignore tiny noise-level differences
    box_count = 0
    for c in contours:
        if cv2.contourArea(c) > min_area:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(boxed_image, (x, y), (x + w, y + h), (203, 58, 168), 2)
            box_count += 1

    return score, diff, boxed_image, box_count


def to_display(img_bgr):
    """Convert an OpenCV BGR image to RGB for display in Streamlit."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def to_png_bytes(img_bgr):
    rgb = to_display(img_bgr)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# ---------- Main UI ----------

col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("Original image", type=["png", "jpg", "jpeg"], key="img1")
with col2:
    file2 = st.file_uploader("Comparison image (e.g. AI-generated / edited)", type=["png", "jpg", "jpeg"], key="img2")

if file1 and file2:
    img1 = load_image(file1)
    img2 = load_image(file2)
    img2_resized = resize_to_match(img1, img2)

    if st.button("Compare images"):
        with st.spinner("Analyzing images..."):
            score, diff, boxed, box_count = compare_images(img1, img2_resized)

        st.markdown("### Results")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f"<div class='metric-box'><h3>{score*100:.1f}%</h3>Similarity</div>",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"<div class='metric-box'><h3>{box_count}</h3>Regions changed</div>",
                unsafe_allow_html=True,
            )
        with m3:
            verdict = "Likely identical" if score > 0.97 else (
                "Minor differences" if score > 0.85 else "Significant differences"
            )
            st.markdown(
                f"<div class='metric-box'><h3>{verdict}</h3>Verdict</div>",
                unsafe_allow_html=True,
            )

        st.markdown("### Visual comparison")
        v1, v2, v3 = st.columns(3)
        with v1:
            st.image(to_display(img1), caption="Original", use_container_width=True)
        with v2:
            st.image(to_display(img2_resized), caption="Comparison image", use_container_width=True)
        with v3:
            st.image(diff, caption="Difference map", use_container_width=True)

        st.markdown("### Detected differences")
        st.image(to_display(boxed), caption="Highlighted regions of change", use_container_width=True)

        st.download_button(
            "Download annotated result",
            data=to_png_bytes(boxed),
            file_name="comparison_result.png",
            mime="image/png",
        )
else:
    st.info("Upload two images above to get started.")

st.markdown("---")
st.caption(
    "Built with Streamlit, OpenCV, and scikit-image (SSIM algorithm). "
    "This technique is used in real-world digital forensics to detect "
    "image tampering and manipulation."
)
