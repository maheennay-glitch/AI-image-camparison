"""
AI Image Comparison
--------------------
Compares two images (e.g. an original photo vs. an AI-generated or
edited version) and highlights where they differ.

Core technique: Structural Similarity Index (SSIM), a standard method
used in digital forensics / tamper-detection to measure how similar
two images are, pixel-region by pixel-region.

Also includes a heuristic "AI-generation likelihood" analysis for a
single image, based on classical forensic signal-processing cues
(noise uniformity, frequency-domain anomalies, compression-error
consistency) rather than a trained deep-learning classifier.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
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

    html, body, [class*="css"] {
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
    .feature-card {
        background: white;
        border: 1px solid #EED6F0;
        border-radius: 14px;
        padding: 22px 20px;
        height: 100%;
        box-shadow: 0 2px 10px rgba(166, 58, 203, 0.06);
    }
    .feature-card h4 {
        font-family: 'Poppins', sans-serif;
        color: #6B2FA0;
        margin-bottom: 6px;
    }
    .feature-card p {
        color: #555;
        font-size: 0.92rem;
        margin: 0;
    }
    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, #A63ACB, #E85DA0);
        color: white;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .usecase-pill {
        display: inline-block;
        background: #F7E6F7;
        color: #6B2FA0;
        border-radius: 999px;
        padding: 6px 14px;
        margin: 4px 6px 4px 0;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .analysis-box {
        background: #1E1B2E;
        color: #F1E9FB;
        border-radius: 14px;
        padding: 22px 26px;
        font-family: 'Courier New', monospace;
        line-height: 1.7;
        box-shadow: 0 4px 18px rgba(107, 47, 160, 0.25);
    }
    .analysis-box .label-line {
        color: #E85DA0;
        font-weight: bold;
    }
    .analysis-box .value-line {
        color: #FFFFFF;
        font-size: 1.1rem;
    }
    .analysis-box .important {
        color: #CBB6E0;
        font-size: 0.85rem;
        margin-top: 10px;
        display: block;
    }
    .signal-bar-track {
        background: #F0E0F0;
        border-radius: 6px;
        height: 10px;
        width: 100%;
        overflow: hidden;
    }
    .signal-bar-fill {
        background: linear-gradient(90deg, #A63ACB, #E85DA0);
        height: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 AI Image Comparison")
st.write(
    "Upload two images to detect differences, measure similarity, "
    "and visualize where an image has been altered or AI-generated "
    "content has been introduced. Or run a single image through the "
    "AI-generation likelihood check."
)

# ---------- Helper functions: comparison ----------

def load_image(uploaded_file):
    """Read an uploaded file into an OpenCV (BGR) image."""
    image = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def resize_to_match(img1, img2):
    """Resize the second image to match the first, so they can be compared."""
    h, w = img1.shape[:2]
    return cv2.resize(img2, (w, h))


def compute_ssim(gray1, gray2):
    """
    A self-contained implementation of the Structural Similarity Index (SSIM),
    using only OpenCV/numpy (no scikit-image dependency needed).

    Returns: (mean_score, full_ssim_map) where full_ssim_map is float32 in [-1, 1].
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = gray1.astype(np.float64)
    img2 = gray2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = (
        ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
        / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    )

    return float(ssim_map.mean()), ssim_map


def compare_images(img1, img2):
    """
    Returns:
        score       - float, 0 to 1 (1 = identical)
        diff_image  - grayscale difference map
        boxed_image - copy of img2 with bounding boxes around differing regions
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    score, ssim_map = compute_ssim(gray1, gray2)

    # Normalize the SSIM map to a 0-255 grayscale difference image
    diff = ((1 - ssim_map) * 255).clip(0, 255).astype("uint8")
    diff = cv2.resize(diff, (gray1.shape[1], gray1.shape[0]))

    # Threshold the diff map to find regions that changed significantly
    # (diff is high where images differ, so plain BINARY + OTSU picks those out)
    thresh = cv2.threshold(
        diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
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


# ---------- Helper functions: AI-generation likelihood ----------

def _patch_stat(gray, patch_size, fn):
    """Run fn over non-overlapping patches of gray and return the list of results."""
    h, w = gray.shape
    values = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            block = gray[y:y + patch_size, x:x + patch_size]
            values.append(fn(block))
    return np.array(values) if values else np.array([0.0])


def analyze_ai_likelihood(img_bgr):
    """
    Heuristic forensic analysis estimating whether a single image is
    likely AI-generated / synthetic. This is NOT a trained deep-learning
    classifier -- it combines three classical signal-processing cues:

      1. Noise uniformity: real camera sensor noise varies unevenly
         across an image; many generative models produce noise that is
         unnaturally smooth or uniform.
      2. Frequency-domain anomalies: some generative pipelines leave
         faint periodic/grid artifacts visible in the FFT magnitude
         spectrum that natural photos don't show.
      3. Compression-error consistency (a lightweight Error Level
         Analysis): real camera JPEGs re-compress consistently across
         the frame; spliced or re-generated regions often don't.

    Returns: (label, confidence_pct, detail_scores) where detail_scores
    is a dict of sub-signal name -> score in [0, 1] (higher = more
    "suspicious" / more consistent with AI-generation on that signal).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape
    patch = max(16, min(h, w) // 16)

    # --- Signal 1: noise uniformity ---
    def lap_var(block):
        return cv2.Laplacian(block, cv2.CV_64F).var()

    variances = _patch_stat(gray, patch, lap_var)
    noise_mean = float(np.mean(variances)) + 1e-6
    noise_uniformity = float(np.std(variances)) / noise_mean
    # Lower uniformity spread => suspiciously smooth/consistent noise
    noise_score = float(np.clip(1 - noise_uniformity, 0, 1))

    # --- Signal 2: frequency-domain anomalies ---
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    cy, cx = h // 2, w // 2
    r = max(10, min(h, w) // 10)
    center_region = magnitude[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r]
    outer_region = magnitude.copy()
    outer_region[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r] = 0
    outer_energy = float(outer_region.mean())
    center_energy = float(center_region.mean()) + 1e-6
    freq_ratio = outer_energy / center_energy
    freq_score = float(np.clip((freq_ratio - 0.3) / 0.4, 0, 1))

    # --- Signal 3: compression-error consistency (lightweight ELA) ---
    _, enc = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    ela = cv2.absdiff(img_bgr, recompressed)
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY).astype(np.float64)

    def mean_val(block):
        return block.mean()

    ela_patches = _patch_stat(ela_gray, patch, mean_val)
    ela_mean = float(np.mean(ela_patches)) + 1e-6
    ela_uniformity = float(np.std(ela_patches)) / ela_mean
    ela_score = float(np.clip(1 - ela_uniformity / 2, 0, 1))

    # --- Combine into a single confidence score ---
    combined = 0.40 * noise_score + 0.35 * freq_score + 0.25 * ela_score
    confidence = float(np.clip(combined * 100, 1, 99))

    if confidence >= 70:
        label = "Potentially AI-generated"
    elif confidence >= 45:
        label = "Uncertain / mixed signals"
    else:
        label = "Likely authentic / camera-captured"

    details = {
        "Noise uniformity": noise_score,
        "Frequency-domain anomaly": freq_score,
        "Compression-error consistency": ela_score,
    }
    return label, confidence, details


def render_signal_bar(name, value):
    pct = int(round(value * 100))
    st.markdown(
        f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#6B2FA0;">
                <span>{name}</span><span>{pct}%</span>
            </div>
            <div class="signal-bar-track">
                <div class="signal-bar-fill" style="width:{pct}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- Main UI ----------

tab_compare, tab_detect = st.tabs(["🆚 Compare two images", "🕵️ AI-generation likelihood"])

with tab_compare:
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
        st.markdown("")
        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown(
                """<div class='feature-card'>
                <h4>🎯 Similarity score</h4>
                <p>Get a precise percentage measuring how structurally similar
                your two images are, powered by the SSIM algorithm.</p>
                </div>""",
                unsafe_allow_html=True,
            )
        with f2:
            st.markdown(
                """<div class='feature-card'>
                <h4>📦 Region detection</h4>
                <p>Automatically finds and draws bounding boxes around every
                area that changed between your two images.</p>
                </div>""",
                unsafe_allow_html=True,
            )
        with f3:
            st.markdown(
                """<div class='feature-card'>
                <h4>⬇️ Downloadable results</h4>
                <p>Export the annotated comparison as a PNG to include in a
                report, case file, or presentation.</p>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### How it works")
        h1, h2, h3 = st.columns(3)
        with h1:
            st.markdown(
                """<span class='step-badge'>1</span>

**Upload two images**

An original photo and a comparison version (edited, AI-generated, or a later copy).""",
            )
        with h2:
            st.markdown(
                """<span class='step-badge'>2</span>

**Run the comparison**

The app aligns both images and computes a structural similarity map between them.""",
            )
        with h3:
            st.markdown(
                """<span class='step-badge'>3</span>

**Review the findings**

See the similarity score, a visual diff map, and every changed region boxed and labeled.""",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Where this is used")
        st.markdown(
            """
            <span class='usecase-pill'>🕵️ Digital forensics</span>
            <span class='usecase-pill'>🖼️ Deepfake / AI-edit detection</span>
            <span class='usecase-pill'>📄 Document tamper checks</span>
            <span class='usecase-pill'>🔐 Fraud & authenticity verification</span>
            <span class='usecase-pill'>📸 Before/after QA</span>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("⬆️ Upload two images above and click **Compare images** to see it in action.")

with tab_detect:
    st.write(
        "Upload a single image to estimate how likely it is to be AI-generated "
        "or synthetic, based on classical forensic signal-processing cues "
        "(noise patterns, frequency-domain artifacts, and compression-error "
        "consistency) -- not a trained deep-learning classifier."
    )

    detect_file = st.file_uploader(
        "Image to analyze", type=["png", "jpg", "jpeg"], key="detect_img"
    )

    if detect_file:
        detect_img = load_image(detect_file)

        if st.button("Analyze image"):
            with st.spinner("Running forensic signal analysis..."):
                label, confidence, details = analyze_ai_likelihood(detect_img)

            d1, d2 = st.columns([1, 1])
            with d1:
                st.image(to_display(detect_img), caption="Uploaded image", use_container_width=True)
            with d2:
                st.markdown(
                    f"""
                    <div class="analysis-box">
                        <span class="label-line">AI IMAGE ANALYSIS</span><br><br>
                        <span class="label-line">Classification:</span><br>
                        <span class="value-line">{label}</span><br><br>
                        <span class="label-line">Confidence:</span><br>
                        <span class="value-line">{confidence:.0f}%</span><br>
                        <span class="important">
                        Important: this result comes from heuristic image-forensics
                        signals, not a trained AI-detection model. Treat it as a
                        probabilistic signal for further review, not proof of
                        AI generation or authenticity.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("#### Signal breakdown")
            for name, value in details.items():
                render_signal_bar(name, value)
    else:
        st.info("⬆️ Upload one image above and click **Analyze image** to see its AI-generation likelihood.")

st.markdown("---")
st.caption(
    "Built with Streamlit and OpenCV, using a self-contained SSIM (Structural "
    "Similarity Index) implementation and heuristic forensic signal analysis. "
    "These techniques are inspired by real-world digital forensics methods for "
    "detecting image tampering and manipulation, but are simplified for this "
    "student project and should not be relied on as definitive proof."
)
