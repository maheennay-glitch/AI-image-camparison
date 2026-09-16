# 🔍 AI Image Comparison

A web app that compares two images, measures how similar they are, and
visually highlights any differences — including differences introduced by
AI-generated or edited content.

This project applies **image forensics** techniques (used in digital
security to detect tampering and manipulation) via the **Structural
Similarity Index (SSIM)**.

## Features

- Upload an original image and a comparison image (e.g. an AI-edited version)
- Similarity score (0–100%)
- Automatic detection and bounding-box highlighting of changed regions
- Visual side-by-side comparison and difference map
- Download the annotated result

## Why this matters for security

Detecting whether an image has been altered — manually or by AI — is a
real problem in digital forensics, fraud prevention, and media
verification. This project is a small, working example of that kind of
tamper-detection pipeline.

## Tech stack

- Python
- [Streamlit](https://streamlit.io/) — web UI
- OpenCV — image processing
- scikit-image — SSIM comparison algorithm
- Pillow, NumPy

## Run it locally

```bash
git clone https://github.com/YOUR-USERNAME/ai-image-comparison.git
cd ai-image-comparison
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Project structure

```
ai-image-comparison/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md
```

## Live demo

_Add a link here once deployed on [Streamlit Community Cloud](https://streamlit.io/cloud) (free — see setup guide)._

## Design

- Color palette: Purple `#A63ACB`, Pink `#E85DA0`, White
- Typography: Poppins (headings), Inter (body)
