# Image OCR System

An intelligent image-region OCR pipeline: crop a region from an image (interactively or via
coordinates), automatically assess and repair its quality (deblurring, super-resolution,
contrast/illumination fixes), run it through EasyOCR/Tesseract, and fall back to a
vision-language model (via [OpenRouter](https://openrouter.ai/)) when confidence is low.

## How it works

1. **Load & select** — load the input image and crop a region of interest, either
   interactively with an OpenCV `selectROI()` window or non-interactively with `--roi x,y,w,h`.
2. **Quality check** — analyze the crop for blurriness, low contrast, and low resolution.
3. **Preprocessing** — if needed, generate a set of enhancement candidates: deblurring
   (Wiener / Richardson-Lucy deconvolution), super-resolution upscaling, CLAHE contrast
   correction, sharpening, and deskewing.
4. **OCR** — run each candidate through EasyOCR (with an optional Tesseract fallback) and
   score the results.
5. **VLM fallback** — if OCR confidence is still low, send the crop to a vision-language
   model through the OpenRouter API for a final read.
6. **Output** — the best crop, the processed crop, and a JSON result (text + confidence +
   metadata) are written to `output/`.

## Project structure

```
.
├── main.py                # CLI entry point
├── src/
│   ├── config.py           # All tunable thresholds/paths in one place
│   ├── image_loader.py     # Image loading & validation
│   ├── selector.py         # ROI selection (interactive or coordinate-based)
│   ├── quality_checker.py  # Blur/contrast/resolution analysis
│   ├── preprocessing.py    # Candidate generation (contrast, sharpen, deskew, etc.)
│   ├── deblur.py            # Wiener / Richardson-Lucy deconvolution
│   ├── super_resolution.py # Upscaling for small/low-res crops
│   ├── ocr_engine.py        # EasyOCR + Tesseract wrapper and scoring
│   ├── vlm_ocr.py           # OpenRouter vision-language-model fallback
│   └── pipeline.py          # Orchestrates the full flow
├── input/                  # Sample input images
└── output/                  # Pipeline results (git-ignored, kept via .gitkeep)
```

## Setup

```bash
git clone <your-repo-url>
cd image-ocr-system
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy the env template and add your own [OpenRouter](https://openrouter.ai/) API key
(only needed for the VLM fallback step):

```bash
cp .env.example .env
```

> **Tesseract** is optional (used only as a fallback OCR engine) and must be installed
> separately as a system binary — see the comment in `requirements.txt`.

## Usage

Interactive region selection (opens a GUI window — requires a display):

```bash
python main.py input/sample_image.png
```

Non-interactive, e.g. on a headless machine/server/container:

```bash
python main.py input/sample_image.png --roi 5,133,205,10
```

Skip the before/after preview windows at the end:

```bash
python main.py input/sample_image.png --no-preview
```

Results are written to `output/selected_crop.png`, `output/processed_crop.png`, and
`output/result.json`.

## Configuration

All thresholds (blur, contrast, resolution, confidence bands, etc.) live in
`src/config.py` and can be tuned without touching pipeline logic.

## License

[MIT](LICENSE)
