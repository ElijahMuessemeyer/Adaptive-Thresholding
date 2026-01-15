# Adaptive Thresholding Processor

Comprehensive image processing system comparing different adaptive thresholding techniques for various scene types.

## Features

- **Multiple Preprocessing Levels**: Light, medium, and heavy preprocessing pipelines
- **Scene Type Analysis**: Indoor, outdoor, and close-up image processing
- **Synthetic Test Images**: Generates controlled test scenarios
- **Comparative Visualization**: Side-by-side method comparison
- **Results Tracking**: JSON output of processing results

## Techniques Implemented

- Gaussian adaptive thresholding
- Mean adaptive thresholding
- Otsu's binarization
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Bilateral filtering

## Usage

```bash
python adaptive_thresholding.py
```

## Requirements

- Python 3.8+
- OpenCV (cv2)
- NumPy
- Matplotlib
