# Installation

## From GitHub

```bash
pip install git+https://github.com/Shlhnj/strategicc.git
```

This installs the core package with its required dependencies:

- `numpy>=1.24`
- `pandas>=2.0`
- `Pillow>=10.0`
- `matplotlib>=3.7`

These four cover the simulation engine, SEEA-EA accounting, Stock & Flow, and all plotting — everything except the calibration module and MP4 animation export.

## Optional dependencies

### Calibration module (`strategicc.calibration`)

Deriving inputs from a historical LULC time series zip requires `rasterio`:

```bash
pip install rasterio
```

If you don't install it, the rest of the package works fine — you'll just need to build `Transitions.csv`, `TransitionMultipliers.csv`, and any age raster by hand or by other means instead of using `load_lulc_timeseries()` / `compute_age_raster()` / `compute_transition_rates()`.

### MP4 animation export

`strategicc.animate()` can save GIFs out of the box (uses Pillow, already a core dependency). MP4 output additionally requires the `ffmpeg` binary to be installed and on your system `PATH` — this is a system package, not a Python package:

```bash
# Ubuntu / Debian / Colab
apt-get install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Conda
conda install ffmpeg
```

If `ffmpeg` isn't found, `animate(..., output_format="mp4")` raises a clear `RuntimeError` telling you to install it or use `output_format="gif"` instead.

### Development / testing

```bash
pip install -e ".[dev]"
```

Installs `pytest` and `pytest-cov` for running the test suite.

## Google Colab

Colab already has `numpy`, `pandas`, `matplotlib`, `Pillow`, and `ffmpeg` preinstalled. You only need:

```python
!pip install git+https://github.com/Shlhnj/strategicc.git --quiet
!pip install rasterio --quiet   # only if using the calibration module
```

## Verifying the install

```python
import strategicc
print(strategicc.__version__)

from strategicc import StrategiccEngine, animate
print("Core import OK")
```
