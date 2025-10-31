### Artery Feature Extraction Pipeline

This repository provides a fully automated pipeline to measure arterial wall thickness (intima & media) and compute derived morphometric features from histology whole-slide images (WSIs) and their corresponding GeoJSON annotations.

It reads artery‐specific annotations (media, intima, lumen contours), computes per-artery thickness maps, area fractions, and intima-to-media ratios, and saves per-artery feature summaries in CSV and JSON formats.

### Overview

| Component             | Description                                                                                                                              |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Input WSIs**        | `.svs` or `.ndpi` whole-slide images (e.g., trichrome stain)                                                                             |
| **Input annotations** | `.geojson` files containing labeled polygons (“Media”, “Intima”, “Lumen”)                                                                |
| **Outputs**           | JSON file (`thickness.json`) with raw measurements, and CSV file (`artery_features_no_labels.csv`) with aggregated morphometric features |
| **Visualization**     | Overlay figures saved under `results/figures/`                                                                                           |

### Directory Structure

```
artery_thickness/
├── artery_analysis.py           # Main CLI script (this file)
├── helper.py                    # Supporting geometry & analysis utilities
├── environment.yml             # Optional dependency list
├── data/
│   ├── wsi/                     # Input WSIs (.svs/.ndpi)
│   └── ann_geojson/             # Corresponding annotation files (.geojson)
└── results/
    ├── thickness.json           # Raw per-artery measurements
    ├── artery_features_no_labels.csv  # Aggregated feature summary
    └── figures/                 # Visualization outputs

```

### Usage

You can run the pipeline with a single command using the CLI interface.

Example:
python artery_analysis.py \
  --wsi_dir /path/to/wsi \
  --ann_dir /path/to/annotations \
  --save_dir /path/to/save/results


### Output Files
| File                                | Description                                                            |
| ----------------------------------- | ---------------------------------------------------------------------- |
| **`thickness.json`**                | Raw artery-level thickness and area measurements (per artery per WSI). |
| **`artery_features_no_labels.csv`** | Final computed morphometric feature table.                             |
| **`figures/`**                      | Optional visualization overlays for QA.                                |


Each row in the CSV corresponds to a unique (WSI_ID, Artery_ID) pair, including features like:

| Feature                   | Description                                      |
| ------------------------- | ------------------------------------------------ |
| `Thickness_Media_um`      | Media layer thickness (μm)                       |
| `Thickness_Intima_um`     | Intima layer thickness (μm)                      |
| `Media Area (microm2)`    | Media area (μm²)                                 |
| `Intima Area (microm2)`   | Intima area (μm²)                                |
| `Lumen Area (microm2)`    | Lumen area (μm²)                                 |
| `Ratio Intima/Media Area` | Ratio of intima to media area                    |
| `Media Area Frac`         | Fraction of total artery area occupied by media  |
| `Intima Area Frac`        | Fraction of total artery area occupied by intima |
| `Lumen Area Frac`         | Fraction of total artery area occupied by lumen  |


Absolutely 👍 — here’s a **clean, polished version** of your *Computation Flow* section, formatted properly for a **GitHub README.md**.
It uses nested bullet points, consistent tense, and Markdown formatting that renders nicely on GitHub:

---

### Computation Flow

The artery wall thickness analysis pipeline proceeds through the following stages:

1. **Load Annotations**

   * Reads polygonal regions from `.geojson` annotation files.
   * Cleans coordinate data and filters polygons based on their labels (`Media`, `Intima`, `Lumen`).

2. **Determine Region of Interest (ROI)**

   * Computes the global bounding box encompassing all annotated artery polygons.
   * Defines the region of the slide to extract for downstream analysis.

3. **Read Slide Region**

   * Extracts the corresponding tissue region from the whole-slide image (WSI) using **OpenSlide**.
   * Converts RGBA images to RGB for consistency in visualization and processing.

4. **Polygon Processing**

   * Transforms global annotation coordinates to local (crop-level) coordinates.
   * Crops and overlays contours on the extracted image for visual validation.

5. **Measurement**

   * Computes pixel-level areas for lumen, intima, and media.
   * Converts pixel areas to physical units (µm²) using the slide’s **microns-per-pixel (MPP)** metadata.
   * Measures **intima** and **media** thickness along evenly spaced radial angles.

6. **Feature Extraction**

   * Aggregates morphometric descriptors such as mean, standard deviation, and ratios.
   * Computes structural metrics like area fractions and intima/media ratios.
   * Outputs results in both **structured CSV** and **JSON** formats for easy downstream analysis.

---

Would you like me to also format the next section (“Output Files”) in the same polished GitHub style for consistency?

```
Author

Harishwar Reddy Kasireddy
Ph.D. Student, Electrical & Computer Engineering
University of Florida — Sarder Lab (Computational Pathology)
harishwarreddy.k@ufl.edu
```

## Relation to Prior Work

This repository is a **re-implementation (engineering reproduction)** of the core measurement pipeline described in:

> **Zhou J., Li X., Demeke D., Dinh T.A., et al.** “Characterization of arteriosclerosis based on computer-aided measurements of intra-arterial intima-media thickness.” *Journal of Medical Imaging* 11(5):057501, 2024. doi:10.1117/1.JMI.11.5.057501. ([ResearchGate][1])
