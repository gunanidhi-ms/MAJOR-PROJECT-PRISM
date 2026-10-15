# PRISM: Predictive Radiology Intelligence & Screening Model

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg?logo=react)](https://reactjs.org/)
[![WebSocket](https://img.shields.io/badge/WebSocket-Live-green.svg)]()
[![DICOM](https://img.shields.io/badge/DICOM-C--STORE-blueviolet.svg)]()
[![Medical Imaging](https://img.shields.io/badge/Medical%20Imaging-Research-ff69b4.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Phase 1: Real-Time Slice-Level Emergency Screening Engine**

*(Animated demonstration of the live pipeline rendering incoming DICOM slices with bounding boxes)*
![PRISM Live Demonstration](docs/assets/prism_demo.gif)

---

## Project Goals

**Current Status (Phase 1)** — ✅ Complete
- [x] Live DICOM SCP network ingestion
- [x] Real-time latency (Target: <50ms per slice)
- [x] Universal statistical triage (disease-agnostic)
- [x] Dynamic geometry and bounding box generation
- [x] Live WebSocket-driven frontend dashboard

**Current Status (Phase 2)** — 🚧 In Progress
- [x] **Package 0:** Schema contracts frozen (`phase1_handoff.json`, `organ_statistics.json`, `findings_output.json`, shared `Candidate` dataclass)
- [x] **Package 1:** Phase 1 → Phase 2 handoff infrastructure (Volume Accumulator, Finding Tracker, extended DICOM tags, dual-path Phase 2 trigger)
- [x] **Package 2:** Volume Assembly & Segmentation Runner
- [x] **Package 3:** Organ Baseline & Region Detection
- [x] **Package 4:** Path A — 3D Clustering of Phase 1 Seeds
- [x] **Package 5:** Path B — Independent Organ-Wide Sweep
- [x] **Package 6:** Technical Suppression Cascade, Merge & 3D Re-validation
- [x] Package 7: Confidence Fusion, Scoring & Phase 3 Handoff
- [x] Package 8: Testing, Calibration & Final Report

**Report Enrichment (cross-phase)** — ✅ Implemented (see [Section 19](#19-report-enrichment-clinical-context-localisation--impression-synthesis))
- [x] Clinical indication, DICOM-derived technique and prior-study comparison in the report
- [x] Intra-organ sub-localisation and organ-contact commentary
- [x] Anatomical positions (side, vertebral level) for unassigned ("Unclassified") lesions
- [x] Organ-volume deviations promoted to a first-class Impression category
- [x] Grouped, de-templated Impression synthesis

**ML False Positive Reduction** — 🔄 Roadmap (see [Section 20](#20-ml-false-positive-reduction-fpr--roadmap))
- [ ] XGBoost/LightGBM classifier on existing radiomic features
- [ ] Evaluation on KiTS23 / LIDC-IDRI public datasets (IoU-based, no medical knowledge required)
- [ ] AUROC, sensitivity at 2 FP/image, FP-reduction % metrics
- [ ] `fpr_score` passed through to Phase 3 for report confidence annotation

---

## Table of Contents

1. [Running PRISM (Quick Start & One-Command Scripts)](#1-running-prism-quick-start)
2. [Project Overview](#2-project-overview)
3. [Motivation](#3-motivation)
4. [Design Decisions](#4-design-decisions)
5. [System Architecture & Data Flow](#5-system-architecture--data-flow)
6. [Universal Phase 1 Triage Engine](#6-universal-phase-1-triage-engine)
7. [Validation Methodology](#7-validation-methodology)
8. [Module Breakdown](#8-module-breakdown)
9. [Frontend Architecture & Automated Workflow](#9-frontend-architecture)
10. [Data Models](#10-data-models)
11. [Configuration & Model Selection](#11-configuration)
12. [Developer Guide](#12-developer-guide)
13. [Phase 2 Development & Package Architecture](#13-phase-2-development--handoff-infrastructure)
14. [Limitations](#14-limitations)
15. [Repository Structure](#15-repository-structure)
16. [License / References](#16-license--references)
17. [End-to-End Diagnostic Workflow & Architecture Updates](#17-end-to-end-diagnostic-workflow--architecture-updates)
18. [Complete Step-by-Step User Guide](#18-complete-step-by-step-user-guide)
19. [Report Enrichment: Clinical Context, Localisation & Impression Synthesis](#19-report-enrichment-clinical-context-localisation--impression-synthesis)
20. [ML False Positive Reduction (FPR) — Roadmap](#20-ml-false-positive-reduction-fpr--roadmap)

---

## 1. Running PRISM (Quick Start)

PRISM provides automated installation and one-command launch scripts for **Windows** (PowerShell), **Linux**, and **macOS** (Bash).

### Prerequisites
- **Python 3.10+** (Python 3.10 – 3.12 recommended for full PyTorch CUDA GPU builds)
- **Node.js 18+** & **npm**
- **Ollama** (for Phase 3 LLM report generation; auto-installed and configured by setup scripts)
- **GPU (Optional):** NVIDIA GPU with CUDA support or Apple Silicon (MPS). Automatically detected; the system seamlessly falls back to CPU if no GPU is present.

---

### One-Click Installation

The repository includes automated installer scripts that create the Python virtual environment, detect your GPU and install CUDA-accelerated PyTorch (or CPU fallback), install all dependencies (including `psutil` for dynamic memory scaling), pull the recommended `llama3.2:3b` model, and install frontend packages.

#### Windows (PowerShell)
```powershell
.\install_prism.ps1
```

#### Linux & macOS (Bash)
```bash
chmod +x install_prism.sh start_prism.sh
./install_prism.sh
```

---

### One-Command Startup

To run the entire PRISM Autonomous Pipeline (Backend Orchestrator, Phase 1 Ingestion, Phase 2 Segmentation, Phase 3 Reporting Service, and React Frontend):

#### Windows (PowerShell)
```powershell
.\start_prism.ps1
```
*This script starts the backend and frontend, checks Ollama health, sets environment variables, performs health polling on `:8001` and `:8005`, displays a live hardware diagnostic profile (GPU, RAM, and dynamic Phase 2 timeout formula), records process IDs in `.prism_pids.json` for automatic tab-close termination, and auto-opens `http://localhost:5173/` in your browser.*

#### Linux / macOS (Bash)
```bash
./start_prism.sh
```

---

### Manual Setup & Component Details

If you prefer to set up or run individual components manually:

#### 1. Ollama LLM Backend (Phase 3 AI Reporting)

Phase 3 uses **`llama3.2:3b`** (2.0 GB) as the standard, lightweight, high-performance medical reporting model. It fits entirely in 4 GB VRAM (e.g. NVIDIA RTX 3050 Ti) and generates reports in 2–3 seconds with zero out-of-memory errors.

**Manual Installation:**
- **Windows:** `winget install --id Ollama.Ollama --accept-source-agreements --accept-package-agreements` or from https://ollama.com/download/windows
- **Linux / macOS:** `curl -fsSL https://ollama.com/install.sh | sh`

**Pull Model:**
```bash
ollama pull llama3.2:3b
```

#### 2. Python Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# Linux / macOS:
# source venv/bin/activate

# Install dependencies (CPU or CUDA)
pip install -r requirements.txt
```

#### 3. Frontend Setup

```bash
cd prism-react
npm install
npm run dev
```

#### 4. Orchestrator Backend Startup

```bash
# From project root with venv activated
python orchestrator.py
```

### Troubleshooting
- **Address already in use (Ports 11112, 8001, 8005, 5173):** Check for conflicting processes with `netstat -ano` on Windows or `lsof -i` on Linux/macOS. Use the Shutdown button in the UI to cleanly terminate all background services.
- **Frontend not receiving data:** Verify the WebSocket server is running at `ws://localhost:8001/ws/alerts`.
- **Phase 3 reports use template fallback:** Ollama is unavailable, intentionally disabled (`SKIP_LLM=true`), or the model is missing. Ensure `ollama serve` is running and `llama3.2:3b` is pulled (`ollama list`).
- **Ollama model architecture error (`mllama`):** Do not use `llama3.2-vision` for text chat; use `llama3.2:3b`. The install and start scripts automatically purge incompatible models.
- **Phase 2 Timeout on Large Volumes:** Phase 2 automatically computes an adaptive timeout based on GPU and RAM (`compute_dynamic_timeout`). For CPU-only systems with <8 GB RAM, timeouts automatically scale up to 1800s.




---

## 2. Project Overview

**PRISM (Predictive Radiology Intelligence & Screening Model)** is an open-source, research-oriented medical imaging framework designed to perform real-time CT triage while a scan is still being acquired.

Unlike conventional Computer-Aided Diagnosis (CAD) systems that analyze an entire reconstructed CT volume after image acquisition has completed, PRISM begins processing immediately as each DICOM slice is transmitted from the CT scanner. This enables analysis to occur concurrently with image acquisition rather than after the examination has finished.

The objective of PRISM Phase 1 is not to diagnose disease. Instead, it continuously evaluates every incoming slice for statistically significant density abnormalities that may represent clinically important findings requiring immediate attention.

When an abnormal region is detected, the system:

- Identifies the abnormal region.
- Generates a spatial bounding box.
- Computes quantitative measurements.
- Assigns a confidence estimate.
- Calculates an emergency score.
- Streams the result to the frontend through WebSockets in real time.

This allows radiologists to begin reviewing potentially abnormal slices before the entire CT examination has completed, reducing the Time-to-First-Finding (TTFF) without interrupting the existing clinical workflow.

PRISM follows a modular multi-phase architecture.

**Phase 1** performs universal statistical triage.

Future phases may perform:

- Organ segmentation
- Disease classification
- Anatomical reasoning
- Clinical interpretation
- Structured report generation

By separating triage from diagnosis, each phase remains specialized, explainable, and independently improvable.

---

## 3. Motivation

### Why PRISM Exists

Modern CT scanners generate hundreds to thousands of slices within seconds. Although image acquisition is rapid, interpretation is often delayed because every completed examination enters a centralized PACS queue where studies are typically reviewed in chronological or priority order.

During busy emergency department workflows, even highly urgent studies may remain unreviewed for several minutes while the patient has already left the scanner.

PRISM was developed to reduce this delay.

Instead of waiting until an examination is complete, PRISM intercepts the DICOM stream directly from the scanner and evaluates every slice as it is generated.

This transforms CT analysis from a post-acquisition workflow into a continuous real-time screening process.

The goal is not to replace radiologists.

The goal is to provide an intelligent early-warning mechanism that immediately highlights slices containing statistically unusual structures, allowing radiologists to prioritize potentially critical examinations earlier.

---

### Why Slice-Level Triage Matters

Most existing AI systems require the entire CT volume before analysis can begin.

Their workflow typically consists of:

CT Acquisition

↓

Volume Reconstruction

↓

PACS Storage

↓

AI Processing

↓

Results Returned

↓

Radiologist Review

This introduces unavoidable latency because every stage depends upon the previous stage completing first.

PRISM fundamentally changes this workflow.

Instead of processing completed studies, PRISM evaluates slices individually as soon as they appear on the DICOM network.

The processing pipeline therefore becomes:

Slice Acquired

↓

Slice Received

↓

HU Conversion

↓

Statistical Triage

↓

Frontend Alert

↓

Next Slice

Since every slice is processed independently, analysis overlaps with image acquisition instead of waiting for reconstruction.

This significantly reduces the Time-to-First-Finding while maintaining continuous throughput.

---

### Why PRISM Phase 1 Does Not Perform Diagnosis

One of the fundamental design principles of PRISM is strict phase separation.

Phase 1 intentionally avoids diagnosis.

Its responsibility is to answer only three universal questions:

1. Is this slice statistically normal?
2. If not, where is the abnormality?
3. How urgent does this abnormality appear?

Notice that none of these questions require understanding diseases.

Phase 1 deliberately does not attempt to determine:

- which organ is affected,
- what disease is present,
- whether the finding represents hemorrhage, tumor, infarction, infection, trauma, or artifact,
- what treatment should be performed.

Those questions require considerably more anatomical understanding, clinical reasoning, volumetric context, and disease-specific knowledge.

They belong in later phases.

By limiting the scope of Phase 1, the system remains:

- extremely fast,
- deterministic,
- explainable,
- computationally lightweight,
- independent of disease-specific training.

This separation also prevents the triage engine from becoming tightly coupled to a particular pathology or segmentation model.

Instead, Phase 1 simply identifies regions that deserve immediate human attention.

Diagnosis always remains the responsibility of subsequent processing stages and ultimately the interpreting radiologist.

---

## 4. Design Decisions

Reviewers often ask why PRISM is built the way it is. Here are the core engineering choices guiding the architecture:

### Why no Deep Learning in Phase 1?
Deep learning models are typically pathology-specific (e.g., a Subdural Hematoma detector) and highly sensitive to out-of-distribution data (new scanner kernels, metallic artifacts). They also require heavy GPU overhead. Phase 1 must be instantaneous and catch *any* anomaly, known or unknown. Statistical outlier detection achieves this universally on standard CPUs. DL is reserved for Phase 2 organ segmentation.

### Why no Organ Segmentation?
Segmenting organs in 2D is unreliable, and segmenting in 3D requires waiting for the entire scan to finish. By omitting segmentation, Phase 1 can process a slice the millisecond it hits the network card, maximizing speed.

### Why Statistical Detection?
Instead of hardcoding rules (e.g., "Blood is always exactly 60 HU"), statistical detection asks: *Is this voxel mathematically impossible for this specific patient's baseline?* This allows the algorithm to generalize automatically across contrasting scanner calibrations and patient demographics.

### Why Hounsfield Units (HU)?
Raw DICOM pixel values are proprietary integers. HU is a standard physical scale (Air = -1000, Water = 0). Vectorized conversion to HU ensures that PRISM's math behaves identically whether the scan was taken on a GE, Siemens, or Canon machine.

### Why Slice-by-Slice?
Scanners transmit slices sequentially. By analyzing them individually as a 2D matrix, we eliminate the memory overhead and latency of assembling a 3D volumetric matrix in RAM.

### Why an Asynchronous Pipeline?
DICOM transmission over TCP is chaotic; packets drop, and slices arrive out-of-order via multiple threads. An async, queue-based architecture decouples the slow I/O network operations from the high-speed mathematical CPU operations.

### Why WebSockets instead of REST?
A REST architecture requires the frontend to constantly poll the backend, introducing artificial latency and heavy HTTP overhead. WebSockets allow the backend to instantly push alerts directly to the React canvas at 20+ frames per second.

### Why Phase 1 Does Not Use Deep Learning

A common question is why PRISM does not use state-of-the-art deep learning models such as TotalSegmentator, nnU-Net, MONAI pipelines, or disease-specific neural networks during Phase 1.

The answer is architectural rather than computational.

Phase 1 is solving a fundamentally different problem.

Deep learning models attempt to understand anatomy.

PRISM Phase 1 attempts only to detect statistically abnormal image content.

These objectives are intentionally separated.

---

### Why TotalSegmentator Is Not Used

Models such as TotalSegmentator are remarkable tools for anatomical segmentation.

However, they require an entirely reconstructed 3D CT volume before inference can begin.

Their workflow typically involves:

Complete CT Volume

↓

3D Preprocessing

↓

GPU Inference

↓

Organ Segmentation

↓

Anatomical Labels

Only after segmentation can disease-specific algorithms begin.

This workflow is ideal for diagnosis.

It is not ideal for real-time triage.

Phase 1 cannot wait for:

- the complete examination,
- volumetric reconstruction,
- GPU scheduling,
- segmentation,
- post-processing.

Instead, PRISM processes every slice independently the moment it arrives.

The result is immediate statistical screening rather than delayed anatomical understanding.

---

### Why Phase 1 Does Not Perform Organ Segmentation

Organ segmentation answers questions such as:

- Where is the liver?
- Where are the kidneys?
- Where is the spleen?

Phase 1 does not need these answers.

Instead, it asks:

"Is there anything statistically unexpected within this slice?"

This distinction is important.

A statistically unusual region can be detected without knowing which organ contains it.

Avoiding segmentation provides several advantages:

- zero dependency on anatomical models,
- reduced computational cost,
- no GPU requirement,
- immediate processing,
- applicability across any body region.

The same statistical engine can therefore process:

- head CT,
- neck CT,
- chest CT,
- abdomen CT,
- pelvis CT,
- spine CT,
- extremity CT,

without changing models.

---

### Why Statistical Detection Instead of Disease Detection

Most AI systems attempt to classify known diseases.

PRISM Phase 1 does not.

Instead, it compares every voxel against an adaptive statistical baseline computed directly from the current slice.

Rather than asking:

"Is this hemorrhage?"

it asks:

"Is this density significantly different from the expected physiological distribution within this slice?"

This allows Phase 1 to remain disease-agnostic.

Previously unseen diseases, scanner variations, or unexpected abnormalities can still appear as statistical outliers without requiring explicit training data.

---

### What Phase 1 Can Do

Phase 1 currently performs the following functions completely automatically:

- receives live DICOM slices,
- converts raw pixels into Hounsfield Units,
- removes scanner padding,
- constructs a robust body mask,
- identifies scan type,
- estimates anatomical body region,
- suppresses expected anatomy,
- computes adaptive statistical baselines,
- detects statistical outlier regions,
- groups pixels into connected abnormalities,
- filters imaging artifacts,
- generates bounding boxes,
- computes quantitative measurements,
- estimates confidence,
- calculates emergency scores,
- streams results to the frontend in real time.

Importantly, none of these operations require prior knowledge of a specific disease.

Instead, they provide a universal triage layer that can operate before any organ segmentation or disease classification begins.

Phase 1 therefore functions as the front gate of the PRISM pipeline, rapidly identifying slices that deserve immediate clinical attention while leaving detailed anatomical interpretation to future phases.

## 5. System Architecture & Complete Data Flow

PRISM is orchestrated through a highly concurrent pipeline using independent Python modules connected by thread-safe queues.

1. **CT Scanner** generates Slice 42.
2. **Scanner** opens a TCP socket and initiates a DICOM Association (C-STORE) with PRISM on port `11112`.
3. `dicom_listener.py` parses the byte-stream, extracts the `PixelData` and spatial/scaling tags (including Phase 2 requirements like `ImagePositionPatient` Z-axis), and pushes it to `slice_buffer.py`.
4. `slice_buffer.py` locks the priority queue, inserts the slice, and checks for contiguous order. If ready, it flushes the slice to `pipeline.py`.
5. `pipeline.py` passes the raw matrix to `hu_transform.py`.
6. `hu_transform.py` vectorizes the array into Hounsfield Units, crops the scanner padding, and returns a float32 matrix.
7. `pipeline.py` passes the float32 matrix to `triage_screen.screen_slice`.
8. `triage_screen.py` executes the 10-step math engine, outputting a `TriageResult` dataclass.
9. **(Parallel Handoff):** `pipeline.py` feeds the HU array and findings to `volume_accumulator.py` to silently build a 3D volume (ordered by True Z-coordinate) without blocking the real-time path.
10. `pipeline.py` converts the result to JSON and pushes it into the `ws_server.py` asyncio queue.
11. `ws_server.py` broadcasts the JSON string over port `8001`.
12. The **React Frontend** receives the WebSocket message, draws the slice to an HTML5 canvas, and renders SVG bounding boxes if an alert is triggered.
13. **(Phase 2 Trigger):** When `dicom_listener.py` detects an Association Release (or `volume_accumulator.py` hits a timeout/slice-count limit), Phase 2 is launched as an isolated subprocess using the assembled 3D volume.

---

## 6. Universal Phase 1 Triage Engine

The core logic of PRISM resides within `triage_screen.py`. It evaluates an unknown slice of human anatomy and extracts clinically actionable emergency alerts in milliseconds. The mechanism is broken into 10 mathematical stages.

### Step 0: Slice Quality Assessment
* **Why this step exists:** A single corrupted network packet can destroy the statistical baseline, triggering hundreds of false alerts.
* **Inputs:** 2D numpy array (float32 HU).
* **Processing:** Checks array dimensionality, boundary extremes (`< -2000` or `> +5000`), and total body pixel count.
* **Outputs:** Boolean validity flag and failure reason string.
* **Complexity:** $O(N)$
* **Failure Cases:** A slice with extreme metallic artifact covering the whole FOV may be incorrectly rejected as corrupted.
* **Example:**
  > **Input:** Array with max value `32000` (Network corruption).
  > **Output:** Rejected. "Corrupted HU values".

### Step 1: Robust Body Mask Generation
* **Why this step exists:** To prevent the statistical baseline from being skewed heavily toward `-1000 HU` by analyzing empty room air or the scanner table.
* **Inputs:** 2D numpy array (float32 HU).
* **Processing:** Thresholds at `HU > -600`. Performs binary opening, closing, and hole-filling to solidify internal body cavities (like lungs). Finally, dilates the mask outward by 15 pixels using `distance_transform_edt` to encompass subcutaneous fat.
* **Outputs:** Boolean 2D array (Body Mask).
* **Complexity:** $O(N)$ with multiple morphological passes.
* **Failure Cases:** Scans with limbs pressed against the chest may cause the mask to merge the two regions inappropriately.
* **Example:**
  > **Input:** Chest slice with arms in FOV.
  > **Output:** A unified mask covering the torso and arms, completely excluding the air gap between them.

### Step 2A & 2B: Scan Type & Body Region Identification
* **Why this step exists:** The engine must adapt to what it is looking at; normal contrast-enhanced blood looks like a hemorrhage in a non-contrast scan.
* **Inputs:** HU Array, Body Mask, DICOM Header Hint.
* **Processing:** Calculates the 99th percentile (P99) of soft tissue to flag `CONTRAST` vs `NON-CONTRAST`. Uses tissue fractions (e.g., Lung fraction > 25%) to dynamically identify the region (Chest, Head, Abdomen, etc.).
* **Outputs:** Strings for Scan Type and Body Region.
* **Complexity:** $O(M \log M)$ for percentiles on $M$ pixels.
* **Failure Cases:** Pathologically obliterated lungs (e.g., massive fibrosis) may drop the lung fraction, causing a Chest scan to be misidentified as an Abdomen.
* **Example:**
  > **Input:** Soft tissue P99 = `165 HU`, Lung Fraction = `31%`.
  > **Output:** Scan Type: `CONTRAST`, Region: `CHEST`.

Here is the updated version reflecting your current implementation.

---

### Step 3: Expected Normal Anatomy Suppression (The Search Mask)

* **Why this step exists:** Eliminates expected anatomical structures that would otherwise dominate the statistical analysis and generate false positives. Normal lungs, bowel gas, cortical bone, and outer body tissues should not be interpreted as abnormalities.
* **Inputs:** HU Array, Body Mask, Body Region.
* **Processing:** Computes an adaptive bone threshold `max(Body_P99, 350 HU)` and suppresses bone after 3 iterations of morphological dilation to capture partial-volume edge voxels. Region-specific suppression is then applied. For **CHEST**, a 2-tier approach suppresses large bilateral lung fields (`>5000 px` with typical lung density) and small central airways (`<2000 px`), leaving unusual-density air for pneumothorax detection. For **ABDOMEN**, only compact, moderate-density internal gas pockets are suppressed to preserve pathological free air (pneumoperitoneum). **HEAD** applies deeper skull erosion and suppresses CSF ventricles, while **NECK** and **PELVIS** have dedicated airway/gas suppression. A universal cleanup removes residual background air (`< -500 HU`) from regions not actively analyzing air.
* **Outputs:** Boolean 2D Search Mask.
* **Complexity:** Heavy $O(N)$ due to Connected Component Analysis, distance transforms, and morphological operations.
* **Failure Cases:** Pathological air collections connected to normal lung fields via thin gaps may be unintentionally suppressed if the morphological structures merge them.
* **Example:**

  > **Input:** Chest CT containing normal lungs (`-850 HU`, `55000 px`), trachea (`1500 px`), and a pneumothorax crescent (`3500 px`).
  >
  > **Output:** Lungs and trachea are suppressed. The pneumothorax crescent remains in the Search Mask for statistical evaluation.

---

### Step 4: Adaptive Statistical Baseline

* **Why this step exists:** Establishes a robust statistical representation of normal tissue while preventing abnormal regions from influencing their own reference distribution.
* **Inputs:** HU Array, Search Mask.
* **Processing:** Subsamples search-mask pixels (`[::4]`) for efficiency. Computes percentiles (`P0.5`–`P99.5`), then calculates the Mean and Standard Deviation only from the **trimmed P5–P95 distribution**. Also computes Median, MAD (Median Absolute Deviation), IQR, and percentile statistics.
* **Outputs:** Dictionary containing robust slice statistics.
* **Complexity:** $O(K \log K)$ where $K$ is the number of sampled pixels.
* **Failure Cases:** Extremely small search masks may produce unstable statistical estimates due to insufficient reference pixels.
* **Example:**

  > **Input:** Brain CT containing a large hyperdense hemorrhage.
  >
  > **Output:** Mean and Standard Deviation are computed from the trimmed normal tissue distribution, preventing the hemorrhage from shifting the baseline.

---

### Step 5 & 6: Statistical Outlier Detection, Connected Component Analysis, and Local Context Validation

* **Why this step exists:** Detects statistically abnormal tissue while minimizing false positives through multiple independent statistical tests and local neighborhood validation.
* **Inputs:** HU Array, Search Mask, Baseline Statistics.
* **Processing:** Computes Z-score, MAD-score, percentile, and IQR-based outlier measures for every pixel. A voxel is accepted only if it satisfies **at least two of four independent statistical criteria** (majority voting). Connected Component Analysis groups abnormal voxels into candidate regions. Each component is then validated using both global statistics and a **local neighborhood that excludes the component itself**, preventing the abnormality from contaminating its own reference statistics.
* **Outputs:** List of validated candidate components containing bounding boxes, centroids, area, and HU measurements.
* **Complexity:** $O(N)$ for statistical calculations and Connected Component Analysis.
* **Failure Cases:** Very diffuse abnormalities with low local contrast may not accumulate sufficient statistical evidence to satisfy the majority-voting criteria.
* **Example:**

  > **Input:** Baseline Mean = `40 HU`, Trimmed Std = `10 HU`, Candidate Region = `95 HU`.
  >
  > **Output:** The region satisfies multiple statistical tests (Z-score, MAD-score, percentile), survives Connected Component Analysis, passes local context validation, and proceeds to geometric filtering.

---

### Step 7: Geometric Filtering and Confidence Estimation

* **Why this step exists:** Removes geometrically implausible detections and assigns confidence based on multiple independent sources of evidence instead of assuming every statistical outlier is clinically meaningful.
* **Inputs:** Validated candidate components, Search Mask, Global Statistics.
* **Processing:** Rejects components that are highly elongated (Aspect Ratio `>6`), have low solidity (`<0.15`), or significantly overlap the Search Mask boundary. Confidence begins conservatively at **0.4** and increases only through strong statistical evidence, larger component area, high solidity, and strong local contrast. A severity score is then computed from statistical extremity and physical size.
* **Outputs:** Final list of `Finding` objects with confidence, severity score, bounding box, and anomaly type.
* **Complexity:** $O(C)$ where $C$ is the number of validated candidate components.
* **Failure Cases:** Small but clinically significant abnormalities may receive lower confidence because of their limited size, while unusually shaped true abnormalities may fail geometric filtering.
* **Example:**

  > **Input:** Compact hyperdense lesion (`Area = 1200 px`, `Global Z = 7.2`, `Evidence = 5`, `High Local Contrast`).
  >
  > **Output:** Component passes all geometric filters, receives high confidence (`≈0.9`), and is reported as a statistical finding.


### Step 8: Confidence Estimation
* **Why this step exists:** To provide transparency to the radiologist; determining if the engine thinks a finding is a "maybe" or a "definitely".
* **Inputs:** Filtered components, Baseline Statistics.
* **Processing:** Assigns base confidence `0.5`. Adds bonuses for high Z-scores ($+0.2$), large physical area ($+0.2$), and high solidity ($+0.1$). Caps at `1.0`. Computes a continuous Severity Score.
* **Outputs:** List of finalized `Finding` dataclasses.
* **Complexity:** $O(C)$.
* **Failure Cases:** Very small but highly lethal aneurysms might receive a low confidence score due to their small footprint.
* **Example:**
  > **Input:** Massive hyperdense bleed (Z = `8.5`, Area = `3000px`).
  > **Output:** Base `0.5` + Z-bonus `0.2` + Size-bonus `0.2` = Confidence `0.9` (90%).

### Step 9 & 10: Emergency Scoring & Action Recommendation
* **Why this step exists:** To collapse complex mathematical findings into a single, clinically actionable directive.
* **Inputs:** List of `Finding` objects.
* **Processing:** Points = `Severity * Confidence^1.5 * sqrt(Area / 500) * 40` (capped at 50 per finding). Total slice points mapped to: `CONTINUE` (0-19), `WATCH` (20-44), `URGENT REVIEW` (45-79), or `IMMEDIATE ALERT` (80-100).
* **Outputs:** Total integer score and Action string.
* **Complexity:** $O(C)$.
* **Example:**
  > **Scenario A (Small Anomaly):** Small pocket of bowel gas. Total area = `150px`.
  > **Calculation:** Severity (0.3) * Conf (0.4)^1.5 * sqrt(150/500) * 40 = `2.1 points`.
  > **Output:** `2 points` $\rightarrow$ **CONTINUE**.
  > 
  > **Scenario B (Large Anomaly):** Massive cranial bleed. Total area = `6000px`.
  > **Calculation:** Severity (0.67) * Conf (0.65)^1.5 * sqrt(6000/500) * 40 = `48.8 points` (Capped at 50).
  > **Output:** `49 points` $\rightarrow$ **URGENT REVIEW**.

---

## 7. Validation Methodology

Our goal is to continually validate the statistical thresholds against diverse datasets. Note that the current performance metrics are based on preliminary validation tests and serve as design targets rather than formal clinical guarantees.

**Datasets Evaluated:**
- **Healthy Baselines:** Routine non-contrast and contrast-enhanced CTs (to tune Expected Anatomy Suppression and minimize false positives).
- **Trauma Scans:** Acute cases containing massive hemorrhages, pneumothoraces, and foreign bodies (to ensure high sensitivity and proper Z-score outlier flagging).
- **Body Regions:** Balanced distribution across Head, Chest, Abdomen, Pelvis, and Extremity.

**Evaluation Metrics (Design Targets):**
- **False Positive Rate:** Target $< 5\%$. The primary metric improved by Step 3 (Anatomy Suppression).
- **False Negative Rate:** Heavily monitored. Geometric filtering (Step 7) is intentionally loose to prevent accidentally discarding linear pathologies (like subdural bleeds).
- **Latency:** Target $< 50ms$ per slice on commercial CPUs. Preliminary tests show execution times averaging `20ms - 45ms`.
- **Throughput:** Target $20+$ frames per second (fps) visualization on the React frontend via Canvas rendering.

---

## 8. Module Breakdown

### `dicom_listener.py`
The TCP gateway. Utilizes `pynetdicom.AE` to establish a DICOM C-STORE SCP on port `11112`. Decodes incoming byte-streams into `PixelData` arrays, extracts spatial/scaling tags (including Phase 2 critical fields like `ImagePositionPatient` Z-axis, `SeriesInstanceUID`, `PixelSpacing`, and `SliceThickness`), captures patient demographics (ID, Name, Age, Sex), saves incoming `.dcm` files to disk, and pushes slice dictionaries to the asynchronous queue. Features thread-safe association counters and non-blocking event hooks.

### `slice_buffer.py`
The priority reordering queue. DICOM slices routinely arrive out-of-order over network connections. The buffer uses an `OrderedDict` with `threading.Lock` to sort slices by `InstanceNumber`. Includes a timeout watchdog and a gap-detection mechanism to prevent pipeline stalls if a slice packet is dropped over the network. Automatically resets its internal sequence pointer when a new series starts.

### `hu_transform.py`
Vectorized pixel standardization. Executes `HU = Pixel * Slope + Intercept`. Disables hard-clipping during ingestion to preserve scanner padding (e.g., `-3024 HU`). Implements an `auto_crop` algorithm to strip artificial padding before processing.

### `pipeline.py`
The central Phase 1 orchestrator. Runs an asynchronous worker loop that pulls from `SliceBuffer`, checks for patient transitions (`_check_patient_change`), executes `triage_screen.py`, feeds the `VolumeAccumulator`, formats `TriageResult` payloads to JSON, and pushes them to `ws_server`. When DICOM associations release, `on_association_released()` offloads queue draining, buffer flushing, and Phase 2 triggering to an independent background thread (`assoc-release-worker`), preventing pynetdicom handler thread deadlocks. Exposes a module-level `get_pipeline()` accessor.

### `volume_accumulator.py`
The Phase 1 $\rightarrow$ Phase 2 volumetric bridge. Silently accumulates HU-transformed 2D slices and per-slice findings into a spatially ordered 3D volume. Uses the true `ImagePositionPatient` Z-coordinate for sorting (never `InstanceNumber`). Built with reentrant `threading.RLock()` to guarantee deadlock-free execution when full resets (`reset_all()`) are called on patient transitions. Maintains a `_completed_series_uids` guard that prevents re-segmentation of the same series while allowing clean multi-patient succession.

### `dicom_archive.py`
Automated patient archive manager. Registers incoming slices by `StudyInstanceUID` and patient demographics. On session termination (via UI Stop/Close or scanner completion), packages raw DICOM files, analysis records, and reports into a compressed `.zip` archive named by Patient ID and timestamp, stores it in `patient_archives/`, and clears temporary disk storage.

### `finding_tracker.py`
Cross-slice finding logic. Links 2D statistical outliers across consecutive slices using centroid proximity, enforcing a minimum Z-axis persistence to filter out single-slice noise before handing candidates to Phase 2.

### `ws_server.py`
The FastAPI and `asyncio` WebSocket telemetry server. Serves real-time alert broadcasts over `ws://localhost:8001/ws/alerts`. Manages scanner start/stop subprocesses (`/api/scanner/start`, `/api/scanner/stop`), explicit pipeline memory resets for new patients, session closure & archival (`/api/archive/close-session`), and graceful full-system shutdown (`/api/system/shutdown`).

---

## 9. Frontend Architecture & Automated Workflow

The frontend is a modern **React 18** application scaffolded with **Vite** and styled with a dark-mode, glassmorphic clinical design system in `viewer.css`.

### Unified 3-State Scanner Control & Session Machine
The scanner control panel in `LiveViewer.jsx` is driven by a deterministic 3-state machine that guarantees patient data integrity:

```
[READY] ──────▶ Start Scan ──────▶ [SCANNING] ──────▶ Stop Scan / Done ──────▶ [DONE]
   ▲                                  │                                            │
   │                                  ▼                                            ▼
   └──────────── Form Cleared ◀── Archive Confirmed ◀── Close & Zip Dialog ────────┘
```

1. **State: `ready` (Green `▶ Start Scan`)**
   - Requires mandatory patient metadata: **Patient ID**, **Patient Name**, **Age**, and **Sex** (with inline field validation).
   - Clicking "Start Scan" resets frontend memory, initializes WebSocket alerts, and launches the DICOM stream.
2. **State: `scanning` (Red `⏹ Stop Scan & Close Record`)**
   - Displays a live glowing green indicator badge with real-time stream count (`Live — Streaming CT slices (N received)`).
   - Clicking "Stop Scan" immediately halts the backend scanner and opens the confirmation dialog to archive and zip the patient record.
3. **State: `done` (Amber `🗜️ Archive & Close Patient Record`)**
   - Automatically activates when the scanner completes replay.
   - Pulses with a gentle amber glow (`@keyframes subtlePulse`), clearly designating the endpoint of the current patient's examination.
   - **Never** resets to "Start Scan" prematurely — holds all slices and findings in memory until the radiologist explicitly clicks "Archive & Close".
   - Confirming the archive triggers `/api/archive/close-session`, zips the patient record, clears frontend state, resets patient form fields, and returns the machine to `ready`.

### Automated End-to-End Clinical Flow
1. **New Report Trigger (`/` → `/live`):** Clicking "New Report" on the Dashboard routes directly to the Live CT Viewport (`/live`).
2. **Real-Time DICOM Ingestion & Triage (Phase 1):** Slices stream at 20+ fps via WebSocket. The viewport renders windowed CT imagery, interactive window/level presets, and live statistical bounding box overlays.
3. **Autonomous 3D Segmentation & AI Sweep (Phase 2):** When slice accumulation completes, Phase 2 is launched automatically with hardware-adaptive timeout scaling. A floating **Pipeline Progress Panel** displays active phase indicators (Phase 1 ✅, Phase 2 🔄, Phase 3 ⏳), an animated progress bar, and a live elapsed timer.
4. **AI Report Synthesis & Auto-Navigation (Phase 3 → `/report/:studyId`):** When Phase 3 completes, the Live Viewer displays a 3-second countdown and navigates directly to the **Draft Report Editor**. The generated report follows a full clinical radiology format: **Indication**, **Technique**, **Findings** (every organ segmented by TotalSegmentator, grouped by body system — normal organs listed as unremarkable with HU density and volume, anomalous organs with quantitative measurements), and **Impression** (concise bullet summary of significant findings).
5. **Review, In-Place Editing, Signing, and PDF Export:** The radiologist reviews findings, makes edits, draws a digital signature, locks the finalized report, and downloads a publication-grade PDF.
6. **Graceful Auto-Shutdown:** When the browser tab is closed or the radiologist clicks "Shutdown System", browser Beacon signals trigger `/api/system/shutdown`, cleanly killing all backend and frontend background processes.



---

## 10. Data Models

The system communicates via a rigorously typed JSON schema.

### Example `TriageResult` Payload:
```json
{
  "slice_number": 42,
  "body_region": "chest",
  "scan_type": "contrast",
  "action": "IMMEDIATE ALERT",
  "emergency_score": 85,
  "processing_time_ms": 32.4,
  "statistics": {
    "mean": 24.5,
    "std": 145.2,
    "median": 40.0,
    "iqr": 60.5,
    "mad": 25.1
  },
  "findings": [
    {
      "anomaly_type": "Statistical Hyperdense",
      "bbox": [150, 200, 45, 30],
      "centroid": [172.5, 215.0],
      "area": 1250,
      "mean_hu": 210.5,
      "confidence": 0.85,
      "severity_score": 0.9
    },
    {
      "anomaly_type": "Extreme Air",
      "bbox": [300, 180, 80, 90],
      "centroid": [340.0, 225.0],
      "area": 6400,
      "mean_hu": -980.5,
      "confidence": 0.95,
      "severity_score": 0.98
    }
  ]
}
```

---

## 11. Configuration & Model Selection

PRISM environment configuration is managed via the root `.env` file (created automatically by the startup scripts if absent):

```bash
# Phase 1: DICOM Ingestion & Real-Time Alerts
DICOM_PORT=11112                              # Port for DICOM C-STORE SCP listener
WS_PORT=8001                                  # Port for WebSocket alert broadcasts

# Phase 2 -> Phase 3 Handoff Bridge
PHASE3_ENABLED=true                           # Enables automated trigger of Phase 3 after Phase 2
PHASE3_URL=http://127.0.0.1:8005/api/v1/generate-report

# Phase 3: Reporting Service & LLM Configuration
PHASE3_PORT=8005                              # Port for FastAPI reporting service
OLLAMA_HOST=http://127.0.0.1:11434            # Ollama REST API server
OLLAMA_MODEL=llama3.2:3b                      # Recommended reporting model (2.0 GB)
OLLAMA_TEMPERATURE=0.2                        # Low temperature for clinical determinism
OLLAMA_TIMEOUT=120                            # HTTP timeout in seconds
OLLAMA_AUTO_PULL=true                         # Auto-pulls model if missing
SKIP_LLM=false                                # Set true to bypass LLM and use template fallback directly
REPORTS_DIR=./storage/reports                 # JSON report persistence directory
```

### Phase 3 Report Architecture

PRISM Phase 3 generates reports using a two-tier pipeline:

1. **Deterministic Template Engine (`template_engine.py`):** Converts Phase 2 TotalSegmentator output directly into a structured clinical report with zero hallucination. All numerical values (HU, volumes, dimensions) come exclusively from Phase 2 `statistics.json`. Produces four mandatory sections:
   - **INDICATION:** Derived from patient demographics, modality, body region
   - **TECHNIQUE:** Modality, contrast phase (inferred from organ HU values), multiplanar reconstruction
   - **FINDINGS:** Every organ TotalSegmentator segmented, grouped by body system. Normal organs appear as `Unremarkable. Mean density X HU. Volume Y cc.`; anomalous organs list quantitative anomaly detail.
   - **IMPRESSION:** Auto-generated bullet summary of anomalous organs; `"No acute or significant findings"` for normal studies.

2. **LLM Grammar Refinement (`llm_service.py`):** `llama3.2:3b` polishes prose fluency without altering any factual content. A strict system prompt forbids altering numbers, section headers, or organ entries. Validator rejects and falls back to template output if any number is altered.

| Model | Size | Hardware Fit | Inference Time | Recommendation |
|---|---|---|---|---|
| **`llama3.2:3b`** | **2.0 GB** | **100% in 4GB GPU VRAM** | **2–3 seconds** | **Standard (Recommended)** — High fluency, zero OOM, instant. |
| **`qwen2.5:3b`** | 1.9 GB | Fits in 4GB GPU VRAM | 2–3 seconds | Excellent alternative. |
| **`meditron:7b`** | 4.1 GB | Exceeds 4GB VRAM (spills to RAM) | 15–20 seconds | Slower, unnecessary given deterministic template base. |
| **`llama3.2-vision`** | 7.8 GB | Requires >8GB VRAM | Fails (`mllama`) | Incompatible with text chat endpoints. |

---

## 12. Developer Guide

We welcome contributions to the open-source codebase.

* **Adding a new Preprocessing Step:** Insert your function into `triage_screen.py` before Step 4. Ensure operations are vectorized; avoid `for` loops over pixels.
* **Adding a new Statistic:** Update `_step_4_adaptive_baseline`. Use subsampled arrays (e.g., `pixels[::4]`) for computationally heavy operations like percentiles.
* **Modifying the Emergency Score:** Adjust `_step_9_10_score_and_action`. If you introduce a new finding type (e.g., "Metallic Artifact"), ensure it does not artificially inflate the severity multiplier.
* **Adding Frontend Overlays:** Modify the React `OverlayEngine` component. Findings with a specific `anomaly_type` can be mapped to new SVG stroke colors or icons.
* **Writing Tests:** Replay your test datasets using `replay_sender.py`. Monitor the terminal logs to ensure your logic changes do not increase the average `processing_time_ms` above 50ms.
* **Adding Replay Datasets:** Place new DICOM folders into `sample_dicoms/`. The `replay_sender.py` will recursively find and stream them.

---

## 13. Phase 2 Development — Handoff Infrastructure

Phase 2 transforms PRISM from a slice-level triage engine into a volumetric analysis system. The development is organized into 8 work packages with strict dependency ordering. This section documents the completed packages.

### Package 0 — Schema Contracts (✅ Complete)

Before any implementation code, three JSON schemas and one shared Python dataclass were frozen to prevent integration mismatches:

| Schema File | Purpose | Location |
|---|---|---|
| `phase1_handoff.json` | Defines the complete handoff payload from Phase 1 to Phase 2 | `schemas/` |
| `organ_statistics.json` | Per-organ statistical baseline (one entry per TotalSegmentator label) | `schemas/` |
| `findings_output.json` | Final findings handed to Phase 3 (one entry per surviving candidate) | `schemas/` |
| `candidate.py` | Shared `Candidate`, `ShapeFeatures`, `DensityHU` dataclasses | `schemas/` |

The `Candidate` dataclass includes lifecycle-annotated fields (`[CREATION]`, `[MERGE]`, `[FILTER]`, `[SCORE]`) ensuring every downstream package knows exactly which fields it's responsible for populating. Optional fields like `texture_features` default to `{}` when `--radiomics` is disabled.

### Package 1 — Phase 1 → Phase 2 Handoff Infrastructure (✅ Complete)

**Objective:** Make Phase 1 retain what it currently discards, and emit a clean, schema-conformant handoff when a series is complete — without slowing the existing real-time alert path.

#### What Was Built

**1. Extended DICOM Tag Extraction** (`dicom_listener.py` — modified)

7 new DICOM tags are now captured alongside the existing fields:

| Tag | DICOM ID | Purpose |
|---|---|---|
| `SeriesInstanceUID` | (0020,000E) | Unique series identifier for volume grouping |
| `StudyInstanceUID` | (0020,000D) | Study-level identifier |
| `PixelSpacing` | (0028,0030) | In-plane pixel dimensions [row_mm, col_mm] |
| `SliceThickness` | (0018,0050) | Nominal slice thickness |
| `ImagePositionPatient` | (0020,0032) | Full [x, y, z] position vector for spatial ordering |
| `ImageOrientationPatient` | (0020,0037) | 6 direction cosines for orientation |
| `Modality` | (0008,0060) | CT vs MR routing |

All tags include safe fallback defaults when absent from the DICOM dataset.

**2. Volume Accumulator** (`volume_accumulator.py` — new)

Thread-safe accumulator that collects HU-transformed 2D slices alongside per-slice findings. Key design decisions:

- **Spatial ordering by ImagePositionPatient Z** — never by InstanceNumber. These routinely disagree in clinical DICOM data, and using InstanceNumber produces subtly misaligned volumes.
- **Z-spacing computed from actual slice positions** (median of inter-slice distances), not from the `SliceThickness` tag which is nominal and often wrong.
- **Two readiness paths:** association-release (`mark_association_released()`) and stream-stall fallback (`STREAM_STALL_TIMEOUT = 15.0s`), both subject to the minimum-slice sanity gate (`min_slices = 20`) and one-shot trigger guard (`_triggered`).
- **Slice-count floor** — Minimum viable volume size (default: 20 slices). This acts purely as a mandatory sanity gate to prevent segmenting near-empty sequences and is *never* a trigger by itself.
- **One-shot state guard** — Guarantees Phase 2 is triggered at most once per series execution via the `_triggered` state flag.
- **Lost Packet Drop** — `SliceBuffer` will forcefully drop a delayed slice and unblock the sequence if the slice fails to arrive within a short gap-timeout while later slices actively arrive.
- **Non-blocking** — `add()` completes in < 5ms for a 512×512 slice (measured), consuming < 10% of the real-time path's 50ms budget.

```python
class VolumeAccumulator:
    def __init__(self, min_slices=20): ...
    def add(self, instance_number, hu_array, findings, spacing_meta): ...
    def mark_association_released(self) -> None: ...
    def mark_stream_stalled(self) -> None: ...
    def is_ready(self) -> bool: ...
    def export_volume(self) -> tuple[np.ndarray, list[list], list[int], tuple]: ...
    def reset(self) -> None: ...
```

**3. Finding Tracker** (`finding_tracker.py` — new)

Links per-slice 2D findings into multi-slice "tracks" for 3D persistence validation:

- **Centroid-based linking:** Greedy nearest-neighbor matching across consecutive slices with configurable drift threshold (default: 15px)
- **Persistence filter:** Tracks spanning ≥ `min_slices` (default: 3) consecutive slices are considered spatially persistent
- **Zero DICOM dependency:** Operates purely on lists of Finding objects, fully testable with synthetic data

```python
def link_findings_across_slices(ordered_findings, max_centroid_drift_px=15) -> list[list]: ...
def track_persistence_ok(track, min_slices=3) -> bool: ...
```

**4. Pipeline Integration** (`pipeline.py` — modified)

- Accumulator fed HU-transformed arrays (not raw pixels) and triage findings after each slice processes
- Two readiness paths (association-release and stream-stall fallback) wired into the pipeline flush loop
- Phase 2 spawn uses `multiprocessing.Process` (not `threading.Thread`) for clean C++ tensor memory teardown
- Real-time WebSocket alert path completely unchanged

#### Multiple Trigger Bug & Production Fix

During initial integration testing, a critical trigger bug was identified where Phase 2 re-triggered continuously on every incoming slice once the slice count exceeded `min_slices`.

* **The Problems Identified**:
  1. **Unbounded Re-firing**: `min_slices` previously acted as an automatic trigger, causing `is_ready()` to evaluate `True` repeatedly for every single slice past the 20-slice threshold.
  2. **Connection-Release Race Condition**: `EVT_RELEASED` (association close) fired Phase 2 before the background worker thread finished processing buffered/queued slices, causing premature export of partial volumes.
  3. **Concurrent Stream Interference**: Active secondary DICOM connections could trigger Phase 2 prematurely before all associations completed.

* **The Production Fixes Implemented**:
  1. **One-Shot State Guard (`_triggered` flag)**: Added an explicit `_triggered` state variable in `VolumeAccumulator`. Once `export_volume()` executes, `is_ready()` strictly returns `False` until `reset()` is called, guaranteeing a maximum of one Phase 2 trigger per series.
  2. **`min_slices` as Sanity Floor Gate ONLY**: `min_slices` (default: 20) was refactored into a strict gate condition. It prevents segmenting near-empty sequences, but `min_slices` alone **never** triggers Phase 2.
  3. **Strict 5-Step Pipeline Sequencing (`on_association_released`)**: When a DICOM association closes, `pipeline.py` enforces a deterministic sequence:
     - Check `listener.has_active_association()` (defer if active peers remain)
     - Drain worker thread queue via `self._queue.join()`
     - Flush remaining buffered slices via `self.buffer.flush_all()` and process through 2D triage
     - Signal `accumulator.mark_association_released()`
     - Execute `self._check_phase2_trigger()` exactly once

#### Acceptance Checklist Results

| Criterion | Status | Evidence |
|---|---|---|
| All 7 new DICOM tags captured | ✅ Pass | `TestDicomTagExtraction` (4 tests) |
| VolumeAccumulator orders by ImagePositionPatient Z | ✅ Pass | `test_ordering_disagrees_with_instance_number` — deliberately shuffled Z vs InstanceNumber |
| Association-release trigger fires independently | ✅ Pass | `test_association_release_trigger` |
| Slice-count floor acts as mandatory sanity gate | ✅ Pass | `test_slice_floor_gate` — verified min_slices alone does not trigger |
| Timeout fallback fires independently | ✅ Pass | `test_timeout_trigger` |
| Finding tracker: smooth drift → 1 track, persistent | ✅ Pass | `test_smooth_drift_single_track` |
| Finding tracker: large jump → separate tracks, not persistent | ✅ Pass | `test_large_jump_separate_tracks` |
| Accumulator.add() < 5ms per 512×512 slice | ✅ Pass | `test_accumulator_add_is_fast` |
| Thread safety (concurrent adds) | ✅ Pass | `test_accumulator_thread_safe` |
| No existing test regressions | ✅ Pass | 136/136 tests passing cleanly |

#### Test Results

```
25 passed in 0.76s  (Package 1 tests)
136 passed, 2 skipped in 23.64s  (Full test suite — zero regressions)
```

### Package 2 — Volume Assembly & Segmentation Runner (✅ Complete)

**Objective:** Convert the accumulated stack of HU slices into a correctly oriented, correctly spaced 3D NIfTI volume, run TotalSegmentator on it inside a properly isolated subprocess, parse the output metrics, and guarantee that memory is fully released.

#### Package 2 Output Structure & Intermediate Artifacts

Each Phase 2 execution isolates outputs by creating a separate `run_<timestamp>/` directory under `phase2_work/`. Below is the distinction between Package 2 artifacts and downstream Package 3 outputs:

* **`run_<timestamp>/`** — Isolates each Phase 2 execution so outputs from concurrent or consecutive runs do not overwrite each other.
* **`temp_seg/`** — Consolidated output folder containing all run artifacts:
  * **Package 2 Direct Artifacts (Stages 1 & 2)**:
    * **`temp_volume.nii.gz`** — Assembled 3D NIfTI volume created from accumulated DICOM HU slices (`volume_builder.py`).
    * **`temp_seg.nii`** — 3D multi-label organ segmentation mask produced by TotalSegmentator (`segment_runner.py`).
    * **Raw `statistics.json` (Intermediate Artifact)** — TotalSegmentator is executed with `--statistics`, initially writing a raw `statistics.json` file containing un-trimmed mean HU and voxel counts. Package 2 (`segment_runner.py`) reads, parses, and normalizes this raw file in memory into `ts_stats`.

##### Stage-by-Stage Statistics Lifecycle

```text
Package 2 (Stage 2 execution):
    temp_volume.nii.gz
          ↓
    TotalSegmentator (--fast --statistics --ml)
          ↓
    temp_seg.nii  +  raw statistics.json (intermediate artifact)
          ↓
    segment_runner.py parses & normalizes raw statistics in memory

Package 3 (Stage 3 execution):
    temp_seg.nii  +  original 3D HU volume
          ↓
    organ_baseline.py (percentile-trimmed baselines & physical volume in cc)
          ↓
    final temp_seg/statistics.json (replaces/overwrites raw file on disk)
```

> [!NOTE]
> **Single Final Statistics File on Disk:**
> TotalSegmentator's raw `statistics.json` is a legitimate intermediate output of Stage 2 parsed in memory by Package 2. During Stage 3, Package 3 independently calculates patient-specific baselines directly from `temp_seg.nii` and the original 3D HU volume array using `organ_baseline.py`, writing the final `temp_seg/statistics.json` to disk and removing the loose intermediate Stage 2 file during consolidation. There are **not** two final statistics files on disk.

#### Package 2 Setup & Installation

To run Package 2, a developer must install the required dependencies and verify the `TotalSegmentator` environment:

1. **Install Dependencies:**
   From the repository root, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   This installs the medical imaging and support packages: `SimpleITK` (for NIfTI volume building), `psutil` (for CPU/RAM resource monitoring), and `TotalSegmentator` (the segmentation subprocess executable).

2. **TotalSegmentator & Model Setup:**
   - **Required Model Weights:** TotalSegmentator requires the pretrained model weights for the selected task (e.g., `total` for CT, `total_mr` for MR). The required weights must be available through the TotalSegmentator setup before running Package 2.
   - **Verification Command:** Verify that the executable is successfully installed and available on your system path by running:
     ```bash
     TotalSegmentator --help
     ```

3. **Integrated Pipeline Execution:**
   The Phase 1 ingestion pipeline (`phase1_ingestion/pipeline.py`) triggers Package 2 and spawns a `multiprocessing.Process` running the Package 2 lifecycle manager (`phase2_segmentation/lifecycle_manager.py`) for NIfTI assembly and segmentation.
   To run the entire pipeline end-to-end:
   - Start the pipeline listener daemon:
     ```bash
     python -m phase1_ingestion.pipeline
     ```
   - In a separate terminal, stream a series of sample slices using the replay sender:
     ```bash
     python phase1_ingestion/replay_sender.py
     ```

#### What Was Built

**1. NIfTI Volume Assembly** ([`volume_builder.py`](phase2_segmentation/volume_builder.py) — new)

- **SimpleITK Spacing Translation** — Correctly translates and swaps the accumulator's `(row_mm, col_mm, z_mm)` order to SimpleITK's internal `(X, Y, Z)` coordinate mapping `(col_mm, row_mm, z_mm)` to avoid metric distortion.
- **Physical Geometry Mapping** — Employs direction cosines derived from patient orientation matrices to resolve the 3D volume grid geometry.
- **Verification Boundaries** — Enforces minimum resolution floors, thickness thresholds, and non-empty voxel metrics.

```python
def assemble_nifti(volume_array: np.ndarray, spacing_meta: tuple, out_path: str = "temp_volume.nii.gz", orientation_patient: Optional[list] = None) -> str: ...
```

**2. TotalSegmentator Subprocess Isolation & Dynamic Hardware Timeout Wrapper** ([`segment_runner.py`](phase2_segmentation/segment_runner.py) — new)

- **Subprocess Isolation** — Invokes `TotalSegmentator` as a separate command-line subprocess to guarantee C++-backed tensor memory (ONNX/PyTorch) is freed upon process exit.
- **Hardware-Adaptive Dynamic Timeout (`compute_dynamic_timeout`)** — Dynamically scales subprocess execution limits based on real-time hardware detection:
  - **GPU Acceleration**: Detects CUDA (PyTorch/`nvidia-smi`), Apple Silicon MPS, or ROCm $\rightarrow$ base timeout 120s.
  - **CPU Fallback**: Base timeout 600s.
  - **RAM-Scaled Multiplier**: Queries available system memory via `psutil` (<4 GB $\rightarrow$ 4.0×, <8 GB $\rightarrow$ 3.0×, <16 GB $\rightarrow$ 2.0×, <32 GB $\rightarrow$ 1.5×, $\ge$32 GB $\rightarrow$ 1.0×).
  - **Volume Size Bonus**: Adds +45s to +180s for NIfTI volumes > 50 MB.
  - **Clamped Safety Bounds**: Hard floor at 60s, ceiling at 1800s (30 mins) to prevent infinite hangs on underpowered machines.
- **Modality-Driven Routing** — Dynamically adjusts segmentation pipeline parameters based on modality (`total` for CT, `total_mr` for MR scans).
- **Schema Normalization** — Transforms TotalSegmentator v2.17+ simplified output keys (`intensity`, `volume`) to match the quantitative parameters defined in `organ_statistics.json`.

```python
def compute_dynamic_timeout(nifti_path: Optional[str] = None) -> int: ...
def run_totalsegmentator(nifti_path: str, out_dir: str = DEFAULT_OUT_DIR, modality: str = "CT", timeout: Optional[int] = None) -> tuple[str, dict]: ...
```


**3. Isolated Process Lifecycle and Safety Manager** ([`lifecycle_manager.py`](phase2_segmentation/lifecycle_manager.py) — new)

- **Resource Watchdog** — Monitors active process RSS memory allocations via a background polling thread (`RAMWatchdog`), logging and asserting that peak memory usage remains within the 3.0 GB budget.
- **Temporary File Retention for Inspection** — Retains `temp_volume.nii.gz`, `temp_seg.nii`, and `statistics.json` in the execution run directory on disk for developer validation, manual inspection, and testing.

```python
@dataclass
class Phase2Result:
    success: bool = False
    nifti_path: str = ""
    segmentation_dir: str = ""
    organ_stats: dict = field(default_factory=dict)
    peak_ram_gb: float = 0.0
    total_time_sec: float = 0.0
    volume_shape: tuple = ()
    num_organs_detected: int = 0
    region: str = "unknown"

class Phase2LifecycleManager:
    def __init__(self, work_dir: str = DEFAULT_WORK_DIR, ram_budget_gb: float = 3.0, cleanup_on_success: bool = False, cleanup_on_failure: bool = False): ...
    def run(self, volume: np.ndarray, spacing: Tuple[float, float, float], series_meta: Optional[Dict] = None) -> Phase2Result: ...

def run_phase2(volume: np.ndarray, findings_per_slice: list, instance_numbers: list, spacing: Tuple[float, float, float], series_meta: Optional[Dict[str, str]] = None) -> Phase2Result: ...
```

#### Acceptance Checklist Results

| Criterion | Status | Evidence |
|---|---|---|
| SimpleITK Spacing Swap Verification | ✅ Pass | `test_volume_to_nifti_pipeline` |
| TotalSegmentator Task and Parameter Discovery | ✅ Pass | `test_build_ct_command`, `test_build_mr_command` |
| Output Schema Key Normalization | ✅ Pass | `test_normalize_organ_entry_complete`, `test_parse_statistics_dict_format` |
| Execution Timeout Guards | ✅ Pass | `test_subprocess_timeout` |
| RSS RAM Watchdog Alerts | ✅ Pass | `test_ram_watchdog_budget_exceeded` |
| Temporary File Handling | ✅ Pass | `test_cleanup_on_success`, `test_cleanup_on_failure` |
| Thread-Safe Parallel Executions | ✅ Pass | `test_concurrent_safety` |
| End-to-End Pipeline Validation | ✅ Pass | `test_full_lifecycle_with_test_data` |

#### Test Results

```
65 passed, 2 skipped in 24.63s  (Package 2 tests)
136 passed, 2 skipped in 23.64s  (Full test suite — zero regressions)
```

#### Package 2 — Radiomics Handoff Note

* **Original Plan Requirement**: The original Package 2 implementation plan specifies running TotalSegmentator with flags: `--fast --statistics --radiomics --ml` to generate both `statistics.json` and `statistics_radiomics.json`.
* **Current Execution State**: The current implementation intentionally runs TotalSegmentator with flags: `--fast --statistics --ml` (omitting `--radiomics`).
* **Environment Context**: The project environment currently uses **Python 3.12.6**, where PyRadiomics is not currently installed/compatible due to legacy CPython C-extension build constraints under Python 3.12+.
* **Current Generated Outputs**: Package 2 execution produces `temp_seg.nii` and TotalSegmentator's basic `statistics.json`, but does not currently produce `statistics_radiomics.json`.
* **Package 3 Status (Unaffected)**: Package 3 is **NOT** affected by this pending radiomics item. Package 3 calculates its own final patient-specific organ statistics from the TotalSegmentator segmentation mask (`temp_seg.nii`) and the original 3D HU volume array, writing the final `temp_seg/statistics.json` and `temp_seg/region.txt`.
* **Handoff Instructions for Future Developers (Package 4+)**:
  1. Establish a Python/PyRadiomics/TotalSegmentator environment compatible with the original Package 2 requirement.
  2. Verify that `--radiomics` works with TotalSegmentator.
  3. Verify that `statistics_radiomics.json` is generated alongside segmentation outputs.
  4. Complete the Package 2 radiomics integration so that `texture_features` can be populated for later packages.
* **Note**: This is a known pending handoff item/deviation from the original Package 2 plan, **NOT** a failure of the current Package 3 patient-specific statistics implementation.

### Package 3 — Organ Baseline & Region Detection (✅ Complete)

**Objective:** Compute every organ's patient-specific statistical baseline (never a population lookup table) from the patient's own voxel HU intensity array, and determine the broad body region (e.g. neuro, KUB, cardiothoracic, etc.) covered by the scan based on the set of organ labels identified by TotalSegmentator.

#### What Was Built

**1. Patient-Adaptive Organ Baseline Calculator** ([`organ_baseline.py`](phase2_segmentation/organ_baseline.py) — new)

- **Voxel-Level Percentile Trimming** — Isolates voxel intensities inside each organ's segmentation mask and sorts them to find P5 and P95 percentiles.
- **Robust Statistical Summarization** — Computes trimmed mean and trimmed standard deviation on the `[P5, P95]` subset to prevent anomaly inflation. Computes median, median absolute deviation (MAD), first quartile (Q1), third quartile (Q3), and interquartile range (IQR) on the full array.
- **Physical Volume Tracking** — Maps the voxel count and scales it by the calculated physical voxel volume (cc) to yield precise per-organ volumes.
- **No Population Hardcoding** — Functions purely on patient data without relying on population HU lookup tables.

```python
def compute_organ_baseline(organ_voxels_hu: np.ndarray, voxel_volume_cc: float = 0.0) -> dict: ...
```

**2. Scan Body Region Classifier** ([`region_detector.py`](phase2_segmentation/region_detector.py) — new)

- **Label Mapping** — Maps TotalSegmentator's class labels (including wildcards like `vertebrae_*` and `rib_*`) to structural regions: `neuro`, `cardiothoracic`, `abdomen_pelvis`, `spine`, `ortho`.
- **KUB Clinical Override** — Detects focused renal scans (`kub`) if kidneys and urinary bladder are present, but liver and spleen are absent, bypassing generic voting blocks.
- **Majority-Vote Resolution** — Applies a tally-weighted majority vote across detected structure categories to determine the dominant scan region, logging mixed scan distributions.

```python
def detect_region(present_labels: list[str]) -> str: ...
```

**3. Integration & Flow Execution** ([`lifecycle_manager.py`](phase2_segmentation/lifecycle_manager.py) — modified)

- Seamlessly wires the baseline calculator and region classifier into Stage 3 of the Phase 2 lifecycle process.
- Extracts unique label values from the generated `temp_seg.nii` mask and maps them to Hounsfield Unit intensities in the source volume to compute patient-specific statistics.
- Writes patient-specific baseline statistics to `statistics.json` and body region classification to `region.txt` inside `phase2_work/run_<timestamp>/temp_seg/`.
- Returns the resolved region classification and statistical baseline dictionary in `Phase2Result`.

#### Package 3 Generated Artifacts

Package 3 generates two primary clinical metadata artifacts inside `phase2_work/run_<timestamp>/temp_seg/`:

* **`statistics.json`** — Rich patient-adaptive organ statistics (P5–P95 percentile-trimmed mean/std, median, MAD, Q1, Q3, IQR, and physical volume in cc) computed by `organ_baseline.py` overlaying the HU volume on `temp_seg.nii` (replaces raw TS stats).
* **`region.txt`** — Classified scan body region (`neuro`, `cardiothoracic`, `abdomen_pelvis`, `spine`, `ortho`, `kub`) generated by `region_detector.py`.

#### Acceptance Checklist Results

| Criterion | Status | Evidence |
|---|---|---|
| Hand-built Region Fixture Classifications | ✅ Pass | `test_pure_chest_classification`, `test_pure_abdomen_classification`, `test_head_classification` |
| Mixed-Region Handling and Decision Logging | ✅ Pass | `test_mixed_classification` |
| KUB Clinical Override Rules | ✅ Pass | `test_kub_clinical_override` |
| Spine and Ortho Wildcard Label Mapping | ✅ Pass | `test_spine_wildcards`, `test_ortho_wildcards` |
| Baseline Math Verification vs. Ground Truth | ✅ Pass | `test_baseline_math_correctness` |
| Tiny Voxel Array Exception Resilience | ✅ Pass | `test_tiny_baseline`, `test_empty_baseline` |
| Integrated End-to-End Pipeline Verification | ✅ Pass | `test_successful_pipeline_with_baselines` |

#### Test Results

```
11 passed in 0.23s  (Package 3 unit tests)
136 passed, 2 skipped in 23.64s  (Full test suite — zero regressions)
```

---

### Package 5 — Path B: Independent Organ-Wide Sweep (✅ Complete)

**Objective:** Independently scan every TotalSegmentator-segmented organ for statistical outlier regions using the patient-adaptive baselines from Package 3. Path B produces `Candidate` objects that are entirely agnostic of Phase 1's 2D findings — it catches what Path A misses by looking at the full 3D organ volume rather than only the voxels under Phase 1 seeds.

#### Why Path B Exists

Path A (Package 4) is fast and leverages Phase 1's real-time evidence, but it is fundamentally limited to the 2D spatial footprint of Phase 1's detections. A lesion that Phase 1 missed — because it fell below the per-slice anomaly threshold, or sat in a region with high local variance — will never become a Path A candidate. Path B provides an independent, exhaustive second opinion: for every organ TotalSegmentator labelled, it computes connected components of voxels whose HU values deviate from the patient's own organ baseline and promotes any large, persistent component to a `Candidate`. When both paths find the same 3D region, Package 6 marks it `corroborated=True`, producing a strong double-confirmed signal for the clinical gate.

#### What Was Built

**1. Organ-Wide Statistical Sweep** ([`organ_sweep.py`](phase2_segmentation/organ_sweep.py) — new)

- **Patient-Adaptive Thresholds, Not Population Norms** — Outlier detection uses the organ's own `trimmed_mean` and `trimmed_std` from Package 3's baseline, not a fixed HU window. A liver at 80 HU in this patient is normal; a cluster at 140 HU in the same liver is flagged. This prevents both false positives in naturally dense organs and false negatives in naturally hypodense ones.
- **Connected-Component Voxel Counting, Not Bounding-Box Approximation** — `scipy.ndimage.label` identifies connected outlier regions at the true voxel level. `volume_cc` is computed from the actual component mask (not the full bbox rectangle), making it strictly more accurate than Path A's whole-bbox estimate.
- **3D Persistence, Not 2D Slice Count** — Persistence is measured by the number of unique Z-slices the component spans (`len(np.unique(z_idx)) >= min_persistent_slices`), not by a cross-slice tracking chain. This is the correct 3D definition for an organ-derived component.
- **Organ Context at Birth** — Unlike Path A (which always starts with `organ_label="unclassified"`), every Path B candidate is born with `organ_label`, `organ_local_zscore`, and `organ_overlap_fraction` already populated — no backfill step needed for Path B survivors.
- **Uniform Coordinate Convention** — `bbox_3d = [x_min, y_min, z_min, x_max, y_max, z_max]` and `centroid_3d = [X, Y, Z]` — identical to Path A's convention, confirmed by cross-checking both implementations before Package 6's merge step was designed.

```python
def sweep_organs(
    volume: np.ndarray,            # (Z, Y, X) HU volume
    seg_arr: np.ndarray,           # (Z, Y, X) TotalSegmentator label mask
    value_to_label: dict,          # {int → str} from class_map
    organ_stats: dict,             # {str → baseline} from organ_baseline.py
    spacing: tuple,                # (row_mm, col_mm, z_mm)
    outlier_sigma: float = 2.5,
    min_component_voxels: int = 10,
    min_persistent_slices: int = 3,
) -> list[Candidate]: ...
```

#### Acceptance Checklist Results

| Criterion | Status | Evidence |
|---|---|---|
| Hyperdense component is detected as a Candidate | ✅ Pass | `test_hyperdense_component_detected` |
| Homogeneous organ produces no candidates | ✅ Pass | `test_no_outliers_returns_empty` |
| Tiny components below size threshold are filtered | ✅ Pass | `test_tiny_component_filtered_out` |
| Single-slice component fails persistence check | ✅ Pass | `test_single_slice_component_not_persistent` |
| No regressions in previous packages | ✅ Pass | Full suite |

#### Test Results
4 passed in 0.09s  (Package 5 tests)
Full suite — zero regressions against Packages 1–4.

---

### Package 6 — Technical Suppression Cascade, Merge & 3D Re-validation (✅ Complete)

**Objective:** Unify the Path A and Path B candidate lists into a single fully-attributed list, resolve the Path A organ context gap, compute 3D shape descriptors (sphericity, margin curvature variance), and apply a rule-driven clinical suppression cascade — all before Package 7's confidence fusion. Package 6 does NOT compute `fused_confidence`, `gate`, or `emergency_score`; those are `[SCORE]` lifecycle fields owned exclusively by Package 7.

#### What Was Built

**1. 3D IoU-Based Corroboration Merge** ([`candidate_merger.py`](phase2_segmentation/candidate_merger.py) — new)

- **Greedy IoU Matching with Inclusive-Bounds Correction** — Merges Path A and Path B lists by 3D bounding-box IoU (threshold: 0.3, confirmed in design review). Uses inclusive integer voxel bounds throughout: `(x_max − x_min + 1)` not `(x_max − x_min)`. Without the `+1`, single-voxel-wide bounding boxes return volume=0 and IoU=0, silently breaking corroboration for the most common overlap case.
- **Path B Spatial Fields Win on Merge, Path A Provenance Always Preserved** — When two candidates are merged: `centroid_3d`, `bbox_3d`, `organ_label`, `organ_local_zscore`, `organ_overlap_fraction`, `shape`, and `density_hu` come from Path B (mask-derived, more accurate); `phase1_confidence`, `phase1_severity`, `phase1_anomaly_type`, and `candidate_id` come from Path A (unique provenance).
- **Organ Context Backfill for Path A Survivors** — Unmatched Path A candidates carry `organ_label="unclassified"` from `seed_clusterer.py`. The backfill step looks up each candidate's `bbox_3d` in `seg_arr`, takes the majority-vote (most frequent non-zero) TotalSegmentator label, and computes `organ_local_zscore` and `organ_overlap_fraction` using exactly the same formulas as `organ_sweep.py`. Note: the z-score uses `density_hu.mean` as a scalar point estimate — an acknowledged approximation because Path A candidates carry no component mask.
- **Shape Geometry from HU Approximation Mask** — `sphericity` and `margin_curvature_variance` (both `0.0` from P4 and P5) are computed from an HU-threshold-derived binary mask within each candidate's bbox. Surface area uses the mean of all three voxel face orientations: `(row_mm×col_mm + row_mm×z_mm + col_mm×z_mm) / 3` — critical for anisotropic CT spacing (e.g. thick-slice acquisitions where z_mm >> col_mm).

**2. Rule-Driven Clinical Suppression Filter**

Five rules evaluated in order (first match wins). Returns ALL candidates — never silently drops any. Every rejected candidate carries a machine-readable `suppression_reason` string for audit traceability.

| Rule key | Condition |
|---|---|
| `"not_persistent"` | `persistence_ok=False` AND not corroborated |
| `"lung_nodule_<6mm"` | Lung label, `long_axis_mm < 6.0`, `region="cardiothoracic"` |
| `"simple_renal_cyst"` | Renal label, `density_hu.mean < 20 HU`, `std < 15 HU`, `sphericity > 0.7` |
| `"population_common"` | `organ_local_zscore < 2.5` AND not corroborated |

Candidates with `organ_label="unclassified"` after backfill are **never auto-suppressed** — Package 7 routes them to `gate="MANUAL_REVIEW"`.

Corroborated candidates are exempt from `"not_persistent"` and `"population_common"`; organ-specific size/density rules still apply.

```python
def merge_candidates(
    path_a: list, path_b: list,
    seg_arr: np.ndarray,   # (Z, Y, X) TotalSegmentator label mask
    volume: np.ndarray,    # (Z, Y, X) HU volume
    value_to_label: dict,  # {int → str}
    organ_stats: dict,     # {str → baseline} from organ_baseline.py
    spacing: tuple,        # (row_mm, col_mm, z_mm)
    region: str,           # from region_detector.py
    iou_threshold: float = 0.3,
) -> list[Candidate]: ...

def clinical_filter(
    candidates: list,
    region: str,
    z_suppress_threshold: float = 2.5,  # keyword param — testable without monkeypatching
) -> list[Candidate]: ...
```

#### Pre-Implementation Audit

Before any code was written, a field-by-field audit verified every Candidate field read or written by Package 6 against the frozen schema, Package 4, and Package 5. Five high/medium findings were identified and resolved:
- **H1/H2:** `organ_label`, `organ_local_zscore`, `organ_overlap_fraction` absent for Path A → fixed by backfill step.
- **H3:** IoU formula used `(max-min)` instead of inclusive `(max-min+1)` → corrected.
- **H4:** `fused_confidence`, `gate`, `emergency_score` must not be touched by P6 → explicitly enforced.
- **M1/M2:** Shape geometry approximation and `volume_cc` semantic difference documented in module docstring.

#### Acceptance Checklist Results

| Criterion | Status | Evidence |
|---|---|---|
| Overlapping candidates corroborated with IoU ≥ 0.3 | ✅ Pass | `test_corroborated_merge_high_iou` |
| H3 fix: single-voxel bbox IoU == 1.0 | ✅ Pass | `test_iou_identical_single_voxel_boxes` |
| Path B spatial fields win on merge | ✅ Pass | `test_corroborated_merge_high_iou` |
| Path A phase1 provenance preserved | ✅ Pass | `test_corroborated_merge_high_iou` |
| Backfill assigns organ_label from seg_arr | ✅ Pass | `test_backfill_assigns_organ_label` |
| Backfill z-score matches P5 formula | ✅ Pass | `test_backfill_zscore_is_point_estimate` |
| Unclassified candidates NOT suppressed | ✅ Pass | `test_unclassified_candidate_never_suppressed` |
| Corroborated candidates exempt from population_common | ✅ Pass | `test_corroborated_exempt_from_population_common` |
| `fused_confidence`, `gate`, `emergency_score` untouched | ✅ Pass | `test_fused_confidence_untouched_full_pipeline` |
| All candidates returned — none silently dropped | ✅ Pass | `test_all_candidates_returned_none_dropped` |
| No regressions in Packages 1–5 | ✅ Pass | Full suite |

#### Test Results
34 passed in 1.04s  (Package 6 tests)
176 passed, 3 skipped in 25.41s  (Full test suite — zero regressions)
The 3 skips are pre-existing TotalSegmentator GPU integration tests unrelated to Package 6.

---

### Package 7 — Confidence Fusion, Scoring & Phase 3 Handoff (✅ Complete)

**Objective:** Compute final merged confidence scores, apply the clinical routing gate rules (e.g., auto-confirm vs. manual review), and package everything into the frozen `findings_output.json` contract for handoff to Phase 3.

#### What Was Built

**1. Scoring & Gating Engine** ([`scorer.py`](phase2_segmentation/scorer.py) — new)
- **Fused Confidence (`fused_confidence`)**: Blends `phase1_confidence` (if from Path A) and `organ_overlap_fraction` (if from Path B). Corroborated candidates receive a 10% bonus (capped at 1.0). For Path B-only candidates, it's strictly based on the overlap fraction. For Path A-only candidates, it's based on Phase 1's confidence.
- **Emergency Score (`emergency_score`)**: Reuses the validated non-linear Phase 1 formula: `severity * confidence^1.5 * sqrt(volume_cc/2.0) * 40`. Caps at 50 to prevent run-away scores.
- **Clinical Gating Rules (`gate`)**: Evaluates `suppressed` status, corroboration, confidence, and organ context to assign a clinical routing directive.
  - `"SUPPRESSED"`: If `suppressed == True` (from Package 6).
  - `"AUTO_CONFIRMED"`: If `corroborated == True` and `fused_confidence >= 0.85` and `emergency_score >= 35`.
  - `"MANUAL_REVIEW"`: All other non-suppressed candidates.

**2. Handoff Serialization** ([`phase2_api.py`](phase2_segmentation/phase2_api.py))
- Converts the final list of `Candidate` objects into a `StructuredFindings` payload.
- Passes both `organ_stats` (full TotalSegmentator `statistics.json` dict) and `region` (detected body region) into the payload, ensuring Phase 3 receives the complete anatomical picture, not just anomaly candidates.
- Writes `findings.json` to the work directory alongside `temp_seg.nii` and `statistics.json`.

**3. Comprehensive Organ Coverage** ([`findings_adapter.py`](phase2_segmentation/findings_adapter.py))
- Iterates `organ_stats` as the **primary organ list** (every organ TotalSegmentator detected).
- Normal organs (no candidates passing confidence/volume thresholds) appear in `StructuredFindings.organs` with `status="normal"`, whole-organ HU mean, and volume.
- Anomalous organs have their `status="anomaly_detected"` and full quantitative anomaly detail.
- Organs with candidates but not in `organ_stats` are also included as a safety net.
- This ensures Phase 3 always generates a complete organ-by-organ report, matching clinical radiology standards.

#### Acceptance Checklist Results

| Criterion | Status | Evidence |
|---|---|---|
| Corroborated high-confidence candidate auto-confirms | ✅ Pass | `test_fused_confidence_corroborated` |
| Suppressed candidate correctly gated | ✅ Pass | `test_scorer_handles_suppressed_candidates` |
| Serialization adheres to schema | ✅ Pass | `test_export_findings_creates_json` |
| Full lifecycle generates findings JSON | ✅ Pass | `test_full_phase2_pipeline_p7_outputs` |

---

### Package 8 — Testing, Calibration & Final Report (✅ Complete)

**Objective:** Verify threshold tolerances, edge case behaviors, and overall pipeline robustness across the entire Phase 2 progression. Update documentation to reflect final configurations.

#### What Was Built

**1. Calibration Tests** ([`test_package8_calibration.py`](phase2_segmentation/tests/test_package8_calibration.py) — new)
- Validated edge cases: single-voxel candidates, boundary IoU merges, non-corroborated high-score candidates, etc.
- Tuned `z_suppress_threshold` and `iou_threshold` limits via robust test suites without changing functional code unexpectedly.

**2. Full System Integration Tests** ([`test_integration.py`](phase2_segmentation/tests/test_integration.py) — modified)
- Mocked out heavy PyTorch/SimpleITK dependencies to guarantee tests run reliably in environments lacking native C++ extensions or GPU compute.
- Proven end-to-end traversal of Stages 1-5 without regression.

#### Test Results
All 12 unit tests for Packages 7 & 8 passed cleanly.
Integration tests successfully validate complete pipeline traversal.
3 skips remain for pre-existing GPU integration tests.

---

## 14. Limitations

* **Lack of 3D Context:** Slices are evaluated in isolation to ensure zero latency. A thin blood vessel curving into the Z-axis may temporarily appear as an isolated dense circle. 
* **Metallic Artifacts:** Dental amalgams, hip replacements, or pacemakers cause massive beam-hardening streaks. While Step 7 Geometric Filtering attempts to suppress thin streaks, massive scatter can distort the local Mean and Std Dev, potentially blinding the statistical detector in that region.
* **No Stateful Tracking:** The engine evaluates Slice 42 without knowing the outcome of Slice 41. It relies on the frontend or Phase 2 to aggregate slice-level findings into a cohesive volume.

---

## 15. Repository Structure

```text
MAJOR-PROJECT-PRISM/
├── orchestrator.py                # Autonomous End-to-End Orchestrator (Phase 1 + 2 + 3)
├── install_prism.ps1              # Windows One-Click Installer (CUDA / GPU auto-detection)
├── start_prism.ps1                # Windows One-Command Startup (Ollama preflight + Browser launch)
├── install_prism.sh               # Linux & macOS Installer (CUDA / MPS / CPU fallback)
├── start_prism.sh                 # Linux & macOS One-Command Startup (Health check + Process tracking)
├── requirements.txt               # Unified Python dependencies (including psutil)
├── .prism_pids.json               # Runtime process tracking for clean automatic shutdown
│
├── phase1_ingestion/              # Phase 1: Real-Time CT Ingestion & 2D Screening
│   ├── dicom_listener.py          # DICOM C-STORE SCP Server (+ Phase 2 tag extraction)
│   ├── slice_buffer.py            # Priority Reordering Queue (auto-resetting sequence pointer)
│   ├── hu_transform.py            # Hounsfield Unit Math & Padding Removal
│   ├── triage_screen.py           # The 10-Step Statistical Engine
│   ├── pipeline.py                # Main Orchestrator (+ async assoc-release-worker)
│   ├── volume_accumulator.py      # Phase 2 Volume Accumulator (RLock reentrant thread safety)
│   ├── dicom_archive.py           # Patient Archive Manager & Automated Zipping Engine
│   ├── finding_tracker.py         # Cross-Slice Finding Tracker
│   ├── ws_server.py               # WebSocket Telemetry Server & Scanner Lifecycle Manager
│   ├── ct_machine_emulator.py     # Replay Utility Core & Patient Metadata Injection
│   └── replay_sender.py           # CLI for Replay Utility
│
├── phase2_segmentation/           # Phase 2: Multi-Organ TotalSegmentator & 3D AI
│   ├── volume_builder.py          # NIfTI Volume Assembly
│   ├── segment_runner.py          # TotalSegmentator Subprocess Wrapper + compute_dynamic_timeout()
│   ├── organ_baseline.py          # Patient-Adaptive Organ Baseline Calculator
│   ├── region_detector.py         # Scan Body Region Classifier & KUB Override
│   ├── seed_clusterer.py          # Path A: 3D Clustering of Phase 1 Seeds
│   ├── organ_sweep.py             # Path B: Independent Organ-Wide Sweep
│   ├── candidate_merger.py        # 3D IoU Merge, Backfill, Shape Geometry & Clinical Filter
│   ├── scorer.py                  # Fused Confidence, Emergency Scoring & Clinical Gating
│   ├── findings_adapter.py        # StructuredFindings Adapter & Series Instance UID bridge
│   ├── phase2_api.py              # Phase 3 HTTP Handoff Trigger
│   └── lifecycle_manager.py       # Subprocess & RAM Watchdog Orchestrator
│
├── phase3_reporting/              # Phase 3: Zero-Hallucination Clinical Report Generation
│   ├── app/                       # FastAPI Application Core & Config
│   ├── api/                       # REST Endpoints (/generate-report, /report/{id}, /sign)
│   ├── services/
│   │   ├── template_engine.py     # Deterministic Clinical Template Engine
│   │   ├── llm_service.py         # Ollama LLM Service (llama3.2:3b grammar refinement)
│   │   └── validator.py           # Output Validation & Medical Hallucination Guard
│   ├── storage/
│   │   └── json_storage.py        # JSON Report Storage Engine (Microsecond Audit Trail & Patient Lookup)
│   └── scripts/
│       └── ollama_setup.py        # Ollama Preflight & Model Setup Helper
│
├── schemas/                       # Frozen Cross-Phase Interface Contracts
│   ├── phase1_handoff.json        # Phase 1 → Phase 2 handoff schema
│   ├── organ_statistics.json      # Per-organ baseline schema
│   ├── findings_output.json       # Phase 3 findings output schema
│   └── candidate.py               # Shared Candidate/ShapeFeatures/DensityHU dataclasses
│
├── prism-react/                   # Modern React Workstation Frontend
│   ├── src/
│   │   ├── components/            # UI Components (Badges, Buttons, Modals, Toasts)
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx      # Landing Dashboard & Instant "New Report" Routing
│   │   │   ├── LiveViewer.jsx     # Phase 1 Live CT Viewport + 3-State Session Machine
│   │   │   ├── ReportDetail.tsx   # Draft Report Editor, Digital Signatures & PDF Export
│   │   │   ├── Generate.tsx       # Manual JSON Report Synthesis
│   │   │   └── ReportsList.tsx    # Archive & Patient Study Index
│   │   ├── styles/                # viewer.css (Responsive Shell & State Animations), globals.css
│   │   ├── api/                   # prismApi.ts (Typed Client)
│   │   └── utils/                 # pdfGenerator.ts
│   ├── package.json
│   └── vite.config.ts
│
├── sample_dicoms/                 # Sample CT Datasets for Simulation
└── README.md                      # Complete Project Documentation
```

## 16. License / References

This project is developed as an open-source medical imaging research initiative. 

* **License:** MIT License. See `LICENSE` for details.
* **DICOM Standard:** [NEMA DICOM PS3](https://www.dicomstandard.org/)
* **TotalSegmentator:** [Wasserthal et al. (2023)](https://github.com/wasserth/TotalSegmentator)
* **Hounsfield Unit Mathematics:** Radiographic attenuation standardization algorithms.

---

## 17. End-to-End Diagnostic Workflow & Architecture Updates

Recent updates introduce a unified, responsive clinical workstation integrating all 3 diagnostic phases with strict session safety, multi-patient reusability without backend restart, deterministic report synthesis, and instant archival.

### Key Architectural Updates

```
+---------------------------------------------------------------------------------------------+
|                                    PRISM Fixed Header                                       |
|  [ Patient Metadata Bar: Directory | ID* | Name* | Age* | Sex* | Speed ] [ Shutdown System ]|
|  [ Stepper: 1: Phase 1 (Ingestion)  -->  2: Phase 2 (TotalSegmentator)  -->  3: Phase 3 (Studio) ] |
+---------------------------------------------------------------------------------------------+
                                               |
                        +----------------------+----------------------+
                        |                                             |
                        v                                             v
          +---------------------------+                 +---------------------------+
          | Phase 1 Live CT Viewport  |                 | Phase 3 Report Studio     |
          | ------------------------- |                 | ------------------------- |
          | - Left: Scanner Controls  |                 | - Clinical Findings       |
          |   (3-State Action Button) |                 |   (Real-time editable)    |
          | - Center: CT 2D Viewer    |                 | - Clinical Impression     |
          | - Right: Findings & Score |                 | - Digital Signature Pad   |
          +---------------------------+                 | - Lock-on-Sign Protection |
                                                        | - "Edit Report" Revisions |
                                                        | - 1-Click PDF Export      |
                                                        +---------------------------+
```

1. **Patient Session Lifecycle & 3-State Machine**:
   - **Mandatory Patient Demographics**: Patient ID, Patient Name, Age, and Sex are strictly required to start a scan, ensuring every archive, JSON finding, and generated clinical report is uniquely indexed and auditable.
   - **3-State Control Flow**:
     - `ready` (`▶ Start Scan`): Validates patient fields and starts the live DICOM C-STORE stream.
     - `scanning` (`⏹ Stop Scan & Close Record`): Shows real-time received slice counter with a pulsing live indicator; clicking Stop prompts to finalize and zip the record.
     - `done` (`🗜️ Archive & Close Patient Record`): Gently pulses in amber when replay finishes, clearly marking the end of the current patient's examination without prematurely jumping back to "Start Scan". Holding this state allows full inspection before explicit archival.
   - **Automated Archival & Multi-Patient Succession**: On session closure, all DICOM files in `incoming/` are compressed into a `.zip` archive named with the Patient ID and timestamp inside `patient_archives/`. The frontend slice memory, accumulator, and buffer are wiped clean, allowing the next patient scan to start immediately in the same running terminal session without backend restart.

2. **Backend Resilience & Deadlock-Free Concurrency**:
   - **Reentrant Volume Accumulator (`RLock`)**: Employs `threading.RLock` so full resets (`reset_all()`) can execute safely on patient transitions without self-deadlocking the ingestion worker.
   - **Asynchronous Association Cleanup**: In `pipeline.py`, association-release processing (`on_association_released`) is offloaded to a daemon thread (`assoc-release-worker`) so pynetdicom's `EVT_RELEASED` handler returns instantly, enabling back-to-back patient connections.
   - **Explicit API Reset**: `/api/scanner/start` invokes `pipeline.accumulator.reset_all()` and `pipeline.buffer.reset()` via a singleton accessor (`get_pipeline()`), ensuring even duplicate test series can be rescanned cleanly.
   - **Hardware-Adaptive Phase 2 Timeout**: TotalSegmentator subprocess timeout is dynamically calculated (`compute_dynamic_timeout`) at runtime:
     $$\text{Timeout} = (\text{Base: 120s GPU / 600s CPU}) \times \text{RAM Multiplier (1.0}\times\text{–4.0}\times) + \text{Volume Size Bonus (0–180s)}$$
     Clamped between 60s and 1800s (30 min) to protect against hangs while giving CPU systems sufficient headroom.

3. **Graceful Browser Shutdown & Process Tracking**:
   - Runtime process IDs are tracked in `.prism_pids.json`.
   - When the user closes the browser tab or clicks "Shutdown System", browser Beacon signals hit `/api/system/shutdown`, cleanly closing all backend and frontend terminal windows and releasing all ports (`11112`, `8001`, `8005`, `5173`).

4. **Clinical Report Studio & Revision System**:
   - **Live Editable Findings & Impressions**: Radiologists can freely edit, format, or type new clinical observations.
   - **Digital Signature Pad**: Draw with mouse or touch, instant `Apply Signature` transition with transparent PNG vector export and paper badge preview.
   - **Lock-on-Sign Safety**: Clicking `Sign & Finalize Report` immediately locks inputs to prevent tampering.
   - **In-Place Revisions (`Edit Report`)**: Unlocks finalized findings for revisions and replaces the stored report with the updated version upon re-signing.
   - **Zero-Hallucination PDF Generator**: Exports standardized medical PDF reports with patient demographics, numerical HU telemetry, findings, and digital signatures.

---

## 18. Complete Step-by-Step User Guide

Follow these steps to run PRISM on your computer from start to finish.

### Step 1: One-Click Installation (Per Operating System)

#### A. Windows (PowerShell)
1. **Clone the repository:**
   ```powershell
   git clone https://github.com/gunanidhi-ms/MAJOR-PROJECT-PRISM.git
   cd MAJOR-PROJECT-PRISM
   ```
2. **Run One-Click Installer (`install_prism.ps1`):**
   ```powershell
   .\install_prism.ps1
   ```
   *This automated script creates the Python virtual environment, detects your NVIDIA GPU and configures CUDA PyTorch (with graceful CPU fallback), installs all dependencies across all three phases, installs Ollama via `winget` (if missing), pulls the recommended `llama3.2:3b` model, and runs `npm install` for the React frontend.*

#### B. Linux & macOS (Bash)
1. **Clone & Install:**
   ```bash
   git clone https://github.com/gunanidhi-ms/MAJOR-PROJECT-PRISM.git
   cd MAJOR-PROJECT-PRISM
   chmod +x install_prism.sh start_prism.sh
   ./install_prism.sh
   ```

---

### Step 2: One-Command Startup

Start the entire end-to-end PRISM ecosystem with a single command.

**Windows:**
```powershell
.\start_prism.ps1
```
*(Automatically verifies Ollama health, sets GPU environment flags, checks backend readiness on `:8001` and `:8005`, displays the active hardware profile, launches the backend orchestrator and React frontend, and opens `http://localhost:5173/` in your browser).*

**Linux / macOS:**
```bash
./start_prism.sh
```

---

### Step 3: Triggering a New Scan & Automated Pipeline

1. On the **Dashboard**, click the primary **`New Report`** button.
2. The UI instantly routes to the **Phase 1 Live CT Viewport** (`/live`).
3. In the left-hand **Scanner Control** panel (expandable via `▲ Hide` / `▼ Show`):
   - **Dataset Directory**: Enter the path to your CT DICOM series (default test series provided).
   - **Patient ID / Name / Age / Sex**: Enter mandatory patient demographics (all 4 fields are required to start).
   - **Speed (Delay ms)**: Replay speed in milliseconds (default: `30ms`).
4. Click **`▶ Start Scan`**:
   - Live DICOM slice streaming initiates immediately over port `11112`.
   - The action button transitions to red **`⏹ Stop Scan & Close Record`** with a live pulsing counter.

---

### Step 4: Phase 1 — Real-Time CT Ingestion & 2D Screening

1. **Live Viewport Rendering:**
   - Incoming slices stream at 20+ fps, rendering to the high-contrast CT viewer.
   - Interactive windowing presets (`DICOM Default`, `Soft Tissue`, `Lung`, `Bone`, `Mediastinum`, `Brain`, `Liver`) adjust Hounsfield Unit contrast in real time.
   - Anomaly bounding boxes (`GGO`, `NOD`, `HYPO`, `LYTIC`, `HEM`) and emergency score indexes (0–100) render directly over the image.
2. **Navigation:**
   - Use the bottom range slider, keyboard arrow keys, mouse wheel, or toggle **`▶ Play`** for automatic cine scrolling.

---

### Step 5: Autonomous Phase 2 & Phase 3 Execution

1. Once all slices are ingested, Phase 1 completes and the backend automatically triggers **Phase 2 (TotalSegmentator & 3D AI)**.
2. A floating **Pipeline Progress Panel** appears in the bottom-right corner:
   - **Phase 1 (CT Ingestion)**: ✅ Complete with total slices and flagged anomaly count.
   - **Phase 2 (Segmentation & AI)**: 🔄 Analyzing CT findings with live elapsed seconds timer (backed by adaptive hardware timeout).
   - **Phase 3 (Report Generation)**: ⏳ Generating draft report with AI (`llama3.2:3b`).
   - Animated progress bar reflects overall pipeline traversal.

---

### Step 6: Automated Landing on Draft Report Editor

1. When Phase 3 completes report synthesis, the Live Viewer triggers an automated **3-second navigation countdown** (`Draft Report Ready! Navigating in 3s...`).
2. You can click **`Go Now →`** or wait for the timer to automatically land on the **Draft Report Editor** (`/report/:studyId`).
3. A welcome banner confirms: *"🎉 Pipeline Complete! Your CT scan has been processed through all 3 phases."*

---

### Step 7: Review, In-Place Editing, Signing & PDF Export

1. **Review & In-Place Text Editing:**
   - The **Findings** and **Impression** sections contain the synthesized clinical observations.
   - Click inside the text areas to edit, refine phrasing, or add notes.
2. **Apply Digital Signature:**
   - Click **`✍ Sign Report`** in the top action bar.
   - Enter Radiologist Name and Designation.
   - Draw your digital signature on the interactive signature pad and click **`Confirm Sign & Lock`**.
   - The report transitions to `Signed` status and locks against accidental tampering.
3. **In-Place Revisions (`Edit Report`):**
   - If revisions are needed after signing, click **`Edit Report`** to unlock fields and re-sign when updated.
4. **1-Click Standardized PDF Export:**
   - Enter a custom filename (optional) and click **`Download PDF`** to generate a publication-ready clinical report with patient demographics, numerical HU telemetry, organ metrics, and digital signature.

---

### Step 8: Multi-Patient Continuation & Archival

1. To transition to a new patient, click the amber **`🗜️ Archive & Close Patient Record`** button in the Live Viewer.
2. Confirm the archive modal to automatically zip all raw DICOMs, pipeline telemetry, and reports into `patient_archives/`.
3. The interface resets all patient fields and slice buffers back to `ready` — you can immediately enter the next patient's details and click `▶ Start Scan` without restarting the backend or frontend!

---

## 19. Report Enrichment: Clinical Context, Localisation & Impression Synthesis

This section documents the cross-phase upgrade that makes the draft report read like a CT report instead of a measurement dump. Every new statement is either read from the DICOM header or is a geometric fact about the segmentation — PRISM still never diagnoses.

### 19.1 What changed, by phase

| Phase | Change | File |
|---|---|---|
| **1 – Ingestion** | Reads technique/clinical tags (scanner, model, kVp, kernel, contrast agent/route/volume, patient position, study date, referrer, accession, reason for study / requested procedure) from every slice | `phase1_ingestion/dicom_listener.py` |
| | Accumulator keeps the first non-empty value per tag and exposes them (`protocol`, `clinical_indication`, `technique`, …) in `get_series_metadata()` | `phase1_ingestion/volume_accumulator.py` |
| | **Volume QC** at export: duplicate slice positions, z-gaps, thick slices, non-square pixels, inconsistent/non-standard orientation → `volume_qc` | `phase1_ingestion/volume_accumulator.py` |
| | **Prior-study lookup**: earlier archived studies for the same `PatientID` (current StudyUID excluded) | `phase1_ingestion/prior_study_lookup.py`, `pipeline.py` |
| **2 – Segmentation** | New `anatomy_context.py`: orientation inferred from anatomy, organ FOV-truncation flag, true surface contacts, intra-organ sub-location, and side / vertebral level / paravertebral position for unassigned lesions | `phase2_segmentation/anatomy_context.py` |
| | Adapter forwards indication, technique, comparison, QC, `sub_location`, `adjacent_structures`, `touches_fov_edge`; unassigned lesions now carry shape/HU and a position description | `phase2_segmentation/findings_adapter.py` |
| | Fixed: organs were emitted twice when `organ_stats` was empty | `phase2_segmentation/findings_adapter.py` |
| **3 – Reporting** | New optional contract fields | `phase3_reporting/app/models.py` |
| | New `COMPARISON:` section; header-driven `TECHNIQUE:`; `INDICATION:` includes the clinical question | `phase3_reporting/services/template_engine.py` |
| | Volume deviations are a separate, first-listed Impression category; Impression is grouped by body system | `phase3_reporting/services/template_engine.py` |
| | LLM prompt preserves the new headers and numbered lists | `phase3_reporting/app/prompts.py` |

### 19.2 Data flow

```
DICOM tags ─► dicom_listener ─► pipeline.spacing_meta ─► VolumeAccumulator
                                                         ├─ technique / indication / study_date
                                                         └─ volume_qc (at export)
pipeline ── prior_study_lookup(exports/) ─► series_meta.prior_studies
series_meta ─► Phase 2 ─► TotalSegmentator label map
                         └─ anatomy_context.build_anatomy_context()
                              orientation · contacts · sub_location · unassigned position
findings_adapter ─► StructuredFindings JSON ─► Phase 3 template engine ─► report
```

### 19.3 New optional fields in the Phase 2 → Phase 3 payload

All fields are optional; older payloads still validate.

```jsonc
{
  "indication": "abdominal pain",
  "technique": { "manufacturer": "...", "model": "...", "kvp": "120",
                 "convolution_kernel": "B30f", "contrast_agent": "...",
                 "contrast_route": "IV", "slice_thickness_mm": 5.0 },
  "comparison": { "prior_studies": [ { "archive": "...zip", "archived_on": "2026-01-02" } ] },
  "volume_qc": { "slice_count": 120, "z_extent_mm": 600.0, "warnings": [] },
  "orientation_verified": true,
  "organs": [ {
    "organ": "liver",
    "adjacent_structures": ["stomach", "portal_vein_and_splenic_vein"],
    "touches_fov_edge": false,
    "anomalies": [ { "sub_location": "lateral, posterior portion", "...": "..." } ]
  }, {
    "organ": "unclassified", "unassigned": true,
    "anomalies": [ { "location": "right paravertebral region, at the T10 level; closest segmented structures: ..." } ]
  } ]
}
```

### 19.4 Behaviour reference

- **Indication.** Shown verbatim when the header carries one; otherwise the report states it was not provided. Keyword routing (`_INDICATION_RULES`) maps the indication to relevant body systems so the Impression can say whether abnormalities fall in the relevant systems. It never infers a diagnosis.
- **Technique.** Scanner, slice thickness, kVp, kernel and contrast come from the header. If contrast tags are absent the report says *"not recorded in the DICOM header"* (with an inference hint from vascular HU worded as such). Oral contrast is reported only if the header says so. The old unverifiable "multiplanar reformats" claim was removed.
- **Comparison.** Lists archived priors (date from the archive filename) and states that automated image comparison is **not** performed; otherwise "No prior comparator study available".
- **Organ coverage filter.** TotalSegmentator attempts all 104 organ labels on every scan. When a scan does not cover an organ (e.g. an abdomen/pelvis CT that barely clips the lung bases), TS still returns a tiny fragment mask. `findings_adapter` now suppresses any organ whose measured volume falls below a per-class minimum coverage threshold (`_MIN_COVERAGE_CC`). This prevents ghost entries such as `Vertebrae T12: 0.1 cc`, `Rib Left 11: 0.3 cc`, or `Right Upper Lobe: 14.4 cc` appearing in an abdomen report. Suppressed organs are logged at INFO level.
- **Sub-localisation.** Centroid position inside the host organ in patient terms (lungs: central/peripheral, anterior/posterior, upper/lower; other organs: medial/lateral or right/left-sided, anterior/posterior, superior/inferior). Skipped for axes < 15 mm.
- **Orientation safety.** Left/right, anterior/posterior and superior/inferior are inferred from paired L/R organs, spine-vs-body centroid and vertebral order. If any axis cannot be verified, directional wording is suppressed and only vertebral level and nearest structures are reported.
- **Organ relationships.** Contacts come from mask dilation (~3 mm), not bounding-box overlap. Large lesions (≥ 50 mm) add a mass-effect advisory; for hepatic lesions the report states that biliary dilatation and portal/hepatic venous patency are **not assessable** from segmentation and need radiologist review (TotalSegmentator does not segment the intrahepatic biliary tree).
- **Unclassified lesions.** One numbered line each with position (side, paravertebral, vertebral level, nearest structures), size, volume and HU — replacing the repeated "within the Unclassified".
- **Volume deviations.** Reduced/enlarged organs (with "marked" at < 50 % / > 150 % of the reference bound) lead the Impression. Volumes under 10 % of the lower bound add a verification caveat (true absence/atrophy vs segmentation limitation vs incomplete coverage); organs touching the scan boundary are flagged as partially included.
- **Impression.** Numbered points grouped by body system with counts, bilateral flag and largest lesion size, a cross-system summary, one shared closing statement (no longer repeated on every line) and QC caveats.

### 19.5 Validator / template alignment

The validator and template engine were originally out of sync (12 pre-existing test failures). In the current release all discrepancies are resolved:

| Issue | Fix |
|---|---|
| Density normaliser applied mm-unit logic to HU strings | Added `_norm_hu()` — lowercases, collapses whitespace, does **not** append `mm` |
| Range patterns (`-10 to 45 HU`) only captured the bound closest to `HU` | Extractor now anchors on every `HU` token and adds **all** numbers in the preceding 40-char window |
| Validator required `sphericity` and density `std` in the report text | Both removed from mandatory set — they are internal metrics not printed by the template |
| Validator treated `volume_cc` as a linear measurement | `_is_volume_key()` now excludes volume keys from the mm-measurement string |
| Organ-label mismatch (lung-lobe, kidney cyst, Oesophagus) | Validator accepts either the raw JSON key words or the display label produced by `TemplateEngine._label()` |
| Invented-number hallucination undetected | Added `reference_text` parameter; numbers present in LLM output but absent from the template are rejected |
| Ghost organs in report (ribs, vertebrae, heart on abdomen CT) | `_MIN_COVERAGE_CC` per-class filter in `findings_adapter` — organs below ~20% of expected volume excluded |

### 19.6 Known limitations

- The Phase 1 **triage screen** heuristics were not changed; `phase1_ingestion/tests/test_triage_screen.py` has pre-existing failures unrelated to this work.
- Prior studies are matched by `PatientID` and archive filename only; no image registration or lesion-level comparison is done.
- Reference volume ranges are generic adult values (`_ORGAN_REFS`); there is no age/sex adjustment yet.
- Sub-location is approximate (organ bounding-box thirds), not a true lobar-segment (e.g. Couinaud) assignment. The report text says "approximate location" explicitly.
- Volume caveat: when an organ `touches_fov_edge`, the report text states the measured volume **may be underestimated if the organ extends beyond the scan boundary**. This is flagged in the Impression too, but cannot be auto-corrected without a larger FOV rescan.


### 19.7 Tests

```powershell
# Full Phase 3 suite (88 tests, all pass)
cd phase3_reporting; ..\venv\Scripts\python.exe -m pytest tests -q

# Phase 3 report content and validator (focused)
cd phase3_reporting; ..\venv\Scripts\python.exe -m pytest tests/test_report_enrichment.py tests/test_validator.py -v

# Phase 2 anatomy context + adapter
cd ..; venv\Scripts\python.exe -m pytest phase2_segmentation/tests/test_anatomy_context.py phase2_segmentation/tests/test_findings_adapter.py -q
```

#### 19.7.1 Clinical indication UI field

The **Generate** page (`prism-react/src/pages/Generate.tsx`) now has a free-text **Clinical Indication** field above the Generate button. When filled in it is injected into the payload as the `indication` key, overriding the DICOM-tag value from Phase 1. Left blank, the DICOM-derived value (or the "not provided" fallback) is used unchanged.


---

## 20. ML False Positive Reduction (FPR) — Roadmap

### 20.1 The problem

PRISM's Phase 1 triage and Phase 2 candidate detection are currently **rule-based** (z-score thresholds, HU range checks). Rules do not understand anatomy — a bowel loop at the same HU as a renal cyst looks identical to a threshold. This produces false positives that consume radiologist review time.

A **False Positive Reduction (FPR)** classifier sits between Phase 2 and Phase 3 and scores each candidate: *"Is this a real finding or a detection artefact?"*

### 20.2 What is already ML in PRISM

| Component | Technology | ML? |
|---|---|---|
| Phase 1 triage | Z-score / intensity statistics | ❌ Rule-based |
| Phase 2 segmentation | **TotalSegmentator** (nnU-Net, 104 organs) | ✅ Deep learning |
| Phase 2 candidate scoring | Statistical thresholds | ❌ Rule-based |
| Phase 3 report | Template + Ollama LLM | ✅ LLM |
| **FPR classifier** | XGBoost / LightGBM (planned) | 🔄 Planned |

The segmentation is already deep learning. FPR adds a trained second-stage classifier on top of the features already extracted.

### 20.3 Where false positives originate

```
Phase 1 triage
  └── flags any voxel cluster with unusual HU / z-score
        → normal fat lobules, vessels, bowel loops → all flagged

Phase 2 candidate detection
  └── anything outside expected HU range for that organ
        → motion artefact, partial-volume effect, beam-hardening → flagged
```

### 20.4 Proposed FPR architecture

```
Phase 2 output (candidates + features)
         │
         ▼
 ┌───────────────────┐
 │   FPR Classifier  │  inputs (already computed by PRISM):
 │  XGBoost / LGBM   │  • mean_density_hu, min_hu, max_hu, std_hu
 │                   │  • volume_cc, long_axis_mm, short_axis_mm
 │                   │  • sphericity, margin type
 │                   │  • sub_location (e.g. "peripheral, lower")
 │                   │  • adjacent_structures (e.g. portal vein)
 │                   │  • touches_fov_edge
 │                   │  • organ (e.g. liver, right_kidney)
 │                   │  • patient age, sex
 └───────────────────┘
         │
         ▼
  fpr_score  (0.0 – 1.0 true-positive probability)
         │
   threshold (e.g. 0.3)
    ├── score ≥ threshold → forwarded to Phase 3 (report it)
    └── score <  threshold → suppressed (logged, not reported)
```

No new feature extraction is needed — every input listed above is already produced by `findings_adapter.py` and `anatomy_context.py`.

### 20.5 How to evaluate without a radiologist

> **You do not need to be a doctor to evaluate this model.**
> The doctors already annotated the public datasets. Your role is systems engineering: run PRISM, compare outputs to the labels, measure numbers.

#### 20.5.1 Public datasets (ground truth provided)

| Dataset | Organs / findings | Access |
|---|---|---|
| **KiTS23** | Kidney + tumour masks | `kits-challenge.s3.us-east-2.amazonaws.com` |
| **LiTS** | Liver + lesion masks | `competitions.codalab.org/competitions/17094` |
| **LIDC-IDRI** | Lung nodules + 4-radiologist agreement | `wiki.cancerimagingarchive.net` |
| **TCIA** | Many organ-specific annotated CTs | `cancerimagingarchive.net` |

#### 20.5.2 Evaluation logic (pure geometry, no medical knowledge)

```python
for candidate in prism_phase2_candidates:
    # IoU = Intersection over Union — pure 3D geometry
    overlap = compute_iou(candidate.bbox_3d, ground_truth_mask)

    if overlap > 0.1:          # candidate overlaps a labelled finding
        label = "true_positive"
    else:                       # candidate is in normal tissue
        label = "false_positive"

# You never need to know *why* something is a tumour.
# You only check: does your box overlap the annotated box?
```

IoU is geometry. No medical interpretation required.

#### 20.5.3 Proxy metrics (no labels needed at all)

While building, before you have annotated data:

| Proxy metric | What it signals |
|---|---|
| Flagging rate on DICOM studies tagged `StudyDescription = Normal` | Lower = fewer FPs |
| Sphericity distribution of candidates | Real lesions cluster 0.7–1.0; FPs are irregular |
| Candidate HU vs known tissue-type HU bands | FPs often fall inside normal fat or vessel ranges |
| Count of candidates that overlap normal TotalSegmentator mask voxels | Overlap = likely FP |

#### 20.5.4 TotalSegmentator as pseudo-ground-truth

You already run TotalSegmentator. Any PRISM candidate that sits **inside a normal organ mask** but was not expected (e.g. high-HU voxel inside liver parenchyma that TotalSegmentator says is normal liver) can be treated as a pseudo-FP label. This gives you a self-supervised training signal **from your own pipeline** without any external dataset.

### 20.6 Metrics to report

| Metric | Definition | Target |
|---|---|---|
| **AUROC** | Rank quality of TP vs FP | > 0.85 |
| **Sensitivity at 2 FP/image** | Clinical CAD standard | > 0.80 |
| **FP reduction %** | `(fp_before − fp_after) / fp_before × 100` | > 40 % |
| **False negative rate** | Real findings suppressed by FPR model | < 5 % |

Compute AUROC with `sklearn.metrics.roc_auc_score`. FROC curve with `pylidc` (for LIDC-IDRI).

### 20.7 What to write in your project report

> *"Ground truth annotations were sourced from the KiTS23 challenge dataset, where labels were provided by board-certified radiologists. The author's contribution was designing and evaluating an automated false positive reduction classifier using these pre-existing annotations. No independent medical interpretation was performed."*

Every published medical imaging ML paper says this. It is the standard framing for engineering-track work.

### 20.8 Planned file structure

```
phase2_segmentation/
  fpr_model/
    feature_extractor.py   ← pulls features from StructuredFindings + anatomy_context
    train.py               ← XGBoost training script, cross-validation
    predict.py             ← scores each candidate at inference time
    evaluate.py            ← FROC curve, AUROC, FP reduction %
    model.pkl              ← trained artifact (not committed; generated by train.py)
    datasets/
      kits23_loader.py     ← loads KiTS23 masks and PRISM candidate pairs
      lidc_loader.py       ← loads LIDC-IDRI annotations
```

The `fpr_score` field is already defined on the `Anomaly` model in `phase3_reporting/app/models.py` as an optional float. The template engine and validator will pass it through unchanged.

### 20.9 Recommended execution order

```
Step 1 — Download KiTS23 (kidney) or LIDC-IDRI (lung)  ~10–20 GB
Step 2 — Run PRISM Phase 2 on 20–50 CTs → save candidates as JSON
Step 3 — Run evaluate.py: compare candidates vs masks → generate TP/FP labels
Step 4 — Train XGBoost on the labelled features (train.py)
Step 5 — Measure AUROC and FROC before/after FPR model
Step 6 — Report: "FPR reduced false positives by X% at 80% sensitivity"
```

> [!NOTE]
> Steps 1–6 require no medical knowledge. Step 3 is pure 3D bounding-box overlap (IoU). Step 4 is standard supervised ML. Step 5 is `sklearn` metrics.

> [!TIP]
> Start with KiTS23 — it has the simplest file structure (one NIfTI volume, one NIfTI segmentation per patient) and a Python data loader on PyPI: `pip install kits23`.
