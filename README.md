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

**Current Status (Phase 1)**
- [x] Live DICOM SCP network ingestion
- [x] Real-time latency (Target: <50ms per slice)
- [x] Universal statistical triage (disease-agnostic)
- [x] Dynamic geometry and bounding box generation
- [x] Live WebSocket-driven frontend dashboard

**Future Roadmap (Phase 2 & Beyond)**
- [ ] Multi-slice Z-axis state persistence (3D tracking)
- [ ] Phase 2: Deep Learning Organ Segmentation (U-Net)
- [ ] Disease Classification & Volumetric Radiomics
- [ ] Large-scale formal clinical validation

---

## Table of Contents

1. [Running PRISM (Quick Start)](#1-running-prism-quick-start)
2. [Project Overview](#2-project-overview)
3. [Motivation](#3-motivation)
4. [Design Decisions](#4-design-decisions)
5. [System Architecture & Data Flow](#5-system-architecture--data-flow)
6. [Universal Phase 1 Triage Engine](#6-universal-phase-1-triage-engine)
7. [Validation Methodology](#7-validation-methodology)
8. [Module Breakdown](#8-module-breakdown)
9. [Frontend Architecture](#9-frontend-architecture)
10. [Data Models](#10-data-models)
11. [Configuration](#11-configuration)
12. [Developer Guide](#12-developer-guide)
13. [Limitations](#13-limitations)
14. [Repository Structure](#14-repository-structure)
15. [License / References](#15-license--references)

---

## 1. Running PRISM (Quick Start)

The PRISM system requires both the asynchronous Python backend and the React-based frontend dashboard to run concurrently.

### Prerequisites
- **Python 3.10+**
- **Node.js 18+**
- **Python Libraries:** `pydicom`, `numpy`, `scipy`, `websockets`

```bash
pip install pydicom numpy scipy websockets
```

### Backend Startup
Run the backend from the project root to ensure correct module resolution.
```bash
cd D:\MAJOR-PROJECT-PRISM
python -m phase1_ingestion.pipeline
```
*Expected Output:* The terminal will display the PRISM initialization banner. The DICOM SCP will bind to port `11112` and the WebSocket server to port `8001`.

### Frontend Startup
Open a separate terminal to initialize the React application.
```bash
cd D:\MAJOR-PROJECT-PRISM\frontend
npm install   # Only required on first run
npm run dev
```
*Expected Output:* The Vite server will start at `http://localhost:5173`. Open this URL in a modern web browser to access the dashboard.

### Replay Sender (Simulating a Scanner)
To test the pipeline without a live CT scanner, PRISM includes a replay utility that streams pre-existing DICOM files via the C-STORE protocol.
```bash
cd D:\MAJOR-PROJECT-PRISM
python -m phase1_ingestion.replay_sender
```
*Expected Output:* The sender pushes local DICOMs to port `11112`. The backend will log processing speeds, and the frontend will instantly render slices and overlays.

### Troubleshooting
- **Errno 10048 (Address already in use):** Ensure no other instances of PRISM, Orthanc, or dcm4chee are running on ports 11112 or 8001.
- **Frontend not receiving data:** Verify the WebSocket server is running and the frontend is successfully connecting to `ws://localhost:8001/ws/alerts`.

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
3. `dicom_listener.py` parses the byte-stream, extracts the `PixelData`, scaling tags, and pushes it to `slice_buffer.py`.
4. `slice_buffer.py` locks the priority queue, inserts the slice, and checks for contiguous order. If ready, it flushes the slice to `pipeline.py`.
5. `pipeline.py` passes the raw matrix to `hu_transform.py`.
6. `hu_transform.py` vectorizes the array into Hounsfield Units, crops the scanner padding, and returns a float32 matrix.
7. `pipeline.py` passes the float32 matrix to `triage_screen.screen_slice`.
8. `triage_screen.py` executes the 10-step math engine, outputting a `TriageResult` dataclass.
9. `pipeline.py` converts the result to JSON and pushes it into the `ws_server.py` asyncio queue.
10. `ws_server.py` broadcasts the JSON string over port `8001`.
11. The **React Frontend** receives the WebSocket message, draws the slice to an HTML5 canvas, and renders SVG bounding boxes if an alert is triggered.

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
The TCP gateway. Utilizes `pydicom.net.AE` to establish a DICOM SCP. Decodes incoming byte-streams into `PixelData` arrays and pushes them to the buffer, ensuring network I/O is never blocked by mathematical processing.

### `slice_buffer.py`
The priority reordering queue. DICOM slices arrive out-of-order. The buffer uses a `threading.Lock` and `heapq` to sort slices by `InstanceNumber`. Includes a timeout watchdog to prevent pipeline stalls if a slice packet is dropped over the network.

### `hu_transform.py`
Vectorized pixel standardization. Executes `HU = Pixel * Slope + Intercept`. Disables hard-clipping during ingestion to preserve scanner padding (e.g., `-3024 HU`). Implements an `auto_crop` algorithm to strip artificial padding before processing.

### `pipeline.py`
The central orchestrator. An infinite worker thread that pulls from the `SliceBuffer`, calls `hu_transform`, executes `triage_screen.py`, formats the `TriageResult` to JSON, and pushes it to the `ws_server`. Wraps execution in global `try/except` blocks to ensure fault tolerance.

### `triage_screen.py`
The mathematical brain containing the 10-step triage algorithm. Relies entirely on `numpy` and `scipy.ndimage` for CPU-bound optimization.

### `ws_server.py`
An `asyncio` WebSocket server running in a daemon thread. Fans out the JSON alerts from `pipeline.py` to all connected React clients concurrently.

---

## 9. Frontend Architecture

The frontend is a **React 18** application scaffolded with **Vite**.

* **Dashboard UI:** A dark-mode, glassmorphic interface designed for high-contrast radiology viewing environments.
* **WebSocket Hook (`useWebSocket`):** Manages the persistent WebSocket connection, handles auto-reconnection with exponential backoff, and parses the high-velocity JSON stream.
* **Canvas Renderer:** Maps raw `mean_hu` values to grayscale pixels on an HTML5 `<canvas>`, providing hardware-accelerated rendering capable of handling 20+ fps.
* **Overlay Engine:** Maps server-side bounding boxes `[x, y, w, h]` to client-side coordinates, drawing distinct SVG boxes directly over suspected pathologies. Colors adapt based on severity.
* **Telemetry Panel:** Displays live metrics including Latency, Current Scan Type, Body Region, and overall Processing Status.

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

## 11. Configuration

PRISM can be tuned by modifying constants in the backend scripts:
* **DICOM_PORT:** `11112` (`dicom_listener.py`)
* **WS_PORT:** `8001` (`ws_server.py`)
* **BUFFER_TIMEOUT_SEC:** `2.5` (`slice_buffer.py`) - Adjust if your network has high latency jitter.
* **LATENCY_TARGET_MS:** `50.0` - Logging threshold for performance warnings.

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

## 13. Limitations

* **Lack of 3D Context:** Slices are evaluated in isolation to ensure zero latency. A thin blood vessel curving into the Z-axis may temporarily appear as an isolated dense circle. 
* **Metallic Artifacts:** Dental amalgams, hip replacements, or pacemakers cause massive beam-hardening streaks. While Step 7 Geometric Filtering attempts to suppress thin streaks, massive scatter can distort the local Mean and Std Dev, potentially blinding the statistical detector in that region.
* **No Stateful Tracking:** The engine evaluates Slice 42 without knowing the outcome of Slice 41. It relies on the frontend or Phase 2 to aggregate slice-level findings into a cohesive volume.

---

## 14. Repository Structure

```text
MAJOR-PROJECT-PRISM/
├── phase1_ingestion/            # Backend Processing & Networking
│   ├── dicom_listener.py        # DICOM SCP Server
│   ├── slice_buffer.py          # Priority Reordering Queue
│   ├── hu_transform.py          # Hounsfield Unit Math
│   ├── triage_screen.py         # The 10-Step Statistical Engine
│   ├── pipeline.py              # Main Orchestrator Loop
│   ├── ws_server.py             # WebSocket Telemetry Server
│   ├── ct_machine_emulator.py   # Replay Utility Core
│   └── replay_sender.py         # CLI for Replay Utility
│
├── frontend/                    # React Dashboard
│   ├── src/
│   │   ├── components/          # UI Components
│   │   ├── hooks/               # useWebSocket logic
│   │   └── App.jsx              # Main View
│   ├── package.json
│   └── vite.config.js
│
├── sample_dicoms/               # Test Data
└── README.md                    # This Document
```

---

## 15. License / References

This project is developed as an open-source medical imaging research initiative. 

* **License:** MIT License. See `LICENSE` for details.
* **DICOM Standard:** [NEMA DICOM PS3](https://www.dicomstandard.org/)
* **Hounsfield Unit Mathematics:** Radiographic attenuation standardization algorithms.
* **Core Libraries:** `pydicom` (DICOM parsing), `scipy.ndimage` (Morphology & Spatial algorithms).