# PRISM – Phase 3: Autonomous Radiology Report Generator

Phase 3 is the report generation module of the **PRISM** pipeline. It converts structured findings JSON into a professional draft radiology report using a deterministic template engine, optional LLM refinement (Ollama + Mistral), and a validation framework that verifies the generated report before it is returned.

---

# Features

- Deterministic template-based report generation
- Optional LLM refinement using Ollama (Mistral 7B)
- Automatic fallback to template if the LLM is unavailable
- Multi-stage validation of LLM output
- REST API built with FastAPI
- Interactive Swagger documentation
- JSON-based report storage
- Modular architecture for future database integration

---

# Phase 3 Workflow

```
Structured Findings JSON
        │
        ▼
Template Engine
        │
        ▼
Ollama (Grammar & Fluency)
        │
        ▼
Validator
        │
        ├── Pass → LLM Report
        └── Fail → Template Report
        │
        ▼
JSON Storage
        │
        ▼
FastAPI REST API
```

The structured findings JSON is the **source of truth**. The LLM is used only to improve readability and is not allowed to modify clinical facts.

---

# Project Structure

```
phase3_reporting/

├── app/
├── api/
├── services/
├── storage/
├── sample_data/
├── reports/
├── tests/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# Prerequisites

- Python 3.10+
- Git
- Ollama

---

# Installation

## 1. Clone the repository

```bash
git clone <repository-url>
cd MAJOR-PROJECT-PRISM/phase3_reporting
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure environment

Copy:

```text
.env.example
```

to

```text
.env
```

No changes are required for the default configuration.

---

# Installing Ollama

Download Ollama from:

https://ollama.com/download

Verify installation:

```bash
ollama --version
```

Download the model used by this project:

```bash
ollama pull mistral:7b
```

Verify:

```bash
ollama list
```

The model should appear as:

```text
mistral:7b
```

If the Ollama desktop application is running, no additional setup is required.

---

# Running the Project

Start the API server:

```bash
uvicorn app.main:app --reload
```

You should see output similar to:

```text
PRISM Phase 3 – Radiology Report Generator

Ollama host : http://localhost:11434

Ollama model : mistral:7b

Application startup complete
```

---

# Testing the API

Open Swagger:

```
http://localhost:8000/docs
```

Swagger automatically documents every endpoint and allows testing directly from the browser.

---

# Testing Workflow

The recommended order is:

## 1. Generate Report

```
POST /generate-report
```

Use any JSON from:

```
sample_data/
```

Examples:

```
normal_case.json

single_lesion.json

multi_lesion.json

multi_organ.json
```

---

## 2. Retrieve Report

```
GET /report/{study_id}
```

---

## 3. Update Report

```
PUT /report/{study_id}
```

Used by the radiologist to edit findings or add an impression.

---

## 4. Sign Report

```
POST /sign-report
```

Marks the report as signed.

---

## 5. List Reports

```
GET /reports
```

Returns all stored reports.

---

## 6. Health Check

```
GET /health
```

Shows server status and Ollama connectivity.

---

# Validation

Every LLM-generated report is validated before it is returned.

The validator checks:

- Numeric values
- Measurements
- Density values
- Medical terminology

If any validation fails:

```
LLM Output

↓

Rejected

↓

Template Report Returned
```

---

# Configuration

Configuration is stored in `.env`.

Important variables:

```ini
OLLAMA_HOST=http://localhost:11434

OLLAMA_MODEL=mistral:7b

SKIP_LLM=false

REPORTS_DIR=reports
```

To disable the LLM completely:

```ini
SKIP_LLM=true
```

The system will generate reports using only the deterministic template engine.

---

# Running Tests

Run all unit tests:

```bash
pytest tests -v
```

---

# Notes

- Reports are stored as JSON files inside the `reports/` directory.
- Swagger documentation is available at `/docs`.
- ReDoc documentation is available at `/redoc`.
- The LLM is used only for language refinement.
- The structured findings JSON remains the source of truth.

---

# License

This project is part of the PRISM research project.