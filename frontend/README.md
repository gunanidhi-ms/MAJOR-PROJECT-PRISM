# PRISM Frontend

Django web interface for the PRISM Phase 3 Radiology Report Generator.

## Tech Stack

- **Django Templates** — server-rendered pages
- **HTML5** — semantic markup
- **Tailwind CSS** — utility-first styling (CDN)
- **JavaScript (ES6)** — API client and page logic
- **Alpine.js** — reactive UI components

## Prerequisites

1. **Python 3.10+**
2. **FastAPI backend running** on port 8000 (see `../phase3_reporting/`)

## Quick Start

### 1. Start the FastAPI backend

```bash
cd ../phase3_reporting
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend runs at: http://127.0.0.1:8000

### 2. Start the Django frontend

```bash
cd frontend
python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt
copy .env.example .env
python manage.py migrate
python manage.py runserver 8080
```

Frontend runs at: http://127.0.0.1:8080

## Pages

| URL | Description |
|-----|-------------|
| `/` | Dashboard — stats, recent reports, quick-start cases |
| `/generate/` | Generate report from structured findings JSON |
| `/reports/` | List all stored reports |
| `/report/<study_id>/` | View, edit, and sign a report |

## Workflow

1. Open **Generate Report** and pick a sample case (or paste JSON)
2. Click **Generate Draft Report** — calls FastAPI `/api/v1/generate-report`
3. Review and edit **Findings** and **Impression** on the report page
4. **Save Changes** then **Sign Report** to finalise

## Configuration

Copy `.env.example` to `.env`:

```env
PRISM_API_BASE_URL=http://127.0.0.1:8000/api/v1
DJANGO_SECRET_KEY=your-secret-key
DJANGO_DEBUG=True
```

## Project Structure

```
frontend/
├── manage.py
├── requirements.txt
├── prism_frontend/       # Django project settings
├── reports/              # Main app (views, API client)
├── templates/            # Django HTML templates
├── static/               # CSS & JavaScript
└── sample_data/          # Example findings JSON
```
