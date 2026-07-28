"""
Django views for the PRISM radiology report frontend.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from reports.api_client import PrismAPIError, get_api_client

logger = logging.getLogger(__name__)

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"

SAMPLE_CASES = [
    {
        "id": "normal_case",
        "title": "Normal CT KUB",
        "description": "All organs within normal limits — baseline study.",
        "modality": "CT",
        "protocol": "CT KUB",
        "image": "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=600&h=400&fit=crop",
        "badge": "Normal",
        "badge_class": "bg-emerald-100 text-emerald-800",
    },
    {
        "id": "single_lesion",
        "title": "Single Renal Lesion",
        "description": "Well-defined hypodense lesion in the right kidney lower pole.",
        "modality": "CT",
        "protocol": "CT KUB",
        "image": "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=600&h=400&fit=crop",
        "badge": "Anomaly",
        "badge_class": "bg-amber-100 text-amber-800",
    },
    {
        "id": "multi_lesion",
        "title": "Multiple Renal Lesions",
        "description": "Bilateral renal lesions requiring detailed characterisation.",
        "modality": "CT",
        "protocol": "CT KUB",
        "image": "https://images.unsplash.com/photo-1582719478250-c89cae4dc85b?w=600&h=400&fit=crop",
        "badge": "Complex",
        "badge_class": "bg-rose-100 text-rose-800",
    },
    {
        "id": "multi_organ",
        "title": "Multi-Organ Findings",
        "description": "Hepatic and renal abnormalities on contrast-enhanced CT.",
        "modality": "CT",
        "protocol": "CT Abdomen",
        "image": "https://images.unsplash.com/photo-1530497610245-94d3c16cda28?w=600&h=400&fit=crop",
        "badge": "Multi-organ",
        "badge_class": "bg-violet-100 text-violet-800",
    },
]


def _load_sample(case_id: str) -> dict | None:
    path = SAMPLE_DATA_DIR / f"{case_id}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _safe_health() -> dict:
    try:
        return get_api_client().health()
    except PrismAPIError as exc:
        return {
            "status": "error",
            "ollama_reachable": False,
            "ollama_model": "—",
            "error": str(exc),
        }


def _safe_list_reports() -> dict:
    try:
        return get_api_client().list_reports()
    except PrismAPIError as exc:
        return {"total": 0, "reports": [], "error": str(exc)}


@require_GET
def dashboard(request):
    health = _safe_health()
    data = _safe_list_reports()
    reports = data.get("reports", [])
    draft_count = sum(1 for r in reports if r.get("status") == "draft")
    signed_count = sum(1 for r in reports if r.get("status") == "signed")

    context = {
        "health": health,
        "reports": reports,
        "total_reports": data.get("total", 0),
        "draft_count": draft_count,
        "signed_count": signed_count,
        "api_error": data.get("error"),
        "sample_cases": SAMPLE_CASES[:3],
    }
    return render(request, "reports/dashboard.html", context)


@require_GET
def generate_page(request):
    health = _safe_health()
    selected = request.GET.get("case", "normal_case")
    sample_data = _load_sample(selected) or _load_sample("normal_case")
    context = {
        "health": health,
        "sample_cases": SAMPLE_CASES,
        "selected_case": selected,
        "sample_data": sample_data or {},
        "sample_json": json.dumps(sample_data, indent=2) if sample_data else "{}",
    }
    return render(request, "reports/generate.html", context)


@require_GET
def report_detail(request, study_id: str):
    health = _safe_health()
    error = None
    report = None
    try:
        report = get_api_client().get_report(study_id)
    except PrismAPIError as exc:
        error = str(exc)

    context = {
        "health": health,
        "study_id": study_id,
        "report": report,
        "report_json": report,
        "error": error,
    }
    return render(request, "reports/report_detail.html", context)


@require_GET
def reports_list(request):
    health = _safe_health()
    data = _safe_list_reports()
    context = {
        "health": health,
        "reports": data.get("reports", []),
        "total_reports": data.get("total", 0),
        "api_error": data.get("error"),
    }
    return render(request, "reports/reports_list.html", context)


# ── JSON proxy endpoints (called from Alpine.js) ──────────────────────


@require_http_methods(["GET"])
def api_health_proxy(request):
    try:
        return JsonResponse(get_api_client().health())
    except PrismAPIError as exc:
        return JsonResponse({"status": "error", "detail": str(exc)}, status=exc.status_code)


@require_http_methods(["GET"])
def api_list_reports_proxy(request):
    try:
        return JsonResponse(get_api_client().list_reports())
    except PrismAPIError as exc:
        return JsonResponse({"detail": str(exc)}, status=exc.status_code)


@require_http_methods(["GET"])
def api_get_report_proxy(request, study_id: str):
    try:
        return JsonResponse(get_api_client().get_report(study_id))
    except PrismAPIError as exc:
        return JsonResponse({"detail": str(exc)}, status=exc.status_code)


@require_POST
def api_generate_proxy(request):
    try:
        payload = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON body."}, status=400)

    try:
        result = get_api_client().generate_report(payload)
        return JsonResponse(result)
    except PrismAPIError as exc:
        return JsonResponse({"detail": str(exc)}, status=exc.status_code)


@require_http_methods(["PUT", "POST"])
def api_update_proxy(request, study_id: str):
    try:
        payload = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON body."}, status=400)

    try:
        result = get_api_client().update_report(
            study_id,
            findings=payload.get("findings"),
            impression=payload.get("impression"),
        )
        return JsonResponse(result)
    except PrismAPIError as exc:
        return JsonResponse({"detail": str(exc)}, status=exc.status_code)


@require_POST
def api_sign_proxy(request):
    try:
        payload = json.loads(request.body)
        study_id = payload.get("study_id", "")
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON body."}, status=400)

    if not study_id:
        return JsonResponse({"detail": "study_id is required."}, status=422)

    try:
        result = get_api_client().sign_report(study_id)
        return JsonResponse(result)
    except PrismAPIError as exc:
        return JsonResponse({"detail": str(exc)}, status=exc.status_code)


@require_GET
def api_sample_case(request, case_id: str):
    data = _load_sample(case_id)
    if data is None:
        return JsonResponse({"detail": f"Sample case '{case_id}' not found."}, status=404)
    return JsonResponse(data)


@require_POST
def generate_and_redirect(request):
    """Server-side form POST: generate report then redirect to editor."""
    case_id = request.POST.get("case_id", "normal_case")
    raw_json = request.POST.get("findings_json", "").strip()

    if raw_json:
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            return render(
                request,
                "reports/generate.html",
                {
                    "health": _safe_health(),
                    "sample_cases": SAMPLE_CASES,
                    "selected_case": case_id,
                    "sample_json": raw_json,
                    "form_error": "Invalid JSON. Please check syntax and try again.",
                },
            )
    else:
        payload = _load_sample(case_id)
        if payload is None:
            return redirect("generate")

    try:
        result = get_api_client().generate_report(payload)
        return redirect("report_detail", study_id=result["study_id"])
    except PrismAPIError as exc:
        return render(
            request,
            "reports/generate.html",
            {
                "health": _safe_health(),
                "sample_cases": SAMPLE_CASES,
                "selected_case": case_id,
                "sample_json": json.dumps(payload, indent=2) if payload else raw_json,
                "form_error": str(exc),
            },
        )
