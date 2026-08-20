from django.urls import path, re_path

from reports import views

urlpatterns = [
    # ── JSON API proxies (unchanged — called by React frontend) ──────────
    path("api/health/",                         views.api_health_proxy,       name="api_health"),
    path("api/reports/",                        views.api_list_reports_proxy, name="api_reports"),
    path("api/report/<str:study_id>/",          views.api_get_report_proxy,   name="api_get_report"),
    path("api/report/<str:study_id>/update/",   views.api_update_proxy,       name="api_update_report"),
    path("api/generate/",                       views.api_generate_proxy,     name="api_generate"),
    path("api/sign/",                           views.api_sign_proxy,         name="api_sign"),
    path("api/sample/<str:case_id>/",           views.api_sample_case,        name="api_sample_case"),

    # ── React SPA catch-all — must be LAST ───────────────────────────────
    # All non-API routes are handled by the React frontend.
    re_path(r"^.*$", views.index_view, name="react_spa"),
]
