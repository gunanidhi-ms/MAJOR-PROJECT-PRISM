from django.urls import path

from reports import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("generate/", views.generate_page, name="generate"),
    path("generate/submit/", views.generate_and_redirect, name="generate_submit"),
    path("reports/", views.reports_list, name="reports_list"),
    path("report/<str:study_id>/", views.report_detail, name="report_detail"),
    # JSON proxies for Alpine.js / fetch
    path("api/health/", views.api_health_proxy, name="api_health"),
    path("api/reports/", views.api_list_reports_proxy, name="api_reports"),
    path("api/report/<str:study_id>/", views.api_get_report_proxy, name="api_get_report"),
    path("api/report/<str:study_id>/update/", views.api_update_proxy, name="api_update_report"),
    path("api/generate/", views.api_generate_proxy, name="api_generate"),
    path("api/sign/", views.api_sign_proxy, name="api_sign"),
    path("api/sample/<str:case_id>/", views.api_sample_case, name="api_sample_case"),
]
