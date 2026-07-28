from django import template

register = template.Library()


@register.filter
def status_badge(status: str) -> str:
    mapping = {
        "draft": "bg-amber-100 text-amber-800 ring-amber-200",
        "signed": "bg-emerald-100 text-emerald-800 ring-emerald-200",
    }
    return mapping.get(status, "bg-slate-100 text-slate-700 ring-slate-200")


@register.filter
def source_badge(source: str) -> str:
    mapping = {
        "llm": "bg-violet-100 text-violet-800 ring-violet-200",
        "template": "bg-sky-100 text-sky-800 ring-sky-200",
    }
    return mapping.get(source, "bg-slate-100 text-slate-700 ring-slate-200")
