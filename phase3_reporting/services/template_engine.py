"""
template_engine.py
------------------
Deterministic, AI-free radiology findings generator.

This is the SOURCE OF TRUTH for all factual content in the report.
Every number, measurement, organ name, and location that appears in the
final report originates here - never from the LLM.

Design constraints
~~~~~~~~~~~~~~~~~~
* No AI / LLM calls.
* No disease-specific or organ-specific hardcoded logic.
* Supports an arbitrary number of organs and anomalies per organ.
* Returns a structured ``TemplateResult`` so callers can extract both
  the raw text and the list of key terms used for validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

from app.models import Anomaly, AnomalyShape, DensityHU, OrganFinding, StructuredFindings
from app.utils import format_dimension_string, humanise_organ_name

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Module-level helpers
# ====================================================================== #


def _fmt(value: float) -> str:
    """
    Format a float cleanly: removes unnecessary trailing '.0'.

    Examples
    --------
    >>> _fmt(18.0)
    '18'
    """
    if value == int(value):
        return str(int(value))
    return str(value)


# ====================================================================== #
#  Result dataclass
# ====================================================================== #


@dataclass
class TemplateResult:
    """
    Container returned by the template engine.

    Attributes
    ----------
    findings_text : str
        Full findings section as a single multi-paragraph string.
    organ_sections : list[str]
        Individual paragraph for each organ (for debugging / testing).
    """
    findings_text: str
    organ_sections: List[str] = field(default_factory=list)


# ====================================================================== #
#  Template Engine
# ====================================================================== #


class TemplateEngine:
    """
    Converts a ``StructuredFindings`` object into a deterministic
    radiology findings text.
    """

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def generate(self, findings: StructuredFindings) -> TemplateResult:
        """
        Generate the complete findings section.
        """
        logger.debug(
            "Template engine generating findings for study_id=%s (%d organs)",
            findings.study_id,
            len(findings.organs),
        )

        header = self._build_header(findings)
        organ_sections: List[str] = []

        for organ in findings.organs:
            section = self._build_organ_section(organ)
            organ_sections.append(section)
            logger.debug("Generated section for organ=%s", organ.organ)

        full_text = header + "\n\n" + "\n\n".join(organ_sections)
        full_text = full_text.strip()

        result = TemplateResult(
            findings_text=full_text,
            organ_sections=organ_sections,
        )

        logger.info(
            "Template engine completed: study_id=%s, %d chars",
            findings.study_id,
            len(full_text),
        )
        return result

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    def _build_header(self, findings: StructuredFindings) -> str:
        parts = ["FINDINGS"]
        if findings.protocol:
            parts.append(findings.protocol.upper())
        elif findings.modality:
            parts.append(findings.modality.upper())
        return " - ".join(parts)

    # ------------------------------------------------------------------ #

    def _build_organ_section(self, organ: OrganFinding) -> str:
        human_name = humanise_organ_name(organ.organ)

        if organ.status == "not_visualised":
            return self._not_visualised_text(human_name, organ)

        lines: List[str] = []

        if organ.status == "normal":
            lines.append(self._normal_summary(human_name, organ))
        else:
            lines.append(self._anomaly_summary(human_name, organ))

        dim_sentence = self._dimension_sentence(organ)
        if dim_sentence:
            lines.append(dim_sentence)

        density_sentence = self._organ_density_sentence(organ)
        if density_sentence:
            lines.append(density_sentence)

        if organ.anomalies:
            for anomaly in organ.anomalies:
                anomaly_block = self._build_anomaly_block(human_name, anomaly)
                lines.append(anomaly_block)

        return " ".join(lines)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _not_visualised_text(human_name: str, organ: OrganFinding) -> str:
        location_clause = ""
        if organ.location:
            location_clause = f" in the {organ.location}"
        return (
            f"The {human_name}{location_clause} is not visualised on this study."
        )

    @staticmethod
    def _normal_summary(human_name: str, organ: OrganFinding) -> str:
        location_clause = ""
        if organ.location:
            location_clause = f", located in the {organ.location},"
        return (
            f"The {human_name}{location_clause} appears normal in size "
            f"and morphology."
        )

    @staticmethod
    def _anomaly_summary(human_name: str, organ: OrganFinding) -> str:
        location_clause = ""
        if organ.location:
            location_clause = f" in the {organ.location}"
        n = len(organ.anomalies) if organ.anomalies else 1
        anomaly_word = "anomaly" if n == 1 else "anomalies"
        return (
            f"The {human_name}{location_clause} demonstrates "
            f"{n} {anomaly_word} as detailed below."
        )

    @staticmethod
    def _dimension_sentence(organ: OrganFinding) -> str:
        if not organ.dimensions_mm:
            return ""
        dims = organ.dimensions_mm.as_dict()
        dims = {k: v for k, v in dims.items() if v is not None}
        if not dims:
            return ""
        dim_str = format_dimension_string(dims)
        return f"Dimensions measure {dim_str}."

    @staticmethod
    def _organ_density_sentence(organ: OrganFinding) -> str:
        if organ.mean_density_hu is None:
            return ""
        return f"Mean parenchymal density measures {_fmt(organ.mean_density_hu)} HU."

    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_anomaly_block(organ_name: str, anomaly: Anomaly) -> str:
        parts: List[str] = []

        margin_str = ""
        if anomaly.shape and anomaly.shape.margin:
            margin_str = f"{anomaly.shape.margin} "

        atype = anomaly.type.lower()
        article = "An" if atype[0] in "aeiou" else "A"

        location_clause = f"in the {anomaly.location} of the {organ_name}"

        measurement_str = ""
        volume_str = ""

        if anomaly.shape:
            s = anomaly.shape
            axes: List[str] = []
            if s.long_axis_mm is not None:
                axes.append(_fmt(s.long_axis_mm))
            if s.short_axis_mm is not None:
                axes.append(_fmt(s.short_axis_mm))

            if axes:
                measurement_str = " x ".join(axes) + " mm"

            if s.volume_cc is not None:
                volume_str = f" (volume {_fmt(s.volume_cc)} cc)"

        sphericity_str = ""
        if anomaly.shape and anomaly.shape.sphericity is not None:
            sphericity_str = (
                f" Sphericity index: {_fmt(anomaly.shape.sphericity)}."
            )

        if measurement_str:
            main_sentence = (
                f"{article} {margin_str}{atype} is identified {location_clause} "
                f"measuring {measurement_str}{volume_str}."
            )
        else:
            main_sentence = (
                f"{article} {margin_str}{atype} is identified {location_clause}."
            )

        parts.append(main_sentence)

        if sphericity_str:
            parts.append(sphericity_str)

        if anomaly.density_hu:
            d = anomaly.density_hu
            density_parts: List[str] = [f"Mean density is {_fmt(d.mean)} HU"]
            extras: List[str] = []
            if d.min is not None and d.max is not None:
                extras.append(f"range: {_fmt(d.min)} to {_fmt(d.max)} HU")
            if d.std is not None:
                extras.append(f"SD {_fmt(d.std)} HU")
            if extras:
                density_parts.append(f"({'; '.join(extras)})")
            parts.append(" ".join(density_parts) + ".")

        if anomaly.confidence is not None:
            pct = round(anomaly.confidence * 100)
            parts.append(f"Detection confidence: {pct}%.")

        return " ".join(parts)
