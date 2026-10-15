from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.models import Anomaly, OrganFinding, StructuredFindings
from app.utils import humanise_organ_name, format_dimension_string

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Organ normal reference ranges
# Sources: standard radiological atlases, TotalSegmentator reference cohort
# Each entry: (volume_cc_min, volume_cc_max, hu_min_normal, hu_max_normal, label)
# hu ranges are for non-contrast unless noted.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OrganRef:
    label: str                     # Human readable organ label
    vol_min: Optional[float]       # cc - lower bound of normal volume
    vol_max: Optional[float]       # cc - upper bound of normal volume
    hu_min: float                  # Lower bound of expected mean HU
    hu_max: float                  # Upper bound of expected mean HU
    clinical_note: str = ""        # Fixed clinical context sentence


_ORGAN_REFS: Dict[str, OrganRef] = {
    # Solid abdominal organs
    "liver": OrganRef(
        label="Liver", vol_min=1000, vol_max=2200,
        hu_min=50, hu_max=70,
        clinical_note="The hepatic parenchyma is evaluated for focal lesions, fatty infiltration, and biliary dilatation."
    ),
    "spleen": OrganRef(
        label="Spleen", vol_min=60, vol_max=350,
        hu_min=40, hu_max=65,
        clinical_note="Splenic size and parenchymal homogeneity are assessed."
    ),
    "pancreas": OrganRef(
        label="Pancreas", vol_min=30, vol_max=120,
        hu_min=35, hu_max=65,
        clinical_note="The pancreatic head, body, and tail are evaluated for focal lesions and ductal dilatation."
    ),
    "gallbladder": OrganRef(
        label="Gallbladder", vol_min=5, vol_max=60,
        hu_min=0, hu_max=30,
        clinical_note="Gallbladder wall thickness and intraluminal contents are assessed."
    ),
    # Renal
    "kidney_right": OrganRef(
        label="Right Kidney", vol_min=100, vol_max=250,
        hu_min=25, hu_max=45,
        clinical_note="The renal cortex, medulla, and collecting system are evaluated. Attenuation is measured in non-contrast phase."
    ),
    "kidney_left": OrganRef(
        label="Left Kidney", vol_min=100, hu_min=25,
        vol_max=250, hu_max=45,
        clinical_note="The renal cortex, medulla, and collecting system are evaluated."
    ),
    "kidney_cyst_left": OrganRef(
        label="Left Renal Cyst", vol_min=None, vol_max=None,
        hu_min=-10, hu_max=20,
        clinical_note="Simple renal cysts measure near water density."
    ),
    "kidney_cyst_right": OrganRef(
        label="Right Renal Cyst", vol_min=None, vol_max=None,
        hu_min=-10, hu_max=20,
        clinical_note="Simple renal cysts measure near water density."
    ),
    # Adrenals
    "adrenal_gland_right": OrganRef(
        label="Right Adrenal Gland", vol_min=2, vol_max=15,
        hu_min=0, hu_max=40,
        clinical_note="Adrenal morphology and density are assessed for nodular lesions."
    ),
    "adrenal_gland_left": OrganRef(
        label="Left Adrenal Gland", vol_min=2, vol_max=15,
        hu_min=0, hu_max=40,
        clinical_note="Adrenal morphology and density are assessed for nodular lesions."
    ),
    # GI
    "stomach": OrganRef(
        label="Stomach", vol_min=None, vol_max=None,
        hu_min=30, hu_max=80,
        clinical_note="Gastric wall thickness and distension are noted."
    ),
    "small_bowel": OrganRef(
        label="Small Bowel", vol_min=None, vol_max=None,
        hu_min=20, hu_max=80,
        clinical_note="Bowel wall thickness and calibre are assessed."
    ),
    "duodenum": OrganRef(
        label="Duodenum", vol_min=None, vol_max=None,
        hu_min=20, hu_max=80,
        clinical_note="Duodenal wall and periduodenal fat are assessed."
    ),
    "colon": OrganRef(
        label="Colon", vol_min=None, vol_max=None,
        hu_min=-200, hu_max=80,
        clinical_note="Colonic wall, haustra, and intraluminal content are assessed."
    ),
    "esophagus": OrganRef(
        label="Oesophagus", vol_min=None, vol_max=None,
        hu_min=20, hu_max=60,
        clinical_note="Oesophageal wall thickness and lumen are assessed."
    ),
    # Vascular
    "aorta": OrganRef(
        label="Aorta", vol_min=None, vol_max=None,
        hu_min=35, hu_max=60,
        clinical_note="Aortic calibre is assessed throughout the visualised extent."
    ),
    "inferior_vena_cava": OrganRef(
        label="Inferior Vena Cava", vol_min=None, vol_max=None,
        hu_min=30, hu_max=60,
        clinical_note="IVC calibre and patency are assessed."
    ),
    "portal_vein_and_splenic_vein": OrganRef(
        label="Portal and Splenic Vein", vol_min=None, vol_max=None,
        hu_min=50, hu_max=90,
        clinical_note="Portal venous system patency is assessed."
    ),
    # Thoracic
    "heart": OrganRef(
        label="Heart", vol_min=None, vol_max=None,
        hu_min=40, hu_max=80,
        clinical_note="Cardiac size and pericardium are assessed on this non-dedicated cardiac study."
    ),
    "lung_upper_lobe_left": OrganRef(
        label="Left Upper Lobe", vol_min=None, vol_max=None,
        hu_min=-900, hu_max=-700,
        clinical_note="Pulmonary parenchyma is evaluated for consolidation, nodules, and interstitial changes."
    ),
    "lung_lower_lobe_left": OrganRef(
        label="Left Lower Lobe", vol_min=None, vol_max=None,
        hu_min=-900, hu_max=-700,
        clinical_note="Pulmonary parenchyma is evaluated for consolidation, nodules, and interstitial changes."
    ),
    "lung_upper_lobe_right": OrganRef(
        label="Right Upper Lobe", vol_min=None, vol_max=None,
        hu_min=-900, hu_max=-700,
        clinical_note="Pulmonary parenchyma is evaluated for consolidation, nodules, and interstitial changes."
    ),
    "lung_middle_lobe_right": OrganRef(
        label="Right Middle Lobe", vol_min=None, vol_max=None,
        hu_min=-900, hu_max=-700,
        clinical_note="Pulmonary parenchyma is evaluated for consolidation, nodules, and interstitial changes."
    ),
    "lung_lower_lobe_right": OrganRef(
        label="Right Lower Lobe", vol_min=None, vol_max=None,
        hu_min=-900, hu_max=-700,
        clinical_note="Pulmonary parenchyma is evaluated for consolidation, nodules, and interstitial changes."
    ),
    "trachea": OrganRef(
        label="Trachea", vol_min=None, vol_max=None,
        hu_min=-900, hu_max=-600,
        clinical_note="Tracheal lumen patency and wall are assessed."
    ),
    # Spinal cord
    "spinal_cord": OrganRef(
        label="Spinal Cord", vol_min=None, vol_max=None,
        hu_min=30, hu_max=60,
        clinical_note="The spinal cord and canal are assessed on this non-dedicated spine study."
    ),
    # Urinary
    "urinary_bladder": OrganRef(
        label="Urinary Bladder", vol_min=None, vol_max=None,
        hu_min=0, hu_max=25,
        clinical_note="Bladder wall thickness, intraluminal contents, and filling are assessed."
    ),
    "prostate": OrganRef(
        label="Prostate", vol_min=15, vol_max=70,
        hu_min=30, hu_max=60,
        clinical_note="Prostatic dimensions and density are assessed."
    ),
    "uterus": OrganRef(
        label="Uterus", vol_min=30, vol_max=100,
        hu_min=30, hu_max=60,
        clinical_note="Uterine size and morphology are assessed."
    ),
}

# Body system grouping for report section ordering
_ORGAN_SYSTEM_MAP: Dict[str, str] = {
    "brain": "Neurological", "spinal_cord": "Neurological",
    "thyroid_gland": "Head & Neck", "trachea": "Head & Neck",
    "heart": "Cardiovascular", "heart_myocardium": "Cardiovascular",
    "heart_atrium_left": "Cardiovascular", "heart_ventricle_left": "Cardiovascular",
    "heart_atrium_right": "Cardiovascular", "heart_ventricle_right": "Cardiovascular",
    "aorta": "Cardiovascular", "pulmonary_artery": "Cardiovascular",
    "inferior_vena_cava": "Cardiovascular", "portal_vein_and_splenic_vein": "Cardiovascular",
    "iliac_artery_left": "Cardiovascular", "iliac_artery_right": "Cardiovascular",
    "iliac_vena_left": "Cardiovascular", "iliac_vena_right": "Cardiovascular",
    "celiac_trunk": "Cardiovascular", "pericardial_effusion": "Cardiovascular",
    "lung_upper_lobe_left": "Pulmonary", "lung_lower_lobe_left": "Pulmonary",
    "lung_upper_lobe_right": "Pulmonary", "lung_middle_lobe_right": "Pulmonary",
    "lung_lower_lobe_right": "Pulmonary", "pleural_effusion_left": "Pulmonary",
    "pleural_effusion_right": "Pulmonary",
    "liver": "Hepatobiliary", "gallbladder": "Hepatobiliary", "bile_duct": "Hepatobiliary",
    "spleen": "Abdominal Solid Organs", "pancreas": "Abdominal Solid Organs",
    "adrenal_gland_right": "Abdominal Solid Organs", "adrenal_gland_left": "Abdominal Solid Organs",
    "esophagus": "Gastrointestinal", "stomach": "Gastrointestinal",
    "duodenum": "Gastrointestinal", "small_bowel": "Gastrointestinal", "colon": "Gastrointestinal",
    "urinary_bladder": "Genitourinary", "kidney_right": "Genitourinary",
    "kidney_left": "Genitourinary", "kidney_cyst_left": "Genitourinary",
    "kidney_cyst_right": "Genitourinary", "prostate": "Genitourinary",
    "uterus": "Genitourinary", "ovary_left": "Genitourinary", "ovary_right": "Genitourinary",
}

_SYSTEM_ORDER = [
    "Neurological", "Head & Neck", "Cardiovascular", "Pulmonary",
    "Hepatobiliary", "Abdominal Solid Organs", "Gastrointestinal",
    "Genitourinary", "Musculoskeletal", "Soft Tissue", "Other",
]

# Vertebrae, ribs, muscles → Musculoskeletal
for _k in list(_ORGAN_SYSTEM_MAP):
    pass
def _organ_system(organ_name: str) -> str:
    if organ_name in _ORGAN_SYSTEM_MAP:
        return _ORGAN_SYSTEM_MAP[organ_name]
    low = organ_name.lower()
    if any(x in low for x in ("vertebrae", "rib_", "sacrum", "sternum", "hip", "femur",
                               "humerus", "scapula", "clavicula", "costal")):
        return "Musculoskeletal"
    if any(x in low for x in ("gluteus", "iliopsoas", "autochthon")):
        return "Soft Tissue"
    return "Other"


def _fmt(v: float) -> str:
    if v == int(v):
        return str(int(v))
    return f"{v:.1f}"


@dataclass
class TemplateResult:
    findings_text: str
    organ_sections: List[str] = field(default_factory=list)



# ──────────────────────────────────────────────────────────────────────────────
# Clinical-indication → relevant body systems (keyword routing, no inference)
# A value of None means "all systems are relevant to this indication".
# ──────────────────────────────────────────────────────────────────────────────

_INDICATION_RULES: List[Tuple[Tuple[str, ...], Optional[set]]] = [
    (("trauma", "accident", "polytrauma", "injur", "fall "), None),
    (("fever", "sepsis", "infect", "pyrexia", "unwell"), None),
    (("follow", "staging", "restaging", "oncolog", "tumou", "tumor", "cancer",
      "malignan", "metasta", "lymphoma", "mass"), None),
    (("liver", "hepat", "jaundice", "biliary", "cholecyst", "gallbladder", "gallstone"),
     {"Hepatobiliary"}),
    (("kidney", "renal", "stone", "calcul", "colic", "flank", "haematuria", "hematuria",
      "kub", "ureter", "bladder", "urinary", "prostat"), {"Genitourinary"}),
    (("chest", "lung", "pulmon", "cough", "dyspn", "haemoptysis", "hemoptysis",
      "pneumo", "pleur"), {"Pulmonary"}),
    (("abdom", "pelvi", "epigastric", "vomit", "bowel", "appendic", "diverticul",
      "pancrea", "gastric", "stomach", "colitis"),
     {"Hepatobiliary", "Abdominal Solid Organs", "Gastrointestinal", "Genitourinary"}),
]


def _relevant_systems(indication: str):
    """Return (set_of_systems | None for all, matched?)."""
    low = f" {indication.lower()} "
    systems: set = set()
    matched = False
    for keywords, systems_for in _INDICATION_RULES:
        if any(k in low for k in keywords):
            matched = True
            if systems_for is None:
                return None, True
            systems |= systems_for
    return (systems if matched else set()), matched


def _plural(n: int, singular: str, plural: Optional[str] = None) -> str:
    return singular if n == 1 else (plural or singular + "s")


def _age_phrase(age: str) -> str:
    """'021Y' → '21-year-old'. Falls back to the raw string if unparseable."""
    import re
    m = re.fullmatch(r"\s*0*(\d+)\s*([DWMYdwmy])\s*", age or "")
    if not m:
        return age
    n, unit = int(m.group(1)), m.group(2).upper()
    word = {"D": "day", "W": "week", "M": "month", "Y": "year"}[unit]
    return f"{n}-{word}-old"


@dataclass
class _VolDeviation:
    organ: str
    label: str
    vol: float
    kind: str            # "reduced" | "enlarged"
    marked: bool
    vol_min: float
    vol_max: float
    near_absent: bool    # < 10% of lower reference bound
    truncated: bool      # organ cut by the scan boundary

    @property
    def phrase(self) -> str:
        word = ("markedly " if self.marked else "") + self.kind
        if self.truncated:
            word += ", partially outside the scanned volume"
        return (f"{self.label} {word} ({_fmt(self.vol)} cc; reference "
                f"{_fmt(self.vol_min)}–{_fmt(self.vol_max)} cc)")


class TemplateEngine:

    def generate(self, findings: StructuredFindings) -> TemplateResult:
        logger.debug("Template engine: study_id=%s, %d organs", findings.study_id, len(findings.organs))

        deviations = self._volume_deviations(findings)

        sections = [
            self._build_indication(findings),
            self._build_technique(findings),
            self._build_comparison(findings),
        ]
        findings_section, organ_sections = self._build_findings(findings)
        sections.append(findings_section)
        sections.append(self._build_impression(findings, deviations))

        full_text = "\n\n".join(s for s in sections if s.strip())
        logger.info("Template engine complete: %d chars, %d organ sections", len(full_text), len(organ_sections))
        return TemplateResult(findings_text=full_text, organ_sections=organ_sections)

    # ── Shared helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _label(organ_key: str) -> str:
        ref = _ORGAN_REFS.get(organ_key)
        return ref.label if ref else humanise_organ_name(organ_key).title()

    @staticmethod
    def _is_unassigned(organ: OrganFinding) -> bool:
        return bool(organ.unassigned) or organ.organ.lower() == "unclassified"

    @staticmethod
    def _organ_volume(organ: OrganFinding) -> Optional[float]:
        if organ.dimensions_mm:
            return organ.dimensions_mm.as_dict().get("volume_cc")
        return None

    def _volume_deviations(self, findings: StructuredFindings) -> List[_VolDeviation]:
        """Organ-volume deviations — first-class findings, separate from HU/lesion anomalies."""
        out: List[_VolDeviation] = []
        for organ in findings.organs:
            ref = _ORGAN_REFS.get(organ.organ)
            vol = self._organ_volume(organ)
            if not ref or vol is None or ref.vol_min is None or ref.vol_max is None:
                continue
            if vol < ref.vol_min:
                out.append(_VolDeviation(
                    organ.organ, ref.label, vol, "reduced", vol < 0.5 * ref.vol_min,
                    ref.vol_min, ref.vol_max, vol < 0.1 * ref.vol_min,
                    bool(organ.touches_fov_edge),
                ))
            elif vol > ref.vol_max:
                out.append(_VolDeviation(
                    organ.organ, ref.label, vol, "enlarged", vol > 1.5 * ref.vol_max,
                    ref.vol_min, ref.vol_max, False, bool(organ.touches_fov_edge),
                ))
        return out

    # ── Section builders ────────────────────────────────────────────────────

    def _build_indication(self, findings: StructuredFindings) -> str:
        modality = (findings.modality or "CT").upper()
        region = (findings.region or "").replace("_", " ").title()
        protocol = findings.protocol or ""

        lines = ["INDICATION:"]
        patient_str = ""
        parts = []
        if findings.patient_age:
            parts.append(_age_phrase(findings.patient_age))
        if findings.patient_sex:
            parts.append("male" if findings.patient_sex.upper() in ("M", "MALE") else "female")
        if parts:
            patient_str = f"{' '.join(parts)} patient. ".capitalize() if not parts[0][0].isdigit() \
                else f"{' '.join(parts)} patient. "

        description = protocol or (f"{modality} of the {region}" if region else modality)
        lines.append(f"{patient_str}{description}.")

        if findings.indication:
            lines.append(f"Clinical indication: {findings.indication}.")
        else:
            lines.append(
                "Clinical indication: not provided in the study header; "
                "findings are reported without a specific clinical question."
            )
        return "\n".join(lines)

    def _build_technique(self, findings: StructuredFindings) -> str:
        modality = (findings.modality or "CT").upper()
        tech = findings.technique or {}
        qc = findings.volume_qc or {}

        region = (findings.region or "").replace("_", " ").lower()
        region_phrase = f" of the {region}" if region else ""

        scanner = " ".join(str(x) for x in (tech.get("manufacturer"), tech.get("model")) if x)
        first = f"Volumetric {modality}{region_phrase} acquired"
        if scanner:
            first += f" on a {scanner} scanner"
        lines = ["TECHNIQUE:", first + "."]

        params = []
        if tech.get("slice_thickness_mm"):
            params.append(f"slice thickness {_fmt(float(tech['slice_thickness_mm']))} mm")
        if tech.get("kvp"):
            params.append(f"{tech['kvp']} kVp")
        if tech.get("convolution_kernel"):
            params.append(f"reconstruction kernel {tech['convolution_kernel']}")
        if tech.get("patient_position"):
            params.append(f"patient position {tech['patient_position']}")
        if params:
            lines.append("Acquisition parameters: " + ", ".join(params) + ".")
        if qc.get("slice_count") and qc.get("z_extent_mm"):
            lines.append(
                f"Coverage: {qc['slice_count']} axial slices spanning {_fmt(float(qc['z_extent_mm']))} mm."
            )

        # Contrast — from header tags when present; otherwise state it is unrecorded
        agent = (tech.get("contrast_agent") or "").strip()
        route = (tech.get("contrast_route") or "").strip()
        if agent or route:
            bits = []
            if agent:
                bits.append(agent)
            if route:
                bits.append(f"route {route}")
            if tech.get("contrast_volume_ml"):
                bits.append(f"{tech['contrast_volume_ml']} mL")
            lines.append("Contrast: " + ", ".join(bits) + " (per DICOM header).")
        else:
            lines.append(f"Contrast administration is not recorded in the DICOM header{self._contrast_hint(findings)}.")

        oral = any("oral" in (str(v) or "").lower() for v in (agent, route))
        lines.append(
            "Oral contrast: documented in the DICOM header." if oral
            else "Oral contrast: not recorded in the DICOM header."
        )

        if qc.get("warnings"):
            lines.append("Series quality notes: " + "; ".join(qc["warnings"]) + ".")
        return "\n".join(lines)

    @staticmethod
    def _contrast_hint(findings: StructuredFindings) -> str:
        """Vascular attenuation as a cue only — worded as an inference, never as fact."""
        vals = {o.organ: o.mean_density_hu for o in findings.organs if o.mean_density_hu is not None}
        vessel = [vals[k] for k in ("aorta", "portal_vein_and_splenic_vein", "inferior_vena_cava") if k in vals]
        if not vessel:
            return ""
        if max(vessel) >= 150:
            return "; vascular attenuation is consistent with a contrast-enhanced acquisition"
        if max(vessel) < 80:
            return "; vascular attenuation is more consistent with a non-contrast acquisition"
        return ""

    def _build_comparison(self, findings: StructuredFindings) -> str:
        priors = (findings.comparison or {}).get("prior_studies") or []
        lines = ["COMPARISON:"]
        if priors:
            dates = ", ".join(p.get("archived_on", "") for p in priors if p.get("archived_on"))
            lines.append(
                f"{len(priors)} prior {_plural(len(priors), 'study')} for this patient "
                f"{_plural(len(priors), 'is', 'are')} on file in the PRISM archive"
                + (f" (archived {dates})" if dates else "")
                + ". Automated image comparison between studies is not performed; "
                "radiologist comparison with the prior study is advised."
            )
        else:
            lines.append("No prior comparator study available in the PRISM archive for this patient.")
        return "\n".join(lines)

    def _build_findings(self, findings: StructuredFindings):
        system_map: Dict[str, List[OrganFinding]] = {}
        unassigned: List[OrganFinding] = []
        for organ in findings.organs:
            if self._is_unassigned(organ):
                unassigned.append(organ)
                continue
            system_map.setdefault(_organ_system(organ.organ), []).append(organ)

        lines = ["FINDINGS:"]
        organ_sections: List[str] = []

        ordered = [s for s in _SYSTEM_ORDER if s in system_map] + \
                  [s for s in system_map if s not in _SYSTEM_ORDER]
        for system in ordered:
            lines.append(f"\n{system}:")
            for organ in system_map[system]:
                section = self._build_organ_paragraph(organ)
                for ln in section.split("\n"):
                    lines.append(f"  {ln}")
                organ_sections.append(section)

        for organ in unassigned:
            lines.append("\nUnclassified Lesions (not assigned to a segmented organ):")
            section = self._build_unassigned_paragraph(organ)
            for ln in section.split("\n"):
                lines.append(f"  {ln}")
            organ_sections.append(section)

        return "\n".join(lines), organ_sections

    def _build_organ_paragraph(self, organ: OrganFinding) -> str:
        ref = _ORGAN_REFS.get(organ.organ)
        name = self._label(organ.organ)

        if organ.status == "not_visualised":
            return f"{name}: Not visualised on this study."

        heading = f"{name} ({organ.location}):" if organ.location else f"{name}:"
        sentences = [heading]

        linear = self._linear_dimensions(organ)
        if linear:
            sentences.append(f"Dimensions {linear}.")

        vol = self._organ_volume(organ)
        if vol is None and organ.mean_density_hu is not None:
            vol = getattr(organ, "volume_cc", None)

        vol_sentence = self._vol_sentence(name, vol, ref, bool(organ.touches_fov_edge))
        if vol_sentence:
            sentences.append(vol_sentence)
        if organ.touches_fov_edge:
            sentences.append(
                "The organ extends to the edge of the scanned volume, so the measured volume "
                "under-represents its true size."
            )

        hu = organ.mean_density_hu
        if hu is not None:
            sentences.append(self._density_sentence(name, hu, organ, ref))

        if ref and ref.clinical_note:
            sentences.append(ref.clinical_note)

        for anomaly in (organ.anomalies or []):
            sentences.append(self._anomaly_sentence(anomaly, name))

        sentences.extend(self._relationship_sentences(organ, name, vol, ref))

        if organ.status == "normal" and not organ.anomalies:
            sentences.append("No focal lesions, calculi, or significant abnormalities identified.")

        return " ".join(sentences)

    def _relationship_sentences(
        self, organ: OrganFinding, name: str, vol: Optional[float], ref: Optional[OrganRef]
    ) -> List[str]:
        """Organ-relationship commentary — only for abnormal organs, from measured contacts."""
        deviant = bool(ref and vol is not None and ref.vol_min is not None and ref.vol_max is not None
                       and (vol < ref.vol_min or vol > ref.vol_max))
        if not (organ.anomalies or deviant):
            return []

        out: List[str] = []
        if organ.adjacent_structures:
            names = ", ".join(self._label(n) for n in organ.adjacent_structures)
            out.append(f"The {name} is in contact with: {names}.")

        # Large lesion: mass-effect / involvement advisory (no diagnosis)
        large = [a for a in (organ.anomalies or [])
                 if a.shape and a.shape.long_axis_mm and a.shape.long_axis_mm >= 50]
        if large:
            out.append(
                "Given the size of the lesion, mass effect on and involvement of adjacent "
                "structures warrants radiologist assessment."
            )

        if _organ_system(organ.organ) == "Hepatobiliary" and organ.organ == "liver" and organ.anomalies:
            out.append(
                "Intrahepatic biliary dilatation, portal and hepatic venous patency, and vascular "
                "involvement are not assessable from the automated segmentation and require "
                "radiologist review."
            )
        return out

    def _build_unassigned_paragraph(self, organ: OrganFinding) -> str:
        """One line per lesion, each with its own anatomical position — no repeated bare label."""
        anomalies = organ.anomalies or []
        n = len(anomalies)
        lines = [
            f"Unclassified: {n} focal {_plural(n, 'density', 'densities')} could not be assigned to a "
            f"segmented organ and {_plural(n, 'requires', 'require')} radiologist anatomical correlation."
        ]
        for i, a in enumerate(anomalies, 1):
            atype = a.type.lower()
            size = ""
            vol = ""
            if a.shape:
                axes = [_fmt(v) for v in (a.shape.long_axis_mm, a.shape.short_axis_mm) if v]
                if axes:
                    size = f" measuring {' x '.join(axes)} mm"
                if a.shape.volume_cc:
                    vol = f" (volume {_fmt(a.shape.volume_cc)} cc)"
            dens = ""
            if a.density_hu:
                d = a.density_hu
                dens = f" Attenuation {_fmt(d.mean)} HU"
                if d.min is not None and d.max is not None:
                    dens += f" (range {_fmt(d.min)} to {_fmt(d.max)} HU)"
                dens += "."
            where = a.location if a.location and a.location.lower() != "unclassified" \
                else "position not determined"
            lines.append(f"{i}. {atype.capitalize()}: {where}{size}{vol}.{dens}")
        return "\n".join(lines)

    @staticmethod
    def _linear_dimensions(organ: OrganFinding) -> str:
        """Linear (mm) dimensions only; volume keys are reported in cc elsewhere."""
        if not organ.dimensions_mm:
            return ""
        dims = {
            k: v for k, v in organ.dimensions_mm.as_dict().items()
            if v is not None and not (k.lower() == "volume" or k.lower().endswith("_cc"))
        }
        return format_dimension_string(dims) if dims else ""

    def _vol_sentence(
        self, name: str, vol: Optional[float], ref: Optional[OrganRef], truncated: bool = False
    ) -> str:
        if vol is None:
            return ""
        vol_str = f"{_fmt(vol)} cc"
        if ref and ref.vol_min is not None and ref.vol_max is not None:
            if vol < ref.vol_min and truncated:
                return (f"Volume measures {vol_str}, below the reference range ({_fmt(ref.vol_min)}\u2013{_fmt(ref.vol_max)} cc); "
                        "the measured volume may be underestimated because the organ extends beyond the scan boundary.")
            if vol < ref.vol_min:
                return f"Volume measures {vol_str}, which is reduced compared to the normal range ({_fmt(ref.vol_min)}–{_fmt(ref.vol_max)} cc), suggesting atrophy or surgical reduction."
            elif vol > ref.vol_max:
                return f"Volume measures {vol_str}, which is enlarged beyond the normal range ({_fmt(ref.vol_min)}–{_fmt(ref.vol_max)} cc)."
            else:
                return f"Volume measures {vol_str}, within the normal range."
        return f"Volume measures {vol_str}."

    def _density_sentence(
        self, name: str, hu: float, organ: OrganFinding, ref: Optional[OrganRef]
    ) -> str:
        hu_str = f"{_fmt(hu)} HU"

        if ref:
            if hu < ref.hu_min:
                if hu < -100:
                    desc = f"Mean parenchymal attenuation is {hu_str}, markedly reduced, consistent with low-density content (fluid, fat, or air)."
                else:
                    desc = f"Mean parenchymal attenuation is {hu_str}, reduced compared to expected ({_fmt(ref.hu_min)}–{_fmt(ref.hu_max)} HU), raising the possibility of fatty infiltration or oedema."
            elif hu > ref.hu_max:
                if hu > 350:
                    desc = f"Mean attenuation is {hu_str}, consistent with dense mineralised/osseous tissue."
                elif hu > 200:
                    desc = f"Mean attenuation is {hu_str}, markedly hyperdense, suggesting calcification or mineralisation."
                else:
                    desc = f"Mean parenchymal attenuation is {hu_str}, mildly elevated, which may reflect contrast enhancement or haemorrhagic content."
            else:
                desc = f"Mean parenchymal attenuation is {hu_str}, within the expected range for this organ."
        else:
            desc = f"Mean attenuation measures {hu_str}."

        extra = organ.model_extra or {}
        iqr = extra.get("iqr") or extra.get("trimmed_std")
        if iqr and float(iqr) > 60 and organ.organ not in ("colon", "stomach", "small_bowel"):
            desc += f" The attenuation distribution is heterogeneous (IQR {_fmt(float(iqr))} HU), which warrants attention."

        return desc

    def _anomaly_sentence(self, anomaly: Anomaly, organ_name: str) -> str:
        atype = anomaly.type.lower()
        article = "An" if atype[0] in "aeiou" else "A"

        size_str = ""
        vol_str = ""
        if anomaly.shape:
            axes = []
            if anomaly.shape.long_axis_mm:
                axes.append(f"{_fmt(anomaly.shape.long_axis_mm)}")
            if anomaly.shape.short_axis_mm:
                axes.append(f"{_fmt(anomaly.shape.short_axis_mm)}")
            if axes:
                size_str = f" measuring {' x '.join(axes)} mm"
            if anomaly.shape.volume_cc:
                vol_str = f" (volume {_fmt(anomaly.shape.volume_cc)} cc)"
            if anomaly.shape.margin:
                atype = f"{anomaly.shape.margin} {atype}"

        density_str = ""
        if anomaly.density_hu:
            d = anomaly.density_hu
            density_str = f" Attenuation {_fmt(d.mean)} HU"
            if d.min is not None and d.max is not None:
                density_str += f" (range {_fmt(d.min)} to {_fmt(d.max)} HU)"
            density_str += "."

        location = anomaly.location or organ_name
        where = f"within the {location}"
        if anomaly.sub_location:
            where += f", in the {anomaly.sub_location} (approximate location)"
        return f"{article} {atype} is identified {where}{size_str}{vol_str}.{density_str}"

    # ── Impression ──────────────────────────────────────────────────────────

    def _build_impression(
        self, findings: StructuredFindings, deviations: Optional[List[_VolDeviation]] = None
    ) -> str:
        deviations = deviations if deviations is not None else self._volume_deviations(findings)
        anomalous = [o for o in findings.organs if o.status == "anomaly_detected" and not self._is_unassigned(o)]
        unassigned = [o for o in findings.organs if self._is_unassigned(o) and o.anomalies]
        qc = findings.volume_qc or {}

        lines = ["IMPRESSION:"]
        points: List[str] = []
        abnormal_systems: List[str] = []

        # 1) Organ-volume deviations — prominent, own category, listed first
        if deviations:
            reduced = [d for d in deviations if d.kind == "reduced"]
            enlarged = [d for d in deviations if d.kind == "enlarged"]
            if reduced:
                points.append(
                    "Reduced organ volume: " + "; ".join(d.phrase for d in reduced) + ". "
                    "Measured volumes may be underestimated if an organ extends beyond the scan "
                    "boundary or is under-segmented."
                )
            if enlarged:
                points.append(
                    "Enlarged organ volume: " + "; ".join(d.phrase for d in enlarged) + "."
                )
            near_absent = [d for d in reduced if d.near_absent]
            if near_absent:
                names = ", ".join(d.label for d in near_absent)
                partial = [d.label for d in near_absent if d.truncated]
                caveat = (
                    f"Near-absent measured volume ({names}) — the measured volume may be underestimated "
                    "if the organ extends beyond the scan boundary or is under-segmented. Confirm on the "
                    "source images whether this reflects true absence/atrophy, surgical removal, a "
                    "segmentation limitation, or incomplete scan coverage."
                )
                if partial:
                    caveat += f" {', '.join(partial)} extend to the edge of the scanned volume."
                points.append(caveat)
            abnormal_systems += [_organ_system(d.organ) for d in deviations]

        # 2) Focal lesions, grouped by body system (not one templated line per organ)
        by_system: Dict[str, List[OrganFinding]] = {}
        for organ in anomalous:
            by_system.setdefault(_organ_system(organ.organ), []).append(organ)

        for system in [s for s in _SYSTEM_ORDER if s in by_system] + \
                      [s for s in by_system if s not in _SYSTEM_ORDER]:
            organs = by_system[system]
            lesions = [a for o in organs for a in (o.anomalies or [])]
            abnormal_systems.append(system)
            if not lesions:
                points.append(f"{system}: density deviation involving "
                              + ", ".join(self._label(o.organ) for o in organs) + ".")
                continue

            n = len(lesions)
            where = ", ".join(
                f"{self._label(o.organ)} ({len(o.anomalies)})" for o in organs if o.anomalies
            )
            sides = {("left" if o.organ.endswith("_left") else "right" if o.organ.endswith("_right") else "")
                     for o in organs if o.anomalies}
            bilateral = {"left", "right"} <= sides
            multi = "multiple " if n > 1 else ""
            types = sorted({a.type.lower() for a in lesions})
            largest = [a.shape.long_axis_mm for a in lesions if a.shape and a.shape.long_axis_mm]
            tail = f"; largest {_fmt(max(largest))} mm" if largest else ""
            points.append(
                f"{system}: {multi}{', '.join(types)} ({n} {_plural(n, 'lesion')}"
                f"{', bilateral' if bilateral else ''}) involving {where}{tail}."
            )

        # 3) Unassigned lesions
        if unassigned:
            n = sum(len(o.anomalies) for o in unassigned)
            points.append(
                f"{n} additional focal {_plural(n, 'density', 'densities')} not assigned to a segmented organ "
                "(individual positions listed under Findings) — radiologist anatomical correlation required."
            )

        # 4) Synthesis across systems
        distinct = list(dict.fromkeys(abnormal_systems))
        if len(distinct) >= 2:
            points.append(
                f"Abnormalities span {len(distinct)} body systems ({', '.join(distinct)})."
            )

        # 5) Tie-back to the clinical indication
        if findings.indication and (points or True):
            relevant, matched = _relevant_systems(findings.indication)
            if matched:
                if relevant is None:
                    if distinct:
                        points.append(
                            f"Relation to the stated indication ({findings.indication}): "
                            f"abnormalities are reported in {', '.join(distinct)}."
                        )
                else:
                    hit = [s for s in distinct if s in relevant]
                    if hit:
                        points.append(
                            f"Relation to the stated indication ({findings.indication}): abnormalities "
                            f"involve the relevant system(s): {', '.join(hit)}."
                        )
                    elif distinct:
                        points.append(
                            f"Relation to the stated indication ({findings.indication}): no abnormality "
                            "identified in the systems most relevant to it; abnormalities elsewhere are "
                            "listed above."
                        )

        # 6) Data-quality caveat
        if qc.get("warnings"):
            points.append("Series quality caveats: " + "; ".join(qc["warnings"]) + ".")

        if not points:
            lines.append("No acute or significant findings identified on this study. All imaged structures appear within normal limits.")
        else:
            for i, p in enumerate(points, 1):
                lines.append(f"{i}. {p}")
            lines.append(
                "These findings warrant further clinical evaluation; radiological correlation "
                "with clinical history is recommended."
            )

        lines.append("\nRECOMMENDATIONS:")
        if not points:
            lines.append("Routine clinical follow-up as indicated.")
        else:
            if deviations:
                lines.append(
                    "Confirm organ volumes on the source images and review scan coverage before "
                    "clinical interpretation of the volume findings."
                )
            lines.append("Correlation with clinical findings and laboratory investigations is advised.")
            lines.append("Radiologist review of the study and clinical correlation with referring clinician is recommended.")

        return "\n".join(lines)
