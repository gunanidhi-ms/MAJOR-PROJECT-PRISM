"""
utils.py
--------
Shared utility helpers used across Phase 3.

Kept deliberately small; only truly cross-cutting concerns live here.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Set

from app.config import get_settings


# ====================================================================== #
#  Logging
# ====================================================================== #

def setup_logging() -> logging.Logger:
    """
    Configure and return the root logger for Phase 3.

    Call once from main.py; every other module uses
    ``logging.getLogger(__name__)``.
    """
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # Ensure logs directory exists if a file path is provided
    if settings.log_file:
        log_dir = os.path.dirname(settings.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(settings.log_file)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("phase3")


# ====================================================================== #
#  Organ / anatomy helpers
# ====================================================================== #

def humanise_organ_name(raw: str) -> str:
    """
    Convert a snake_case organ key to a title-cased human-readable string.

    Examples
    --------
    >>> humanise_organ_name("left_kidney")
    'left kidney'
    >>> humanise_organ_name("urinary_bladder")
    'urinary bladder'
    >>> humanise_organ_name("Liver")
    'liver'
    """
    return raw.replace("_", " ").strip().lower()


# ====================================================================== #
#  Number / measurement extraction  (used by validator)
# ====================================================================== #

# Matches integers and decimals, including those embedded in measurements
# like "22.4 × 14.1 mm" or "98 × 45 × 40 mm", and negative numbers like "-10".
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_numbers(text: str) -> Set[str]:
    """
    Return the set of all numeric strings found in *text*.

    The strings are returned as-is (not converted to float) so that
    trailing-zero differences (e.g. "18" vs "18.0") are caught by the
    validator as mismatches – the LLM should never reformat numbers.

    Parameters
    ----------
    text : str
        Any free-form radiology text.

    Returns
    -------
    set[str]
        Deduplicated set of numeric strings.
    """
    return set(_NUMBER_RE.findall(text))


def extract_words(text: str) -> Set[str]:
    """
    Return the set of lower-cased alphabetic words found in *text*.

    Used by the validator to check organ names and location descriptors.

    Parameters
    ----------
    text : str

    Returns
    -------
    set[str]
    """
    return set(re.findall(r"[a-zA-Z]+", text.lower()))


def format_dimension_string(dims: dict) -> str:
    """
    Format a dimensions dict as a '× '-joined measurement string.

    Parameters
    ----------
    dims : dict
        Mapping of dimension name → numeric value.
        Keys like ``length``, ``width``, ``ap``, ``diameter`` are common.

    Returns
    -------
    str
        E.g. ``"98 × 45 × 40 mm"``
    """
    # Preferred ordering for common keys; unknowns appended after.
    preferred_order = ["length", "width", "ap", "diameter", "height", "depth"]
    ordered_keys = [k for k in preferred_order if k in dims]
    ordered_keys += [k for k in dims if k not in preferred_order]

    values = [str(dims[k]) for k in ordered_keys if dims[k] is not None]
    return " × ".join(values) + " mm" if values else ""
