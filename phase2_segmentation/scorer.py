"""
scorer.py — Package 7: Confidence Fusion & Emergency Scoring

Takes the merged, filtered candidate list from Package 6 and computes
three missing [SCORE] fields for every non-suppressed candidate:
  - fused_confidence: [0.0, 1.0] synthesizing 6 evidence sources
  - gate: "AUTO_CONFIRMED" or "MANUAL_REVIEW"
  - emergency_score: 0-100 priority score

Lifecycle annotations from schemas/candidate.py:
  [SCORE]  fused_confidence, gate, emergency_score
"""

import math
from schemas.candidate import Candidate

CONFIDENCE_GATE_THRESHOLD = 0.70

def score_candidates(candidates: list, region: str) -> list:
    """
    In-place: computes [SCORE] fields for all non-suppressed candidates.
    Suppressed candidates are left untouched (0.0, "", 0).

    Args:
        candidates: List of Candidate objects from candidate_merger.py
        region: Scan body region (e.g. "cardiothoracic")

    Returns:
        The same list (identity return).
    """
    for c in candidates:
        if c.suppressed:
            continue
            
        # Stage 1: Fused Confidence Calculation
        fused_conf, evidence_count = _compute_fused_confidence(c)
        
        # Evidence count gate
        if evidence_count < 2:
            fused_conf = min(fused_conf, 0.55)
            
        c.fused_confidence = round(fused_conf, 4)
        
        # Stage 2: Gate Assignment
        if c.organ_label == "unclassified":
            c.gate = "MANUAL_REVIEW"
        elif c.fused_confidence >= CONFIDENCE_GATE_THRESHOLD:
            c.gate = "AUTO_CONFIRMED"
        else:
            c.gate = "MANUAL_REVIEW"
            
        # Stage 3: Emergency Score
        c.emergency_score = _compute_emergency_score(c)

    return candidates

def _compute_fused_confidence(c: Candidate) -> tuple[float, int]:
    """
    Calculates weighted confidence and number of contributing evidence sources.
    Returns (fused_confidence, evidence_count).
    """
    score = 0.0
    evidence_count = 0
    
    # 1. phase1_confidence (Path A provenance) - 0.30 weight
    if c.phase1_confidence > 0:
        score += c.phase1_confidence * 0.30
        evidence_count += 1
        
    # 2. organ_local_zscore (Organ deviation) - 0.25 weight
    if c.organ_local_zscore > 0:
        z_norm = min(c.organ_local_zscore / 6.0, 1.0)
        score += z_norm * 0.25
        evidence_count += 1
        
    # 3. corroborated flag - 0.20 weight
    if c.corroborated:
        score += 1.0 * 0.20
        evidence_count += 1
        
    # 4. persistence_ok flag - 0.10 weight
    if c.persistence_ok:
        score += 1.0 * 0.10
        evidence_count += 1
        
    # 5. shape.sphericity - 0.10 weight
    if c.shape.sphericity > 0:
        score += c.shape.sphericity * 0.10
        evidence_count += 1
        
    # 6. density_hu.std (heterogeneity) - 0.05 weight
    if c.density_hu.std > 0:
        std_norm = min(c.density_hu.std / 100.0, 1.0)
        score += std_norm * 0.05
        evidence_count += 1
        
    return score, evidence_count

def _compute_emergency_score(c: Candidate) -> int:
    """
    Calculates 0-100 emergency priority score.
    """
    points = c.fused_confidence * 40.0
    
    z_norm = min(c.organ_local_zscore / 6.0, 1.0)
    points += z_norm * 25.0
    
    if c.corroborated:
        points += 20.0
        
    vol_norm = min(c.shape.volume_cc / 50.0, 1.0)
    points += vol_norm * 15.0
    
    score = int(round(points))
    return min(max(score, 0), 100)
