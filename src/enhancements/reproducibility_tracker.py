"""
ReproducibilityTracker module for modeling the reproducibility crisis in academic research.
"""

import json
import uuid
import random
from datetime import date, datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import math

from src.core.exceptions import ValidationError, PeerReviewError


class ReplicationOutcome(Enum):
    """Possible outcomes of replication attempts."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    INCONCLUSIVE = "inconclusive"


class QuestionablePractice(Enum):
    """Types of questionable research practices."""
    P_HACKING = "p_hacking"
    DATA_FABRICATION = "data_fabrication"
    DATA_FALSIFICATION = "data_falsification"
    CHERRY_PICKING = "cherry_picking"
    SELECTIVE_REPORTING = "selective_reporting"
    DUPLICATE_PUBLICATION = "duplicate_publication"
    PLAGIARISM = "plagiarism"
    GHOST_AUTHORSHIP = "ghost_authorship"


@dataclass
class ReplicationAttempt:
    """Represents a replication attempt of a research paper."""
    attempt_id: str
    original_paper_id: str
    replicating_researcher_id: str
    replicating_institution: str
    attempt_date: date
    outcome: ReplicationOutcome
    success_rate: float
    confidence_level: float
    methodology_similarity: float
    sample_size_ratio: float
    effect_size_ratio: Optional[float] = None
    notes: Optional[str] = None
    publication_status: Optional[str] = None
    
    def __post_init__(self):
        """Validate replication attempt data."""
        if not (0.0 <= self.success_rate <= 1.0):
            raise ValidationError("success_rate", self.success_rate, "between 0.0 and 1.0")
        if not (0.0 <= self.confidence_level <= 1.0):
            raise ValidationError("confidence_level", self.confidence_level, "between 0.0 and 1.0")
        if not (0.0 <= self.methodology_similarity <= 1.0):
            raise ValidationError("methodology_similarity", self.methodology_similarity, "between 0.0 and 1.0")
        if self.sample_size_ratio < 0:
            raise ValidationError("sample_size_ratio", self.sample_size_ratio, "non-negative")