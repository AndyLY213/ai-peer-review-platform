"""Data structures describing coordinator events for integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class EventType(str, Enum):
    RESEARCHER_REGISTERED = "researcher_registered"
    PAPER_SUBMITTED = "paper_submitted"
    REVIEW_SUBMITTED = "review_submitted"
    INVITATION_DECISION = "invitation_decision"
    TOKEN_TRANSACTION = "token_transaction"


@dataclass
class ResearcherProfile:
    researcher_id: str
    name: str
    specialty: str
    personality: Optional[str]
    career_stage: str
    bias_profile: Dict[str, float] = field(default_factory=dict)
    strategic_profile: Dict[str, float] = field(default_factory=dict)


@dataclass
class PaperSubmission:
    paper_id: str
    title: str
    field: str
    author_id: str
    venue_id: Optional[str]
    priority_score: float = 0.0


@dataclass
class ReviewDecision:
    review_id: str
    paper_id: str
    reviewer_id: str
    scores: Dict[str, float]
    recommendation: str
    confidence: float
    thought_process: str


@dataclass
class InvitationDecision:
    paper_id: str
    reviewer_id: str
    author_id: str
    token_offer: int
    accepted: bool
    thought_process: str


@dataclass
class TokenEvent:
    transaction_id: str
    from_agent: Optional[str]
    to_agent: Optional[str]
    amount: int
    reason: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class SimulationEvent:
    event_type: EventType
    timestamp: datetime
    payload: Dict[str, object]
