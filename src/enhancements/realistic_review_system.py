"""
Enhanced Review System for Realistic Peer Review Simulation.

This module implements a more realistic review system with:
- Multi-dimensional scoring
- Review quality assessment
- Venue-specific standards
- Temporal dynamics
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import random

class ReviewDecision(Enum):
    """Possible review decisions."""
    ACCEPT = "accept"
    MINOR_REVISION = "minor_revision"
    MAJOR_REVISION = "major_revision"
    REJECT = "reject"

class VenueType(Enum):
    """Types of publication venues."""
    TOP_CONFERENCE = "top_conference"      # 5% acceptance rate
    MID_CONFERENCE = "mid_conference"      # 25% acceptance rate
    LOW_CONFERENCE = "low_conference"      # 50% acceptance rate
    TOP_JOURNAL = "top_journal"            # 2% acceptance rate
    SPECIALIZED_JOURNAL = "specialized_journal"  # 15% acceptance rate
    GENERAL_JOURNAL = "general_journal"    # 40% acceptance rate

class ResearcherLevel(Enum):
    """Academic hierarchy levels."""
    GRADUATE_STUDENT = "graduate_student"
    POSTDOC = "postdoc"
    ASSISTANT_PROF = "assistant_prof"
    ASSOCIATE_PROF = "associate_prof"
    FULL_PROF = "full_prof"
    EMERITUS = "emeritus"

@dataclass
class ReviewCriteria:
    """Multi-dimensional review scoring."""
    novelty: int           # 1-10: How novel/original is the work?
    technical_quality: int # 1-10: Technical soundness and rigor
    clarity: int          # 1-10: Writing quality and presentation
    significance: int     # 1-10: Impact and importance to field
    reproducibility: int  # 1-10: Can results be reproduced?
    related_work: int     # 1-10: Coverage of related work
    
    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        weights = {
            'novelty': 0.25,
            'technical_quality': 0.25,
            'significance': 0.20,
            'clarity': 0.15,
            'reproducibility': 0.10,
            'related_work': 0.05
        }
        
        return (
            self.novelty * weights['novelty'] +
            self.technical_quality * weights['technical_quality'] +
            self.significance * weights['significance'] +
            self.clarity * weights['clarity'] +
            self.reproducibility * weights['reproducibility'] +
            self.related_work * weights['related_work']
        )

@dataclass
class DetailedReview:
    """Comprehensive review structure."""
    reviewer_id: str
    paper_id: str
    venue_id: str
    
    # Scoring
    criteria: ReviewCriteria
    confidence: int  # 1-5: Reviewer's confidence in assessment
    
    # Decision
    recommendation: ReviewDecision
    
    # Detailed feedback
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    detailed_comments: str
    questions_for_authors: List[str]
    
    # Metadata
    review_length: int  # Word count
    time_spent: int     # Minutes spent reviewing
    timestamp: datetime
    is_late: bool       # Was review submitted after deadline?
    
    def meets_quality_standards(self, venue_type: VenueType) -> bool:
        """Check if review meets venue quality standards."""
        min_lengths = {
            VenueType.TOP_CONFERENCE: 800,
            VenueType.MID_CONFERENCE: 500,
            VenueType.LOW_CONFERENCE: 300,
            VenueType.TOP_JOURNAL: 1000,
            VenueType.SPECIALIZED_JOURNAL: 600,
            VenueType.GENERAL_JOURNAL: 400
        }
        
        return (
            self.review_length >= min_lengths.get(venue_type, 300) and
            len(self.strengths) >= 2 and
            len(self.weaknesses) >= 1 and
            self.confidence >= 3
        )

@dataclass
class Venue:
    """Publication venue with specific characteristics."""
    id: str
    name: str
    venue_type: VenueType
    field: str
    
    # Venue characteristics
    acceptance_rate: float
    prestige_score: int  # 1-10
    review_deadline_weeks: int
    min_reviewers: int
    
    # Review standards
    min_review_length: int
    requires_detailed_scores: bool
    allows_borderline_reviews: bool
    
    def get_acceptance_threshold(self) -> float:
        """Get score threshold for acceptance."""
        thresholds = {
            VenueType.TOP_CONFERENCE: 8.0,
            VenueType.MID_CONFERENCE: 6.5,
            VenueType.LOW_CONFERENCE: 5.0,
            VenueType.TOP_JOURNAL: 8.5,
            VenueType.SPECIALIZED_JOURNAL: 7.0,
            VenueType.GENERAL_JOURNAL: 5.5
        }
        return thresholds.get(self.venue_type, 6.0)

@dataclass
class EnhancedResearcher:
    """Researcher with realistic attributes."""
    id: str
    name: str
    level: ResearcherLevel
    institution_tier: int  # 1-3 (1 = top tier)
    specialty: str
    
    # Reputation metrics
    h_index: int
    total_citations: int
    years_active: int
    
    # Review behavior
    review_quality_score: float  # 0-1
    review_speed: float         # 0-1 (1 = always on time)
    review_harshness: float     # 0-1 (1 = very harsh)
    
    # Biases
    prestige_bias: float        # 0-1 (favor prestigious authors)
    novelty_bias: float         # 0-1 (favor novel work)
    confirmation_bias: float    # 0-1 (favor confirming work)
    
    # Availability
    current_workload: int       # Number of pending reviews
    max_reviews_per_month: int
    is_available: bool
    
    def get_reputation_multiplier(self) -> float:
        """Get reputation-based influence multiplier."""
        level_multipliers = {
            ResearcherLevel.GRADUATE_STUDENT: 0.5,
            ResearcherLevel.POSTDOC: 0.7,
            ResearcherLevel.ASSISTANT_PROF: 1.0,
            ResearcherLevel.ASSOCIATE_PROF: 1.3,
            ResearcherLevel.FULL_PROF: 1.5,
            ResearcherLevel.EMERITUS: 1.2
        }
        
        base = level_multipliers.get(self.level, 1.0)
        institution_bonus = (4 - self.institution_tier) * 0.1
        h_index_bonus = min(self.h_index / 50, 0.5)
        
        return base + institution_bonus + h_index_bonus
    
    def generate_biased_review(self, paper: Dict, author_reputation: float) -> DetailedReview:
        """Generate a review with realistic biases."""
        # Base scores
        base_scores = {
            'novelty': random.randint(4, 8),
            'technical_quality': random.randint(5, 9),
            'clarity': random.randint(4, 8),
            'significance': random.randint(3, 7),
            'reproducibility': random.randint(4, 7),
            'related_work': random.randint(5, 8)
        }
        
        # Apply biases
        if self.prestige_bias > 0.5 and author_reputation > 0.7:
            # Boost scores for prestigious authors
            for key in base_scores:
                base_scores[key] = min(10, base_scores[key] + 1)
        
        if self.novelty_bias > 0.7:
            # Heavily weight novelty
            base_scores['novelty'] = min(10, base_scores['novelty'] + 1)
            base_scores['significance'] = min(10, base_scores['significance'] + 1)
        
        # Apply harshness
        if self.review_harshness > 0.6:
            for key in base_scores:
                base_scores[key] = max(1, base_scores[key] - 1)
        
        criteria = ReviewCriteria(**base_scores)
        
        # Determine recommendation based on overall score
        overall = criteria.overall_score()
        if overall >= 8.0:
            recommendation = ReviewDecision.ACCEPT
        elif overall >= 6.5:
            recommendation = ReviewDecision.MINOR_REVISION
        elif overall >= 4.5:
            recommendation = ReviewDecision.MAJOR_REVISION
        else:
            recommendation = ReviewDecision.REJECT
        
        # Generate review content based on scores and biases
        strengths = self._generate_strengths(criteria)
        weaknesses = self._generate_weaknesses(criteria)
        
        return DetailedReview(
            reviewer_id=self.id,
            paper_id=paper['id'],
            venue_id="",  # To be set by venue
            criteria=criteria,
            confidence=random.randint(3, 5),
            recommendation=recommendation,
            summary=f"This paper presents work in {paper.get('field', 'the field')}...",
            strengths=strengths,
            weaknesses=weaknesses,
            detailed_comments=self._generate_detailed_comments(criteria, recommendation),
            questions_for_authors=self._generate_questions(criteria),
            review_length=random.randint(400, 1200),
            time_spent=random.randint(60, 240),
            timestamp=datetime.now(),
            is_late=random.random() > self.review_speed
        )
    
    def _generate_strengths(self, criteria: ReviewCriteria) -> List[str]:
        """Generate strengths based on scores."""
        strengths = []
        if criteria.novelty >= 7:
            strengths.append("The approach is novel and innovative")
        if criteria.technical_quality >= 7:
            strengths.append("The technical execution is sound")
        if criteria.clarity >= 7:
            strengths.append("The paper is well-written and clear")
        if criteria.significance >= 7:
            strengths.append("The work addresses an important problem")
        
        # Ensure at least 2 strengths
        if len(strengths) < 2:
            strengths.extend([
                "The experimental setup is reasonable",
                "The related work section is adequate"
            ])
        
        return strengths[:4]  # Limit to 4 strengths
    
    def _generate_weaknesses(self, criteria: ReviewCriteria) -> List[str]:
        """Generate weaknesses based on scores."""
        weaknesses = []
        if criteria.novelty <= 4:
            weaknesses.append("The novelty is limited")
        if criteria.technical_quality <= 4:
            weaknesses.append("Technical issues need to be addressed")
        if criteria.clarity <= 4:
            weaknesses.append("The presentation could be improved")
        if criteria.reproducibility <= 4:
            weaknesses.append("Reproducibility concerns")
        
        # Ensure at least 1 weakness
        if len(weaknesses) == 0:
            weaknesses.append("Minor presentation issues")
        
        return weaknesses[:3]  # Limit to 3 weaknesses
    
    def _generate_detailed_comments(self, criteria: ReviewCriteria, recommendation: ReviewDecision) -> str:
        """Generate detailed comments based on review."""
        if recommendation == ReviewDecision.ACCEPT:
            return "This is a solid contribution that merits publication..."
        elif recommendation == ReviewDecision.MINOR_REVISION:
            return "This work has merit but requires minor revisions..."
        elif recommendation == ReviewDecision.MAJOR_REVISION:
            return "While the core idea has potential, significant revisions are needed..."
        else:
            return "Unfortunately, this work has fundamental issues that prevent publication..."
    
    def _generate_questions(self, criteria: ReviewCriteria) -> List[str]:
        """Generate questions for authors."""
        questions = []
        if criteria.technical_quality <= 6:
            questions.append("Can you provide more details on the methodology?")
        if criteria.reproducibility <= 5:
            questions.append("Will you make the code and data available?")
        if criteria.related_work <= 5:
            questions.append("How does this compare to recent work by X et al.?")
        
        return questions

class RealisticReviewSystem:
    """Enhanced review system with realistic dynamics."""
    
    def __init__(self):
        self.venues: Dict[str, Venue] = {}
        self.researchers: Dict[str, EnhancedResearcher] = {}
        self.active_submissions: Dict[str, Dict] = {}
        self.review_history: List[DetailedReview] = []
    
    def add_venue(self, venue: Venue):
        """Add a publication venue."""
        self.venues[venue.id] = venue
    
    def add_researcher(self, researcher: EnhancedResearcher):
        """Add a researcher to the system."""
        self.researchers[researcher.id] = researcher
    
    def submit_paper(self, paper: Dict, venue_id: str, author_ids: List[str]) -> str:
        """Submit a paper to a venue."""
        submission_id = f"sub_{len(self.active_submissions):04d}"
        
        self.active_submissions[submission_id] = {
            'paper': paper,
            'venue_id': venue_id,
            'author_ids': author_ids,
            'submission_date': datetime.now(),
            'status': 'under_review',
            'assigned_reviewers': [],
            'reviews': [],
            'decision': None
        }
        
        # Assign reviewers
        self._assign_reviewers(submission_id)
        
        return submission_id
    
    def _assign_reviewers(self, submission_id: str):
        """Assign reviewers to a submission."""
        submission = self.active_submissions[submission_id]
        venue = self.venues[submission['venue_id']]
        paper = submission['paper']
        
        # Find eligible reviewers
        eligible_reviewers = []
        for researcher_id, researcher in self.researchers.items():
            if (researcher_id not in submission['author_ids'] and
                researcher.specialty == paper.get('field', '') and
                researcher.is_available and
                researcher.current_workload < researcher.max_reviews_per_month):
                eligible_reviewers.append(researcher_id)
        
        # Select reviewers (prefer higher reputation for top venues)
        if venue.venue_type in [VenueType.TOP_CONFERENCE, VenueType.TOP_JOURNAL]:
            eligible_reviewers.sort(
                key=lambda r_id: self.researchers[r_id].get_reputation_multiplier(),
                reverse=True
            )
        
        # Assign required number of reviewers
        num_reviewers = min(venue.min_reviewers, len(eligible_reviewers))
        selected_reviewers = eligible_reviewers[:num_reviewers]
        
        submission['assigned_reviewers'] = selected_reviewers
        
        # Update reviewer workloads
        for reviewer_id in selected_reviewers:
            self.researchers[reviewer_id].current_workload += 1
    
    def generate_reviews(self, submission_id: str) -> List[DetailedReview]:
        """Generate reviews for a submission."""
        submission = self.active_submissions[submission_id]
        venue = self.venues[submission['venue_id']]
        paper = submission['paper']
        
        reviews = []
        for reviewer_id in submission['assigned_reviewers']:
            reviewer = self.researchers[reviewer_id]
            
            # Calculate author reputation
            author_reputation = sum(
                self.researchers[author_id].get_reputation_multiplier()
                for author_id in submission['author_ids']
                if author_id in self.researchers
            ) / len(submission['author_ids'])
            
            # Generate review
            review = reviewer.generate_biased_review(paper, author_reputation)
            review.venue_id = venue.id
            
            # Check if review meets venue standards
            if not review.meets_quality_standards(venue.venue_type):
                # Low-quality review - reduce reviewer's quality score
                reviewer.review_quality_score *= 0.95
            
            reviews.append(review)
            self.review_history.append(review)
        
        submission['reviews'] = reviews
        return reviews
    
    def make_decision(self, submission_id: str) -> ReviewDecision:
        """Make publication decision based on reviews."""
        submission = self.active_submissions[submission_id]
        venue = self.venues[submission['venue_id']]
        reviews = submission['reviews']
        
        if not reviews:
            return ReviewDecision.REJECT
        
        # Calculate weighted average score
        total_score = 0
        total_weight = 0
        
        for review in reviews:
            reviewer = self.researchers[review.reviewer_id]
            weight = reviewer.get_reputation_multiplier() * reviewer.review_quality_score
            total_score += review.criteria.overall_score() * weight
            total_weight += weight
        
        avg_score = total_score / total_weight if total_weight > 0 else 0
        
        # Apply venue-specific threshold
        threshold = venue.get_acceptance_threshold()
        
        if avg_score >= threshold:
            decision = ReviewDecision.ACCEPT
        elif avg_score >= threshold - 1.5:
            decision = ReviewDecision.MINOR_REVISION
        elif avg_score >= threshold - 3.0:
            decision = ReviewDecision.MAJOR_REVISION
        else:
            decision = ReviewDecision.REJECT
        
        submission['decision'] = decision
        submission['status'] = 'decided'
        
        # Update researcher workloads
        for reviewer_id in submission['assigned_reviewers']:
            self.researchers[reviewer_id].current_workload -= 1
        
        return decision
    
    def get_venue_statistics(self, venue_id: str) -> Dict:
        """Get statistics for a venue."""
        venue_reviews = [r for r in self.review_history if r.venue_id == venue_id]
        
        if not venue_reviews:
            return {}
        
        return {
            'total_reviews': len(venue_reviews),
            'avg_review_length': sum(r.review_length for r in venue_reviews) / len(venue_reviews),
            'late_review_rate': sum(1 for r in venue_reviews if r.is_late) / len(venue_reviews),
            'avg_confidence': sum(r.confidence for r in venue_reviews) / len(venue_reviews),
            'decision_distribution': {
                decision.value: sum(1 for r in venue_reviews if r.recommendation == decision)
                for decision in ReviewDecision
            }
        }