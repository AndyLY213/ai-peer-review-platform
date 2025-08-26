"""
Anchoring Bias Model

This module implements anchoring bias modeling for the peer review simulation.
Anchoring bias occurs when reviewers are influenced by earlier review scores,
causing later reviewers to anchor their judgments around previously submitted scores.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import math
from datetime import datetime

from src.enhancements.bias_engine import BiasModel, BiasType
from src.data.enhanced_models import EnhancedResearcher, StructuredReview, BiasEffect
from src.core.exceptions import BiasSystemError
from src.core.logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class ReviewAnchor:
    """Represents an anchoring point from a previous review."""
    review_id: str
    reviewer_id: str
    overall_score: float
    confidence_level: int
    submission_time: datetime
    anchor_strength: float = 1.0  # How strong this anchor is (0-1)
    
    def __post_init__(self):
        """Calculate anchor strength based on confidence and other factors."""
        self._calculate_anchor_strength()
    
    def _calculate_anchor_strength(self):
        """Calculate how strong this review serves as an anchor."""
        # Higher confidence reviews serve as stronger anchors
        confidence_factor = self.confidence_level / 5.0
        
        # More recent reviews have stronger anchoring effect (recency bias)
        # This is simplified - in practice would consider actual time differences
        recency_factor = 1.0  # Assume all reviews are equally recent for now
        
        # Combine factors
        self.anchor_strength = confidence_factor * recency_factor


class AnchoringBiasModel(BiasModel):
    """
    Models anchoring bias in peer review.
    
    Anchoring bias occurs when:
    1. Later reviewers are influenced by earlier review scores
    2. High-confidence reviews create stronger anchoring effects
    3. Multiple previous reviews create compound anchoring
    4. Reviewers adjust their scores toward the anchor values
    """
    
    def __init__(self, configuration):
        """Initialize anchoring bias model."""
        super().__init__(configuration)
        
        # Extract parameters from configuration
        params = configuration.parameters
        self.influence_decay = params.get('influence_decay', 0.8)
        self.confidence_weight = params.get('confidence_weight', 0.3)
        self.max_influence = params.get('max_influence', 1.0)
        
        # Track review history for anchoring calculations
        self.review_history: Dict[str, List[ReviewAnchor]] = {}  # paper_id -> list of anchors
    
    def calculate_bias_effect(
        self, 
        reviewer: EnhancedResearcher,
        review: StructuredReview,
        context: Dict[str, Any]
    ) -> BiasEffect:
        """
        Calculate anchoring bias effect based on previous reviews.
        
        Args:
            reviewer: The researcher conducting the review
            review: The review being written
            context: Additional context including previous reviews
            
        Returns:
            BiasEffect representing the anchoring bias impact
        """
        try:
            paper_id = review.paper_id
            
            # Get previous reviews from context or stored history
            previous_reviews = context.get('previous_reviews', [])
            if not previous_reviews and paper_id in self.review_history:
                previous_reviews = self.review_history[paper_id]
            
            if not previous_reviews:
                return BiasEffect(
                    bias_type=self.bias_type.value,
                    strength=0.0,
                    score_adjustment=0.0,
                    description="No previous reviews available for anchoring"
                )
            
            # Convert previous reviews to anchors if needed
            anchors = self._convert_to_anchors(previous_reviews)
            
            # Calculate anchoring effect
            anchor_score = self._calculate_anchor_score(anchors)
            
            # Get effective bias strength
            effective_strength = self.get_effective_strength(reviewer, context)
            
            # Get reviewer's current intended score (from context or review)
            reviewer_intended_score = context.get('intended_score', 
                                                review.criteria_scores.get_average_score())
            
            # Calculate score adjustment toward anchor
            score_adjustment = self._calculate_score_adjustment(
                reviewer_intended_score, anchor_score, effective_strength, anchors
            )
            
            # Create bias effect description
            description = self._generate_bias_description(
                anchor_score, score_adjustment, len(anchors)
            )
            
            logger.debug(
                f"Anchoring bias for reviewer {reviewer.id}: "
                f"anchor_score={anchor_score:.3f}, adjustment={score_adjustment:.3f}"
            )
            
            return BiasEffect(
                bias_type=self.bias_type.value,
                strength=effective_strength,
                score_adjustment=score_adjustment,
                description=description
            )
            
        except Exception as e:
            logger.error(f"Error calculating anchoring bias: {e}")
            # Return neutral bias effect on error
            return BiasEffect(
                bias_type=self.bias_type.value,
                strength=0.0,
                score_adjustment=0.0,
                description="Error in anchoring bias calculation"
            )
    
    def _convert_to_anchors(self, previous_reviews: List[Any]) -> List[ReviewAnchor]:
        """
        Convert previous reviews to ReviewAnchor objects.
        
        Args:
            previous_reviews: List of previous reviews (can be StructuredReview or dict)
            
        Returns:
            List of ReviewAnchor objects
        """
        anchors = []
        
        for review in previous_reviews:
            if isinstance(review, StructuredReview):
                anchor = ReviewAnchor(
                    review_id=review.review_id,
                    reviewer_id=review.reviewer_id,
                    overall_score=review.criteria_scores.get_average_score(),
                    confidence_level=review.confidence_level,
                    submission_time=review.submission_timestamp
                )
            elif isinstance(review, dict):
                # Handle dictionary format
                overall_score = review.get('overall_score', 5.0)
                if 'criteria_scores' in review:
                    # Calculate from criteria scores if available
                    criteria = review['criteria_scores']
                    if isinstance(criteria, dict):
                        scores = [criteria.get(dim, 5.0) for dim in 
                                ['novelty', 'technical_quality', 'clarity', 'significance', 'reproducibility', 'related_work']]
                        overall_score = sum(scores) / len(scores)
                
                anchor = ReviewAnchor(
                    review_id=review.get('review_id', 'unknown'),
                    reviewer_id=review.get('reviewer_id', 'unknown'),
                    overall_score=overall_score,
                    confidence_level=review.get('confidence_level', 3),
                    submission_time=datetime.fromisoformat(review['submission_time']) 
                                  if 'submission_time' in review 
                                  else datetime.now()
                )
            elif isinstance(review, ReviewAnchor):
                anchor = review
            else:
                logger.warning(f"Unknown review format: {type(review)}")
                continue
            
            anchors.append(anchor)
        
        return anchors
    
    def _calculate_anchor_score(self, anchors: List[ReviewAnchor]) -> float:
        """
        Calculate the effective anchor score from multiple previous reviews.
        
        Args:
            anchors: List of review anchors
            
        Returns:
            Weighted average anchor score
        """
        if not anchors:
            return 5.0  # Default neutral score
        
        # Sort anchors by submission time (most recent first)
        sorted_anchors = sorted(anchors, key=lambda a: a.submission_time, reverse=True)
        
        # Calculate weighted average with decay for older reviews
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for i, anchor in enumerate(sorted_anchors):
            # Apply decay for review order (more recent = higher weight)
            order_weight = (self.influence_decay ** i)
            
            # Weight by anchor strength (confidence-based)
            anchor_weight = anchor.anchor_strength * order_weight
            
            total_weighted_score += anchor.overall_score * anchor_weight
            total_weight += anchor_weight
        
        if total_weight == 0:
            return 5.0  # Default neutral score
        
        return total_weighted_score / total_weight
    
    def _calculate_score_adjustment(
        self,
        intended_score: float,
        anchor_score: float,
        effective_strength: float,
        anchors: List[ReviewAnchor]
    ) -> float:
        """
        Calculate the score adjustment toward the anchor.
        
        Args:
            intended_score: Reviewer's intended score without bias
            anchor_score: The anchor score to pull toward
            effective_strength: Effective bias strength
            anchors: List of anchors for additional context
            
        Returns:
            Score adjustment (can be positive or negative)
        """
        # Calculate base adjustment toward anchor
        score_difference = anchor_score - intended_score
        
        # Apply bias strength
        base_adjustment = score_difference * effective_strength
        
        # Apply maximum influence limit
        max_adjustment = self.max_influence
        base_adjustment = max(-max_adjustment, min(max_adjustment, base_adjustment))
        
        # Adjust based on anchor consensus
        anchor_consensus = self._calculate_anchor_consensus(anchors)
        consensus_multiplier = 0.5 + (anchor_consensus * 0.5)  # 0.5 to 1.0
        
        # Apply consensus multiplier
        final_adjustment = base_adjustment * consensus_multiplier
        
        return final_adjustment
    
    def _calculate_anchor_consensus(self, anchors: List[ReviewAnchor]) -> float:
        """
        Calculate how much consensus there is among the anchors.
        
        Args:
            anchors: List of review anchors
            
        Returns:
            Consensus score (0-1, where 1 is perfect consensus)
        """
        if len(anchors) <= 1:
            return 1.0  # Perfect consensus with single anchor
        
        scores = [anchor.overall_score for anchor in anchors]
        
        # Calculate standard deviation
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = math.sqrt(variance)
        
        # Convert to consensus score (lower std_dev = higher consensus)
        # Normalize by reasonable score range (assume scores 1-10, so max std_dev ~4.5)
        max_std_dev = 4.5
        consensus = max(0.0, 1.0 - (std_dev / max_std_dev))
        
        return consensus
    
    def _generate_bias_description(
        self, 
        anchor_score: float, 
        score_adjustment: float,
        num_anchors: int
    ) -> str:
        """Generate a human-readable description of the anchoring effect."""
        if abs(score_adjustment) < 0.05:
            return f"Minimal anchoring bias - {num_anchors} previous review(s) had little influence"
        
        direction = "toward" if score_adjustment > 0 else "away from"
        anchor_level = "high" if anchor_score > 6.5 else "low" if anchor_score < 4.5 else "moderate"
        
        return (f"Anchoring bias - pulled {direction} {anchor_level} anchor score "
                f"({anchor_score:.1f}) from {num_anchors} previous review(s) "
                f"({score_adjustment:+.2f})")
    
    def is_applicable(self, reviewer: EnhancedResearcher, context: Dict[str, Any]) -> bool:
        """
        Check if anchoring bias is applicable for this reviewer and context.
        
        Args:
            reviewer: The researcher conducting the review
            context: Additional context for applicability check
            
        Returns:
            True if bias should be applied, False otherwise
        """
        # Check if bias is enabled
        if not super().is_applicable(reviewer, context):
            return False
        
        # Check if we have previous reviews to anchor on
        previous_reviews = context.get('previous_reviews', [])
        paper_id = context.get('paper_id')
        
        if not previous_reviews and paper_id and paper_id in self.review_history:
            previous_reviews = self.review_history[paper_id]
        
        if not previous_reviews:
            return False
        
        # Check if reviewer has any susceptibility to anchoring bias
        anchoring_susceptibility = reviewer.cognitive_biases.get('anchoring', 0.0)
        if anchoring_susceptibility <= 0.0:
            return False
        
        return True
    
    def add_review_to_history(
        self, 
        paper_id: str, 
        review: StructuredReview
    ):
        """
        Add a completed review to the history for future anchoring.
        
        Args:
            paper_id: ID of the paper being reviewed
            review: The completed review
        """
        if paper_id not in self.review_history:
            self.review_history[paper_id] = []
        
        anchor = ReviewAnchor(
            review_id=review.review_id,
            reviewer_id=review.reviewer_id,
            overall_score=review.criteria_scores.get_average_score(),
            confidence_level=review.confidence_level,
            submission_time=review.submission_timestamp
        )
        
        self.review_history[paper_id].append(anchor)
        
        # Keep history manageable (limit to last 10 reviews per paper)
        if len(self.review_history[paper_id]) > 10:
            self.review_history[paper_id] = self.review_history[paper_id][-10:]
        
        logger.debug(f"Added review {review.review_id} to anchoring history for paper {paper_id}")
    
    def get_paper_review_history(self, paper_id: str) -> List[ReviewAnchor]:
        """
        Get the review history for a specific paper.
        
        Args:
            paper_id: ID of the paper
            
        Returns:
            List of review anchors for the paper
        """
        return self.review_history.get(paper_id, [])
    
    def clear_paper_history(self, paper_id: str):
        """
        Clear the review history for a specific paper.
        
        Args:
            paper_id: ID of the paper
        """
        if paper_id in self.review_history:
            del self.review_history[paper_id]
            logger.info(f"Cleared review history for paper {paper_id}")
    
    def get_anchoring_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about anchoring effects.
        
        Returns:
            Dictionary containing anchoring statistics
        """
        total_papers = len(self.review_history)
        total_reviews = sum(len(reviews) for reviews in self.review_history.values())
        
        if total_reviews == 0:
            return {
                "total_papers_with_history": 0,
                "total_reviews_in_history": 0,
                "average_reviews_per_paper": 0.0,
                "average_anchor_strength": 0.0
            }
        
        # Calculate average anchor strength
        all_anchors = []
        for reviews in self.review_history.values():
            all_anchors.extend(reviews)
        
        avg_anchor_strength = sum(anchor.anchor_strength for anchor in all_anchors) / len(all_anchors)
        
        return {
            "total_papers_with_history": total_papers,
            "total_reviews_in_history": total_reviews,
            "average_reviews_per_paper": total_reviews / total_papers if total_papers > 0 else 0.0,
            "average_anchor_strength": avg_anchor_strength,
            "papers_with_multiple_reviews": sum(1 for reviews in self.review_history.values() if len(reviews) > 1)
        }
    
    def simulate_review_order_effect(
        self, 
        base_scores: List[float], 
        reviewer_susceptibilities: List[float]
    ) -> List[float]:
        """
        Simulate the effect of review order on a sequence of reviews.
        
        Args:
            base_scores: List of intended scores without anchoring bias
            reviewer_susceptibilities: List of anchoring susceptibilities for each reviewer
            
        Returns:
            List of bias-adjusted scores showing anchoring effects
        """
        if len(base_scores) != len(reviewer_susceptibilities):
            raise ValueError("base_scores and reviewer_susceptibilities must have same length")
        
        adjusted_scores = []
        anchors = []
        
        for i, (base_score, susceptibility) in enumerate(zip(base_scores, reviewer_susceptibilities)):
            if i == 0:
                # First reviewer has no anchoring bias
                adjusted_score = base_score
            else:
                # Calculate anchoring effect from previous reviews
                anchor_score = self._calculate_anchor_score(anchors)
                
                # Calculate adjustment
                score_difference = anchor_score - base_score
                adjustment = score_difference * susceptibility
                
                # Apply maximum influence limit
                adjustment = max(-self.max_influence, min(self.max_influence, adjustment))
                
                adjusted_score = base_score + adjustment
            
            adjusted_scores.append(adjusted_score)
            
            # Add this review as an anchor for future reviews
            anchor = ReviewAnchor(
                review_id=f"sim_review_{i}",
                reviewer_id=f"sim_reviewer_{i}",
                overall_score=adjusted_score,
                confidence_level=3,  # Default confidence
                submission_time=datetime.now()
            )
            anchors.append(anchor)
        
        return adjusted_scores
    
    def reset_history(self):
        """Reset all review history."""
        self.review_history.clear()
        logger.info("Reset all anchoring bias history")