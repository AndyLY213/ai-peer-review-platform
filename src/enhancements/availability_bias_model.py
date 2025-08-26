"""
Availability Bias Model

This module implements availability bias modeling for the peer review simulation.
Availability bias occurs when reviewers are influenced by recent exposure to similar
work, making them more likely to judge papers based on easily recalled examples
rather than comprehensive evaluation.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
import re

from src.enhancements.bias_engine import BiasModel, BiasType
from src.data.enhanced_models import EnhancedResearcher, StructuredReview, BiasEffect
from src.core.exceptions import BiasSystemError
from src.core.logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class RecentExposure:
    """Represents recent exposure to a paper or research topic."""
    paper_id: str
    title: str
    abstract: str
    keywords: List[str]
    exposure_date: datetime
    exposure_type: str  # 'reviewed', 'read', 'cited', 'discussed'
    relevance_score: float = 0.0  # How relevant this exposure is (0-1)
    
    def __post_init__(self):
        """Calculate relevance score and other derived fields."""
        self._calculate_relevance_score()
    
    def _calculate_relevance_score(self):
        """Calculate how relevant this exposure is for availability bias."""
        # Base relevance from exposure type
        type_weights = {
            'reviewed': 1.0,    # Highest relevance - actively reviewed
            'cited': 0.8,       # High relevance - used in own work
            'read': 0.6,        # Medium relevance - read recently
            'discussed': 0.4    # Lower relevance - heard about
        }
        
        base_relevance = type_weights.get(self.exposure_type, 0.5)
        
        # Adjust for recency (more recent = more available)
        days_ago = (datetime.now() - self.exposure_date).days
        recency_factor = math.exp(-days_ago / 30.0)  # Exponential decay over 30 days
        
        self.relevance_score = base_relevance * recency_factor


@dataclass
class TopicSimilarity:
    """Represents similarity between two research topics or papers."""
    similarity_score: float  # 0-1 scale
    matching_keywords: List[str] = field(default_factory=list)
    semantic_similarity: float = 0.0  # Future: could use embeddings
    
    def get_overall_similarity(self) -> float:
        """Get overall similarity combining different factors."""
        # For now, just use keyword-based similarity
        # In future, could combine with semantic similarity from embeddings
        return self.similarity_score


class AvailabilityBiasModel(BiasModel):
    """
    Models availability bias in peer review.
    
    Availability bias occurs when:
    1. Reviewers judge papers based on easily recalled similar work
    2. Recent exposure to similar papers influences current evaluation
    3. Memorable examples (positive or negative) skew judgment
    4. Recency of exposure affects the strength of bias
    """
    
    def __init__(self, configuration):
        """Initialize availability bias model."""
        super().__init__(configuration)
        
        # Extract parameters from configuration
        params = configuration.parameters
        self.recency_window_days = params.get('recency_window_days', 30)
        self.similarity_threshold = params.get('similarity_threshold', 0.6)
        self.max_adjustment = params.get('max_adjustment', 1.0)
        
        # Track recent exposures for each reviewer
        self.reviewer_exposures: Dict[str, List[RecentExposure]] = {}
        
        # Common research keywords for similarity calculation
        self.research_keywords = self._initialize_research_keywords()
    
    def _initialize_research_keywords(self) -> Dict[str, Set[str]]:
        """Initialize common research keywords by field."""
        return {
            "AI": {
                "machine learning", "deep learning", "neural networks", "artificial intelligence",
                "reinforcement learning", "supervised learning", "unsupervised learning",
                "computer vision", "natural language processing", "robotics", "expert systems",
                "knowledge representation", "planning", "search algorithms", "optimization"
            },
            "NLP": {
                "natural language processing", "computational linguistics", "text mining",
                "sentiment analysis", "machine translation", "information extraction",
                "question answering", "dialogue systems", "language models", "parsing",
                "named entity recognition", "part-of-speech tagging", "semantic analysis",
                "discourse analysis", "text classification", "text generation"
            },
            "Computer Vision": {
                "computer vision", "image processing", "pattern recognition", "object detection",
                "image classification", "segmentation", "feature extraction", "optical character recognition",
                "face recognition", "medical imaging", "video analysis", "3d reconstruction",
                "image enhancement", "stereo vision", "motion detection", "tracking"
            },
            "Machine Learning": {
                "machine learning", "statistical learning", "data mining", "predictive modeling",
                "classification", "regression", "clustering", "dimensionality reduction",
                "feature selection", "model selection", "cross-validation", "ensemble methods",
                "decision trees", "support vector machines", "bayesian methods", "neural networks"
            }
        }
    
    def calculate_bias_effect(
        self, 
        reviewer: EnhancedResearcher,
        review: StructuredReview,
        context: Dict[str, Any]
    ) -> BiasEffect:
        """
        Calculate availability bias effect based on recent exposure to similar work.
        
        Args:
            reviewer: The researcher conducting the review
            review: The review being written
            context: Additional context including paper content
            
        Returns:
            BiasEffect representing the availability bias impact
        """
        try:
            # Get current paper information
            paper_content = context.get('paper_content', {})
            paper_title = paper_content.get('title', '')
            paper_abstract = paper_content.get('abstract', '')
            paper_keywords = paper_content.get('keywords', [])
            
            if not paper_title and not paper_abstract:
                return BiasEffect(
                    bias_type=self.bias_type.value,
                    strength=0.0,
                    score_adjustment=0.0,
                    description="No paper content available for availability bias"
                )
            
            # Get reviewer's recent exposures
            recent_exposures = self._get_recent_exposures(reviewer.id)
            
            if not recent_exposures:
                return BiasEffect(
                    bias_type=self.bias_type.value,
                    strength=0.0,
                    score_adjustment=0.0,
                    description="No recent exposures for availability bias"
                )
            
            # Find most similar recent exposure
            most_similar_exposure, similarity = self._find_most_similar_exposure(
                paper_title, paper_abstract, paper_keywords, recent_exposures
            )
            
            if similarity.get_overall_similarity() < self.similarity_threshold:
                return BiasEffect(
                    bias_type=self.bias_type.value,
                    strength=0.0,
                    score_adjustment=0.0,
                    description="No sufficiently similar recent exposures"
                )
            
            # Get effective bias strength
            effective_strength = self.get_effective_strength(reviewer, context)
            
            # Calculate score adjustment based on similarity and recency
            score_adjustment = self._calculate_score_adjustment(
                most_similar_exposure, similarity, effective_strength, context
            )
            
            # Create bias effect description
            description = self._generate_bias_description(
                most_similar_exposure, similarity, score_adjustment
            )
            
            logger.debug(
                f"Availability bias for reviewer {reviewer.id}: "
                f"similarity={similarity.get_overall_similarity():.3f}, "
                f"adjustment={score_adjustment:.3f}"
            )
            
            return BiasEffect(
                bias_type=self.bias_type.value,
                strength=effective_strength,
                score_adjustment=score_adjustment,
                description=description
            )
            
        except Exception as e:
            logger.error(f"Error calculating availability bias: {e}")
            # Return neutral bias effect on error
            return BiasEffect(
                bias_type=self.bias_type.value,
                strength=0.0,
                score_adjustment=0.0,
                description="Error in availability bias calculation"
            )
    
    def _get_recent_exposures(self, reviewer_id: str) -> List[RecentExposure]:
        """
        Get recent exposures for a reviewer within the recency window.
        
        Args:
            reviewer_id: ID of the reviewer
            
        Returns:
            List of recent exposures within the time window
        """
        if reviewer_id not in self.reviewer_exposures:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=self.recency_window_days)
        
        recent_exposures = [
            exposure for exposure in self.reviewer_exposures[reviewer_id]
            if exposure.exposure_date >= cutoff_date
        ]
        
        # Sort by relevance score (most relevant first)
        recent_exposures.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return recent_exposures
    
    def _find_most_similar_exposure(
        self,
        paper_title: str,
        paper_abstract: str,
        paper_keywords: List[str],
        recent_exposures: List[RecentExposure]
    ) -> tuple[RecentExposure, TopicSimilarity]:
        """
        Find the most similar recent exposure to the current paper.
        
        Args:
            paper_title: Title of current paper
            paper_abstract: Abstract of current paper
            paper_keywords: Keywords of current paper
            recent_exposures: List of recent exposures to compare against
            
        Returns:
            Tuple of (most similar exposure, similarity object)
        """
        max_similarity = 0.0
        most_similar = None
        best_similarity_obj = None
        
        current_text = f"{paper_title} {paper_abstract} {' '.join(paper_keywords)}".lower()
        current_keywords = set(paper_keywords + self._extract_keywords_from_text(current_text))
        
        for exposure in recent_exposures:
            exposure_text = f"{exposure.title} {exposure.abstract} {' '.join(exposure.keywords)}".lower()
            exposure_keywords = set(exposure.keywords + self._extract_keywords_from_text(exposure_text))
            
            # Calculate keyword-based similarity
            similarity_obj = self._calculate_topic_similarity(
                current_keywords, exposure_keywords, current_text, exposure_text
            )
            
            overall_similarity = similarity_obj.get_overall_similarity()
            
            if overall_similarity > max_similarity:
                max_similarity = overall_similarity
                most_similar = exposure
                best_similarity_obj = similarity_obj
        
        if most_similar is None:
            # Return a dummy exposure with zero similarity
            dummy_exposure = RecentExposure(
                paper_id="none",
                title="",
                abstract="",
                keywords=[],
                exposure_date=datetime.now(),
                exposure_type="none"
            )
            dummy_similarity = TopicSimilarity(similarity_score=0.0)
            return dummy_exposure, dummy_similarity
        
        return most_similar, best_similarity_obj
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """
        Extract research keywords from text using predefined keyword sets.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of found keywords
        """
        found_keywords = []
        
        # Check all research keyword sets
        for field, keywords in self.research_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    found_keywords.append(keyword)
        
        return found_keywords
    
    def _calculate_topic_similarity(
        self,
        keywords1: Set[str],
        keywords2: Set[str],
        text1: str,
        text2: str
    ) -> TopicSimilarity:
        """
        Calculate similarity between two topics based on keywords and text.
        
        Args:
            keywords1: Keywords from first topic
            keywords2: Keywords from second topic
            text1: Text from first topic
            text2: Text from second topic
            
        Returns:
            TopicSimilarity object
        """
        # Calculate Jaccard similarity for keywords
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        if len(union) == 0:
            jaccard_similarity = 0.0
        else:
            jaccard_similarity = len(intersection) / len(union)
        
        # Calculate text overlap similarity (simplified)
        text1_words = set(text1.lower().split())
        text2_words = set(text2.lower().split())
        
        text_intersection = text1_words.intersection(text2_words)
        text_union = text1_words.union(text2_words)
        
        if len(text_union) == 0:
            text_similarity = 0.0
        else:
            text_similarity = len(text_intersection) / len(text_union)
        
        # Combine similarities (weight keywords more heavily)
        overall_similarity = (jaccard_similarity * 0.7) + (text_similarity * 0.3)
        
        return TopicSimilarity(
            similarity_score=overall_similarity,
            matching_keywords=list(intersection),
            semantic_similarity=text_similarity
        )
    
    def _calculate_score_adjustment(
        self,
        similar_exposure: RecentExposure,
        similarity: TopicSimilarity,
        effective_strength: float,
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate score adjustment based on similar exposure.
        
        Args:
            similar_exposure: The most similar recent exposure
            similarity: Similarity metrics
            effective_strength: Effective bias strength
            context: Additional context
            
        Returns:
            Score adjustment (can be positive or negative)
        """
        # Base adjustment depends on the type of exposure and its outcome
        base_adjustment = self._determine_exposure_valence(similar_exposure, context)
        
        # Scale by similarity strength
        similarity_factor = similarity.get_overall_similarity()
        
        # Scale by recency (more recent = stronger effect)
        days_ago = (datetime.now() - similar_exposure.exposure_date).days
        recency_factor = math.exp(-days_ago / 15.0)  # Stronger decay than relevance
        
        # Scale by exposure relevance
        relevance_factor = similar_exposure.relevance_score
        
        # Combine all factors
        total_adjustment = (base_adjustment * similarity_factor * 
                          recency_factor * relevance_factor * effective_strength)
        
        # Apply maximum adjustment limit
        total_adjustment = max(-self.max_adjustment, 
                             min(self.max_adjustment, total_adjustment))
        
        return total_adjustment
    
    def _determine_exposure_valence(
        self, 
        exposure: RecentExposure, 
        context: Dict[str, Any]
    ) -> float:
        """
        Determine whether the exposure was positive or negative.
        
        This is simplified - in practice would need more sophisticated analysis
        of the exposure outcome.
        
        Args:
            exposure: The exposure to analyze
            context: Additional context
            
        Returns:
            Valence score (-1 to +1, where +1 is very positive)
        """
        # Simple heuristics based on exposure type
        if exposure.exposure_type == 'reviewed':
            # If reviewer gave high scores to similar work, bias positively
            # This is simplified - would need actual review data
            return 0.3  # Slight positive bias for reviewed work
        elif exposure.exposure_type == 'cited':
            # If cited in own work, likely positive
            return 0.5
        elif exposure.exposure_type == 'read':
            # Neutral to slightly positive for read papers
            return 0.1
        elif exposure.exposure_type == 'discussed':
            # Could be positive or negative - assume neutral
            return 0.0
        else:
            return 0.0
    
    def _generate_bias_description(
        self,
        similar_exposure: RecentExposure,
        similarity: TopicSimilarity,
        score_adjustment: float
    ) -> str:
        """Generate a human-readable description of the availability bias."""
        if abs(score_adjustment) < 0.05:
            return "Minimal availability bias - recent exposure had little influence"
        
        days_ago = (datetime.now() - similar_exposure.exposure_date).days
        
        direction = "positively" if score_adjustment > 0 else "negatively"
        similarity_level = "highly" if similarity.get_overall_similarity() > 0.75 else "moderately"
        
        return (f"Availability bias - {similarity_level} similar paper "
                f"{similar_exposure.exposure_type} {days_ago} days ago "
                f"influenced judgment {direction} ({score_adjustment:+.2f})")
    
    def is_applicable(self, reviewer: EnhancedResearcher, context: Dict[str, Any]) -> bool:
        """
        Check if availability bias is applicable for this reviewer and context.
        
        Args:
            reviewer: The researcher conducting the review
            context: Additional context for applicability check
            
        Returns:
            True if bias should be applied, False otherwise
        """
        # Check if bias is enabled
        if not super().is_applicable(reviewer, context):
            return False
        
        # Check if we have paper content to analyze
        paper_content = context.get('paper_content', {})
        if not paper_content.get('title') and not paper_content.get('abstract'):
            return False
        
        # Check if reviewer has any susceptibility to availability bias
        availability_susceptibility = reviewer.cognitive_biases.get('availability', 0.0)
        if availability_susceptibility <= 0.0:
            return False
        
        # Check if reviewer has recent exposures
        recent_exposures = self._get_recent_exposures(reviewer.id)
        if not recent_exposures:
            return False
        
        return True
    
    def add_exposure(
        self,
        reviewer_id: str,
        paper_id: str,
        title: str,
        abstract: str,
        keywords: List[str],
        exposure_type: str,
        exposure_date: Optional[datetime] = None
    ):
        """
        Add a recent exposure for a reviewer.
        
        Args:
            reviewer_id: ID of the reviewer
            paper_id: ID of the paper they were exposed to
            title: Title of the paper
            abstract: Abstract of the paper
            keywords: Keywords of the paper
            exposure_type: Type of exposure ('reviewed', 'read', 'cited', 'discussed')
            exposure_date: Date of exposure (defaults to now)
        """
        if exposure_date is None:
            exposure_date = datetime.now()
        
        if reviewer_id not in self.reviewer_exposures:
            self.reviewer_exposures[reviewer_id] = []
        
        exposure = RecentExposure(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            keywords=keywords,
            exposure_date=exposure_date,
            exposure_type=exposure_type
        )
        
        self.reviewer_exposures[reviewer_id].append(exposure)
        
        # Keep exposure history manageable (limit to last 50 exposures)
        if len(self.reviewer_exposures[reviewer_id]) > 50:
            self.reviewer_exposures[reviewer_id] = self.reviewer_exposures[reviewer_id][-50:]
        
        logger.debug(f"Added {exposure_type} exposure for reviewer {reviewer_id}: {title[:50]}...")
    
    def get_reviewer_exposures(
        self, 
        reviewer_id: str, 
        days_back: Optional[int] = None
    ) -> List[RecentExposure]:
        """
        Get exposures for a specific reviewer.
        
        Args:
            reviewer_id: ID of the reviewer
            days_back: Number of days to look back (defaults to recency window)
            
        Returns:
            List of exposures within the specified time window
        """
        if reviewer_id not in self.reviewer_exposures:
            return []
        
        if days_back is None:
            days_back = self.recency_window_days
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        return [
            exposure for exposure in self.reviewer_exposures[reviewer_id]
            if exposure.exposure_date >= cutoff_date
        ]
    
    def clear_reviewer_exposures(self, reviewer_id: str):
        """
        Clear all exposures for a specific reviewer.
        
        Args:
            reviewer_id: ID of the reviewer
        """
        if reviewer_id in self.reviewer_exposures:
            del self.reviewer_exposures[reviewer_id]
            logger.info(f"Cleared exposures for reviewer {reviewer_id}")
    
    def get_availability_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about availability bias effects.
        
        Returns:
            Dictionary containing availability bias statistics
        """
        total_reviewers = len(self.reviewer_exposures)
        total_exposures = sum(len(exposures) for exposures in self.reviewer_exposures.values())
        
        if total_exposures == 0:
            return {
                "total_reviewers_with_exposures": 0,
                "total_exposures": 0,
                "average_exposures_per_reviewer": 0.0,
                "exposure_types": {},
                "average_relevance_score": 0.0
            }
        
        # Count exposure types
        exposure_types = {}
        all_exposures = []
        
        for exposures in self.reviewer_exposures.values():
            all_exposures.extend(exposures)
            for exposure in exposures:
                exposure_types[exposure.exposure_type] = exposure_types.get(exposure.exposure_type, 0) + 1
        
        # Calculate average relevance
        avg_relevance = sum(exp.relevance_score for exp in all_exposures) / len(all_exposures)
        
        return {
            "total_reviewers_with_exposures": total_reviewers,
            "total_exposures": total_exposures,
            "average_exposures_per_reviewer": total_exposures / total_reviewers if total_reviewers > 0 else 0.0,
            "exposure_types": exposure_types,
            "average_relevance_score": avg_relevance,
            "recent_exposures": sum(1 for exp in all_exposures 
                                  if (datetime.now() - exp.exposure_date).days <= self.recency_window_days)
        }
    
    def cleanup_old_exposures(self):
        """Remove exposures older than the maximum retention period."""
        max_retention_days = self.recency_window_days * 3  # Keep 3x the recency window
        cutoff_date = datetime.now() - timedelta(days=max_retention_days)
        
        cleaned_count = 0
        for reviewer_id in list(self.reviewer_exposures.keys()):
            original_count = len(self.reviewer_exposures[reviewer_id])
            
            self.reviewer_exposures[reviewer_id] = [
                exposure for exposure in self.reviewer_exposures[reviewer_id]
                if exposure.exposure_date >= cutoff_date
            ]
            
            cleaned_count += original_count - len(self.reviewer_exposures[reviewer_id])
            
            # Remove reviewer if no exposures left
            if not self.reviewer_exposures[reviewer_id]:
                del self.reviewer_exposures[reviewer_id]
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old exposures")
    
    def reset_all_exposures(self):
        """Reset all exposure history."""
        self.reviewer_exposures.clear()
        logger.info("Reset all availability bias exposure history")