"""
Publication Reform Manager

This module implements the PublicationReformManager class for alternative metrics,
post-publication review support, new evaluation models, and reform impact assessment
and adoption tracking.
"""

import json
import uuid
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path
import statistics

from src.core.exceptions import ValidationError, DatabaseError
from src.core.logging_config import get_logger
from src.data.enhanced_models import EnhancedResearcher, StructuredReview, EnhancedVenue, VenueType


logger = get_logger(__name__)


class ReformType(Enum):
    """Types of publication reforms."""
    ALTERNATIVE_METRICS = "Alternative Metrics"
    POST_PUBLICATION_REVIEW = "Post-Publication Review"
    OPEN_PEER_REVIEW = "Open Peer Review"
    PREPRINT_FIRST = "Preprint First"
    CONTINUOUS_EVALUATION = "Continuous Evaluation"
    COLLABORATIVE_REVIEW = "Collaborative Review"
    AI_ASSISTED_REVIEW = "AI-Assisted Review"
    REPRODUCIBILITY_BADGES = "Reproducibility Badges"


class MetricType(Enum):
    """Types of alternative metrics."""
    ALTMETRIC_SCORE = "Altmetric Score"
    SOCIAL_MEDIA_MENTIONS = "Social Media Mentions"
    DOWNLOAD_COUNT = "Download Count"
    GITHUB_STARS = "GitHub Stars"
    REPLICATION_SUCCESS_RATE = "Replication Success Rate"
    PEER_REVIEW_QUALITY = "Peer Review Quality"
    COMMUNITY_RATING = "Community Rating"
    USAGE_IMPACT = "Usage Impact"


class AdoptionStage(Enum):
    """Stages of reform adoption."""
    EXPERIMENTAL = "Experimental"
    PILOT = "Pilot"
    PARTIAL_ADOPTION = "Partial Adoption"
    WIDESPREAD_ADOPTION = "Widespread Adoption"
    STANDARD_PRACTICE = "Standard Practice"


@dataclass
class AlternativeMetric:
    """Represents an alternative evaluation metric."""
    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    confidence: float = 1.0  # 0-1 scale
    
    def __post_init__(self):
        """Validate metric data."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValidationError("confidence", self.confidence, "float between 0.0 and 1.0")


@dataclass
class PostPublicationReview:
    """Represents a post-publication review."""
    review_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    paper_id: str = ""
    reviewer_id: str = ""
    review_type: str = "post_publication"  # "post_publication", "community", "expert"
    
    # Review content
    rating: float = 5.0  # 1-10 scale
    summary: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    submission_date: datetime = field(default_factory=datetime.now)
    is_anonymous: bool = True
    is_verified_reviewer: bool = False
    helpfulness_votes: int = 0
    
    def __post_init__(self):
        """Validate review data."""
        if not (1.0 <= self.rating <= 10.0):
            raise ValidationError("rating", self.rating, "float between 1.0 and 10.0")


@dataclass
class ReformImplementation:
    """Represents the implementation of a specific reform."""
    reform_type: ReformType
    venue_id: str
    implementation_date: date
    adoption_stage: AdoptionStage = AdoptionStage.EXPERIMENTAL
    
    # Configuration
    parameters: Dict[str, Any] = field(default_factory=dict)
    success_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Impact tracking
    adoption_rate: float = 0.0  # 0-1 scale
    researcher_satisfaction: float = 0.0  # 0-1 scale
    quality_improvement: float = 0.0  # Change in review/paper quality
    efficiency_gain: float = 0.0  # Change in process efficiency
    
    def __post_init__(self):
        """Validate implementation data."""
        for metric in [self.adoption_rate, self.researcher_satisfaction]:
            if not (0.0 <= metric <= 1.0):
                raise ValidationError("metric", metric, "float between 0.0 and 1.0")


@dataclass
class EvaluationModel:
    """Represents a new evaluation model."""
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Model components
    traditional_weight: float = 0.5  # Weight of traditional peer review
    alternative_metrics_weight: float = 0.3  # Weight of alternative metrics
    post_pub_review_weight: float = 0.2  # Weight of post-publication reviews
    
    # Metric definitions
    included_metrics: List[MetricType] = field(default_factory=list)
    metric_weights: Dict[str, float] = field(default_factory=dict)
    
    # Validation and thresholds
    min_traditional_reviews: int = 2
    min_post_pub_reviews: int = 0
    acceptance_threshold: float = 6.0
    
    def __post_init__(self):
        """Validate model weights sum to 1.0."""
        total_weight = (self.traditional_weight + 
                       self.alternative_metrics_weight + 
                       self.post_pub_review_weight)
        if abs(total_weight - 1.0) > 0.01:
            raise ValidationError("weights", total_weight, "weights must sum to 1.0")


class PublicationReformManager:
    """
    Manages publication reform initiatives including alternative metrics,
    post-publication review, and new evaluation models.
    """
    
    def __init__(self, data_dir: str = "data/reforms"):
        """Initialize the reform manager."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for reform data
        self.reforms: Dict[str, ReformImplementation] = {}
        self.evaluation_models: Dict[str, EvaluationModel] = {}
        self.alternative_metrics: Dict[str, List[AlternativeMetric]] = {}
        self.post_pub_reviews: Dict[str, List[PostPublicationReview]] = {}
        
        # Tracking data
        self.adoption_history: List[Dict[str, Any]] = []
        self.impact_assessments: Dict[str, Dict[str, float]] = {}
        
        self._load_data()
        logger.info("PublicationReformManager initialized")
    
    def implement_reform(self, venue_id: str, reform_type: ReformType, 
                        parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Implement a publication reform at a specific venue.
        
        Args:
            venue_id: ID of the venue implementing the reform
            reform_type: Type of reform to implement
            parameters: Configuration parameters for the reform
            
        Returns:
            Reform implementation ID
        """
        try:
            reform_id = f"{venue_id}_{reform_type.value}_{datetime.now().strftime('%Y%m%d')}"
            
            reform = ReformImplementation(
                reform_type=reform_type,
                venue_id=venue_id,
                implementation_date=date.today(),
                parameters=parameters or {}
            )
            
            self.reforms[reform_id] = reform
            
            # Initialize tracking for this reform
            self.impact_assessments[reform_id] = {
                'baseline_quality': 0.0,
                'baseline_efficiency': 0.0,
                'baseline_satisfaction': 0.0
            }
            
            # Log adoption event
            self.adoption_history.append({
                'reform_id': reform_id,
                'venue_id': venue_id,
                'reform_type': reform_type.value,
                'date': date.today().isoformat(),
                'stage': AdoptionStage.EXPERIMENTAL.value
            })
            
            logger.info(f"Implemented reform {reform_type.value} at venue {venue_id}")
            return reform_id
            
        except Exception as e:
            logger.error(f"Error implementing reform: {e}")
            raise DatabaseError(f"Failed to implement reform: {e}")
    
    def create_evaluation_model(self, name: str, description: str,
                               traditional_weight: float = 0.5,
                               alternative_weight: float = 0.3,
                               post_pub_weight: float = 0.2,
                               included_metrics: Optional[List[MetricType]] = None) -> str:
        """
        Create a new evaluation model combining traditional and alternative approaches.
        
        Args:
            name: Name of the evaluation model
            description: Description of the model
            traditional_weight: Weight for traditional peer review
            alternative_weight: Weight for alternative metrics
            post_pub_weight: Weight for post-publication reviews
            included_metrics: List of alternative metrics to include
            
        Returns:
            Model ID
        """
        try:
            model = EvaluationModel(
                name=name,
                description=description,
                traditional_weight=traditional_weight,
                alternative_metrics_weight=alternative_weight,
                post_pub_review_weight=post_pub_weight,
                included_metrics=included_metrics or []
            )
            
            self.evaluation_models[model.model_id] = model
            
            logger.info(f"Created evaluation model: {name}")
            return model.model_id
            
        except Exception as e:
            logger.error(f"Error creating evaluation model: {e}")
            raise ValidationError("evaluation_model", str(e), "valid model parameters")
    
    def add_alternative_metric(self, paper_id: str, metric_type: MetricType,
                              value: float, source: str = "",
                              confidence: float = 1.0) -> None:
        """
        Add an alternative metric for a paper.
        
        Args:
            paper_id: ID of the paper
            metric_type: Type of alternative metric
            value: Metric value
            source: Source of the metric data
            confidence: Confidence in the metric (0-1)
        """
        try:
            metric = AlternativeMetric(
                metric_type=metric_type,
                value=value,
                source=source,
                confidence=confidence
            )
            
            if paper_id not in self.alternative_metrics:
                self.alternative_metrics[paper_id] = []
            
            self.alternative_metrics[paper_id].append(metric)
            
            logger.debug(f"Added {metric_type.value} metric for paper {paper_id}: {value}")
            
        except Exception as e:
            logger.error(f"Error adding alternative metric: {e}")
            raise ValidationError("alternative_metric", str(e), "valid metric data")
    
    def submit_post_publication_review(self, paper_id: str, reviewer_id: str,
                                     rating: float, summary: str,
                                     strengths: Optional[List[str]] = None,
                                     weaknesses: Optional[List[str]] = None,
                                     is_anonymous: bool = True) -> str:
        """
        Submit a post-publication review.
        
        Args:
            paper_id: ID of the paper being reviewed
            reviewer_id: ID of the reviewer
            rating: Overall rating (1-10)
            summary: Review summary
            strengths: List of paper strengths
            weaknesses: List of paper weaknesses
            is_anonymous: Whether the review is anonymous
            
        Returns:
            Review ID
        """
        try:
            review = PostPublicationReview(
                paper_id=paper_id,
                reviewer_id=reviewer_id,
                rating=rating,
                summary=summary,
                strengths=strengths or [],
                weaknesses=weaknesses or [],
                is_anonymous=is_anonymous
            )
            
            if paper_id not in self.post_pub_reviews:
                self.post_pub_reviews[paper_id] = []
            
            self.post_pub_reviews[paper_id].append(review)
            
            logger.info(f"Submitted post-publication review for paper {paper_id}")
            return review.review_id
            
        except Exception as e:
            logger.error(f"Error submitting post-publication review: {e}")
            raise ValidationError("post_pub_review", str(e), "valid review data")
    
    def evaluate_paper_with_model(self, paper_id: str, model_id: str,
                                 traditional_reviews: List[StructuredReview]) -> Dict[str, float]:
        """
        Evaluate a paper using a specific evaluation model.
        
        Args:
            paper_id: ID of the paper to evaluate
            model_id: ID of the evaluation model to use
            traditional_reviews: List of traditional peer reviews
            
        Returns:
            Dictionary with evaluation scores and components
        """
        try:
            if model_id not in self.evaluation_models:
                raise ValidationError("model_id", model_id, "existing evaluation model")
            
            model = self.evaluation_models[model_id]
            
            # Calculate traditional review score
            if traditional_reviews:
                traditional_score = statistics.mean([
                    review.criteria_scores.get_average_score() 
                    for review in traditional_reviews
                ])
            else:
                traditional_score = 0.0
            
            # Calculate alternative metrics score
            alt_metrics_score = self._calculate_alternative_metrics_score(
                paper_id, model.included_metrics, model.metric_weights
            )
            
            # Calculate post-publication review score
            post_pub_score = self._calculate_post_pub_score(paper_id)
            
            # Combine scores according to model weights
            final_score = (
                traditional_score * model.traditional_weight +
                alt_metrics_score * model.alternative_metrics_weight +
                post_pub_score * model.post_pub_review_weight
            )
            
            return {
                'final_score': final_score,
                'traditional_score': traditional_score,
                'alternative_metrics_score': alt_metrics_score,
                'post_publication_score': post_pub_score,
                'model_id': model_id,
                'evaluation_date': datetime.now().isoformat()
            }
            
        except ValidationError:
            raise  # Re-raise ValidationError as-is
        except Exception as e:
            logger.error(f"Error evaluating paper with model: {e}")
            raise DatabaseError(f"Failed to evaluate paper: {e}")
    
    def track_reform_adoption(self, reform_id: str, adoption_rate: float,
                             researcher_satisfaction: float,
                             quality_improvement: float = 0.0,
                             efficiency_gain: float = 0.0) -> None:
        """
        Track the adoption and impact of a reform.
        
        Args:
            reform_id: ID of the reform implementation
            adoption_rate: Current adoption rate (0-1)
            researcher_satisfaction: Researcher satisfaction (0-1)
            quality_improvement: Change in quality metrics
            efficiency_gain: Change in efficiency metrics
        """
        try:
            if reform_id not in self.reforms:
                raise ValidationError("reform_id", reform_id, "existing reform")
            
            reform = self.reforms[reform_id]
            reform.adoption_rate = adoption_rate
            reform.researcher_satisfaction = researcher_satisfaction
            reform.quality_improvement = quality_improvement
            reform.efficiency_gain = efficiency_gain
            
            # Update adoption stage based on adoption rate
            if adoption_rate >= 0.8:
                reform.adoption_stage = AdoptionStage.STANDARD_PRACTICE
            elif adoption_rate >= 0.6:
                reform.adoption_stage = AdoptionStage.WIDESPREAD_ADOPTION
            elif adoption_rate >= 0.3:
                reform.adoption_stage = AdoptionStage.PARTIAL_ADOPTION
            elif adoption_rate >= 0.1:
                reform.adoption_stage = AdoptionStage.PILOT
            else:
                reform.adoption_stage = AdoptionStage.EXPERIMENTAL
            
            # Log tracking update
            self.adoption_history.append({
                'reform_id': reform_id,
                'venue_id': reform.venue_id,
                'adoption_rate': adoption_rate,
                'satisfaction': researcher_satisfaction,
                'quality_change': quality_improvement,
                'efficiency_change': efficiency_gain,
                'stage': reform.adoption_stage.value,
                'date': date.today().isoformat()
            })
            
            logger.info(f"Updated reform tracking for {reform_id}: "
                       f"adoption={adoption_rate:.2f}, satisfaction={researcher_satisfaction:.2f}")
            
        except Exception as e:
            logger.error(f"Error tracking reform adoption: {e}")
            raise ValidationError("reform_tracking", str(e), "valid tracking data")
    
    def assess_reform_impact(self, reform_id: str) -> Dict[str, Any]:
        """
        Assess the overall impact of a reform implementation.
        
        Args:
            reform_id: ID of the reform to assess
            
        Returns:
            Dictionary with impact assessment results
        """
        try:
            if reform_id not in self.reforms:
                raise ValidationError("reform_id", reform_id, "existing reform")
            
            reform = self.reforms[reform_id]
            
            # Calculate time since implementation
            days_since_implementation = (date.today() - reform.implementation_date).days
            
            # Get historical data for this reform
            reform_history = [
                entry for entry in self.adoption_history
                if entry['reform_id'] == reform_id
            ]
            
            # Calculate trends
            adoption_trend = self._calculate_trend([
                entry.get('adoption_rate', 0.0) for entry in reform_history[-5:]
                if 'adoption_rate' in entry
            ])
            
            satisfaction_trend = self._calculate_trend([
                entry.get('satisfaction', 0.0) for entry in reform_history[-5:]
                if 'satisfaction' in entry
            ])
            
            # Overall impact score
            impact_score = (
                reform.adoption_rate * 0.3 +
                reform.researcher_satisfaction * 0.3 +
                max(0, reform.quality_improvement) * 0.2 +
                max(0, reform.efficiency_gain) * 0.2
            )
            
            return {
                'reform_id': reform_id,
                'reform_type': reform.reform_type.value,
                'venue_id': reform.venue_id,
                'implementation_date': reform.implementation_date.isoformat(),
                'days_active': days_since_implementation,
                'current_stage': reform.adoption_stage.value,
                'adoption_rate': reform.adoption_rate,
                'researcher_satisfaction': reform.researcher_satisfaction,
                'quality_improvement': reform.quality_improvement,
                'efficiency_gain': reform.efficiency_gain,
                'adoption_trend': adoption_trend,
                'satisfaction_trend': satisfaction_trend,
                'overall_impact_score': impact_score,
                'assessment_date': datetime.now().isoformat()
            }
            
        except ValidationError:
            raise  # Re-raise ValidationError as-is
        except Exception as e:
            logger.error(f"Error assessing reform impact: {e}")
            raise DatabaseError(f"Failed to assess reform impact: {e}")
    
    def get_reform_recommendations(self, venue_id: str, 
                                 venue_type: VenueType) -> List[Dict[str, Any]]:
        """
        Get recommendations for reforms suitable for a specific venue.
        
        Args:
            venue_id: ID of the venue
            venue_type: Type of venue
            
        Returns:
            List of reform recommendations
        """
        try:
            recommendations = []
            
            # Define reform suitability by venue type
            venue_suitability = {
                VenueType.TOP_CONFERENCE: [
                    (ReformType.ALTERNATIVE_METRICS, 0.8),
                    (ReformType.POST_PUBLICATION_REVIEW, 0.9),
                    (ReformType.REPRODUCIBILITY_BADGES, 0.9),
                    (ReformType.OPEN_PEER_REVIEW, 0.6)
                ],
                VenueType.MID_CONFERENCE: [
                    (ReformType.ALTERNATIVE_METRICS, 0.7),
                    (ReformType.POST_PUBLICATION_REVIEW, 0.8),
                    (ReformType.COLLABORATIVE_REVIEW, 0.7),
                    (ReformType.PREPRINT_FIRST, 0.8)
                ],
                VenueType.LOW_CONFERENCE: [
                    (ReformType.COLLABORATIVE_REVIEW, 0.9),
                    (ReformType.AI_ASSISTED_REVIEW, 0.8),
                    (ReformType.CONTINUOUS_EVALUATION, 0.7),
                    (ReformType.PREPRINT_FIRST, 0.9)
                ],
                VenueType.TOP_JOURNAL: [
                    (ReformType.POST_PUBLICATION_REVIEW, 0.9),
                    (ReformType.REPRODUCIBILITY_BADGES, 0.9),
                    (ReformType.ALTERNATIVE_METRICS, 0.8),
                    (ReformType.OPEN_PEER_REVIEW, 0.7)
                ],
                VenueType.SPECIALIZED_JOURNAL: [
                    (ReformType.ALTERNATIVE_METRICS, 0.8),
                    (ReformType.POST_PUBLICATION_REVIEW, 0.8),
                    (ReformType.COLLABORATIVE_REVIEW, 0.8),
                    (ReformType.CONTINUOUS_EVALUATION, 0.7)
                ],
                VenueType.GENERAL_JOURNAL: [
                    (ReformType.AI_ASSISTED_REVIEW, 0.8),
                    (ReformType.COLLABORATIVE_REVIEW, 0.8),
                    (ReformType.PREPRINT_FIRST, 0.8),
                    (ReformType.CONTINUOUS_EVALUATION, 0.7)
                ]
            }
            
            suitable_reforms = venue_suitability.get(venue_type, [])
            
            for reform_type, suitability_score in suitable_reforms:
                # Check if already implemented
                existing_reform = any(
                    reform.venue_id == venue_id and reform.reform_type == reform_type
                    for reform in self.reforms.values()
                )
                
                if not existing_reform:
                    recommendations.append({
                        'reform_type': reform_type.value,
                        'suitability_score': suitability_score,
                        'description': self._get_reform_description(reform_type),
                        'expected_benefits': self._get_expected_benefits(reform_type),
                        'implementation_complexity': self._get_implementation_complexity(reform_type)
                    })
            
            # Sort by suitability score
            recommendations.sort(key=lambda x: x['suitability_score'], reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting reform recommendations: {e}")
            raise DatabaseError(f"Failed to get recommendations: {e}")
    
    def _calculate_alternative_metrics_score(self, paper_id: str, 
                                           included_metrics: List[MetricType],
                                           metric_weights: Dict[str, float]) -> float:
        """Calculate score from alternative metrics."""
        if paper_id not in self.alternative_metrics:
            return 0.0
        
        metrics = self.alternative_metrics[paper_id]
        total_score = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            if metric.metric_type in included_metrics:
                weight = metric_weights.get(metric.metric_type.value, 1.0)
                # Normalize different metric types to 0-10 scale
                normalized_value = self._normalize_metric_value(metric.metric_type, metric.value)
                total_score += normalized_value * weight * metric.confidence
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_post_pub_score(self, paper_id: str) -> float:
        """Calculate score from post-publication reviews."""
        if paper_id not in self.post_pub_reviews:
            return 0.0
        
        reviews = self.post_pub_reviews[paper_id]
        if not reviews:
            return 0.0
        
        # Weight reviews by helpfulness votes and reviewer verification
        weighted_scores = []
        for review in reviews:
            weight = 1.0
            if review.is_verified_reviewer:
                weight *= 1.5
            if review.helpfulness_votes > 0:
                weight *= (1.0 + min(review.helpfulness_votes / 10.0, 0.5))
            
            weighted_scores.append(review.rating * weight)
        
        return statistics.mean(weighted_scores)
    
    def _normalize_metric_value(self, metric_type: MetricType, value: float) -> float:
        """Normalize different metric types to 0-10 scale."""
        # Define normalization ranges for different metrics
        normalization_ranges = {
            MetricType.ALTMETRIC_SCORE: (0, 100),
            MetricType.SOCIAL_MEDIA_MENTIONS: (0, 1000),
            MetricType.DOWNLOAD_COUNT: (0, 10000),
            MetricType.GITHUB_STARS: (0, 1000),
            MetricType.REPLICATION_SUCCESS_RATE: (0, 1),
            MetricType.PEER_REVIEW_QUALITY: (0, 10),
            MetricType.COMMUNITY_RATING: (0, 10),
            MetricType.USAGE_IMPACT: (0, 100)
        }
        
        min_val, max_val = normalization_ranges.get(metric_type, (0, 10))
        
        # Normalize to 0-10 scale
        if max_val > min_val:
            normalized = ((value - min_val) / (max_val - min_val)) * 10.0
            return max(0.0, min(10.0, normalized))
        else:
            return value
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _get_reform_description(self, reform_type: ReformType) -> str:
        """Get description for a reform type."""
        descriptions = {
            ReformType.ALTERNATIVE_METRICS: "Incorporate social media mentions, downloads, and usage metrics alongside traditional citations",
            ReformType.POST_PUBLICATION_REVIEW: "Enable continuous peer review after publication with community feedback",
            ReformType.OPEN_PEER_REVIEW: "Make reviewer identities and reviews publicly available",
            ReformType.PREPRINT_FIRST: "Require preprint submission before formal peer review",
            ReformType.CONTINUOUS_EVALUATION: "Ongoing evaluation and updating of published work",
            ReformType.COLLABORATIVE_REVIEW: "Multiple reviewers collaborate on single comprehensive review",
            ReformType.AI_ASSISTED_REVIEW: "Use AI tools to assist reviewers in evaluation process",
            ReformType.REPRODUCIBILITY_BADGES: "Award badges for reproducible research with verified results"
        }
        return descriptions.get(reform_type, "Reform description not available")
    
    def _get_expected_benefits(self, reform_type: ReformType) -> List[str]:
        """Get expected benefits for a reform type."""
        benefits = {
            ReformType.ALTERNATIVE_METRICS: [
                "Broader impact assessment",
                "Faster recognition of important work",
                "Reduced citation bias"
            ],
            ReformType.POST_PUBLICATION_REVIEW: [
                "Continuous quality improvement",
                "Community-driven evaluation",
                "Error detection and correction"
            ],
            ReformType.OPEN_PEER_REVIEW: [
                "Increased accountability",
                "Reduced bias",
                "Educational value for reviewers"
            ],
            ReformType.PREPRINT_FIRST: [
                "Faster dissemination",
                "Early feedback incorporation",
                "Reduced publication delays"
            ],
            ReformType.CONTINUOUS_EVALUATION: [
                "Living documents",
                "Ongoing quality improvement",
                "Adaptation to new findings"
            ],
            ReformType.COLLABORATIVE_REVIEW: [
                "More comprehensive reviews",
                "Reduced reviewer burden",
                "Knowledge sharing among reviewers"
            ],
            ReformType.AI_ASSISTED_REVIEW: [
                "Faster review process",
                "Consistency in evaluation",
                "Detection of technical issues"
            ],
            ReformType.REPRODUCIBILITY_BADGES: [
                "Incentivized reproducibility",
                "Quality assurance",
                "Trust in research findings"
            ]
        }
        return benefits.get(reform_type, ["Benefits not specified"])
    
    def _get_implementation_complexity(self, reform_type: ReformType) -> str:
        """Get implementation complexity for a reform type."""
        complexity = {
            ReformType.ALTERNATIVE_METRICS: "Medium",
            ReformType.POST_PUBLICATION_REVIEW: "High",
            ReformType.OPEN_PEER_REVIEW: "Low",
            ReformType.PREPRINT_FIRST: "Medium",
            ReformType.CONTINUOUS_EVALUATION: "High",
            ReformType.COLLABORATIVE_REVIEW: "Medium",
            ReformType.AI_ASSISTED_REVIEW: "High",
            ReformType.REPRODUCIBILITY_BADGES: "Medium"
        }
        return complexity.get(reform_type, "Unknown")
    
    def _load_data(self):
        """Load existing reform data from files."""
        try:
            # Load reforms
            reforms_file = self.data_dir / "reforms.json"
            if reforms_file.exists():
                with open(reforms_file, 'r') as f:
                    data = json.load(f)
                    for reform_id, reform_data in data.items():
                        reform_data['reform_type'] = ReformType(reform_data['reform_type'])
                        reform_data['adoption_stage'] = AdoptionStage(reform_data['adoption_stage'])
                        reform_data['implementation_date'] = date.fromisoformat(reform_data['implementation_date'])
                        self.reforms[reform_id] = ReformImplementation(**reform_data)
            
            # Load evaluation models
            models_file = self.data_dir / "evaluation_models.json"
            if models_file.exists():
                with open(models_file, 'r') as f:
                    data = json.load(f)
                    for model_id, model_data in data.items():
                        if 'included_metrics' in model_data:
                            model_data['included_metrics'] = [
                                MetricType(m) for m in model_data['included_metrics']
                            ]
                        self.evaluation_models[model_id] = EvaluationModel(**model_data)
            
            # Load adoption history
            history_file = self.data_dir / "adoption_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.adoption_history = json.load(f)
            
            logger.info("Loaded existing reform data")
            
        except Exception as e:
            logger.warning(f"Could not load existing data: {e}")
    
    def save_data(self):
        """Save reform data to files."""
        try:
            # Save reforms
            reforms_data = {}
            for reform_id, reform in self.reforms.items():
                reform_dict = asdict(reform)
                reform_dict['reform_type'] = reform.reform_type.value
                reform_dict['adoption_stage'] = reform.adoption_stage.value
                reform_dict['implementation_date'] = reform.implementation_date.isoformat()
                reforms_data[reform_id] = reform_dict
            
            with open(self.data_dir / "reforms.json", 'w') as f:
                json.dump(reforms_data, f, indent=2)
            
            # Save evaluation models
            models_data = {}
            for model_id, model in self.evaluation_models.items():
                model_dict = asdict(model)
                model_dict['included_metrics'] = [m.value for m in model.included_metrics]
                models_data[model_id] = model_dict
            
            with open(self.data_dir / "evaluation_models.json", 'w') as f:
                json.dump(models_data, f, indent=2)
            
            # Save adoption history
            with open(self.data_dir / "adoption_history.json", 'w') as f:
                json.dump(self.adoption_history, f, indent=2)
            
            logger.info("Saved reform data")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise DatabaseError(f"Failed to save reform data: {e}")