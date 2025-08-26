"""
Review History Tracking System

This module implements review history tracking with ReviewQualityMetric class to track
reviewer performance, reliability scoring based on review timeliness and quality,
and logic to maintain historical review quality data.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import statistics
from pathlib import Path
import json

from src.data.enhanced_models import (
    EnhancedResearcher, StructuredReview, ReviewQualityMetric,
    ResearcherLevel, ReviewDecision
)
from src.core.exceptions import ValidationError, DatabaseError
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class ReliabilityCategory(Enum):
    """Categories for reviewer reliability."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    AVERAGE = "Average"
    POOR = "Poor"
    UNRELIABLE = "Unreliable"


@dataclass
class ReviewPerformanceMetrics:
    """Comprehensive performance metrics for a reviewer."""
    reviewer_id: str
    total_reviews: int = 0
    avg_quality_score: float = 0.0
    avg_timeliness_score: float = 0.0
    reliability_score: float = 0.0
    reliability_category: ReliabilityCategory = ReliabilityCategory.AVERAGE
    
    # Detailed metrics
    on_time_reviews: int = 0
    late_reviews: int = 0
    avg_days_early_late: float = 0.0  # Negative = early, positive = late
    
    # Quality breakdown
    high_quality_reviews: int = 0  # Quality score >= 0.8
    medium_quality_reviews: int = 0  # Quality score 0.5-0.8
    low_quality_reviews: int = 0  # Quality score < 0.5
    
    # Consistency metrics
    quality_consistency: float = 0.0  # Standard deviation of quality scores
    timeliness_consistency: float = 0.0  # Standard deviation of timeliness scores
    
    # Trend analysis
    recent_performance_trend: str = "stable"  # "improving", "declining", "stable"
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            'reviewer_id': self.reviewer_id,
            'total_reviews': self.total_reviews,
            'avg_quality_score': self.avg_quality_score,
            'avg_timeliness_score': self.avg_timeliness_score,
            'reliability_score': self.reliability_score,
            'reliability_category': self.reliability_category.value,
            'on_time_reviews': self.on_time_reviews,
            'late_reviews': self.late_reviews,
            'avg_days_early_late': self.avg_days_early_late,
            'high_quality_reviews': self.high_quality_reviews,
            'medium_quality_reviews': self.medium_quality_reviews,
            'low_quality_reviews': self.low_quality_reviews,
            'quality_consistency': self.quality_consistency,
            'timeliness_consistency': self.timeliness_consistency,
            'recent_performance_trend': self.recent_performance_trend,
            'last_updated': self.last_updated.isoformat()
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewPerformanceMetrics':
        """Create from dictionary."""
        if 'reliability_category' in data:
            data['reliability_category'] = ReliabilityCategory(data['reliability_category'])
        if 'last_updated' in data:
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)


class ReviewHistoryTracker:
    """
    Comprehensive review history tracking system.
    
    This class provides functionality to:
    - Track reviewer performance over time
    - Calculate reliability scores based on timeliness and quality
    - Maintain historical review quality data
    - Analyze performance trends and patterns
    - Generate performance reports and recommendations
    """
    
    # Quality score thresholds
    HIGH_QUALITY_THRESHOLD = 0.8
    MEDIUM_QUALITY_THRESHOLD = 0.5
    
    # Timeliness scoring parameters
    EXCELLENT_TIMELINESS_DAYS = -2  # 2 days early or more
    GOOD_TIMELINESS_DAYS = 0       # On time
    POOR_TIMELINESS_DAYS = 3       # 3 days late or more
    
    # Reliability score weights
    QUALITY_WEIGHT = 0.6      # 60% weight for quality
    TIMELINESS_WEIGHT = 0.4   # 40% weight for timeliness
    
    # Performance trend analysis window (months)
    TREND_ANALYSIS_MONTHS = 6
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the review history tracker.
        
        Args:
            data_dir: Directory to store tracking data (optional)
        """
        logger.info("Initializing Review History Tracking System")
        self.data_dir = data_dir or Path("peer_review_workspace/review_history")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for performance metrics
        self.performance_metrics: Dict[str, ReviewPerformanceMetrics] = {}
        
        # Load existing data
        self._load_performance_data()
    
    def track_review_quality(self, review: StructuredReview, deadline: Optional[datetime] = None) -> ReviewQualityMetric:
        """
        Track quality metrics for a completed review.
        
        Args:
            review: The completed structured review
            deadline: The original deadline for the review
            
        Returns:
            ReviewQualityMetric object with calculated scores
        """
        try:
            # Calculate timeliness score
            timeliness_score = self._calculate_timeliness_score(review, deadline)
            
            # Create quality metric
            quality_metric = ReviewQualityMetric(
                review_id=review.review_id,
                quality_score=review.quality_score,
                timeliness_score=timeliness_score,
                timestamp=review.submission_timestamp
            )
            
            # Update reviewer's performance metrics
            self._update_reviewer_performance(review.reviewer_id, quality_metric, review, deadline)
            
            logger.info(f"Tracked review quality for reviewer {review.reviewer_id}: "
                       f"Quality={quality_metric.quality_score:.2f}, "
                       f"Timeliness={quality_metric.timeliness_score:.2f}")
            
            return quality_metric
            
        except Exception as e:
            logger.error(f"Error tracking review quality: {e}")
            raise DatabaseError(f"Failed to track review quality: {e}")
    
    def _calculate_timeliness_score(self, review: StructuredReview, deadline: Optional[datetime]) -> float:
        """
        Calculate timeliness score based on submission time vs deadline.
        
        Args:
            review: The review to score
            deadline: The deadline for the review
            
        Returns:
            Timeliness score (0.0 to 1.0)
        """
        if deadline is None:
            # If no deadline specified, use review's deadline or assume on-time
            deadline = review.deadline or review.submission_timestamp
        
        # Calculate days difference (negative = early, positive = late)
        days_diff = (review.submission_timestamp - deadline).days
        
        # Score based on timeliness
        if days_diff <= self.EXCELLENT_TIMELINESS_DAYS:
            return 1.0  # Excellent - submitted early
        elif days_diff <= self.GOOD_TIMELINESS_DAYS:
            return 0.9  # Good - on time
        elif days_diff <= 1:
            return 0.7  # Acceptable - 1 day late
        elif days_diff <= self.POOR_TIMELINESS_DAYS:
            return 0.5  # Poor - 2-3 days late
        else:
            # Very poor - more than 3 days late
            # Exponential decay for very late submissions
            return max(0.1, 0.5 * (0.8 ** (days_diff - self.POOR_TIMELINESS_DAYS)))
    
    def _update_reviewer_performance(self, reviewer_id: str, quality_metric: ReviewQualityMetric,
                                   review: StructuredReview, deadline: Optional[datetime]):
        """Update performance metrics for a reviewer."""
        if reviewer_id not in self.performance_metrics:
            self.performance_metrics[reviewer_id] = ReviewPerformanceMetrics(reviewer_id=reviewer_id)
        
        metrics = self.performance_metrics[reviewer_id]
        
        # Update basic counts
        metrics.total_reviews += 1
        
        # Update timeliness tracking
        if deadline:
            days_diff = (review.submission_timestamp - deadline).days
            if days_diff <= 0:
                metrics.on_time_reviews += 1
            else:
                metrics.late_reviews += 1
            
            # Update average days early/late
            total_days = metrics.avg_days_early_late * (metrics.total_reviews - 1) + days_diff
            metrics.avg_days_early_late = total_days / metrics.total_reviews
        
        # Update quality breakdown
        if quality_metric.quality_score >= self.HIGH_QUALITY_THRESHOLD:
            metrics.high_quality_reviews += 1
        elif quality_metric.quality_score >= self.MEDIUM_QUALITY_THRESHOLD:
            metrics.medium_quality_reviews += 1
        else:
            metrics.low_quality_reviews += 1
        
        # Recalculate averages and reliability
        self._recalculate_performance_metrics(reviewer_id)
        
        # Save updated metrics
        self._save_performance_data()
    
    def _recalculate_performance_metrics(self, reviewer_id: str):
        """Recalculate comprehensive performance metrics for a reviewer."""
        if reviewer_id not in self.performance_metrics:
            return
        
        metrics = self.performance_metrics[reviewer_id]
        
        # Get all quality metrics for this reviewer
        quality_scores = []
        timeliness_scores = []
        
        # Load historical data to recalculate averages
        history_file = self.data_dir / f"{reviewer_id}_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                    for record in history_data.get('reviews', []):
                        quality_scores.append(record.get('quality_score', 0.0))
                        timeliness_scores.append(record.get('timeliness_score', 0.0))
            except Exception as e:
                logger.warning(f"Error loading history for {reviewer_id}: {e}")
        
        # If no historical data, create temporary data from current metrics
        if not quality_scores and metrics.total_reviews > 0:
            # Estimate scores based on current metrics
            high_quality_ratio = metrics.high_quality_reviews / metrics.total_reviews
            medium_quality_ratio = metrics.medium_quality_reviews / metrics.total_reviews
            low_quality_ratio = metrics.low_quality_reviews / metrics.total_reviews
            
            # Estimate average quality score
            estimated_quality = (high_quality_ratio * 0.9 + 
                               medium_quality_ratio * 0.65 + 
                               low_quality_ratio * 0.35)
            quality_scores = [estimated_quality] * metrics.total_reviews
            
            # Estimate timeliness scores
            on_time_ratio = metrics.on_time_reviews / metrics.total_reviews if metrics.total_reviews > 0 else 0.5
            estimated_timeliness = on_time_ratio * 0.9 + (1 - on_time_ratio) * 0.5
            timeliness_scores = [estimated_timeliness] * metrics.total_reviews
        
        # Calculate averages
        if quality_scores:
            metrics.avg_quality_score = statistics.mean(quality_scores)
            metrics.quality_consistency = statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0
        
        if timeliness_scores:
            metrics.avg_timeliness_score = statistics.mean(timeliness_scores)
            metrics.timeliness_consistency = statistics.stdev(timeliness_scores) if len(timeliness_scores) > 1 else 0.0
        
        # Calculate overall reliability score
        metrics.reliability_score = (
            metrics.avg_quality_score * self.QUALITY_WEIGHT +
            metrics.avg_timeliness_score * self.TIMELINESS_WEIGHT
        )
        
        # Determine reliability category
        metrics.reliability_category = self._determine_reliability_category(metrics.reliability_score)
        
        # Analyze performance trend
        metrics.recent_performance_trend = self._analyze_performance_trend(quality_scores, timeliness_scores)
        
        metrics.last_updated = datetime.now()
    
    def _determine_reliability_category(self, reliability_score: float) -> ReliabilityCategory:
        """Determine reliability category based on score."""
        if reliability_score >= 0.9:
            return ReliabilityCategory.EXCELLENT
        elif reliability_score >= 0.8:
            return ReliabilityCategory.GOOD
        elif reliability_score >= 0.6:
            return ReliabilityCategory.AVERAGE
        elif reliability_score >= 0.4:
            return ReliabilityCategory.POOR
        else:
            return ReliabilityCategory.UNRELIABLE
    
    def _analyze_performance_trend(self, quality_scores: List[float], timeliness_scores: List[float]) -> str:
        """Analyze recent performance trend."""
        if len(quality_scores) < 4:
            return "stable"  # Not enough data for trend analysis
        
        # Look at recent reviews (last 25% or minimum 3)
        recent_count = max(3, len(quality_scores) // 4)
        recent_quality = quality_scores[-recent_count:]
        recent_timeliness = timeliness_scores[-recent_count:] if timeliness_scores else []
        
        # Compare recent performance to earlier performance
        earlier_quality = quality_scores[:-recent_count] if len(quality_scores) > recent_count else quality_scores
        earlier_timeliness = timeliness_scores[:-recent_count] if len(timeliness_scores) > recent_count else timeliness_scores
        
        if not earlier_quality:
            return "stable"
        
        recent_avg_quality = statistics.mean(recent_quality)
        earlier_avg_quality = statistics.mean(earlier_quality)
        
        recent_avg_timeliness = statistics.mean(recent_timeliness) if recent_timeliness else 0.5
        earlier_avg_timeliness = statistics.mean(earlier_timeliness) if earlier_timeliness else 0.5
        
        # Calculate overall trend
        quality_trend = recent_avg_quality - earlier_avg_quality
        timeliness_trend = recent_avg_timeliness - earlier_avg_timeliness
        
        overall_trend = quality_trend * self.QUALITY_WEIGHT + timeliness_trend * self.TIMELINESS_WEIGHT
        
        if overall_trend > 0.1:
            return "improving"
        elif overall_trend < -0.1:
            return "declining"
        else:
            return "stable"
    
    def get_reviewer_performance(self, reviewer_id: str) -> Optional[ReviewPerformanceMetrics]:
        """
        Get performance metrics for a specific reviewer.
        
        Args:
            reviewer_id: ID of the reviewer
            
        Returns:
            ReviewPerformanceMetrics object or None if not found
        """
        return self.performance_metrics.get(reviewer_id)
    
    def get_reliability_score(self, reviewer_id: str) -> float:
        """
        Get reliability score for a reviewer.
        
        Args:
            reviewer_id: ID of the reviewer
            
        Returns:
            Reliability score (0.0 to 1.0)
        """
        metrics = self.get_reviewer_performance(reviewer_id)
        return metrics.reliability_score if metrics else 0.0
    
    def get_top_reviewers(self, limit: int = 10, min_reviews: int = 5) -> List[Tuple[str, ReviewPerformanceMetrics]]:
        """
        Get top reviewers by reliability score.
        
        Args:
            limit: Maximum number of reviewers to return
            min_reviews: Minimum number of reviews required
            
        Returns:
            List of (reviewer_id, metrics) tuples sorted by reliability score
        """
        eligible_reviewers = [
            (reviewer_id, metrics) for reviewer_id, metrics in self.performance_metrics.items()
            if metrics.total_reviews >= min_reviews
        ]
        
        # Sort by reliability score (descending)
        eligible_reviewers.sort(key=lambda x: x[1].reliability_score, reverse=True)
        
        return eligible_reviewers[:limit]
    
    def get_reviewers_needing_improvement(self, threshold: float = 0.6) -> List[Tuple[str, ReviewPerformanceMetrics]]:
        """
        Get reviewers with reliability scores below threshold.
        
        Args:
            threshold: Reliability score threshold
            
        Returns:
            List of (reviewer_id, metrics) tuples for reviewers needing improvement
        """
        return [
            (reviewer_id, metrics) for reviewer_id, metrics in self.performance_metrics.items()
            if metrics.reliability_score < threshold and metrics.total_reviews >= 3
        ]
    
    def generate_performance_report(self, reviewer_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive performance report for a reviewer.
        
        Args:
            reviewer_id: ID of the reviewer
            
        Returns:
            Dictionary with detailed performance report
        """
        metrics = self.get_reviewer_performance(reviewer_id)
        if not metrics:
            return {'error': f'No performance data found for reviewer {reviewer_id}'}
        
        # Calculate percentiles compared to all reviewers
        all_reliability_scores = [m.reliability_score for m in self.performance_metrics.values() if m.total_reviews >= 3]
        reliability_percentile = self._calculate_percentile(metrics.reliability_score, all_reliability_scores)
        
        all_quality_scores = [m.avg_quality_score for m in self.performance_metrics.values() if m.total_reviews >= 3]
        quality_percentile = self._calculate_percentile(metrics.avg_quality_score, all_quality_scores)
        
        all_timeliness_scores = [m.avg_timeliness_score for m in self.performance_metrics.values() if m.total_reviews >= 3]
        timeliness_percentile = self._calculate_percentile(metrics.avg_timeliness_score, all_timeliness_scores)
        
        report = {
            'reviewer_id': reviewer_id,
            'summary': {
                'total_reviews': metrics.total_reviews,
                'reliability_score': metrics.reliability_score,
                'reliability_category': metrics.reliability_category.value,
                'performance_trend': metrics.recent_performance_trend
            },
            'quality_metrics': {
                'avg_quality_score': metrics.avg_quality_score,
                'quality_percentile': quality_percentile,
                'quality_consistency': metrics.quality_consistency,
                'high_quality_reviews': metrics.high_quality_reviews,
                'medium_quality_reviews': metrics.medium_quality_reviews,
                'low_quality_reviews': metrics.low_quality_reviews
            },
            'timeliness_metrics': {
                'avg_timeliness_score': metrics.avg_timeliness_score,
                'timeliness_percentile': timeliness_percentile,
                'timeliness_consistency': metrics.timeliness_consistency,
                'on_time_reviews': metrics.on_time_reviews,
                'late_reviews': metrics.late_reviews,
                'avg_days_early_late': metrics.avg_days_early_late
            },
            'percentiles': {
                'reliability': reliability_percentile,
                'quality': quality_percentile,
                'timeliness': timeliness_percentile
            },
            'recommendations': self._generate_improvement_recommendations(metrics),
            'last_updated': metrics.last_updated.isoformat()
        }
        
        return report
    
    def _calculate_percentile(self, value: float, all_values: List[float]) -> float:
        """Calculate percentile rank of a value within a list."""
        if not all_values:
            return 0.5
        
        sorted_values = sorted(all_values)
        rank = sum(1 for v in sorted_values if v < value)
        return rank / len(sorted_values)
    
    def _generate_improvement_recommendations(self, metrics: ReviewPerformanceMetrics) -> List[str]:
        """Generate improvement recommendations based on performance metrics."""
        recommendations = []
        
        # Quality recommendations
        if metrics.avg_quality_score < 0.6:
            recommendations.append("Focus on providing more detailed and constructive feedback")
            recommendations.append("Ensure all required review sections are thoroughly completed")
        
        if metrics.quality_consistency > 0.3:
            recommendations.append("Work on maintaining consistent review quality across all reviews")
        
        # Timeliness recommendations
        if metrics.avg_timeliness_score < 0.7:
            recommendations.append("Improve review submission timeliness - aim to submit reviews early")
            recommendations.append("Better time management and deadline tracking needed")
        
        if metrics.late_reviews > metrics.on_time_reviews:
            recommendations.append("Prioritize meeting review deadlines to maintain reliability")
        
        # Trend-based recommendations
        if metrics.recent_performance_trend == "declining":
            recommendations.append("Recent performance shows decline - consider reducing review load temporarily")
            recommendations.append("Review recent submissions to identify areas for improvement")
        
        # Category-specific recommendations
        if metrics.reliability_category == ReliabilityCategory.POOR:
            recommendations.append("Consider taking a break from reviewing to focus on improvement")
            recommendations.append("Seek mentorship or training on effective peer review practices")
        elif metrics.reliability_category == ReliabilityCategory.UNRELIABLE:
            recommendations.append("Immediate improvement needed - consider suspension from review duties")
        
        return recommendations
    
    def update_review_helpfulness(self, review_id: str, helpfulness_rating: float):
        """
        Update helpfulness rating for a review (from authors' feedback).
        
        Args:
            review_id: ID of the review
            helpfulness_rating: Rating from authors (0.0 to 1.0)
        """
        # Find the review in history and update helpfulness
        for reviewer_id, metrics in self.performance_metrics.items():
            history_file = self.data_dir / f"{reviewer_id}_history.json"
            if history_file.exists():
                try:
                    with open(history_file, 'r') as f:
                        history_data = json.load(f)
                    
                    # Update the specific review
                    for record in history_data.get('reviews', []):
                        if record.get('review_id') == review_id:
                            record['helpfulness_rating'] = helpfulness_rating
                            break
                    
                    # Save updated history
                    with open(history_file, 'w') as f:
                        json.dump(history_data, f, indent=2)
                    
                    logger.info(f"Updated helpfulness rating for review {review_id}: {helpfulness_rating}")
                    return
                    
                except Exception as e:
                    logger.error(f"Error updating helpfulness rating: {e}")
        
        logger.warning(f"Review {review_id} not found for helpfulness update")
    
    def _save_performance_data(self):
        """Save performance metrics to disk."""
        try:
            # Save overall metrics
            metrics_file = self.data_dir / "performance_metrics.json"
            metrics_data = {
                reviewer_id: metrics.to_dict()
                for reviewer_id, metrics in self.performance_metrics.items()
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.debug("Saved performance metrics to disk")
            
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def _load_performance_data(self):
        """Load performance metrics from disk."""
        try:
            metrics_file = self.data_dir / "performance_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                self.performance_metrics = {
                    reviewer_id: ReviewPerformanceMetrics.from_dict(data)
                    for reviewer_id, data in metrics_data.items()
                }
                
                logger.info(f"Loaded performance metrics for {len(self.performance_metrics)} reviewers")
            
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get overall system statistics for review tracking.
        
        Returns:
            Dictionary with system-wide statistics
        """
        if not self.performance_metrics:
            return {'total_reviewers': 0, 'total_reviews': 0}
        
        total_reviews = sum(m.total_reviews for m in self.performance_metrics.values())
        avg_reliability = statistics.mean([m.reliability_score for m in self.performance_metrics.values()])
        
        reliability_distribution = {}
        for category in ReliabilityCategory:
            count = sum(1 for m in self.performance_metrics.values() if m.reliability_category == category)
            reliability_distribution[category.value] = count
        
        return {
            'total_reviewers': len(self.performance_metrics),
            'total_reviews': total_reviews,
            'avg_reliability_score': avg_reliability,
            'reliability_distribution': reliability_distribution,
            'reviewers_with_good_reliability': sum(1 for m in self.performance_metrics.values() 
                                                 if m.reliability_score >= 0.8),
            'reviewers_needing_improvement': sum(1 for m in self.performance_metrics.values() 
                                               if m.reliability_score < 0.6)
        }