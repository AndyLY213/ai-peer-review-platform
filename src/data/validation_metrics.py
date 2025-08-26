"""
Real Data Validation Framework

This module implements ValidationMetrics class to compare simulation vs. real data patterns,
statistical comparison utilities (KL divergence, Wasserstein distance, correlation analysis),
baseline statistics calculator from PeerRead training data, continuous validation monitoring
with automated alerts for deviations, and realism indicators for review quality, reviewer
behavior, and venue characteristics.
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import entropy, pearsonr, spearmanr, ks_2samp, wasserstein_distance
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import warnings
from collections import defaultdict, Counter
import statistics

from src.core.exceptions import ValidationError, DatasetError
from src.core.logging_config import get_logger
from src.data.peerread_loader import PeerReadLoader, VenueCharacteristics, PeerReadReview, PeerReadPaper
from src.data.enhanced_models import StructuredReview, EnhancedResearcher, EnhancedVenue


logger = get_logger(__name__)


@dataclass
class StatisticalComparison:
    """Results of statistical comparison between simulation and real data."""
    metric_name: str
    simulation_values: List[float]
    real_values: List[float]
    
    # Statistical measures
    kl_divergence: Optional[float] = None
    wasserstein_distance: Optional[float] = None
    pearson_correlation: Optional[Tuple[float, float]] = None  # (correlation, p-value)
    spearman_correlation: Optional[Tuple[float, float]] = None  # (correlation, p-value)
    ks_statistic: Optional[Tuple[float, float]] = None  # (statistic, p-value)
    
    # Descriptive statistics
    simulation_mean: float = 0.0
    real_mean: float = 0.0
    simulation_std: float = 0.0
    real_std: float = 0.0
    
    # Validation results
    is_similar: bool = False
    similarity_score: float = 0.0  # 0-1 scale
    deviation_level: str = "UNKNOWN"  # LOW, MEDIUM, HIGH, CRITICAL
    
    def __post_init__(self):
        """Calculate all statistical measures."""
        self._calculate_descriptive_stats()
        self._calculate_statistical_measures()
        self._assess_similarity()
    
    def _calculate_descriptive_stats(self):
        """Calculate basic descriptive statistics."""
        if self.simulation_values:
            self.simulation_mean = statistics.mean(self.simulation_values)
            self.simulation_std = statistics.stdev(self.simulation_values) if len(self.simulation_values) > 1 else 0.0
        
        if self.real_values:
            self.real_mean = statistics.mean(self.real_values)
            self.real_std = statistics.stdev(self.real_values) if len(self.real_values) > 1 else 0.0
    
    def _calculate_statistical_measures(self):
        """Calculate advanced statistical comparison measures."""
        if not self.simulation_values or not self.real_values:
            return
        
        try:
            # KL Divergence (requires probability distributions)
            self.kl_divergence = self._calculate_kl_divergence()
            
            # Wasserstein Distance
            self.wasserstein_distance = wasserstein_distance(self.simulation_values, self.real_values)
            
            # Correlation measures
            if len(self.simulation_values) == len(self.real_values):
                self.pearson_correlation = pearsonr(self.simulation_values, self.real_values)
                self.spearman_correlation = spearmanr(self.simulation_values, self.real_values)
            
            # Kolmogorov-Smirnov test
            self.ks_statistic = ks_2samp(self.simulation_values, self.real_values)
            
        except Exception as e:
            logger.warning(f"Error calculating statistical measures for {self.metric_name}: {e}")
    
    def _calculate_kl_divergence(self) -> Optional[float]:
        """Calculate KL divergence between distributions."""
        try:
            # Create histograms with same bins
            min_val = min(min(self.simulation_values), min(self.real_values))
            max_val = max(max(self.simulation_values), max(self.real_values))
            bins = np.linspace(min_val, max_val, 20)
            
            sim_hist, _ = np.histogram(self.simulation_values, bins=bins, density=True)
            real_hist, _ = np.histogram(self.real_values, bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            sim_hist = sim_hist + epsilon
            real_hist = real_hist + epsilon
            
            # Normalize to probabilities
            sim_hist = sim_hist / np.sum(sim_hist)
            real_hist = real_hist / np.sum(real_hist)
            
            return entropy(sim_hist, real_hist)
        
        except Exception as e:
            logger.warning(f"Could not calculate KL divergence: {e}")
            return None
    
    def _assess_similarity(self):
        """Assess overall similarity and deviation level."""
        similarity_factors = []
        
        # Mean difference factor (normalized)
        if self.real_mean != 0:
            mean_diff = abs(self.simulation_mean - self.real_mean) / abs(self.real_mean)
            similarity_factors.append(max(0, 1 - mean_diff))
        
        # Standard deviation similarity
        if self.real_std != 0:
            std_diff = abs(self.simulation_std - self.real_std) / self.real_std
            similarity_factors.append(max(0, 1 - std_diff))
        
        # KS test p-value (higher p-value = more similar)
        if self.ks_statistic and self.ks_statistic[1] is not None:
            similarity_factors.append(self.ks_statistic[1])
        
        # Wasserstein distance (lower = more similar, normalize by data range)
        if self.wasserstein_distance is not None:
            data_range = max(max(self.simulation_values), max(self.real_values)) - \
                        min(min(self.simulation_values), min(self.real_values))
            if data_range > 0:
                normalized_wasserstein = 1 - (self.wasserstein_distance / data_range)
                similarity_factors.append(max(0, normalized_wasserstein))
        
        # Calculate overall similarity score
        if similarity_factors:
            self.similarity_score = statistics.mean(similarity_factors)
        
        # Determine deviation level
        if self.similarity_score >= 0.8:
            self.deviation_level = "LOW"
            self.is_similar = True
        elif self.similarity_score >= 0.6:
            self.deviation_level = "MEDIUM"
            self.is_similar = True
        elif self.similarity_score >= 0.4:
            self.deviation_level = "HIGH"
            self.is_similar = False
        else:
            self.deviation_level = "CRITICAL"
            self.is_similar = False


@dataclass
class RealismIndicator:
    """Indicator of realism for a specific aspect of the simulation."""
    aspect_name: str  # e.g., "review_quality", "reviewer_behavior", "venue_characteristics"
    indicator_type: str  # e.g., "score_distribution", "length_distribution", "acceptance_rate"
    
    current_value: float
    expected_value: float
    tolerance: float = 0.1  # Acceptable deviation (10% by default)
    
    is_realistic: bool = field(init=False)
    deviation_percentage: float = field(init=False)
    alert_level: str = field(init=False)  # OK, WARNING, CRITICAL
    
    def __post_init__(self):
        """Calculate realism assessment."""
        self._assess_realism()
    
    def _assess_realism(self):
        """Assess whether the current value is realistic."""
        if self.expected_value == 0:
            self.deviation_percentage = abs(self.current_value)
        else:
            self.deviation_percentage = abs(self.current_value - self.expected_value) / abs(self.expected_value)
        
        self.is_realistic = self.deviation_percentage <= self.tolerance
        
        if self.deviation_percentage <= self.tolerance:
            self.alert_level = "OK"
        elif self.deviation_percentage <= self.tolerance * 2:
            self.alert_level = "WARNING"
        else:
            self.alert_level = "CRITICAL"


@dataclass
class ValidationAlert:
    """Alert for validation deviations."""
    timestamp: datetime
    alert_level: str  # WARNING, CRITICAL
    aspect: str
    metric: str
    message: str
    current_value: float
    expected_value: float
    deviation_percentage: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'alert_level': self.alert_level,
            'aspect': self.aspect,
            'metric': self.metric,
            'message': self.message,
            'current_value': self.current_value,
            'expected_value': self.expected_value,
            'deviation_percentage': self.deviation_percentage
        }


@dataclass
class BaselineStatistics:
    """Baseline statistics calculated from PeerRead training data."""
    venue_name: str
    
    # Review score statistics
    score_means: Dict[str, float] = field(default_factory=dict)
    score_stds: Dict[str, float] = field(default_factory=dict)
    score_distributions: Dict[str, List[float]] = field(default_factory=dict)
    
    # Review length statistics
    review_length_mean: float = 0.0
    review_length_std: float = 0.0
    review_length_distribution: List[int] = field(default_factory=list)
    
    # Venue characteristics
    acceptance_rate: float = 0.0
    reviews_per_paper_mean: float = 0.0
    reviews_per_paper_std: float = 0.0
    
    # Reviewer behavior patterns
    confidence_distribution: List[int] = field(default_factory=list)
    recommendation_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Temporal patterns
    review_submission_patterns: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'venue_name': self.venue_name,
            'score_means': self.score_means,
            'score_stds': self.score_stds,
            'score_distributions': self.score_distributions,
            'review_length_mean': self.review_length_mean,
            'review_length_std': self.review_length_std,
            'review_length_distribution': self.review_length_distribution,
            'acceptance_rate': self.acceptance_rate,
            'reviews_per_paper_mean': self.reviews_per_paper_mean,
            'reviews_per_paper_std': self.reviews_per_paper_std,
            'confidence_distribution': self.confidence_distribution,
            'recommendation_distribution': self.recommendation_distribution,
            'review_submission_patterns': self.review_submission_patterns
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaselineStatistics':
        """Create from dictionary."""
        return cls(**data)


class ValidationMetrics:
    """
    ValidationMetrics class to compare simulation vs. real data patterns with statistical
    comparison utilities, baseline statistics calculator, continuous validation monitoring,
    and realism indicators.
    """
    
    def __init__(self, peerread_loader: Optional[PeerReadLoader] = None):
        """
        Initialize ValidationMetrics.
        
        Args:
            peerread_loader: Optional PeerReadLoader instance for real data
        """
        self.peerread_loader = peerread_loader or PeerReadLoader()
        self.baseline_statistics: Dict[str, BaselineStatistics] = {}
        self.validation_alerts: List[ValidationAlert] = []
        self.monitoring_enabled = True
        
        # Thresholds for automated alerts
        self.alert_thresholds = {
            'score_deviation': 0.2,  # 20% deviation triggers warning
            'length_deviation': 0.3,  # 30% deviation triggers warning
            'acceptance_rate_deviation': 0.15,  # 15% deviation triggers warning
            'critical_threshold_multiplier': 2.0  # 2x warning threshold = critical
        }
        
        self._initialize_baseline_statistics()
    
    def _initialize_baseline_statistics(self):
        """Initialize baseline statistics from PeerRead data."""
        try:
            venues = self.peerread_loader.load_all_venues()
            for venue_name, venue_chars in venues.items():
                self.baseline_statistics[venue_name] = self._calculate_baseline_from_venue(venue_chars)
            logger.info(f"Initialized baseline statistics for {len(self.baseline_statistics)} venues")
        except Exception as e:
            logger.error(f"Failed to initialize baseline statistics: {e}")
    
    def _calculate_baseline_from_venue(self, venue_chars: VenueCharacteristics) -> BaselineStatistics:
        """Calculate baseline statistics from venue characteristics."""
        baseline = BaselineStatistics(venue_name=venue_chars.name)
        
        # Score statistics
        score_fields = ['impact', 'substance', 'soundness', 'originality', 'clarity', 'meaningful_comparison']
        for field in score_fields:
            scores = getattr(venue_chars, f"{field}_scores", [])
            if scores:
                baseline.score_means[field] = statistics.mean(scores)
                baseline.score_stds[field] = statistics.stdev(scores) if len(scores) > 1 else 0.0
                baseline.score_distributions[field] = scores.copy()
        
        # Review length statistics
        papers = self.peerread_loader.get_papers_by_venue(venue_chars.name)
        if papers:
            review_lengths = []
            confidence_scores = []
            recommendations = []
            
            for paper in papers:
                for review in paper.reviews:
                    if review.comments:
                        review_lengths.append(len(review.comments))
                    if review.reviewer_confidence:
                        confidence_scores.append(review.reviewer_confidence)
                    if review.recommendation:
                        recommendations.append(review.recommendation)
            
            if review_lengths:
                baseline.review_length_mean = statistics.mean(review_lengths)
                baseline.review_length_std = statistics.stdev(review_lengths) if len(review_lengths) > 1 else 0.0
                baseline.review_length_distribution = review_lengths
            
            if confidence_scores:
                baseline.confidence_distribution = confidence_scores
            
            if recommendations:
                rec_counts = Counter(recommendations)
                total_recs = sum(rec_counts.values())
                baseline.recommendation_distribution = {
                    str(k): v / total_recs for k, v in rec_counts.items()
                }
        
        # Venue characteristics
        baseline.acceptance_rate = venue_chars.acceptance_rate
        baseline.reviews_per_paper_mean = venue_chars.avg_reviews_per_paper
        
        return baseline
    
    def compare_simulation_to_real(self, 
                                 simulation_data: Dict[str, List[float]], 
                                 venue_name: str) -> Dict[str, StatisticalComparison]:
        """
        Compare simulation data to real data patterns.
        
        Args:
            simulation_data: Dictionary with metric names as keys and lists of values
            venue_name: Name of venue to compare against
            
        Returns:
            Dictionary of statistical comparisons
        """
        if venue_name not in self.baseline_statistics:
            raise ValidationError("venue_name", venue_name, "venue with baseline statistics")
        
        baseline = self.baseline_statistics[venue_name]
        comparisons = {}
        
        for metric_name, sim_values in simulation_data.items():
            real_values = self._get_real_values_for_metric(metric_name, baseline)
            
            if real_values:
                comparison = StatisticalComparison(
                    metric_name=metric_name,
                    simulation_values=sim_values,
                    real_values=real_values
                )
                comparisons[metric_name] = comparison
                
                # Generate alerts if monitoring is enabled
                if self.monitoring_enabled:
                    self._check_for_alerts(comparison, venue_name)
        
        return comparisons
    
    def _get_real_values_for_metric(self, metric_name: str, baseline: BaselineStatistics) -> List[float]:
        """Get real values for a specific metric from baseline statistics."""
        if metric_name in baseline.score_distributions:
            return [float(x) for x in baseline.score_distributions[metric_name]]
        elif metric_name == "review_length":
            return [float(x) for x in baseline.review_length_distribution]
        elif metric_name == "confidence":
            return [float(x) for x in baseline.confidence_distribution]
        elif metric_name in baseline.score_means:
            # If we don't have distribution, create synthetic data around mean
            mean = baseline.score_means[metric_name]
            std = baseline.score_stds.get(metric_name, mean * 0.2)
            return [mean + np.random.normal(0, std) for _ in range(100)]
        else:
            logger.warning(f"No real data available for metric: {metric_name}")
            return []
    
    def calculate_kl_divergence(self, sim_values: List[float], real_values: List[float]) -> Optional[float]:
        """
        Calculate KL divergence between simulation and real data distributions.
        
        Args:
            sim_values: Simulation values
            real_values: Real data values
            
        Returns:
            KL divergence value or None if calculation fails
        """
        try:
            comparison = StatisticalComparison("temp", sim_values, real_values)
            return comparison.kl_divergence
        except Exception as e:
            logger.error(f"Failed to calculate KL divergence: {e}")
            return None
    
    def calculate_wasserstein_distance(self, sim_values: List[float], real_values: List[float]) -> Optional[float]:
        """
        Calculate Wasserstein distance between simulation and real data.
        
        Args:
            sim_values: Simulation values
            real_values: Real data values
            
        Returns:
            Wasserstein distance or None if calculation fails
        """
        try:
            return wasserstein_distance(sim_values, real_values)
        except Exception as e:
            logger.error(f"Failed to calculate Wasserstein distance: {e}")
            return None
    
    def calculate_correlation_analysis(self, sim_values: List[float], real_values: List[float]) -> Dict[str, Tuple[float, float]]:
        """
        Calculate correlation analysis between simulation and real data.
        
        Args:
            sim_values: Simulation values
            real_values: Real data values
            
        Returns:
            Dictionary with correlation results
        """
        results = {}
        
        try:
            if len(sim_values) == len(real_values):
                results['pearson'] = pearsonr(sim_values, real_values)
                results['spearman'] = spearmanr(sim_values, real_values)
            else:
                logger.warning("Cannot calculate correlation: different array lengths")
        except Exception as e:
            logger.error(f"Failed to calculate correlations: {e}")
        
        return results
    
    def calculate_baseline_statistics_from_peerread(self, venue_name: str) -> BaselineStatistics:
        """
        Calculate baseline statistics from PeerRead training data for a specific venue.
        
        Args:
            venue_name: Name of venue to calculate statistics for
            
        Returns:
            BaselineStatistics object
        """
        venue_chars = self.peerread_loader.get_venue_statistics(venue_name)
        if not venue_chars:
            raise ValidationError("venue_name", venue_name, "valid venue in PeerRead dataset")
        
        baseline = self._calculate_baseline_from_venue(venue_chars)
        self.baseline_statistics[venue_name] = baseline
        return baseline
    
    def enable_continuous_monitoring(self, enable: bool = True):
        """
        Enable or disable continuous validation monitoring.
        
        Args:
            enable: Whether to enable monitoring
        """
        self.monitoring_enabled = enable
        logger.info(f"Continuous monitoring {'enabled' if enable else 'disabled'}")
    
    def _check_for_alerts(self, comparison: StatisticalComparison, venue_name: str):
        """Check for validation alerts based on comparison results."""
        current_time = datetime.now()
        
        # Check mean deviation
        if comparison.real_mean != 0:
            mean_deviation = abs(comparison.simulation_mean - comparison.real_mean) / abs(comparison.real_mean)
            
            if mean_deviation > self.alert_thresholds['score_deviation']:
                alert_level = "WARNING"
                if mean_deviation > self.alert_thresholds['score_deviation'] * self.alert_thresholds['critical_threshold_multiplier']:
                    alert_level = "CRITICAL"
                
                alert = ValidationAlert(
                    timestamp=current_time,
                    alert_level=alert_level,
                    aspect=venue_name,
                    metric=f"{comparison.metric_name}_mean",
                    message=f"Mean deviation of {mean_deviation:.2%} detected for {comparison.metric_name}",
                    current_value=comparison.simulation_mean,
                    expected_value=comparison.real_mean,
                    deviation_percentage=mean_deviation
                )
                self.validation_alerts.append(alert)
                logger.warning(f"Validation alert: {alert.message}")
        
        # Check similarity score
        if comparison.similarity_score < 0.6:  # Below medium similarity
            alert_level = "WARNING" if comparison.similarity_score >= 0.4 else "CRITICAL"
            
            alert = ValidationAlert(
                timestamp=current_time,
                alert_level=alert_level,
                aspect=venue_name,
                metric=f"{comparison.metric_name}_similarity",
                message=f"Low similarity score ({comparison.similarity_score:.2f}) for {comparison.metric_name}",
                current_value=comparison.similarity_score,
                expected_value=0.8,  # Target similarity
                deviation_percentage=(0.8 - comparison.similarity_score) / 0.8
            )
            self.validation_alerts.append(alert)
            logger.warning(f"Validation alert: {alert.message}")
    
    def create_realism_indicators(self, 
                                simulation_reviews: List[StructuredReview],
                                simulation_researchers: List[EnhancedResearcher],
                                simulation_venues: List[EnhancedVenue]) -> Dict[str, List[RealismIndicator]]:
        """
        Create realism indicators for review quality, reviewer behavior, and venue characteristics.
        
        Args:
            simulation_reviews: List of simulation reviews
            simulation_researchers: List of simulation researchers
            simulation_venues: List of simulation venues
            
        Returns:
            Dictionary of realism indicators by aspect
        """
        indicators = {
            'review_quality': [],
            'reviewer_behavior': [],
            'venue_characteristics': []
        }
        
        # Review quality indicators
        indicators['review_quality'].extend(self._create_review_quality_indicators(simulation_reviews))
        
        # Reviewer behavior indicators
        indicators['reviewer_behavior'].extend(self._create_reviewer_behavior_indicators(simulation_researchers))
        
        # Venue characteristics indicators
        indicators['venue_characteristics'].extend(self._create_venue_characteristics_indicators(simulation_venues))
        
        return indicators
    
    def _create_review_quality_indicators(self, reviews: List[StructuredReview]) -> List[RealismIndicator]:
        """Create realism indicators for review quality."""
        indicators = []
        
        if not reviews:
            return indicators
        
        # Average review length
        review_lengths = [r.review_length for r in reviews]
        avg_length = statistics.mean(review_lengths)
        
        # Compare with baseline (using average across all venues)
        expected_length = 0
        count = 0
        for baseline in self.baseline_statistics.values():
            if baseline.review_length_mean > 0:
                expected_length += baseline.review_length_mean
                count += 1
        
        if count > 0:
            expected_length /= count
            indicators.append(RealismIndicator(
                aspect_name="review_quality",
                indicator_type="average_length",
                current_value=avg_length,
                expected_value=expected_length,
                tolerance=0.3  # 30% tolerance
            ))
        
        # Score distribution indicators
        score_fields = ['novelty', 'technical_quality', 'clarity', 'significance', 'reproducibility', 'related_work']
        for field in score_fields:
            scores = [getattr(r.criteria_scores, field) for r in reviews]
            if scores:
                avg_score = statistics.mean(scores)
                
                # Expected score (average from baselines, mapped from 1-5 to 1-10 scale)
                expected_score = 0
                count = 0
                for baseline in self.baseline_statistics.values():
                    # Map PeerRead field names to our field names
                    peerread_field = self._map_field_to_peerread(field)
                    if peerread_field in baseline.score_means:
                        expected_score += baseline.score_means[peerread_field] * 2.0  # Scale 1-5 to 1-10
                        count += 1
                
                if count > 0:
                    expected_score /= count
                    indicators.append(RealismIndicator(
                        aspect_name="review_quality",
                        indicator_type=f"{field}_score",
                        current_value=avg_score,
                        expected_value=expected_score,
                        tolerance=0.2  # 20% tolerance
                    ))
        
        return indicators
    
    def _create_reviewer_behavior_indicators(self, researchers: List[EnhancedResearcher]) -> List[RealismIndicator]:
        """Create realism indicators for reviewer behavior."""
        indicators = []
        
        if not researchers:
            return indicators
        
        # H-index distribution
        h_indices = [r.h_index for r in researchers]
        avg_h_index = statistics.mean(h_indices)
        
        # Expected h-index varies by career level, use reasonable baseline
        expected_h_index = 15  # Reasonable average for academic reviewers
        
        indicators.append(RealismIndicator(
            aspect_name="reviewer_behavior",
            indicator_type="average_h_index",
            current_value=avg_h_index,
            expected_value=expected_h_index,
            tolerance=0.4  # 40% tolerance
        ))
        
        # Years active distribution
        years_active = [r.years_active for r in researchers]
        avg_years = statistics.mean(years_active)
        expected_years = 10  # Reasonable average
        
        indicators.append(RealismIndicator(
            aspect_name="reviewer_behavior",
            indicator_type="average_years_active",
            current_value=avg_years,
            expected_value=expected_years,
            tolerance=0.5  # 50% tolerance
        ))
        
        return indicators
    
    def _create_venue_characteristics_indicators(self, venues: List[EnhancedVenue]) -> List[RealismIndicator]:
        """Create realism indicators for venue characteristics."""
        indicators = []
        
        for venue in venues:
            # Check if we have baseline for this venue
            baseline = None
            for baseline_venue in self.baseline_statistics.values():
                if baseline_venue.venue_name.lower() in venue.name.lower():
                    baseline = baseline_venue
                    break
            
            if baseline:
                # Acceptance rate indicator
                indicators.append(RealismIndicator(
                    aspect_name="venue_characteristics",
                    indicator_type=f"{venue.name}_acceptance_rate",
                    current_value=venue.acceptance_rate,
                    expected_value=baseline.acceptance_rate,
                    tolerance=0.15  # 15% tolerance
                ))
        
        return indicators
    
    def _map_field_to_peerread(self, field: str) -> str:
        """Map our field names to PeerRead field names."""
        mapping = {
            'novelty': 'originality',
            'technical_quality': 'substance',
            'clarity': 'clarity',
            'significance': 'impact',
            'reproducibility': 'soundness',  # Approximate mapping
            'related_work': 'meaningful_comparison'
        }
        return mapping.get(field, field)
    
    def get_recent_alerts(self, hours: int = 24) -> List[ValidationAlert]:
        """
        Get validation alerts from the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.validation_alerts if alert.timestamp >= cutoff_time]
    
    def clear_alerts(self):
        """Clear all validation alerts."""
        self.validation_alerts.clear()
        logger.info("Validation alerts cleared")
    
    def export_validation_report(self, output_path: str):
        """
        Export comprehensive validation report.
        
        Args:
            output_path: Path to save the report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'baseline_statistics': {name: baseline.to_dict() for name, baseline in self.baseline_statistics.items()},
            'recent_alerts': [alert.to_dict() for alert in self.get_recent_alerts(168)],  # Last week
            'alert_thresholds': self.alert_thresholds,
            'monitoring_enabled': self.monitoring_enabled
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report exported to {output_path}")
    
    def import_baseline_statistics(self, file_path: str):
        """
        Import baseline statistics from file.
        
        Args:
            file_path: Path to baseline statistics file
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.baseline_statistics = {
                name: BaselineStatistics.from_dict(baseline_data)
                for name, baseline_data in data.items()
            }
            
            logger.info(f"Imported baseline statistics for {len(self.baseline_statistics)} venues")
        
        except Exception as e:
            logger.error(f"Failed to import baseline statistics: {e}")
            raise ValidationError("file_path", file_path, "valid baseline statistics file")
    
    def export_baseline_statistics(self, file_path: str):
        """
        Export baseline statistics to file.
        
        Args:
            file_path: Path to save baseline statistics
        """
        data = {name: baseline.to_dict() for name, baseline in self.baseline_statistics.items()}
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported baseline statistics to {file_path}")