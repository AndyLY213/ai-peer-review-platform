"""
Venue Statistics and Tracking System with Continuous Calibration

This module implements comprehensive venue statistics tracking with continuous
calibration against PeerRead baseline data, including submission and acceptance
tracking, historical trend analysis, dynamic acceptance rate calculation,
and venue realism validation metrics.
"""

import json
import statistics
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
from collections import defaultdict, deque
import numpy as np
from scipy import stats as scipy_stats

from src.data.enhanced_models import EnhancedVenue, VenueType
from src.data.peerread_loader import PeerReadLoader, VenueCharacteristics
from src.core.exceptions import ValidationError, DatasetError
from src.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SubmissionRecord:
    """Record of a paper submission to a venue."""
    paper_id: str
    venue_id: str
    submission_date: datetime
    decision: Optional[str] = None  # "accept", "reject", "revision"
    decision_date: Optional[datetime] = None
    review_scores: List[float] = field(default_factory=list)
    average_score: Optional[float] = None
    reviewer_count: int = 0
    review_quality_score: Optional[float] = None
    
    def __post_init__(self):
        """Calculate average score if review scores are provided."""
        if self.review_scores and not self.average_score:
            self.average_score = statistics.mean(self.review_scores)


@dataclass
class AcceptanceRecord:
    """Record of acceptance statistics for a time period."""
    venue_id: str
    period_start: datetime
    period_end: datetime
    total_submissions: int
    total_acceptances: int
    acceptance_rate: float
    average_review_score: float
    score_distribution: Dict[str, List[float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate acceptance rate."""
        if self.total_submissions > 0:
            calculated_rate = self.total_acceptances / self.total_submissions
            if abs(calculated_rate - self.acceptance_rate) > 0.01:
                logger.warning(f"Acceptance rate mismatch: calculated {calculated_rate:.3f}, recorded {self.acceptance_rate:.3f}")


@dataclass
class TrendData:
    """Historical trend data for a venue."""
    venue_id: str
    metric_name: str  # "acceptance_rate", "avg_score", "submission_count"
    time_series: List[Tuple[datetime, float]] = field(default_factory=list)
    trend_direction: Optional[str] = None  # "increasing", "decreasing", "stable"
    trend_strength: Optional[float] = None  # correlation coefficient
    
    def add_data_point(self, timestamp: datetime, value: float):
        """Add a new data point to the time series."""
        self.time_series.append((timestamp, value))
        self.time_series.sort(key=lambda x: x[0])  # Keep sorted by time
        self._calculate_trend()
    
    def _calculate_trend(self):
        """Calculate trend direction and strength."""
        if len(self.time_series) < 3:
            return
        
        # Extract values and convert timestamps to numeric
        timestamps = [t.timestamp() for t, v in self.time_series]
        values = [v for t, v in self.time_series]
        
        # Check if all values are the same (constant)
        if len(set(values)) <= 1:
            self.trend_direction = "stable"
            self.trend_strength = 0.0
            return
        
        # Calculate correlation coefficient
        try:
            correlation, p_value = scipy_stats.pearsonr(timestamps, values)
            self.trend_strength = correlation
            
            # Determine trend direction
            if abs(correlation) < 0.1:
                self.trend_direction = "stable"
            elif correlation > 0:
                self.trend_direction = "increasing"
            else:
                self.trend_direction = "decreasing"
        except Exception:
            # Fallback for any correlation calculation issues
            self.trend_direction = "stable"
            self.trend_strength = 0.0


@dataclass
class RealismMetrics:
    """Metrics for validating venue realism against PeerRead data."""
    venue_id: str
    peerread_baseline: Dict[str, float]
    current_metrics: Dict[str, float]
    accuracy_scores: Dict[str, float] = field(default_factory=dict)
    overall_realism_score: float = 0.0
    
    def __post_init__(self):
        """Calculate accuracy scores and overall realism."""
        self._calculate_accuracy_scores()
        self._calculate_overall_realism()
    
    def _calculate_accuracy_scores(self):
        """Calculate accuracy scores for each metric."""
        for metric, baseline_value in self.peerread_baseline.items():
            if metric in self.current_metrics:
                current_value = self.current_metrics[metric]
                
                if baseline_value == 0:
                    # Handle division by zero
                    accuracy = 1.0 if current_value == 0 else 0.0
                else:
                    # Calculate percentage error and convert to accuracy (0-1)
                    error = abs(current_value - baseline_value) / baseline_value
                    accuracy = max(0.0, 1.0 - error)
                
                self.accuracy_scores[metric] = accuracy
    
    def _calculate_overall_realism(self):
        """Calculate overall realism score as weighted average."""
        if not self.accuracy_scores:
            self.overall_realism_score = 0.0
            return
        
        # Weight different metrics by importance
        weights = {
            "acceptance_rate": 0.3,
            "avg_review_score": 0.2,
            "score_std": 0.15,
            "review_count": 0.15,
            "review_length": 0.1,
            "default": 0.1
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, accuracy in self.accuracy_scores.items():
            weight = weights.get(metric, weights["default"])
            weighted_sum += accuracy * weight
            total_weight += weight
        
        self.overall_realism_score = weighted_sum / total_weight if total_weight > 0 else 0.0


class VenueStats:
    """
    Comprehensive venue statistics tracking with continuous PeerRead calibration.
    
    This class tracks submission and acceptance data, maintains historical trends,
    calculates dynamic acceptance rates, and validates venue realism against
    PeerRead baseline data.
    """
    
    def __init__(self, data_dir: Optional[Path] = None, peerread_path: Optional[str] = None):
        """
        Initialize venue statistics system.
        
        Args:
            data_dir: Directory for storing statistics data
            peerread_path: Path to PeerRead dataset for calibration
        """
        self.data_dir = data_dir or Path("data/venue_stats")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Core data structures
        self.submission_records: Dict[str, List[SubmissionRecord]] = defaultdict(list)
        self.acceptance_records: Dict[str, List[AcceptanceRecord]] = defaultdict(list)
        self.trend_data: Dict[str, Dict[str, TrendData]] = defaultdict(dict)
        self.realism_metrics: Dict[str, RealismMetrics] = {}
        
        # PeerRead integration
        self.peerread_loader: Optional[PeerReadLoader] = None
        self.peerread_baselines: Dict[str, VenueCharacteristics] = {}
        
        if peerread_path:
            self._initialize_peerread_integration(peerread_path)
        
        # Configuration
        self.realism_threshold = 0.8  # Minimum realism score
        self.acceptance_rate_tolerance = 0.05  # 5% tolerance for acceptance rates
        self.recalibration_interval_days = 30  # Recalibrate every 30 days
        self.max_history_days = 365 * 3  # Keep 3 years of history
        
        # Load existing data
        self._load_existing_data()
    
    def _initialize_peerread_integration(self, peerread_path: str):
        """Initialize PeerRead integration for baseline calibration."""
        try:
            self.peerread_loader = PeerReadLoader(peerread_path)
            self.peerread_baselines = self.peerread_loader.load_all_venues()
            logger.info(f"Loaded PeerRead baselines for {len(self.peerread_baselines)} venues")
        except Exception as e:
            logger.error(f"Failed to initialize PeerRead integration: {e}")
            self.peerread_loader = None
    
    def record_submission(self, paper_id: str, venue_id: str, submission_date: Optional[datetime] = None) -> SubmissionRecord:
        """
        Record a paper submission to a venue.
        
        Args:
            paper_id: Unique paper identifier
            venue_id: Venue identifier
            submission_date: Submission timestamp (defaults to now)
            
        Returns:
            SubmissionRecord: Created submission record
        """
        if submission_date is None:
            submission_date = datetime.now()
        
        record = SubmissionRecord(
            paper_id=paper_id,
            venue_id=venue_id,
            submission_date=submission_date
        )
        
        self.submission_records[venue_id].append(record)
        
        # Update trend data
        self._update_submission_trends(venue_id, submission_date)
        
        logger.debug(f"Recorded submission {paper_id} to venue {venue_id}")
        return record
    
    def record_decision(self, paper_id: str, venue_id: str, decision: str, 
                       review_scores: Optional[List[float]] = None,
                       decision_date: Optional[datetime] = None) -> bool:
        """
        Record a review decision for a submitted paper.
        
        Args:
            paper_id: Paper identifier
            venue_id: Venue identifier
            decision: Decision ("accept", "reject", "revision")
            review_scores: List of review scores
            decision_date: Decision timestamp (defaults to now)
            
        Returns:
            bool: True if decision was recorded successfully
        """
        if decision_date is None:
            decision_date = datetime.now()
        
        # Find the submission record
        submission_record = None
        for record in self.submission_records[venue_id]:
            if record.paper_id == paper_id:
                submission_record = record
                break
        
        if not submission_record:
            logger.error(f"No submission record found for paper {paper_id} at venue {venue_id}")
            return False
        
        # Update submission record
        submission_record.decision = decision
        submission_record.decision_date = decision_date
        if review_scores:
            submission_record.review_scores = review_scores
            submission_record.average_score = statistics.mean(review_scores)
            submission_record.reviewer_count = len(review_scores)
        
        # Update acceptance trends
        self._update_acceptance_trends(venue_id, decision, decision_date)
        
        logger.debug(f"Recorded decision '{decision}' for paper {paper_id} at venue {venue_id}")
        return True
    
    def calculate_dynamic_acceptance_rate(self, venue_id: str, 
                                        time_window_days: int = 365) -> float:
        """
        Calculate dynamic acceptance rate for a venue over a time window.
        
        Args:
            venue_id: Venue identifier
            time_window_days: Time window in days for calculation
            
        Returns:
            float: Calculated acceptance rate (0.0 to 1.0)
        """
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        
        # Get recent submissions with decisions
        recent_submissions = [
            record for record in self.submission_records[venue_id]
            if record.submission_date >= cutoff_date and record.decision is not None
        ]
        
        if not recent_submissions:
            # Fall back to PeerRead baseline if available
            if venue_id in self.peerread_baselines:
                return self.peerread_baselines[venue_id].acceptance_rate
            return 0.0
        
        # Calculate acceptance rate
        acceptances = sum(1 for record in recent_submissions if record.decision == "accept")
        acceptance_rate = acceptances / len(recent_submissions)
        
        # Validate against PeerRead baseline
        if venue_id in self.peerread_baselines:
            baseline_rate = self.peerread_baselines[venue_id].acceptance_rate
            if abs(acceptance_rate - baseline_rate) > self.acceptance_rate_tolerance:
                logger.warning(f"Venue {venue_id} acceptance rate {acceptance_rate:.3f} "
                             f"deviates from baseline {baseline_rate:.3f}")
        
        return acceptance_rate
    
    def get_historical_trends(self, venue_id: str, metric: str = "acceptance_rate") -> Optional[TrendData]:
        """
        Get historical trend data for a venue metric.
        
        Args:
            venue_id: Venue identifier
            metric: Metric name ("acceptance_rate", "avg_score", "submission_count")
            
        Returns:
            TrendData or None if no data available
        """
        return self.trend_data[venue_id].get(metric)
    
    def validate_venue_realism(self, venue_id: str) -> RealismMetrics:
        """
        Validate venue realism against PeerRead baseline data.
        
        Args:
            venue_id: Venue identifier
            
        Returns:
            RealismMetrics: Realism validation results
        """
        # Get current venue metrics
        current_metrics = self._calculate_current_metrics(venue_id)
        
        # Get PeerRead baseline if available
        peerread_baseline = {}
        if venue_id in self.peerread_baselines:
            baseline_data = self.peerread_baselines[venue_id]
            peerread_baseline = {
                "acceptance_rate": baseline_data.acceptance_rate,
                "avg_review_score": baseline_data.substance_mean,  # Use substance as proxy
                "review_count": baseline_data.avg_reviews_per_paper,
                "review_length": baseline_data.avg_review_length
            }
        
        # Create realism metrics
        realism_metrics = RealismMetrics(
            venue_id=venue_id,
            peerread_baseline=peerread_baseline,
            current_metrics=current_metrics
        )
        
        # Store for future reference
        self.realism_metrics[venue_id] = realism_metrics
        
        # Log warning if realism is low
        if realism_metrics.overall_realism_score < self.realism_threshold:
            logger.warning(f"Venue {venue_id} realism score {realism_metrics.overall_realism_score:.3f} "
                         f"below threshold {self.realism_threshold}")
        
        return realism_metrics
    
    def recalibrate_venue(self, venue_id: str, venue: EnhancedVenue) -> bool:
        """
        Recalibrate venue parameters based on recent data and PeerRead baselines.
        
        Args:
            venue_id: Venue identifier
            venue: EnhancedVenue object to recalibrate
            
        Returns:
            bool: True if recalibration was performed
        """
        if venue_id not in self.peerread_baselines:
            logger.warning(f"No PeerRead baseline available for venue {venue_id}")
            return False
        
        baseline = self.peerread_baselines[venue_id]
        current_metrics = self._calculate_current_metrics(venue_id)
        
        # Recalibrate acceptance rate
        current_rate = venue.acceptance_rate  # Use venue's current rate, not calculated metrics
        baseline_rate = baseline.acceptance_rate
        
        # If current rate deviates significantly, adjust towards baseline
        if abs(current_rate - baseline_rate) > self.acceptance_rate_tolerance:
            # Weighted average: 70% baseline, 30% current
            new_rate = 0.7 * baseline_rate + 0.3 * current_rate
            venue.acceptance_rate = new_rate
            logger.info(f"Recalibrated venue {venue_id} acceptance rate: "
                      f"{current_rate:.3f} -> {new_rate:.3f} (baseline: {baseline_rate:.3f})")
        
        # Also check calculated metrics if available
        if "acceptance_rate" in current_metrics:
            calculated_rate = current_metrics["acceptance_rate"]
            if abs(calculated_rate - baseline_rate) > self.acceptance_rate_tolerance:
                # Use calculated rate in the adjustment if it's more recent
                new_rate = 0.7 * baseline_rate + 0.3 * calculated_rate
                venue.acceptance_rate = new_rate
                logger.info(f"Recalibrated venue {venue_id} acceptance rate based on recent data: "
                          f"{calculated_rate:.3f} -> {new_rate:.3f} (baseline: {baseline_rate:.3f})")
        
        # Recalibrate score distributions
        if hasattr(venue, 'score_distributions') and baseline.impact_scores:
            venue.score_distributions = {
                "IMPACT": baseline.impact_scores[-10:],  # Use recent scores
                "SUBSTANCE": baseline.substance_scores[-10:],
                "SOUNDNESS_CORRECTNESS": baseline.soundness_scores[-10:],
                "ORIGINALITY": baseline.originality_scores[-10:],
                "CLARITY": baseline.clarity_scores[-10:],
                "MEANINGFUL_COMPARISON": baseline.meaningful_comparison_scores[-10:]
            }
        
        return True
    
    def get_venue_statistics(self, venue_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a venue.
        
        Args:
            venue_id: Venue identifier
            
        Returns:
            Dictionary with venue statistics
        """
        stats = {
            "venue_id": venue_id,
            "total_submissions": len(self.submission_records[venue_id]),
            "submissions_with_decisions": 0,
            "acceptance_rate": 0.0,
            "average_review_score": 0.0,
            "review_score_std": 0.0,
            "average_review_count": 0.0,
            "trends": {},
            "realism_score": 0.0,
            "last_updated": datetime.now().isoformat()
        }
        
        # Calculate basic statistics
        submissions = self.submission_records[venue_id]
        decided_submissions = [s for s in submissions if s.decision is not None]
        
        if decided_submissions:
            stats["submissions_with_decisions"] = len(decided_submissions)
            
            # Acceptance rate
            acceptances = sum(1 for s in decided_submissions if s.decision == "accept")
            stats["acceptance_rate"] = acceptances / len(decided_submissions)
            
            # Review scores
            all_scores = []
            review_counts = []
            for submission in decided_submissions:
                if submission.review_scores:
                    all_scores.extend(submission.review_scores)
                    review_counts.append(len(submission.review_scores))
            
            if all_scores:
                stats["average_review_score"] = statistics.mean(all_scores)
                stats["review_score_std"] = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
            
            if review_counts:
                stats["average_review_count"] = statistics.mean(review_counts)
        
        # Add trend information
        for metric_name, trend_data in self.trend_data[venue_id].items():
            stats["trends"][metric_name] = {
                "direction": trend_data.trend_direction,
                "strength": trend_data.trend_strength,
                "data_points": len(trend_data.time_series)
            }
        
        # Add realism score
        if venue_id in self.realism_metrics:
            stats["realism_score"] = self.realism_metrics[venue_id].overall_realism_score
        
        return stats
    
    def _calculate_current_metrics(self, venue_id: str) -> Dict[str, float]:
        """Calculate current metrics for a venue."""
        metrics = {}
        
        # Get recent submissions (last 365 days)
        cutoff_date = datetime.now() - timedelta(days=365)
        recent_submissions = [
            record for record in self.submission_records[venue_id]
            if record.submission_date >= cutoff_date
        ]
        
        if not recent_submissions:
            return metrics
        
        # Acceptance rate
        decided_submissions = [s for s in recent_submissions if s.decision is not None]
        if decided_submissions:
            acceptances = sum(1 for s in decided_submissions if s.decision == "accept")
            metrics["acceptance_rate"] = acceptances / len(decided_submissions)
        
        # Review scores
        all_scores = []
        review_counts = []
        for submission in decided_submissions:
            if submission.review_scores:
                all_scores.extend(submission.review_scores)
                review_counts.append(len(submission.review_scores))
        
        if all_scores:
            metrics["avg_review_score"] = statistics.mean(all_scores)
            metrics["score_std"] = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
        
        if review_counts:
            metrics["review_count"] = statistics.mean(review_counts)
        
        return metrics
    
    def _update_submission_trends(self, venue_id: str, submission_date: datetime):
        """Update submission count trends."""
        if "submission_count" not in self.trend_data[venue_id]:
            self.trend_data[venue_id]["submission_count"] = TrendData(
                venue_id=venue_id,
                metric_name="submission_count"
            )
        
        # Count submissions in the current month
        month_start = submission_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_submissions = sum(
            1 for record in self.submission_records[venue_id]
            if record.submission_date >= month_start and record.submission_date < month_start + timedelta(days=32)
        )
        
        trend_data = self.trend_data[venue_id]["submission_count"]
        trend_data.add_data_point(month_start, float(month_submissions))
    
    def _update_acceptance_trends(self, venue_id: str, decision: str, decision_date: datetime):
        """Update acceptance rate trends."""
        if "acceptance_rate" not in self.trend_data[venue_id]:
            self.trend_data[venue_id]["acceptance_rate"] = TrendData(
                venue_id=venue_id,
                metric_name="acceptance_rate"
            )
        
        # Calculate acceptance rate for the current month
        month_start = decision_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_end = month_start + timedelta(days=32)
        
        month_decisions = [
            record for record in self.submission_records[venue_id]
            if (record.decision_date and 
                record.decision_date >= month_start and 
                record.decision_date < month_end)
        ]
        
        if month_decisions:
            acceptances = sum(1 for record in month_decisions if record.decision == "accept")
            acceptance_rate = acceptances / len(month_decisions)
            
            trend_data = self.trend_data[venue_id]["acceptance_rate"]
            trend_data.add_data_point(month_start, acceptance_rate)
    
    def _load_existing_data(self):
        """Load existing statistics data from disk."""
        stats_file = self.data_dir / "venue_statistics.json"
        if not stats_file.exists():
            return
        
        try:
            with open(stats_file, 'r') as f:
                data = json.load(f)
            
            # Load submission records
            for venue_id, records_data in data.get("submission_records", {}).items():
                for record_data in records_data:
                    record = SubmissionRecord(
                        paper_id=record_data["paper_id"],
                        venue_id=record_data["venue_id"],
                        submission_date=datetime.fromisoformat(record_data["submission_date"]),
                        decision=record_data.get("decision"),
                        decision_date=datetime.fromisoformat(record_data["decision_date"]) if record_data.get("decision_date") else None,
                        review_scores=record_data.get("review_scores", []),
                        reviewer_count=record_data.get("reviewer_count", 0)
                    )
                    self.submission_records[venue_id].append(record)
            
            logger.info(f"Loaded statistics for {len(self.submission_records)} venues")
            
        except Exception as e:
            logger.error(f"Failed to load existing statistics data: {e}")
    
    def save_data(self):
        """Save statistics data to disk."""
        stats_file = self.data_dir / "venue_statistics.json"
        
        try:
            # Prepare data for serialization
            data = {
                "submission_records": {},
                "last_updated": datetime.now().isoformat()
            }
            
            for venue_id, records in self.submission_records.items():
                data["submission_records"][venue_id] = []
                for record in records:
                    record_data = {
                        "paper_id": record.paper_id,
                        "venue_id": record.venue_id,
                        "submission_date": record.submission_date.isoformat(),
                        "decision": record.decision,
                        "decision_date": record.decision_date.isoformat() if record.decision_date else None,
                        "review_scores": record.review_scores,
                        "reviewer_count": record.reviewer_count,
                        "average_score": record.average_score
                    }
                    data["submission_records"][venue_id].append(record_data)
            
            with open(stats_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved statistics data to {stats_file}")
            
        except Exception as e:
            logger.error(f"Failed to save statistics data: {e}")
    
    def cleanup_old_data(self, max_age_days: int = None):
        """
        Clean up old statistics data beyond the retention period.
        
        Args:
            max_age_days: Maximum age in days (defaults to self.max_history_days)
        """
        if max_age_days is None:
            max_age_days = self.max_history_days
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Clean up submission records
        for venue_id in list(self.submission_records.keys()):
            original_count = len(self.submission_records[venue_id])
            self.submission_records[venue_id] = [
                record for record in self.submission_records[venue_id]
                if record.submission_date >= cutoff_date
            ]
            cleaned_count = original_count - len(self.submission_records[venue_id])
            
            if cleaned_count > 0:
                logger.info(f"Cleaned {cleaned_count} old records for venue {venue_id}")
        
        # Clean up trend data
        for venue_id in self.trend_data:
            for metric_name in self.trend_data[venue_id]:
                trend_data = self.trend_data[venue_id][metric_name]
                original_count = len(trend_data.time_series)
                trend_data.time_series = [
                    (timestamp, value) for timestamp, value in trend_data.time_series
                    if timestamp >= cutoff_date
                ]
                cleaned_count = original_count - len(trend_data.time_series)
                
                if cleaned_count > 0:
                    logger.debug(f"Cleaned {cleaned_count} old trend points for {venue_id}.{metric_name}")