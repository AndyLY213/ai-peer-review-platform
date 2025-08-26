"""
Review Trading Detector

This module implements review trading detection functionality to identify quid pro quo
arrangements where researchers exchange favorable reviews. It tracks mutual review
patterns and detects suspicious trading behavior.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from collections import defaultdict, Counter
import itertools

from src.data.enhanced_models import ReviewDecision
from src.core.exceptions import ValidationError
from src.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ReviewExchange:
    """Record of a review exchange between two researchers."""
    exchange_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reviewer_a: str = ""  # First reviewer ID
    reviewer_b: str = ""  # Second reviewer ID
    paper_a: str = ""     # Paper reviewed by B for A
    paper_b: str = ""     # Paper reviewed by A for B
    review_a_date: datetime = field(default_factory=datetime.now)
    review_b_date: datetime = field(default_factory=datetime.now)
    score_a: float = 0.0  # Score given by B to A's paper
    score_b: float = 0.0  # Score given by A to B's paper
    decision_a: ReviewDecision = ReviewDecision.MAJOR_REVISION
    decision_b: ReviewDecision = ReviewDecision.MAJOR_REVISION
    time_gap_days: int = 0  # Days between the two reviews
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.time_gap_days = abs((self.review_b_date - self.review_a_date).days)


@dataclass
class TradingPattern:
    """Pattern of review trading between researchers."""
    researcher_a: str
    researcher_b: str
    exchanges: List[ReviewExchange] = field(default_factory=list)
    total_exchanges: int = 0
    mutual_favorable_count: int = 0  # Both gave favorable reviews
    average_score_difference: float = 0.0  # Difference from baseline scores
    average_time_gap: float = 0.0  # Average days between reciprocal reviews
    trading_score: float = 0.0  # 0-1 scale indicating trading likelihood
    confidence_level: float = 0.0  # Statistical confidence in trading detection
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate trading pattern metrics."""
        if not self.exchanges:
            return
        
        self.total_exchanges = len(self.exchanges)
        
        # Count mutual favorable reviews (both accept or minor revision)
        favorable_decisions = {ReviewDecision.ACCEPT, ReviewDecision.MINOR_REVISION}
        self.mutual_favorable_count = sum(
            1 for ex in self.exchanges 
            if ex.decision_a in favorable_decisions and ex.decision_b in favorable_decisions
        )
        
        # Calculate average time gap
        if self.exchanges:
            self.average_time_gap = sum(ex.time_gap_days for ex in self.exchanges) / len(self.exchanges)
        
        # Calculate trading score based on multiple factors
        self._calculate_trading_score()
    
    def _calculate_trading_score(self):
        """Calculate trading likelihood score (0-1 scale)."""
        if not self.exchanges:
            self.trading_score = 0.0
            return
        
        # Factor 1: Frequency of exchanges (normalized to 10 exchanges)
        frequency_factor = min(1.0, self.total_exchanges / 10.0)
        
        # Factor 2: Mutual favorability rate
        favorability_factor = self.mutual_favorable_count / self.total_exchanges
        
        # Factor 3: Time proximity (closer in time = more suspicious)
        # Normalize to 30 days - closer reviews are more suspicious
        time_factor = max(0.0, 1.0 - (self.average_time_gap / 30.0))
        time_factor = min(1.0, time_factor)
        
        # Factor 4: Score correlation (both giving similar high scores)
        score_correlation_factor = self._calculate_score_correlation()
        
        # Combine factors with weights
        self.trading_score = (
            frequency_factor * 0.3 +
            favorability_factor * 0.3 +
            time_factor * 0.2 +
            score_correlation_factor * 0.2
        )
        
        # Calculate confidence based on sample size and consistency
        self._calculate_confidence()
    
    def _calculate_score_correlation(self) -> float:
        """Calculate correlation in scores given by the pair."""
        if len(self.exchanges) < 2:
            return 0.0
        
        scores_a = [ex.score_a for ex in self.exchanges]
        scores_b = [ex.score_b for ex in self.exchanges]
        
        # Simple correlation measure: how often both scores are above average
        avg_score_a = sum(scores_a) / len(scores_a)
        avg_score_b = sum(scores_b) / len(scores_b)
        
        both_high_count = sum(
            1 for sa, sb in zip(scores_a, scores_b)
            if sa > avg_score_a and sb > avg_score_b
        )
        
        return both_high_count / len(self.exchanges)
    
    def _calculate_confidence(self):
        """Calculate statistical confidence in trading detection."""
        if not self.exchanges:
            self.confidence_level = 0.0
            return
        
        # Base confidence on sample size
        sample_factor = min(1.0, self.total_exchanges / 5.0)  # Normalize to 5 exchanges
        
        # Consistency factor: how consistent the pattern is
        if self.total_exchanges > 1:
            # Measure consistency in favorability
            favorability_rates = []
            for i in range(len(self.exchanges)):
                # Calculate favorability for each exchange
                favorable_a = self.exchanges[i].decision_a in {ReviewDecision.ACCEPT, ReviewDecision.MINOR_REVISION}
                favorable_b = self.exchanges[i].decision_b in {ReviewDecision.ACCEPT, ReviewDecision.MINOR_REVISION}
                favorability_rates.append(1.0 if (favorable_a and favorable_b) else 0.0)
            
            # Calculate variance in favorability
            mean_fav = sum(favorability_rates) / len(favorability_rates)
            variance = sum((x - mean_fav) ** 2 for x in favorability_rates) / len(favorability_rates)
            consistency_factor = 1.0 - variance  # Lower variance = higher consistency
        else:
            consistency_factor = 0.5
        
        self.confidence_level = (sample_factor * 0.6 + consistency_factor * 0.4)


@dataclass
class ResearcherTradingProfile:
    """Profile of a researcher's review trading behavior."""
    researcher_id: str
    total_trading_partners: int = 0
    total_exchanges: int = 0
    average_trading_score: float = 0.0
    most_frequent_partners: List[Tuple[str, int]] = field(default_factory=list)  # (partner_id, exchange_count)
    trading_tendency: float = 0.0  # 0-1 scale
    suspicious_patterns: List[str] = field(default_factory=list)  # Pattern IDs
    
    def update_profile(self, trading_patterns: List[TradingPattern]):
        """Update profile based on trading patterns."""
        if not trading_patterns:
            return
        
        self.total_trading_partners = len(trading_patterns)
        self.total_exchanges = sum(p.total_exchanges for p in trading_patterns)
        
        if trading_patterns:
            self.average_trading_score = sum(p.trading_score for p in trading_patterns) / len(trading_patterns)
        
        # Identify most frequent partners
        partner_counts = [(p.researcher_b if p.researcher_a == self.researcher_id else p.researcher_a, 
                          p.total_exchanges) for p in trading_patterns]
        self.most_frequent_partners = sorted(partner_counts, key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate trading tendency
        high_score_patterns = sum(1 for p in trading_patterns if p.trading_score > 0.5)
        self.trading_tendency = high_score_patterns / len(trading_patterns) if trading_patterns else 0.0
        
        # Identify suspicious patterns
        self.suspicious_patterns = [
            f"{p.researcher_a}-{p.researcher_b}" for p in trading_patterns 
            if p.trading_score > 0.7 and p.confidence_level > 0.6
        ]


class ReviewTradingDetector:
    """
    Detects review trading patterns and quid pro quo arrangements between researchers.
    
    This class analyzes review patterns to identify suspicious mutual reviewing
    arrangements where researchers may be exchanging favorable reviews.
    """
    
    def __init__(self, data_dir: str = "data/review_trading"):
        """
        Initialize the review trading detector.
        
        Args:
            data_dir: Directory to store review trading data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for tracking data
        self.review_records: Dict[str, Dict] = {}  # review_id -> review info
        self.researcher_reviews: Dict[str, List[str]] = defaultdict(list)  # researcher_id -> review_ids
        self.paper_reviews: Dict[str, List[str]] = defaultdict(list)  # paper_id -> review_ids
        self.trading_patterns: Dict[str, TradingPattern] = {}  # pattern_id -> pattern
        self.researcher_profiles: Dict[str, ResearcherTradingProfile] = {}  # researcher_id -> profile
        
        # Configuration
        self.min_trading_score = 0.5  # Minimum score to consider as trading
        self.max_time_gap_days = 90  # Maximum days between reviews to consider as potential trading
        self.min_exchanges_for_pattern = 2  # Minimum exchanges to establish a pattern
        
        logger.info(f"ReviewTradingDetector initialized with data directory: {self.data_dir}")
    
    def record_review(self, review_id: str, reviewer_id: str, paper_id: str, author_id: str,
                     review_date: datetime, score: float, decision: ReviewDecision) -> None:
        """
        Record a review for trading analysis.
        
        Args:
            review_id: Unique identifier for the review
            reviewer_id: ID of the reviewer
            paper_id: ID of the paper being reviewed
            author_id: ID of the paper's author
            review_date: Date when review was submitted
            score: Numerical score given in review
            decision: Review decision
        """
        review_record = {
            'review_id': review_id,
            'reviewer_id': reviewer_id,
            'paper_id': paper_id,
            'author_id': author_id,
            'review_date': review_date,
            'score': score,
            'decision': decision
        }
        
        self.review_records[review_id] = review_record
        self.researcher_reviews[reviewer_id].append(review_id)
        self.paper_reviews[paper_id].append(review_id)
        
        logger.debug(f"Recorded review {review_id} by {reviewer_id} for paper {paper_id}")
        
        # Update trading patterns
        self._update_trading_patterns(reviewer_id, author_id)
    
    def _update_trading_patterns(self, reviewer_id: str, author_id: str):
        """Update trading patterns when a new review is recorded."""
        # Look for reciprocal reviews between reviewer and author
        reviewer_reviews = self.researcher_reviews[reviewer_id]
        author_reviews = self.researcher_reviews[author_id]
        
        # Find reviews where author reviewed reviewer's papers
        reciprocal_pairs = []
        
        for rev_review_id in reviewer_reviews:
            rev_review = self.review_records[rev_review_id]
            if rev_review['author_id'] == author_id:  # Reviewer reviewed author's paper
                
                # Look for author's reviews of reviewer's papers
                for auth_review_id in author_reviews:
                    auth_review = self.review_records[auth_review_id]
                    if auth_review['author_id'] == reviewer_id:  # Author reviewed reviewer's paper
                        
                        # Check time proximity
                        time_gap = abs((auth_review['review_date'] - rev_review['review_date']).days)
                        if time_gap <= self.max_time_gap_days:
                            reciprocal_pairs.append((rev_review_id, auth_review_id))
        
        # Create or update trading patterns
        for rev_review_id, auth_review_id in reciprocal_pairs:
            self._create_or_update_pattern(rev_review_id, auth_review_id)
    
    def _create_or_update_pattern(self, review_a_id: str, review_b_id: str):
        """Create or update a trading pattern from two reciprocal reviews."""
        review_a = self.review_records[review_a_id]
        review_b = self.review_records[review_b_id]
        
        researcher_a = review_a['reviewer_id']
        researcher_b = review_b['reviewer_id']
        
        # Create pattern ID (consistent ordering)
        if researcher_a < researcher_b:
            pattern_id = f"{researcher_a}-{researcher_b}"
        else:
            pattern_id = f"{researcher_b}-{researcher_a}"
            researcher_a, researcher_b = researcher_b, researcher_a
            review_a, review_b = review_b, review_a
        
        # Create exchange record
        exchange = ReviewExchange(
            reviewer_a=researcher_a,
            reviewer_b=researcher_b,
            paper_a=review_a['paper_id'],
            paper_b=review_b['paper_id'],
            review_a_date=review_a['review_date'],
            review_b_date=review_b['review_date'],
            score_a=review_a['score'],
            score_b=review_b['score'],
            decision_a=review_a['decision'],
            decision_b=review_b['decision']
        )
        
        # Create or update pattern
        if pattern_id not in self.trading_patterns:
            self.trading_patterns[pattern_id] = TradingPattern(
                researcher_a=researcher_a,
                researcher_b=researcher_b
            )
        
        pattern = self.trading_patterns[pattern_id]
        
        # Check if this exchange already exists
        existing_exchange = None
        for ex in pattern.exchanges:
            if ((ex.paper_a == exchange.paper_a and ex.paper_b == exchange.paper_b) or
                (ex.paper_a == exchange.paper_b and ex.paper_b == exchange.paper_a)):
                existing_exchange = ex
                break
        
        if not existing_exchange:
            pattern.exchanges.append(exchange)
            pattern.__post_init__()  # Recalculate metrics
            
            # Update researcher profiles
            self._update_researcher_profiles([researcher_a, researcher_b])
    
    def _update_researcher_profiles(self, researcher_ids: List[str]):
        """Update researcher trading profiles."""
        for researcher_id in researcher_ids:
            # Get all patterns involving this researcher
            researcher_patterns = [
                p for p in self.trading_patterns.values()
                if p.researcher_a == researcher_id or p.researcher_b == researcher_id
            ]
            
            if not researcher_patterns:
                continue
            
            # Create or update profile
            if researcher_id not in self.researcher_profiles:
                self.researcher_profiles[researcher_id] = ResearcherTradingProfile(
                    researcher_id=researcher_id
                )
            
            profile = self.researcher_profiles[researcher_id]
            profile.update_profile(researcher_patterns)
    
    def detect_trading_patterns(self, min_score: Optional[float] = None) -> List[TradingPattern]:
        """
        Detect suspicious review trading patterns.
        
        Args:
            min_score: Minimum trading score threshold (uses default if None)
            
        Returns:
            List of suspicious trading patterns
        """
        threshold = min_score if min_score is not None else self.min_trading_score
        
        suspicious_patterns = [
            pattern for pattern in self.trading_patterns.values()
            if (pattern.trading_score >= threshold and 
                pattern.total_exchanges >= self.min_exchanges_for_pattern)
        ]
        
        return sorted(suspicious_patterns, key=lambda x: x.trading_score, reverse=True)
    
    def get_researcher_trading_behavior(self, researcher_id: str) -> Optional[ResearcherTradingProfile]:
        """
        Get trading behavior profile for a researcher.
        
        Args:
            researcher_id: Researcher identifier
            
        Returns:
            ResearcherTradingProfile if available, None otherwise
        """
        return self.researcher_profiles.get(researcher_id)
    
    def analyze_network_clusters(self) -> Dict[str, List[str]]:
        """
        Analyze trading networks to identify clusters of researchers.
        
        Returns:
            Dictionary mapping cluster IDs to lists of researcher IDs
        """
        # Build adjacency list of trading relationships
        adjacency = defaultdict(set)
        
        for pattern in self.trading_patterns.values():
            if pattern.trading_score >= self.min_trading_score:
                adjacency[pattern.researcher_a].add(pattern.researcher_b)
                adjacency[pattern.researcher_b].add(pattern.researcher_a)
        
        # Find connected components (clusters)
        visited = set()
        clusters = {}
        cluster_id = 0
        
        def dfs(node: str, cluster: List[str]):
            if node in visited:
                return
            visited.add(node)
            cluster.append(node)
            for neighbor in adjacency[node]:
                dfs(neighbor, cluster)
        
        for researcher in adjacency:
            if researcher not in visited:
                cluster = []
                dfs(researcher, cluster)
                if len(cluster) > 1:  # Only include clusters with multiple researchers
                    clusters[f"cluster_{cluster_id}"] = cluster
                    cluster_id += 1
        
        return clusters
    
    def calculate_trading_statistics(self) -> Dict[str, float]:
        """
        Calculate overall trading statistics.
        
        Returns:
            Dictionary with trading statistics
        """
        if not self.trading_patterns:
            return {
                'total_patterns': 0,
                'suspicious_patterns': 0,
                'trading_rate': 0.0,
                'average_trading_score': 0.0,
                'average_exchanges_per_pattern': 0.0,
                'total_researchers_involved': 0
            }
        
        total_patterns = len(self.trading_patterns)
        suspicious_patterns = sum(1 for p in self.trading_patterns.values() 
                                if p.trading_score >= self.min_trading_score)
        
        total_exchanges = sum(p.total_exchanges for p in self.trading_patterns.values())
        total_trading_score = sum(p.trading_score for p in self.trading_patterns.values())
        
        # Count unique researchers involved
        researchers_involved = set()
        for pattern in self.trading_patterns.values():
            researchers_involved.add(pattern.researcher_a)
            researchers_involved.add(pattern.researcher_b)
        
        return {
            'total_patterns': total_patterns,
            'suspicious_patterns': suspicious_patterns,
            'trading_rate': suspicious_patterns / total_patterns if total_patterns > 0 else 0.0,
            'average_trading_score': total_trading_score / total_patterns if total_patterns > 0 else 0.0,
            'average_exchanges_per_pattern': total_exchanges / total_patterns if total_patterns > 0 else 0.0,
            'total_researchers_involved': len(researchers_involved)
        }
    
    def predict_trading_likelihood(self, researcher_a: str, researcher_b: str) -> float:
        """
        Predict likelihood of trading between two researchers.
        
        Args:
            researcher_a: First researcher ID
            researcher_b: Second researcher ID
            
        Returns:
            Predicted trading likelihood (0-1 scale)
        """
        # Check if pattern already exists
        pattern_id = f"{min(researcher_a, researcher_b)}-{max(researcher_a, researcher_b)}"
        if pattern_id in self.trading_patterns:
            return self.trading_patterns[pattern_id].trading_score
        
        # Predict based on individual trading tendencies
        profile_a = self.researcher_profiles.get(researcher_a)
        profile_b = self.researcher_profiles.get(researcher_b)
        
        if not profile_a or not profile_b:
            return 0.1  # Low baseline probability
        
        # Combine individual trading tendencies
        combined_tendency = (profile_a.trading_tendency + profile_b.trading_tendency) / 2
        
        # Adjust based on how many partners they typically have
        partner_factor_a = min(1.0, profile_a.total_trading_partners / 5.0)
        partner_factor_b = min(1.0, profile_b.total_trading_partners / 5.0)
        partner_factor = (partner_factor_a + partner_factor_b) / 2
        
        # Combine factors
        likelihood = combined_tendency * 0.7 + partner_factor * 0.3
        
        return min(1.0, likelihood)
    
    def generate_trading_report(self, researcher_id: Optional[str] = None) -> Dict:
        """
        Generate comprehensive trading report.
        
        Args:
            researcher_id: If provided, focus on specific researcher
            
        Returns:
            Dictionary containing comprehensive trading analysis
        """
        if researcher_id:
            # Researcher-specific report
            profile = self.researcher_profiles.get(researcher_id)
            if not profile:
                return {'error': f'No trading data found for researcher {researcher_id}'}
            
            # Get patterns involving this researcher
            researcher_patterns = [
                p for p in self.trading_patterns.values()
                if p.researcher_a == researcher_id or p.researcher_b == researcher_id
            ]
            
            return {
                'researcher_id': researcher_id,
                'profile': {
                    'total_trading_partners': profile.total_trading_partners,
                    'total_exchanges': profile.total_exchanges,
                    'average_trading_score': profile.average_trading_score,
                    'trading_tendency': profile.trading_tendency,
                    'most_frequent_partners': profile.most_frequent_partners,
                    'suspicious_patterns': profile.suspicious_patterns
                },
                'patterns': [self._pattern_to_dict(p) for p in researcher_patterns],
                'risk_level': self._calculate_risk_level(profile)
            }
        else:
            # Global report
            stats = self.calculate_trading_statistics()
            clusters = self.analyze_network_clusters()
            suspicious_patterns = self.detect_trading_patterns()
            
            return {
                'statistics': stats,
                'network_clusters': clusters,
                'suspicious_patterns': [self._pattern_to_dict(p) for p in suspicious_patterns[:10]],  # Top 10
                'top_traders': self._get_top_traders(5)
            }
    
    def _calculate_risk_level(self, profile: ResearcherTradingProfile) -> str:
        """Calculate risk level for a researcher."""
        if profile.trading_tendency >= 0.7 and profile.average_trading_score >= 0.6:
            return "HIGH"
        elif profile.trading_tendency >= 0.4 and profile.average_trading_score >= 0.4:
            return "MEDIUM"
        elif profile.trading_tendency >= 0.2 or profile.average_trading_score >= 0.3:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _get_top_traders(self, limit: int) -> List[Dict]:
        """Get top traders by trading score."""
        sorted_profiles = sorted(
            self.researcher_profiles.values(),
            key=lambda x: x.average_trading_score,
            reverse=True
        )
        
        return [
            {
                'researcher_id': p.researcher_id,
                'trading_score': p.average_trading_score,
                'trading_tendency': p.trading_tendency,
                'total_partners': p.total_trading_partners,
                'total_exchanges': p.total_exchanges
            }
            for p in sorted_profiles[:limit]
        ]
    
    def save_data(self, filename: str = "review_trading_data.json"):
        """Save review trading data to file."""
        data = {
            'review_records': {
                review_id: {
                    **record,
                    'review_date': record['review_date'].isoformat(),
                    'decision': record['decision'].value
                }
                for review_id, record in self.review_records.items()
            },
            'trading_patterns': {
                pattern_id: self._pattern_to_dict(pattern)
                for pattern_id, pattern in self.trading_patterns.items()
            },
            'researcher_profiles': {
                researcher_id: self._profile_to_dict(profile)
                for researcher_id, profile in self.researcher_profiles.items()
            }
        }
        
        filepath = self.data_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Review trading data saved to {filepath}")
    
    def load_data(self, filename: str = "review_trading_data.json"):
        """Load review trading data from file."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            logger.warning(f"Review trading data file not found: {filepath}")
            return
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load review records
            self.review_records = {}
            for review_id, record_data in data.get('review_records', {}).items():
                record_data['review_date'] = datetime.fromisoformat(record_data['review_date'])
                record_data['decision'] = ReviewDecision(record_data['decision'])
                self.review_records[review_id] = record_data
            
            # Rebuild researcher_reviews and paper_reviews
            self.researcher_reviews = defaultdict(list)
            self.paper_reviews = defaultdict(list)
            for review_id, record in self.review_records.items():
                self.researcher_reviews[record['reviewer_id']].append(review_id)
                self.paper_reviews[record['paper_id']].append(review_id)
            
            # Load trading patterns
            self.trading_patterns = {}
            for pattern_id, pattern_data in data.get('trading_patterns', {}).items():
                self.trading_patterns[pattern_id] = self._dict_to_pattern(pattern_data)
            
            # Load researcher profiles
            self.researcher_profiles = {}
            for researcher_id, profile_data in data.get('researcher_profiles', {}).items():
                self.researcher_profiles[researcher_id] = self._dict_to_profile(profile_data)
            
            logger.info(f"Review trading data loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading review trading data: {e}")
            raise
    
    def _pattern_to_dict(self, pattern: TradingPattern) -> Dict:
        """Convert trading pattern to dictionary."""
        return {
            'researcher_a': pattern.researcher_a,
            'researcher_b': pattern.researcher_b,
            'exchanges': [self._exchange_to_dict(ex) for ex in pattern.exchanges],
            'total_exchanges': pattern.total_exchanges,
            'mutual_favorable_count': pattern.mutual_favorable_count,
            'average_score_difference': pattern.average_score_difference,
            'average_time_gap': pattern.average_time_gap,
            'trading_score': pattern.trading_score,
            'confidence_level': pattern.confidence_level
        }
    
    def _dict_to_pattern(self, data: Dict) -> TradingPattern:
        """Convert dictionary to trading pattern."""
        pattern = TradingPattern(
            researcher_a=data['researcher_a'],
            researcher_b=data['researcher_b'],
            exchanges=[self._dict_to_exchange(ex) for ex in data['exchanges']],
            total_exchanges=data['total_exchanges'],
            mutual_favorable_count=data['mutual_favorable_count'],
            average_score_difference=data['average_score_difference'],
            average_time_gap=data['average_time_gap'],
            trading_score=data['trading_score'],
            confidence_level=data['confidence_level']
        )
        return pattern
    
    def _exchange_to_dict(self, exchange: ReviewExchange) -> Dict:
        """Convert review exchange to dictionary."""
        return {
            'exchange_id': exchange.exchange_id,
            'reviewer_a': exchange.reviewer_a,
            'reviewer_b': exchange.reviewer_b,
            'paper_a': exchange.paper_a,
            'paper_b': exchange.paper_b,
            'review_a_date': exchange.review_a_date.isoformat(),
            'review_b_date': exchange.review_b_date.isoformat(),
            'score_a': exchange.score_a,
            'score_b': exchange.score_b,
            'decision_a': exchange.decision_a.value,
            'decision_b': exchange.decision_b.value,
            'time_gap_days': exchange.time_gap_days
        }
    
    def _dict_to_exchange(self, data: Dict) -> ReviewExchange:
        """Convert dictionary to review exchange."""
        return ReviewExchange(
            exchange_id=data['exchange_id'],
            reviewer_a=data['reviewer_a'],
            reviewer_b=data['reviewer_b'],
            paper_a=data['paper_a'],
            paper_b=data['paper_b'],
            review_a_date=datetime.fromisoformat(data['review_a_date']),
            review_b_date=datetime.fromisoformat(data['review_b_date']),
            score_a=data['score_a'],
            score_b=data['score_b'],
            decision_a=ReviewDecision(data['decision_a']),
            decision_b=ReviewDecision(data['decision_b']),
            time_gap_days=data['time_gap_days']
        )
    
    def _profile_to_dict(self, profile: ResearcherTradingProfile) -> Dict:
        """Convert researcher profile to dictionary."""
        return {
            'researcher_id': profile.researcher_id,
            'total_trading_partners': profile.total_trading_partners,
            'total_exchanges': profile.total_exchanges,
            'average_trading_score': profile.average_trading_score,
            'most_frequent_partners': profile.most_frequent_partners,
            'trading_tendency': profile.trading_tendency,
            'suspicious_patterns': profile.suspicious_patterns
        }
    
    def _dict_to_profile(self, data: Dict) -> ResearcherTradingProfile:
        """Convert dictionary to researcher profile."""
        return ResearcherTradingProfile(
            researcher_id=data['researcher_id'],
            total_trading_partners=data['total_trading_partners'],
            total_exchanges=data['total_exchanges'],
            average_trading_score=data['average_trading_score'],
            most_frequent_partners=data['most_frequent_partners'],
            trading_tendency=data['trading_tendency'],
            suspicious_patterns=data['suspicious_patterns']
        )