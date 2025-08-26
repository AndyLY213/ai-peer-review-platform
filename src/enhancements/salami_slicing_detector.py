"""
Salami Slicing Detection System

This module implements the SalamiSlicingDetector class for minimal publishable unit analysis,
identifying research broken into small pieces, and detecting strategic publication splitting.
"""

import json
import uuid
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
import math
from statistics import mean, stdev
import re
from difflib import SequenceMatcher

from src.core.exceptions import ValidationError, NetworkError
from src.core.logging_config import get_logger
from src.data.enhanced_models import EnhancedResearcher, PublicationRecord, VenueType


logger = get_logger(__name__)


@dataclass
class SalamiPattern:
    """Represents a detected salami slicing pattern."""
    pattern_id: str
    researcher_id: str
    paper_ids: List[str]
    pattern_type: str  # "incremental_splitting", "data_splitting", "method_splitting", "temporal_splitting"
    detection_date: date
    strength_score: float  # 0-1 scale indicating pattern strength
    total_papers: int
    time_span_months: int
    content_similarity: float  # Average similarity between papers
    venue_diversity: float  # Diversity of venues used
    suspicious_indicators: List[str]
    evidence: Dict[str, Any]
    
    def __post_init__(self):
        """Validate pattern data."""
        if len(self.paper_ids) < 2:
            raise ValidationError("paper_ids", "too few", "at least 2 papers")
        if not 0 <= self.strength_score <= 1:
            raise ValidationError("strength_score", self.strength_score, "0-1 range")


@dataclass
class PaperSimilarity:
    """Represents similarity between two papers."""
    paper_a: str
    paper_b: str
    title_similarity: float
    abstract_similarity: float
    keyword_similarity: float
    author_overlap: float
    venue_similarity: float
    temporal_proximity: float  # Days between publications
    overall_similarity: float = field(init=False)
    
    def __post_init__(self):
        """Calculate overall similarity score."""
        self.overall_similarity = (
            self.title_similarity * 0.25 +
            self.abstract_similarity * 0.30 +
            self.keyword_similarity * 0.20 +
            self.author_overlap * 0.15 +
            self.venue_similarity * 0.05 +
            (1.0 - min(1.0, self.temporal_proximity / 365.0)) * 0.05  # Closer in time = more similar
        )


@dataclass
class ResearcherSalamiProfile:
    """Profile of a researcher's salami slicing behavior."""
    researcher_id: str
    total_papers: int = 0
    suspected_salami_papers: int = 0
    salami_patterns: List[str] = field(default_factory=list)  # Pattern IDs
    average_pattern_strength: float = 0.0
    publication_frequency: float = 0.0  # Papers per year
    venue_shopping_tendency: float = 0.0  # Tendency to use lower-tier venues
    incremental_publication_score: float = 0.0  # 0-1 scale
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    
    def update_profile(self, patterns: List[SalamiPattern], total_papers: int, years_active: int):
        """Update profile based on detected patterns."""
        self.total_papers = total_papers
        self.suspected_salami_papers = sum(len(p.paper_ids) for p in patterns)
        self.salami_patterns = [p.pattern_id for p in patterns]
        
        if patterns:
            self.average_pattern_strength = mean([p.strength_score for p in patterns])
        
        if years_active > 0:
            self.publication_frequency = total_papers / years_active
        
        # Calculate incremental publication score
        if total_papers > 0:
            self.incremental_publication_score = self.suspected_salami_papers / total_papers
        
        # Determine risk level
        self._calculate_risk_level()
    
    def _calculate_risk_level(self):
        """Calculate risk level based on multiple factors."""
        risk_score = (
            self.incremental_publication_score * 0.4 +
            self.average_pattern_strength * 0.3 +
            min(1.0, self.publication_frequency / 10.0) * 0.2 +  # Normalize to 10 papers/year
            self.venue_shopping_tendency * 0.1
        )
        
        if risk_score >= 0.8:
            self.risk_level = "CRITICAL"
        elif risk_score >= 0.6:
            self.risk_level = "HIGH"
        elif risk_score >= 0.3:
            self.risk_level = "MEDIUM"
        else:
            self.risk_level = "LOW"


class SalamiSlicingDetector:
    """
    Detects salami slicing patterns where researchers break work into minimal publishable units.
    
    This class implements comprehensive salami slicing detection including:
    - Content similarity analysis
    - Temporal publication pattern analysis
    - Venue selection pattern analysis
    - Incremental contribution detection
    """
    
    def __init__(self, similarity_threshold: float = 0.7, min_papers_for_pattern: int = 3,
                 max_time_span_months: int = 24):
        """
        Initialize salami slicing detector.
        
        Args:
            similarity_threshold: Minimum similarity to consider papers related
            min_papers_for_pattern: Minimum papers needed to establish a pattern
            max_time_span_months: Maximum time span for related papers
        """
        self.similarity_threshold = similarity_threshold
        self.min_papers_for_pattern = min_papers_for_pattern
        self.max_time_span_months = max_time_span_months
        
        # Detection results
        self.detected_patterns: List[SalamiPattern] = []
        self.researcher_profiles: Dict[str, ResearcherSalamiProfile] = {}
        
        # Paper data
        self.papers: Dict[str, Dict[str, Any]] = {}  # paper_id -> paper info
        self.researcher_papers: Dict[str, List[str]] = defaultdict(list)  # researcher_id -> paper_ids
        self.paper_similarities: Dict[Tuple[str, str], PaperSimilarity] = {}
        
        logger.info(f"Initialized SalamiSlicingDetector with similarity_threshold={similarity_threshold}, "
                   f"min_papers_for_pattern={min_papers_for_pattern}, max_time_span_months={max_time_span_months}")
    
    def add_paper(self, paper_id: str, title: str, abstract: str, authors: List[str],
                  keywords: List[str], venue_id: str, venue_type: VenueType,
                  publication_date: date, primary_author: str) -> None:
        """
        Add a paper for salami slicing analysis.
        
        Args:
            paper_id: Unique identifier for the paper
            title: Paper title
            abstract: Paper abstract
            authors: List of author IDs
            keywords: List of keywords
            venue_id: Venue identifier
            venue_type: Type of venue
            publication_date: Date of publication
            primary_author: Primary/corresponding author ID
        """
        paper_info = {
            'paper_id': paper_id,
            'title': title,
            'abstract': abstract,
            'authors': authors,
            'keywords': keywords,
            'venue_id': venue_id,
            'venue_type': venue_type,
            'publication_date': publication_date,
            'primary_author': primary_author
        }
        
        self.papers[paper_id] = paper_info
        
        # Add to researcher's paper list
        for author in authors:
            self.researcher_papers[author].append(paper_id)
        
        logger.debug(f"Added paper {paper_id} by {primary_author} to salami slicing analysis")
    
    def calculate_paper_similarity(self, paper_a_id: str, paper_b_id: str) -> PaperSimilarity:
        """
        Calculate similarity between two papers.
        
        Args:
            paper_a_id: First paper ID
            paper_b_id: Second paper ID
            
        Returns:
            PaperSimilarity: Similarity metrics between the papers
        """
        if paper_a_id not in self.papers or paper_b_id not in self.papers:
            raise ValidationError("paper_id", "not found", "existing paper ID")
        
        paper_a = self.papers[paper_a_id]
        paper_b = self.papers[paper_b_id]
        
        # Title similarity
        title_similarity = self._calculate_text_similarity(paper_a['title'], paper_b['title'])
        
        # Abstract similarity
        abstract_similarity = self._calculate_text_similarity(paper_a['abstract'], paper_b['abstract'])
        
        # Keyword similarity
        keyword_similarity = self._calculate_keyword_similarity(paper_a['keywords'], paper_b['keywords'])
        
        # Author overlap
        authors_a = set(paper_a['authors'])
        authors_b = set(paper_b['authors'])
        author_overlap = len(authors_a.intersection(authors_b)) / len(authors_a.union(authors_b)) if authors_a.union(authors_b) else 0.0
        
        # Venue similarity
        venue_similarity = 1.0 if paper_a['venue_type'] == paper_b['venue_type'] else 0.0
        
        # Temporal proximity
        date_diff = abs((paper_a['publication_date'] - paper_b['publication_date']).days)
        temporal_proximity = date_diff
        
        similarity = PaperSimilarity(
            paper_a=paper_a_id,
            paper_b=paper_b_id,
            title_similarity=title_similarity,
            abstract_similarity=abstract_similarity,
            keyword_similarity=keyword_similarity,
            author_overlap=author_overlap,
            venue_similarity=venue_similarity,
            temporal_proximity=temporal_proximity
        )
        
        return similarity
    
    def _calculate_text_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate similarity between two text strings."""
        if not text_a or not text_b:
            return 0.0
        
        # Normalize text
        text_a = text_a.lower().strip()
        text_b = text_b.lower().strip()
        
        # Use sequence matcher for similarity
        similarity = SequenceMatcher(None, text_a, text_b).ratio()
        
        # Also check for common phrases and technical terms
        words_a = set(re.findall(r'\b\w+\b', text_a))
        words_b = set(re.findall(r'\b\w+\b', text_b))
        
        if words_a and words_b:
            word_overlap = len(words_a.intersection(words_b)) / len(words_a.union(words_b))
            # Combine sequence similarity with word overlap
            similarity = (similarity * 0.7) + (word_overlap * 0.3)
        
        return similarity
    
    def _calculate_keyword_similarity(self, keywords_a: List[str], keywords_b: List[str]) -> float:
        """Calculate similarity between keyword lists."""
        if not keywords_a or not keywords_b:
            return 0.0
        
        set_a = set(kw.lower().strip() for kw in keywords_a)
        set_b = set(kw.lower().strip() for kw in keywords_b)
        
        if not set_a or not set_b:
            return 0.0
        
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        
        return intersection / union if union > 0 else 0.0
    
    def detect_salami_patterns(self, researcher_id: str) -> List[SalamiPattern]:
        """
        Detect salami slicing patterns for a specific researcher.
        
        Args:
            researcher_id: Researcher to analyze
            
        Returns:
            List[SalamiPattern]: Detected salami slicing patterns
        """
        if researcher_id not in self.researcher_papers:
            return []
        
        paper_ids = self.researcher_papers[researcher_id]
        if len(paper_ids) < self.min_papers_for_pattern:
            return []
        
        patterns = []
        
        # Calculate similarities between all paper pairs
        similarities = []
        for i, paper_a in enumerate(paper_ids):
            for paper_b in paper_ids[i+1:]:
                similarity = self.calculate_paper_similarity(paper_a, paper_b)
                similarities.append(similarity)
                
                # Cache similarity
                pair_key = tuple(sorted([paper_a, paper_b]))
                self.paper_similarities[pair_key] = similarity
        
        # Group highly similar papers
        similar_groups = self._find_similar_paper_groups(paper_ids, similarities)
        
        # Analyze each group for salami patterns
        for group in similar_groups:
            if len(group) >= self.min_papers_for_pattern:
                pattern = self._analyze_paper_group_for_salami(researcher_id, group)
                if pattern and pattern.strength_score >= 0.3:  # Minimum threshold
                    patterns.append(pattern)
        
        # Update researcher profile
        self._update_researcher_profile(researcher_id, patterns)
        
        return patterns
    
    def _find_similar_paper_groups(self, paper_ids: List[str], similarities: List[PaperSimilarity]) -> List[List[str]]:
        """Find groups of similar papers using clustering."""
        # Create adjacency list of similar papers
        adjacency = defaultdict(set)
        
        for similarity in similarities:
            if similarity.overall_similarity >= self.similarity_threshold:
                adjacency[similarity.paper_a].add(similarity.paper_b)
                adjacency[similarity.paper_b].add(similarity.paper_a)
        
        # Find connected components (groups)
        visited = set()
        groups = []
        
        def dfs(paper_id: str, group: List[str]):
            if paper_id in visited:
                return
            visited.add(paper_id)
            group.append(paper_id)
            for neighbor in adjacency[paper_id]:
                dfs(neighbor, group)
        
        for paper_id in paper_ids:
            if paper_id not in visited:
                group = []
                dfs(paper_id, group)
                if len(group) >= 2:  # Only include groups with multiple papers
                    groups.append(group)
        
        return groups
    
    def _analyze_paper_group_for_salami(self, researcher_id: str, paper_ids: List[str]) -> Optional[SalamiPattern]:
        """Analyze a group of similar papers for salami slicing patterns."""
        if len(paper_ids) < 2:
            return None
        
        papers = [self.papers[pid] for pid in paper_ids]
        
        # Sort by publication date
        papers.sort(key=lambda p: p['publication_date'])
        
        # Calculate temporal span
        first_date = papers[0]['publication_date']
        last_date = papers[-1]['publication_date']
        time_span_months = ((last_date - first_date).days / 30.44)  # Average days per month
        
        if time_span_months > self.max_time_span_months:
            return None  # Too spread out in time
        
        # Calculate content similarity metrics
        similarities = []
        for i, paper_a in enumerate(papers):
            for paper_b in papers[i+1:]:
                pair_key = tuple(sorted([paper_a['paper_id'], paper_b['paper_id']]))
                if pair_key in self.paper_similarities:
                    similarities.append(self.paper_similarities[pair_key].overall_similarity)
        
        if not similarities:
            return None
        
        avg_similarity = mean(similarities)
        
        # Analyze venue diversity
        venues = [p['venue_type'] for p in papers]
        venue_diversity = len(set(venues)) / len(venues) if venues else 0.0
        
        # Determine pattern type and calculate strength
        pattern_type, strength_score, indicators = self._classify_salami_pattern(papers, avg_similarity, venue_diversity)
        
        if strength_score < 0.3:  # Minimum threshold
            return None
        
        pattern = SalamiPattern(
            pattern_id=str(uuid.uuid4()),
            researcher_id=researcher_id,
            paper_ids=paper_ids,
            pattern_type=pattern_type,
            detection_date=date.today(),
            strength_score=strength_score,
            total_papers=len(paper_ids),
            time_span_months=int(time_span_months),
            content_similarity=avg_similarity,
            venue_diversity=venue_diversity,
            suspicious_indicators=indicators,
            evidence={
                'publication_dates': [p['publication_date'].isoformat() for p in papers],
                'venues': [p['venue_id'] for p in papers],
                'venue_types': [p['venue_type'].value for p in papers],
                'similarities': similarities,
                'avg_similarity': avg_similarity,
                'time_span_months': time_span_months
            }
        )
        
        return pattern
    
    def _classify_salami_pattern(self, papers: List[Dict], avg_similarity: float, 
                               venue_diversity: float) -> Tuple[str, float, List[str]]:
        """Classify the type of salami slicing pattern and calculate strength."""
        indicators = []
        strength_factors = []
        
        # Analyze publication timing
        dates = [p['publication_date'] for p in papers]
        time_gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        avg_gap = mean(time_gaps) if time_gaps else 0
        
        # Incremental splitting indicators
        if avg_similarity > 0.8:
            indicators.append("Very high content similarity")
            strength_factors.append(0.9)
        elif avg_similarity > 0.6:
            indicators.append("High content similarity")
            strength_factors.append(0.7)
        
        # Temporal splitting indicators
        if avg_gap < 90:  # Less than 3 months between papers
            indicators.append("Rapid sequential publication")
            strength_factors.append(0.8)
        elif avg_gap < 180:  # Less than 6 months
            indicators.append("Quick sequential publication")
            strength_factors.append(0.6)
        
        # Venue shopping indicators
        venue_types = [p['venue_type'] for p in papers]
        venue_prestige_decline = self._detect_venue_prestige_decline(venue_types)
        if venue_prestige_decline:
            indicators.append("Declining venue prestige")
            strength_factors.append(0.7)
        
        if venue_diversity < 0.5:  # Low venue diversity
            indicators.append("Limited venue diversity")
            strength_factors.append(0.5)
        
        # Method/data splitting indicators
        titles = [p['title'] for p in papers]
        if self._detect_incremental_titles(titles):
            indicators.append("Incremental title patterns")
            strength_factors.append(0.8)
        
        # Determine pattern type
        if avg_similarity > 0.8 and avg_gap < 120:
            pattern_type = "incremental_splitting"
        elif venue_prestige_decline:
            pattern_type = "venue_shopping_splitting"
        elif self._detect_data_splitting_pattern(papers):
            pattern_type = "data_splitting"
        elif avg_gap < 90:
            pattern_type = "temporal_splitting"
        else:
            pattern_type = "method_splitting"
        
        # Calculate overall strength score
        if strength_factors:
            strength_score = mean(strength_factors)
            # Boost score for multiple indicators
            if len(indicators) >= 3:
                strength_score = min(1.0, strength_score * 1.2)
        else:
            strength_score = 0.0
        
        return pattern_type, strength_score, indicators
    
    def _detect_venue_prestige_decline(self, venue_types: List[VenueType]) -> bool:
        """Detect if there's a decline in venue prestige over time."""
        prestige_scores = {
            VenueType.TOP_JOURNAL: 10,
            VenueType.TOP_CONFERENCE: 9,
            VenueType.SPECIALIZED_JOURNAL: 7,
            VenueType.MID_CONFERENCE: 6,
            VenueType.GENERAL_JOURNAL: 5,
            VenueType.LOW_CONFERENCE: 4,
            VenueType.WORKSHOP: 3,
            VenueType.PREPRINT: 2
        }
        
        scores = [prestige_scores.get(vt, 5) for vt in venue_types]
        
        # Check for declining trend
        if len(scores) < 2:
            return False
        
        declining_count = 0
        for i in range(1, len(scores)):
            if scores[i] < scores[i-1]:
                declining_count += 1
        
        return declining_count >= len(scores) // 2  # At least half are declining
    
    def _detect_incremental_titles(self, titles: List[str]) -> bool:
        """Detect incremental patterns in paper titles."""
        # Look for common incremental patterns
        incremental_patterns = [
            r'part\s+\d+', r'volume\s+\d+', r'chapter\s+\d+',
            r'study\s+\d+', r'analysis\s+\d+', r'approach\s+\d+',
            r'method\s+\d+', r'algorithm\s+\d+', r'model\s+\d+',
            r':\s*\d+', r'i{1,3}$', r'\d+$'  # Roman numerals or numbers at end
        ]
        
        incremental_count = 0
        for title in titles:
            title_lower = title.lower()
            for pattern in incremental_patterns:
                if re.search(pattern, title_lower):
                    incremental_count += 1
                    break
        
        return incremental_count >= len(titles) // 2  # At least half have incremental patterns
    
    def _detect_data_splitting_pattern(self, papers: List[Dict]) -> bool:
        """Detect if papers appear to split data or experiments."""
        # Look for keywords indicating data/experiment splitting
        splitting_keywords = [
            'dataset', 'subset', 'sample', 'cohort', 'group',
            'experiment', 'trial', 'study', 'analysis',
            'phase', 'stage', 'part', 'section'
        ]
        
        keyword_matches = 0
        for paper in papers:
            title_abstract = (paper['title'] + ' ' + paper['abstract']).lower()
            for keyword in splitting_keywords:
                if keyword in title_abstract:
                    keyword_matches += 1
                    break
        
        return keyword_matches >= len(papers) * 0.6  # At least 60% have splitting keywords
    
    def _update_researcher_profile(self, researcher_id: str, patterns: List[SalamiPattern]):
        """Update researcher's salami slicing profile."""
        total_papers = len(self.researcher_papers[researcher_id])
        
        # Calculate years active (rough estimate)
        paper_dates = [self.papers[pid]['publication_date'] for pid in self.researcher_papers[researcher_id]]
        if paper_dates:
            earliest = min(paper_dates)
            latest = max(paper_dates)
            years_active = max(1, (latest - earliest).days / 365.25)
        else:
            years_active = 1
        
        # Create or update profile
        if researcher_id not in self.researcher_profiles:
            self.researcher_profiles[researcher_id] = ResearcherSalamiProfile(researcher_id=researcher_id)
        
        profile = self.researcher_profiles[researcher_id]
        profile.update_profile(patterns, total_papers, years_active)
        
        # Calculate venue shopping tendency
        venues = [self.papers[pid]['venue_type'] for pid in self.researcher_papers[researcher_id]]
        venue_prestige_decline = self._detect_venue_prestige_decline(venues)
        profile.venue_shopping_tendency = 1.0 if venue_prestige_decline else 0.0
    
    def analyze_all_researchers(self) -> Dict[str, List[SalamiPattern]]:
        """
        Analyze all researchers for salami slicing patterns.
        
        Returns:
            Dict[str, List[SalamiPattern]]: Mapping of researcher IDs to their patterns
        """
        all_patterns = {}
        
        for researcher_id in self.researcher_papers:
            patterns = self.detect_salami_patterns(researcher_id)
            if patterns:
                all_patterns[researcher_id] = patterns
                self.detected_patterns.extend(patterns)
        
        logger.info(f"Analyzed {len(self.researcher_papers)} researchers, "
                   f"found {len(self.detected_patterns)} salami slicing patterns")
        
        return all_patterns
    
    def get_researcher_salami_behavior(self, researcher_id: str) -> Optional[ResearcherSalamiProfile]:
        """
        Get salami slicing behavior profile for a researcher.
        
        Args:
            researcher_id: Researcher identifier
            
        Returns:
            ResearcherSalamiProfile if available, None otherwise
        """
        return self.researcher_profiles.get(researcher_id)
    
    def calculate_salami_statistics(self) -> Dict[str, Any]:
        """
        Calculate overall salami slicing statistics.
        
        Returns:
            Dict[str, Any]: Comprehensive statistics
        """
        if not self.detected_patterns:
            return {
                'total_patterns': 0,
                'total_researchers': len(self.researcher_profiles),
                'total_papers_involved': 0,
                'pattern_types': {},
                'average_pattern_strength': 0.0,
                'high_risk_researchers': 0,
                'researchers_by_risk': {
                    risk: len([p for p in self.researcher_profiles.values() if p.risk_level == risk])
                    for risk in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
                }
            }
        
        pattern_types = Counter([p.pattern_type for p in self.detected_patterns])
        total_papers_involved = sum(p.total_papers for p in self.detected_patterns)
        avg_strength = mean([p.strength_score for p in self.detected_patterns])
        
        high_risk_researchers = len([
            p for p in self.researcher_profiles.values()
            if p.risk_level in ['HIGH', 'CRITICAL']
        ])
        
        return {
            'total_patterns': len(self.detected_patterns),
            'total_researchers': len(self.researcher_profiles),
            'total_papers_involved': total_papers_involved,
            'pattern_types': dict(pattern_types),
            'average_pattern_strength': avg_strength,
            'high_risk_researchers': high_risk_researchers,
            'researchers_by_risk': {
                risk: len([p for p in self.researcher_profiles.values() if p.risk_level == risk])
                for risk in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            }
        }
    
    def generate_salami_report(self, researcher_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive salami slicing report.
        
        Args:
            researcher_id: If provided, focus on specific researcher
            
        Returns:
            Dict[str, Any]: Comprehensive report
        """
        if researcher_id:
            # Researcher-specific report
            profile = self.researcher_profiles.get(researcher_id)
            if not profile:
                return {'error': f'No salami slicing data found for researcher {researcher_id}'}
            
            researcher_patterns = [p for p in self.detected_patterns if p.researcher_id == researcher_id]
            
            return {
                'researcher_id': researcher_id,
                'profile': {
                    'total_papers': profile.total_papers,
                    'suspected_salami_papers': profile.suspected_salami_papers,
                    'salami_rate': profile.incremental_publication_score,
                    'average_pattern_strength': profile.average_pattern_strength,
                    'publication_frequency': profile.publication_frequency,
                    'risk_level': profile.risk_level
                },
                'patterns': [self._pattern_to_dict(p) for p in researcher_patterns],
                'recommendations': self._generate_recommendations(profile)
            }
        else:
            # Global report
            stats = self.calculate_salami_statistics()
            top_patterns = sorted(self.detected_patterns, key=lambda x: x.strength_score, reverse=True)[:10]
            
            return {
                'statistics': stats,
                'top_patterns': [self._pattern_to_dict(p) for p in top_patterns],
                'high_risk_researchers': [
                    {
                        'researcher_id': rid,
                        'risk_level': profile.risk_level,
                        'salami_rate': profile.incremental_publication_score,
                        'pattern_count': len(profile.salami_patterns)
                    }
                    for rid, profile in self.researcher_profiles.items()
                    if profile.risk_level in ['HIGH', 'CRITICAL']
                ]
            }
    
    def _generate_recommendations(self, profile: ResearcherSalamiProfile) -> List[str]:
        """Generate recommendations based on researcher profile."""
        recommendations = []
        
        if profile.risk_level == 'CRITICAL':
            recommendations.append("Immediate review of publication practices recommended")
            recommendations.append("Consider consolidating related work into comprehensive papers")
        elif profile.risk_level == 'HIGH':
            recommendations.append("Review publication strategy to ensure substantial contributions")
            recommendations.append("Focus on quality over quantity in publications")
        
        if profile.incremental_publication_score > 0.5:
            recommendations.append("High rate of potentially incremental publications detected")
        
        if profile.publication_frequency > 8:
            recommendations.append("Very high publication frequency may indicate salami slicing")
        
        if profile.venue_shopping_tendency > 0.5:
            recommendations.append("Pattern of declining venue prestige detected")
        
        return recommendations
    
    def _pattern_to_dict(self, pattern: SalamiPattern) -> Dict[str, Any]:
        """Convert pattern to dictionary for serialization."""
        return {
            'pattern_id': pattern.pattern_id,
            'researcher_id': pattern.researcher_id,
            'paper_ids': pattern.paper_ids,
            'pattern_type': pattern.pattern_type,
            'detection_date': pattern.detection_date.isoformat(),
            'strength_score': pattern.strength_score,
            'total_papers': pattern.total_papers,
            'time_span_months': pattern.time_span_months,
            'content_similarity': pattern.content_similarity,
            'venue_diversity': pattern.venue_diversity,
            'suspicious_indicators': pattern.suspicious_indicators,
            'evidence': pattern.evidence
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'config': {
                'similarity_threshold': self.similarity_threshold,
                'min_papers_for_pattern': self.min_papers_for_pattern,
                'max_time_span_months': self.max_time_span_months
            },
            'detected_patterns': [self._pattern_to_dict(p) for p in self.detected_patterns],
            'researcher_profiles': {
                rid: {
                    'researcher_id': profile.researcher_id,
                    'total_papers': profile.total_papers,
                    'suspected_salami_papers': profile.suspected_salami_papers,
                    'salami_patterns': profile.salami_patterns,
                    'average_pattern_strength': profile.average_pattern_strength,
                    'publication_frequency': profile.publication_frequency,
                    'venue_shopping_tendency': profile.venue_shopping_tendency,
                    'incremental_publication_score': profile.incremental_publication_score,
                    'risk_level': profile.risk_level
                }
                for rid, profile in self.researcher_profiles.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SalamiSlicingDetector':
        """Create from dictionary."""
        config = data.get('config', {})
        detector = cls(
            similarity_threshold=config.get('similarity_threshold', 0.7),
            min_papers_for_pattern=config.get('min_papers_for_pattern', 3),
            max_time_span_months=config.get('max_time_span_months', 24)
        )
        
        # Load detected patterns
        for pattern_data in data.get('detected_patterns', []):
            pattern = SalamiPattern(
                pattern_id=pattern_data['pattern_id'],
                researcher_id=pattern_data['researcher_id'],
                paper_ids=pattern_data['paper_ids'],
                pattern_type=pattern_data['pattern_type'],
                detection_date=date.fromisoformat(pattern_data['detection_date']),
                strength_score=pattern_data['strength_score'],
                total_papers=pattern_data['total_papers'],
                time_span_months=pattern_data['time_span_months'],
                content_similarity=pattern_data['content_similarity'],
                venue_diversity=pattern_data['venue_diversity'],
                suspicious_indicators=pattern_data['suspicious_indicators'],
                evidence=pattern_data['evidence']
            )
            detector.detected_patterns.append(pattern)
        
        # Load researcher profiles
        for rid, profile_data in data.get('researcher_profiles', {}).items():
            profile = ResearcherSalamiProfile(
                researcher_id=profile_data['researcher_id'],
                total_papers=profile_data['total_papers'],
                suspected_salami_papers=profile_data['suspected_salami_papers'],
                salami_patterns=profile_data['salami_patterns'],
                average_pattern_strength=profile_data['average_pattern_strength'],
                publication_frequency=profile_data['publication_frequency'],
                venue_shopping_tendency=profile_data['venue_shopping_tendency'],
                incremental_publication_score=profile_data['incremental_publication_score'],
                risk_level=profile_data['risk_level']
            )
            detector.researcher_profiles[rid] = profile
        
        return detector