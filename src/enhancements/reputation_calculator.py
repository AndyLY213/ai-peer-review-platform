"""
Reputation Scoring System

This module implements a comprehensive reputation scoring system for researchers
with h-index, citations, years active, and institutional tier calculations.
Provides logic to combine multiple reputation metrics into overall scores.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import math

from src.data.enhanced_models import EnhancedResearcher, ResearcherLevel
from src.core.exceptions import ValidationError
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class InstitutionTier(Enum):
    """Institution tier classifications."""
    TIER_1 = 1  # Top-tier institutions (e.g., MIT, Stanford, Harvard)
    TIER_2 = 2  # Mid-tier institutions (strong regional universities)
    TIER_3 = 3  # Lower-tier institutions (smaller colleges, teaching-focused)


@dataclass
class ReputationMetrics:
    """Container for individual reputation metrics."""
    h_index_score: float
    citation_score: float
    experience_score: float
    institutional_score: float
    productivity_score: float
    impact_score: float
    overall_score: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'h_index_score': self.h_index_score,
            'citation_score': self.citation_score,
            'experience_score': self.experience_score,
            'institutional_score': self.institutional_score,
            'productivity_score': self.productivity_score,
            'impact_score': self.impact_score,
            'overall_score': self.overall_score
        }


class ReputationCalculator:
    """
    Comprehensive reputation scoring system for researchers.
    
    This class implements reputation calculations based on:
    - H-index with field-normalized scoring
    - Total citations with recency weighting
    - Years active with experience curves
    - Institutional tier with influence calculations
    - Publication productivity metrics
    - Research impact indicators
    """
    
    # H-index benchmarks by career level (50th percentile values)
    H_INDEX_BENCHMARKS = {
        ResearcherLevel.GRADUATE_STUDENT: 2,
        ResearcherLevel.POSTDOC: 5,
        ResearcherLevel.ASSISTANT_PROF: 12,
        ResearcherLevel.ASSOCIATE_PROF: 20,
        ResearcherLevel.FULL_PROF: 35,
        ResearcherLevel.EMERITUS: 45
    }
    
    # Citation benchmarks by career level (50th percentile values)
    CITATION_BENCHMARKS = {
        ResearcherLevel.GRADUATE_STUDENT: 50,
        ResearcherLevel.POSTDOC: 200,
        ResearcherLevel.ASSISTANT_PROF: 800,
        ResearcherLevel.ASSOCIATE_PROF: 2000,
        ResearcherLevel.FULL_PROF: 5000,
        ResearcherLevel.EMERITUS: 8000
    }
    
    # Years active benchmarks by career level
    EXPERIENCE_BENCHMARKS = {
        ResearcherLevel.GRADUATE_STUDENT: 3,
        ResearcherLevel.POSTDOC: 2,
        ResearcherLevel.ASSISTANT_PROF: 6,
        ResearcherLevel.ASSOCIATE_PROF: 12,
        ResearcherLevel.FULL_PROF: 20,
        ResearcherLevel.EMERITUS: 35
    }
    
    # Institutional tier influence multipliers
    INSTITUTIONAL_MULTIPLIERS = {
        InstitutionTier.TIER_1: 1.3,  # 30% bonus for top institutions
        InstitutionTier.TIER_2: 1.0,  # Baseline for mid-tier
        InstitutionTier.TIER_3: 0.8   # 20% penalty for lower-tier
    }
    
    # Institutional tier prestige scores
    INSTITUTIONAL_PRESTIGE = {
        InstitutionTier.TIER_1: 0.9,  # High prestige
        InstitutionTier.TIER_2: 0.6,  # Medium prestige
        InstitutionTier.TIER_3: 0.3   # Lower prestige
    }
    
    def __init__(self):
        """Initialize the reputation calculator."""
        logger.info("Initializing Reputation Calculator System")
    
    def calculate_reputation_score(self, researcher: EnhancedResearcher) -> float:
        """
        Calculate comprehensive reputation score for a researcher.
        
        Args:
            researcher: The researcher to calculate reputation for
            
        Returns:
            Overall reputation score (0.0 to 1.0)
        """
        try:
            metrics = self.calculate_detailed_metrics(researcher)
            return metrics.overall_score
        except Exception as e:
            logger.error(f"Error calculating reputation for {researcher.name}: {e}")
            return 0.0
    
    def calculate_detailed_metrics(self, researcher: EnhancedResearcher) -> ReputationMetrics:
        """
        Calculate detailed reputation metrics for a researcher.
        
        Args:
            researcher: The researcher to analyze
            
        Returns:
            ReputationMetrics object with detailed breakdown
        """
        # Calculate individual metric scores
        h_index_score = self._calculate_h_index_score(researcher)
        citation_score = self._calculate_citation_score(researcher)
        experience_score = self._calculate_experience_score(researcher)
        institutional_score = self._calculate_institutional_score(researcher)
        productivity_score = self._calculate_productivity_score(researcher)
        impact_score = self._calculate_impact_score(researcher)
        
        # Combine metrics with weights
        overall_score = self._combine_metrics(
            h_index_score, citation_score, experience_score,
            institutional_score, productivity_score, impact_score
        )
        
        metrics = ReputationMetrics(
            h_index_score=h_index_score,
            citation_score=citation_score,
            experience_score=experience_score,
            institutional_score=institutional_score,
            productivity_score=productivity_score,
            impact_score=impact_score,
            overall_score=overall_score
        )
        
        logger.debug(f"Calculated reputation metrics for {researcher.name}: {metrics.to_dict()}")
        return metrics
    
    def _calculate_h_index_score(self, researcher: EnhancedResearcher) -> float:
        """Calculate normalized h-index score relative to career level expectations."""
        benchmark = self.H_INDEX_BENCHMARKS.get(researcher.level, 10)
        
        if benchmark == 0:
            return 1.0 if researcher.h_index > 0 else 0.0
        
        # Use logarithmic scaling to handle wide range of h-index values
        if researcher.h_index <= 0:
            return 0.0
        
        # Score relative to benchmark with diminishing returns
        ratio = researcher.h_index / benchmark
        score = min(1.0, math.log(1 + ratio) / math.log(3))  # Caps at ~1.0 when ratio = 2
        
        return max(0.0, score)
    
    def _calculate_citation_score(self, researcher: EnhancedResearcher) -> float:
        """Calculate normalized citation score with recency weighting."""
        benchmark = self.CITATION_BENCHMARKS.get(researcher.level, 500)
        
        if benchmark == 0:
            return 1.0 if researcher.total_citations > 0 else 0.0
        
        if researcher.total_citations <= 0:
            return 0.0
        
        # Use square root scaling for citations to reduce extreme values
        ratio = researcher.total_citations / benchmark
        score = min(1.0, math.sqrt(ratio))
        
        # Apply recency weighting if publication history is available
        if researcher.publication_history:
            recency_weight = self._calculate_recency_weight(researcher)
            score *= recency_weight
        
        return min(1.0, max(0.0, score))
    
    def _calculate_experience_score(self, researcher: EnhancedResearcher) -> float:
        """Calculate experience score with diminishing returns for very long careers."""
        benchmark = self.EXPERIENCE_BENCHMARKS.get(researcher.level, 5)
        
        if researcher.years_active <= 0:
            return 0.0
        
        # Experience score with diminishing returns after benchmark
        if researcher.years_active <= benchmark:
            score = researcher.years_active / benchmark
        else:
            # Diminishing returns for experience beyond benchmark
            excess_years = researcher.years_active - benchmark
            score = 1.0 + (math.log(1 + excess_years) / math.log(10)) * 0.2
        
        return min(1.0, max(0.0, score))
    
    def _calculate_institutional_score(self, researcher: EnhancedResearcher) -> float:
        """Calculate institutional prestige score."""
        try:
            tier = InstitutionTier(researcher.institution_tier)
            return self.INSTITUTIONAL_PRESTIGE[tier]
        except ValueError:
            logger.warning(f"Invalid institution tier: {researcher.institution_tier}, using tier 2")
            return self.INSTITUTIONAL_PRESTIGE[InstitutionTier.TIER_2]
    
    def _calculate_productivity_score(self, researcher: EnhancedResearcher) -> float:
        """Calculate productivity score based on publications per year."""
        if researcher.years_active <= 0:
            return 0.0
        
        publications_per_year = len(researcher.publication_history) / researcher.years_active
        
        # Expected publications per year by level
        expected_productivity = {
            ResearcherLevel.GRADUATE_STUDENT: 0.5,
            ResearcherLevel.POSTDOC: 1.5,
            ResearcherLevel.ASSISTANT_PROF: 2.0,
            ResearcherLevel.ASSOCIATE_PROF: 2.5,
            ResearcherLevel.FULL_PROF: 2.0,  # Quality over quantity
            ResearcherLevel.EMERITUS: 1.0
        }
        
        expected = expected_productivity.get(researcher.level, 1.5)
        
        if expected == 0:
            return 1.0 if publications_per_year > 0 else 0.0
        
        ratio = publications_per_year / expected
        score = min(1.0, ratio)  # Linear up to expected, then capped
        
        return max(0.0, score)
    
    def _calculate_impact_score(self, researcher: EnhancedResearcher) -> float:
        """Calculate research impact score based on citations per paper."""
        if not researcher.publication_history:
            return 0.0
        
        total_citations = sum(pub.citations for pub in researcher.publication_history)
        total_papers = len(researcher.publication_history)
        
        if total_papers == 0:
            return 0.0
        
        citations_per_paper = total_citations / total_papers
        
        # Expected citations per paper by level
        expected_impact = {
            ResearcherLevel.GRADUATE_STUDENT: 5,
            ResearcherLevel.POSTDOC: 10,
            ResearcherLevel.ASSISTANT_PROF: 15,
            ResearcherLevel.ASSOCIATE_PROF: 25,
            ResearcherLevel.FULL_PROF: 40,
            ResearcherLevel.EMERITUS: 50
        }
        
        expected = expected_impact.get(researcher.level, 15)
        
        if expected == 0:
            return 1.0 if citations_per_paper > 0 else 0.0
        
        ratio = citations_per_paper / expected
        score = min(1.0, math.sqrt(ratio))  # Square root scaling for impact
        
        return max(0.0, score)
    
    def _calculate_recency_weight(self, researcher: EnhancedResearcher) -> float:
        """Calculate recency weight based on recent publication activity."""
        if not researcher.publication_history:
            return 1.0
        
        current_year = 2024  # Could be made dynamic
        recent_years = 5
        
        recent_publications = [
            pub for pub in researcher.publication_history
            if current_year - pub.year <= recent_years
        ]
        
        if not recent_publications:
            return 0.7  # Penalty for no recent publications
        
        recent_ratio = len(recent_publications) / len(researcher.publication_history)
        
        # Weight between 0.7 (no recent work) and 1.1 (all recent work)
        weight = 0.7 + (recent_ratio * 0.4)
        
        return min(1.1, weight)
    
    def _combine_metrics(self, h_index_score: float, citation_score: float,
                        experience_score: float, institutional_score: float,
                        productivity_score: float, impact_score: float) -> float:
        """
        Combine individual metrics into overall reputation score.
        
        Weights are designed to balance different aspects of academic reputation:
        - H-index and citations are most important (research quality)
        - Impact and productivity measure research effectiveness
        - Experience provides career context
        - Institution provides prestige context
        """
        weights = {
            'h_index': 0.25,      # Primary research quality indicator
            'citations': 0.20,    # Overall research influence
            'impact': 0.20,       # Research effectiveness
            'productivity': 0.15, # Research output
            'experience': 0.10,   # Career development
            'institutional': 0.10 # Prestige context
        }
        
        overall_score = (
            h_index_score * weights['h_index'] +
            citation_score * weights['citations'] +
            impact_score * weights['impact'] +
            productivity_score * weights['productivity'] +
            experience_score * weights['experience'] +
            institutional_score * weights['institutional']
        )
        
        return min(1.0, max(0.0, overall_score))
    
    def get_institutional_influence_multiplier(self, researcher: EnhancedResearcher) -> float:
        """
        Get institutional influence multiplier for review weight calculations.
        
        Args:
            researcher: The researcher to get multiplier for
            
        Returns:
            Influence multiplier based on institutional tier
        """
        try:
            tier = InstitutionTier(researcher.institution_tier)
            multiplier = self.INSTITUTIONAL_MULTIPLIERS[tier]
            
            logger.debug(f"Institutional multiplier for {researcher.name} (Tier {tier.value}): {multiplier}")
            return multiplier
        except ValueError:
            logger.warning(f"Invalid institution tier: {researcher.institution_tier}, using tier 2 multiplier")
            return self.INSTITUTIONAL_MULTIPLIERS[InstitutionTier.TIER_2]
    
    def compare_researchers(self, researcher1: EnhancedResearcher, 
                          researcher2: EnhancedResearcher) -> Dict[str, any]:
        """
        Compare reputation metrics between two researchers.
        
        Args:
            researcher1: First researcher to compare
            researcher2: Second researcher to compare
            
        Returns:
            Dictionary with comparison results
        """
        metrics1 = self.calculate_detailed_metrics(researcher1)
        metrics2 = self.calculate_detailed_metrics(researcher2)
        
        comparison = {
            'researcher1': {
                'name': researcher1.name,
                'level': researcher1.level.value,
                'metrics': metrics1.to_dict()
            },
            'researcher2': {
                'name': researcher2.name,
                'level': researcher2.level.value,
                'metrics': metrics2.to_dict()
            },
            'differences': {
                'overall_score': metrics1.overall_score - metrics2.overall_score,
                'h_index_score': metrics1.h_index_score - metrics2.h_index_score,
                'citation_score': metrics1.citation_score - metrics2.citation_score,
                'experience_score': metrics1.experience_score - metrics2.experience_score,
                'institutional_score': metrics1.institutional_score - metrics2.institutional_score,
                'productivity_score': metrics1.productivity_score - metrics2.productivity_score,
                'impact_score': metrics1.impact_score - metrics2.impact_score
            },
            'winner': researcher1.name if metrics1.overall_score > metrics2.overall_score else researcher2.name
        }
        
        logger.info(f"Reputation comparison: {researcher1.name} vs {researcher2.name} - "
                   f"Winner: {comparison['winner']} "
                   f"(Scores: {metrics1.overall_score:.3f} vs {metrics2.overall_score:.3f})")
        
        return comparison
    
    def get_reputation_percentile(self, researcher: EnhancedResearcher, 
                                 peer_group: List[EnhancedResearcher]) -> float:
        """
        Calculate researcher's reputation percentile within a peer group.
        
        Args:
            researcher: The researcher to evaluate
            peer_group: List of peer researchers for comparison
            
        Returns:
            Percentile rank (0.0 to 1.0)
        """
        if not peer_group:
            return 0.5  # Default to median if no peer group
        
        researcher_score = self.calculate_reputation_score(researcher)
        peer_scores = [self.calculate_reputation_score(peer) for peer in peer_group]
        
        # Count how many peers have lower scores
        lower_count = sum(1 for score in peer_scores if score < researcher_score)
        
        # Calculate percentile
        percentile = lower_count / len(peer_scores) if peer_scores else 0.5
        
        logger.debug(f"Reputation percentile for {researcher.name}: {percentile:.2f} "
                    f"(Score: {researcher_score:.3f}, Peer group size: {len(peer_group)})")
        
        return percentile
    
    def get_level_appropriate_benchmarks(self, level: ResearcherLevel) -> Dict[str, float]:
        """
        Get appropriate benchmarks for a given career level.
        
        Args:
            level: The career level to get benchmarks for
            
        Returns:
            Dictionary with benchmark values
        """
        return {
            'h_index': self.H_INDEX_BENCHMARKS.get(level, 10),
            'citations': self.CITATION_BENCHMARKS.get(level, 500),
            'years_experience': self.EXPERIENCE_BENCHMARKS.get(level, 5)
        }
    
    def validate_researcher_metrics(self, researcher: EnhancedResearcher) -> List[str]:
        """
        Validate researcher metrics for consistency and reasonableness.
        
        Args:
            researcher: The researcher to validate
            
        Returns:
            List of validation warnings/issues
        """
        warnings = []
        
        # Check for negative values
        if researcher.h_index < 0:
            warnings.append("H-index cannot be negative")
        
        if researcher.total_citations < 0:
            warnings.append("Total citations cannot be negative")
        
        if researcher.years_active < 0:
            warnings.append("Years active cannot be negative")
        
        # Check for unrealistic h-index vs citations
        if researcher.h_index > researcher.total_citations:
            warnings.append("H-index cannot exceed total citations")
        
        # Check for level-appropriate metrics
        benchmarks = self.get_level_appropriate_benchmarks(researcher.level)
        
        if researcher.h_index > benchmarks['h_index'] * 5:
            warnings.append(f"H-index ({researcher.h_index}) seems very high for {researcher.level.value}")
        
        if researcher.total_citations > benchmarks['citations'] * 10:
            warnings.append(f"Citations ({researcher.total_citations}) seem very high for {researcher.level.value}")
        
        # Check institutional tier
        if not (1 <= researcher.institution_tier <= 3):
            warnings.append(f"Institution tier ({researcher.institution_tier}) must be 1, 2, or 3")
        
        return warnings