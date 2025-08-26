"""
Academic Hierarchy Management System

This module implements academic hierarchy management with seniority level definitions,
reputation multiplier calculations, and hierarchy-based functionality for the peer review simulation.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from src.data.enhanced_models import EnhancedResearcher, ResearcherLevel
from src.core.exceptions import ValidationError
from src.core.logging_config import get_logger
from src.enhancements.reputation_calculator import ReputationCalculator

logger = get_logger(__name__)


class AcademicHierarchy:
    """
    Manages academic hierarchy with seniority level definitions and reputation calculations.
    
    This class provides functionality to:
    - Define and manage the six academic seniority levels
    - Calculate reputation multipliers based on seniority
    - Manage hierarchy-based privileges and responsibilities
    - Track career progression through academic ranks
    """
    
    # Seniority level definitions with characteristics
    LEVEL_DEFINITIONS = {
        ResearcherLevel.GRADUATE_STUDENT: {
            'description': 'PhD student conducting research under supervision',
            'typical_years_experience': (0, 6),
            'typical_h_index_range': (0, 5),
            'base_reputation_multiplier': 0.3,
            'max_reviews_per_month': 2,
            'review_weight': 0.5,
            'can_be_senior_reviewer': False,
            'can_mentor': False,
            'typical_publication_pressure': 0.8,  # High pressure to publish
        },
        ResearcherLevel.POSTDOC: {
            'description': 'Postdoctoral researcher with recent PhD',
            'typical_years_experience': (0, 4),
            'typical_h_index_range': (2, 12),
            'base_reputation_multiplier': 0.5,
            'max_reviews_per_month': 3,
            'review_weight': 0.7,
            'can_be_senior_reviewer': False,
            'can_mentor': True,
            'typical_publication_pressure': 0.9,  # Very high pressure
        },
        ResearcherLevel.ASSISTANT_PROF: {
            'description': 'Assistant Professor on tenure track',
            'typical_years_experience': (1, 8),
            'typical_h_index_range': (5, 20),
            'base_reputation_multiplier': 1.0,  # Baseline
            'max_reviews_per_month': 4,
            'review_weight': 1.0,
            'can_be_senior_reviewer': True,
            'can_mentor': True,
            'typical_publication_pressure': 0.9,  # Very high pressure for tenure
        },
        ResearcherLevel.ASSOCIATE_PROF: {
            'description': 'Associate Professor with tenure',
            'typical_years_experience': (6, 15),
            'typical_h_index_range': (12, 35),
            'base_reputation_multiplier': 1.3,
            'max_reviews_per_month': 6,
            'review_weight': 1.3,
            'can_be_senior_reviewer': True,
            'can_mentor': True,
            'typical_publication_pressure': 0.6,  # Moderate pressure
        },
        ResearcherLevel.FULL_PROF: {
            'description': 'Full Professor with established reputation',
            'typical_years_experience': (10, 40),
            'typical_h_index_range': (20, 80),
            'base_reputation_multiplier': 1.5,
            'max_reviews_per_month': 8,
            'review_weight': 1.5,
            'can_be_senior_reviewer': True,
            'can_mentor': True,
            'typical_publication_pressure': 0.4,  # Lower pressure, focus on quality
        },
        ResearcherLevel.EMERITUS: {
            'description': 'Emeritus Professor, retired but active',
            'typical_years_experience': (20, 60),
            'typical_h_index_range': (25, 100),
            'base_reputation_multiplier': 1.2,
            'max_reviews_per_month': 3,
            'review_weight': 1.4,  # High respect but limited availability
            'can_be_senior_reviewer': True,
            'can_mentor': True,
            'typical_publication_pressure': 0.1,  # Very low pressure
        }
    }
    
    # Career progression requirements
    PROMOTION_REQUIREMENTS = {
        ResearcherLevel.POSTDOC: {
            'from_level': ResearcherLevel.GRADUATE_STUDENT,
            'min_years_experience': 4,
            'min_publications': 3,
            'min_h_index': 2,
            'requirements': ['PhD completion', 'Dissertation defense']
        },
        ResearcherLevel.ASSISTANT_PROF: {
            'from_level': ResearcherLevel.POSTDOC,
            'min_years_experience': 1,
            'min_publications': 5,
            'min_h_index': 5,
            'requirements': ['Job market success', 'Research independence']
        },
        ResearcherLevel.ASSOCIATE_PROF: {
            'from_level': ResearcherLevel.ASSISTANT_PROF,
            'min_years_experience': 6,
            'min_publications': 15,
            'min_h_index': 12,
            'requirements': ['Tenure achievement', 'External recognition', 'Service record']
        },
        ResearcherLevel.FULL_PROF: {
            'from_level': ResearcherLevel.ASSOCIATE_PROF,
            'min_years_experience': 12,
            'min_publications': 30,
            'min_h_index': 20,
            'requirements': ['International recognition', 'Leadership in field', 'Major grants']
        },
        ResearcherLevel.EMERITUS: {
            'from_level': ResearcherLevel.FULL_PROF,
            'min_years_experience': 20,
            'min_publications': 40,
            'min_h_index': 25,
            'requirements': ['Retirement', 'Distinguished career', 'Continued activity']
        }
    }
    
    def __init__(self):
        """Initialize the academic hierarchy manager."""
        logger.info("Initializing Academic Hierarchy Management System")
        self.reputation_calculator = ReputationCalculator()
    
    def get_level_definition(self, level: ResearcherLevel) -> Dict:
        """
        Get the definition and characteristics of a seniority level.
        
        Args:
            level: The researcher level to get definition for
            
        Returns:
            Dictionary containing level characteristics
        """
        if level not in self.LEVEL_DEFINITIONS:
            raise ValidationError("level", level, "valid ResearcherLevel")
        
        return self.LEVEL_DEFINITIONS[level].copy()
    
    def get_all_levels(self) -> List[ResearcherLevel]:
        """
        Get all available seniority levels in hierarchical order.
        
        Returns:
            List of ResearcherLevel enum values in order from junior to senior
        """
        return [
            ResearcherLevel.GRADUATE_STUDENT,
            ResearcherLevel.POSTDOC,
            ResearcherLevel.ASSISTANT_PROF,
            ResearcherLevel.ASSOCIATE_PROF,
            ResearcherLevel.FULL_PROF,
            ResearcherLevel.EMERITUS
        ]
    
    def calculate_reputation_multiplier(self, researcher: EnhancedResearcher) -> float:
        """
        Calculate reputation multiplier based on seniority level and individual metrics.
        
        This implements the requirement that Full Professors have 1.5x influence
        compared to Assistant Professors baseline, with additional factors for
        individual performance.
        
        Args:
            researcher: The researcher to calculate multiplier for
            
        Returns:
            Reputation multiplier (typically 0.3 to 2.0)
        """
        if researcher.level not in self.LEVEL_DEFINITIONS:
            logger.warning(f"Unknown researcher level: {researcher.level}, using Assistant Prof baseline")
            base_multiplier = 1.0
        else:
            base_multiplier = self.LEVEL_DEFINITIONS[researcher.level]['base_reputation_multiplier']
        
        # Individual performance adjustments
        performance_bonus = self._calculate_performance_bonus(researcher)
        institutional_bonus = self._calculate_institutional_bonus(researcher)
        experience_bonus = self._calculate_experience_bonus(researcher)
        
        # Combine all factors
        total_multiplier = base_multiplier * (1 + performance_bonus + institutional_bonus + experience_bonus)
        
        # Cap the multiplier to reasonable bounds
        total_multiplier = max(0.1, min(2.0, total_multiplier))
        
        logger.debug(f"Calculated reputation multiplier for {researcher.name}: {total_multiplier:.2f} "
                    f"(base: {base_multiplier:.2f}, performance: {performance_bonus:.2f}, "
                    f"institutional: {institutional_bonus:.2f}, experience: {experience_bonus:.2f})")
        
        return total_multiplier
    
    def _calculate_performance_bonus(self, researcher: EnhancedResearcher) -> float:
        """Calculate performance bonus based on h-index and citations relative to level expectations."""
        level_def = self.LEVEL_DEFINITIONS.get(researcher.level, {})
        expected_h_index_range = level_def.get('typical_h_index_range', (0, 10))
        
        # Calculate how researcher performs relative to expectations
        expected_min, expected_max = expected_h_index_range
        expected_mid = (expected_min + expected_max) / 2
        
        if expected_mid > 0:
            h_index_ratio = researcher.h_index / expected_mid
            # Bonus ranges from -0.3 to +0.5 based on performance
            performance_bonus = min(0.5, max(-0.3, (h_index_ratio - 1.0) * 0.3))
        else:
            performance_bonus = 0.0
        
        return performance_bonus
    
    def _calculate_institutional_bonus(self, researcher: EnhancedResearcher) -> float:
        """Calculate institutional bonus based on institution tier."""
        # Institution tier: 1 = top tier, 2 = mid tier, 3 = lower tier
        tier_bonuses = {1: 0.2, 2: 0.0, 3: -0.1}
        return tier_bonuses.get(researcher.institution_tier, 0.0)
    
    def _calculate_experience_bonus(self, researcher: EnhancedResearcher) -> float:
        """Calculate experience bonus based on years active relative to level expectations."""
        level_def = self.LEVEL_DEFINITIONS.get(researcher.level, {})
        expected_years_range = level_def.get('typical_years_experience', (0, 10))
        
        expected_min, expected_max = expected_years_range
        
        # Bonus for being experienced within the level
        if researcher.years_active >= expected_max:
            return 0.1  # Bonus for being very experienced
        elif researcher.years_active >= expected_min:
            return 0.05  # Small bonus for adequate experience
        else:
            return -0.1  # Penalty for being inexperienced for the level
    
    def get_max_reviews_per_month(self, level: ResearcherLevel) -> int:
        """
        Get maximum reviews per month based on seniority level.
        
        Args:
            level: The researcher level
            
        Returns:
            Maximum number of reviews per month
        """
        level_def = self.get_level_definition(level)
        return level_def['max_reviews_per_month']
    
    def get_review_weight(self, level: ResearcherLevel) -> float:
        """
        Get review weight multiplier based on seniority level.
        
        Args:
            level: The researcher level
            
        Returns:
            Review weight multiplier
        """
        level_def = self.get_level_definition(level)
        return level_def['review_weight']
    
    def can_be_senior_reviewer(self, level: ResearcherLevel) -> bool:
        """
        Check if researcher at this level can serve as senior reviewer.
        
        Args:
            level: The researcher level
            
        Returns:
            True if can serve as senior reviewer
        """
        level_def = self.get_level_definition(level)
        return level_def['can_be_senior_reviewer']
    
    def can_mentor(self, level: ResearcherLevel) -> bool:
        """
        Check if researcher at this level can mentor others.
        
        Args:
            level: The researcher level
            
        Returns:
            True if can mentor
        """
        level_def = self.get_level_definition(level)
        return level_def['can_mentor']
    
    def get_typical_publication_pressure(self, level: ResearcherLevel) -> float:
        """
        Get typical publication pressure for this level.
        
        Args:
            level: The researcher level
            
        Returns:
            Publication pressure (0.0 to 1.0)
        """
        level_def = self.get_level_definition(level)
        return level_def['typical_publication_pressure']
    
    def is_promotion_eligible(self, researcher: EnhancedResearcher, target_level: ResearcherLevel) -> Tuple[bool, List[str]]:
        """
        Check if researcher is eligible for promotion to target level.
        
        Args:
            researcher: The researcher to check
            target_level: The level to promote to
            
        Returns:
            Tuple of (is_eligible, list_of_missing_requirements)
        """
        if target_level not in self.PROMOTION_REQUIREMENTS:
            return False, [f"No promotion path defined to {target_level.value}"]
        
        requirements = self.PROMOTION_REQUIREMENTS[target_level]
        missing_requirements = []
        
        # Check if promoting from correct level
        if researcher.level != requirements['from_level']:
            missing_requirements.append(f"Must be {requirements['from_level'].value} to promote to {target_level.value}")
        
        # Check years experience
        if researcher.years_active < requirements['min_years_experience']:
            missing_requirements.append(f"Need {requirements['min_years_experience']} years experience, have {researcher.years_active}")
        
        # Check publications
        if len(researcher.publication_history) < requirements['min_publications']:
            missing_requirements.append(f"Need {requirements['min_publications']} publications, have {len(researcher.publication_history)}")
        
        # Check h-index
        if researcher.h_index < requirements['min_h_index']:
            missing_requirements.append(f"Need h-index of {requirements['min_h_index']}, have {researcher.h_index}")
        
        is_eligible = len(missing_requirements) == 0
        
        logger.info(f"Promotion eligibility check for {researcher.name} to {target_level.value}: "
                   f"{'Eligible' if is_eligible else 'Not eligible'}")
        if missing_requirements:
            logger.info(f"Missing requirements: {missing_requirements}")
        
        return is_eligible, missing_requirements
    
    def get_promotion_requirements(self, target_level: ResearcherLevel) -> Dict:
        """
        Get promotion requirements for a target level.
        
        Args:
            target_level: The level to get requirements for
            
        Returns:
            Dictionary of promotion requirements
        """
        if target_level not in self.PROMOTION_REQUIREMENTS:
            raise ValidationError("target_level", target_level, "level with defined promotion requirements")
        
        return self.PROMOTION_REQUIREMENTS[target_level].copy()
    
    def suggest_career_progression(self, researcher: EnhancedResearcher) -> Dict:
        """
        Suggest next career steps for a researcher.
        
        Args:
            researcher: The researcher to analyze
            
        Returns:
            Dictionary with career progression suggestions
        """
        current_level = researcher.level
        all_levels = self.get_all_levels()
        
        try:
            current_index = all_levels.index(current_level)
            if current_index < len(all_levels) - 1:
                next_level = all_levels[current_index + 1]
                is_eligible, missing_reqs = self.is_promotion_eligible(researcher, next_level)
                
                return {
                    'current_level': current_level.value,
                    'next_level': next_level.value,
                    'is_eligible': is_eligible,
                    'missing_requirements': missing_reqs,
                    'promotion_timeline': self._estimate_promotion_timeline(researcher, next_level),
                    'recommendations': self._generate_career_recommendations(researcher, next_level)
                }
            else:
                return {
                    'current_level': current_level.value,
                    'next_level': None,
                    'is_eligible': False,
                    'missing_requirements': [],
                    'message': 'Already at highest academic level'
                }
        except ValueError:
            logger.error(f"Unknown researcher level: {current_level}")
            return {
                'current_level': current_level.value,
                'error': 'Unknown researcher level'
            }
    
    def _estimate_promotion_timeline(self, researcher: EnhancedResearcher, target_level: ResearcherLevel) -> str:
        """Estimate timeline for promotion based on current progress."""
        is_eligible, missing_reqs = self.is_promotion_eligible(researcher, target_level)
        
        if is_eligible:
            return "Ready for promotion"
        
        requirements = self.PROMOTION_REQUIREMENTS[target_level]
        
        # Estimate years needed based on missing requirements
        years_needed = []
        
        if researcher.years_active < requirements['min_years_experience']:
            years_needed.append(requirements['min_years_experience'] - researcher.years_active)
        
        if len(researcher.publication_history) < requirements['min_publications']:
            pubs_needed = requirements['min_publications'] - len(researcher.publication_history)
            # Assume 2-3 publications per year
            years_needed.append(pubs_needed / 2.5)
        
        if researcher.h_index < requirements['min_h_index']:
            h_index_gap = requirements['min_h_index'] - researcher.h_index
            # Assume h-index grows by 1-2 per year
            years_needed.append(h_index_gap / 1.5)
        
        if years_needed:
            estimated_years = max(years_needed)
            return f"Approximately {estimated_years:.1f} years"
        else:
            return "Soon (administrative requirements only)"
    
    def _generate_career_recommendations(self, researcher: EnhancedResearcher, target_level: ResearcherLevel) -> List[str]:
        """Generate specific career recommendations for promotion."""
        is_eligible, missing_reqs = self.is_promotion_eligible(researcher, target_level)
        
        if is_eligible:
            return ["Apply for promotion", "Prepare promotion dossier", "Seek letters of recommendation"]
        
        recommendations = []
        requirements = self.PROMOTION_REQUIREMENTS[target_level]
        
        if researcher.years_active < requirements['min_years_experience']:
            recommendations.append("Continue building experience and track record")
        
        if len(researcher.publication_history) < requirements['min_publications']:
            pubs_needed = requirements['min_publications'] - len(researcher.publication_history)
            recommendations.append(f"Publish {pubs_needed} more papers in high-quality venues")
        
        if researcher.h_index < requirements['min_h_index']:
            recommendations.append("Focus on impactful research that will be cited")
            recommendations.append("Collaborate with established researchers")
        
        # Level-specific recommendations
        if target_level == ResearcherLevel.ASSOCIATE_PROF:
            recommendations.extend([
                "Build external visibility through conference presentations",
                "Take on service roles in the community",
                "Develop independent research program"
            ])
        elif target_level == ResearcherLevel.FULL_PROF:
            recommendations.extend([
                "Seek leadership roles in major conferences",
                "Apply for significant research grants",
                "Mentor junior researchers",
                "Build international collaborations"
            ])
        
        return recommendations
    
    def calculate_comprehensive_reputation_score(self, researcher: EnhancedResearcher) -> float:
        """
        Calculate comprehensive reputation score using the ReputationCalculator.
        
        This method provides a more sophisticated reputation calculation than the
        basic multiplier approach, incorporating h-index, citations, years active,
        and institutional tier with proper normalization and weighting.
        
        Args:
            researcher: The researcher to calculate reputation for
            
        Returns:
            Comprehensive reputation score (0.0 to 1.0)
        """
        return self.reputation_calculator.calculate_reputation_score(researcher)
    
    def get_detailed_reputation_metrics(self, researcher: EnhancedResearcher) -> Dict:
        """
        Get detailed breakdown of reputation metrics for a researcher.
        
        Args:
            researcher: The researcher to analyze
            
        Returns:
            Dictionary with detailed reputation metrics
        """
        metrics = self.reputation_calculator.calculate_detailed_metrics(researcher)
        return metrics.to_dict()
    
    def get_institutional_influence_multiplier(self, researcher: EnhancedResearcher) -> float:
        """
        Get institutional influence multiplier for review weight calculations.
        
        Args:
            researcher: The researcher to get multiplier for
            
        Returns:
            Influence multiplier based on institutional tier
        """
        return self.reputation_calculator.get_institutional_influence_multiplier(researcher)
    
    def compare_researcher_reputations(self, researcher1: EnhancedResearcher, 
                                     researcher2: EnhancedResearcher) -> Dict:
        """
        Compare reputation metrics between two researchers.
        
        Args:
            researcher1: First researcher to compare
            researcher2: Second researcher to compare
            
        Returns:
            Dictionary with detailed comparison results
        """
        return self.reputation_calculator.compare_researchers(researcher1, researcher2)
    
    def get_hierarchy_statistics(self, researchers: List[EnhancedResearcher]) -> Dict:
        """
        Get statistics about the academic hierarchy distribution.
        
        Args:
            researchers: List of researchers to analyze
            
        Returns:
            Dictionary with hierarchy statistics
        """
        level_counts = {}
        level_metrics = {}
        
        for level in self.get_all_levels():
            level_researchers = [r for r in researchers if r.level == level]
            level_counts[level.value] = len(level_researchers)
            
            if level_researchers:
                avg_h_index = sum(r.h_index for r in level_researchers) / len(level_researchers)
                avg_years = sum(r.years_active for r in level_researchers) / len(level_researchers)
                avg_reputation = sum(r.reputation_score for r in level_researchers) / len(level_researchers)
                
                level_metrics[level.value] = {
                    'count': len(level_researchers),
                    'avg_h_index': avg_h_index,
                    'avg_years_active': avg_years,
                    'avg_reputation_score': avg_reputation,
                    'avg_reputation_multiplier': sum(self.calculate_reputation_multiplier(r) for r in level_researchers) / len(level_researchers)
                }
            else:
                level_metrics[level.value] = {
                    'count': 0,
                    'avg_h_index': 0,
                    'avg_years_active': 0,
                    'avg_reputation_score': 0,
                    'avg_reputation_multiplier': 0
                }
        
        return {
            'total_researchers': len(researchers),
            'level_distribution': level_counts,
            'level_metrics': level_metrics,
            'hierarchy_diversity': len([count for count in level_counts.values() if count > 0])
        }