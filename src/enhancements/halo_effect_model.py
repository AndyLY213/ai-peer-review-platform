"""
Halo Effect Bias Model

This module implements halo effect bias modeling for the peer review simulation.
Halo effect occurs when reviewers are influenced by the reputation or prestige of
authors, leading to higher scores for papers from well-known researchers or
prestigious institutions.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import math

from src.enhancements.bias_engine import BiasModel, BiasType
from src.data.enhanced_models import EnhancedResearcher, StructuredReview, BiasEffect
from src.core.exceptions import BiasSystemError
from src.core.logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class AuthorPrestige:
    """Represents the prestige metrics of an author."""
    author_id: str
    h_index: int = 0
    total_citations: int = 0
    institution_tier: int = 3  # 1-3, where 1 is most prestigious
    years_active: int = 0
    notable_awards: List[str] = None
    prestige_score: float = 0.0  # Calculated overall prestige
    
    def __post_init__(self):
        """Calculate prestige score after initialization."""
        if self.notable_awards is None:
            self.notable_awards = []
        self._calculate_prestige_score()
    
    def _calculate_prestige_score(self):
        """Calculate overall prestige score from individual metrics."""
        # H-index contribution (normalized to 0-1)
        h_index_score = min(1.0, self.h_index / 50.0)
        
        # Citations contribution (normalized to 0-1)
        citation_score = min(1.0, self.total_citations / 5000.0)
        
        # Institution tier contribution (inverted since 1 is best)
        institution_score = (4 - self.institution_tier) / 3.0
        
        # Experience contribution
        experience_score = min(1.0, self.years_active / 25.0)
        
        # Awards bonus
        awards_bonus = min(0.3, len(self.notable_awards) * 0.1)
        
        # Weighted combination
        base_score = (
            h_index_score * 0.3 +
            citation_score * 0.25 +
            institution_score * 0.2 +
            experience_score * 0.15 +
            awards_bonus * 0.1
        )
        
        # Add awards bonus but cap at reasonable maximum
        self.prestige_score = min(1.0, base_score + awards_bonus)


class HaloEffectModel(BiasModel):
    """
    Models halo effect bias in peer review.
    
    Halo effect occurs when:
    1. Reviewers give higher scores to papers from prestigious authors
    2. Institutional prestige influences review scores
    3. Past reputation affects current paper evaluation
    4. Well-known authors receive more lenient reviews
    """
    
    def __init__(self, configuration):
        """Initialize halo effect bias model."""
        super().__init__(configuration)
        
        # Extract parameters from configuration
        params = configuration.parameters
        self.reputation_threshold = params.get('reputation_threshold', 0.7)
        self.max_score_boost = params.get('max_score_boost', 2.0)
        self.prestige_factor = params.get('prestige_factor', 0.5)
        
        # Author prestige database (simplified - would be loaded from external source)
        self.author_prestige_db: Dict[str, AuthorPrestige] = {}
        self._initialize_sample_prestige_data()
    
    def _initialize_sample_prestige_data(self):
        """Initialize sample prestige data for testing purposes."""
        # In a real implementation, this would be loaded from a comprehensive database
        sample_authors = [
            AuthorPrestige(
                author_id="prestigious_author_1",
                h_index=45,
                total_citations=8000,
                institution_tier=1,
                years_active=20,
                notable_awards=["Turing Award", "Best Paper Award"]
            ),
            AuthorPrestige(
                author_id="mid_tier_author_1",
                h_index=25,
                total_citations=2000,
                institution_tier=2,
                years_active=12,
                notable_awards=["Best Paper Award"]
            ),
            AuthorPrestige(
                author_id="junior_author_1",
                h_index=8,
                total_citations=200,
                institution_tier=3,
                years_active=3,
                notable_awards=[]
            )
        ]
        
        for author in sample_authors:
            self.author_prestige_db[author.author_id] = author
    
    def calculate_bias_effect(
        self, 
        reviewer: EnhancedResearcher,
        review: StructuredReview,
        context: Dict[str, Any]
    ) -> BiasEffect:
        """
        Calculate halo effect bias based on author prestige and reviewer susceptibility.
        
        Args:
            reviewer: The researcher conducting the review
            review: The review being written
            context: Additional context including paper authors
            
        Returns:
            BiasEffect representing the halo effect impact
        """
        try:
            # Get paper authors from context
            paper_authors = context.get('paper_authors', [])
            if not paper_authors:
                return BiasEffect(
                    bias_type=self.bias_type.value,
                    strength=0.0,
                    score_adjustment=0.0,
                    description="No author information available for halo effect"
                )
            
            # Calculate maximum prestige among authors
            max_author_prestige = self._calculate_max_author_prestige(paper_authors)
            
            # Get effective bias strength
            effective_strength = self.get_effective_strength(reviewer, context)
            
            # Calculate reviewer's susceptibility to prestige
            reviewer_prestige_bias = self._calculate_reviewer_prestige_bias(reviewer)
            
            # Calculate score adjustment based on author prestige
            score_adjustment = self._calculate_score_adjustment(
                max_author_prestige, effective_strength, reviewer_prestige_bias
            )
            
            # Create bias effect description
            description = self._generate_bias_description(
                max_author_prestige, score_adjustment, paper_authors
            )
            
            logger.debug(
                f"Halo effect for reviewer {reviewer.id}: "
                f"max_prestige={max_author_prestige:.3f}, adjustment={score_adjustment:.3f}"
            )
            
            return BiasEffect(
                bias_type=self.bias_type.value,
                strength=effective_strength,
                score_adjustment=score_adjustment,
                description=description
            )
            
        except Exception as e:
            logger.error(f"Error calculating halo effect bias: {e}")
            # Return neutral bias effect on error
            return BiasEffect(
                bias_type=self.bias_type.value,
                strength=0.0,
                score_adjustment=0.0,
                description="Error in halo effect calculation"
            )
    
    def _calculate_max_author_prestige(self, paper_authors: List[str]) -> float:
        """
        Calculate the maximum prestige score among paper authors.
        
        Args:
            paper_authors: List of author IDs
            
        Returns:
            Maximum prestige score (0-1 scale)
        """
        if not paper_authors:
            return 0.0
        
        max_prestige = 0.0
        
        for author_id in paper_authors:
            # Get author prestige from database
            author_prestige = self.author_prestige_db.get(author_id)
            
            if author_prestige:
                max_prestige = max(max_prestige, author_prestige.prestige_score)
            else:
                # For unknown authors, estimate prestige from author_id if it contains info
                # This is a fallback for testing - in practice would query external DB
                estimated_prestige = self._estimate_author_prestige(author_id)
                max_prestige = max(max_prestige, estimated_prestige)
        
        return max_prestige
    
    def _estimate_author_prestige(self, author_id: str) -> float:
        """
        Estimate author prestige for unknown authors.
        
        This is a simplified fallback method. In practice, this would query
        external databases or use more sophisticated estimation methods.
        """
        # Simple heuristic based on author_id patterns
        if "senior" in author_id.lower() or "prof" in author_id.lower():
            return 0.7
        elif "postdoc" in author_id.lower() or "junior" in author_id.lower():
            return 0.3
        elif "student" in author_id.lower() or "phd" in author_id.lower():
            return 0.1
        else:
            # Default to moderate prestige for unknown authors
            return 0.4
    
    def _calculate_reviewer_prestige_bias(self, reviewer: EnhancedResearcher) -> float:
        """
        Calculate how susceptible the reviewer is to prestige bias.
        
        Args:
            reviewer: The reviewer
            
        Returns:
            Prestige bias susceptibility (0-1 scale)
        """
        # Base susceptibility from reviewer's cognitive biases
        base_susceptibility = reviewer.cognitive_biases.get('halo_effect', 0.2)
        
        # Adjust based on reviewer's own prestige (less prestigious reviewers more susceptible)
        reviewer_prestige = reviewer.reputation_score
        prestige_adjustment = (1.0 - reviewer_prestige) * 0.3
        
        # Adjust based on career stage (junior researchers more susceptible)
        career_stage_adjustments = {
            "Early Career": 0.2,
            "Mid Career": 0.0,
            "Senior Career": -0.1,
            "Emeritus Career": -0.2
        }
        
        career_adjustment = career_stage_adjustments.get(reviewer.career_stage.value, 0.0)
        
        # Combine factors
        total_susceptibility = base_susceptibility + prestige_adjustment + career_adjustment
        
        # Debug logging
        logger.debug(
            f"Reviewer {reviewer.id} prestige bias calculation: "
            f"base={base_susceptibility}, prestige_adj={prestige_adjustment}, "
            f"career_adj={career_adjustment}, total={total_susceptibility}"
        )
        
        # Ensure within bounds
        return max(0.0, min(1.0, total_susceptibility))
    
    def _calculate_score_adjustment(
        self, 
        author_prestige: float, 
        effective_strength: float,
        reviewer_prestige_bias: float
    ) -> float:
        """
        Calculate the score adjustment based on author prestige and reviewer bias.
        
        Args:
            author_prestige: Maximum author prestige score (0-1)
            effective_strength: Effective bias strength (0-1)
            reviewer_prestige_bias: Reviewer's susceptibility to prestige bias (0-1)
            
        Returns:
            Score adjustment (0 to max_score_boost)
        """
        # Only apply positive bias if author prestige exceeds threshold
        if author_prestige < self.reputation_threshold:
            return 0.0
        
        # Calculate base adjustment
        prestige_factor = (author_prestige - self.reputation_threshold) / (1.0 - self.reputation_threshold)
        base_adjustment = prestige_factor * self.max_score_boost
        
        # Apply bias strength and reviewer susceptibility
        final_adjustment = base_adjustment * effective_strength * reviewer_prestige_bias
        
        # Apply non-linear scaling for very high prestige
        if author_prestige > 0.9:
            final_adjustment *= 1.2  # 20% bonus for extremely prestigious authors
        
        return final_adjustment
    
    def _generate_bias_description(
        self, 
        author_prestige: float, 
        score_adjustment: float,
        paper_authors: List[str]
    ) -> str:
        """Generate a human-readable description of the halo effect."""
        if score_adjustment < 0.05:
            return "Minimal halo effect - author prestige below threshold or reviewer not susceptible"
        
        prestige_level = "moderate"
        if author_prestige > 0.95:
            prestige_level = "extremely high"
        elif author_prestige > 0.85:
            prestige_level = "very high"
        elif author_prestige > 0.7:
            prestige_level = "high"
        
        author_count = len(paper_authors)
        author_text = f"{author_count} author{'s' if author_count > 1 else ''}"
        
        return (f"Halo effect bias - {prestige_level} prestige {author_text} "
                f"boosted review score (+{score_adjustment:.2f})")
    
    def is_applicable(self, reviewer: EnhancedResearcher, context: Dict[str, Any]) -> bool:
        """
        Check if halo effect bias is applicable for this reviewer and context.
        
        Args:
            reviewer: The researcher conducting the review
            context: Additional context for applicability check
            
        Returns:
            True if bias should be applied, False otherwise
        """
        # Check if bias is enabled
        if not super().is_applicable(reviewer, context):
            return False
        
        # Check if we have author information
        paper_authors = context.get('paper_authors', [])
        if not paper_authors:
            return False
        
        # Check if reviewer has any susceptibility to halo effect
        halo_susceptibility = reviewer.cognitive_biases.get('halo_effect', 0.0)
        if halo_susceptibility <= 0.0:
            return False
        
        return True
    
    def add_author_prestige(
        self,
        author_id: str,
        h_index: int,
        total_citations: int,
        institution_tier: int,
        years_active: int,
        notable_awards: List[str] = None
    ):
        """
        Add or update author prestige information.
        
        Args:
            author_id: Unique identifier for the author
            h_index: Author's h-index
            total_citations: Total citation count
            institution_tier: Institution tier (1-3, 1 is most prestigious)
            years_active: Years of active research
            notable_awards: List of notable awards
        """
        if notable_awards is None:
            notable_awards = []
        
        author_prestige = AuthorPrestige(
            author_id=author_id,
            h_index=h_index,
            total_citations=total_citations,
            institution_tier=institution_tier,
            years_active=years_active,
            notable_awards=notable_awards
        )
        
        self.author_prestige_db[author_id] = author_prestige
        logger.info(f"Added/updated prestige for author {author_id}: score={author_prestige.prestige_score:.3f}")
    
    def get_author_prestige(self, author_id: str) -> Optional[AuthorPrestige]:
        """
        Get prestige information for an author.
        
        Args:
            author_id: Author identifier
            
        Returns:
            AuthorPrestige object if found, None otherwise
        """
        return self.author_prestige_db.get(author_id)
    
    def update_author_prestige_score(self, author_id: str, new_score: float):
        """
        Manually update an author's prestige score.
        
        Args:
            author_id: Author identifier
            new_score: New prestige score (0-1)
        """
        if author_id in self.author_prestige_db:
            self.author_prestige_db[author_id].prestige_score = max(0.0, min(1.0, new_score))
            logger.info(f"Updated prestige score for {author_id}: {new_score:.3f}")
        else:
            logger.warning(f"Author {author_id} not found in prestige database")
    
    def get_prestige_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the prestige database.
        
        Returns:
            Dictionary containing prestige statistics
        """
        if not self.author_prestige_db:
            return {"total_authors": 0, "average_prestige": 0.0}
        
        prestige_scores = [author.prestige_score for author in self.author_prestige_db.values()]
        
        return {
            "total_authors": len(self.author_prestige_db),
            "average_prestige": sum(prestige_scores) / len(prestige_scores),
            "max_prestige": max(prestige_scores),
            "min_prestige": min(prestige_scores),
            "high_prestige_authors": sum(1 for score in prestige_scores if score > 0.8),
            "low_prestige_authors": sum(1 for score in prestige_scores if score < 0.3)
        }
    
    def simulate_author_network_effect(self, paper_authors: List[str]) -> float:
        """
        Simulate network effects where co-authorship with prestigious authors
        can boost the perceived prestige of other authors.
        
        Args:
            paper_authors: List of author IDs on the paper
            
        Returns:
            Network effect multiplier (1.0 = no effect, >1.0 = positive effect)
        """
        if len(paper_authors) <= 1:
            return 1.0  # No network effect for single authors
        
        # Calculate average prestige of all authors
        total_prestige = 0.0
        known_authors = 0
        
        for author_id in paper_authors:
            author_prestige = self.author_prestige_db.get(author_id)
            if author_prestige:
                total_prestige += author_prestige.prestige_score
                known_authors += 1
            else:
                # Use estimated prestige for unknown authors
                total_prestige += self._estimate_author_prestige(author_id)
                known_authors += 1
        
        if known_authors == 0:
            return 1.0
        
        average_prestige = total_prestige / known_authors
        
        # Network effect is stronger when there are more authors and higher average prestige
        author_count_factor = min(1.2, 1.0 + (len(paper_authors) - 1) * 0.05)
        prestige_factor = 1.0 + (average_prestige * 0.1)
        
        return author_count_factor * prestige_factor