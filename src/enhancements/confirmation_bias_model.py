"""
Confirmation Bias Model

This module implements confirmation bias modeling for the peer review simulation.
Confirmation bias occurs when reviewers favor papers that align with their existing
research beliefs and are more critical of papers that challenge their views.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import math

from src.enhancements.bias_engine import BiasModel, BiasType
from src.data.enhanced_models import EnhancedResearcher, StructuredReview, BiasEffect
from src.core.exceptions import BiasSystemError
from src.core.logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class ResearchBelief:
    """Represents a research belief or position held by a researcher."""
    topic: str
    position: str  # The researcher's stance on this topic
    strength: float  # How strongly they hold this belief (0-1)
    keywords: List[str]  # Keywords associated with this belief


class ConfirmationBiasModel(BiasModel):
    """
    Models confirmation bias in peer review.
    
    Confirmation bias occurs when reviewers:
    1. Give higher scores to papers that align with their research beliefs
    2. Give lower scores to papers that contradict their beliefs
    3. Are more critical in their evaluation of contradictory work
    4. Are more lenient in evaluating supportive work
    """
    
    def __init__(self, configuration):
        """Initialize confirmation bias model."""
        super().__init__(configuration)
        
        # Extract parameters from configuration
        params = configuration.parameters
        self.belief_alignment_threshold = params.get('belief_alignment_threshold', 0.7)
        self.max_score_adjustment = params.get('max_score_adjustment', 1.5)
        
        # Research belief database (simplified - in practice this would be more sophisticated)
        self.research_beliefs = self._initialize_research_beliefs()
    
    def _initialize_research_beliefs(self) -> Dict[str, List[ResearchBelief]]:
        """
        Initialize a database of research beliefs by specialty.
        
        In a real implementation, this would be loaded from a comprehensive database
        or learned from researcher publication history.
        """
        beliefs_db = {
            "AI": [
                ResearchBelief(
                    topic="deep_learning_vs_symbolic",
                    position="deep_learning_superior",
                    strength=0.8,
                    keywords=["neural networks", "deep learning", "end-to-end", "representation learning"]
                ),
                ResearchBelief(
                    topic="deep_learning_vs_symbolic",
                    position="symbolic_ai_important",
                    strength=0.7,
                    keywords=["symbolic reasoning", "logic", "knowledge representation", "interpretability"]
                ),
                ResearchBelief(
                    topic="interpretability",
                    position="interpretability_crucial",
                    strength=0.9,
                    keywords=["explainable AI", "interpretability", "transparency", "fairness"]
                ),
                ResearchBelief(
                    topic="interpretability",
                    position="performance_over_interpretability",
                    strength=0.6,
                    keywords=["state-of-the-art", "performance", "accuracy", "benchmarks"]
                )
            ],
            "NLP": [
                ResearchBelief(
                    topic="transformer_architecture",
                    position="transformers_revolutionary",
                    strength=0.9,
                    keywords=["transformer", "attention", "BERT", "GPT", "self-attention"]
                ),
                ResearchBelief(
                    topic="linguistic_features",
                    position="linguistic_features_important",
                    strength=0.7,
                    keywords=["syntax", "semantics", "linguistic structure", "grammar"]
                ),
                ResearchBelief(
                    topic="multilingual_models",
                    position="multilingual_beneficial",
                    strength=0.8,
                    keywords=["multilingual", "cross-lingual", "language transfer", "zero-shot"]
                )
            ],
            "Computer Vision": [
                ResearchBelief(
                    topic="cnn_vs_transformer",
                    position="cnn_still_relevant",
                    strength=0.6,
                    keywords=["CNN", "convolutional", "inductive bias", "spatial structure"]
                ),
                ResearchBelief(
                    topic="cnn_vs_transformer",
                    position="vision_transformers_superior",
                    strength=0.8,
                    keywords=["vision transformer", "ViT", "attention", "patch embedding"]
                ),
                ResearchBelief(
                    topic="data_augmentation",
                    position="augmentation_essential",
                    strength=0.9,
                    keywords=["data augmentation", "regularization", "generalization", "robustness"]
                )
            ],
            "Machine Learning": [
                ResearchBelief(
                    topic="theoretical_vs_empirical",
                    position="theory_important",
                    strength=0.7,
                    keywords=["theoretical analysis", "convergence", "generalization bounds", "PAC learning"]
                ),
                ResearchBelief(
                    topic="theoretical_vs_empirical",
                    position="empirical_results_sufficient",
                    strength=0.6,
                    keywords=["empirical evaluation", "experiments", "benchmarks", "practical performance"]
                ),
                ResearchBelief(
                    topic="optimization",
                    position="adam_overrated",
                    strength=0.5,
                    keywords=["SGD", "momentum", "learning rate scheduling", "optimization theory"]
                )
            ]
        }
        
        return beliefs_db
    
    def calculate_bias_effect(
        self, 
        reviewer: EnhancedResearcher,
        review: StructuredReview,
        context: Dict[str, Any]
    ) -> BiasEffect:
        """
        Calculate confirmation bias effect based on alignment between paper and reviewer beliefs.
        
        Args:
            reviewer: The researcher conducting the review
            review: The review being written
            context: Additional context including paper content
            
        Returns:
            BiasEffect representing the confirmation bias impact
        """
        try:
            # Get paper content from context
            paper_content = context.get('paper_content', {})
            paper_title = paper_content.get('title', '')
            paper_abstract = paper_content.get('abstract', '')
            paper_keywords = paper_content.get('keywords', [])
            
            # Check if we have sufficient paper content
            if not paper_title and not paper_abstract:
                return BiasEffect(
                    bias_type=self.bias_type.value,
                    strength=0.0,
                    score_adjustment=0.0,
                    description="No paper content available for bias calculation"
                )
            
            # Calculate belief alignment
            alignment_score = self._calculate_belief_alignment(
                reviewer, paper_title, paper_abstract, paper_keywords
            )
            
            # Get effective bias strength
            effective_strength = self.get_effective_strength(reviewer, context)
            
            # Calculate score adjustment based on alignment
            score_adjustment = self._calculate_score_adjustment(
                alignment_score, effective_strength
            )
            
            # Create bias effect description
            description = self._generate_bias_description(alignment_score, score_adjustment)
            
            logger.debug(
                f"Confirmation bias for reviewer {reviewer.id}: "
                f"alignment={alignment_score:.3f}, adjustment={score_adjustment:.3f}"
            )
            
            return BiasEffect(
                bias_type=self.bias_type.value,
                strength=effective_strength,
                score_adjustment=score_adjustment,
                description=description
            )
            
        except Exception as e:
            logger.error(f"Error calculating confirmation bias: {e}")
            # Return neutral bias effect on error
            return BiasEffect(
                bias_type=self.bias_type.value,
                strength=0.0,
                score_adjustment=0.0,
                description="Error in confirmation bias calculation"
            )
    
    def _calculate_belief_alignment(
        self,
        reviewer: EnhancedResearcher,
        paper_title: str,
        paper_abstract: str,
        paper_keywords: List[str]
    ) -> float:
        """
        Calculate how well the paper aligns with the reviewer's research beliefs.
        
        Args:
            reviewer: The reviewer
            paper_title: Title of the paper
            paper_abstract: Abstract of the paper
            paper_keywords: Keywords from the paper
            
        Returns:
            Alignment score from -1 (strongly contradicts) to +1 (strongly supports)
        """
        # Get reviewer's research beliefs based on specialty
        reviewer_beliefs = self.research_beliefs.get(reviewer.specialty, [])
        
        if not reviewer_beliefs:
            # No beliefs defined for this specialty, return neutral
            return 0.0
        
        # Combine paper text for analysis
        paper_text = f"{paper_title} {paper_abstract} {' '.join(paper_keywords)}".lower()
        
        # Calculate alignment with each belief
        belief_alignments = []
        
        for belief in reviewer_beliefs:
            # Check if paper contains keywords related to this belief OR contradictory phrases
            keyword_matches = sum(1 for keyword in belief.keywords 
                                if keyword.lower() in paper_text)
            
            # Also check for contradictory phrases
            contradictory_phrases = {
                "deep_learning_superior": ["symbolic", "rule-based", "logic", "interpretable"],
                "symbolic_ai_important": ["end-to-end", "black box", "neural", "deep learning"],
                "interpretability_crucial": ["performance", "accuracy", "state-of-the-art", "black box"],
                "performance_over_interpretability": ["explainable", "interpretable", "transparent"],
                "transformers_revolutionary": ["CNN", "convolutional", "recurrent"],
                "cnn_still_relevant": ["transformer", "attention", "self-attention"],
                "theory_important": ["empirical", "experimental", "practical"],
                "empirical_results_sufficient": ["theoretical", "proof", "bounds"]
            }
            
            contradictory = contradictory_phrases.get(belief.position, [])
            contradiction_matches = sum(1 for phrase in contradictory 
                                      if phrase.lower() in paper_text)
            
            # Consider belief relevant if it has either keyword matches or contradiction matches
            if keyword_matches > 0 or contradiction_matches > 0:
                # Calculate alignment strength based on keyword matches and belief strength
                keyword_ratio = keyword_matches / len(belief.keywords) if keyword_matches > 0 else 0
                contradiction_ratio = contradiction_matches / len(contradictory) if contradictory and contradiction_matches > 0 else 0
                
                # Use the stronger signal
                alignment_strength = max(keyword_ratio, contradiction_ratio) * belief.strength
                
                # Determine if alignment is positive or negative based on belief position
                alignment_direction = self._determine_alignment_direction(
                    belief, paper_text
                )
                
                belief_alignments.append(alignment_strength * alignment_direction)
        
        if not belief_alignments:
            return 0.0  # No relevant beliefs found
        
        # Return average alignment across all relevant beliefs
        return sum(belief_alignments) / len(belief_alignments)
    
    def _determine_alignment_direction(self, belief: ResearchBelief, paper_text: str) -> float:
        """
        Determine if the paper supports (+1) or contradicts (-1) the belief.
        
        This is a simplified implementation. In practice, this would require
        sophisticated NLP analysis to understand the paper's stance.
        """
        # Simple heuristic: if paper contains belief keywords, assume alignment
        # In practice, this would need semantic analysis to determine stance
        
        # Look for contradictory keywords or phrases
        contradictory_phrases = {
            "deep_learning_superior": ["symbolic", "rule-based", "logic", "interpretable"],
            "symbolic_ai_important": ["end-to-end", "black box", "neural", "deep learning"],
            "interpretability_crucial": ["performance", "accuracy", "state-of-the-art", "black box"],
            "performance_over_interpretability": ["explainable", "interpretable", "transparent"],
            "transformers_revolutionary": ["CNN", "convolutional", "recurrent"],
            "cnn_still_relevant": ["transformer", "attention", "self-attention"],
            "theory_important": ["empirical", "experimental", "practical"],
            "empirical_results_sufficient": ["theoretical", "proof", "bounds"]
        }
        
        contradictory = contradictory_phrases.get(belief.position, [])
        contradiction_count = sum(1 for phrase in contradictory 
                                if phrase.lower() in paper_text)
        
        support_count = sum(1 for keyword in belief.keywords 
                          if keyword.lower() in paper_text)
        
        # Give more weight to contradictory evidence
        if contradiction_count > 0:
            return -1.0  # Paper contradicts the belief
        elif support_count > 0:
            return 1.0  # Paper supports the belief
        else:
            return 0.0  # Neutral or unclear
    
    def _calculate_score_adjustment(self, alignment_score: float, effective_strength: float) -> float:
        """
        Calculate the score adjustment based on belief alignment and bias strength.
        
        Args:
            alignment_score: How well paper aligns with beliefs (-1 to +1)
            effective_strength: Effective bias strength (0-1)
            
        Returns:
            Score adjustment (can be positive or negative)
        """
        # Base adjustment proportional to alignment and strength
        base_adjustment = alignment_score * effective_strength * self.max_score_adjustment
        
        # Apply non-linear scaling to make strong alignments/contradictions more impactful
        scaled_adjustment = base_adjustment * (1 + abs(alignment_score) * 0.5)
        
        return scaled_adjustment
    
    def _generate_bias_description(self, alignment_score: float, score_adjustment: float) -> str:
        """Generate a human-readable description of the bias effect."""
        if abs(score_adjustment) < 0.05:
            return "Minimal confirmation bias - paper is neutral to reviewer's beliefs"
        
        if score_adjustment > 0:
            if alignment_score > 0.5:
                return f"Strong positive confirmation bias - paper strongly supports reviewer's beliefs (+{score_adjustment:.2f})"
            else:
                return f"Moderate positive confirmation bias - paper somewhat supports reviewer's beliefs (+{score_adjustment:.2f})"
        else:
            if alignment_score < -0.5:
                return f"Strong negative confirmation bias - paper contradicts reviewer's beliefs ({score_adjustment:.2f})"
            else:
                return f"Moderate negative confirmation bias - paper somewhat contradicts reviewer's beliefs ({score_adjustment:.2f})"
    
    def is_applicable(self, reviewer: EnhancedResearcher, context: Dict[str, Any]) -> bool:
        """
        Check if confirmation bias is applicable for this reviewer and context.
        
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
        
        # Check if reviewer has beliefs defined for their specialty
        reviewer_beliefs = self.research_beliefs.get(reviewer.specialty, [])
        if not reviewer_beliefs:
            return False
        
        return True
    
    def add_research_belief(
        self, 
        specialty: str, 
        topic: str, 
        position: str, 
        strength: float,
        keywords: List[str]
    ):
        """
        Add a new research belief to the database.
        
        Args:
            specialty: Research specialty (e.g., "AI", "NLP")
            topic: Topic of the belief (e.g., "interpretability")
            position: Position on the topic (e.g., "interpretability_crucial")
            strength: Strength of the belief (0-1)
            keywords: Keywords associated with this belief
        """
        if specialty not in self.research_beliefs:
            self.research_beliefs[specialty] = []
        
        belief = ResearchBelief(
            topic=topic,
            position=position,
            strength=strength,
            keywords=keywords
        )
        
        self.research_beliefs[specialty].append(belief)
        logger.info(f"Added research belief for {specialty}: {topic} - {position}")
    
    def get_researcher_beliefs(self, researcher: EnhancedResearcher) -> List[ResearchBelief]:
        """
        Get the research beliefs associated with a researcher's specialty.
        
        Args:
            researcher: The researcher
            
        Returns:
            List of research beliefs for the researcher's specialty
        """
        return self.research_beliefs.get(researcher.specialty, [])
    
    def update_belief_strength(self, specialty: str, topic: str, position: str, new_strength: float):
        """
        Update the strength of a specific belief.
        
        Args:
            specialty: Research specialty
            topic: Topic of the belief
            position: Position on the topic
            new_strength: New strength value (0-1)
        """
        beliefs = self.research_beliefs.get(specialty, [])
        
        for belief in beliefs:
            if belief.topic == topic and belief.position == position:
                belief.strength = new_strength
                logger.info(f"Updated belief strength for {specialty}/{topic}/{position}: {new_strength}")
                return
        
        logger.warning(f"Belief not found for update: {specialty}/{topic}/{position}")