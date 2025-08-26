"""
Cognitive Bias Engine Infrastructure

This module provides the core infrastructure for modeling and applying cognitive biases
in the peer review simulation system. It includes the central BiasEngine class for
coordinating bias applications, BiasEffect class for representing individual bias impacts,
and a configurable bias strength system.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import random
import json
from pathlib import Path

from src.core.exceptions import ValidationError, BiasSystemError
from src.core.logging_config import get_logger
from src.data.enhanced_models import EnhancedResearcher, StructuredReview, BiasEffect


logger = get_logger(__name__)


class BiasType(Enum):
    """Types of cognitive biases modeled in the system."""
    CONFIRMATION = "confirmation"
    HALO_EFFECT = "halo_effect"
    ANCHORING = "anchoring"
    AVAILABILITY = "availability"


@dataclass
class BiasConfiguration:
    """Configuration for bias strength and application parameters."""
    bias_type: BiasType
    base_strength: float = 0.3  # 0-1 scale, default strength
    min_strength: float = 0.0   # Minimum possible strength
    max_strength: float = 1.0   # Maximum possible strength
    enabled: bool = True        # Whether this bias is enabled
    
    # Bias-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_strength_values()
    
    def _validate_strength_values(self):
        """Validate that strength values are in valid ranges."""
        for strength_name, strength_value in [
            ("base_strength", self.base_strength),
            ("min_strength", self.min_strength),
            ("max_strength", self.max_strength)
        ]:
            if not (0.0 <= strength_value <= 1.0):
                raise ValidationError(
                    strength_name, strength_value, 
                    "float between 0.0 and 1.0"
                )
        
        if self.min_strength > self.max_strength:
            raise ValidationError(
                "min_strength", self.min_strength,
                f"value <= max_strength ({self.max_strength})"
            )
        
        if not (self.min_strength <= self.base_strength <= self.max_strength):
            raise ValidationError(
                "base_strength", self.base_strength,
                f"value between min_strength ({self.min_strength}) and max_strength ({self.max_strength})"
            )


class BiasModel(ABC):
    """Abstract base class for cognitive bias models."""
    
    def __init__(self, configuration: BiasConfiguration):
        """Initialize bias model with configuration."""
        self.configuration = configuration
        self.bias_type = configuration.bias_type
    
    @abstractmethod
    def calculate_bias_effect(
        self, 
        reviewer: EnhancedResearcher,
        review: StructuredReview,
        context: Dict[str, Any]
    ) -> BiasEffect:
        """
        Calculate the bias effect for a given review context.
        
        Args:
            reviewer: The researcher conducting the review
            review: The review being written
            context: Additional context for bias calculation
            
        Returns:
            BiasEffect representing the impact of this bias
        """
        pass
    
    def is_applicable(
        self, 
        reviewer: EnhancedResearcher,
        context: Dict[str, Any]
    ) -> bool:
        """
        Check if this bias is applicable in the given context.
        
        Args:
            reviewer: The researcher conducting the review
            context: Additional context for applicability check
            
        Returns:
            True if bias should be applied, False otherwise
        """
        return self.configuration.enabled
    
    def get_effective_strength(
        self, 
        reviewer: EnhancedResearcher,
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate the effective bias strength for this reviewer and context.
        
        Args:
            reviewer: The researcher conducting the review
            context: Additional context for strength calculation
            
        Returns:
            Effective bias strength (0-1 scale)
        """
        # Get base strength from configuration
        base_strength = self.configuration.base_strength
        
        # Get reviewer-specific bias susceptibility
        reviewer_susceptibility = reviewer.cognitive_biases.get(
            self.bias_type.value, 0.3
        )
        
        # Combine base strength with reviewer susceptibility
        effective_strength = base_strength * reviewer_susceptibility
        
        # Ensure within configured bounds
        effective_strength = max(
            self.configuration.min_strength,
            min(self.configuration.max_strength, effective_strength)
        )
        
        return effective_strength


class BiasEngine:
    """
    Central engine for coordinating and applying cognitive biases in peer review.
    
    This class manages multiple bias models, applies them to reviews, and tracks
    their cumulative effects on review outcomes.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the bias engine.
        
        Args:
            config_path: Optional path to bias configuration file
        """
        self.bias_models: Dict[BiasType, BiasModel] = {}
        self.configurations: Dict[BiasType, BiasConfiguration] = {}
        self.bias_history: List[Dict[str, Any]] = []
        
        # Load configuration
        if config_path:
            self.load_configuration(config_path)
        else:
            self._initialize_default_configurations()
    
    def _initialize_default_configurations(self):
        """Initialize default bias configurations."""
        default_configs = {
            BiasType.CONFIRMATION: BiasConfiguration(
                bias_type=BiasType.CONFIRMATION,
                base_strength=0.3,
                parameters={
                    'belief_alignment_threshold': 0.7,
                    'max_score_adjustment': 1.5
                }
            ),
            BiasType.HALO_EFFECT: BiasConfiguration(
                bias_type=BiasType.HALO_EFFECT,
                base_strength=0.2,
                parameters={
                    'reputation_threshold': 0.7,
                    'max_score_boost': 2.0,
                    'prestige_factor': 0.5
                }
            ),
            BiasType.ANCHORING: BiasConfiguration(
                bias_type=BiasType.ANCHORING,
                base_strength=0.4,
                parameters={
                    'influence_decay': 0.8,
                    'confidence_weight': 0.3,
                    'max_influence': 1.0
                }
            ),
            BiasType.AVAILABILITY: BiasConfiguration(
                bias_type=BiasType.AVAILABILITY,
                base_strength=0.3,
                parameters={
                    'recency_window_days': 30,
                    'similarity_threshold': 0.6,
                    'max_adjustment': 1.0
                }
            )
        }
        
        self.configurations = default_configs
        logger.info("Initialized default bias configurations")
    
    def register_bias_model(self, bias_model: BiasModel):
        """
        Register a bias model with the engine.
        
        Args:
            bias_model: The bias model to register
        """
        bias_type = bias_model.bias_type
        self.bias_models[bias_type] = bias_model
        logger.info(f"Registered bias model: {bias_type.value}")
    
    def configure_bias_strength(
        self, 
        bias_type: BiasType, 
        strength: float,
        **parameters
    ):
        """
        Configure the strength and parameters for a specific bias type.
        
        Args:
            bias_type: The type of bias to configure
            strength: The bias strength (0-1 scale)
            **parameters: Additional bias-specific parameters
        """
        if bias_type not in self.configurations:
            raise BiasSystemError(f"Unknown bias type: {bias_type}")
        
        # Update configuration
        config = self.configurations[bias_type]
        config.base_strength = strength
        config.parameters.update(parameters)
        
        logger.info(f"Updated configuration for {bias_type.value}: strength={strength}")
    
    def apply_biases(
        self,
        reviewer: EnhancedResearcher,
        review: StructuredReview,
        context: Dict[str, Any]
    ) -> List[BiasEffect]:
        """
        Apply all applicable biases to a review.
        
        Args:
            reviewer: The researcher conducting the review
            review: The review being written
            context: Additional context for bias application
            
        Returns:
            List of BiasEffect objects representing applied biases
        """
        applied_biases = []
        
        try:
            for bias_type, bias_model in self.bias_models.items():
                if bias_model.is_applicable(reviewer, context):
                    bias_effect = bias_model.calculate_bias_effect(
                        reviewer, review, context
                    )
                    
                    if bias_effect.score_adjustment != 0.0:
                        applied_biases.append(bias_effect)
                        logger.debug(
                            f"Applied {bias_type.value} bias: "
                            f"adjustment={bias_effect.score_adjustment:.3f}"
                        )
            
            # Track bias application in history
            self._record_bias_application(reviewer.id, review.review_id, applied_biases)
            
        except Exception as e:
            logger.error(f"Error applying biases: {e}")
            raise BiasSystemError(f"Failed to apply biases: {e}")
        
        return applied_biases
    
    def calculate_bias_adjusted_scores(
        self,
        original_scores: Dict[str, float],
        bias_effects: List[BiasEffect]
    ) -> Dict[str, float]:
        """
        Calculate bias-adjusted scores from original scores and bias effects.
        
        Args:
            original_scores: Original review scores by dimension
            bias_effects: List of bias effects to apply
            
        Returns:
            Dictionary of bias-adjusted scores
        """
        adjusted_scores = original_scores.copy()
        
        # Apply each bias effect
        for bias_effect in bias_effects:
            # Apply bias adjustment to all dimensions (simplified approach)
            # In a more sophisticated implementation, different biases might
            # affect different dimensions differently
            for dimension in adjusted_scores:
                adjusted_scores[dimension] += bias_effect.score_adjustment
                
                # Ensure scores stay within valid range (1-10)
                adjusted_scores[dimension] = max(1.0, min(10.0, adjusted_scores[dimension]))
        
        return adjusted_scores
    
    def get_bias_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about bias applications.
        
        Returns:
            Dictionary containing bias application statistics
        """
        if not self.bias_history:
            return {"total_applications": 0, "bias_breakdown": {}}
        
        # Count applications by bias type
        bias_counts = {}
        total_applications = len(self.bias_history)
        
        for record in self.bias_history:
            for bias_effect in record.get("applied_biases", []):
                bias_type = bias_effect.bias_type
                bias_counts[bias_type] = bias_counts.get(bias_type, 0) + 1
        
        # Calculate average adjustments by bias type
        bias_adjustments = {}
        for record in self.bias_history:
            for bias_effect in record.get("applied_biases", []):
                bias_type = bias_effect.bias_type
                if bias_type not in bias_adjustments:
                    bias_adjustments[bias_type] = []
                bias_adjustments[bias_type].append(bias_effect.score_adjustment)
        
        # Calculate averages
        bias_averages = {}
        for bias_type, adjustments in bias_adjustments.items():
            bias_averages[bias_type] = sum(adjustments) / len(adjustments)
        
        return {
            "total_applications": total_applications,
            "bias_breakdown": bias_counts,
            "average_adjustments": bias_averages,
            "configurations": {
                bt.value: {
                    "base_strength": config.base_strength,
                    "enabled": config.enabled
                }
                for bt, config in self.configurations.items()
            }
        }
    
    def _record_bias_application(
        self,
        reviewer_id: str,
        review_id: str,
        applied_biases: List[BiasEffect]
    ):
        """Record bias application in history."""
        record = {
            "reviewer_id": reviewer_id,
            "review_id": review_id,
            "applied_biases": applied_biases,
            "timestamp": logger.handlers[0].formatter.formatTime(
                logging.LogRecord("", 0, "", 0, "", (), None)
            ) if logger.handlers else "unknown"
        }
        
        self.bias_history.append(record)
        
        # Keep history size manageable
        if len(self.bias_history) > 10000:
            self.bias_history = self.bias_history[-5000:]
    
    def load_configuration(self, config_path: str):
        """
        Load bias configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Configuration file not found: {config_path}")
                self._initialize_default_configurations()
                return
            
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Parse configurations
            for bias_type_str, config_dict in config_data.items():
                try:
                    bias_type = BiasType(bias_type_str)
                    config = BiasConfiguration(
                        bias_type=bias_type,
                        base_strength=config_dict.get("base_strength", 0.3),
                        min_strength=config_dict.get("min_strength", 0.0),
                        max_strength=config_dict.get("max_strength", 1.0),
                        enabled=config_dict.get("enabled", True),
                        parameters=config_dict.get("parameters", {})
                    )
                    self.configurations[bias_type] = config
                except ValueError:
                    logger.warning(f"Unknown bias type in configuration: {bias_type_str}")
            
            logger.info(f"Loaded bias configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading bias configuration: {e}")
            self._initialize_default_configurations()
    
    def save_configuration(self, config_path: str):
        """
        Save current bias configuration to file.
        
        Args:
            config_path: Path to save configuration file
        """
        try:
            config_data = {}
            for bias_type, config in self.configurations.items():
                config_data[bias_type.value] = {
                    "base_strength": config.base_strength,
                    "min_strength": config.min_strength,
                    "max_strength": config.max_strength,
                    "enabled": config.enabled,
                    "parameters": config.parameters
                }
            
            config_file = Path(config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Saved bias configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving bias configuration: {e}")
            raise BiasSystemError(f"Failed to save configuration: {e}")
    
    def reset_bias_history(self):
        """Reset the bias application history."""
        self.bias_history.clear()
        logger.info("Reset bias application history")
    
    def disable_bias(self, bias_type: BiasType):
        """
        Disable a specific bias type.
        
        Args:
            bias_type: The bias type to disable
        """
        if bias_type in self.configurations:
            self.configurations[bias_type].enabled = False
            logger.info(f"Disabled bias: {bias_type.value}")
        else:
            logger.warning(f"Cannot disable unknown bias type: {bias_type}")
    
    def enable_bias(self, bias_type: BiasType):
        """
        Enable a specific bias type.
        
        Args:
            bias_type: The bias type to enable
        """
        if bias_type in self.configurations:
            self.configurations[bias_type].enabled = True
            logger.info(f"Enabled bias: {bias_type.value}")
        else:
            logger.warning(f"Cannot enable unknown bias type: {bias_type}")