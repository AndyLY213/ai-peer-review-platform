"""
Unit tests for the Bias Engine Infrastructure.

Tests the BiasEngine class, BiasConfiguration, BiasModel abstract base class,
and bias strength configuration system.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

from src.enhancements.bias_engine import (
    BiasEngine, BiasConfiguration, BiasModel, BiasType
)
from src.data.enhanced_models import (
    EnhancedResearcher, StructuredReview, BiasEffect, 
    ResearcherLevel, ReviewDecision, EnhancedReviewCriteria
)
from src.core.exceptions import ValidationError, BiasSystemError


class TestBiasConfiguration:
    """Test cases for BiasConfiguration class."""
    
    def test_valid_configuration_creation(self):
        """Test creating a valid bias configuration."""
        config = BiasConfiguration(
            bias_type=BiasType.CONFIRMATION,
            base_strength=0.5,
            min_strength=0.1,
            max_strength=0.9,
            enabled=True,
            parameters={"test_param": "test_value"}
        )
        
        assert config.bias_type == BiasType.CONFIRMATION
        assert config.base_strength == 0.5
        assert config.min_strength == 0.1
        assert config.max_strength == 0.9
        assert config.enabled is True
        assert config.parameters["test_param"] == "test_value"
    
    def test_default_configuration_values(self):
        """Test default configuration values."""
        config = BiasConfiguration(bias_type=BiasType.HALO_EFFECT)
        
        assert config.base_strength == 0.3
        assert config.min_strength == 0.0
        assert config.max_strength == 1.0
        assert config.enabled is True
        assert config.parameters == {}
    
    def test_invalid_strength_values(self):
        """Test validation of invalid strength values."""
        # Base strength out of range
        with pytest.raises(ValidationError):
            BiasConfiguration(
                bias_type=BiasType.CONFIRMATION,
                base_strength=1.5
            )
        
        with pytest.raises(ValidationError):
            BiasConfiguration(
                bias_type=BiasType.CONFIRMATION,
                base_strength=-0.1
            )
        
        # Min strength greater than max strength
        with pytest.raises(ValidationError):
            BiasConfiguration(
                bias_type=BiasType.CONFIRMATION,
                min_strength=0.8,
                max_strength=0.6
            )
        
        # Base strength outside min/max range
        with pytest.raises(ValidationError):
            BiasConfiguration(
                bias_type=BiasType.CONFIRMATION,
                base_strength=0.9,
                min_strength=0.1,
                max_strength=0.5
            )


class MockBiasModel(BiasModel):
    """Mock bias model for testing."""
    
    def calculate_bias_effect(self, reviewer, review, context):
        """Mock implementation that returns a fixed bias effect."""
        return BiasEffect(
            bias_type=self.bias_type.value,
            strength=0.5,
            score_adjustment=0.3,
            description="Mock bias effect"
        )


class TestBiasModel:
    """Test cases for BiasModel abstract base class."""
    
    def test_bias_model_initialization(self):
        """Test bias model initialization."""
        config = BiasConfiguration(bias_type=BiasType.CONFIRMATION)
        model = MockBiasModel(config)
        
        assert model.configuration == config
        assert model.bias_type == BiasType.CONFIRMATION
    
    def test_is_applicable_default(self):
        """Test default is_applicable implementation."""
        config = BiasConfiguration(bias_type=BiasType.CONFIRMATION, enabled=True)
        model = MockBiasModel(config)
        
        researcher = EnhancedResearcher(
            id="test_researcher",
            name="Test Researcher",
            specialty="AI"
        )
        
        assert model.is_applicable(researcher, {}) is True
        
        # Test with disabled bias
        config.enabled = False
        assert model.is_applicable(researcher, {}) is False
    
    def test_get_effective_strength(self):
        """Test effective strength calculation."""
        config = BiasConfiguration(
            bias_type=BiasType.CONFIRMATION,
            base_strength=0.6,
            min_strength=0.1,
            max_strength=0.9
        )
        model = MockBiasModel(config)
        
        researcher = EnhancedResearcher(
            id="test_researcher",
            name="Test Researcher",
            specialty="AI",
            cognitive_biases={"confirmation": 0.8}
        )
        
        # Base strength (0.6) * reviewer susceptibility (0.8) = 0.48
        effective_strength = model.get_effective_strength(researcher, {})
        assert effective_strength == 0.48
    
    def test_effective_strength_bounds(self):
        """Test that effective strength respects min/max bounds."""
        config = BiasConfiguration(
            bias_type=BiasType.CONFIRMATION,
            base_strength=0.5,  # Valid base strength within min/max range
            min_strength=0.3,
            max_strength=0.7
        )
        model = MockBiasModel(config)
        
        # Test upper bound - high susceptibility should be capped
        researcher_high = EnhancedResearcher(
            id="test_researcher",
            name="Test Researcher",
            specialty="AI",
            cognitive_biases={"confirmation": 2.0}  # Very high susceptibility
        )
        
        effective_strength = model.get_effective_strength(researcher_high, {})
        assert effective_strength == 0.7  # Capped at max_strength
        
        # Test lower bound - low susceptibility should be raised to minimum
        researcher_low = EnhancedResearcher(
            id="test_researcher",
            name="Test Researcher",
            specialty="AI",
            cognitive_biases={"confirmation": 0.1}  # Very low susceptibility
        )
        
        effective_strength = model.get_effective_strength(researcher_low, {})
        assert effective_strength == 0.3  # Raised to min_strength


class TestBiasEngine:
    """Test cases for BiasEngine class."""
    
    def test_bias_engine_initialization(self):
        """Test bias engine initialization with default configuration."""
        engine = BiasEngine()
        
        # Check that default configurations are loaded
        assert len(engine.configurations) == 4
        assert BiasType.CONFIRMATION in engine.configurations
        assert BiasType.HALO_EFFECT in engine.configurations
        assert BiasType.ANCHORING in engine.configurations
        assert BiasType.AVAILABILITY in engine.configurations
        
        # Check default values
        confirmation_config = engine.configurations[BiasType.CONFIRMATION]
        assert confirmation_config.base_strength == 0.3
        assert confirmation_config.enabled is True
    
    def test_register_bias_model(self):
        """Test registering bias models."""
        engine = BiasEngine()
        config = BiasConfiguration(bias_type=BiasType.CONFIRMATION)
        model = MockBiasModel(config)
        
        engine.register_bias_model(model)
        
        assert BiasType.CONFIRMATION in engine.bias_models
        assert engine.bias_models[BiasType.CONFIRMATION] == model
    
    def test_configure_bias_strength(self):
        """Test configuring bias strength."""
        engine = BiasEngine()
        
        engine.configure_bias_strength(
            BiasType.CONFIRMATION, 
            0.7,
            test_param="test_value"
        )
        
        config = engine.configurations[BiasType.CONFIRMATION]
        assert config.base_strength == 0.7
        assert config.parameters["test_param"] == "test_value"
    
    def test_configure_unknown_bias_type(self):
        """Test configuring unknown bias type raises error."""
        engine = BiasEngine()
        
        # Remove a bias type to simulate unknown type
        del engine.configurations[BiasType.CONFIRMATION]
        
        with pytest.raises(BiasSystemError):
            engine.configure_bias_strength(BiasType.CONFIRMATION, 0.5)
    
    def test_apply_biases(self):
        """Test applying biases to a review."""
        engine = BiasEngine()
        
        # Register mock bias model
        config = BiasConfiguration(bias_type=BiasType.CONFIRMATION)
        model = MockBiasModel(config)
        engine.register_bias_model(model)
        
        # Create test data
        reviewer = EnhancedResearcher(
            id="test_reviewer",
            name="Test Reviewer",
            specialty="AI"
        )
        
        review = StructuredReview(
            reviewer_id="test_reviewer",
            paper_id="test_paper",
            venue_id="test_venue"
        )
        
        context = {"test_context": "value"}
        
        # Apply biases
        applied_biases = engine.apply_biases(reviewer, review, context)
        
        assert len(applied_biases) == 1
        assert applied_biases[0].bias_type == "confirmation"
        assert applied_biases[0].score_adjustment == 0.3
        
        # Check that bias application was recorded
        assert len(engine.bias_history) == 1
        assert engine.bias_history[0]["reviewer_id"] == "test_reviewer"
        assert engine.bias_history[0]["review_id"] == review.review_id
    
    def test_calculate_bias_adjusted_scores(self):
        """Test calculating bias-adjusted scores."""
        engine = BiasEngine()
        
        original_scores = {
            "novelty": 6.0,
            "technical_quality": 7.0,
            "clarity": 5.0
        }
        
        bias_effects = [
            BiasEffect(
                bias_type="confirmation",
                strength=0.5,
                score_adjustment=0.5
            ),
            BiasEffect(
                bias_type="halo_effect",
                strength=0.3,
                score_adjustment=-0.2
            )
        ]
        
        adjusted_scores = engine.calculate_bias_adjusted_scores(
            original_scores, bias_effects
        )
        
        # Each score should be adjusted by +0.5 - 0.2 = +0.3
        assert adjusted_scores["novelty"] == 6.3
        assert adjusted_scores["technical_quality"] == 7.3
        assert adjusted_scores["clarity"] == 5.3
    
    def test_bias_adjusted_scores_bounds(self):
        """Test that bias-adjusted scores stay within bounds."""
        engine = BiasEngine()
        
        original_scores = {"novelty": 9.5, "clarity": 1.2}
        
        bias_effects = [
            BiasEffect(
                bias_type="confirmation",
                strength=0.8,
                score_adjustment=1.0  # Large positive adjustment
            )
        ]
        
        adjusted_scores = engine.calculate_bias_adjusted_scores(
            original_scores, bias_effects
        )
        
        # Scores should be capped at 10.0 and floored at 1.0
        assert adjusted_scores["novelty"] == 10.0
        assert adjusted_scores["clarity"] == 2.2
    
    def test_get_bias_statistics_empty(self):
        """Test getting bias statistics with no history."""
        engine = BiasEngine()
        
        stats = engine.get_bias_statistics()
        
        assert stats["total_applications"] == 0
        assert stats["bias_breakdown"] == {}
    
    def test_get_bias_statistics_with_data(self):
        """Test getting bias statistics with application history."""
        engine = BiasEngine()
        
        # Add mock bias history
        bias_effects = [
            BiasEffect("confirmation", 0.5, 0.3),
            BiasEffect("halo_effect", 0.3, 0.1)
        ]
        
        engine.bias_history = [
            {
                "reviewer_id": "reviewer1",
                "review_id": "review1",
                "applied_biases": bias_effects
            }
        ]
        
        stats = engine.get_bias_statistics()
        
        assert stats["total_applications"] == 1
        assert stats["bias_breakdown"]["confirmation"] == 1
        assert stats["bias_breakdown"]["halo_effect"] == 1
        assert stats["average_adjustments"]["confirmation"] == 0.3
        assert stats["average_adjustments"]["halo_effect"] == 0.1
    
    def test_disable_enable_bias(self):
        """Test disabling and enabling bias types."""
        engine = BiasEngine()
        
        # Initially enabled
        assert engine.configurations[BiasType.CONFIRMATION].enabled is True
        
        # Disable
        engine.disable_bias(BiasType.CONFIRMATION)
        assert engine.configurations[BiasType.CONFIRMATION].enabled is False
        
        # Enable
        engine.enable_bias(BiasType.CONFIRMATION)
        assert engine.configurations[BiasType.CONFIRMATION].enabled is True
    
    def test_reset_bias_history(self):
        """Test resetting bias history."""
        engine = BiasEngine()
        
        # Add some history
        engine.bias_history = [{"test": "data"}]
        assert len(engine.bias_history) == 1
        
        # Reset
        engine.reset_bias_history()
        assert len(engine.bias_history) == 0
    
    def test_load_configuration_from_file(self):
        """Test loading configuration from file."""
        config_data = {
            "confirmation": {
                "base_strength": 0.8,
                "min_strength": 0.2,
                "max_strength": 0.9,
                "enabled": False,
                "parameters": {"test_param": "test_value"}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            engine = BiasEngine(config_path)
            
            config = engine.configurations[BiasType.CONFIRMATION]
            assert config.base_strength == 0.8
            assert config.min_strength == 0.2
            assert config.max_strength == 0.9
            assert config.enabled is False
            assert config.parameters["test_param"] == "test_value"
        
        finally:
            Path(config_path).unlink()
    
    def test_load_configuration_nonexistent_file(self):
        """Test loading configuration from nonexistent file."""
        engine = BiasEngine("nonexistent_file.json")
        
        # Should fall back to default configuration
        assert len(engine.configurations) == 4
        assert engine.configurations[BiasType.CONFIRMATION].base_strength == 0.3
    
    def test_save_configuration(self):
        """Test saving configuration to file."""
        engine = BiasEngine()
        
        # Modify a configuration
        engine.configure_bias_strength(BiasType.CONFIRMATION, 0.7, test_param="test")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            engine.save_configuration(config_path)
            
            # Load and verify
            with open(config_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["confirmation"]["base_strength"] == 0.7
            assert saved_data["confirmation"]["parameters"]["test_param"] == "test"
        
        finally:
            Path(config_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])