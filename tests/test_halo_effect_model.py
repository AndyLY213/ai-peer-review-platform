"""
Unit tests for the Halo Effect Bias Model.

Tests the HaloEffectModel class and its ability to model prestige-based bias
in peer review based on author reputation and reviewer susceptibility.
"""

import pytest
from unittest.mock import Mock, patch

from src.enhancements.halo_effect_model import (
    HaloEffectModel, AuthorPrestige
)
from src.enhancements.bias_engine import BiasConfiguration, BiasType
from src.data.enhanced_models import (
    EnhancedResearcher, StructuredReview, BiasEffect,
    ResearcherLevel, ReviewDecision, CareerStage
)


class TestAuthorPrestige:
    """Test cases for AuthorPrestige dataclass."""
    
    def test_author_prestige_creation(self):
        """Test creating an author prestige object."""
        prestige = AuthorPrestige(
            author_id="test_author",
            h_index=30,
            total_citations=2000,
            institution_tier=1,
            years_active=15,
            notable_awards=["Best Paper Award", "Young Researcher Award"]
        )
        
        assert prestige.author_id == "test_author"
        assert prestige.h_index == 30
        assert prestige.total_citations == 2000
        assert prestige.institution_tier == 1
        assert prestige.years_active == 15
        assert len(prestige.notable_awards) == 2
        assert prestige.prestige_score > 0  # Should be calculated automatically
    
    def test_prestige_score_calculation(self):
        """Test prestige score calculation."""
        # High prestige author
        high_prestige = AuthorPrestige(
            author_id="high_prestige",
            h_index=50,
            total_citations=5000,
            institution_tier=1,
            years_active=25,
            notable_awards=["Turing Award", "Best Paper Award"]
        )
        
        # Low prestige author
        low_prestige = AuthorPrestige(
            author_id="low_prestige",
            h_index=5,
            total_citations=100,
            institution_tier=3,
            years_active=2,
            notable_awards=[]
        )
        
        assert high_prestige.prestige_score > low_prestige.prestige_score
        assert 0.0 <= low_prestige.prestige_score <= 1.0
        assert 0.0 <= high_prestige.prestige_score <= 1.0
    
    def test_prestige_score_bounds(self):
        """Test that prestige scores are within valid bounds."""
        # Extreme values
        extreme_prestige = AuthorPrestige(
            author_id="extreme",
            h_index=100,
            total_citations=10000,
            institution_tier=1,
            years_active=50,
            notable_awards=["Award1", "Award2", "Award3", "Award4", "Award5"]
        )
        
        # Should not exceed reasonable bounds even with extreme values
        assert extreme_prestige.prestige_score <= 1.5  # Allow some bonus from awards


class TestHaloEffectModel:
    """Test cases for HaloEffectModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = BiasConfiguration(
            bias_type=BiasType.HALO_EFFECT,
            base_strength=0.4,
            parameters={
                'reputation_threshold': 0.7,
                'max_score_boost': 2.0,
                'prestige_factor': 0.5
            }
        )
        self.model = HaloEffectModel(self.config)
        
        self.reviewer = EnhancedResearcher(
            id="test_reviewer",
            name="Test Reviewer",
            specialty="AI",
            level=ResearcherLevel.ASSISTANT_PROF,
            career_stage=CareerStage.EARLY_CAREER,
            reputation_score=0.5,
            cognitive_biases={"halo_effect": 0.6}
        )
        
        self.review = StructuredReview(
            reviewer_id="test_reviewer",
            paper_id="test_paper",
            venue_id="test_venue"
        )
    
    def test_model_initialization(self):
        """Test halo effect model initialization."""
        assert self.model.bias_type == BiasType.HALO_EFFECT
        assert self.model.reputation_threshold == 0.7
        assert self.model.max_score_boost == 2.0
        assert self.model.prestige_factor == 0.5
        assert len(self.model.author_prestige_db) > 0  # Should have sample data
    
    def test_sample_prestige_data(self):
        """Test that sample prestige data is properly initialized."""
        assert "prestigious_author_1" in self.model.author_prestige_db
        assert "mid_tier_author_1" in self.model.author_prestige_db
        assert "junior_author_1" in self.model.author_prestige_db
        
        # Check prestige scores are ordered correctly
        prestigious = self.model.author_prestige_db["prestigious_author_1"]
        mid_tier = self.model.author_prestige_db["mid_tier_author_1"]
        junior = self.model.author_prestige_db["junior_author_1"]
        
        assert prestigious.prestige_score > mid_tier.prestige_score > junior.prestige_score
    
    def test_is_applicable_with_valid_context(self):
        """Test is_applicable with valid context."""
        context = {
            'paper_authors': ['prestigious_author_1', 'mid_tier_author_1']
        }
        
        assert self.model.is_applicable(self.reviewer, context) is True
    
    def test_is_applicable_without_authors(self):
        """Test is_applicable without author information."""
        context = {}
        
        assert self.model.is_applicable(self.reviewer, context) is False
    
    def test_is_applicable_no_susceptibility(self):
        """Test is_applicable with reviewer having no halo effect susceptibility."""
        reviewer = EnhancedResearcher(
            id="test_reviewer",
            name="Test Reviewer",
            specialty="AI",
            cognitive_biases={"halo_effect": 0.0}  # No susceptibility
        )
        
        context = {'paper_authors': ['prestigious_author_1']}
        
        assert self.model.is_applicable(reviewer, context) is False
    
    def test_calculate_max_author_prestige_known_authors(self):
        """Test calculating maximum prestige for known authors."""
        paper_authors = ["prestigious_author_1", "junior_author_1"]
        
        max_prestige = self.model._calculate_max_author_prestige(paper_authors)
        
        # Should return the prestige of the most prestigious author
        expected_max = self.model.author_prestige_db["prestigious_author_1"].prestige_score
        assert max_prestige == expected_max
    
    def test_calculate_max_author_prestige_unknown_authors(self):
        """Test calculating maximum prestige for unknown authors."""
        paper_authors = ["unknown_senior_prof", "unknown_student"]
        
        max_prestige = self.model._calculate_max_author_prestige(paper_authors)
        
        # Should use estimation and return higher prestige for "senior_prof"
        assert max_prestige > 0.0
        assert max_prestige <= 1.0
    
    def test_estimate_author_prestige(self):
        """Test author prestige estimation for unknown authors."""
        # Test different patterns
        senior_prestige = self.model._estimate_author_prestige("senior_researcher")
        student_prestige = self.model._estimate_author_prestige("phd_student")
        unknown_prestige = self.model._estimate_author_prestige("random_author")
        
        assert senior_prestige > student_prestige
        assert senior_prestige == 0.7  # Expected value for "senior"
        assert student_prestige == 0.1  # Expected value for "student"
        assert unknown_prestige == 0.4  # Default value
    
    def test_calculate_reviewer_prestige_bias(self):
        """Test calculation of reviewer's susceptibility to prestige bias."""
        # Test with different reviewer profiles
        junior_reviewer = EnhancedResearcher(
            id="junior",
            name="Junior Reviewer",
            specialty="AI",
            level=ResearcherLevel.POSTDOC,  # Will result in EARLY_CAREER
            years_active=3,
            reputation_score=0.2,
            cognitive_biases={"halo_effect": 0.3}
        )
        
        senior_reviewer = EnhancedResearcher(
            id="senior",
            name="Senior Reviewer",
            specialty="AI",
            level=ResearcherLevel.FULL_PROF,  # Will result in SENIOR_CAREER
            years_active=25,
            reputation_score=0.9,
            cognitive_biases={"halo_effect": 0.3}
        )
        
        junior_bias = self.model._calculate_reviewer_prestige_bias(junior_reviewer)
        senior_bias = self.model._calculate_reviewer_prestige_bias(senior_reviewer)
        
        # Calculate expected values manually to verify
        # Junior: base=0.3, prestige_adj=(1.0-0.2)*0.3=0.24, career_adj=0.2 -> 0.74
        # Senior: base=0.3, prestige_adj=(1.0-0.9)*0.3=0.03, career_adj=-0.1 -> 0.23
        
        # Junior reviewer should be more susceptible to prestige bias
        assert junior_bias > senior_bias
        assert 0.0 <= junior_bias <= 1.0
        assert 0.0 <= senior_bias <= 1.0
    
    def test_calculate_score_adjustment_high_prestige(self):
        """Test score adjustment calculation for high prestige authors."""
        author_prestige = 0.9  # High prestige
        effective_strength = 0.5
        reviewer_prestige_bias = 0.6
        
        adjustment = self.model._calculate_score_adjustment(
            author_prestige, effective_strength, reviewer_prestige_bias
        )
        
        assert adjustment > 0  # Should boost score
        assert adjustment <= self.model.max_score_boost
    
    def test_calculate_score_adjustment_low_prestige(self):
        """Test score adjustment calculation for low prestige authors."""
        author_prestige = 0.5  # Below threshold (0.7)
        effective_strength = 0.5
        reviewer_prestige_bias = 0.6
        
        adjustment = self.model._calculate_score_adjustment(
            author_prestige, effective_strength, reviewer_prestige_bias
        )
        
        assert adjustment == 0.0  # No boost for low prestige
    
    def test_calculate_score_adjustment_extremely_high_prestige(self):
        """Test score adjustment with extremely high prestige (bonus multiplier)."""
        author_prestige = 0.95  # Extremely high prestige
        effective_strength = 0.5
        reviewer_prestige_bias = 0.6
        
        adjustment = self.model._calculate_score_adjustment(
            author_prestige, effective_strength, reviewer_prestige_bias
        )
        
        # Should get bonus multiplier for extremely high prestige
        base_adjustment = self.model._calculate_score_adjustment(
            0.85, effective_strength, reviewer_prestige_bias  # High but not extreme
        )
        
        assert adjustment > base_adjustment
    
    def test_calculate_bias_effect_high_prestige(self):
        """Test calculating bias effect with high prestige authors."""
        context = {
            'paper_authors': ['prestigious_author_1']
        }
        
        bias_effect = self.model.calculate_bias_effect(self.reviewer, self.review, context)
        
        assert bias_effect.bias_type == "halo_effect"
        assert bias_effect.strength > 0
        assert bias_effect.score_adjustment > 0  # Positive adjustment
        assert "halo effect" in bias_effect.description.lower()
    
    def test_calculate_bias_effect_low_prestige(self):
        """Test calculating bias effect with low prestige authors."""
        context = {
            'paper_authors': ['junior_author_1']
        }
        
        bias_effect = self.model.calculate_bias_effect(self.reviewer, self.review, context)
        
        assert bias_effect.bias_type == "halo_effect"
        assert bias_effect.score_adjustment == 0.0  # No boost for low prestige
        assert "minimal" in bias_effect.description.lower()
    
    def test_calculate_bias_effect_no_authors(self):
        """Test calculating bias effect without author information."""
        context = {}
        
        bias_effect = self.model.calculate_bias_effect(self.reviewer, self.review, context)
        
        assert bias_effect.bias_type == "halo_effect"
        assert bias_effect.strength == 0.0
        assert bias_effect.score_adjustment == 0.0
        assert "no author information" in bias_effect.description.lower()
    
    def test_generate_bias_description(self):
        """Test bias description generation."""
        # Extremely high prestige with significant boost
        desc = self.model._generate_bias_description(0.97, 1.5, ["author1", "author2"])
        assert "extremely high prestige" in desc.lower()
        assert "2 authors" in desc
        assert "+1.50" in desc
        
        # Moderate prestige with small boost
        desc = self.model._generate_bias_description(0.75, 0.3, ["author1"])
        assert "high prestige" in desc.lower()
        assert "1 author" in desc
        assert "+0.30" in desc
        
        # Minimal effect
        desc = self.model._generate_bias_description(0.6, 0.02, ["author1"])
        assert "minimal" in desc.lower()
    
    def test_add_author_prestige(self):
        """Test adding author prestige information."""
        initial_count = len(self.model.author_prestige_db)
        
        self.model.add_author_prestige(
            author_id="new_author",
            h_index=20,
            total_citations=1000,
            institution_tier=2,
            years_active=10,
            notable_awards=["Best Paper Award"]
        )
        
        assert len(self.model.author_prestige_db) == initial_count + 1
        assert "new_author" in self.model.author_prestige_db
        
        new_author = self.model.author_prestige_db["new_author"]
        assert new_author.h_index == 20
        assert new_author.total_citations == 1000
        assert new_author.institution_tier == 2
        assert new_author.years_active == 10
        assert "Best Paper Award" in new_author.notable_awards
        assert new_author.prestige_score > 0
    
    def test_get_author_prestige(self):
        """Test getting author prestige information."""
        # Existing author
        prestige = self.model.get_author_prestige("prestigious_author_1")
        assert prestige is not None
        assert prestige.author_id == "prestigious_author_1"
        
        # Non-existent author
        prestige = self.model.get_author_prestige("non_existent_author")
        assert prestige is None
    
    def test_update_author_prestige_score(self):
        """Test updating author prestige score."""
        author_id = "prestigious_author_1"
        original_score = self.model.author_prestige_db[author_id].prestige_score
        new_score = 0.95
        
        self.model.update_author_prestige_score(author_id, new_score)
        
        updated_score = self.model.author_prestige_db[author_id].prestige_score
        assert updated_score == new_score
        assert updated_score != original_score
    
    def test_update_author_prestige_score_bounds(self):
        """Test that prestige score updates respect bounds."""
        author_id = "prestigious_author_1"
        
        # Test upper bound
        self.model.update_author_prestige_score(author_id, 1.5)
        assert self.model.author_prestige_db[author_id].prestige_score == 1.0
        
        # Test lower bound
        self.model.update_author_prestige_score(author_id, -0.5)
        assert self.model.author_prestige_db[author_id].prestige_score == 0.0
    
    def test_update_author_prestige_score_nonexistent(self):
        """Test updating prestige score for non-existent author."""
        # Should not raise error, just log warning
        self.model.update_author_prestige_score("non_existent", 0.8)
        # Test passes if no exception is raised
    
    def test_get_prestige_statistics(self):
        """Test getting prestige database statistics."""
        stats = self.model.get_prestige_statistics()
        
        assert "total_authors" in stats
        assert "average_prestige" in stats
        assert "max_prestige" in stats
        assert "min_prestige" in stats
        assert "high_prestige_authors" in stats
        assert "low_prestige_authors" in stats
        
        assert stats["total_authors"] > 0
        assert 0.0 <= stats["average_prestige"] <= 1.0
        assert stats["max_prestige"] >= stats["min_prestige"]
    
    def test_get_prestige_statistics_empty_db(self):
        """Test getting statistics with empty database."""
        # Create model with empty database
        empty_model = HaloEffectModel(self.config)
        empty_model.author_prestige_db.clear()
        
        stats = empty_model.get_prestige_statistics()
        
        assert stats["total_authors"] == 0
        assert stats["average_prestige"] == 0.0
    
    def test_simulate_author_network_effect_single_author(self):
        """Test network effect simulation with single author."""
        paper_authors = ["prestigious_author_1"]
        
        network_effect = self.model.simulate_author_network_effect(paper_authors)
        
        assert network_effect == 1.0  # No network effect for single author
    
    def test_simulate_author_network_effect_multiple_authors(self):
        """Test network effect simulation with multiple authors."""
        paper_authors = ["prestigious_author_1", "mid_tier_author_1", "junior_author_1"]
        
        network_effect = self.model.simulate_author_network_effect(paper_authors)
        
        assert network_effect > 1.0  # Should have positive network effect
        assert network_effect <= 1.5  # Should be reasonable
    
    def test_simulate_author_network_effect_unknown_authors(self):
        """Test network effect simulation with unknown authors."""
        paper_authors = ["unknown_author_1", "unknown_author_2"]
        
        network_effect = self.model.simulate_author_network_effect(paper_authors)
        
        assert network_effect >= 1.0  # Should not be negative
        assert network_effect <= 1.5  # Should be reasonable


if __name__ == "__main__":
    pytest.main([__file__])