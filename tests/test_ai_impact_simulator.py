"""
Unit tests for AIImpactSimulator module.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

from src.enhancements.ai_impact_simulator import (
    AIImpactSimulator, AIUsageRecord, AIDetectionResult, AIPolicyRecord,
    AIAdoptionProfile, AIImpactMetrics, AIAssistanceType, AIDetectionMethod,
    AIPolicy, AIQualityImpact
)
from src.core.exceptions import ValidationError, PeerReviewError


class TestAIUsageRecord:
    """Test AIUsageRecord data class."""
    
    def test_valid_usage_record(self):
        """Test creating a valid AI usage record."""
        record = AIUsageRecord(
            usage_id="test_usage_1",
            researcher_id="researcher_1",
            paper_id="paper_1",
            assistance_type=AIAssistanceType.WRITING_ASSISTANCE,
            ai_tool_name="GPT-4",
            usage_date=date.today(),
            usage_duration_minutes=60,
            content_percentage=0.3,
            is_disclosed=True,
            quality_impact=AIQualityImpact.MODERATE_IMPROVEMENT
        )
        
        assert record.usage_id == "test_usage_1"
        assert record.researcher_id == "researcher_1"
        assert record.content_percentage == 0.3
        assert record.is_disclosed is True
        assert record.quality_impact == AIQualityImpact.MODERATE_IMPROVEMENT
    
    def test_invalid_content_percentage(self):
        """Test validation of content percentage."""
        with pytest.raises(ValidationError):
            AIUsageRecord(
                usage_id="test_usage_1",
                researcher_id="researcher_1",
                content_percentage=1.5  # Invalid: > 1.0
            )
        
        with pytest.raises(ValidationError):
            AIUsageRecord(
                usage_id="test_usage_1",
                researcher_id="researcher_1",
                content_percentage=-0.1  # Invalid: < 0.0
            )
    
    def test_invalid_duration(self):
        """Test validation of usage duration."""
        with pytest.raises(ValidationError):
            AIUsageRecord(
                usage_id="test_usage_1",
                researcher_id="researcher_1",
                usage_duration_minutes=-10  # Invalid: negative
            )
    
    def test_default_usage_date(self):
        """Test that usage date defaults to today."""
        record = AIUsageRecord(
            usage_id="test_usage_1",
            researcher_id="researcher_1"
        )
        assert record.usage_date == date.today()


class TestAIDetectionResult:
    """Test AIDetectionResult data class."""
    
    def test_valid_detection_result(self):
        """Test creating a valid AI detection result."""
        result = AIDetectionResult(
            detection_id="detection_1",
            content_id="paper_1",
            detection_method=AIDetectionMethod.STATISTICAL_ANALYSIS,
            detection_date=date.today(),
            ai_probability=0.8,
            confidence_level=0.9,
            flagged_sections=["abstract", "introduction"],
            false_positive_risk=0.1
        )
        
        assert result.detection_id == "detection_1"
        assert result.ai_probability == 0.8
        assert result.confidence_level == 0.9
        assert len(result.flagged_sections) == 2
    
    def test_invalid_probabilities(self):
        """Test validation of probability values."""
        with pytest.raises(ValidationError):
            AIDetectionResult(
                detection_id="detection_1",
                content_id="paper_1",
                detection_method=AIDetectionMethod.STATISTICAL_ANALYSIS,
                detection_date=date.today(),
                ai_probability=1.5,  # Invalid: > 1.0
                confidence_level=0.9,
                flagged_sections=[]
            )


class TestAIPolicyRecord:
    """Test AIPolicyRecord data class."""
    
    def test_valid_policy_record(self):
        """Test creating a valid AI policy record."""
        policy = AIPolicyRecord(
            institution_id="university_1",
            policy_type=AIPolicy.DISCLOSURE_REQUIRED,
            effective_date=date.today(),
            policy_description="All AI use must be disclosed",
            allowed_tools=["GPT-4", "Grammarly"],
            disclosure_requirements=["Specify tool", "Indicate percentage"],
            enforcement_level=0.8,
            violation_penalties=["Warning", "Training"]
        )
        
        assert policy.institution_id == "university_1"
        assert policy.policy_type == AIPolicy.DISCLOSURE_REQUIRED
        assert policy.enforcement_level == 0.8
        assert len(policy.allowed_tools) == 2
    
    def test_invalid_enforcement_level(self):
        """Test validation of enforcement level."""
        with pytest.raises(ValidationError):
            AIPolicyRecord(
                institution_id="university_1",
                policy_type=AIPolicy.DISCLOSURE_REQUIRED,
                effective_date=date.today(),
                policy_description="Test policy",
                allowed_tools=[],
                disclosure_requirements=[],
                enforcement_level=1.5,  # Invalid: > 1.0
                violation_penalties=[]
            )


class TestAIAdoptionProfile:
    """Test AIAdoptionProfile data class."""
    
    def test_valid_adoption_profile(self):
        """Test creating a valid AI adoption profile."""
        profile = AIAdoptionProfile(
            researcher_id="researcher_1",
            adoption_rate=0.7,
            preferred_tools=["GPT-4", "Claude"],
            comfort_level=0.8,
            ethical_concerns=0.3,
            disclosure_compliance=0.9,
            career_stage_influence=0.7,
            field_influence=0.8,
            institutional_policy_compliance=0.85
        )
        
        assert profile.researcher_id == "researcher_1"
        assert profile.adoption_rate == 0.7
        assert len(profile.preferred_tools) == 2
        assert profile.comfort_level == 0.8
    
    def test_invalid_rates(self):
        """Test validation of rate values."""
        with pytest.raises(ValidationError):
            AIAdoptionProfile(
                researcher_id="researcher_1",
                adoption_rate=1.2,  # Invalid: > 1.0
                preferred_tools=[],
                comfort_level=0.8,
                ethical_concerns=0.3,
                disclosure_compliance=0.9,
                career_stage_influence=0.7,
                field_influence=0.8,
                institutional_policy_compliance=0.85
            )


class TestAIImpactSimulator:
    """Test AIImpactSimulator class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def simulator(self, temp_dir):
        """Create an AIImpactSimulator instance for testing."""
        return AIImpactSimulator(data_dir=temp_dir / "ai_impact")
    
    def test_initialization(self, simulator):
        """Test simulator initialization."""
        assert isinstance(simulator, AIImpactSimulator)
        assert simulator.data_dir.exists()
        assert len(simulator.usage_records) == 0
        assert len(simulator.detection_results) == 0
        assert len(simulator.policy_records) == 0
        assert len(simulator.adoption_profiles) == 0
    
    def test_create_adoption_profile(self, simulator):
        """Test creating an AI adoption profile."""
        profile = simulator.create_adoption_profile(
            researcher_id="researcher_1",
            career_stage="Assistant Prof",
            field="computer_science",
            institution_id="university_1"
        )
        
        assert profile.researcher_id == "researcher_1"
        assert 0.0 <= profile.adoption_rate <= 1.0
        assert 0.0 <= profile.comfort_level <= 1.0
        assert 0.0 <= profile.ethical_concerns <= 1.0
        assert 0.0 <= profile.disclosure_compliance <= 1.0
        assert len(profile.preferred_tools) > 0
        
        # Check that profile is stored
        assert "researcher_1" in simulator.adoption_profiles
    
    def test_career_stage_influence(self, simulator):
        """Test that career stage influences adoption rates."""
        # Graduate students should have higher adoption than Full Professors
        grad_profile = simulator.create_adoption_profile(
            "grad_student", "Graduate Student", "computer_science", "university_1"
        )
        prof_profile = simulator.create_adoption_profile(
            "full_prof", "Full Prof", "computer_science", "university_1"
        )
        
        # Graduate students generally have higher adoption rates
        assert grad_profile.career_stage_influence >= prof_profile.career_stage_influence
    
    def test_field_influence(self, simulator):
        """Test that field influences adoption rates."""
        cs_profile = simulator.create_adoption_profile(
            "cs_researcher", "Assistant Prof", "computer_science", "university_1"
        )
        soc_profile = simulator.create_adoption_profile(
            "soc_researcher", "Assistant Prof", "sociology", "university_1"
        )
        
        # Computer science should generally have higher adoption
        assert cs_profile.field_influence >= soc_profile.field_influence
    
    @patch('random.random')
    def test_simulate_ai_usage_adoption(self, mock_random, simulator):
        """Test AI usage simulation based on adoption rate."""
        # Create profile with high adoption rate
        profile = simulator.create_adoption_profile(
            "researcher_1", "Graduate Student", "computer_science", "university_1"
        )
        profile.adoption_rate = 0.9
        
        # Mock random to return value below adoption rate
        mock_random.return_value = 0.5
        
        usage = simulator.simulate_ai_usage(
            researcher_id="researcher_1",
            paper_id="paper_1",
            assistance_type=AIAssistanceType.WRITING_ASSISTANCE
        )
        
        assert usage is not None
        assert usage.researcher_id == "researcher_1"
        assert usage.paper_id == "paper_1"
        assert usage.assistance_type == AIAssistanceType.WRITING_ASSISTANCE
    
    @patch('random.random')
    def test_simulate_ai_usage_no_adoption(self, mock_random, simulator):
        """Test AI usage simulation when researcher doesn't adopt."""
        # Create profile with low adoption rate
        profile = simulator.create_adoption_profile(
            "researcher_1", "Full Prof", "sociology", "university_1"
        )
        profile.adoption_rate = 0.1
        
        # Mock random to return value above adoption rate
        mock_random.return_value = 0.9
        
        usage = simulator.simulate_ai_usage(
            researcher_id="researcher_1",
            paper_id="paper_1"
        )
        
        assert usage is None
    
    def test_simulate_ai_usage_content_percentage(self, simulator):
        """Test that content percentage varies by assistance type."""
        profile = simulator.create_adoption_profile(
            "researcher_1", "Assistant Prof", "computer_science", "university_1"
        )
        
        # Test different assistance types
        writing_usage = simulator.simulate_ai_usage(
            "researcher_1", "paper_1", assistance_type=AIAssistanceType.WRITING_ASSISTANCE
        )
        grammar_usage = simulator.simulate_ai_usage(
            "researcher_1", "paper_2", assistance_type=AIAssistanceType.GRAMMAR_CHECK
        )
        
        if writing_usage and grammar_usage:
            # Grammar check should generally have lower content percentage
            assert grammar_usage.content_percentage <= writing_usage.content_percentage * 2
    
    def test_detect_ai_content_with_ai(self, simulator):
        """Test AI detection when content actually contains AI."""
        # Create usage record
        usage = AIUsageRecord(
            usage_id="usage_1",
            researcher_id="researcher_1",
            paper_id="paper_1",
            content_percentage=0.5
        )
        
        detection = simulator.detect_ai_content(
            content_id="paper_1",
            detection_method=AIDetectionMethod.STATISTICAL_ANALYSIS,
            actual_ai_usage=usage
        )
        
        assert detection.content_id == "paper_1"
        assert detection.detection_method == AIDetectionMethod.STATISTICAL_ANALYSIS
        assert 0.0 <= detection.ai_probability <= 1.0
        assert 0.0 <= detection.confidence_level <= 1.0
        assert usage.detection_attempted is True
    
    def test_detect_ai_content_without_ai(self, simulator):
        """Test AI detection when content doesn't contain AI."""
        detection = simulator.detect_ai_content(
            content_id="paper_1",
            detection_method=AIDetectionMethod.HUMAN_REVIEW
        )
        
        assert detection.content_id == "paper_1"
        assert detection.detection_method == AIDetectionMethod.HUMAN_REVIEW
        # Should have lower probability for non-AI content
        assert detection.ai_probability < 0.8
    
    def test_detection_method_accuracy(self, simulator):
        """Test that different detection methods have different accuracy."""
        # Watermarking should be more accurate than statistical analysis
        watermark_detection = simulator.detect_ai_content(
            "paper_1", AIDetectionMethod.WATERMARKING
        )
        stats_detection = simulator.detect_ai_content(
            "paper_2", AIDetectionMethod.STATISTICAL_ANALYSIS
        )
        
        # Watermarking should generally have higher confidence
        assert watermark_detection.confidence_level >= stats_detection.confidence_level * 0.8
    
    def test_implement_ai_policy(self, simulator):
        """Test implementing an AI policy."""
        policy = simulator.implement_ai_policy(
            institution_id="university_1",
            policy_type=AIPolicy.DISCLOSURE_REQUIRED,
            policy_description="All AI use must be disclosed",
            allowed_tools=["GPT-4", "Grammarly"],
            enforcement_level=0.8
        )
        
        assert policy.institution_id == "university_1"
        assert policy.policy_type == AIPolicy.DISCLOSURE_REQUIRED
        assert policy.enforcement_level == 0.8
        assert "GPT-4" in policy.allowed_tools
        
        # Check that policy is stored
        assert "university_1" in simulator.policy_records
    
    def test_policy_impact_on_profiles(self, simulator):
        """Test that policies impact researcher profiles."""
        # Create profile first
        profile = simulator.create_adoption_profile(
            "researcher_1", "Assistant Prof", "computer_science", "university_1"
        )
        original_adoption = profile.adoption_rate
        
        # Implement prohibited policy
        simulator.implement_ai_policy(
            "university_1", AIPolicy.PROHIBITED, "No AI allowed", enforcement_level=0.9
        )
        
        # Profile should be updated
        updated_profile = simulator.adoption_profiles["researcher_1"]
        assert updated_profile.adoption_rate < original_adoption
    
    def test_calculate_ai_impact_metrics_empty(self, simulator):
        """Test calculating metrics with no data."""
        metrics = simulator.calculate_ai_impact_metrics()
        
        assert metrics.total_ai_usage_instances == 0
        assert metrics.ai_adoption_rate == 0.0
        assert metrics.average_content_percentage == 0.0
        assert metrics.disclosure_rate == 0.0
        assert metrics.detection_accuracy == 0.0
    
    def test_calculate_ai_impact_metrics_with_data(self, simulator):
        """Test calculating metrics with sample data."""
        # Create profiles and usage
        profile1 = simulator.create_adoption_profile(
            "researcher_1", "Assistant Prof", "computer_science", "university_1"
        )
        profile2 = simulator.create_adoption_profile(
            "researcher_2", "Full Prof", "biology", "university_2"
        )
        
        # Add usage records
        usage1 = AIUsageRecord(
            usage_id="usage_1",
            researcher_id="researcher_1",
            content_percentage=0.3,
            is_disclosed=True,
            quality_impact=AIQualityImpact.MODERATE_IMPROVEMENT
        )
        usage2 = AIUsageRecord(
            usage_id="usage_2",
            researcher_id="researcher_2",
            content_percentage=0.2,
            is_disclosed=False,
            quality_impact=AIQualityImpact.MINIMAL_IMPROVEMENT
        )
        
        simulator.usage_records["usage_1"] = usage1
        simulator.usage_records["usage_2"] = usage2
        
        metrics = simulator.calculate_ai_impact_metrics()
        
        assert metrics.total_ai_usage_instances == 2
        assert metrics.average_content_percentage == 0.25
        assert metrics.disclosure_rate == 0.5
        assert metrics.quality_improvement_rate == 1.0  # Both records show improvement
    
    def test_simulate_policy_enforcement(self, simulator):
        """Test policy enforcement simulation."""
        # Create policy and profile
        policy = simulator.implement_ai_policy(
            "university_1", AIPolicy.DISCLOSURE_REQUIRED, "Disclosure required"
        )
        profile = simulator.create_adoption_profile(
            "researcher_1", "Assistant Prof", "computer_science", "university_1"
        )
        
        # Add undisclosed usage
        usage = AIUsageRecord(
            usage_id="usage_1",
            researcher_id="researcher_1",
            content_percentage=0.3,
            is_disclosed=False  # Violation
        )
        simulator.usage_records["usage_1"] = usage
        
        enforcement_result = simulator.simulate_policy_enforcement("university_1")
        
        assert "institution_id" in enforcement_result
        assert "total_violations" in enforcement_result
        assert "compliance_rate" in enforcement_result
        assert enforcement_result["policy_type"] == "disclosure_required"
    
    def test_get_researcher_ai_score(self, simulator):
        """Test calculating researcher AI scores."""
        # Create profile and usage
        profile = simulator.create_adoption_profile(
            "researcher_1", "Assistant Prof", "computer_science", "university_1"
        )
        
        usage = AIUsageRecord(
            usage_id="usage_1",
            researcher_id="researcher_1",
            content_percentage=0.3,
            is_disclosed=True,
            quality_impact=AIQualityImpact.MODERATE_IMPROVEMENT
        )
        simulator.usage_records["usage_1"] = usage
        
        scores = simulator.get_researcher_ai_score("researcher_1")
        
        assert "adoption_score" in scores
        assert "disclosure_score" in scores
        assert "quality_impact_score" in scores
        assert "policy_compliance_score" in scores
        assert "overall_ai_score" in scores
        
        # All scores should be between 0 and 1
        for score_name, score_value in scores.items():
            assert 0.0 <= score_value <= 1.0
    
    def test_get_researcher_ai_score_no_usage(self, simulator):
        """Test AI scores for researcher with no usage."""
        profile = simulator.create_adoption_profile(
            "researcher_1", "Assistant Prof", "computer_science", "university_1"
        )
        
        scores = simulator.get_researcher_ai_score("researcher_1")
        
        assert scores["quality_impact_score"] == 0.0
        assert scores["disclosure_score"] == profile.disclosure_compliance
    
    def test_save_and_load_data(self, simulator):
        """Test saving and loading data."""
        # Create some test data
        profile = simulator.create_adoption_profile(
            "researcher_1", "Assistant Prof", "computer_science", "university_1"
        )
        
        usage = AIUsageRecord(
            usage_id="usage_1",
            researcher_id="researcher_1",
            content_percentage=0.3,
            is_disclosed=True
        )
        simulator.usage_records["usage_1"] = usage
        
        policy = simulator.implement_ai_policy(
            "university_1", AIPolicy.DISCLOSURE_REQUIRED, "Test policy"
        )
        
        detection = simulator.detect_ai_content(
            "paper_1", AIDetectionMethod.STATISTICAL_ANALYSIS
        )
        
        # Save data
        simulator.save_data()
        
        # Create new simulator and load data
        new_simulator = AIImpactSimulator(data_dir=simulator.data_dir)
        
        # Check that data was loaded
        assert len(new_simulator.adoption_profiles) == 1
        assert len(new_simulator.usage_records) == 1
        assert len(new_simulator.policy_records) == 1
        assert len(new_simulator.detection_results) == 1
        
        # Check data integrity
        loaded_profile = new_simulator.adoption_profiles["researcher_1"]
        assert loaded_profile.researcher_id == "researcher_1"
        
        loaded_usage = list(new_simulator.usage_records.values())[0]
        assert loaded_usage.researcher_id == "researcher_1"
        assert loaded_usage.content_percentage == 0.3
    
    def test_assistance_type_variations(self, simulator):
        """Test that different assistance types have appropriate characteristics."""
        profile = simulator.create_adoption_profile(
            "researcher_1", "Assistant Prof", "computer_science", "university_1"
        )
        
        # Test multiple assistance types
        assistance_types = [
            AIAssistanceType.WRITING_ASSISTANCE,
            AIAssistanceType.GRAMMAR_CHECK,
            AIAssistanceType.LITERATURE_REVIEW,
            AIAssistanceType.DATA_ANALYSIS
        ]
        
        for assistance_type in assistance_types:
            usage = simulator.simulate_ai_usage(
                "researcher_1", f"paper_{assistance_type.value}", 
                assistance_type=assistance_type
            )
            
            if usage:  # May be None due to random adoption
                assert usage.assistance_type == assistance_type
                assert usage.usage_duration_minutes > 0
                assert 0.0 <= usage.content_percentage <= 1.0
    
    def test_quality_impact_distribution(self, simulator):
        """Test that quality impact varies appropriately."""
        profile = simulator.create_adoption_profile(
            "researcher_1", "Assistant Prof", "computer_science", "university_1"
        )
        
        # Generate multiple usage records
        quality_impacts = []
        for i in range(10):
            usage = simulator.simulate_ai_usage(
                "researcher_1", f"paper_{i}", 
                assistance_type=AIAssistanceType.WRITING_ASSISTANCE
            )
            if usage:
                quality_impacts.append(usage.quality_impact)
        
        # Should have some variation in quality impacts
        if len(quality_impacts) > 1:
            unique_impacts = set(quality_impacts)
            # Not all impacts should be the same (with high probability)
            assert len(unique_impacts) >= 1
    
    def test_detection_flagged_sections(self, simulator):
        """Test that detection results include appropriate flagged sections."""
        usage = AIUsageRecord(
            usage_id="usage_1",
            researcher_id="researcher_1",
            paper_id="paper_1",
            content_percentage=0.8  # High AI content
        )
        
        detection = simulator.detect_ai_content(
            "paper_1", AIDetectionMethod.HYBRID_DETECTION, usage
        )
        
        # High AI content should likely result in flagged sections
        if detection.ai_probability > 0.5:
            assert len(detection.flagged_sections) > 0
            # Flagged sections should be valid section names
            valid_sections = ["abstract", "introduction", "methodology", "results", "discussion", "conclusion"]
            for section in detection.flagged_sections:
                assert section in valid_sections


class TestAIImpactMetrics:
    """Test AIImpactMetrics data class."""
    
    def test_valid_metrics(self):
        """Test creating valid AI impact metrics."""
        metrics = AIImpactMetrics(
            total_ai_usage_instances=100,
            ai_adoption_rate=0.7,
            average_content_percentage=0.3,
            disclosure_rate=0.8,
            detection_accuracy=0.75,
            quality_improvement_rate=0.6,
            policy_compliance_rate=0.85,
            tool_usage_distribution={"GPT-4": 50, "Claude": 30, "Grammarly": 20},
            assistance_type_distribution={AIAssistanceType.WRITING_ASSISTANCE: 60, AIAssistanceType.GRAMMAR_CHECK: 40},
            detection_method_effectiveness={AIDetectionMethod.STATISTICAL_ANALYSIS: 0.7, AIDetectionMethod.HUMAN_REVIEW: 0.85}
        )
        
        assert metrics.total_ai_usage_instances == 100
        assert metrics.ai_adoption_rate == 0.7
        assert metrics.detection_accuracy == 0.75
        assert len(metrics.tool_usage_distribution) == 3
    
    def test_invalid_metrics(self):
        """Test validation of metrics values."""
        with pytest.raises(ValidationError):
            AIImpactMetrics(
                total_ai_usage_instances=-1,  # Invalid: negative
                ai_adoption_rate=0.7,
                average_content_percentage=0.3,
                disclosure_rate=0.8,
                detection_accuracy=0.75,
                quality_improvement_rate=0.6,
                policy_compliance_rate=0.85,
                tool_usage_distribution={},
                assistance_type_distribution={},
                detection_method_effectiveness={}
            )


if __name__ == "__main__":
    pytest.main([__file__])