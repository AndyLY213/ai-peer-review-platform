"""
Unit tests for PublicationReformManager

Tests the publication reform system including alternative metrics,
post-publication review, new evaluation models, and reform impact assessment.
"""

import pytest
import tempfile
import shutil
from datetime import datetime, date, timedelta
from pathlib import Path
import json

from src.enhancements.publication_reform_manager import (
    PublicationReformManager, ReformType, MetricType, AdoptionStage,
    AlternativeMetric, PostPublicationReview, ReformImplementation,
    EvaluationModel
)
from src.data.enhanced_models import (
    StructuredReview, EnhancedReviewCriteria, VenueType, ReviewDecision
)
from src.core.exceptions import ValidationError, DatabaseError


class TestPublicationReformManager:
    """Test cases for PublicationReformManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = PublicationReformManager(data_dir=self.temp_dir)
        
        # Test data
        self.venue_id = "test_venue_001"
        self.paper_id = "test_paper_001"
        self.reviewer_id = "test_reviewer_001"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test manager initialization."""
        assert isinstance(self.manager, PublicationReformManager)
        assert Path(self.temp_dir).exists()
        assert self.manager.reforms == {}
        assert self.manager.evaluation_models == {}
        assert self.manager.alternative_metrics == {}
        assert self.manager.post_pub_reviews == {}
    
    def test_implement_reform(self):
        """Test implementing a publication reform."""
        parameters = {"threshold": 0.8, "pilot_duration": 6}
        
        reform_id = self.manager.implement_reform(
            venue_id=self.venue_id,
            reform_type=ReformType.ALTERNATIVE_METRICS,
            parameters=parameters
        )
        
        assert reform_id in self.manager.reforms
        reform = self.manager.reforms[reform_id]
        assert reform.venue_id == self.venue_id
        assert reform.reform_type == ReformType.ALTERNATIVE_METRICS
        assert reform.parameters == parameters
        assert reform.adoption_stage == AdoptionStage.EXPERIMENTAL
        assert len(self.manager.adoption_history) == 1
    
    def test_implement_reform_without_parameters(self):
        """Test implementing reform without parameters."""
        reform_id = self.manager.implement_reform(
            venue_id=self.venue_id,
            reform_type=ReformType.POST_PUBLICATION_REVIEW
        )
        
        reform = self.manager.reforms[reform_id]
        assert reform.parameters == {}
    
    def test_create_evaluation_model(self):
        """Test creating a new evaluation model."""
        model_id = self.manager.create_evaluation_model(
            name="Hybrid Model",
            description="Combines traditional and alternative metrics",
            traditional_weight=0.4,
            alternative_weight=0.4,
            post_pub_weight=0.2,
            included_metrics=[MetricType.ALTMETRIC_SCORE, MetricType.DOWNLOAD_COUNT]
        )
        
        assert model_id in self.manager.evaluation_models
        model = self.manager.evaluation_models[model_id]
        assert model.name == "Hybrid Model"
        assert model.traditional_weight == 0.4
        assert model.alternative_metrics_weight == 0.4
        assert model.post_pub_review_weight == 0.2
        assert MetricType.ALTMETRIC_SCORE in model.included_metrics
        assert MetricType.DOWNLOAD_COUNT in model.included_metrics
    
    def test_create_evaluation_model_invalid_weights(self):
        """Test creating evaluation model with invalid weights."""
        with pytest.raises(ValidationError):
            self.manager.create_evaluation_model(
                name="Invalid Model",
                description="Weights don't sum to 1.0",
                traditional_weight=0.6,
                alternative_weight=0.6,  # Total > 1.0
                post_pub_weight=0.2
            )
    
    def test_add_alternative_metric(self):
        """Test adding alternative metrics."""
        self.manager.add_alternative_metric(
            paper_id=self.paper_id,
            metric_type=MetricType.ALTMETRIC_SCORE,
            value=85.5,
            source="Altmetric API",
            confidence=0.9
        )
        
        assert self.paper_id in self.manager.alternative_metrics
        metrics = self.manager.alternative_metrics[self.paper_id]
        assert len(metrics) == 1
        
        metric = metrics[0]
        assert metric.metric_type == MetricType.ALTMETRIC_SCORE
        assert metric.value == 85.5
        assert metric.source == "Altmetric API"
        assert metric.confidence == 0.9
    
    def test_add_multiple_alternative_metrics(self):
        """Test adding multiple alternative metrics for same paper."""
        self.manager.add_alternative_metric(
            paper_id=self.paper_id,
            metric_type=MetricType.ALTMETRIC_SCORE,
            value=85.5
        )
        
        self.manager.add_alternative_metric(
            paper_id=self.paper_id,
            metric_type=MetricType.DOWNLOAD_COUNT,
            value=1250
        )
        
        metrics = self.manager.alternative_metrics[self.paper_id]
        assert len(metrics) == 2
        assert any(m.metric_type == MetricType.ALTMETRIC_SCORE for m in metrics)
        assert any(m.metric_type == MetricType.DOWNLOAD_COUNT for m in metrics)
    
    def test_add_alternative_metric_invalid_confidence(self):
        """Test adding alternative metric with invalid confidence."""
        with pytest.raises(ValidationError):
            self.manager.add_alternative_metric(
                paper_id=self.paper_id,
                metric_type=MetricType.ALTMETRIC_SCORE,
                value=85.5,
                confidence=1.5  # Invalid confidence > 1.0
            )
    
    def test_submit_post_publication_review(self):
        """Test submitting a post-publication review."""
        review_id = self.manager.submit_post_publication_review(
            paper_id=self.paper_id,
            reviewer_id=self.reviewer_id,
            rating=8.5,
            summary="Excellent work with minor issues",
            strengths=["Novel approach", "Clear presentation"],
            weaknesses=["Limited evaluation"],
            is_anonymous=False
        )
        
        assert self.paper_id in self.manager.post_pub_reviews
        reviews = self.manager.post_pub_reviews[self.paper_id]
        assert len(reviews) == 1
        
        review = reviews[0]
        assert review.review_id == review_id
        assert review.paper_id == self.paper_id
        assert review.reviewer_id == self.reviewer_id
        assert review.rating == 8.5
        assert review.summary == "Excellent work with minor issues"
        assert len(review.strengths) == 2
        assert len(review.weaknesses) == 1
        assert not review.is_anonymous
    
    def test_submit_post_publication_review_invalid_rating(self):
        """Test submitting post-publication review with invalid rating."""
        with pytest.raises(ValidationError):
            self.manager.submit_post_publication_review(
                paper_id=self.paper_id,
                reviewer_id=self.reviewer_id,
                rating=11.0,  # Invalid rating > 10.0
                summary="Test review"
            )
    
    def test_evaluate_paper_with_model(self):
        """Test evaluating a paper using an evaluation model."""
        # Create evaluation model
        model_id = self.manager.create_evaluation_model(
            name="Test Model",
            description="Test evaluation model",
            traditional_weight=0.5,
            alternative_weight=0.3,
            post_pub_weight=0.2,
            included_metrics=[MetricType.ALTMETRIC_SCORE]
        )
        
        # Add alternative metrics
        self.manager.add_alternative_metric(
            paper_id=self.paper_id,
            metric_type=MetricType.ALTMETRIC_SCORE,
            value=80.0
        )
        
        # Add post-publication review
        self.manager.submit_post_publication_review(
            paper_id=self.paper_id,
            reviewer_id=self.reviewer_id,
            rating=7.5,
            summary="Good work"
        )
        
        # Create traditional reviews
        traditional_reviews = [
            StructuredReview(
                reviewer_id="reviewer1",
                paper_id=self.paper_id,
                venue_id=self.venue_id,
                criteria_scores=EnhancedReviewCriteria(
                    novelty=8.0, technical_quality=7.5, clarity=8.5,
                    significance=7.0, reproducibility=6.5, related_work=7.5
                )
            )
        ]
        
        # Evaluate paper
        result = self.manager.evaluate_paper_with_model(
            paper_id=self.paper_id,
            model_id=model_id,
            traditional_reviews=traditional_reviews
        )
        
        assert 'final_score' in result
        assert 'traditional_score' in result
        assert 'alternative_metrics_score' in result
        assert 'post_publication_score' in result
        assert result['model_id'] == model_id
        assert 'evaluation_date' in result
        
        # Check that scores are reasonable
        assert 0 <= result['final_score'] <= 10
        assert 0 <= result['traditional_score'] <= 10
        assert 0 <= result['alternative_metrics_score'] <= 10
        assert 0 <= result['post_publication_score'] <= 10
    
    def test_evaluate_paper_nonexistent_model(self):
        """Test evaluating paper with non-existent model."""
        with pytest.raises(ValidationError):
            self.manager.evaluate_paper_with_model(
                paper_id=self.paper_id,
                model_id="nonexistent_model",
                traditional_reviews=[]
            )
    
    def test_track_reform_adoption(self):
        """Test tracking reform adoption and impact."""
        # Implement a reform first
        reform_id = self.manager.implement_reform(
            venue_id=self.venue_id,
            reform_type=ReformType.ALTERNATIVE_METRICS
        )
        
        # Track adoption
        self.manager.track_reform_adoption(
            reform_id=reform_id,
            adoption_rate=0.6,
            researcher_satisfaction=0.8,
            quality_improvement=0.15,
            efficiency_gain=0.1
        )
        
        reform = self.manager.reforms[reform_id]
        assert reform.adoption_rate == 0.6
        assert reform.researcher_satisfaction == 0.8
        assert reform.quality_improvement == 0.15
        assert reform.efficiency_gain == 0.1
        assert reform.adoption_stage == AdoptionStage.WIDESPREAD_ADOPTION
        
        # Check adoption history updated
        assert len(self.manager.adoption_history) == 2  # Initial + tracking update
    
    def test_track_reform_adoption_stage_transitions(self):
        """Test adoption stage transitions based on adoption rate."""
        reform_id = self.manager.implement_reform(
            venue_id=self.venue_id,
            reform_type=ReformType.POST_PUBLICATION_REVIEW
        )
        
        # Test different adoption rates
        test_cases = [
            (0.05, AdoptionStage.EXPERIMENTAL),
            (0.15, AdoptionStage.PILOT),
            (0.4, AdoptionStage.PARTIAL_ADOPTION),
            (0.7, AdoptionStage.WIDESPREAD_ADOPTION),
            (0.9, AdoptionStage.STANDARD_PRACTICE)
        ]
        
        for adoption_rate, expected_stage in test_cases:
            self.manager.track_reform_adoption(
                reform_id=reform_id,
                adoption_rate=adoption_rate,
                researcher_satisfaction=0.7
            )
            
            reform = self.manager.reforms[reform_id]
            assert reform.adoption_stage == expected_stage
    
    def test_track_reform_adoption_nonexistent_reform(self):
        """Test tracking adoption for non-existent reform."""
        with pytest.raises(ValidationError):
            self.manager.track_reform_adoption(
                reform_id="nonexistent_reform",
                adoption_rate=0.5,
                researcher_satisfaction=0.7
            )
    
    def test_assess_reform_impact(self):
        """Test assessing reform impact."""
        # Implement and track a reform
        reform_id = self.manager.implement_reform(
            venue_id=self.venue_id,
            reform_type=ReformType.ALTERNATIVE_METRICS
        )
        
        self.manager.track_reform_adoption(
            reform_id=reform_id,
            adoption_rate=0.7,
            researcher_satisfaction=0.8,
            quality_improvement=0.2,
            efficiency_gain=0.15
        )
        
        # Assess impact
        impact = self.manager.assess_reform_impact(reform_id)
        
        assert impact['reform_id'] == reform_id
        assert impact['reform_type'] == ReformType.ALTERNATIVE_METRICS.value
        assert impact['venue_id'] == self.venue_id
        assert 'implementation_date' in impact
        assert 'days_active' in impact
        assert impact['current_stage'] == AdoptionStage.WIDESPREAD_ADOPTION.value
        assert impact['adoption_rate'] == 0.7
        assert impact['researcher_satisfaction'] == 0.8
        assert impact['quality_improvement'] == 0.2
        assert impact['efficiency_gain'] == 0.15
        assert 'overall_impact_score' in impact
        assert 'assessment_date' in impact
        
        # Check impact score is reasonable
        assert 0 <= impact['overall_impact_score'] <= 1
    
    def test_assess_reform_impact_nonexistent_reform(self):
        """Test assessing impact for non-existent reform."""
        with pytest.raises(ValidationError):
            self.manager.assess_reform_impact("nonexistent_reform")
    
    def test_get_reform_recommendations(self):
        """Test getting reform recommendations for different venue types."""
        # Test recommendations for top conference
        recommendations = self.manager.get_reform_recommendations(
            venue_id=self.venue_id,
            venue_type=VenueType.TOP_CONFERENCE
        )
        
        assert len(recommendations) > 0
        assert all('reform_type' in rec for rec in recommendations)
        assert all('suitability_score' in rec for rec in recommendations)
        assert all('description' in rec for rec in recommendations)
        assert all('expected_benefits' in rec for rec in recommendations)
        assert all('implementation_complexity' in rec for rec in recommendations)
        
        # Check recommendations are sorted by suitability
        suitability_scores = [rec['suitability_score'] for rec in recommendations]
        assert suitability_scores == sorted(suitability_scores, reverse=True)
    
    def test_get_reform_recommendations_excludes_existing(self):
        """Test that recommendations exclude already implemented reforms."""
        # Implement a reform
        self.manager.implement_reform(
            venue_id=self.venue_id,
            reform_type=ReformType.ALTERNATIVE_METRICS
        )
        
        # Get recommendations
        recommendations = self.manager.get_reform_recommendations(
            venue_id=self.venue_id,
            venue_type=VenueType.TOP_CONFERENCE
        )
        
        # Check that implemented reform is not recommended
        reform_types = [rec['reform_type'] for rec in recommendations]
        assert ReformType.ALTERNATIVE_METRICS.value not in reform_types
    
    def test_normalize_metric_value(self):
        """Test metric value normalization."""
        # Test different metric types
        test_cases = [
            (MetricType.ALTMETRIC_SCORE, 50, 5.0),  # 50/100 * 10 = 5.0
            (MetricType.DOWNLOAD_COUNT, 5000, 5.0),  # 5000/10000 * 10 = 5.0
            (MetricType.REPLICATION_SUCCESS_RATE, 0.8, 8.0),  # 0.8/1 * 10 = 8.0
            (MetricType.PEER_REVIEW_QUALITY, 7.5, 7.5),  # Already 0-10 scale
        ]
        
        for metric_type, value, expected in test_cases:
            normalized = self.manager._normalize_metric_value(metric_type, value)
            assert abs(normalized - expected) < 0.1
    
    def test_calculate_trend(self):
        """Test trend calculation."""
        # Test increasing trend
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        trend = self.manager._calculate_trend(increasing_values)
        assert trend == "increasing"
        
        # Test decreasing trend
        decreasing_values = [5.0, 4.0, 3.0, 2.0, 1.0]
        trend = self.manager._calculate_trend(decreasing_values)
        assert trend == "decreasing"
        
        # Test stable trend
        stable_values = [3.0, 3.1, 2.9, 3.0, 3.1]
        trend = self.manager._calculate_trend(stable_values)
        assert trend == "stable"
        
        # Test insufficient data
        insufficient_values = [1.0]
        trend = self.manager._calculate_trend(insufficient_values)
        assert trend == "insufficient_data"
    
    def test_save_and_load_data(self):
        """Test saving and loading reform data."""
        # Create some test data
        reform_id = self.manager.implement_reform(
            venue_id=self.venue_id,
            reform_type=ReformType.ALTERNATIVE_METRICS
        )
        
        model_id = self.manager.create_evaluation_model(
            name="Test Model",
            description="Test model for save/load",
            included_metrics=[MetricType.ALTMETRIC_SCORE]
        )
        
        # Save data
        self.manager.save_data()
        
        # Create new manager and load data
        new_manager = PublicationReformManager(data_dir=self.temp_dir)
        
        # Check data was loaded correctly
        assert reform_id in new_manager.reforms
        assert model_id in new_manager.evaluation_models
        assert len(new_manager.adoption_history) > 0
        
        # Check reform data
        reform = new_manager.reforms[reform_id]
        assert reform.venue_id == self.venue_id
        assert reform.reform_type == ReformType.ALTERNATIVE_METRICS
        
        # Check model data
        model = new_manager.evaluation_models[model_id]
        assert model.name == "Test Model"
        assert MetricType.ALTMETRIC_SCORE in model.included_metrics
    
    def test_alternative_metric_validation(self):
        """Test AlternativeMetric validation."""
        # Valid metric
        metric = AlternativeMetric(
            metric_type=MetricType.ALTMETRIC_SCORE,
            value=85.5,
            confidence=0.9
        )
        assert metric.confidence == 0.9
        
        # Invalid confidence
        with pytest.raises(ValidationError):
            AlternativeMetric(
                metric_type=MetricType.ALTMETRIC_SCORE,
                value=85.5,
                confidence=1.5
            )
    
    def test_post_publication_review_validation(self):
        """Test PostPublicationReview validation."""
        # Valid review
        review = PostPublicationReview(
            paper_id=self.paper_id,
            reviewer_id=self.reviewer_id,
            rating=8.5
        )
        assert review.rating == 8.5
        
        # Invalid rating
        with pytest.raises(ValidationError):
            PostPublicationReview(
                paper_id=self.paper_id,
                reviewer_id=self.reviewer_id,
                rating=11.0
            )
    
    def test_evaluation_model_validation(self):
        """Test EvaluationModel validation."""
        # Valid model
        model = EvaluationModel(
            name="Test Model",
            traditional_weight=0.5,
            alternative_metrics_weight=0.3,
            post_pub_review_weight=0.2
        )
        assert abs(model.traditional_weight + model.alternative_metrics_weight + 
                  model.post_pub_review_weight - 1.0) < 0.01
        
        # Invalid weights (don't sum to 1.0)
        with pytest.raises(ValidationError):
            EvaluationModel(
                name="Invalid Model",
                traditional_weight=0.6,
                alternative_metrics_weight=0.6,
                post_pub_review_weight=0.2
            )
    
    def test_reform_implementation_validation(self):
        """Test ReformImplementation validation."""
        # Valid implementation
        reform = ReformImplementation(
            reform_type=ReformType.ALTERNATIVE_METRICS,
            venue_id=self.venue_id,
            implementation_date=date.today(),
            adoption_rate=0.5,
            researcher_satisfaction=0.7
        )
        assert reform.adoption_rate == 0.5
        assert reform.researcher_satisfaction == 0.7
        
        # Invalid adoption rate
        with pytest.raises(ValidationError):
            ReformImplementation(
                reform_type=ReformType.ALTERNATIVE_METRICS,
                venue_id=self.venue_id,
                implementation_date=date.today(),
                adoption_rate=1.5  # Invalid > 1.0
            )


if __name__ == "__main__":
    pytest.main([__file__])