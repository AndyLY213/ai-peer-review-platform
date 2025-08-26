"""
Unit Tests for Promotion Criteria Evaluator

Tests the promotion criteria evaluation system including teaching/service/research balance,
promotion readiness evaluation with field-specific weights, and promotion timeline tracking.
"""

import pytest
from datetime import date, timedelta

from src.enhancements.promotion_criteria_evaluator import (
    PromotionCriteriaEvaluator, PromotionType, PromotionOutcome, PromotionWeights,
    ResearchCriteria, TeachingCriteria, ServiceCriteria, PromotionRequirements,
    PromotionEvaluation, PromotionTimeline
)
from src.data.enhanced_models import EnhancedResearcher, ResearcherLevel, CareerStage
from src.core.exceptions import ValidationError, CareerSystemError


def test_promotion_weights_valid():
    """Test valid promotion weights."""
    weights = PromotionWeights(research_weight=0.6, teaching_weight=0.25, service_weight=0.15)
    assert weights.research_weight == 0.6
    assert weights.teaching_weight == 0.25
    assert weights.service_weight == 0.15


def test_promotion_weights_invalid():
    """Test invalid weights that don't sum to 1.0."""
    with pytest.raises(ValidationError):
        PromotionWeights(research_weight=0.5, teaching_weight=0.3, service_weight=0.3)


def test_research_criteria_valid():
    """Test valid research criteria."""
    criteria = ResearchCriteria(
        min_publications=15,
        min_first_author_publications=8,
        min_journal_publications=5,
        min_h_index=12,
        min_citations=200,
        min_external_funding=150000,
        required_collaborations=3
    )
    assert criteria.min_publications == 15
    assert criteria.min_external_funding == 150000


def test_research_criteria_invalid_publications():
    """Test invalid negative publications."""
    with pytest.raises(ValidationError):
        ResearchCriteria(
            min_publications=-1,
            min_first_author_publications=8,
            min_journal_publications=5,
            min_h_index=12,
            min_citations=200,
            min_external_funding=150000,
            required_collaborations=3
        )


def test_research_criteria_invalid_funding():
    """Test invalid negative funding."""
    with pytest.raises(ValidationError):
        ResearchCriteria(
            min_publications=15,
            min_first_author_publications=8,
            min_journal_publications=5,
            min_h_index=12,
            min_citations=200,
            min_external_funding=-1000,
            required_collaborations=3
        )


def test_teaching_criteria_valid():
    """Test valid teaching criteria."""
    criteria = TeachingCriteria(
        min_courses_taught=6,
        min_student_evaluation_score=3.5,
        required_course_development=1,
        mentoring_requirements=2
    )
    assert criteria.min_courses_taught == 6
    assert criteria.min_student_evaluation_score == 3.5


def test_teaching_criteria_invalid_score_low():
    """Test invalid low evaluation score."""
    with pytest.raises(ValidationError):
        TeachingCriteria(
            min_courses_taught=6,
            min_student_evaluation_score=0.5,
            required_course_development=1,
            mentoring_requirements=2
        )


def test_teaching_criteria_invalid_score_high():
    """Test invalid high evaluation score."""
    with pytest.raises(ValidationError):
        TeachingCriteria(
            min_courses_taught=6,
            min_student_evaluation_score=6.0,
            required_course_development=1,
            mentoring_requirements=2
        )


def test_service_criteria_valid():
    """Test valid service criteria."""
    criteria = ServiceCriteria(
        min_internal_service_roles=3,
        min_external_service_roles=2,
        editorial_board_experience=False,
        leadership_roles_required=1
    )
    assert criteria.min_internal_service_roles == 3
    assert criteria.min_external_service_roles == 2


def test_service_criteria_invalid_negative():
    """Test invalid negative service roles."""
    with pytest.raises(ValidationError):
        ServiceCriteria(
            min_internal_service_roles=-1,
            min_external_service_roles=2,
            leadership_roles_required=1
        )


def test_promotion_timeline_years_calculation():
    """Test years in rank calculation."""
    start_date = date.today() - timedelta(days=730)  # 2 years ago
    timeline = PromotionTimeline(
        researcher_id="test_researcher",
        current_rank=ResearcherLevel.ASSISTANT_PROF,
        target_rank=ResearcherLevel.ASSOCIATE_PROF,
        rank_start_date=start_date,
        earliest_eligible_date=date.today() + timedelta(days=1460),
        recommended_application_date=date.today() + timedelta(days=1825)
    )
    
    years = timeline.get_years_in_rank()
    assert 1.9 < years < 2.1  # Approximately 2 years


def test_promotion_timeline_eligibility_not_eligible():
    """Test eligibility check when not eligible."""
    timeline = PromotionTimeline(
        researcher_id="test_researcher",
        current_rank=ResearcherLevel.ASSISTANT_PROF,
        target_rank=ResearcherLevel.ASSOCIATE_PROF,
        rank_start_date=date.today(),
        earliest_eligible_date=date.today() + timedelta(days=365),
        recommended_application_date=date.today() + timedelta(days=730)
    )
    assert not timeline.is_eligible_for_promotion()


def test_promotion_timeline_eligibility_eligible():
    """Test eligibility check when eligible."""
    timeline = PromotionTimeline(
        researcher_id="test_researcher",
        current_rank=ResearcherLevel.ASSISTANT_PROF,
        target_rank=ResearcherLevel.ASSOCIATE_PROF,
        rank_start_date=date.today() - timedelta(days=2555),
        earliest_eligible_date=date.today() - timedelta(days=365),
        recommended_application_date=date.today()
    )
    assert timeline.is_eligible_for_promotion()


@pytest.fixture
def sample_researcher():
    """Create a sample researcher for testing."""
    return EnhancedResearcher(
        id="test_researcher_001",
        name="Dr. Test Researcher",
        specialty="computer_science",
        level=ResearcherLevel.ASSISTANT_PROF,
        institution_tier=1,
        h_index=15,
        total_citations=250,
        years_active=6,
        reputation_score=75.0,
        cognitive_biases={},
        review_behavior=None,
        strategic_behavior=None,
        career_stage=CareerStage.EARLY_CAREER,
        funding_status=None,
        publication_pressure=0.5,
        tenure_timeline=None,
        collaboration_network=set(),
        citation_network=set(),
        institutional_affiliations=["Test University"],
        review_quality_history=[],
        publication_history=[
            {"title": f"Paper {i}", "year": 2020 + i % 4, "first_author": i < 10}
            for i in range(18)
        ],
        career_milestones=[]
    )


def test_evaluator_creation():
    """Test promotion criteria evaluator creation."""
    evaluator = PromotionCriteriaEvaluator()
    assert evaluator is not None
    assert len(evaluator.promotion_requirements) > 0


def test_create_promotion_timeline_assistant_to_associate(sample_researcher):
    """Test creating promotion timeline for assistant to associate professor."""
    evaluator = PromotionCriteriaEvaluator()
    timeline = evaluator.create_promotion_timeline(sample_researcher)
    
    assert timeline.researcher_id == sample_researcher.id
    assert timeline.current_rank == ResearcherLevel.ASSISTANT_PROF
    assert timeline.target_rank == ResearcherLevel.ASSOCIATE_PROF
    assert timeline.milestones_pending
    assert len(timeline.milestones_completed) == 0


def test_create_promotion_timeline_emeritus_error():
    """Test error when creating timeline for emeritus researcher."""
    evaluator = PromotionCriteriaEvaluator()
    emeritus_researcher = EnhancedResearcher(
        id="emeritus_001",
        name="Prof. Emeritus",
        specialty="computer_science",
        level=ResearcherLevel.EMERITUS,
        institution_tier=1,
        h_index=50,
        total_citations=2000,
        years_active=40,
        reputation_score=95.0,
        cognitive_biases={},
        review_behavior=None,
        strategic_behavior=None,
        career_stage=CareerStage.SENIOR,
        funding_status=None,
        publication_pressure=0.1,
        tenure_timeline=None,
        collaboration_network=set(),
        citation_network=set(),
        institutional_affiliations=["Test University"],
        review_quality_history=[],
        publication_history=[],
        career_milestones=[]
    )
    
    with pytest.raises(CareerSystemError):
        evaluator.create_promotion_timeline(emeritus_researcher)


def test_evaluate_promotion_readiness_computer_science(sample_researcher):
    """Test promotion evaluation for computer science field."""
    evaluator = PromotionCriteriaEvaluator()
    # Create timeline first
    evaluator.create_promotion_timeline(sample_researcher)
    
    evaluation = evaluator.evaluate_promotion_readiness(sample_researcher, "computer_science")
    
    assert evaluation.researcher_id == sample_researcher.id
    assert evaluation.promotion_type == PromotionType.ASSISTANT_TO_ASSOCIATE
    assert 0 <= evaluation.research_score <= 100
    assert 0 <= evaluation.teaching_score <= 100
    assert 0 <= evaluation.service_score <= 100
    assert 0 <= evaluation.overall_score <= 100
    assert evaluation.recommendation in [outcome for outcome in PromotionOutcome]
    assert isinstance(evaluation.strengths, list)
    assert isinstance(evaluation.weaknesses, list)
    assert isinstance(evaluation.improvement_areas, list)


def test_evaluate_promotion_readiness_biology(sample_researcher):
    """Test promotion evaluation for biology field."""
    evaluator = PromotionCriteriaEvaluator()
    # Modify researcher for biology
    sample_researcher.specialty = "biology"
    
    # Create timeline first
    evaluator.create_promotion_timeline(sample_researcher)
    
    evaluation = evaluator.evaluate_promotion_readiness(sample_researcher, "biology")
    
    assert evaluation.promotion_type == PromotionType.ASSISTANT_TO_ASSOCIATE
    assert evaluation.overall_score > 0  # Should have some score


def test_evaluate_promotion_readiness_invalid_field(sample_researcher):
    """Test error for invalid field."""
    evaluator = PromotionCriteriaEvaluator()
    with pytest.raises(CareerSystemError):
        evaluator.evaluate_promotion_readiness(sample_researcher, "invalid_field")


def test_evaluate_promotion_readiness_full_professor():
    """Test promotion evaluation for full professor (no promotion available)."""
    evaluator = PromotionCriteriaEvaluator()
    full_prof = EnhancedResearcher(
        id="full_prof_001",
        name="Prof. Full",
        specialty="computer_science",
        level=ResearcherLevel.FULL_PROF,
        institution_tier=1,
        h_index=30,
        total_citations=800,
        years_active=15,
        reputation_score=90.0,
        cognitive_biases={},
        review_behavior=None,
        strategic_behavior=None,
        career_stage=CareerStage.MID_CAREER,
        funding_status=None,
        publication_pressure=0.3,
        tenure_timeline=None,
        collaboration_network=set(),
        citation_network=set(),
        institutional_affiliations=["Test University"],
        review_quality_history=[],
        publication_history=[],
        career_milestones=[]
    )
    
    with pytest.raises(CareerSystemError):
        evaluator.evaluate_promotion_readiness(full_prof)


def test_complete_milestone(sample_researcher):
    """Test completing promotion milestones."""
    evaluator = PromotionCriteriaEvaluator()
    timeline = evaluator.create_promotion_timeline(sample_researcher)
    
    # Get a milestone to complete
    if timeline.milestones_pending:
        milestone = timeline.milestones_pending[0]
        
        # Complete the milestone
        result = evaluator.complete_milestone(sample_researcher.id, milestone)
        assert result is True
        
        # Check milestone was moved
        updated_timeline = evaluator.promotion_timelines[sample_researcher.id]
        assert milestone in updated_timeline.milestones_completed
        assert milestone not in updated_timeline.milestones_pending


def test_complete_milestone_invalid_researcher():
    """Test completing milestone for non-existent researcher."""
    evaluator = PromotionCriteriaEvaluator()
    result = evaluator.complete_milestone("invalid_id", "Some milestone")
    assert result is False


def test_set_custom_requirements():
    """Test setting custom promotion requirements."""
    evaluator = PromotionCriteriaEvaluator()
    custom_requirements = PromotionRequirements(
        promotion_type=PromotionType.ASSISTANT_TO_ASSOCIATE,
        field="custom_field",
        research_criteria=ResearchCriteria(
            min_publications=10,
            min_first_author_publications=5,
            min_journal_publications=3,
            min_h_index=8,
            min_citations=100,
            min_external_funding=50000,
            required_collaborations=2
        ),
        teaching_criteria=TeachingCriteria(
            min_courses_taught=4,
            min_student_evaluation_score=3.0,
            required_course_development=1,
            mentoring_requirements=1
        ),
        service_criteria=ServiceCriteria(
            min_internal_service_roles=2,
            min_external_service_roles=1,
            leadership_roles_required=0
        ),
        weights=PromotionWeights(research_weight=0.7, teaching_weight=0.2, service_weight=0.1),
        minimum_years_in_rank=5
    )
    
    evaluator.set_custom_requirements("custom_field", PromotionType.ASSISTANT_TO_ASSOCIATE, custom_requirements)
    
    # Verify requirements were set
    assert "custom_field" in evaluator.promotion_requirements
    assert PromotionType.ASSISTANT_TO_ASSOCIATE in evaluator.promotion_requirements["custom_field"]
    stored_req = evaluator.promotion_requirements["custom_field"][PromotionType.ASSISTANT_TO_ASSOCIATE]
    assert stored_req.research_criteria.min_publications == 10


def test_get_promotion_statistics_empty():
    """Test getting statistics with no evaluations."""
    evaluator = PromotionCriteriaEvaluator()
    stats = evaluator.get_promotion_statistics()
    assert stats['total_evaluations'] == 0


def test_get_promotion_statistics_with_data(sample_researcher):
    """Test getting statistics with evaluation data."""
    evaluator = PromotionCriteriaEvaluator()
    # Create timeline and evaluation
    evaluator.create_promotion_timeline(sample_researcher)
    evaluator.evaluate_promotion_readiness(sample_researcher, "computer_science")
    
    stats = evaluator.get_promotion_statistics()
    
    assert stats['total_evaluations'] == 1
    assert 'outcome_distribution' in stats
    assert 'average_scores' in stats
    assert 'timeline_statistics' in stats
    assert 'approval_rate' in stats
    
    # Check average scores structure
    avg_scores = stats['average_scores']
    assert 'research' in avg_scores
    assert 'teaching' in avg_scores
    assert 'service' in avg_scores
    assert 'overall' in avg_scores


def test_export_evaluation_data(sample_researcher):
    """Test exporting evaluation data."""
    evaluator = PromotionCriteriaEvaluator()
    # Create timeline and evaluation
    evaluator.create_promotion_timeline(sample_researcher)
    evaluator.evaluate_promotion_readiness(sample_researcher, "computer_science")
    
    export_data = evaluator.export_evaluation_data(sample_researcher.id)
    
    assert export_data['researcher_id'] == sample_researcher.id
    assert 'evaluations' in export_data
    assert 'timeline' in export_data
    assert 'export_date' in export_data
    
    # Check evaluation data structure
    evaluations = export_data['evaluations']
    assert len(evaluations) == 1
    
    eval_data = evaluations[0]
    assert 'evaluation_date' in eval_data
    assert 'promotion_type' in eval_data
    assert 'research_score' in eval_data
    assert 'teaching_score' in eval_data
    assert 'service_score' in eval_data
    assert 'overall_score' in eval_data
    assert 'recommendation' in eval_data
    
    # Check timeline data structure
    timeline_data = export_data['timeline']
    assert 'current_rank' in timeline_data
    assert 'target_rank' in timeline_data
    assert 'years_in_rank' in timeline_data
    assert 'is_eligible' in timeline_data


def test_export_evaluation_data_no_data():
    """Test exporting data for researcher with no evaluations."""
    evaluator = PromotionCriteriaEvaluator()
    export_data = evaluator.export_evaluation_data("nonexistent_researcher")
    
    assert export_data['researcher_id'] == "nonexistent_researcher"
    assert export_data['evaluations'] == []
    assert export_data['timeline'] == {}


def test_promotion_evaluation_valid():
    """Test valid promotion evaluation creation."""
    evaluation = PromotionEvaluation(
        researcher_id="test_001",
        promotion_type=PromotionType.ASSISTANT_TO_ASSOCIATE,
        evaluation_date=date.today(),
        research_score=85.0,
        teaching_score=75.0,
        service_score=70.0,
        overall_score=80.0,
        meets_minimum_requirements=True,
        recommendation=PromotionOutcome.APPROVED,
        strengths=["Strong research record", "Good teaching"],
        weaknesses=["Limited service"],
        improvement_areas=["Increase service contributions"],
        timeline_assessment="On track for promotion"
    )
    
    assert evaluation.researcher_id == "test_001"
    assert evaluation.research_score == 85.0
    assert evaluation.recommendation == PromotionOutcome.APPROVED


def test_promotion_evaluation_invalid_score():
    """Test invalid score ranges in evaluation."""
    with pytest.raises(ValidationError):
        PromotionEvaluation(
            researcher_id="test_001",
            promotion_type=PromotionType.ASSISTANT_TO_ASSOCIATE,
            evaluation_date=date.today(),
            research_score=150.0,  # Invalid: > 100
            teaching_score=75.0,
            service_score=70.0,
            overall_score=80.0,
            meets_minimum_requirements=True,
            recommendation=PromotionOutcome.APPROVED,
            strengths=[],
            weaknesses=[],
            improvement_areas=[],
            timeline_assessment=""
        )


def test_promotion_requirements_valid():
    """Test valid promotion requirements creation."""
    requirements = PromotionRequirements(
        promotion_type=PromotionType.ASSISTANT_TO_ASSOCIATE,
        field="computer_science",
        research_criteria=ResearchCriteria(
            min_publications=15,
            min_first_author_publications=8,
            min_journal_publications=5,
            min_h_index=12,
            min_citations=200,
            min_external_funding=150000,
            required_collaborations=3
        ),
        teaching_criteria=TeachingCriteria(
            min_courses_taught=6,
            min_student_evaluation_score=3.5,
            required_course_development=1,
            mentoring_requirements=2
        ),
        service_criteria=ServiceCriteria(
            min_internal_service_roles=3,
            min_external_service_roles=2,
            leadership_roles_required=1
        ),
        weights=PromotionWeights(research_weight=0.6, teaching_weight=0.25, service_weight=0.15),
        minimum_years_in_rank=6
    )
    
    assert requirements.promotion_type == PromotionType.ASSISTANT_TO_ASSOCIATE
    assert requirements.field == "computer_science"
    assert requirements.minimum_years_in_rank == 6


def test_promotion_requirements_invalid_years():
    """Test invalid minimum years in rank."""
    with pytest.raises(ValidationError):
        PromotionRequirements(
            promotion_type=PromotionType.ASSISTANT_TO_ASSOCIATE,
            field="computer_science",
            research_criteria=ResearchCriteria(
                min_publications=15,
                min_first_author_publications=8,
                min_journal_publications=5,
                min_h_index=12,
                min_citations=200,
                min_external_funding=150000,
                required_collaborations=3
            ),
            teaching_criteria=TeachingCriteria(
                min_courses_taught=6,
                min_student_evaluation_score=3.5,
                required_course_development=1,
                mentoring_requirements=2
            ),
            service_criteria=ServiceCriteria(
                min_internal_service_roles=3,
                min_external_service_roles=2,
                leadership_roles_required=1
            ),
            weights=PromotionWeights(research_weight=0.6, teaching_weight=0.25, service_weight=0.15),
            minimum_years_in_rank=-1  # Invalid: negative
        )
    
    def test_invalid_weights_sum(self):
        """Test invalid weights that don't sum to 1.0."""
        with pytest.raises(ValidationError):
            PromotionWeights(research_weight=0.5, teaching_weight=0.3, service_weight=0.3)


class TestResearchCriteria:
    """Test research criteria validation."""
    
    def test_valid_research_criteria(self):
        """Test valid research criteria."""
        criteria = ResearchCriteria(
            min_publications=15,
            min_first_author_publications=8,
            min_journal_publications=5,
            min_h_index=12,
            min_citations=200,
            min_external_funding=150000,
            required_collaborations=3
        )
        assert criteria.min_publications == 15
        assert criteria.min_external_funding == 150000
    
    def test_invalid_negative_publications(self):
        """Test invalid negative publications."""
        with pytest.raises(ValidationError):
            ResearchCriteria(
                min_publications=-1,
                min_first_author_publications=8,
                min_journal_publications=5,
                min_h_index=12,
                min_citations=200,
                min_external_funding=150000,
                required_collaborations=3
            )
    
    def test_invalid_negative_funding(self):
        """Test invalid negative funding."""
        with pytest.raises(ValidationError):
            ResearchCriteria(
                min_publications=15,
                min_first_author_publications=8,
                min_journal_publications=5,
                min_h_index=12,
                min_citations=200,
                min_external_funding=-1000,
                required_collaborations=3
            )


class TestTeachingCriteria:
    """Test teaching criteria validation."""
    
    def test_valid_teaching_criteria(self):
        """Test valid teaching criteria."""
        criteria = TeachingCriteria(
            min_courses_taught=6,
            min_student_evaluation_score=3.5,
            required_course_development=1,
            mentoring_requirements=2
        )
        assert criteria.min_courses_taught == 6
        assert criteria.min_student_evaluation_score == 3.5
    
    def test_invalid_evaluation_score_low(self):
        """Test invalid low evaluation score."""
        with pytest.raises(ValidationError):
            TeachingCriteria(
                min_courses_taught=6,
                min_student_evaluation_score=0.5,
                required_course_development=1,
                mentoring_requirements=2
            )
    
    def test_invalid_evaluation_score_high(self):
        """Test invalid high evaluation score."""
        with pytest.raises(ValidationError):
            TeachingCriteria(
                min_courses_taught=6,
                min_student_evaluation_score=6.0,
                required_course_development=1,
                mentoring_requirements=2
            )


class TestServiceCriteria:
    """Test service criteria validation."""
    
    def test_valid_service_criteria(self):
        """Test valid service criteria."""
        criteria = ServiceCriteria(
            min_internal_service_roles=3,
            min_external_service_roles=2,
            editorial_board_experience=False,
            leadership_roles_required=1
        )
        assert criteria.min_internal_service_roles == 3
        assert criteria.min_external_service_roles == 2
    
    def test_invalid_negative_service_roles(self):
        """Test invalid negative service roles."""
        with pytest.raises(ValidationError):
            ServiceCriteria(
                min_internal_service_roles=-1,
                min_external_service_roles=2,
                leadership_roles_required=1
            )


class TestPromotionTimeline:
    """Test promotion timeline functionality."""
    
    def test_years_in_rank_calculation(self):
        """Test years in rank calculation."""
        start_date = date.today() - timedelta(days=730)  # 2 years ago
        timeline = PromotionTimeline(
            researcher_id="test_researcher",
            current_rank=ResearcherLevel.ASSISTANT_PROF,
            target_rank=ResearcherLevel.ASSOCIATE_PROF,
            rank_start_date=start_date,
            earliest_eligible_date=date.today() + timedelta(days=1460),  # 4 years from start
            recommended_application_date=date.today() + timedelta(days=1825)  # 5 years from start
        )
        
        years = timeline.get_years_in_rank()
        assert 1.9 < years < 2.1  # Approximately 2 years
    
    def test_eligibility_check_not_eligible(self):
        """Test eligibility check when not eligible."""
        timeline = PromotionTimeline(
            researcher_id="test_researcher",
            current_rank=ResearcherLevel.ASSISTANT_PROF,
            target_rank=ResearcherLevel.ASSOCIATE_PROF,
            rank_start_date=date.today(),
            earliest_eligible_date=date.today() + timedelta(days=365),  # 1 year from now
            recommended_application_date=date.today() + timedelta(days=730)  # 2 years from now
        )
        
        assert not timeline.is_eligible_for_promotion()
    
    def test_eligibility_check_eligible(self):
        """Test eligibility check when eligible."""
        timeline = PromotionTimeline(
            researcher_id="test_researcher",
            current_rank=ResearcherLevel.ASSISTANT_PROF,
            target_rank=ResearcherLevel.ASSOCIATE_PROF,
            rank_start_date=date.today() - timedelta(days=2555),  # 7 years ago
            earliest_eligible_date=date.today() - timedelta(days=365),  # 1 year ago
            recommended_application_date=date.today()
        )
        
        assert timeline.is_eligible_for_promotion()


class TestPromotionCriteriaEvaluator:
    """Test the main promotion criteria evaluator."""
    
    @pytest.fixture
    def evaluator(self):
        """Create a promotion criteria evaluator instance."""
        return PromotionCriteriaEvaluator()
    
    @pytest.fixture
    def sample_researcher(self):
        """Create a sample researcher for testing."""
        return EnhancedResearcher(
            id="test_researcher_001",
            name="Dr. Test Researcher",
            specialty="computer_science",
            level=ResearcherLevel.ASSISTANT_PROF,
            institution_tier=1,
            h_index=15,
            total_citations=250,
            years_active=6,
            reputation_score=75.0,
            cognitive_biases={},
            review_behavior=None,
            strategic_behavior=None,
            career_stage=CareerStage.EARLY_CAREER,
            funding_status=None,
            publication_pressure=0.5,
            tenure_timeline=None,
            collaboration_network=set(),
            citation_network=set(),
            institutional_affiliations=["Test University"],
            review_quality_history=[],
            publication_history=[
                {"title": f"Paper {i}", "year": 2020 + i % 4, "first_author": i < 10}
                for i in range(18)
            ],
            career_milestones=[]
        )
    
    def test_create_promotion_timeline_assistant_to_associate(self, evaluator, sample_researcher):
        """Test creating promotion timeline for assistant to associate professor."""
        timeline = evaluator.create_promotion_timeline(sample_researcher)
        
        assert timeline.researcher_id == sample_researcher.id
        assert timeline.current_rank == ResearcherLevel.ASSISTANT_PROF
        assert timeline.target_rank == ResearcherLevel.ASSOCIATE_PROF
        assert timeline.milestones_pending
        assert len(timeline.milestones_completed) == 0
    
    def test_create_promotion_timeline_emeritus_error(self, evaluator):
        """Test error when creating timeline for emeritus researcher."""
        emeritus_researcher = EnhancedResearcher(
            id="emeritus_001",
            name="Prof. Emeritus",
            specialty="computer_science",
            level=ResearcherLevel.EMERITUS,
            institution_tier=1,
            h_index=50,
            total_citations=2000,
            years_active=40,
            reputation_score=95.0,
            cognitive_biases={},
            review_behavior=None,
            strategic_behavior=None,
            career_stage=CareerStage.SENIOR,
            funding_status=None,
            publication_pressure=0.1,
            tenure_timeline=None,
            collaboration_network=set(),
            citation_network=set(),
            institutional_affiliations=["Test University"],
            review_quality_history=[],
            publication_history=[],
            career_milestones=[]
        )
        
        with pytest.raises(CareerSystemError):
            evaluator.create_promotion_timeline(emeritus_researcher)
    
    def test_evaluate_promotion_readiness_computer_science(self, evaluator, sample_researcher):
        """Test promotion evaluation for computer science field."""
        # Create timeline first
        evaluator.create_promotion_timeline(sample_researcher)
        
        evaluation = evaluator.evaluate_promotion_readiness(sample_researcher, "computer_science")
        
        assert evaluation.researcher_id == sample_researcher.id
        assert evaluation.promotion_type == PromotionType.ASSISTANT_TO_ASSOCIATE
        assert 0 <= evaluation.research_score <= 100
        assert 0 <= evaluation.teaching_score <= 100
        assert 0 <= evaluation.service_score <= 100
        assert 0 <= evaluation.overall_score <= 100
        assert evaluation.recommendation in [outcome for outcome in PromotionOutcome]
        assert isinstance(evaluation.strengths, list)
        assert isinstance(evaluation.weaknesses, list)
        assert isinstance(evaluation.improvement_areas, list)
    
    def test_evaluate_promotion_readiness_biology(self, evaluator, sample_researcher):
        """Test promotion evaluation for biology field."""
        # Modify researcher for biology
        sample_researcher.specialty = "biology"
        
        # Create timeline first
        evaluator.create_promotion_timeline(sample_researcher)
        
        evaluation = evaluator.evaluate_promotion_readiness(sample_researcher, "biology")
        
        assert evaluation.promotion_type == PromotionType.ASSISTANT_TO_ASSOCIATE
        assert evaluation.overall_score > 0  # Should have some score
    
    def test_evaluate_promotion_readiness_invalid_field(self, evaluator, sample_researcher):
        """Test error for invalid field."""
        with pytest.raises(CareerSystemError):
            evaluator.evaluate_promotion_readiness(sample_researcher, "invalid_field")
    
    def test_evaluate_promotion_readiness_full_professor(self, evaluator):
        """Test promotion evaluation for full professor (no promotion available)."""
        full_prof = EnhancedResearcher(
            id="full_prof_001",
            name="Prof. Full",
            specialty="computer_science",
            level=ResearcherLevel.FULL_PROF,
            institution_tier=1,
            h_index=30,
            total_citations=800,
            years_active=15,
            reputation_score=90.0,
            cognitive_biases={},
            review_behavior=None,
            strategic_behavior=None,
            career_stage=CareerStage.MID_CAREER,
            funding_status=None,
            publication_pressure=0.3,
            tenure_timeline=None,
            collaboration_network=set(),
            citation_network=set(),
            institutional_affiliations=["Test University"],
            review_quality_history=[],
            publication_history=[],
            career_milestones=[]
        )
        
        with pytest.raises(CareerSystemError):
            evaluator.evaluate_promotion_readiness(full_prof)
    
    def test_complete_milestone(self, evaluator, sample_researcher):
        """Test completing promotion milestones."""
        timeline = evaluator.create_promotion_timeline(sample_researcher)
        
        # Get a milestone to complete
        if timeline.milestones_pending:
            milestone = timeline.milestones_pending[0]
            
            # Complete the milestone
            result = evaluator.complete_milestone(sample_researcher.id, milestone)
            assert result is True
            
            # Check milestone was moved
            updated_timeline = evaluator.promotion_timelines[sample_researcher.id]
            assert milestone in updated_timeline.milestones_completed
            assert milestone not in updated_timeline.milestones_pending
    
    def test_complete_milestone_invalid_researcher(self, evaluator):
        """Test completing milestone for non-existent researcher."""
        result = evaluator.complete_milestone("invalid_id", "Some milestone")
        assert result is False
    
    def test_set_custom_requirements(self, evaluator):
        """Test setting custom promotion requirements."""
        custom_requirements = PromotionRequirements(
            promotion_type=PromotionType.ASSISTANT_TO_ASSOCIATE,
            field="custom_field",
            research_criteria=ResearchCriteria(
                min_publications=10,
                min_first_author_publications=5,
                min_journal_publications=3,
                min_h_index=8,
                min_citations=100,
                min_external_funding=50000,
                required_collaborations=2
            ),
            teaching_criteria=TeachingCriteria(
                min_courses_taught=4,
                min_student_evaluation_score=3.0,
                required_course_development=1,
                mentoring_requirements=1
            ),
            service_criteria=ServiceCriteria(
                min_internal_service_roles=2,
                min_external_service_roles=1,
                leadership_roles_required=0
            ),
            weights=PromotionWeights(research_weight=0.7, teaching_weight=0.2, service_weight=0.1),
            minimum_years_in_rank=5
        )
        
        evaluator.set_custom_requirements("custom_field", PromotionType.ASSISTANT_TO_ASSOCIATE, custom_requirements)
        
        # Verify requirements were set
        assert "custom_field" in evaluator.promotion_requirements
        assert PromotionType.ASSISTANT_TO_ASSOCIATE in evaluator.promotion_requirements["custom_field"]
        stored_req = evaluator.promotion_requirements["custom_field"][PromotionType.ASSISTANT_TO_ASSOCIATE]
        assert stored_req.research_criteria.min_publications == 10
    
    def test_get_promotion_statistics_empty(self, evaluator):
        """Test getting statistics with no evaluations."""
        stats = evaluator.get_promotion_statistics()
        assert stats['total_evaluations'] == 0
    
    def test_get_promotion_statistics_with_data(self, evaluator, sample_researcher):
        """Test getting statistics with evaluation data."""
        # Create timeline and evaluation
        evaluator.create_promotion_timeline(sample_researcher)
        evaluator.evaluate_promotion_readiness(sample_researcher, "computer_science")
        
        stats = evaluator.get_promotion_statistics()
        
        assert stats['total_evaluations'] == 1
        assert 'outcome_distribution' in stats
        assert 'average_scores' in stats
        assert 'timeline_statistics' in stats
        assert 'approval_rate' in stats
        
        # Check average scores structure
        avg_scores = stats['average_scores']
        assert 'research' in avg_scores
        assert 'teaching' in avg_scores
        assert 'service' in avg_scores
        assert 'overall' in avg_scores
    
    def test_export_evaluation_data(self, evaluator, sample_researcher):
        """Test exporting evaluation data."""
        # Create timeline and evaluation
        evaluator.create_promotion_timeline(sample_researcher)
        evaluator.evaluate_promotion_readiness(sample_researcher, "computer_science")
        
        export_data = evaluator.export_evaluation_data(sample_researcher.id)
        
        assert export_data['researcher_id'] == sample_researcher.id
        assert 'evaluations' in export_data
        assert 'timeline' in export_data
        assert 'export_date' in export_data
        
        # Check evaluation data structure
        evaluations = export_data['evaluations']
        assert len(evaluations) == 1
        
        eval_data = evaluations[0]
        assert 'evaluation_date' in eval_data
        assert 'promotion_type' in eval_data
        assert 'research_score' in eval_data
        assert 'teaching_score' in eval_data
        assert 'service_score' in eval_data
        assert 'overall_score' in eval_data
        assert 'recommendation' in eval_data
        
        # Check timeline data structure
        timeline_data = export_data['timeline']
        assert 'current_rank' in timeline_data
        assert 'target_rank' in timeline_data
        assert 'years_in_rank' in timeline_data
        assert 'is_eligible' in timeline_data
    
    def test_export_evaluation_data_no_data(self, evaluator):
        """Test exporting data for researcher with no evaluations."""
        export_data = evaluator.export_evaluation_data("nonexistent_researcher")
        
        assert export_data['researcher_id'] == "nonexistent_researcher"
        assert export_data['evaluations'] == []
        assert export_data['timeline'] == {}


class TestPromotionEvaluation:
    """Test promotion evaluation data class."""
    
    def test_valid_promotion_evaluation(self):
        """Test valid promotion evaluation creation."""
        evaluation = PromotionEvaluation(
            researcher_id="test_001",
            promotion_type=PromotionType.ASSISTANT_TO_ASSOCIATE,
            evaluation_date=date.today(),
            research_score=85.0,
            teaching_score=75.0,
            service_score=70.0,
            overall_score=80.0,
            meets_minimum_requirements=True,
            recommendation=PromotionOutcome.APPROVED,
            strengths=["Strong research record", "Good teaching"],
            weaknesses=["Limited service"],
            improvement_areas=["Increase service contributions"],
            timeline_assessment="On track for promotion"
        )
        
        assert evaluation.researcher_id == "test_001"
        assert evaluation.research_score == 85.0
        assert evaluation.recommendation == PromotionOutcome.APPROVED
    
    def test_invalid_score_range(self):
        """Test invalid score ranges in evaluation."""
        with pytest.raises(ValidationError):
            PromotionEvaluation(
                researcher_id="test_001",
                promotion_type=PromotionType.ASSISTANT_TO_ASSOCIATE,
                evaluation_date=date.today(),
                research_score=150.0,  # Invalid: > 100
                teaching_score=75.0,
                service_score=70.0,
                overall_score=80.0,
                meets_minimum_requirements=True,
                recommendation=PromotionOutcome.APPROVED,
                strengths=[],
                weaknesses=[],
                improvement_areas=[],
                timeline_assessment=""
            )


class TestPromotionRequirements:
    """Test promotion requirements data class."""
    
    def test_valid_promotion_requirements(self):
        """Test valid promotion requirements creation."""
        requirements = PromotionRequirements(
            promotion_type=PromotionType.ASSISTANT_TO_ASSOCIATE,
            field="computer_science",
            research_criteria=ResearchCriteria(
                min_publications=15,
                min_first_author_publications=8,
                min_journal_publications=5,
                min_h_index=12,
                min_citations=200,
                min_external_funding=150000,
                required_collaborations=3
            ),
            teaching_criteria=TeachingCriteria(
                min_courses_taught=6,
                min_student_evaluation_score=3.5,
                required_course_development=1,
                mentoring_requirements=2
            ),
            service_criteria=ServiceCriteria(
                min_internal_service_roles=3,
                min_external_service_roles=2,
                leadership_roles_required=1
            ),
            weights=PromotionWeights(research_weight=0.6, teaching_weight=0.25, service_weight=0.15),
            minimum_years_in_rank=6
        )
        
        assert requirements.promotion_type == PromotionType.ASSISTANT_TO_ASSOCIATE
        assert requirements.field == "computer_science"
        assert requirements.minimum_years_in_rank == 6
    
    def test_invalid_minimum_years(self):
        """Test invalid minimum years in rank."""
        with pytest.raises(ValidationError):
            PromotionRequirements(
                promotion_type=PromotionType.ASSISTANT_TO_ASSOCIATE,
                field="computer_science",
                research_criteria=ResearchCriteria(
                    min_publications=15,
                    min_first_author_publications=8,
                    min_journal_publications=5,
                    min_h_index=12,
                    min_citations=200,
                    min_external_funding=150000,
                    required_collaborations=3
                ),
                teaching_criteria=TeachingCriteria(
                    min_courses_taught=6,
                    min_student_evaluation_score=3.5,
                    required_course_development=1,
                    mentoring_requirements=2
                ),
                service_criteria=ServiceCriteria(
                    min_internal_service_roles=3,
                    min_external_service_roles=2,
                    leadership_roles_required=1
                ),
                weights=PromotionWeights(research_weight=0.6, teaching_weight=0.25, service_weight=0.15),
                minimum_years_in_rank=-1  # Invalid: negative
            )