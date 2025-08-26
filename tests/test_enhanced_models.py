"""
Unit tests for Enhanced Data Model Classes.

This module provides comprehensive unit tests for the enhanced data model classes
including researchers, reviews, venues, and database migration utilities.
"""

import pytest
import json
import tempfile
from datetime import datetime, date
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.enhanced_models import (
    EnhancedResearcher, StructuredReview, EnhancedVenue,
    EnhancedReviewCriteria, DetailedStrength, DetailedWeakness,
    BiasEffect, ReviewBehaviorProfile, StrategicBehaviorProfile,
    CareerMilestone, PublicationRecord, ReviewQualityMetric,
    TenureTimeline, ReviewRequirements, QualityStandards,
    ReviewerCriteria, DatabaseMigrationUtility,
    ResearcherLevel, VenueType, ReviewDecision, CareerStage, FundingStatus
)
from src.core.exceptions import ValidationError


class TestEnhancedReviewCriteria:
    """Test cases for EnhancedReviewCriteria class."""
    
    def test_valid_criteria_creation(self):
        """Test creating criteria with valid scores."""
        criteria = EnhancedReviewCriteria(
            novelty=8.0,
            technical_quality=7.5,
            clarity=6.0,
            significance=9.0,
            reproducibility=5.5,
            related_work=7.0
        )
        
        assert criteria.novelty == 8.0
        assert criteria.technical_quality == 7.5
        assert criteria.clarity == 6.0
        assert criteria.significance == 9.0
        assert criteria.reproducibility == 5.5
        assert criteria.related_work == 7.0
    
    def test_invalid_score_validation(self):
        """Test validation of score ranges."""
        with pytest.raises(ValidationError):
            EnhancedReviewCriteria(novelty=11.0)  # Above max
        
        with pytest.raises(ValidationError):
            EnhancedReviewCriteria(technical_quality=0.5)  # Below min
    
    def test_average_score_calculation(self):
        """Test average score calculation."""
        criteria = EnhancedReviewCriteria(
            novelty=6.0,
            technical_quality=8.0,
            clarity=7.0,
            significance=9.0,
            reproducibility=5.0,
            related_work=7.0
        )
        
        expected_avg = (6.0 + 8.0 + 7.0 + 9.0 + 5.0 + 7.0) / 6.0
        assert criteria.get_average_score() == expected_avg
    
    def test_weighted_score_calculation(self):
        """Test weighted score calculation."""
        criteria = EnhancedReviewCriteria(
            novelty=6.0,
            technical_quality=8.0,
            clarity=7.0,
            significance=9.0,
            reproducibility=5.0,
            related_work=7.0
        )
        
        # Test with default weights
        weighted_score = criteria.get_weighted_score()
        assert isinstance(weighted_score, float)
        assert 1.0 <= weighted_score <= 10.0
        
        # Test with custom weights
        custom_weights = {
            'novelty': 2.0,
            'technical_quality': 1.0,
            'clarity': 1.0,
            'significance': 1.0,
            'reproducibility': 1.0,
            'related_work': 1.0
        }
        custom_weighted = criteria.get_weighted_score(custom_weights)
        assert isinstance(custom_weighted, float)


class TestStructuredReview:
    """Test cases for StructuredReview class."""
    
    def test_basic_review_creation(self):
        """Test creating a basic structured review."""
        review = StructuredReview(
            reviewer_id="reviewer_1",
            paper_id="paper_1",
            venue_id="venue_1"
        )
        
        assert review.reviewer_id == "reviewer_1"
        assert review.paper_id == "paper_1"
        assert review.venue_id == "venue_1"
        assert review.confidence_level == 3  # Default
        assert review.recommendation == ReviewDecision.MAJOR_REVISION  # Default
        assert isinstance(review.criteria_scores, EnhancedReviewCriteria)
    
    def test_confidence_level_validation(self):
        """Test confidence level validation."""
        with pytest.raises(ValidationError):
            StructuredReview(
                reviewer_id="reviewer_1",
                paper_id="paper_1",
                venue_id="venue_1",
                confidence_level=6  # Invalid
            )
    
    def test_review_length_calculation(self):
        """Test review length calculation."""
        review = StructuredReview(
            reviewer_id="reviewer_1",
            paper_id="paper_1",
            venue_id="venue_1",
            executive_summary="This is a summary.",
            technical_comments="Technical details here.",
            presentation_comments="Presentation feedback."
        )
        
        expected_length = len("This is a summary.") + len("Technical details here.") + len("Presentation feedback.")
        assert review.review_length == expected_length
    
    def test_venue_requirements_check(self):
        """Test venue requirements validation."""
        review = StructuredReview(
            reviewer_id="reviewer_1",
            paper_id="paper_1",
            venue_id="venue_1",
            executive_summary="This is a comprehensive summary of the paper.",
            technical_comments="Detailed technical analysis with multiple points.",
            detailed_strengths=[
                DetailedStrength(category="Technical", description="Strong methodology"),
                DetailedStrength(category="Novelty", description="Novel approach")
            ],
            detailed_weaknesses=[
                DetailedWeakness(category="Clarity", description="Some unclear sections")
            ]
        )
        
        # Should meet requirements with sufficient content
        assert review.meets_venue_requirements(min_word_count=5)
        
        # Should not meet requirements with high word count
        assert not review.meets_venue_requirements(min_word_count=100)
    
    def test_serialization_deserialization(self):
        """Test review serialization and deserialization."""
        original_review = StructuredReview(
            reviewer_id="reviewer_1",
            paper_id="paper_1",
            venue_id="venue_1",
            confidence_level=4,
            recommendation=ReviewDecision.ACCEPT,
            executive_summary="Great paper",
            detailed_strengths=[
                DetailedStrength(category="Technical", description="Strong methodology")
            ],
            applied_biases=[
                BiasEffect(bias_type="confirmation", strength=0.3, score_adjustment=0.5)
            ]
        )
        
        # Serialize to dict
        review_dict = original_review.to_dict()
        assert isinstance(review_dict, dict)
        assert review_dict['reviewer_id'] == "reviewer_1"
        assert review_dict['recommendation'] == "Accept"
        
        # Deserialize from dict
        restored_review = StructuredReview.from_dict(review_dict)
        assert restored_review.reviewer_id == original_review.reviewer_id
        assert restored_review.recommendation == original_review.recommendation
        assert len(restored_review.detailed_strengths) == 1
        assert len(restored_review.applied_biases) == 1


class TestEnhancedResearcher:
    """Test cases for EnhancedResearcher class."""
    
    def test_basic_researcher_creation(self):
        """Test creating a basic enhanced researcher."""
        researcher = EnhancedResearcher(
            id="researcher_1",
            name="Dr. Jane Smith",
            specialty="Machine Learning"
        )
        
        assert researcher.id == "researcher_1"
        assert researcher.name == "Dr. Jane Smith"
        assert researcher.specialty == "Machine Learning"
        assert researcher.level == ResearcherLevel.ASSISTANT_PROF  # Default
        assert researcher.reputation_score > 0  # Should be calculated
    
    def test_reputation_score_calculation(self):
        """Test reputation score calculation."""
        researcher = EnhancedResearcher(
            id="researcher_1",
            name="Dr. Jane Smith",
            specialty="Machine Learning",
            h_index=20,
            total_citations=500,
            years_active=10,
            institution_tier=1,
            level=ResearcherLevel.FULL_PROF
        )
        
        # Should have higher reputation due to good metrics
        assert researcher.reputation_score > 0.5
        
        # Test with lower metrics
        low_researcher = EnhancedResearcher(
            id="researcher_2",
            name="Dr. John Doe",
            specialty="AI",
            h_index=5,
            total_citations=50,
            years_active=2,
            institution_tier=3,
            level=ResearcherLevel.GRADUATE_STUDENT
        )
        
        assert low_researcher.reputation_score < researcher.reputation_score
    
    def test_reputation_multiplier(self):
        """Test reputation multiplier calculation."""
        full_prof = EnhancedResearcher(
            id="prof_1",
            name="Prof. Smith",
            specialty="AI",
            level=ResearcherLevel.FULL_PROF
        )
        
        assistant_prof = EnhancedResearcher(
            id="prof_2",
            name="Dr. Jones",
            specialty="AI",
            level=ResearcherLevel.ASSISTANT_PROF
        )
        
        # Full professor should have higher multiplier
        assert full_prof.get_reputation_multiplier() > assistant_prof.get_reputation_multiplier()
        assert full_prof.get_reputation_multiplier() >= 1.5  # As per requirements
    
    def test_max_reviews_per_month(self):
        """Test maximum reviews per month based on seniority."""
        grad_student = EnhancedResearcher(
            id="grad_1",
            name="Student",
            specialty="AI",
            level=ResearcherLevel.GRADUATE_STUDENT
        )
        
        full_prof = EnhancedResearcher(
            id="prof_1",
            name="Professor",
            specialty="AI",
            level=ResearcherLevel.FULL_PROF
        )
        
        assert grad_student.max_reviews_per_month == 2
        assert full_prof.max_reviews_per_month == 8
    
    def test_can_accept_review(self):
        """Test review acceptance capability."""
        researcher = EnhancedResearcher(
            id="researcher_1",
            name="Dr. Smith",
            specialty="AI",
            current_review_load=2,
            max_reviews_per_month=4,
            availability_status=True
        )
        
        assert researcher.can_accept_review()
        
        # Test with full load
        researcher.current_review_load = 4
        assert not researcher.can_accept_review()
        
        # Test with unavailable status
        researcher.current_review_load = 2
        researcher.availability_status = False
        assert not researcher.can_accept_review()
    
    def test_collaboration_network(self):
        """Test collaboration network management."""
        researcher = EnhancedResearcher(
            id="researcher_1",
            name="Dr. Smith",
            specialty="AI"
        )
        
        researcher.add_collaboration("researcher_2")
        researcher.add_collaboration("researcher_3")
        
        assert "researcher_2" in researcher.collaboration_network
        assert "researcher_3" in researcher.collaboration_network
        assert len(researcher.collaboration_network) == 2
    
    def test_publication_history_update(self):
        """Test publication history updates."""
        researcher = EnhancedResearcher(
            id="researcher_1",
            name="Dr. Smith",
            specialty="AI",
            h_index=5,
            total_citations=100
        )
        
        # Add publications
        pub1 = PublicationRecord(
            paper_id="paper_1",
            title="AI Paper 1",
            venue="ICML",
            year=2023,
            citations=50
        )
        
        pub2 = PublicationRecord(
            paper_id="paper_2",
            title="AI Paper 2",
            venue="NeurIPS",
            year=2024,
            citations=30
        )
        
        researcher.update_publication_history(pub1)
        researcher.update_publication_history(pub2)
        
        assert len(researcher.publication_history) == 2
        assert researcher.total_citations >= 180  # Original + new citations
        assert researcher.h_index >= 2  # Should be updated
    
    def test_institution_tier_validation(self):
        """Test institution tier validation."""
        with pytest.raises(ValidationError):
            EnhancedResearcher(
                id="researcher_1",
                name="Dr. Smith",
                specialty="AI",
                institution_tier=4  # Invalid
            )
    
    def test_serialization_deserialization(self):
        """Test researcher serialization and deserialization."""
        original_researcher = EnhancedResearcher(
            id="researcher_1",
            name="Dr. Jane Smith",
            specialty="Machine Learning",
            level=ResearcherLevel.ASSOCIATE_PROF,
            h_index=15,
            collaboration_network={"researcher_2", "researcher_3"}
        )
        
        # Serialize to dict
        researcher_dict = original_researcher.to_dict()
        assert isinstance(researcher_dict, dict)
        assert researcher_dict['name'] == "Dr. Jane Smith"
        assert researcher_dict['level'] == "Associate Prof"
        assert isinstance(researcher_dict['collaboration_network'], list)
        
        # Deserialize from dict
        restored_researcher = EnhancedResearcher.from_dict(researcher_dict)
        assert restored_researcher.name == original_researcher.name
        assert restored_researcher.level == original_researcher.level
        assert restored_researcher.collaboration_network == original_researcher.collaboration_network


class TestEnhancedVenue:
    """Test cases for EnhancedVenue class."""
    
    def test_basic_venue_creation(self):
        """Test creating a basic enhanced venue."""
        venue = EnhancedVenue(
            id="venue_1",
            name="International Conference on AI",
            venue_type=VenueType.TOP_CONFERENCE,
            field="Artificial Intelligence"
        )
        
        assert venue.id == "venue_1"
        assert venue.name == "International Conference on AI"
        assert venue.venue_type == VenueType.TOP_CONFERENCE
        assert venue.field == "Artificial Intelligence"
        assert venue.acceptance_rate == 0.05  # Should be set by type defaults
        assert venue.prestige_score == 9  # Should be set by type defaults
    
    def test_venue_type_defaults(self):
        """Test that venue type defaults are applied correctly."""
        top_conf = EnhancedVenue(
            id="top_conf",
            name="Top Conference",
            venue_type=VenueType.TOP_CONFERENCE,
            field="AI"
        )
        
        mid_conf = EnhancedVenue(
            id="mid_conf",
            name="Mid Conference",
            venue_type=VenueType.MID_CONFERENCE,
            field="AI"
        )
        
        assert top_conf.acceptance_rate < mid_conf.acceptance_rate
        assert top_conf.prestige_score > mid_conf.prestige_score
        assert top_conf.reviewer_selection_criteria.min_h_index > mid_conf.reviewer_selection_criteria.min_h_index
    
    def test_acceptance_rate_validation(self):
        """Test acceptance rate validation."""
        with pytest.raises(ValidationError):
            EnhancedVenue(
                id="venue_1",
                name="Test Venue",
                venue_type=VenueType.MID_CONFERENCE,
                field="AI",
                acceptance_rate=1.5  # Invalid
            )
    
    def test_reviewer_criteria_check(self):
        """Test reviewer criteria checking."""
        venue = EnhancedVenue(
            id="venue_1",
            name="Top Conference",
            venue_type=VenueType.TOP_CONFERENCE,
            field="AI"
        )
        
        # High-quality researcher should meet criteria
        good_researcher = EnhancedResearcher(
            id="researcher_1",
            name="Dr. Smith",
            specialty="AI",
            level=ResearcherLevel.FULL_PROF,
            h_index=20,
            years_active=10,
            institution_tier=1,
            reputation_score=0.8
        )
        
        assert venue.meets_reviewer_criteria(good_researcher)
        
        # Low-quality researcher should not meet criteria
        poor_researcher = EnhancedResearcher(
            id="researcher_2",
            name="Student",
            specialty="AI",
            level=ResearcherLevel.GRADUATE_STUDENT,
            h_index=2,
            years_active=1,
            institution_tier=3,
            reputation_score=0.1
        )
        
        assert not venue.meets_reviewer_criteria(poor_researcher)
    
    def test_acceptance_probability_calculation(self):
        """Test acceptance probability calculation."""
        venue = EnhancedVenue(
            id="venue_1",
            name="Test Venue",
            venue_type=VenueType.MID_CONFERENCE,
            field="AI"
        )
        
        # High scores should have high acceptance probability
        high_scores = [8.0, 8.5, 9.0]
        high_prob = venue.calculate_acceptance_probability(high_scores)
        assert high_prob > 0.7
        
        # Low scores should have low acceptance probability
        low_scores = [3.0, 4.0, 3.5]
        low_prob = venue.calculate_acceptance_probability(low_scores)
        assert low_prob < 0.3
        
        # Empty scores should return 0
        assert venue.calculate_acceptance_probability([]) == 0.0
    
    def test_submission_record_tracking(self):
        """Test submission record tracking."""
        venue = EnhancedVenue(
            id="venue_1",
            name="Test Venue",
            venue_type=VenueType.MID_CONFERENCE,
            field="AI"
        )
        
        # Add submission records
        venue.add_submission_record("paper_1", True, [8.0, 7.5, 8.5])
        venue.add_submission_record("paper_2", False, [4.0, 5.0, 4.5])
        
        assert len(venue.submission_history) == 2
        assert venue.submission_history[0]['accepted'] == True
        assert venue.submission_history[1]['accepted'] == False
        
        # Check that acceptance trends are updated
        assert len(venue.acceptance_trends) > 0
    
    def test_serialization_deserialization(self):
        """Test venue serialization and deserialization."""
        original_venue = EnhancedVenue(
            id="venue_1",
            name="Test Conference",
            venue_type=VenueType.TOP_CONFERENCE,
            field="AI",
            reviewer_pool={"reviewer_1", "reviewer_2"}
        )
        
        # Serialize to dict
        venue_dict = original_venue.to_dict()
        assert isinstance(venue_dict, dict)
        assert venue_dict['name'] == "Test Conference"
        assert venue_dict['venue_type'] == "Top Conference"
        assert isinstance(venue_dict['reviewer_pool'], list)
        
        # Deserialize from dict
        restored_venue = EnhancedVenue.from_dict(venue_dict)
        assert restored_venue.name == original_venue.name
        assert restored_venue.venue_type == original_venue.venue_type
        assert restored_venue.reviewer_pool == original_venue.reviewer_pool


class TestDatabaseMigrationUtility:
    """Test cases for DatabaseMigrationUtility class."""
    
    def test_migration_utility_initialization(self):
        """Test migration utility initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            utility = DatabaseMigrationUtility(temp_dir)
            assert utility.data_directory == Path(temp_dir)
            assert utility.backup_directory.exists()
    
    def test_researcher_migration(self):
        """Test researcher data migration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            utility = DatabaseMigrationUtility(temp_dir)
            
            # Create old format data
            old_data = {
                "researchers": {
                    "researcher_1": {
                        "name": "Dr. Smith",
                        "specialty": "AI",
                        "level": "Full Prof",
                        "h_index": 20,
                        "citations": 500,
                        "years_active": 15
                    }
                }
            }
            
            old_file = Path(temp_dir) / "old_researchers.json"
            with open(old_file, 'w') as f:
                json.dump(old_data, f)
            
            new_file = Path(temp_dir) / "new_researchers.json"
            
            # Perform migration
            utility.migrate_researchers(str(old_file), str(new_file))
            
            # Verify migration
            assert new_file.exists()
            with open(new_file, 'r') as f:
                migrated_data = json.load(f)
            
            assert "researchers" in migrated_data
            assert "researcher_1" in migrated_data["researchers"]
            
            researcher_data = migrated_data["researchers"]["researcher_1"]
            assert researcher_data["name"] == "Dr. Smith"
            assert researcher_data["level"] == "Full Prof"
            assert researcher_data["h_index"] == 20
    
    def test_paper_migration(self):
        """Test paper data migration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            utility = DatabaseMigrationUtility(temp_dir)
            
            # Create old format data
            old_data = {
                "papers": {
                    "paper_1": {
                        "title": "AI Paper",
                        "authors": ["Dr. Smith"],
                        "reviews": [
                            {
                                "reviewer_id": "reviewer_1",
                                "rating": 7.5,
                                "text": "Good paper with strengths and weaknesses",
                                "confidence": 4
                            }
                        ]
                    }
                }
            }
            
            old_file = Path(temp_dir) / "old_papers.json"
            with open(old_file, 'w') as f:
                json.dump(old_data, f)
            
            new_file = Path(temp_dir) / "new_papers.json"
            
            # Perform migration
            utility.migrate_papers(str(old_file), str(new_file))
            
            # Verify migration
            assert new_file.exists()
            with open(new_file, 'r') as f:
                migrated_data = json.load(f)
            
            assert "papers" in migrated_data
            assert "paper_1" in migrated_data["papers"]
            
            paper_data = migrated_data["papers"]["paper_1"]
            assert len(paper_data["reviews"]) == 1
            
            review_data = paper_data["reviews"][0]
            assert "criteria_scores" in review_data
            assert "executive_summary" in review_data
    
    @patch('shutil.copy2')
    def test_backup_creation(self, mock_copy):
        """Test that backups are created during migration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            utility = DatabaseMigrationUtility(temp_dir)
            
            # Create a file to migrate
            old_file = Path(temp_dir) / "old_data.json"
            old_file.write_text('{"test": "data"}')
            
            new_file = Path(temp_dir) / "new_data.json"
            
            # Perform migration
            utility.migrate_researchers(str(old_file), str(new_file))
            
            # Verify backup was attempted
            mock_copy.assert_called_once()


class TestDataModelIntegration:
    """Integration tests for data model interactions."""
    
    def test_researcher_review_integration(self):
        """Test integration between researcher and review models."""
        researcher = EnhancedResearcher(
            id="researcher_1",
            name="Dr. Smith",
            specialty="AI",
            level=ResearcherLevel.ASSOCIATE_PROF
        )
        
        review = StructuredReview(
            reviewer_id=researcher.id,
            paper_id="paper_1",
            venue_id="venue_1",
            confidence_level=4,
            executive_summary="This paper presents interesting work."
        )
        
        # Test that review can be linked to researcher
        assert review.reviewer_id == researcher.id
        
        # Test that researcher can accept the review based on workload
        assert researcher.can_accept_review()
    
    def test_venue_researcher_compatibility(self):
        """Test compatibility between venue requirements and researcher qualifications."""
        top_venue = EnhancedVenue(
            id="top_venue",
            name="Top Conference",
            venue_type=VenueType.TOP_CONFERENCE,
            field="AI"
        )
        
        qualified_researcher = EnhancedResearcher(
            id="researcher_1",
            name="Dr. Smith",
            specialty="AI",
            level=ResearcherLevel.FULL_PROF,
            h_index=25,
            years_active=15,
            institution_tier=1
        )
        
        unqualified_researcher = EnhancedResearcher(
            id="researcher_2",
            name="Student",
            specialty="AI",
            level=ResearcherLevel.GRADUATE_STUDENT,
            h_index=3,
            years_active=2,
            institution_tier=3
        )
        
        assert top_venue.meets_reviewer_criteria(qualified_researcher)
        assert not top_venue.meets_reviewer_criteria(unqualified_researcher)
    
    def test_complete_review_workflow(self):
        """Test a complete review workflow with all models."""
        # Create venue
        venue = EnhancedVenue(
            id="venue_1",
            name="AI Conference",
            venue_type=VenueType.MID_CONFERENCE,
            field="AI"
        )
        
        # Create researcher
        researcher = EnhancedResearcher(
            id="researcher_1",
            name="Dr. Smith",
            specialty="AI",
            level=ResearcherLevel.ASSOCIATE_PROF,
            h_index=12,
            years_active=8
        )
        
        # Verify researcher can review for venue
        assert venue.meets_reviewer_criteria(researcher)
        assert researcher.can_accept_review()
        
        # Create review
        review = StructuredReview(
            reviewer_id=researcher.id,
            paper_id="paper_1",
            venue_id=venue.id,
            confidence_level=4,
            recommendation=ReviewDecision.ACCEPT,
            executive_summary="Excellent paper with novel contributions.",
            detailed_strengths=[
                DetailedStrength(category="Novelty", description="Novel approach to the problem"),
                DetailedStrength(category="Technical", description="Sound methodology")
            ],
            detailed_weaknesses=[
                DetailedWeakness(category="Clarity", description="Some sections could be clearer")
            ]
        )
        
        # Verify review meets venue requirements
        min_words = venue.review_requirements.min_word_count
        assert review.meets_venue_requirements(min_words)
        
        # Test acceptance probability
        scores = [review.criteria_scores.get_average_score()]
        prob = venue.calculate_acceptance_probability(scores)
        assert 0.0 <= prob <= 1.0