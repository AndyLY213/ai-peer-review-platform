"""
Unit tests for Enhanced Researcher Agent Integration

This module tests the EnhancedResearcherAgent class and its integration with all
enhancement systems including biases, networks, career progression, and funding.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta, date
import json
import uuid

from src.agents.enhanced_researcher_agent import EnhancedResearcherAgent
from src.data.enhanced_models import (
    EnhancedResearcher, StructuredReview, EnhancedReviewCriteria,
    ResearcherLevel, VenueType, ReviewDecision, CareerStage, FundingStatus,
    ReviewBehaviorProfile, StrategicBehaviorProfile, BiasEffect,
    DetailedStrength, DetailedWeakness, ReviewQualityMetric,
    PublicationRecord, CareerMilestone, TenureTimeline
)
from src.core.exceptions import ValidationError, SimulationError


class TestEnhancedResearcherAgent(unittest.TestCase):
    """Test cases for EnhancedResearcherAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_paper_db = Mock()
        self.mock_token_system = Mock()
        self.mock_simulation_coordinator = Mock()
        
        # Create enhanced profile
        self.enhanced_profile = EnhancedResearcher(
            id="test_researcher",
            name="Dr. Test Researcher",
            specialty="Machine Learning",
            level=ResearcherLevel.ASSISTANT_PROF,
            institution_name="Test University",
            institution_tier=2,
            h_index=10,
            total_citations=150,
            years_active=5,
            cognitive_biases={
                'confirmation': 0.3,
                'halo': 0.2,
                'anchoring': 0.4,
                'availability': 0.3
            },
            career_stage=CareerStage.EARLY_CAREER,
            funding_status=FundingStatus.ADEQUATELY_FUNDED,
            publication_pressure=0.6
        )
        
        # Create agent
        self.agent = EnhancedResearcherAgent(
            name="Dr. Test Researcher",
            specialty="Machine Learning",
            system_message="Test system message",
            paper_db=self.mock_paper_db,
            token_system=self.mock_token_system,
            enhanced_profile=self.enhanced_profile,
            simulation_coordinator=self.mock_simulation_coordinator,
            llm_config=False  # Disable LLM client for testing
        )
        
        # Mock paper data
        self.mock_paper = {
            'id': 'paper_123',
            'title': 'Test Paper',
            'authors': ['Author One', 'Author Two'],
            'abstract': 'Test abstract',
            'keywords': ['test', 'paper'],
            'field': 'Machine Learning',
            'owner_id': 'other_researcher'
        }
        
        self.mock_paper_db.get_paper.return_value = self.mock_paper
    
    def test_initialization_with_enhanced_profile(self):
        """Test agent initialization with enhanced profile."""
        self.assertEqual(self.agent.name, "Dr. Test Researcher")
        self.assertEqual(self.agent.specialty, "Machine Learning")
        self.assertEqual(self.agent.enhanced_profile.level, ResearcherLevel.ASSISTANT_PROF)
        self.assertEqual(self.agent.enhanced_profile.h_index, 10)
        self.assertEqual(self.agent.enhanced_profile.total_citations, 150)
        self.assertIsNotNone(self.agent.simulation_coordinator)
        
        # Check performance metrics initialization
        self.assertEqual(self.agent.performance_metrics['reviews_completed'], 0)
        self.assertEqual(self.agent.performance_metrics['papers_published'], 0)
        self.assertEqual(self.agent.performance_metrics['collaborations_formed'], 0)
    
    def test_initialization_with_default_profile(self):
        """Test agent initialization with default enhanced profile."""
        agent = EnhancedResearcherAgent(
            name="Default Researcher",
            specialty="Computer Science",
            system_message="Test message",
            paper_db=self.mock_paper_db,
            token_system=self.mock_token_system,
            llm_config=False  # Disable LLM client for testing
        )
        
        self.assertEqual(agent.enhanced_profile.name, "Default Researcher")
        self.assertEqual(agent.enhanced_profile.specialty, "Computer Science")
        self.assertEqual(agent.enhanced_profile.level, ResearcherLevel.ASSISTANT_PROF)
        self.assertEqual(agent.enhanced_profile.institution_name, "Default University")
    
    def test_get_enhanced_profile(self):
        """Test getting the enhanced profile."""
        profile = self.agent.get_enhanced_profile()
        
        self.assertIsInstance(profile, EnhancedResearcher)
        self.assertEqual(profile.name, "Dr. Test Researcher")
        self.assertEqual(profile.level, ResearcherLevel.ASSISTANT_PROF)
        self.assertEqual(profile.h_index, 10)
    
    def test_update_enhanced_profile(self):
        """Test updating the enhanced profile."""
        updates = {
            'h_index': 15,
            'total_citations': 200,
            'years_active': 6,
            'funding_status': FundingStatus.WELL_FUNDED
        }
        
        result = self.agent.update_enhanced_profile(updates)
        
        self.assertTrue(result)
        self.assertEqual(self.agent.enhanced_profile.h_index, 15)
        self.assertEqual(self.agent.enhanced_profile.total_citations, 200)
        self.assertEqual(self.agent.enhanced_profile.years_active, 6)
        self.assertEqual(self.agent.enhanced_profile.funding_status, FundingStatus.WELL_FUNDED)
        
        # Check that reputation score was recalculated
        self.assertGreater(self.agent.enhanced_profile.reputation_score, 0)
    
    def test_update_enhanced_profile_invalid_field(self):
        """Test updating enhanced profile with invalid field."""
        updates = {
            'invalid_field': 'invalid_value',
            'h_index': 20
        }
        
        result = self.agent.update_enhanced_profile(updates)
        
        # Should still succeed for valid fields
        self.assertTrue(result)
        self.assertEqual(self.agent.enhanced_profile.h_index, 20)
        # Invalid field should be ignored
        self.assertFalse(hasattr(self.agent.enhanced_profile, 'invalid_field'))
    
    @patch('src.agents.enhanced_researcher_agent.StructuredReviewSystem')
    @patch('src.agents.enhanced_researcher_agent.BiasEngine')
    @patch('src.agents.enhanced_researcher_agent.WorkloadTracker')
    def test_generate_enhanced_review_with_systems(self, mock_workload, mock_bias, mock_review_system):
        """Test enhanced review generation with all systems available."""
        # Setup mocks
        mock_review_instance = Mock()
        mock_bias_instance = Mock()
        mock_workload_instance = Mock()
        
        mock_review_system.return_value = mock_review_instance
        mock_bias.return_value = mock_bias_instance
        mock_workload.return_value = mock_workload_instance
        
        # Mock structured review
        mock_structured_review = StructuredReview(
            reviewer_id="Dr. Test Researcher",
            paper_id="paper_123",
            venue_id="venue_456",
            criteria_scores=EnhancedReviewCriteria(
                novelty=7.0, technical_quality=8.0, clarity=6.0,
                significance=7.5, reproducibility=6.5, related_work=7.0
            ),
            confidence_level=4,
            recommendation=ReviewDecision.MINOR_REVISION,
            executive_summary="Test review summary",
            detailed_strengths=[
                DetailedStrength(category="Technical", description="Strong methodology", importance=4)
            ],
            detailed_weaknesses=[
                DetailedWeakness(category="Presentation", description="Unclear figures", severity=2)
            ],
            technical_comments="Technical assessment",
            presentation_comments="Presentation feedback",
            questions_for_authors=["How was the dataset collected?"],
            suggestions_for_improvement=["Improve figure quality"],
            time_spent_minutes=180
        )
        
        mock_review_instance.generate_review.return_value = mock_structured_review
        mock_bias_instance.apply_biases.return_value = mock_structured_review
        mock_workload_instance.check_availability.return_value = (True, "Available", "Available for review")
        
        # Reinitialize agent to pick up mocked systems
        self.agent._initialize_system_integrations()
        
        # Generate review
        result = self.agent.generate_enhanced_review("paper_123", "venue_456")
        
        # Verify result
        self.assertIsInstance(result, StructuredReview)
        self.assertEqual(result.reviewer_id, "Dr. Test Researcher")
        self.assertEqual(result.paper_id, "paper_123")
        self.assertEqual(result.venue_id, "venue_456")
        
        # Verify system calls
        mock_review_instance.generate_review.assert_called_once()
        mock_bias_instance.apply_biases.assert_called_once()
        # Check that workload tracker was called with researcher ID and profile
        self.assertTrue(mock_workload_instance.check_availability.called)
        
        # Check performance metrics updated
        self.assertEqual(self.agent.performance_metrics['reviews_completed'], 1)
    
    def test_generate_enhanced_review_fallback(self):
        """Test enhanced review generation with fallback when systems unavailable."""
        # Mock basic review generation
        basic_review = {
            'summary': 'Basic review summary',
            'strengths': 'Basic strengths',
            'weaknesses': 'Basic weaknesses',
            'technical_correctness': 'Technical assessment',
            'clarity': 'Clarity assessment'
        }
        
        with patch.object(self.agent, 'generate_review', return_value=basic_review):
            result = self.agent.generate_enhanced_review("paper_123", "venue_456")
        
        # Verify fallback structured review
        self.assertIsInstance(result, StructuredReview)
        self.assertEqual(result.reviewer_id, "Dr. Test Researcher")
        self.assertEqual(result.paper_id, "paper_123")
        self.assertEqual(result.venue_id, "venue_456")
        self.assertEqual(result.executive_summary, "Basic review summary")
        self.assertEqual(len(result.detailed_strengths), 1)
        self.assertEqual(len(result.detailed_weaknesses), 1)
    
    def test_generate_enhanced_review_workload_unavailable(self):
        """Test enhanced review generation when reviewer unavailable due to workload."""
        # Mock workload tracker to return unavailable
        mock_workload = Mock()
        mock_workload.check_availability.return_value = (False, "Overloaded", "At capacity")
        self.agent.workload_tracker = mock_workload
        
        with self.assertRaises(ValidationError) as context:
            self.agent.generate_enhanced_review("paper_123", "venue_456")
        
        self.assertIn("not available due to workload", str(context.exception))
    
    def test_generate_enhanced_review_paper_not_found(self):
        """Test enhanced review generation when paper not found."""
        self.mock_paper_db.get_paper.return_value = None
        
        with self.assertRaises(ValidationError) as context:
            self.agent.generate_enhanced_review("nonexistent_paper", "venue_456")
        
        self.assertIn("Paper nonexistent_paper not found", str(context.exception))
    
    def test_execute_strategic_behavior_venue_shopping(self):
        """Test venue shopping strategic behavior."""
        context = {
            'paper_id': 'paper_123',
            'rejection_history': ['top_venue_1', 'mid_venue_2']
        }
        
        # Set high venue shopping tendency
        self.agent.enhanced_profile.strategic_behavior.venue_shopping_tendency = 0.8
        
        with patch.object(self.agent, '_find_lower_tier_venues', return_value=['low_venue_1', 'low_venue_2']):
            result = self.agent.execute_strategic_behavior('venue_shopping', context)
        
        self.assertEqual(result['action'], 'venue_shopping')
        self.assertEqual(result['target_venues'], ['low_venue_1', 'low_venue_2'])
        self.assertEqual(result['shopping_tendency'], 0.8)
        
        # Check behavior tracking
        self.assertEqual(len(self.agent.behavior_history), 1)
        self.assertEqual(self.agent.behavior_history[0]['behavior_type'], 'venue_shopping')
        self.assertEqual(self.agent.performance_metrics['strategic_behaviors_executed'], 1)
    
    def test_execute_strategic_behavior_review_trading(self):
        """Test review trading strategic behavior."""
        context = {
            'potential_partners': ['partner_1', 'partner_2', 'partner_3']
        }
        
        # Set high trading willingness
        self.agent.enhanced_profile.strategic_behavior.review_trading_willingness = 0.6
        
        with patch.object(self.agent, '_evaluate_trading_benefit', return_value=0.7):
            result = self.agent.execute_strategic_behavior('review_trading', context)
        
        self.assertEqual(result['action'], 'review_trading')
        self.assertEqual(result['selected_partners'], ['partner_1', 'partner_2', 'partner_3'])
        self.assertEqual(result['trading_willingness'], 0.6)
    
    def test_execute_strategic_behavior_low_tendency(self):
        """Test strategic behavior with low tendency."""
        context = {
            'paper_id': 'paper_123',
            'rejection_history': ['venue_1']
        }
        
        # Set low venue shopping tendency
        self.agent.enhanced_profile.strategic_behavior.venue_shopping_tendency = 0.2
        
        result = self.agent.execute_strategic_behavior('venue_shopping', context)
        
        self.assertEqual(result['action'], 'no_venue_shopping')
        self.assertIn('Low shopping tendency', result['reason'])
    
    def test_execute_strategic_behavior_invalid_type(self):
        """Test strategic behavior with invalid type."""
        context = {}
        
        with self.assertRaises(ValidationError) as exc_context:
            self.agent.execute_strategic_behavior('invalid_behavior', context)
        
        self.assertIn("Unknown strategic behavior type", str(exc_context.exception))
    
    @patch('src.agents.enhanced_researcher_agent.TenureTrackManager')
    @patch('src.agents.enhanced_researcher_agent.JobMarketSimulator')
    @patch('src.agents.enhanced_researcher_agent.CareerTransitionManager')
    def test_manage_career_progression(self, mock_transition, mock_job_market, mock_tenure):
        """Test career progression management."""
        # Setup mocks
        mock_tenure_instance = Mock()
        mock_job_market_instance = Mock()
        mock_transition_instance = Mock()
        
        mock_tenure.return_value = mock_tenure_instance
        mock_job_market.return_value = mock_job_market_instance
        mock_transition.return_value = mock_transition_instance
        
        # Mock return values
        mock_tenure_instance.evaluate_tenure_progress.return_value = {
            'progress': 0.6,
            'years_remaining': 2.4,
            'on_track': True
        }
        mock_job_market_instance.evaluate_market_position.return_value = {
            'competitiveness': 0.7,
            'market_rank': 'strong'
        }
        mock_transition_instance.evaluate_transition_opportunities.return_value = {
            'industry_opportunities': 3,
            'academic_opportunities': 2
        }
        
        # Reinitialize agent to pick up mocked systems
        self.agent._initialize_system_integrations()
        
        # Test career progression
        result = self.agent.manage_career_progression()
        
        # Verify results
        self.assertIn('tenure_status', result)
        self.assertIn('job_market_position', result)
        self.assertIn('transition_opportunities', result)
        
        self.assertEqual(result['tenure_status']['progress'], 0.6)
        self.assertEqual(result['job_market_position']['competitiveness'], 0.7)
        self.assertEqual(result['transition_opportunities']['industry_opportunities'], 3)
        
        # Verify system calls
        mock_tenure_instance.evaluate_tenure_progress.assert_called_once_with("Dr. Test Researcher")
        mock_job_market_instance.evaluate_market_position.assert_called_once_with("Dr. Test Researcher")
        mock_transition_instance.evaluate_transition_opportunities.assert_called_once_with("Dr. Test Researcher")
    
    @patch('src.agents.enhanced_researcher_agent.FundingSystem')
    def test_manage_funding_lifecycle(self, mock_funding_system):
        """Test funding lifecycle management."""
        # Setup mock
        mock_funding_instance = Mock()
        mock_funding_system.return_value = mock_funding_instance
        
        # Mock return values
        mock_funding_instance.evaluate_funding_status.return_value = {
            'status': FundingStatus.WELL_FUNDED,
            'amount': 500000,
            'duration': 3,
            'newly_funded': True
        }
        mock_funding_instance.calculate_publication_pressure.return_value = 0.4
        mock_funding_instance.evaluate_resource_constraints.return_value = {
            'equipment_access': 0.9,
            'student_funding': 0.8,
            'travel_budget': 0.7
        }
        
        # Reinitialize agent to pick up mocked system
        self.agent._initialize_system_integrations()
        
        # Test funding management
        result = self.agent.manage_funding_lifecycle()
        
        # Verify results
        self.assertIn('current_status', result)
        self.assertIn('publication_pressure', result)
        self.assertIn('resource_constraints', result)
        
        self.assertEqual(result['current_status']['status'], FundingStatus.WELL_FUNDED)
        self.assertEqual(result['publication_pressure'], 0.4)
        self.assertEqual(result['resource_constraints']['equipment_access'], 0.9)
        
        # Check profile updates
        self.assertEqual(self.agent.enhanced_profile.funding_status, FundingStatus.WELL_FUNDED)
        self.assertEqual(self.agent.enhanced_profile.publication_pressure, 0.4)
        
        # Check performance metrics
        self.assertEqual(self.agent.performance_metrics['funding_secured'], 1)
    
    @patch('src.agents.enhanced_researcher_agent.CollaborationNetwork')
    @patch('src.agents.enhanced_researcher_agent.CitationNetwork')
    def test_update_network_relationships_collaboration(self, mock_citation, mock_collaboration):
        """Test updating collaboration network relationships."""
        # Setup mocks
        mock_collab_instance = Mock()
        mock_citation_instance = Mock()
        
        mock_collaboration.return_value = mock_collab_instance
        mock_citation.return_value = mock_citation_instance
        
        # Reinitialize agent to pick up mocked systems
        self.agent._initialize_system_integrations()
        
        # Test collaboration update
        target_researchers = ['researcher_1', 'researcher_2']
        result = self.agent.update_network_relationships('collaboration', target_researchers)
        
        # Verify results
        self.assertEqual(result['collaborations_added'], 2)
        
        # Verify system calls
        mock_collab_instance.add_collaboration.assert_any_call("Dr. Test Researcher", "researcher_1")
        mock_collab_instance.add_collaboration.assert_any_call("Dr. Test Researcher", "researcher_2")
        
        # Check profile updates
        self.assertIn('researcher_1', self.agent.enhanced_profile.collaboration_network)
        self.assertIn('researcher_2', self.agent.enhanced_profile.collaboration_network)
        
        # Check performance metrics
        self.assertEqual(self.agent.performance_metrics['collaborations_formed'], 2)
        
        # Check interaction tracking
        self.assertEqual(len(self.agent.network_interactions), 1)
        self.assertEqual(self.agent.network_interactions[0]['interaction_type'], 'collaboration')
    
    def test_update_network_relationships_citation(self):
        """Test updating citation network relationships."""
        # Mock citation network
        mock_citation = Mock()
        self.agent.citation_network = mock_citation
        
        target_researchers = ['cited_researcher_1']
        result = self.agent.update_network_relationships('citation', target_researchers)
        
        # Verify results
        self.assertEqual(result['citations_added'], 1)
        
        # Verify system calls
        mock_citation.add_citation_relationship.assert_called_once_with("Dr. Test Researcher", "cited_researcher_1")
        
        # Check profile updates
        self.assertIn('cited_researcher_1', self.agent.enhanced_profile.citation_network)
    
    def test_get_comprehensive_status(self):
        """Test getting comprehensive agent status."""
        # Add some test data
        self.agent.performance_metrics['reviews_completed'] = 5
        self.agent.performance_metrics['papers_published'] = 2
        self.agent.enhanced_profile.collaboration_network.add('collaborator_1')
        self.agent.enhanced_profile.citation_network.add('cited_researcher_1')
        
        status = self.agent.get_comprehensive_status()
        
        # Verify basic info
        self.assertEqual(status['basic_info']['name'], "Dr. Test Researcher")
        self.assertEqual(status['basic_info']['specialty'], "Machine Learning")
        self.assertEqual(status['basic_info']['level'], "Assistant Prof")
        
        # Verify enhanced profile
        self.assertIn('enhanced_profile', status)
        self.assertIsInstance(status['enhanced_profile'], dict)
        
        # Verify performance metrics
        self.assertEqual(status['performance_metrics']['reviews_completed'], 5)
        self.assertEqual(status['performance_metrics']['papers_published'], 2)
        
        # Verify workload status
        self.assertIn('current_workload', status)
        self.assertEqual(status['current_workload']['max_reviews'], 4)  # Assistant Prof default
        
        # Verify network status
        self.assertEqual(status['network_status']['collaborations'], 1)
        self.assertEqual(status['network_status']['citations'], 1)
        
        # Verify behavior tracking
        self.assertIn('behavior_tracking', status)
        self.assertIn('strategic_tendency', status['behavior_tracking'])
        
        # Verify system integrations
        self.assertIn('system_integrations', status)
        self.assertIn('career_systems', status['system_integrations'])
    
    def test_track_review_quality(self):
        """Test review quality tracking."""
        # Create a test review with content to generate quality score
        review = StructuredReview(
            reviewer_id="Dr. Test Researcher",
            paper_id="paper_123",
            venue_id="venue_456",
            executive_summary="Test summary with sufficient content",
            detailed_strengths=[
                DetailedStrength(category="Technical", description="Good methodology", importance=4),
                DetailedStrength(category="Novelty", description="Novel approach", importance=3)
            ],
            detailed_weaknesses=[
                DetailedWeakness(category="Presentation", description="Minor issues", severity=2)
            ],
            technical_comments="Technical assessment with content",
            is_late=False
        )
        
        # Track review quality
        self.agent._track_review_quality(review)
        
        # Verify quality metric added
        self.assertEqual(len(self.agent.enhanced_profile.review_quality_history), 1)
        
        quality_metric = self.agent.enhanced_profile.review_quality_history[0]
        self.assertEqual(quality_metric.review_id, review.review_id)
        self.assertGreater(quality_metric.quality_score, 0.0)  # Should be calculated based on content
        self.assertEqual(quality_metric.timeliness_score, 1.0)  # Not late
    
    def test_track_review_quality_late_submission(self):
        """Test review quality tracking for late submission."""
        # Create a late review
        review = StructuredReview(
            reviewer_id="Dr. Test Researcher",
            paper_id="paper_123",
            venue_id="venue_456",
            quality_score=0.7,
            is_late=True
        )
        
        # Track review quality
        self.agent._track_review_quality(review)
        
        # Verify timeliness penalty
        quality_metric = self.agent.enhanced_profile.review_quality_history[0]
        self.assertEqual(quality_metric.timeliness_score, 0.5)  # Penalty for late submission
    
    def test_track_review_quality_history_limit(self):
        """Test review quality history limit."""
        # Add 55 reviews (more than the 50 limit)
        for i in range(55):
            review = StructuredReview(
                reviewer_id="Dr. Test Researcher",
                paper_id=f"paper_{i}",
                venue_id="venue_456",
                quality_score=0.8
            )
            self.agent._track_review_quality(review)
        
        # Verify history is limited to 50
        self.assertEqual(len(self.agent.enhanced_profile.review_quality_history), 50)
        
        # Verify it kept the most recent ones
        last_metric = self.agent.enhanced_profile.review_quality_history[-1]
        self.assertEqual(last_metric.review_id, review.review_id)
    
    def test_helper_methods(self):
        """Test helper methods for strategic behaviors."""
        # Test venue shopping helper
        with patch.object(self.agent, 'venue_registry') as mock_registry:
            mock_venue = Mock()
            mock_venue.acceptance_rate = 0.2
            mock_venue.field = "Machine Learning"
            
            mock_registry.get_venue.return_value = mock_venue
            mock_registry.get_venues_by_field.return_value = [
                Mock(id="venue_1", acceptance_rate=0.3),
                Mock(id="venue_2", acceptance_rate=0.4),
                Mock(id="venue_3", acceptance_rate=0.5),
                Mock(id="venue_4", acceptance_rate=0.6)
            ]
            
            lower_venues = self.agent._find_lower_tier_venues("rejected_venue")
            self.assertEqual(len(lower_venues), 3)  # Returns top 3
        
        # Test trading benefit evaluation
        with patch.object(self.agent, 'reputation_calculator') as mock_calc:
            mock_calc.get_reputation_score.return_value = 0.8
            
            benefit = self.agent._evaluate_trading_benefit("partner_123")
            self.assertIsInstance(benefit, float)
            self.assertGreaterEqual(benefit, 0.0)
            self.assertLessEqual(benefit, 1.0)
        
        # Test citation benefit evaluation
        benefit = self.agent._evaluate_citation_benefit("citation_target")
        self.assertEqual(benefit, 0.7)  # Placeholder value
        
        # Test collaboration value evaluation
        with patch.object(self.agent, 'reputation_calculator') as mock_calc:
            mock_calc.get_reputation_score.return_value = 0.9
            
            value = self.agent._evaluate_collaboration_value("collaborator_123")
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
    
    def test_update_career_milestones(self):
        """Test career milestone updates."""
        progression_results = {
            'tenure_status': {
                'achieved': True,
                'date': datetime.now().date()
            }
        }
        
        initial_milestones = len(self.agent.enhanced_profile.career_milestones)
        
        self.agent._update_career_milestones(progression_results)
        
        # Verify milestone added
        self.assertEqual(len(self.agent.enhanced_profile.career_milestones), initial_milestones + 1)
        
        new_milestone = self.agent.enhanced_profile.career_milestones[-1]
        self.assertEqual(new_milestone.milestone_type, "tenure")
        self.assertEqual(new_milestone.description, "Achieved tenure")
        
        # Verify performance metrics updated
        self.assertEqual(self.agent.performance_metrics['career_milestones_achieved'], 1)


if __name__ == '__main__':
    unittest.main()