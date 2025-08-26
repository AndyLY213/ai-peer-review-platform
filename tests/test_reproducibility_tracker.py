"""
Unit tests for ReproducibilityTracker class.

Tests replication attempt tracking, questionable research practices modeling,
and reproducibility scoring effects on paper quality.
"""

import pytest
import tempfile
import shutil
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.enhancements.reproducibility_tracker import (
    ReproducibilityTracker,
    ReplicationAttempt,
    QuestionablePracticeIncident,
    ReproducibilityScore,
    ReproducibilityCrisisMetrics,
    ReplicationOutcome,
    QuestionablePractice
)
from src.core.exceptions import ValidationError, PeerReviewError


class TestReproducibilityTracker:
    """Test cases for ReproducibilityTracker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ReproducibilityTracker(data_dir=self.temp_dir)
        
        # Test data
        self.paper_id = "paper_123"
        self.researcher_id = "researcher_456"
        self.institution = "Test University"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ReproducibilityTracker initialization."""
        assert isinstance(self.tracker, ReproducibilityTracker)
        assert self.tracker.data_dir == Path(self.temp_dir)
        assert self.tracker.base_replication_rate == 0.05
        assert self.tracker.base_success_rate == 0.40
        assert self.tracker.qrp_base_rate == 0.15
        assert len(self.tracker.replication_attempts) == 0
        assert len(self.tracker.qrp_incidents) == 0
        assert len(self.tracker.reproducibility_scores) == 0
    
    def test_register_replication_attempt_success(self):
        """Test successful replication attempt registration."""
        attempt_id = self.tracker.register_replication_attempt(
            original_paper_id=self.paper_id,
            replicating_researcher_id=self.researcher_id,
            replicating_institution=self.institution,
            outcome=ReplicationOutcome.SUCCESS,
            success_rate=0.8,
            confidence_level=0.9,
            methodology_similarity=0.85,
            sample_size_ratio=1.2,
            effect_size_ratio=0.95,
            notes="Successful replication",
            publication_status="published"
        )
        
        assert attempt_id in self.tracker.replication_attempts
        attempt = self.tracker.replication_attempts[attempt_id]
        assert attempt.original_paper_id == self.paper_id
        assert attempt.outcome == ReplicationOutcome.SUCCESS
        assert attempt.success_rate == 0.8
        assert attempt.confidence_level == 0.9
        assert attempt.methodology_similarity == 0.85
        
        # Check that reproducibility score was updated
        assert self.paper_id in self.tracker.reproducibility_scores
    
    def test_register_replication_attempt_validation_error(self):
        """Test replication attempt registration with invalid data."""
        with pytest.raises(ValidationError):
            self.tracker.register_replication_attempt(
                original_paper_id=self.paper_id,
                replicating_researcher_id=self.researcher_id,
                replicating_institution=self.institution,
                outcome=ReplicationOutcome.SUCCESS,
                success_rate=1.5,  # Invalid: > 1.0
                confidence_level=0.9,
                methodology_similarity=0.85,
                sample_size_ratio=1.2
            )
    
    def test_register_qrp_incident_success(self):
        """Test successful QRP incident registration."""
        incident_id = self.tracker.register_qrp_incident(
            paper_id=self.paper_id,
            researcher_id=self.researcher_id,
            practice_type=QuestionablePractice.P_HACKING,
            severity=0.7,
            detection_method="peer_review",
            evidence_strength=0.8,
            impact_on_results=0.6,
            corrective_action="Retraction issued"
        )
        
        assert incident_id in self.tracker.qrp_incidents
        incident = self.tracker.qrp_incidents[incident_id]
        assert incident.paper_id == self.paper_id
        assert incident.practice_type == QuestionablePractice.P_HACKING
        assert incident.severity == 0.7
        assert incident.evidence_strength == 0.8
        
        # Check that reproducibility score was updated
        assert self.paper_id in self.tracker.reproducibility_scores
    
    def test_register_qrp_incident_validation_error(self):
        """Test QRP incident registration with invalid data."""
        with pytest.raises(ValidationError):
            self.tracker.register_qrp_incident(
                paper_id=self.paper_id,
                researcher_id=self.researcher_id,
                practice_type=QuestionablePractice.P_HACKING,
                severity=1.5,  # Invalid: > 1.0
                detection_method="peer_review",
                evidence_strength=0.8,
                impact_on_results=0.6
            )
    
    def test_calculate_base_reproducibility_score(self):
        """Test base reproducibility score calculation."""
        score = self.tracker.calculate_base_reproducibility_score(
            paper_id=self.paper_id,
            methodology_quality=0.8,
            data_availability=0.9,
            code_availability=0.7,
            documentation_quality=0.6
        )
        
        # Expected: 0.8*0.35 + 0.9*0.25 + 0.7*0.25 + 0.6*0.15 = 0.77
        expected = 0.8 * 0.35 + 0.9 * 0.25 + 0.7 * 0.25 + 0.6 * 0.15
        assert abs(score - expected) < 0.01
        assert 0.0 <= score <= 1.0
    
    def test_update_reproducibility_score_with_replication(self):
        """Test reproducibility score update with replication data."""
        # Register a successful replication
        self.tracker.register_replication_attempt(
            original_paper_id=self.paper_id,
            replicating_researcher_id=self.researcher_id,
            replicating_institution=self.institution,
            outcome=ReplicationOutcome.SUCCESS,
            success_rate=0.9,
            confidence_level=0.8,
            methodology_similarity=0.9,
            sample_size_ratio=1.0
        )
        
        score = self.tracker.reproducibility_scores[self.paper_id]
        assert score.paper_id == self.paper_id
        assert score.replication_adjustment > 0  # Should be positive for successful replication
        assert 0.0 <= score.final_score <= 1.0
    
    def test_update_reproducibility_score_with_qrp(self):
        """Test reproducibility score update with QRP incident."""
        # Register a QRP incident
        self.tracker.register_qrp_incident(
            paper_id=self.paper_id,
            researcher_id=self.researcher_id,
            practice_type=QuestionablePractice.DATA_FABRICATION,
            severity=0.9,
            detection_method="statistical_analysis",
            evidence_strength=0.95,
            impact_on_results=0.8
        )
        
        score = self.tracker.reproducibility_scores[self.paper_id]
        assert score.paper_id == self.paper_id
        assert score.qrp_penalty < 0  # Should be negative for QRP
        assert 0.0 <= score.final_score <= 1.0
    
    def test_simulate_replication_attempts(self):
        """Test simulation of replication attempts."""
        paper_ids = [f"paper_{i}" for i in range(100)]
        
        with patch('random.random') as mock_random:
            # Mock random to ensure some papers get replication attempts
            mock_random.side_effect = [0.01] * 10 + [0.1] * 90  # First 10 get attempts
            
            attempted_papers = self.tracker.simulate_replication_attempts(
                paper_ids=paper_ids,
                field="psychology"
            )
        
        assert len(attempted_papers) == 10
        assert len(self.tracker.replication_attempts) == 10
        
        # Check that all attempted papers have replication attempts
        for paper_id in attempted_papers:
            attempts = [a for a in self.tracker.replication_attempts.values() 
                       if a.original_paper_id == paper_id]
            assert len(attempts) == 1
    
    def test_simulate_qrp_incidents(self):
        """Test simulation of QRP incidents."""
        paper_ids = [f"paper_{i}" for i in range(100)]
        researcher_ids = [f"researcher_{i}" for i in range(50)]
        
        with patch('random.random') as mock_random:
            # Mock random to ensure some papers get QRP incidents
            mock_random.side_effect = [0.01] * 15 + [0.2] * 85  # First 15 get incidents
            
            affected_papers = self.tracker.simulate_qrp_incidents(
                paper_ids=paper_ids,
                researcher_ids=researcher_ids,
                field="psychology"
            )
        
        assert len(affected_papers) == 15
        assert len(self.tracker.qrp_incidents) == 15
        
        # Check that all affected papers have QRP incidents
        for paper_id in affected_papers:
            incidents = [i for i in self.tracker.qrp_incidents.values() 
                        if i.paper_id == paper_id]
            assert len(incidents) == 1
    
    def test_calculate_crisis_metrics(self):
        """Test calculation of reproducibility crisis metrics."""
        # Set up test data
        start_date = date.today() - timedelta(days=365)
        end_date = date.today()
        
        # Add some replication attempts
        for i in range(5):
            self.tracker.register_replication_attempt(
                original_paper_id=f"paper_{i}",
                replicating_researcher_id=f"researcher_{i}",
                replicating_institution="Test University",
                outcome=ReplicationOutcome.SUCCESS if i < 2 else ReplicationOutcome.FAILURE,
                success_rate=0.8 if i < 2 else 0.3,
                confidence_level=0.8,
                methodology_similarity=0.9,
                sample_size_ratio=1.0
            )
        
        # Add some QRP incidents
        for i in range(3):
            self.tracker.register_qrp_incident(
                paper_id=f"paper_{i+10}",
                researcher_id=f"researcher_{i}",
                practice_type=QuestionablePractice.P_HACKING,
                severity=0.5,
                detection_method="peer_review",
                evidence_strength=0.7,
                impact_on_results=0.4
            )
        
        metrics = self.tracker.calculate_crisis_metrics(
            field="psychology",
            start_date=start_date,
            end_date=end_date
        )
        
        assert isinstance(metrics, ReproducibilityCrisisMetrics)
        assert metrics.field == "psychology"
        assert metrics.total_papers == 8  # 5 with replications + 3 with QRP
        assert metrics.papers_with_replication_attempts == 5
        assert metrics.successful_replications == 2
        assert metrics.failed_replications == 3
        assert metrics.qrp_incidents == 3
        assert 0 <= metrics.replication_rate <= 100
        assert 0 <= metrics.success_rate <= 100
        assert 0 <= metrics.qrp_rate <= 100
    
    def test_get_paper_reproducibility_impact_with_data(self):
        """Test getting reproducibility impact for paper with data."""
        # Register replication and QRP for the paper
        self.tracker.register_replication_attempt(
            original_paper_id=self.paper_id,
            replicating_researcher_id=self.researcher_id,
            replicating_institution=self.institution,
            outcome=ReplicationOutcome.SUCCESS,
            success_rate=0.8,
            confidence_level=0.9,
            methodology_similarity=0.85,
            sample_size_ratio=1.0
        )
        
        impact = self.tracker.get_paper_reproducibility_impact(self.paper_id)
        
        assert impact["has_reproducibility_data"] is True
        assert 0.0 <= impact["reproducibility_score"] <= 1.0
        assert -0.3 <= impact["quality_adjustment"] <= 0.3
        assert impact["replication_attempts"] == 1
        assert impact["qrp_incidents"] == 0
        assert "last_updated" in impact
    
    def test_get_paper_reproducibility_impact_no_data(self):
        """Test getting reproducibility impact for paper without data."""
        impact = self.tracker.get_paper_reproducibility_impact("nonexistent_paper")
        
        assert impact["has_reproducibility_data"] is False
        assert impact["reproducibility_score"] == 0.5
        assert impact["quality_adjustment"] == 0.0
        assert impact["confidence_interval"] == (0.4, 0.6)
    
    def test_export_and_load_data(self):
        """Test data export and loading functionality."""
        # Add some test data
        self.tracker.register_replication_attempt(
            original_paper_id=self.paper_id,
            replicating_researcher_id=self.researcher_id,
            replicating_institution=self.institution,
            outcome=ReplicationOutcome.SUCCESS,
            success_rate=0.8,
            confidence_level=0.9,
            methodology_similarity=0.85,
            sample_size_ratio=1.0
        )
        
        # Export data
        exported_data = self.tracker.export_data()
        
        assert "replication_attempts" in exported_data
        assert "qrp_incidents" in exported_data
        assert "reproducibility_scores" in exported_data
        assert "export_timestamp" in exported_data
        assert len(exported_data["replication_attempts"]) == 1
        
        # Save and load data
        filename = "test_reproducibility_data.json"
        self.tracker.save_to_file(filename)
        
        # Create new tracker and load data
        new_tracker = ReproducibilityTracker(data_dir=self.temp_dir)
        new_tracker.load_from_file(filename)
        
        assert len(new_tracker.replication_attempts) == 1
        assert len(new_tracker.reproducibility_scores) == 1
    
    def test_field_specific_parameters(self):
        """Test field-specific replication and success rates."""
        # Test psychology field (higher QRP rate, lower success rate)
        psychology_replication_rate = self.tracker.field_replication_rates["psychology"]
        psychology_success_rate = self.tracker.field_success_rates["psychology"]
        
        assert psychology_replication_rate > self.tracker.base_replication_rate
        assert psychology_success_rate < self.tracker.base_success_rate
        
        # Test mathematics field (lower QRP rate, higher success rate)
        math_replication_rate = self.tracker.field_replication_rates["mathematics"]
        math_success_rate = self.tracker.field_success_rates["mathematics"]
        
        assert math_replication_rate < self.tracker.base_replication_rate
        assert math_success_rate > self.tracker.base_success_rate
    
    def test_replication_adjustment_calculation(self):
        """Test replication adjustment calculation logic."""
        # Add successful replication
        self.tracker.register_replication_attempt(
            original_paper_id=self.paper_id,
            replicating_researcher_id=self.researcher_id,
            replicating_institution=self.institution,
            outcome=ReplicationOutcome.SUCCESS,
            success_rate=0.9,
            confidence_level=0.8,
            methodology_similarity=0.9,
            sample_size_ratio=1.0
        )
        
        adjustment = self.tracker._calculate_replication_adjustment(self.paper_id)
        assert adjustment > 0  # Successful replication should increase score
        assert -0.3 <= adjustment <= 0.3
        
        # Add failed replication
        self.tracker.register_replication_attempt(
            original_paper_id=self.paper_id,
            replicating_researcher_id="another_researcher",
            replicating_institution=self.institution,
            outcome=ReplicationOutcome.FAILURE,
            success_rate=0.2,
            confidence_level=0.7,
            methodology_similarity=0.8,
            sample_size_ratio=1.0
        )
        
        new_adjustment = self.tracker._calculate_replication_adjustment(self.paper_id)
        assert new_adjustment < adjustment  # Failed replication should decrease adjustment
    
    def test_qrp_penalty_calculation(self):
        """Test QRP penalty calculation logic."""
        # Add severe QRP incident
        self.tracker.register_qrp_incident(
            paper_id=self.paper_id,
            researcher_id=self.researcher_id,
            practice_type=QuestionablePractice.DATA_FABRICATION,
            severity=0.9,
            detection_method="statistical_analysis",
            evidence_strength=0.95,
            impact_on_results=0.8
        )
        
        penalty = self.tracker._calculate_qrp_penalty(self.paper_id)
        assert penalty < 0  # QRP should always result in penalty
        assert -0.5 <= penalty <= 0
        
        # Add mild QRP incident
        self.tracker.register_qrp_incident(
            paper_id=self.paper_id,
            researcher_id="another_researcher",
            practice_type=QuestionablePractice.CHERRY_PICKING,
            severity=0.3,
            detection_method="peer_review",
            evidence_strength=0.5,
            impact_on_results=0.2
        )
        
        new_penalty = self.tracker._calculate_qrp_penalty(self.paper_id)
        assert new_penalty < penalty  # Additional QRP should increase penalty (more negative)
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation for reproducibility scores."""
        self.tracker.register_replication_attempt(
            original_paper_id=self.paper_id,
            replicating_researcher_id=self.researcher_id,
            replicating_institution=self.institution,
            outcome=ReplicationOutcome.SUCCESS,
            success_rate=0.8,
            confidence_level=0.9,
            methodology_similarity=0.85,
            sample_size_ratio=1.0
        )
        
        score = self.tracker.reproducibility_scores[self.paper_id]
        lower, upper = score.confidence_interval
        
        assert lower <= score.final_score <= upper
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0
        assert upper > lower


class TestReplicationAttempt:
    """Test cases for ReplicationAttempt dataclass."""
    
    def test_valid_replication_attempt(self):
        """Test creation of valid replication attempt."""
        attempt = ReplicationAttempt(
            attempt_id="attempt_123",
            original_paper_id="paper_456",
            replicating_researcher_id="researcher_789",
            replicating_institution="Test University",
            attempt_date=date.today(),
            outcome=ReplicationOutcome.SUCCESS,
            success_rate=0.8,
            confidence_level=0.9,
            methodology_similarity=0.85,
            sample_size_ratio=1.2,
            effect_size_ratio=0.95,
            notes="Test replication",
            publication_status="published"
        )
        
        assert attempt.attempt_id == "attempt_123"
        assert attempt.outcome == ReplicationOutcome.SUCCESS
        assert attempt.success_rate == 0.8
    
    def test_invalid_success_rate(self):
        """Test validation of success rate."""
        with pytest.raises(ValidationError):
            ReplicationAttempt(
                attempt_id="attempt_123",
                original_paper_id="paper_456",
                replicating_researcher_id="researcher_789",
                replicating_institution="Test University",
                attempt_date=date.today(),
                outcome=ReplicationOutcome.SUCCESS,
                success_rate=1.5,  # Invalid: > 1.0
                confidence_level=0.9,
                methodology_similarity=0.85,
                sample_size_ratio=1.2
            )


class TestQuestionablePracticeIncident:
    """Test cases for QuestionablePracticeIncident dataclass."""
    
    def test_valid_qrp_incident(self):
        """Test creation of valid QRP incident."""
        incident = QuestionablePracticeIncident(
            incident_id="incident_123",
            paper_id="paper_456",
            researcher_id="researcher_789",
            practice_type=QuestionablePractice.P_HACKING,
            severity=0.7,
            detection_method="peer_review",
            detection_date=date.today(),
            evidence_strength=0.8,
            impact_on_results=0.6,
            corrective_action="Retraction"
        )
        
        assert incident.incident_id == "incident_123"
        assert incident.practice_type == QuestionablePractice.P_HACKING
        assert incident.severity == 0.7
    
    def test_invalid_severity(self):
        """Test validation of severity."""
        with pytest.raises(ValidationError):
            QuestionablePracticeIncident(
                incident_id="incident_123",
                paper_id="paper_456",
                researcher_id="researcher_789",
                practice_type=QuestionablePractice.P_HACKING,
                severity=1.5,  # Invalid: > 1.0
                detection_method="peer_review",
                detection_date=date.today(),
                evidence_strength=0.8,
                impact_on_results=0.6
            )


class TestReproducibilityScore:
    """Test cases for ReproducibilityScore dataclass."""
    
    def test_valid_reproducibility_score(self):
        """Test creation of valid reproducibility score."""
        score = ReproducibilityScore(
            paper_id="paper_123",
            base_score=0.7,
            replication_adjustment=0.1,
            qrp_penalty=-0.2,
            final_score=0.6,
            confidence_interval=(0.5, 0.7),
            last_updated=date.today()
        )
        
        assert score.paper_id == "paper_123"
        assert score.base_score == 0.7
        assert score.final_score == 0.6
    
    def test_invalid_base_score(self):
        """Test validation of base score."""
        with pytest.raises(ValidationError):
            ReproducibilityScore(
                paper_id="paper_123",
                base_score=1.5,  # Invalid: > 1.0
                replication_adjustment=0.1,
                qrp_penalty=-0.2,
                final_score=0.6,
                confidence_interval=(0.5, 0.7),
                last_updated=date.today()
            )


class TestReproducibilityCrisisMetrics:
    """Test cases for ReproducibilityCrisisMetrics dataclass."""
    
    def test_valid_crisis_metrics(self):
        """Test creation of valid crisis metrics."""
        start_date = date.today() - timedelta(days=365)
        end_date = date.today()
        
        metrics = ReproducibilityCrisisMetrics(
            field="psychology",
            time_period=(start_date, end_date),
            total_papers=100,
            papers_with_replication_attempts=8,
            successful_replications=3,
            failed_replications=5,
            replication_rate=8.0,
            success_rate=37.5,
            average_reproducibility_score=0.45,
            qrp_incidents=15,
            qrp_rate=15.0
        )
        
        assert metrics.field == "psychology"
        assert metrics.total_papers == 100
        assert metrics.replication_rate == 8.0
        assert metrics.success_rate == 37.5
    
    def test_invalid_total_papers(self):
        """Test validation of total papers."""
        start_date = date.today() - timedelta(days=365)
        end_date = date.today()
        
        with pytest.raises(ValidationError):
            ReproducibilityCrisisMetrics(
                field="psychology",
                time_period=(start_date, end_date),
                total_papers=-5,  # Invalid: negative
                papers_with_replication_attempts=8,
                successful_replications=3,
                failed_replications=5,
                replication_rate=8.0,
                success_rate=37.5,
                average_reproducibility_score=0.45,
                qrp_incidents=15,
                qrp_rate=15.0
            )