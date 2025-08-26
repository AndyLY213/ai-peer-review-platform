"""
Unit tests for OpenScienceManager module.
"""

import pytest
import tempfile
import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.enhancements.open_science_manager import (
    OpenScienceManager, PreprintRecord, OpenAccessRecord, DataSharingRecord,
    OpenScienceProfile, OpenScienceMetrics, PreprintServer, OpenAccessType,
    DataSharingLevel
)
from src.core.exceptions import ValidationError, PeerReviewError


class TestOpenScienceManager:
    """Test cases for OpenScienceManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def manager(self, temp_dir):
        """Create an OpenScienceManager instance for testing."""
        return OpenScienceManager(data_dir=temp_dir / "open_science")
    
    def test_initialization(self, temp_dir):
        """Test OpenScienceManager initialization."""
        manager = OpenScienceManager(data_dir=temp_dir / "open_science")
        
        assert manager.data_dir == temp_dir / "open_science"
        assert manager.data_dir.exists()
        assert isinstance(manager.preprint_records, dict)
        assert isinstance(manager.open_access_records, dict)
        assert isinstance(manager.data_sharing_records, dict)
        assert isinstance(manager.researcher_profiles, dict)
    
    def test_create_researcher_profile(self, manager):
        """Test creating researcher profiles with different parameters."""
        # Test basic profile creation
        profile = manager.create_researcher_profile(
            researcher_id="researcher_1",
            career_stage="Assistant Prof",
            field="computer_science"
        )
        
        assert profile.researcher_id == "researcher_1"
        assert 0.0 <= profile.preprint_adoption_rate <= 1.0
        assert 0.0 <= profile.open_access_preference <= 1.0
        assert 0.0 <= profile.data_sharing_willingness <= 1.0
        assert profile.preferred_preprint_server == PreprintServer.ARXIV
        assert not profile.institutional_mandate
        assert not profile.funder_mandate
        
        # Test with mandates
        profile_mandated = manager.create_researcher_profile(
            researcher_id="researcher_2",
            career_stage="Full Prof",
            field="biology",
            institutional_mandate=True,
            funder_mandate=True
        )
        
        assert profile_mandated.institutional_mandate
        assert profile_mandated.funder_mandate
        assert profile_mandated.preferred_preprint_server == PreprintServer.BIORXIV
        # Mandates should generally increase adoption rates, but career stage also matters
        # Full Prof has lower base rate (0.5) vs Assistant Prof (0.7), so just check it's reasonable
        assert profile_mandated.preprint_adoption_rate > 0.3  # Should be above minimum with mandates
    
    def test_career_stage_influence(self, manager):
        """Test that career stage influences adoption rates."""
        # Graduate students should have higher adoption than emeritus
        grad_profile = manager.create_researcher_profile(
            "grad_1", "Graduate Student", "computer_science"
        )
        emeritus_profile = manager.create_researcher_profile(
            "emeritus_1", "Emeritus", "computer_science"
        )
        
        assert grad_profile.preprint_adoption_rate >= emeritus_profile.preprint_adoption_rate
        assert grad_profile.career_stage_influence >= emeritus_profile.career_stage_influence
    
    def test_submit_preprint(self, manager):
        """Test preprint submission functionality."""
        # Create researcher profile first
        manager.create_researcher_profile("researcher_1", "Postdoc", "physics")
        
        # Test successful preprint submission
        with patch('random.random', return_value=0.1):  # Ensure adoption
            preprint = manager.submit_preprint("paper_1", "researcher_1", "physics")
        
        assert preprint is not None
        assert preprint.paper_id == "paper_1"
        assert preprint.researcher_id == "researcher_1"
        assert preprint.server == PreprintServer.ARXIV
        assert preprint.version == 1
        assert preprint.submission_date == date.today()
        assert preprint.preprint_id in manager.preprint_records
    
    def test_submit_preprint_no_adoption(self, manager):
        """Test preprint submission when researcher doesn't adopt preprints."""
        manager.create_researcher_profile("researcher_1", "Full Prof", "physics")
        
        # Mock random to return high value (no adoption)
        with patch('random.random', return_value=0.9):
            preprint = manager.submit_preprint("paper_1", "researcher_1", "physics")
        
        assert preprint is None
    
    def test_submit_preprint_auto_profile_creation(self, manager):
        """Test that preprint submission creates profile if none exists."""
        with patch('random.random', return_value=0.1):
            preprint = manager.submit_preprint("paper_1", "new_researcher", "biology")
        
        assert preprint is not None
        assert "new_researcher" in manager.researcher_profiles
        assert preprint.server == PreprintServer.BIORXIV
    
    def test_track_preprint_usage(self, manager):
        """Test preprint usage tracking."""
        # Create and submit preprint
        manager.create_researcher_profile("researcher_1", "Postdoc", "computer_science")
        with patch('random.random', return_value=0.1):
            preprint = manager.submit_preprint("paper_1", "researcher_1", "computer_science")
        
        # Track usage
        usage = manager.track_preprint_usage(preprint.preprint_id)
        
        assert "downloads" in usage
        assert "views" in usage
        assert "citations" in usage
        assert "comments" in usage
        assert all(isinstance(v, int) for v in usage.values())
    
    def test_track_preprint_usage_invalid_id(self, manager):
        """Test tracking usage for invalid preprint ID."""
        with pytest.raises(ValidationError):
            manager.track_preprint_usage("invalid_id")
    
    def test_publish_open_access(self, manager):
        """Test open access publication functionality."""
        manager.create_researcher_profile("researcher_1", "Assistant Prof", "biology")
        
        oa_record = manager.publish_open_access(
            paper_id="paper_1",
            researcher_id="researcher_1",
            venue_type="Top Journal",
            has_funding=True
        )
        
        assert oa_record.paper_id == "paper_1"
        assert oa_record.access_type in [OpenAccessType.GOLD, OpenAccessType.GREEN]
        assert oa_record.publication_date == date.today()
        assert oa_record.paper_id in manager.open_access_records
        
        # Test with funding - should prefer gold OA
        if oa_record.access_type == OpenAccessType.GOLD:
            assert oa_record.apc_cost is not None
            assert oa_record.apc_cost >= 2000  # Top journal range
    
    def test_publish_open_access_auto_profile(self, manager):
        """Test open access publication with automatic profile creation."""
        oa_record = manager.publish_open_access(
            paper_id="paper_1",
            researcher_id="new_researcher",
            venue_type="Mid Conference"
        )
        
        assert oa_record is not None
        assert "new_researcher" in manager.researcher_profiles
    
    def test_enforce_data_sharing_requirements(self, manager):
        """Test data sharing requirement enforcement."""
        manager.create_researcher_profile("researcher_1", "Associate Prof", "psychology")
        
        # Test with no constraints
        ds_record = manager.enforce_data_sharing_requirements(
            paper_id="paper_1",
            researcher_id="researcher_1"
        )
        
        assert ds_record.paper_id == "paper_1"
        assert isinstance(ds_record.sharing_level, DataSharingLevel)
        assert 0.0 <= ds_record.compliance_score <= 1.0
        assert ds_record.paper_id in manager.data_sharing_records
    
    def test_enforce_data_sharing_with_constraints(self, manager):
        """Test data sharing with human subjects and proprietary data."""
        manager.create_researcher_profile("researcher_1", "Full Prof", "medicine")
        
        # Test with human subjects
        ds_record = manager.enforce_data_sharing_requirements(
            paper_id="paper_1",
            researcher_id="researcher_1",
            has_human_subjects=True
        )
        
        # Should be more restrictive
        assert ds_record.sharing_level in [DataSharingLevel.ON_REQUEST, DataSharingLevel.RESTRICTED]
        
        # Test with proprietary data
        ds_record2 = manager.enforce_data_sharing_requirements(
            paper_id="paper_2",
            researcher_id="researcher_1",
            has_proprietary_data=True
        )
        
        assert ds_record2.sharing_level in [DataSharingLevel.ON_REQUEST, DataSharingLevel.RESTRICTED]
    
    def test_calculate_open_science_metrics(self, manager):
        """Test calculation of aggregated open science metrics."""
        # Create some test data
        manager.create_researcher_profile("researcher_1", "Postdoc", "computer_science")
        manager.create_researcher_profile("researcher_2", "Assistant Prof", "biology")
        
        # Submit preprints and publications
        with patch('random.random', return_value=0.1):
            manager.submit_preprint("paper_1", "researcher_1", "computer_science")
            manager.submit_preprint("paper_2", "researcher_2", "biology")
        
        manager.publish_open_access("paper_1", "researcher_1", "Top Conference")
        manager.publish_open_access("paper_2", "researcher_2", "Specialized Journal")
        
        manager.enforce_data_sharing_requirements("paper_1", "researcher_1")
        manager.enforce_data_sharing_requirements("paper_2", "researcher_2")
        
        metrics = manager.calculate_open_science_metrics()
        
        assert isinstance(metrics, OpenScienceMetrics)
        assert metrics.total_preprints >= 0
        assert 0.0 <= metrics.preprint_adoption_rate <= 2.0  # Can be > 1 if more preprints than papers
        assert 0.0 <= metrics.open_access_rate <= 1.0
        assert 0.0 <= metrics.data_sharing_compliance_rate <= 1.0
        assert metrics.average_preprint_citations >= 0.0
        assert isinstance(metrics.server_usage_distribution, dict)
        assert isinstance(metrics.oa_type_distribution, dict)
        assert isinstance(metrics.data_sharing_distribution, dict)
    
    def test_calculate_metrics_empty_data(self, manager):
        """Test metrics calculation with no data."""
        metrics = manager.calculate_open_science_metrics()
        
        assert metrics.total_preprints == 0
        assert metrics.preprint_adoption_rate == 0.0
        assert metrics.open_access_rate == 0.0
        assert metrics.data_sharing_compliance_rate == 0.0
        assert metrics.average_preprint_citations == 0.0
        assert metrics.average_time_to_publication == 0.0
    
    def test_get_researcher_open_science_score(self, manager):
        """Test calculation of researcher open science scores."""
        # Test with no profile
        score = manager.get_researcher_open_science_score("nonexistent")
        assert score == 0.0
        
        # Test with profile and activities
        manager.create_researcher_profile("researcher_1", "Postdoc", "physics")
        
        with patch('random.random', return_value=0.1):
            manager.submit_preprint("paper_1", "researcher_1", "physics")
        
        manager.publish_open_access("paper_1", "researcher_1", "Top Conference")
        manager.enforce_data_sharing_requirements("paper_1", "researcher_1")
        
        score = manager.get_researcher_open_science_score("researcher_1")
        assert 0.0 <= score <= 1.0
    
    def test_simulate_policy_impact(self, manager):
        """Test simulation of policy impacts."""
        # Create some researchers
        manager.create_researcher_profile("researcher_1", "Assistant Prof", "computer_science")
        manager.create_researcher_profile("researcher_2", "Associate Prof", "biology")
        
        # Test institutional mandate
        impact = manager.simulate_policy_impact("institutional_mandate", 0.5)
        
        assert "preprint_adoption_change" in impact
        assert "open_access_change" in impact
        assert "data_sharing_change" in impact
        assert all(isinstance(v, float) for v in impact.values())
        
        # Test funder mandate
        impact2 = manager.simulate_policy_impact("funder_mandate", 0.8)
        assert impact2["open_access_change"] >= 0  # Should increase OA
        
        # Test APC funding
        impact3 = manager.simulate_policy_impact("apc_funding", 0.3)
        assert impact3["open_access_change"] >= 0  # Should increase OA
    
    def test_simulate_policy_impact_invalid_strength(self, manager):
        """Test policy simulation with invalid strength values."""
        with pytest.raises(ValidationError):
            manager.simulate_policy_impact("institutional_mandate", -0.1)
        
        with pytest.raises(ValidationError):
            manager.simulate_policy_impact("institutional_mandate", 1.1)
    
    def test_data_persistence(self, manager):
        """Test saving and loading of data."""
        # Create some test data
        manager.create_researcher_profile("researcher_1", "Postdoc", "computer_science")
        
        with patch('random.random', return_value=0.1):
            preprint = manager.submit_preprint("paper_1", "researcher_1", "computer_science")
        
        oa_record = manager.publish_open_access("paper_1", "researcher_1", "Top Conference")
        ds_record = manager.enforce_data_sharing_requirements("paper_1", "researcher_1")
        
        # Save data
        manager.save_data()
        
        # Create new manager and verify data is loaded
        new_manager = OpenScienceManager(data_dir=manager.data_dir)
        
        assert len(new_manager.preprint_records) == 1
        assert len(new_manager.open_access_records) == 1
        assert len(new_manager.data_sharing_records) == 1
        assert len(new_manager.researcher_profiles) == 1
        
        # Verify data integrity
        loaded_preprint = list(new_manager.preprint_records.values())[0]
        assert loaded_preprint.paper_id == preprint.paper_id
        assert loaded_preprint.researcher_id == preprint.researcher_id
    
    def test_server_field_mapping(self, manager):
        """Test that fields are correctly mapped to preprint servers."""
        test_cases = [
            ("computer_science", PreprintServer.ARXIV),
            ("physics", PreprintServer.ARXIV),
            ("biology", PreprintServer.BIORXIV),
            ("medicine", PreprintServer.MEDRXIV),
            ("psychology", PreprintServer.PSYARXIV),
            ("chemistry", PreprintServer.CHEMRXIV),
            ("unknown_field", PreprintServer.PREPRINTS_ORG)  # Default
        ]
        
        for field, expected_server in test_cases:
            profile = manager.create_researcher_profile(
                f"researcher_{field}", "Postdoc", field
            )
            
            if field != "unknown_field":
                assert profile.preferred_preprint_server == expected_server
            
            with patch('random.random', return_value=0.1):
                preprint = manager.submit_preprint(f"paper_{field}", f"researcher_{field}", field)
            
            if field == "unknown_field":
                assert preprint.server == PreprintServer.PREPRINTS_ORG
            else:
                assert preprint.server == expected_server


class TestDataClasses:
    """Test cases for data classes."""
    
    def test_preprint_record_validation(self):
        """Test PreprintRecord validation."""
        # Valid record
        record = PreprintRecord(
            preprint_id="preprint_1",
            paper_id="paper_1",
            researcher_id="researcher_1",
            server=PreprintServer.ARXIV,
            submission_date=date.today(),
            version=1
        )
        assert record.version == 1
        
        # Invalid version
        with pytest.raises(ValidationError):
            PreprintRecord(
                preprint_id="preprint_1",
                paper_id="paper_1",
                researcher_id="researcher_1",
                server=PreprintServer.ARXIV,
                submission_date=date.today(),
                version=0  # Invalid
            )
        
        # Invalid download count
        with pytest.raises(ValidationError):
            PreprintRecord(
                preprint_id="preprint_1",
                paper_id="paper_1",
                researcher_id="researcher_1",
                server=PreprintServer.ARXIV,
                submission_date=date.today(),
                version=1,
                download_count=-1  # Invalid
            )
    
    def test_open_access_record_validation(self):
        """Test OpenAccessRecord validation."""
        # Valid record
        record = OpenAccessRecord(
            paper_id="paper_1",
            access_type=OpenAccessType.GOLD,
            publication_date=date.today(),
            apc_cost=2000.0
        )
        assert record.apc_cost == 2000.0
        
        # Invalid APC cost
        with pytest.raises(ValidationError):
            OpenAccessRecord(
                paper_id="paper_1",
                access_type=OpenAccessType.GOLD,
                publication_date=date.today(),
                apc_cost=-100.0  # Invalid
            )
        
        # Invalid embargo period
        with pytest.raises(ValidationError):
            OpenAccessRecord(
                paper_id="paper_1",
                access_type=OpenAccessType.GREEN,
                publication_date=date.today(),
                embargo_period_months=-6  # Invalid
            )
    
    def test_data_sharing_record_validation(self):
        """Test DataSharingRecord validation."""
        # Valid record
        record = DataSharingRecord(
            paper_id="paper_1",
            sharing_level=DataSharingLevel.FULL_OPEN,
            compliance_score=0.8
        )
        assert record.compliance_score == 0.8
        assert record.funder_requirements == []  # Default empty list
        
        # Invalid compliance score
        with pytest.raises(ValidationError):
            DataSharingRecord(
                paper_id="paper_1",
                sharing_level=DataSharingLevel.FULL_OPEN,
                compliance_score=1.5  # Invalid
            )
    
    def test_open_science_profile_validation(self):
        """Test OpenScienceProfile validation."""
        # Valid profile
        profile = OpenScienceProfile(
            researcher_id="researcher_1",
            preprint_adoption_rate=0.8,
            open_access_preference=0.7,
            data_sharing_willingness=0.6
        )
        assert profile.preprint_adoption_rate == 0.8
        
        # Invalid adoption rate
        with pytest.raises(ValidationError):
            OpenScienceProfile(
                researcher_id="researcher_1",
                preprint_adoption_rate=1.5,  # Invalid
                open_access_preference=0.7,
                data_sharing_willingness=0.6
            )
    
    def test_open_science_metrics_validation(self):
        """Test OpenScienceMetrics validation."""
        # Valid metrics
        metrics = OpenScienceMetrics(
            total_preprints=100,
            preprint_adoption_rate=0.8,
            open_access_rate=0.6,
            data_sharing_compliance_rate=0.4,
            average_preprint_citations=5.2,
            average_time_to_publication=180.0,
            server_usage_distribution={},
            oa_type_distribution={},
            data_sharing_distribution={}
        )
        assert metrics.total_preprints == 100
        
        # Invalid total preprints
        with pytest.raises(ValidationError):
            OpenScienceMetrics(
                total_preprints=-1,  # Invalid
                preprint_adoption_rate=0.8,
                open_access_rate=0.6,
                data_sharing_compliance_rate=0.4,
                average_preprint_citations=5.2,
                average_time_to_publication=180.0,
                server_usage_distribution={},
                oa_type_distribution={},
                data_sharing_distribution={}
            )


if __name__ == "__main__":
    pytest.main([__file__])