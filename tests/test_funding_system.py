"""
Unit tests for Funding Integration System

Tests the FundingAgency, FundingCycle, FundingApplication, and FundingSystem classes
functionality including agency management, cycle creation, application processing,
and funding statistics.
"""

import pytest
from datetime import date, timedelta
from typing import List, Dict

from src.enhancements.funding_system import (
    FundingSystem, FundingAgency, FundingCycle, FundingApplication,
    FundingAgencyType, FundingCycleStatus, ApplicationStatus
)
from src.data.enhanced_models import ResearcherLevel
from src.core.exceptions import ValidationError


class TestFundingAgency:
    """Test cases for FundingAgency class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.nsf_agency = FundingAgency(
            name="Test NSF",
            agency_type=FundingAgencyType.NSF,
            typical_cycle_duration_years=3,
            typical_funding_amount_range=(100000, 800000),
            success_rate=0.20,
            min_career_level=ResearcherLevel.ASSISTANT_PROF,
            application_deadlines_per_year=["February 15", "August 15"]
        )
        
        self.industry_agency = FundingAgency(
            name="Test Industry",
            agency_type=FundingAgencyType.INDUSTRY,
            typical_cycle_duration_years=1,
            typical_funding_amount_range=(50000, 300000),
            success_rate=0.35,
            min_career_level=ResearcherLevel.POSTDOC,
            application_deadlines_per_year=["March 31", "September 30"]
        )
    
    def test_initialization(self):
        """Test FundingAgency initialization."""
        agency = FundingAgency(
            name="Test Agency",
            agency_type=FundingAgencyType.NSF
        )
        assert agency.name == "Test Agency"
        assert agency.agency_type == FundingAgencyType.NSF
        assert agency.agency_id is not None
        assert len(agency.application_deadlines_per_year) > 0  # Should have defaults
    
    def test_initialization_with_invalid_duration(self):
        """Test FundingAgency initialization with invalid duration."""
        with pytest.raises(ValidationError):
            FundingAgency(
                name="Invalid Agency",
                typical_cycle_duration_years=5  # Invalid: must be 1-3
            )
    
    def test_initialization_with_invalid_success_rate(self):
        """Test FundingAgency initialization with invalid success rate."""
        with pytest.raises(ValidationError):
            FundingAgency(
                name="Invalid Agency",
                success_rate=1.5  # Invalid: must be 0.0-1.0
            )
    
    def test_default_deadlines_nsf(self):
        """Test default deadlines for NSF agency."""
        agency = FundingAgency(
            name="NSF Test",
            agency_type=FundingAgencyType.NSF
        )
        assert "February 15" in agency.application_deadlines_per_year
        assert "August 15" in agency.application_deadlines_per_year
    
    def test_default_deadlines_nih(self):
        """Test default deadlines for NIH agency."""
        agency = FundingAgency(
            name="NIH Test",
            agency_type=FundingAgencyType.NIH
        )
        assert "February 5" in agency.application_deadlines_per_year
        assert "June 5" in agency.application_deadlines_per_year
        assert "October 5" in agency.application_deadlines_per_year
    
    def test_get_next_deadline_future(self):
        """Test getting next deadline when future deadlines exist."""
        # Test with a date before first deadline
        test_date = date(2024, 1, 15)
        next_deadline = self.nsf_agency.get_next_deadline(test_date)
        
        assert next_deadline is not None
        assert next_deadline.month == 2  # February
        assert next_deadline.day == 15
        assert next_deadline.year == 2024
    
    def test_get_next_deadline_past_all(self):
        """Test getting next deadline when all current year deadlines have passed."""
        # Test with a date after all deadlines
        test_date = date(2024, 12, 1)
        next_deadline = self.nsf_agency.get_next_deadline(test_date)
        
        assert next_deadline is not None
        assert next_deadline.month == 2  # February of next year
        assert next_deadline.day == 15
        assert next_deadline.year == 2025
    
    def test_get_next_deadline_between_deadlines(self):
        """Test getting next deadline when between two deadlines."""
        # Test with a date between February and August
        test_date = date(2024, 5, 1)
        next_deadline = self.nsf_agency.get_next_deadline(test_date)
        
        assert next_deadline is not None
        assert next_deadline.month == 8  # August
        assert next_deadline.day == 15
        assert next_deadline.year == 2024
    
    def test_is_researcher_eligible_valid(self):
        """Test researcher eligibility for valid cases."""
        # Assistant professor should be eligible for NSF
        is_eligible, issues = self.nsf_agency.is_researcher_eligible(
            ResearcherLevel.ASSISTANT_PROF, 0
        )
        assert is_eligible
        assert len(issues) == 0
        
        # Full professor should be eligible
        is_eligible, issues = self.nsf_agency.is_researcher_eligible(
            ResearcherLevel.FULL_PROF, 1
        )
        assert is_eligible
        assert len(issues) == 0
    
    def test_is_researcher_eligible_career_level_too_low(self):
        """Test researcher eligibility with career level too low."""
        # Graduate student should not be eligible for NSF (requires Assistant Prof+)
        is_eligible, issues = self.nsf_agency.is_researcher_eligible(
            ResearcherLevel.GRADUATE_STUDENT, 0
        )
        assert not is_eligible
        assert len(issues) > 0
        assert "Minimum career level" in issues[0]
    
    def test_is_researcher_eligible_too_many_applications(self):
        """Test researcher eligibility with too many current applications."""
        # Test with maximum applications already submitted
        is_eligible, issues = self.nsf_agency.is_researcher_eligible(
            ResearcherLevel.ASSISTANT_PROF, 2  # NSF allows max 2
        )
        assert not is_eligible
        assert len(issues) > 0
        assert "Maximum" in issues[0] and "applications" in issues[0]
    
    def test_is_researcher_eligible_industry_vs_nsf(self):
        """Test different eligibility requirements between agencies."""
        # Postdoc should be eligible for industry but not NSF
        nsf_eligible, nsf_issues = self.nsf_agency.is_researcher_eligible(
            ResearcherLevel.POSTDOC, 0
        )
        industry_eligible, industry_issues = self.industry_agency.is_researcher_eligible(
            ResearcherLevel.POSTDOC, 0
        )
        
        assert not nsf_eligible
        assert len(nsf_issues) > 0
        assert industry_eligible
        assert len(industry_issues) == 0


class TestFundingCycle:
    """Test cases for FundingCycle class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        today = date.today()
        self.cycle = FundingCycle(
            agency_id="test_agency",
            cycle_name="Test Cycle 2024",
            fiscal_year=2024,
            duration_years=2,
            total_budget=1000000,
            expected_awards=10,
            application_open_date=today,
            application_deadline=today + timedelta(days=90),
            review_start_date=today + timedelta(days=100),
            notification_date=today + timedelta(days=180),
            funding_start_date=today + timedelta(days=210),
            funding_end_date=today + timedelta(days=940)  # ~2.5 years
        )
    
    def test_initialization(self):
        """Test FundingCycle initialization."""
        assert self.cycle.cycle_name == "Test Cycle 2024"
        assert self.cycle.fiscal_year == 2024
        assert self.cycle.duration_years == 2
        assert self.cycle.total_budget == 1000000
        assert self.cycle.expected_awards == 10
        assert self.cycle.status == FundingCycleStatus.PLANNING
        assert len(self.cycle.applications) == 0
        assert len(self.cycle.awarded_applications) == 0
    
    def test_initialization_invalid_timeline(self):
        """Test FundingCycle initialization with invalid timeline."""
        today = date.today()
        with pytest.raises(ValidationError):
            FundingCycle(
                agency_id="test",
                application_open_date=today,
                application_deadline=today - timedelta(days=1),  # Invalid: deadline before open
                review_start_date=today + timedelta(days=10),
                notification_date=today + timedelta(days=20),
                funding_start_date=today + timedelta(days=30),
                funding_end_date=today + timedelta(days=40)
            )
    
    def test_is_application_open_true(self):
        """Test is_application_open when applications are open."""
        # Test during application period
        test_date = self.cycle.application_open_date + timedelta(days=30)
        assert self.cycle.is_application_open(test_date)
        
        # Set status to OPEN
        self.cycle.status = FundingCycleStatus.OPEN
        assert self.cycle.is_application_open(test_date)
    
    def test_is_application_open_false_before(self):
        """Test is_application_open before applications open."""
        test_date = self.cycle.application_open_date - timedelta(days=1)
        assert not self.cycle.is_application_open(test_date)
    
    def test_is_application_open_false_after(self):
        """Test is_application_open after deadline."""
        test_date = self.cycle.application_deadline + timedelta(days=1)
        assert not self.cycle.is_application_open(test_date)
    
    def test_is_application_open_false_wrong_status(self):
        """Test is_application_open with wrong status."""
        self.cycle.status = FundingCycleStatus.COMPLETED
        test_date = self.cycle.application_open_date + timedelta(days=30)
        assert not self.cycle.is_application_open(test_date)
    
    def test_get_days_until_deadline_positive(self):
        """Test get_days_until_deadline with positive days."""
        test_date = self.cycle.application_deadline - timedelta(days=30)
        days_left = self.cycle.get_days_until_deadline(test_date)
        assert days_left == 30
    
    def test_get_days_until_deadline_negative(self):
        """Test get_days_until_deadline with negative days (past deadline)."""
        test_date = self.cycle.application_deadline + timedelta(days=10)
        days_left = self.cycle.get_days_until_deadline(test_date)
        assert days_left == -10
    
    def test_calculate_success_rate_no_applications(self):
        """Test calculate_success_rate with no applications."""
        success_rate = self.cycle.calculate_success_rate()
        assert success_rate == 0.0
    
    def test_calculate_success_rate_with_applications(self):
        """Test calculate_success_rate with applications."""
        # Add some applications
        self.cycle.applications = ["app1", "app2", "app3", "app4", "app5"]
        self.cycle.awarded_applications = ["app1", "app3"]  # 2 out of 5
        
        success_rate = self.cycle.calculate_success_rate()
        assert success_rate == 0.4  # 2/5 = 0.4
    
    def test_get_average_award_amount_no_awards(self):
        """Test get_average_award_amount with no awards."""
        avg_amount = self.cycle.get_average_award_amount()
        assert avg_amount == 0.0
    
    def test_get_average_award_amount_with_awards(self):
        """Test get_average_award_amount with awards."""
        self.cycle.awarded_applications = ["app1", "app2", "app3", "app4"]  # 4 awards
        avg_amount = self.cycle.get_average_award_amount()
        assert avg_amount == 250000.0  # 1,000,000 / 4 = 250,000


class TestFundingApplication:
    """Test cases for FundingApplication class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.application = FundingApplication(
            cycle_id="test_cycle",
            researcher_id="researcher_001",
            title="Test Research Project",
            requested_amount=150000,
            duration_years=2
        )
    
    def test_initialization(self):
        """Test FundingApplication initialization."""
        assert self.application.cycle_id == "test_cycle"
        assert self.application.researcher_id == "researcher_001"
        assert self.application.title == "Test Research Project"
        assert self.application.requested_amount == 150000
        assert self.application.duration_years == 2
        assert self.application.status == ApplicationStatus.DRAFT
        assert not self.application.is_awarded
        assert self.application.awarded_amount == 0
        assert len(self.application.review_scores) == 0
    
    def test_calculate_final_score_no_scores(self):
        """Test calculate_final_score with no review scores."""
        final_score = self.application.calculate_final_score()
        assert final_score == 0.0
        assert self.application.final_score == 0.0
    
    def test_calculate_final_score_with_scores(self):
        """Test calculate_final_score with review scores."""
        self.application.review_scores = [4.5, 3.8, 4.2, 4.0]
        final_score = self.application.calculate_final_score()
        expected = (4.5 + 3.8 + 4.2 + 4.0) / 4
        assert abs(final_score - expected) < 0.001
        assert abs(self.application.final_score - expected) < 0.001
    
    def test_submit_application(self):
        """Test submit_application method."""
        test_date = date(2024, 3, 15)
        self.application.submit_application(test_date)
        
        assert self.application.submission_date == test_date
        assert self.application.status == ApplicationStatus.SUBMITTED
    
    def test_submit_application_default_date(self):
        """Test submit_application with default date."""
        self.application.submit_application()
        
        assert self.application.submission_date is not None
        assert self.application.status == ApplicationStatus.SUBMITTED
    
    def test_start_review(self):
        """Test start_review method."""
        test_date = date(2024, 4, 1)
        self.application.start_review(test_date)
        
        assert self.application.review_start_date == test_date
        assert self.application.status == ApplicationStatus.UNDER_REVIEW
    
    def test_make_decision_awarded(self):
        """Test make_decision for awarded application."""
        test_date = date(2024, 6, 1)
        self.application.make_decision(True, 140000, test_date)
        
        assert self.application.decision_date == test_date
        assert self.application.is_awarded
        assert self.application.awarded_amount == 140000
        assert self.application.status == ApplicationStatus.AWARDED
    
    def test_make_decision_rejected(self):
        """Test make_decision for rejected application."""
        test_date = date(2024, 6, 1)
        self.application.make_decision(False, 0, test_date)
        
        assert self.application.decision_date == test_date
        assert not self.application.is_awarded
        assert self.application.awarded_amount == 0
        assert self.application.status == ApplicationStatus.REJECTED


class TestFundingSystem:
    """Test cases for FundingSystem class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.funding_system = FundingSystem()
    
    def test_initialization(self):
        """Test FundingSystem initialization."""
        assert len(self.funding_system.agencies) >= 3  # Should have NSF, NIH, Industry
        assert len(self.funding_system.cycles) == 0
        assert len(self.funding_system.applications) == 0
    
    def test_default_agencies_created(self):
        """Test that default agencies are created."""
        nsf_agencies = self.funding_system.get_agencies_by_type(FundingAgencyType.NSF)
        nih_agencies = self.funding_system.get_agencies_by_type(FundingAgencyType.NIH)
        industry_agencies = self.funding_system.get_agencies_by_type(FundingAgencyType.INDUSTRY)
        
        assert len(nsf_agencies) >= 1
        assert len(nih_agencies) >= 1
        assert len(industry_agencies) >= 1
        
        # Check NSF characteristics
        nsf = nsf_agencies[0]
        assert nsf.name == "National Science Foundation"
        assert nsf.typical_cycle_duration_years == 3
        assert nsf.success_rate == 0.20
        assert nsf.min_career_level == ResearcherLevel.ASSISTANT_PROF
        
        # Check NIH characteristics
        nih = nih_agencies[0]
        assert nih.name == "National Institutes of Health"
        assert nih.typical_cycle_duration_years == 2
        assert nih.success_rate == 0.18
        assert nih.min_career_level == ResearcherLevel.POSTDOC
        
        # Check Industry characteristics
        industry = industry_agencies[0]
        assert industry.name == "Industry Research Consortium"
        assert industry.typical_cycle_duration_years == 1
        assert industry.success_rate == 0.35
        assert industry.min_career_level == ResearcherLevel.POSTDOC
    
    def test_register_agency(self):
        """Test registering a new agency."""
        new_agency = FundingAgency(
            name="Test Foundation",
            agency_type=FundingAgencyType.PRIVATE_FOUNDATION,
            typical_cycle_duration_years=2,
            success_rate=0.25
        )
        
        agency_id = self.funding_system.register_agency(new_agency)
        
        assert agency_id == new_agency.agency_id
        assert self.funding_system.get_agency(agency_id) == new_agency
    
    def test_get_agency_valid(self):
        """Test getting agency by valid ID."""
        nsf_agencies = self.funding_system.get_agencies_by_type(FundingAgencyType.NSF)
        nsf = nsf_agencies[0]
        
        retrieved_agency = self.funding_system.get_agency(nsf.agency_id)
        assert retrieved_agency == nsf
    
    def test_get_agency_invalid(self):
        """Test getting agency by invalid ID."""
        retrieved_agency = self.funding_system.get_agency("invalid_id")
        assert retrieved_agency is None
    
    def test_get_agencies_by_type(self):
        """Test getting agencies by type."""
        nsf_agencies = self.funding_system.get_agencies_by_type(FundingAgencyType.NSF)
        nih_agencies = self.funding_system.get_agencies_by_type(FundingAgencyType.NIH)
        
        assert len(nsf_agencies) >= 1
        assert len(nih_agencies) >= 1
        assert all(agency.agency_type == FundingAgencyType.NSF for agency in nsf_agencies)
        assert all(agency.agency_type == FundingAgencyType.NIH for agency in nih_agencies)
    
    def test_create_funding_cycle(self):
        """Test creating a funding cycle."""
        nsf_agencies = self.funding_system.get_agencies_by_type(FundingAgencyType.NSF)
        nsf = nsf_agencies[0]
        
        cycle_id = self.funding_system.create_funding_cycle(
            agency_id=nsf.agency_id,
            cycle_name="NSF Test Cycle 2024",
            fiscal_year=2024,
            total_budget=2000000,
            expected_awards=15
        )
        
        cycle = self.funding_system.get_cycle(cycle_id)
        assert cycle is not None
        assert cycle.cycle_name == "NSF Test Cycle 2024"
        assert cycle.fiscal_year == 2024
        assert cycle.total_budget == 2000000
        assert cycle.expected_awards == 15
        assert cycle.agency_id == nsf.agency_id
        assert cycle.duration_years == nsf.typical_cycle_duration_years
    
    def test_create_funding_cycle_invalid_agency(self):
        """Test creating funding cycle with invalid agency."""
        with pytest.raises(ValidationError):
            self.funding_system.create_funding_cycle(
                agency_id="invalid_agency",
                cycle_name="Invalid Cycle"
            )
    
    def test_get_active_cycles_empty(self):
        """Test getting active cycles when none exist."""
        active_cycles = self.funding_system.get_active_cycles()
        assert len(active_cycles) == 0
    
    def test_get_active_cycles_with_cycles(self):
        """Test getting active cycles with existing cycles."""
        nsf_agencies = self.funding_system.get_agencies_by_type(FundingAgencyType.NSF)
        nsf = nsf_agencies[0]
        
        # Create cycle and set to OPEN
        cycle_id = self.funding_system.create_funding_cycle(
            agency_id=nsf.agency_id,
            cycle_name="Active Cycle"
        )
        cycle = self.funding_system.get_cycle(cycle_id)
        cycle.status = FundingCycleStatus.OPEN
        
        active_cycles = self.funding_system.get_active_cycles()
        assert len(active_cycles) == 1
        assert active_cycles[0].cycle_id == cycle_id
    
    def test_get_active_cycles_filtered_by_type(self):
        """Test getting active cycles filtered by agency type."""
        nsf_agencies = self.funding_system.get_agencies_by_type(FundingAgencyType.NSF)
        industry_agencies = self.funding_system.get_agencies_by_type(FundingAgencyType.INDUSTRY)
        
        # Create NSF cycle
        nsf_cycle_id = self.funding_system.create_funding_cycle(
            agency_id=nsf_agencies[0].agency_id,
            cycle_name="NSF Cycle"
        )
        nsf_cycle = self.funding_system.get_cycle(nsf_cycle_id)
        nsf_cycle.status = FundingCycleStatus.OPEN
        
        # Create Industry cycle
        industry_cycle_id = self.funding_system.create_funding_cycle(
            agency_id=industry_agencies[0].agency_id,
            cycle_name="Industry Cycle"
        )
        industry_cycle = self.funding_system.get_cycle(industry_cycle_id)
        industry_cycle.status = FundingCycleStatus.OPEN
        
        # Test filtering
        nsf_cycles = self.funding_system.get_active_cycles(FundingAgencyType.NSF)
        industry_cycles = self.funding_system.get_active_cycles(FundingAgencyType.INDUSTRY)
        
        assert len(nsf_cycles) == 1
        assert len(industry_cycles) == 1
        assert nsf_cycles[0].cycle_id == nsf_cycle_id
        assert industry_cycles[0].cycle_id == industry_cycle_id
    
    def test_submit_application(self):
        """Test submitting an application."""
        # Create cycle
        nsf_agencies = self.funding_system.get_agencies_by_type(FundingAgencyType.NSF)
        cycle_id = self.funding_system.create_funding_cycle(
            agency_id=nsf_agencies[0].agency_id,
            cycle_name="Test Cycle"
        )
        cycle = self.funding_system.get_cycle(cycle_id)
        cycle.status = FundingCycleStatus.OPEN
        
        # Submit application
        app_id = self.funding_system.submit_application(
            cycle_id=cycle_id,
            researcher_id="researcher_001",
            title="Test Research Project",
            requested_amount=200000
        )
        
        # Verify application
        application = self.funding_system.get_application(app_id)
        assert application is not None
        assert application.cycle_id == cycle_id
        assert application.researcher_id == "researcher_001"
        assert application.title == "Test Research Project"
        assert application.requested_amount == 200000
        assert application.status == ApplicationStatus.SUBMITTED
        
        # Verify cycle updated
        assert app_id in cycle.applications
    
    def test_submit_application_cycle_closed(self):
        """Test submitting application to closed cycle."""
        # Create cycle but don't open it
        nsf_agencies = self.funding_system.get_agencies_by_type(FundingAgencyType.NSF)
        cycle_id = self.funding_system.create_funding_cycle(
            agency_id=nsf_agencies[0].agency_id,
            cycle_name="Closed Cycle"
        )
        cycle = self.funding_system.get_cycle(cycle_id)
        cycle.status = FundingCycleStatus.COMPLETED  # Closed
        
        with pytest.raises(ValidationError):
            self.funding_system.submit_application(
                cycle_id=cycle_id,
                researcher_id="researcher_001",
                title="Test Project",
                requested_amount=100000
            )
    
    def test_get_researcher_applications(self):
        """Test getting applications for a researcher."""
        # Create cycle and submit applications
        nsf_agencies = self.funding_system.get_agencies_by_type(FundingAgencyType.NSF)
        cycle_id = self.funding_system.create_funding_cycle(
            agency_id=nsf_agencies[0].agency_id,
            cycle_name="Test Cycle"
        )
        cycle = self.funding_system.get_cycle(cycle_id)
        cycle.status = FundingCycleStatus.OPEN
        
        # Submit multiple applications
        app1_id = self.funding_system.submit_application(
            cycle_id=cycle_id,
            researcher_id="researcher_001",
            title="Project 1",
            requested_amount=100000
        )
        app2_id = self.funding_system.submit_application(
            cycle_id=cycle_id,
            researcher_id="researcher_001",
            title="Project 2",
            requested_amount=150000
        )
        app3_id = self.funding_system.submit_application(
            cycle_id=cycle_id,
            researcher_id="researcher_002",
            title="Project 3",
            requested_amount=120000
        )
        
        # Test getting all applications for researcher_001
        researcher_apps = self.funding_system.get_researcher_applications("researcher_001")
        assert len(researcher_apps) == 2
        app_ids = [app.application_id for app in researcher_apps]
        assert app1_id in app_ids
        assert app2_id in app_ids
        assert app3_id not in app_ids
        
        # Test filtering by status
        submitted_apps = self.funding_system.get_researcher_applications(
            "researcher_001", ApplicationStatus.SUBMITTED
        )
        assert len(submitted_apps) == 2
    
    def test_process_cycle_reviews(self):
        """Test processing reviews for a cycle."""
        # Create cycle with applications
        nsf_agencies = self.funding_system.get_agencies_by_type(FundingAgencyType.NSF)
        cycle_id = self.funding_system.create_funding_cycle(
            agency_id=nsf_agencies[0].agency_id,
            cycle_name="Review Cycle",
            total_budget=500000,
            expected_awards=3
        )
        cycle = self.funding_system.get_cycle(cycle_id)
        cycle.status = FundingCycleStatus.OPEN
        
        # Submit multiple applications
        app_ids = []
        for i in range(5):
            app_id = self.funding_system.submit_application(
                cycle_id=cycle_id,
                researcher_id=f"researcher_{i:03d}",
                title=f"Project {i+1}",
                requested_amount=100000
            )
            app_ids.append(app_id)
        
        # Process reviews
        results = self.funding_system.process_cycle_reviews(cycle_id)
        
        # Verify results
        assert results["cycle_id"] == cycle_id
        assert results["total_applications"] == 5
        assert results["applications_reviewed"] == 5
        assert results["applications_awarded"] <= 3  # Expected awards limit
        assert results["total_awarded_amount"] <= 500000  # Budget limit
        
        # Verify cycle status updated
        updated_cycle = self.funding_system.get_cycle(cycle_id)
        assert updated_cycle.status == FundingCycleStatus.AWARDED
        
        # Verify applications have decisions
        for app_id in app_ids:
            app = self.funding_system.get_application(app_id)
            assert app.status in [ApplicationStatus.AWARDED, ApplicationStatus.REJECTED]
            assert len(app.review_scores) > 0
            assert app.final_score > 0
    
    def test_get_funding_statistics(self):
        """Test getting comprehensive funding statistics."""
        # Create some cycles and applications
        nsf_agencies = self.funding_system.get_agencies_by_type(FundingAgencyType.NSF)
        cycle_id = self.funding_system.create_funding_cycle(
            agency_id=nsf_agencies[0].agency_id,
            cycle_name="Stats Cycle"
        )
        cycle = self.funding_system.get_cycle(cycle_id)
        cycle.status = FundingCycleStatus.OPEN
        
        # Submit applications
        for i in range(3):
            self.funding_system.submit_application(
                cycle_id=cycle_id,
                researcher_id=f"researcher_{i:03d}",
                title=f"Project {i+1}",
                requested_amount=100000 + i * 50000
            )
        
        stats = self.funding_system.get_funding_statistics()
        
        # Verify structure
        assert "total_agencies" in stats
        assert "total_cycles" in stats
        assert "total_applications" in stats
        assert "agency_breakdown" in stats
        assert "cycle_status_breakdown" in stats
        assert "application_status_breakdown" in stats
        assert "success_rates" in stats
        assert "funding_amounts" in stats
        
        # Verify values
        assert stats["total_agencies"] >= 3  # Default agencies
        assert stats["total_cycles"] >= 1
        assert stats["total_applications"] >= 3
        
        # Verify funding amounts
        funding_amounts = stats["funding_amounts"]
        assert funding_amounts["total_requested"] >= 450000  # 100k + 150k + 200k
        assert funding_amounts["average_request"] > 0
        
        # Verify breakdowns have expected keys
        assert FundingAgencyType.NSF.value in stats["agency_breakdown"]
        assert FundingAgencyType.NIH.value in stats["agency_breakdown"]
        assert FundingAgencyType.INDUSTRY.value in stats["agency_breakdown"]


if __name__ == "__main__":
    pytest.main([__file__])