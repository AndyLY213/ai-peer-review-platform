#!/usr/bin/env python3
"""
Unit tests for funding system (simplified version for task 9.1)
"""

import pytest
from datetime import date, timedelta

from test_minimal_funding import FundingSystem, FundingAgency, FundingCycle, FundingAgencyType
from src.data.enhanced_models import ResearcherLevel
from src.core.exceptions import ValidationError


def test_funding_agency_initialization():
    """Test FundingAgency initialization."""
    agency = FundingAgency(
        name="Test Agency",
        agency_type=FundingAgencyType.NSF,
        typical_cycle_duration_years=2,
        success_rate=0.25
    )
    assert agency.name == "Test Agency"
    assert agency.agency_type == FundingAgencyType.NSF
    assert agency.typical_cycle_duration_years == 2
    assert agency.success_rate == 0.25


def test_funding_agency_validation():
    """Test FundingAgency validation."""
    # Test invalid cycle duration
    with pytest.raises(ValidationError):
        FundingAgency(
            name="Invalid Agency",
            typical_cycle_duration_years=5  # Invalid: must be 1-3
        )
    
    # Test invalid success rate
    with pytest.raises(ValidationError):
        FundingAgency(
            name="Invalid Agency",
            success_rate=1.5  # Invalid: must be 0.0-1.0
        )


def test_funding_cycle_initialization():
    """Test FundingCycle initialization."""
    cycle = FundingCycle(
        agency_id="test_agency",
        cycle_name="Test Cycle",
        duration_years=2,
        total_budget=500000,
        expected_awards=5
    )
    assert cycle.cycle_name == "Test Cycle"
    assert cycle.duration_years == 2
    assert cycle.total_budget == 500000
    assert cycle.expected_awards == 5


def test_funding_system_initialization():
    """Test FundingSystem initialization."""
    funding_system = FundingSystem()
    assert len(funding_system.agencies) >= 3  # Should have NSF, NIH, Industry
    assert len(funding_system.cycles) == 0


def test_funding_system_agency_management():
    """Test funding system agency management."""
    funding_system = FundingSystem()
    
    # Test getting agencies by type
    nsf_agencies = funding_system.get_agencies_by_type(FundingAgencyType.NSF)
    nih_agencies = funding_system.get_agencies_by_type(FundingAgencyType.NIH)
    industry_agencies = funding_system.get_agencies_by_type(FundingAgencyType.INDUSTRY)
    
    assert len(nsf_agencies) >= 1
    assert len(nih_agencies) >= 1
    assert len(industry_agencies) >= 1


def test_funding_system_cycle_creation():
    """Test funding system cycle creation."""
    funding_system = FundingSystem()
    
    # Get NSF agency
    nsf_agencies = funding_system.get_agencies_by_type(FundingAgencyType.NSF)
    nsf = nsf_agencies[0]
    
    # Create cycle
    cycle_id = funding_system.create_funding_cycle(
        agency_id=nsf.agency_id,
        cycle_name="Test Cycle 2024",
        total_budget=1000000,
        expected_awards=10
    )
    
    # Verify cycle
    cycle = funding_system.get_cycle(cycle_id)
    assert cycle is not None
    assert cycle.cycle_name == "Test Cycle 2024"
    assert cycle.total_budget == 1000000
    assert cycle.expected_awards == 10
    assert cycle.duration_years == nsf.typical_cycle_duration_years


def test_funding_system_invalid_cycle_creation():
    """Test funding system with invalid cycle creation."""
    funding_system = FundingSystem()
    
    # Test with invalid agency ID
    with pytest.raises(ValidationError):
        funding_system.create_funding_cycle(
            agency_id="invalid_agency",
            cycle_name="Invalid Cycle"
        )


if __name__ == "__main__":
    # Run tests manually
    print("Running unit tests...")
    
    test_funding_agency_initialization()
    print("✓ test_funding_agency_initialization")
    
    test_funding_agency_validation()
    print("✓ test_funding_agency_validation")
    
    test_funding_cycle_initialization()
    print("✓ test_funding_cycle_initialization")
    
    test_funding_system_initialization()
    print("✓ test_funding_system_initialization")
    
    test_funding_system_agency_management()
    print("✓ test_funding_system_agency_management")
    
    test_funding_system_cycle_creation()
    print("✓ test_funding_system_cycle_creation")
    
    test_funding_system_invalid_cycle_creation()
    print("✓ test_funding_system_invalid_cycle_creation")
    
    print("\nAll unit tests passed!")