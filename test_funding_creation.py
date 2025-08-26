#!/usr/bin/env python3

from src.enhancements.funding_system import FundingSystem, FundingAgency, FundingAgencyType
from src.data.enhanced_models import ResearcherLevel

try:
    print("Testing FundingSystem creation...")
    
    # Test creating a simple agency first
    print("1. Creating simple agency...")
    agency = FundingAgency(
        name="Test Agency",
        agency_type=FundingAgencyType.NSF
    )
    print(f"   Created agency: {agency.name}")
    
    # Test creating funding system
    print("2. Creating FundingSystem...")
    fs = FundingSystem()
    print(f"   Created FundingSystem")
    
    # Test registering agency
    print("3. Registering agency...")
    agency_id = fs.register_agency(agency)
    print(f"   Registered agency with ID: {agency_id}")
    print(f"   Total agencies: {len(fs.agencies)}")
    
    # Test creating default agencies manually
    print("4. Testing default agency creation...")
    fs._create_default_agencies()
    print(f"   After creating defaults: {len(fs.agencies)} agencies")
    
    for agency_id, agency in fs.agencies.items():
        print(f"   - {agency.name} ({agency.agency_type.value})")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()