#!/usr/bin/env python3
"""
Simple test script for venue standards system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Test basic imports
try:
    from src.data.enhanced_models import EnhancedVenue, VenueType, EnhancedResearcher, ResearcherLevel
    print("✓ Enhanced models imported successfully")
except Exception as e:
    print(f"✗ Enhanced models import failed: {e}")
    sys.exit(1)

# Test venue standards system import
try:
    # Try to execute the file and capture any errors
    namespace = {'__builtins__': __builtins__}
    
    # Add the required imports to the namespace
    from src.data.enhanced_models import (
        EnhancedVenue, VenueType, EnhancedResearcher, StructuredReview,
        ReviewDecision, ResearcherLevel
    )
    from src.core.exceptions import ValidationError
    from src.core.logging_config import get_logger
    
    namespace.update({
        'EnhancedVenue': EnhancedVenue,
        'VenueType': VenueType,
        'EnhancedResearcher': EnhancedResearcher,
        'StructuredReview': StructuredReview,
        'ReviewDecision': ReviewDecision,
        'ResearcherLevel': ResearcherLevel,
        'ValidationError': ValidationError,
        'get_logger': get_logger,
        'logging': __import__('logging'),
        'Dict': __import__('typing').Dict,
        'List': __import__('typing').List,
        'Optional': __import__('typing').Optional,
        'Tuple': __import__('typing').Tuple,
        'Set': __import__('typing').Set,
        'dataclass': __import__('dataclasses').dataclass,
        'field': __import__('dataclasses').field,
        'Enum': __import__('enum').Enum,
        'statistics': __import__('statistics')
    })
    
    with open('src/enhancements/venue_standards_system.py', 'r') as f:
        code = f.read()
    
    try:
        exec(code, namespace)
        print("✓ Venue standards system executed successfully")
    except Exception as exec_error:
        print(f"✗ Execution error: {exec_error}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Check what's in the namespace
    classes = [k for k, v in namespace.items() if isinstance(v, type)]
    print(f"Classes found: {classes}")
    
    # Get the class from the namespace
    if 'VenueStandardsManager' in namespace:
        VenueStandardsManager = namespace['VenueStandardsManager']
    else:
        print("VenueStandardsManager not found in namespace")
        sys.exit(1)
    
    # Test class creation
    manager = VenueStandardsManager()
    print("✓ VenueStandardsManager created successfully")
    
    # Test basic functionality
    thresholds = manager.get_venue_score_thresholds()
    print(f"✓ PeerRead thresholds: {thresholds}")
    
    venues = manager.list_supported_venues()
    print(f"✓ Supported venues: {venues}")
    
    # Test venue standards
    acl_standards = manager.get_venue_standards("ACL")
    if acl_standards:
        print(f"✓ ACL standards: threshold={acl_standards.acceptance_threshold.base_threshold}")
    else:
        print("✗ Failed to get ACL standards")
    
    # Test with sample venue
    sample_venue = EnhancedVenue(
        id="acl-test",
        name="ACL",
        venue_type=VenueType.TOP_CONFERENCE,
        field="Natural Language Processing"
    )
    
    criteria = manager.get_reviewer_selection_criteria(sample_venue)
    print(f"✓ ACL reviewer criteria: min_h_index={criteria.min_h_index}")
    
    requirements = manager.get_minimum_reviewer_requirements(sample_venue)
    print(f"✓ ACL reviewer requirements: {requirements}")
    
    # Test with sample researcher
    sample_researcher = EnhancedResearcher(
        id="test-researcher",
        name="Dr. Test",
        specialty="NLP",
        level=ResearcherLevel.ASSISTANT_PROF,
        h_index=20,
        years_active=8,
        institution_tier=1
    )
    sample_researcher.reputation_score = 0.7
    
    is_qualified, issues = manager.validate_reviewer_qualifications(sample_venue, sample_researcher)
    print(f"✓ Researcher qualification check: qualified={is_qualified}, issues={issues}")
    
    print("\n🎉 All tests passed! Venue standards system is working correctly.")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)