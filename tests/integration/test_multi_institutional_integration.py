"""
Integration test for Multi-Institutional Collaboration Bonus System

This test demonstrates the integration between the multi-institutional collaboration
system and the existing funding system, showing how collaboration bonuses affect
funding success rates and publication outcomes.
"""

import pytest
from datetime import date, timedelta
from src.enhancements.funding_system import FundingSystem, FundingAgency, FundingAgencyType
from src.enhancements.multi_institutional_collaboration import (
    MultiInstitutionalCollaborationSystem,
    InstitutionProfile,
    CollaborationType
)
from src.data.enhanced_models import ResearcherLevel, EnhancedResearcher


def test_multi_institutional_collaboration_integration():
    """Test integration between collaboration system and funding system."""
    
    # Initialize systems
    funding_system = FundingSystem()
    collaboration_system = MultiInstitutionalCollaborationSystem()
    
    # Create institutions
    mit = InstitutionProfile(
        name="MIT",
        tier=1,
        country="USA",
        institution_type="Academic",
        research_strengths=["AI", "Robotics"],
        reputation_score=0.95
    )
    
    stanford = InstitutionProfile(
        name="Stanford University",
        tier=1,
        country="USA",
        institution_type="Academic",
        research_strengths=["AI", "Computer Science"],
        reputation_score=0.93
    )
    
    oxford = InstitutionProfile(
        name="Oxford University",
        tier=1,
        country="UK",
        institution_type="Academic",
        research_strengths=["AI", "Philosophy"],
        reputation_score=0.92
    )
    
    # Register institutions
    mit_id = collaboration_system.register_institution(mit)
    stanford_id = collaboration_system.register_institution(stanford)
    oxford_id = collaboration_system.register_institution(oxford)
    
    # Create researchers
    researcher_1 = "researcher_1"
    researcher_2 = "researcher_2"
    researcher_3 = "researcher_3"
    
    print("=== Multi-Institutional Collaboration Integration Test ===")
    print(f"Created institutions: MIT, Stanford, Oxford")
    print(f"Created researchers: {researcher_1}, {researcher_2}, {researcher_3}")
    
    # Test 1: Bilateral domestic collaboration
    print("\n--- Test 1: Bilateral Domestic Collaboration ---")
    
    bilateral_project_id = collaboration_system.create_collaboration_project(
        title="AI Safety Research",
        description="Joint research on AI safety between MIT and Stanford",
        participating_institutions=[mit_id, stanford_id],
        participating_researchers=[researcher_1, researcher_2],
        lead_institution=mit_id,
        lead_researcher=researcher_1,
        research_areas=["AI", "Safety"],
        total_funding=500000
    )
    
    # Activate project
    collaboration_system.activate_project(bilateral_project_id)
    
    # Check bonuses
    researcher_1_bonuses = collaboration_system.get_researcher_bonuses(researcher_1)
    funding_multiplier_1 = collaboration_system.calculate_funding_success_multiplier(researcher_1)
    publication_multiplier_1 = collaboration_system.calculate_publication_success_multiplier(researcher_1)
    
    print(f"Researcher 1 funding multiplier: {funding_multiplier_1:.3f}")
    print(f"Researcher 1 publication multiplier: {publication_multiplier_1:.3f}")
    print(f"Researcher 1 bonuses: {len(researcher_1_bonuses)}")
    
    # Test 2: International collaboration
    print("\n--- Test 2: International Collaboration ---")
    
    international_project_id = collaboration_system.create_collaboration_project(
        title="Global AI Ethics Initiative",
        description="International collaboration on AI ethics between MIT and Oxford",
        participating_institutions=[mit_id, oxford_id],
        participating_researchers=[researcher_1, researcher_3],
        lead_institution=mit_id,
        lead_researcher=researcher_1,
        research_areas=["AI", "Ethics"],
        total_funding=750000
    )
    
    # Activate project
    collaboration_system.activate_project(international_project_id)
    
    # Check updated bonuses for researcher_1 (now has two collaborations)
    researcher_1_bonuses_updated = collaboration_system.get_researcher_bonuses(researcher_1)
    funding_multiplier_1_updated = collaboration_system.calculate_funding_success_multiplier(researcher_1)
    publication_multiplier_1_updated = collaboration_system.calculate_publication_success_multiplier(researcher_1)
    
    print(f"Researcher 1 updated funding multiplier: {funding_multiplier_1_updated:.3f}")
    print(f"Researcher 1 updated publication multiplier: {publication_multiplier_1_updated:.3f}")
    print(f"Researcher 1 total bonuses: {len(researcher_1_bonuses_updated)}")
    
    # International collaboration should provide higher bonuses
    assert funding_multiplier_1_updated > funding_multiplier_1
    assert publication_multiplier_1_updated > publication_multiplier_1
    
    # Test 3: Integration with funding system
    print("\n--- Test 3: Funding System Integration ---")
    
    # Get NSF agency
    nsf_agencies = funding_system.get_agencies_by_type(FundingAgencyType.NSF)
    nsf_agency = nsf_agencies[0] if nsf_agencies else None
    
    if nsf_agency:
        # Create funding cycle
        cycle_id = funding_system.create_funding_cycle(
            agency_id=nsf_agency.agency_id,
            cycle_name="AI Research Initiative 2024",
            total_budget=2000000,
            expected_awards=20
        )
        
        # Simulate funding applications with collaboration bonuses
        base_success_rate = nsf_agency.success_rate
        
        # Researcher with no collaborations
        researcher_no_collab = "researcher_no_collab"
        no_collab_multiplier = collaboration_system.calculate_funding_success_multiplier(researcher_no_collab)
        
        # Calculate effective success rates
        researcher_1_effective_rate = base_success_rate * funding_multiplier_1_updated
        no_collab_effective_rate = base_success_rate * no_collab_multiplier
        
        print(f"Base NSF success rate: {base_success_rate:.3f}")
        print(f"Researcher 1 effective success rate: {researcher_1_effective_rate:.3f}")
        print(f"No collaboration researcher effective rate: {no_collab_effective_rate:.3f}")
        print(f"Collaboration advantage: {(researcher_1_effective_rate / no_collab_effective_rate - 1) * 100:.1f}%")
        
        # The researcher with collaborations should have a significant advantage
        assert researcher_1_effective_rate > no_collab_effective_rate
        advantage = (researcher_1_effective_rate / no_collab_effective_rate - 1) * 100
        assert advantage > 20  # At least 20% advantage
    
    # Test 4: Project completion and outcome bonuses
    print("\n--- Test 4: Project Completion and Outcomes ---")
    
    # Complete the bilateral project with successful outcomes
    outcomes = {
        "publications": ["paper_1", "paper_2", "paper_3"],
        "patents": ["patent_1"],
        "other_outcomes": ["best_paper_award"]
    }
    
    collaboration_system.complete_project(bilateral_project_id, outcomes)
    
    # Check project status
    bilateral_project = collaboration_system.projects[bilateral_project_id]
    print(f"Project status: {bilateral_project.status.value}")
    print(f"Publications: {len(bilateral_project.publications)}")
    print(f"Patents: {len(bilateral_project.patents)}")
    
    # Test 5: Collaboration partner suggestions
    print("\n--- Test 5: Collaboration Partner Suggestions ---")
    
    suggestions = collaboration_system.suggest_collaboration_partners(
        researcher_id=researcher_2,
        research_areas=["AI", "Machine Learning", "Ethics"],
        max_suggestions=3
    )
    
    print(f"Collaboration suggestions for researcher_2:")
    for i, (inst_id, score) in enumerate(suggestions, 1):
        institution = collaboration_system.get_institution(inst_id)
        print(f"  {i}. {institution.name} (score: {score:.3f})")
    
    # Test 6: System statistics
    print("\n--- Test 6: System Statistics ---")
    
    stats = collaboration_system.get_collaboration_statistics()
    print(f"Total institutions: {stats['total_institutions']}")
    print(f"Total projects: {stats['total_projects']}")
    print(f"Active projects: {stats['active_projects']}")
    print(f"Completed projects: {stats['completed_projects']}")
    print(f"Total bonuses: {stats['total_bonuses']}")
    print(f"Average funding bonus: {stats['average_funding_bonus']:.3f}")
    print(f"Average publication bonus: {stats['average_publication_bonus']:.3f}")
    print(f"Total funding: ${stats['total_funding']:,}")
    
    # Verify statistics
    assert stats['total_institutions'] == 3
    assert stats['total_projects'] == 2
    assert stats['active_projects'] == 1  # One completed, one active
    assert stats['completed_projects'] == 1
    assert stats['total_bonuses'] > 0
    assert stats['average_funding_bonus'] > 0
    
    print("\n=== Integration Test Completed Successfully ===")
    
    # Return key metrics for verification
    return {
        "funding_multiplier_improvement": funding_multiplier_1_updated - 1.0,
        "publication_multiplier_improvement": publication_multiplier_1_updated - 1.0,
        "collaboration_advantage_percent": advantage if 'advantage' in locals() else 0,
        "total_bonuses": stats['total_bonuses'],
        "projects_created": stats['total_projects']
    }


def test_collaboration_bonus_calculations():
    """Test specific collaboration bonus calculation scenarios."""
    
    collaboration_system = MultiInstitutionalCollaborationSystem()
    
    # Create different types of institutions
    institutions = [
        InstitutionProfile(
            name="Top Academic US",
            tier=1,
            country="USA",
            institution_type="Academic",
            reputation_score=0.95
        ),
        InstitutionProfile(
            name="Industry Partner",
            tier=1,
            country="USA",
            institution_type="Industry",
            reputation_score=0.85
        ),
        InstitutionProfile(
            name="International Academic",
            tier=2,
            country="Germany",
            institution_type="Academic",
            reputation_score=0.80
        ),
        InstitutionProfile(
            name="Government Lab",
            tier=2,
            country="USA",
            institution_type="Government",
            reputation_score=0.75
        )
    ]
    
    # Register institutions
    inst_ids = []
    for inst in institutions:
        inst_id = collaboration_system.register_institution(inst)
        inst_ids.append(inst_id)
    
    print("\n=== Collaboration Bonus Calculation Tests ===")
    
    # Test different collaboration types
    collaboration_scenarios = [
        {
            "name": "Bilateral Academic",
            "institutions": [inst_ids[0], inst_ids[2]],  # US Academic + International Academic
            "type": "International Academic"
        },
        {
            "name": "Industry-Academic",
            "institutions": [inst_ids[0], inst_ids[1]],  # Academic + Industry
            "type": "Industry-Academic"
        },
        {
            "name": "Government-Academic",
            "institutions": [inst_ids[0], inst_ids[3]],  # Academic + Government
            "type": "Government-Academic"
        },
        {
            "name": "Multi-lateral International",
            "institutions": [inst_ids[0], inst_ids[1], inst_ids[2]],  # US Academic + Industry + International
            "type": "Complex Multi-lateral"
        }
    ]
    
    results = []
    
    for scenario in collaboration_scenarios:
        print(f"\n--- {scenario['name']} Collaboration ---")
        
        project_id = collaboration_system.create_collaboration_project(
            title=f"{scenario['name']} Project",
            description=f"Test {scenario['name']} collaboration",
            participating_institutions=scenario['institutions'],
            participating_researchers=["test_researcher"],
            lead_institution=scenario['institutions'][0],
            lead_researcher="test_researcher",
            research_areas=["AI", "Research"],
            total_funding=500000
        )
        
        collaboration_system.activate_project(project_id)
        
        bonuses = collaboration_system.get_researcher_bonuses("test_researcher")
        latest_bonus = bonuses[-1]  # Get the most recent bonus
        
        funding_multiplier = collaboration_system.calculate_funding_success_multiplier("test_researcher")
        publication_multiplier = collaboration_system.calculate_publication_success_multiplier("test_researcher")
        
        print(f"  Funding bonus: {latest_bonus.funding_success_bonus:.3f}")
        print(f"  Publication bonus: {latest_bonus.publication_success_bonus:.3f}")
        print(f"  Reputation bonus: {latest_bonus.reputation_bonus:.3f}")
        print(f"  Network bonus: {latest_bonus.network_expansion_bonus:.3f}")
        print(f"  Total bonus value: {latest_bonus.get_total_bonus_value():.3f}")
        print(f"  Funding multiplier: {funding_multiplier:.3f}")
        print(f"  Publication multiplier: {publication_multiplier:.3f}")
        
        results.append({
            "scenario": scenario['name'],
            "funding_bonus": latest_bonus.funding_success_bonus,
            "publication_bonus": latest_bonus.publication_success_bonus,
            "total_bonus": latest_bonus.get_total_bonus_value(),
            "funding_multiplier": funding_multiplier,
            "publication_multiplier": publication_multiplier
        })
        
        # Clear bonuses for next test
        collaboration_system.bonuses.clear()
        collaboration_system.collaboration_history.clear()
    
    # Verify that more complex collaborations get higher bonuses
    bilateral_result = next(r for r in results if r['scenario'] == 'Bilateral Academic')
    multilateral_result = next(r for r in results if r['scenario'] == 'Multi-lateral International')
    
    assert multilateral_result['total_bonus'] > bilateral_result['total_bonus']
    print(f"\nVerified: Multi-lateral collaboration bonus ({multilateral_result['total_bonus']:.3f}) > "
          f"Bilateral collaboration bonus ({bilateral_result['total_bonus']:.3f})")
    
    return results


if __name__ == "__main__":
    # Run integration test
    metrics = test_multi_institutional_collaboration_integration()
    
    print(f"\nFinal Integration Test Metrics:")
    print(f"  Funding multiplier improvement: {metrics['funding_multiplier_improvement']:.3f}")
    print(f"  Publication multiplier improvement: {metrics['publication_multiplier_improvement']:.3f}")
    print(f"  Collaboration advantage: {metrics['collaboration_advantage_percent']:.1f}%")
    print(f"  Total bonuses created: {metrics['total_bonuses']}")
    print(f"  Projects created: {metrics['projects_created']}")
    
    # Run bonus calculation tests
    bonus_results = test_collaboration_bonus_calculations()
    
    print(f"\nBonus Calculation Test Results:")
    for result in bonus_results:
        print(f"  {result['scenario']}: Total bonus = {result['total_bonus']:.3f}")