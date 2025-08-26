#!/usr/bin/env python3
"""
Integration test for CitationCartelDetector with CitationNetwork.
"""

import sys
from pathlib import Path
from datetime import date

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.enhancements.citation_network import CitationNetwork
from src.enhancements.citation_cartel_detector import CitationCartelDetector
from src.data.enhanced_models import EnhancedResearcher, PublicationRecord, ResearcherLevel


def test_citation_cartel_integration():
    """Test integration between CitationNetwork and CitationCartelDetector."""
    print("Testing Citation Cartel Detection Integration...")
    
    # Create citation network
    citation_network = CitationNetwork()
    
    # Add some citation relationships that form suspicious patterns
    # Mutual citation pair
    citation_network.add_citation(
        citing_paper_id="paper_1",
        cited_paper_id="paper_2",
        citing_author_ids=["researcher_1"],
        cited_author_ids=["researcher_2"],
        citation_date=date(2022, 6, 1)
    )
    citation_network.add_citation(
        citing_paper_id="paper_2",
        cited_paper_id="paper_1",
        citing_author_ids=["researcher_2"],
        cited_author_ids=["researcher_1"],
        citation_date=date(2022, 8, 1)
    )
    
    # Add more citations to make it suspicious
    for i in range(3, 6):  # Add 3 more citations each way
        citation_network.add_citation(
            citing_paper_id=f"paper_{i}",
            cited_paper_id="paper_2",
            citing_author_ids=["researcher_1"],
            cited_author_ids=["researcher_2"],
            citation_date=date(2022, i, 1)
        )
        citation_network.add_citation(
            citing_paper_id=f"paper_{i+10}",
            cited_paper_id="paper_1",
            citing_author_ids=["researcher_2"],
            cited_author_ids=["researcher_1"],
            citation_date=date(2022, i, 15)
        )
    
    # Create a potential citation ring
    citation_network.add_citation(
        citing_paper_id="paper_20",
        cited_paper_id="paper_21",
        citing_author_ids=["researcher_3"],
        cited_author_ids=["researcher_4"],
        citation_date=date(2022, 1, 1)
    )
    citation_network.add_citation(
        citing_paper_id="paper_21",
        cited_paper_id="paper_22",
        citing_author_ids=["researcher_4"],
        cited_author_ids=["researcher_5"],
        citation_date=date(2022, 2, 1)
    )
    citation_network.add_citation(
        citing_paper_id="paper_22",
        cited_paper_id="paper_20",
        citing_author_ids=["researcher_5"],
        cited_author_ids=["researcher_3"],
        citation_date=date(2022, 3, 1)
    )
    
    # Add more citations to strengthen the ring
    for i in range(3):
        citation_network.add_citation(
            citing_paper_id=f"paper_{30+i}",
            cited_paper_id=f"paper_{21+i}",
            citing_author_ids=[f"researcher_{3+i}"],
            cited_author_ids=[f"researcher_{4+((i+1)%3)}"],
            citation_date=date(2022, 4+i, 1)
        )
    
    print(f"Created citation network with {len(citation_network.citation_records)} citations")
    
    # Create citation cartel detector
    detector = CitationCartelDetector(
        min_mutual_citations=3,
        min_ring_size=3,
        suspicion_threshold=0.6
    )
    
    # Load citation data
    detector.load_citation_data(citation_network)
    print(f"Loaded citation data: {len(detector.author_citations)} authors")
    
    # Create sample researchers
    researchers = [
        EnhancedResearcher(
            id=f"researcher_{i}",
            name=f"Dr. Researcher {i}",
            specialty="Computer Science",
            level=ResearcherLevel.ASSISTANT_PROF,
            institution_tier=2,
            h_index=10,
            total_citations=100,
            years_active=5,
            reputation_score=0.7,
            cognitive_biases={},
            review_behavior=None,
            strategic_behavior=None,
            career_stage=None,
            funding_status=None,
            publication_pressure=0.5,
            collaboration_network=set(),
            citation_network=set(),
            institutional_affiliations=[],
            review_quality_history=[],
            publication_history=[],
            career_milestones=[]
        )
        for i in range(1, 6)
    ]
    
    # Analyze citation patterns
    print("\nAnalyzing citation patterns...")
    analysis = detector.analyze_citation_patterns(researchers)
    
    print(f"Total researchers analyzed: {analysis['total_researchers']}")
    print(f"Mutual citation pairs detected: {analysis['suspicious_patterns']['mutual_pairs_count']}")
    print(f"High suspicion pairs: {analysis['suspicious_patterns']['high_suspicion_pairs']}")
    print(f"Citation rings detected: {analysis['suspicious_patterns']['citation_rings_count']}")
    print(f"High confidence rings: {analysis['suspicious_patterns']['high_confidence_rings']}")
    print(f"Total cartels detected: {analysis['cartel_analysis']['total_cartels_detected']}")
    
    # Generate detailed report
    print("\nGenerating cartel report...")
    report = detector.generate_cartel_report()
    
    print(f"Report summary:")
    print(f"  - Total cartels: {report['detection_summary']['total_cartels']}")
    print(f"  - Mutual pairs: {report['detection_summary']['mutual_pairs']}")
    print(f"  - Citation rings: {report['detection_summary']['citation_rings']}")
    print(f"  - High strength cartels: {report['detection_summary']['high_strength_cartels']}")
    
    # Show cartel details
    if report['cartel_details']:
        print(f"\nCartel details:")
        for i, cartel in enumerate(report['cartel_details'][:3]):  # Show first 3
            print(f"  Cartel {i+1}:")
            print(f"    Type: {cartel['type']}")
            print(f"    Members: {cartel['members']}")
            print(f"    Strength: {cartel['strength_score']:.3f}")
            print(f"    Citations: {cartel['total_citations']}")
    
    # Test researcher involvement
    if report['researcher_involvement']:
        print(f"\nResearcher involvement:")
        for researcher_id, involvement in list(report['researcher_involvement'].items())[:3]:
            print(f"  {researcher_id}: {involvement['total_cartels']} cartels, "
                  f"max strength: {involvement['max_strength_score']:.3f}")
    
    # Test serialization
    print(f"\nTesting serialization...")
    detector_dict = detector.to_dict()
    print(f"Serialized detector with {len(detector_dict['detected_cartels'])} cartels")
    
    # Test deserialization
    new_detector = CitationCartelDetector.from_dict(detector_dict)
    print(f"Deserialized detector with {len(new_detector.detected_cartels)} cartels")
    
    print(f"\n✅ Citation cartel detection integration test completed successfully!")
    
    return detector, report


if __name__ == "__main__":
    detector, report = test_citation_cartel_integration()