#!/usr/bin/env python3
"""
Integration test for SalamiSlicingDetector

This script demonstrates the salami slicing detection functionality with realistic examples.
"""

from datetime import date, timedelta
from src.enhancements.salami_slicing_detector import SalamiSlicingDetector
from src.data.enhanced_models import VenueType


def test_salami_slicing_integration():
    """Test salami slicing detection with realistic examples."""
    
    print("=== Salami Slicing Detection Integration Test ===\n")
    
    # Initialize detector
    detector = SalamiSlicingDetector(
        similarity_threshold=0.6,  # Lower threshold for demo
        min_papers_for_pattern=3,
        max_time_span_months=18
    )
    
    # Create realistic salami slicing example
    base_date = date(2023, 1, 1)
    
    # Researcher with clear salami slicing pattern
    salami_papers = [
        {
            'paper_id': 'salami_1',
            'title': 'Deep Learning for Medical Image Analysis: Part I - Preprocessing',
            'abstract': 'This paper presents preprocessing techniques for medical images using deep learning. We focus on data cleaning and normalization methods.',
            'authors': ['researcher_salami'],
            'keywords': ['deep learning', 'medical imaging', 'preprocessing'],
            'venue_id': 'venue_top',
            'venue_type': VenueType.TOP_CONFERENCE,
            'publication_date': base_date,
            'primary_author': 'researcher_salami'
        },
        {
            'paper_id': 'salami_2',
            'title': 'Deep Learning for Medical Image Analysis: Part II - Feature Extraction',
            'abstract': 'Building on our preprocessing work, this paper presents feature extraction techniques for medical images using deep learning methods.',
            'authors': ['researcher_salami'],
            'keywords': ['deep learning', 'medical imaging', 'feature extraction'],
            'venue_id': 'venue_mid',
            'venue_type': VenueType.MID_CONFERENCE,
            'publication_date': base_date + timedelta(days=90),
            'primary_author': 'researcher_salami'
        },
        {
            'paper_id': 'salami_3',
            'title': 'Deep Learning for Medical Image Analysis: Part III - Classification',
            'abstract': 'This paper completes our series by presenting classification techniques for medical images using our preprocessing and feature extraction methods.',
            'authors': ['researcher_salami'],
            'keywords': ['deep learning', 'medical imaging', 'classification'],
            'venue_id': 'venue_low',
            'venue_type': VenueType.LOW_CONFERENCE,
            'publication_date': base_date + timedelta(days=180),
            'primary_author': 'researcher_salami'
        },
        {
            'paper_id': 'salami_4',
            'title': 'Deep Learning Medical Image Analysis: Extended Results',
            'abstract': 'This paper presents extended experimental results for our deep learning approach to medical image analysis with additional datasets.',
            'authors': ['researcher_salami'],
            'keywords': ['deep learning', 'medical imaging', 'experiments'],
            'venue_id': 'venue_workshop',
            'venue_type': VenueType.WORKSHOP,
            'publication_date': base_date + timedelta(days=270),
            'primary_author': 'researcher_salami'
        }
    ]
    
    # Normal researcher with diverse work
    normal_papers = [
        {
            'paper_id': 'normal_1',
            'title': 'Quantum Computing Applications in Cryptography',
            'abstract': 'This paper explores novel applications of quantum computing principles in modern cryptographic systems.',
            'authors': ['researcher_normal'],
            'keywords': ['quantum computing', 'cryptography', 'security'],
            'venue_id': 'venue_top2',
            'venue_type': VenueType.TOP_JOURNAL,
            'publication_date': base_date,
            'primary_author': 'researcher_normal'
        },
        {
            'paper_id': 'normal_2',
            'title': 'Blockchain Technology for Supply Chain Management',
            'abstract': 'We present a comprehensive blockchain-based solution for supply chain transparency and traceability.',
            'authors': ['researcher_normal'],
            'keywords': ['blockchain', 'supply chain', 'transparency'],
            'venue_id': 'venue_spec',
            'venue_type': VenueType.SPECIALIZED_JOURNAL,
            'publication_date': base_date + timedelta(days=120),
            'primary_author': 'researcher_normal'
        },
        {
            'paper_id': 'normal_3',
            'title': 'Machine Learning for Climate Change Prediction',
            'abstract': 'This work applies advanced machine learning techniques to improve climate change prediction models.',
            'authors': ['researcher_normal'],
            'keywords': ['machine learning', 'climate change', 'prediction'],
            'venue_id': 'venue_gen',
            'venue_type': VenueType.GENERAL_JOURNAL,
            'publication_date': base_date + timedelta(days=240),
            'primary_author': 'researcher_normal'
        }
    ]
    
    # Add papers to detector
    print("Adding papers to detector...")
    for paper in salami_papers + normal_papers:
        detector.add_paper(**paper)
    
    print(f"Added {len(salami_papers + normal_papers)} papers total\n")
    
    # Analyze all researchers
    print("Analyzing researchers for salami slicing patterns...")
    all_patterns = detector.analyze_all_researchers()
    
    print(f"Found patterns for {len(all_patterns)} researchers\n")
    
    # Check salami researcher
    print("=== Salami Researcher Analysis ===")
    salami_patterns = detector.detect_salami_patterns('researcher_salami')
    salami_profile = detector.get_researcher_salami_behavior('researcher_salami')
    
    if salami_patterns:
        print(f"Detected {len(salami_patterns)} salami slicing patterns")
        for i, pattern in enumerate(salami_patterns, 1):
            print(f"  Pattern {i}:")
            print(f"    Type: {pattern.pattern_type}")
            print(f"    Strength: {pattern.strength_score:.3f}")
            print(f"    Papers: {len(pattern.paper_ids)}")
            print(f"    Time span: {pattern.time_span_months} months")
            print(f"    Content similarity: {pattern.content_similarity:.3f}")
            print(f"    Indicators: {', '.join(pattern.suspicious_indicators)}")
    else:
        print("No salami slicing patterns detected")
    
    if salami_profile:
        print(f"\nProfile:")
        print(f"  Risk level: {salami_profile.risk_level}")
        print(f"  Salami rate: {salami_profile.incremental_publication_score:.3f}")
        print(f"  Publication frequency: {salami_profile.publication_frequency:.1f} papers/year")
    
    # Check normal researcher
    print("\n=== Normal Researcher Analysis ===")
    normal_patterns = detector.detect_salami_patterns('researcher_normal')
    normal_profile = detector.get_researcher_salami_behavior('researcher_normal')
    
    if normal_patterns:
        print(f"Detected {len(normal_patterns)} patterns (unexpected)")
    else:
        print("No salami slicing patterns detected (expected)")
    
    if normal_profile:
        print(f"\nProfile:")
        print(f"  Risk level: {normal_profile.risk_level}")
        print(f"  Salami rate: {normal_profile.incremental_publication_score:.3f}")
        print(f"  Publication frequency: {normal_profile.publication_frequency:.1f} papers/year")
    
    # Generate comprehensive report
    print("\n=== Overall Statistics ===")
    stats = detector.calculate_salami_statistics()
    
    print(f"Total patterns detected: {stats['total_patterns']}")
    print(f"Total researchers analyzed: {stats['total_researchers']}")
    print(f"Papers involved in patterns: {stats['total_papers_involved']}")
    print(f"High-risk researchers: {stats['high_risk_researchers']}")
    
    if stats['pattern_types']:
        print("Pattern types:")
        for pattern_type, count in stats['pattern_types'].items():
            print(f"  {pattern_type}: {count}")
    
    print("\nResearchers by risk level:")
    for risk_level, count in stats['researchers_by_risk'].items():
        print(f"  {risk_level}: {count}")
    
    # Generate researcher-specific report
    print("\n=== Detailed Report for Salami Researcher ===")
    report = detector.generate_salami_report('researcher_salami')
    
    if 'error' not in report:
        profile = report['profile']
        print(f"Total papers: {profile['total_papers']}")
        print(f"Suspected salami papers: {profile['suspected_salami_papers']}")
        print(f"Salami rate: {profile['salami_rate']:.3f}")
        print(f"Risk level: {profile['risk_level']}")
        
        if report['recommendations']:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
    
    print("\n=== Integration Test Complete ===")
    return detector


if __name__ == '__main__':
    test_salami_slicing_integration()