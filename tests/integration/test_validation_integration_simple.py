#!/usr/bin/env python3
"""
Simple integration test for the ValidationMetrics framework.
This demonstrates the key functionality of the real data validation framework.
"""

import sys
import os
# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from unittest.mock import Mock
from src.data.validation_metrics import ValidationMetrics, BaselineStatistics
from src.data.peerread_loader import PeerReadLoader, VenueCharacteristics, PeerReadReview, PeerReadPaper
from src.data.enhanced_models import (
    StructuredReview, EnhancedResearcher, EnhancedVenue, 
    EnhancedReviewCriteria, ResearcherLevel, VenueType
)


def create_mock_peerread_loader():
    """Create a mock PeerReadLoader with realistic data."""
    loader = Mock(spec=PeerReadLoader)
    
    # Create realistic venue characteristics for ACL
    acl_chars = VenueCharacteristics(
        name="ACL",
        venue_type="conference",
        field="NLP",
        total_papers=200,
        accepted_papers=50,
        acceptance_rate=0.25,
        avg_reviews_per_paper=3.1,
        impact_scores=[3, 3, 4, 3, 4, 2, 3, 4, 3, 3, 4, 2, 3, 3, 4] * 10,  # ~3.2 mean
        substance_scores=[3, 4, 3, 4, 3, 3, 4, 3, 4, 3, 4, 3, 3, 4, 3] * 10,  # ~3.4 mean
        impact_mean=3.2,
        substance_mean=3.4
    )
    
    # Create realistic venue characteristics for NIPS
    nips_chars = VenueCharacteristics(
        name="NIPS",
        venue_type="conference", 
        field="AI",
        total_papers=500,
        accepted_papers=100,
        acceptance_rate=0.20,
        avg_reviews_per_paper=3.5,
        impact_scores=[4, 4, 3, 4, 5, 3, 4, 4, 3, 4, 5, 3, 4, 4, 3] * 10,  # ~3.8 mean
        substance_scores=[4, 3, 4, 4, 3, 4, 4, 3, 4, 3, 4, 4, 3, 4, 4] * 10,  # ~3.7 mean
        impact_mean=3.8,
        substance_mean=3.7
    )
    
    loader.load_all_venues.return_value = {"ACL": acl_chars, "NIPS": nips_chars}
    loader.get_venue_statistics.side_effect = lambda name: {"ACL": acl_chars, "NIPS": nips_chars}.get(name)
    
    # Create mock papers with reviews
    def create_mock_papers(venue_name, venue_chars):
        papers = []
        for i in range(5):  # 5 papers per venue
            paper = PeerReadPaper(f"{venue_name.lower()}_paper_{i}", f"{venue_name} Paper {i}", f"Abstract for {venue_name} paper {i}")
            paper.venue = venue_name
            
            # Add 3 reviews per paper
            for j in range(3):
                review = PeerReadReview(f"{venue_name.lower()}_paper_{i}", f"reviewer_{j}")
                
                # Use realistic score distributions
                if venue_name == "ACL":
                    review.impact = np.random.choice([2, 3, 3, 4, 4], p=[0.1, 0.4, 0.3, 0.15, 0.05])
                    review.substance = np.random.choice([2, 3, 3, 4, 4], p=[0.05, 0.35, 0.35, 0.2, 0.05])
                else:  # NIPS
                    review.impact = np.random.choice([3, 4, 4, 5, 5], p=[0.2, 0.3, 0.3, 0.15, 0.05])
                    review.substance = np.random.choice([3, 4, 4, 4, 5], p=[0.15, 0.35, 0.3, 0.15, 0.05])
                
                review.comments = f"This is a review for {venue_name} paper {i}. " * np.random.randint(15, 40)
                review.reviewer_confidence = np.random.randint(3, 5)
                review.recommendation = np.random.randint(1, 4)
                paper.reviews.append(review)
            
            papers.append(paper)
        return papers
    
    loader.get_papers_by_venue.side_effect = lambda name: create_mock_papers(name, {"ACL": acl_chars, "NIPS": nips_chars}.get(name))
    
    return loader


def create_simulation_data():
    """Create realistic simulation data for testing."""
    
    # Create simulation reviews
    reviews = []
    for i in range(20):
        review = StructuredReview(
            reviewer_id=f"sim_reviewer_{i}",
            paper_id=f"sim_paper_{i % 10}",
            venue_id="ACL"
        )
        
        # Set realistic review content
        review.executive_summary = f"This paper presents interesting work in area {i}. " * 3
        review.technical_comments = f"The technical approach is sound with some minor issues. " * 5
        review.presentation_comments = f"The paper is generally well written. " * 2
        
        # Set realistic scores (mapped from PeerRead 1-5 to our 1-10 scale)
        review.criteria_scores = EnhancedReviewCriteria(
            novelty=np.random.normal(6.4, 1.6),  # ~3.2 * 2
            technical_quality=np.random.normal(6.8, 1.8),  # ~3.4 * 2
            clarity=np.random.normal(6.0, 1.2),
            significance=np.random.normal(6.4, 1.6),
            reproducibility=np.random.normal(6.0, 1.4),
            related_work=np.random.normal(5.8, 1.2)
        )
        
        review._calculate_review_length()
        reviews.append(review)
    
    # Create simulation researchers
    researchers = []
    for i in range(10):
        researcher = EnhancedResearcher(
            id=f"sim_researcher_{i}",
            name=f"Researcher {i}",
            specialty="NLP",
            level=np.random.choice(list(ResearcherLevel)),
            h_index=np.random.randint(5, 25),
            years_active=np.random.randint(3, 20)
        )
        researchers.append(researcher)
    
    # Create simulation venues
    venues = [
        EnhancedVenue(
            id="sim_acl",
            name="ACL",
            venue_type=VenueType.MID_CONFERENCE,
            field="NLP",
            acceptance_rate=0.26  # Slightly different from real 0.25
        )
    ]
    
    return reviews, researchers, venues


def test_statistical_comparisons():
    """Test statistical comparison functionality."""
    print("=== Testing Statistical Comparisons ===")
    
    # Create validation metrics with mock data
    mock_loader = create_mock_peerread_loader()
    validation_metrics = ValidationMetrics(mock_loader)
    
    print(f"Initialized with {len(validation_metrics.baseline_statistics)} venues")
    
    # Create simulation data that's similar to real data
    simulation_data = {
        'impact': list(np.random.normal(3.3, 0.8, 50)),  # Close to ACL baseline of 3.2
        'substance': list(np.random.normal(3.5, 0.9, 50))  # Close to ACL baseline of 3.4
    }
    
    # Compare to ACL baseline
    comparisons = validation_metrics.compare_simulation_to_real(simulation_data, "ACL")
    
    print(f"\nComparisons for ACL:")
    for metric_name, comparison in comparisons.items():
        print(f"  {metric_name}:")
        print(f"    Simulation mean: {comparison.simulation_mean:.2f}")
        print(f"    Real mean: {comparison.real_mean:.2f}")
        print(f"    Wasserstein distance: {comparison.wasserstein_distance:.3f}")
        print(f"    Similarity score: {comparison.similarity_score:.3f}")
        print(f"    Deviation level: {comparison.deviation_level}")
        print(f"    Is similar: {comparison.is_similar}")
    
    return validation_metrics


def test_realism_indicators(validation_metrics):
    """Test realism indicator functionality."""
    print("\n=== Testing Realism Indicators ===")
    
    # Create simulation data
    reviews, researchers, venues = create_simulation_data()
    
    # Create realism indicators
    indicators = validation_metrics.create_realism_indicators(reviews, researchers, venues)
    
    print(f"Created indicators for {len(indicators)} aspects:")
    
    for aspect_name, aspect_indicators in indicators.items():
        print(f"\n  {aspect_name} ({len(aspect_indicators)} indicators):")
        for indicator in aspect_indicators[:3]:  # Show first 3 indicators
            print(f"    {indicator.indicator_type}:")
            print(f"      Current: {indicator.current_value:.2f}")
            print(f"      Expected: {indicator.expected_value:.2f}")
            print(f"      Deviation: {indicator.deviation_percentage:.1%}")
            print(f"      Alert level: {indicator.alert_level}")
            print(f"      Is realistic: {indicator.is_realistic}")


def test_continuous_monitoring(validation_metrics):
    """Test continuous monitoring and alert generation."""
    print("\n=== Testing Continuous Monitoring ===")
    
    # Enable monitoring
    validation_metrics.enable_continuous_monitoring(True)
    
    # Create simulation data with significant deviation to trigger alerts
    problematic_simulation_data = {
        'impact': list(np.random.normal(5.0, 0.5, 30)),  # Much higher than ACL baseline of 3.2
        'substance': list(np.random.normal(2.0, 0.3, 30))  # Much lower than ACL baseline of 3.4
    }
    
    print("Comparing problematic simulation data to trigger alerts...")
    
    # This should generate alerts
    comparisons = validation_metrics.compare_simulation_to_real(problematic_simulation_data, "ACL")
    
    # Check alerts
    recent_alerts = validation_metrics.get_recent_alerts(1)  # Last hour
    print(f"\nGenerated {len(recent_alerts)} alerts:")
    
    for alert in recent_alerts:
        print(f"  {alert.alert_level}: {alert.message}")
        print(f"    Deviation: {alert.deviation_percentage:.1%}")


def test_baseline_statistics(validation_metrics):
    """Test baseline statistics functionality."""
    print("\n=== Testing Baseline Statistics ===")
    
    # Get baseline statistics
    acl_baseline = validation_metrics.baseline_statistics.get("ACL")
    if acl_baseline:
        print(f"ACL Baseline Statistics:")
        print(f"  Acceptance rate: {acl_baseline.acceptance_rate:.1%}")
        print(f"  Reviews per paper: {acl_baseline.reviews_per_paper_mean:.1f}")
        print(f"  Review length mean: {acl_baseline.review_length_mean:.0f} chars")
        print(f"  Score means: {acl_baseline.score_means}")
    
    nips_baseline = validation_metrics.baseline_statistics.get("NIPS")
    if nips_baseline:
        print(f"\nNIPS Baseline Statistics:")
        print(f"  Acceptance rate: {nips_baseline.acceptance_rate:.1%}")
        print(f"  Reviews per paper: {nips_baseline.reviews_per_paper_mean:.1f}")
        print(f"  Score means: {nips_baseline.score_means}")


def test_export_import():
    """Test export and import functionality."""
    print("\n=== Testing Export/Import ===")
    
    mock_loader = create_mock_peerread_loader()
    validation_metrics = ValidationMetrics(mock_loader)
    
    # Export baseline statistics
    baseline_file = "test_baseline_stats.json"
    validation_metrics.export_baseline_statistics(baseline_file)
    print(f"Exported baseline statistics to {baseline_file}")
    
    # Export validation report
    report_file = "test_validation_report.json"
    validation_metrics.export_validation_report(report_file)
    print(f"Exported validation report to {report_file}")
    
    # Test import
    new_validation_metrics = ValidationMetrics()
    new_validation_metrics.import_baseline_statistics(baseline_file)
    print(f"Imported baseline statistics: {len(new_validation_metrics.baseline_statistics)} venues")
    
    # Clean up
    import os
    try:
        os.remove(baseline_file)
        os.remove(report_file)
        print("Cleaned up test files")
    except:
        pass


def main():
    """Run all validation framework tests."""
    print("Real Data Validation Framework Integration Test")
    print("=" * 50)
    
    try:
        # Test statistical comparisons
        validation_metrics = test_statistical_comparisons()
        
        # Test realism indicators
        test_realism_indicators(validation_metrics)
        
        # Test continuous monitoring
        test_continuous_monitoring(validation_metrics)
        
        # Test baseline statistics
        test_baseline_statistics(validation_metrics)
        
        # Test export/import
        test_export_import()
        
        print("\n" + "=" * 50)
        print("✅ All validation framework tests completed successfully!")
        print("\nKey Features Demonstrated:")
        print("- Statistical comparison utilities (KL divergence, Wasserstein distance, correlations)")
        print("- Baseline statistics calculation from PeerRead training data")
        print("- Continuous validation monitoring with automated alerts")
        print("- Realism indicators for review quality, reviewer behavior, and venue characteristics")
        print("- Export/import functionality for baseline statistics and reports")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())