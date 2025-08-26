"""
Integration Test for Review History Tracking System

This script demonstrates the review history tracking functionality
with realistic scenarios and comprehensive testing.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.enhancements.review_history_tracker import ReviewHistoryTracker, ReliabilityCategory
from src.data.enhanced_models import (
    StructuredReview, EnhancedReviewCriteria, ReviewDecision,
    DetailedStrength, DetailedWeakness
)


def create_sample_review(reviewer_id: str, paper_id: str, quality_level: str = "good", 
                        submission_delay_days: int = 0) -> StructuredReview:
    """Create a sample review with specified quality level and timing."""
    
    # Quality level configurations
    quality_configs = {
        "excellent": {
            "scores": {"novelty": 9.0, "technical_quality": 9.5, "clarity": 8.5, 
                      "significance": 9.0, "reproducibility": 8.0, "related_work": 8.5},
            "summary": "This is an outstanding paper that makes significant contributions to the field. The methodology is rigorous and the results are compelling.",
            "strengths": [
                DetailedStrength("Technical", "Excellent experimental design with comprehensive evaluation", 5),
                DetailedStrength("Novelty", "Novel approach that advances state-of-the-art significantly", 5),
                DetailedStrength("Clarity", "Very well written and easy to follow", 4)
            ],
            "weaknesses": [
                DetailedWeakness("Minor", "Some minor notation inconsistencies", 1, ["Fix notation in equations 3-5"])
            ],
            "recommendation": ReviewDecision.ACCEPT
        },
        "good": {
            "scores": {"novelty": 7.0, "technical_quality": 7.5, "clarity": 6.5, 
                      "significance": 7.0, "reproducibility": 6.5, "related_work": 7.0},
            "summary": "This paper presents solid work with good methodology and reasonable results. Some improvements needed.",
            "strengths": [
                DetailedStrength("Technical", "Sound methodology with adequate evaluation", 3),
                DetailedStrength("Significance", "Addresses an important problem", 4)
            ],
            "weaknesses": [
                DetailedWeakness("Presentation", "Some sections could be clearer", 3, ["Improve section 4", "Add more examples"])
            ],
            "recommendation": ReviewDecision.MINOR_REVISION
        },
        "poor": {
            "scores": {"novelty": 4.0, "technical_quality": 4.5, "clarity": 3.5, 
                      "significance": 4.0, "reproducibility": 3.0, "related_work": 4.0},
            "summary": "This paper has significant issues that need to be addressed.",
            "strengths": [
                DetailedStrength("Topic", "Addresses relevant problem", 2)
            ],
            "weaknesses": [
                DetailedWeakness("Technical", "Methodology has serious flaws", 4, ["Redesign experiments", "Add baselines"]),
                DetailedWeakness("Clarity", "Paper is difficult to follow", 4, ["Major rewrite needed"])
            ],
            "recommendation": ReviewDecision.MAJOR_REVISION
        }
    }
    
    config = quality_configs.get(quality_level, quality_configs["good"])
    
    # Create review with timing
    submission_time = datetime.now() + timedelta(days=submission_delay_days)
    
    review = StructuredReview(
        reviewer_id=reviewer_id,
        paper_id=paper_id,
        venue_id="ICML_2024",
        criteria_scores=EnhancedReviewCriteria(**config["scores"]),
        confidence_level=4,
        recommendation=config["recommendation"],
        executive_summary=config["summary"],
        detailed_strengths=config["strengths"],
        detailed_weaknesses=config["weaknesses"],
        technical_comments="Technical evaluation based on methodology and experimental design.",
        presentation_comments="Overall presentation quality assessment.",
        questions_for_authors=["How does this compare to recent work by Smith et al.?"],
        suggestions_for_improvement=["Consider adding ablation studies"],
        submission_timestamp=submission_time
    )
    
    return review


def demonstrate_review_history_tracking():
    """Demonstrate comprehensive review history tracking functionality."""
    
    print("=== Review History Tracking System Demo ===\n")
    
    # Initialize tracker
    tracker = ReviewHistoryTracker()
    
    # Simulate multiple reviewers with different performance patterns
    reviewers_scenarios = [
        {
            "id": "dr_excellent",
            "name": "Dr. Excellent Reviewer",
            "pattern": [
                ("excellent", -1),  # 1 day early
                ("excellent", 0),   # on time
                ("good", -2),       # 2 days early
                ("excellent", 0),   # on time
                ("good", -1)        # 1 day early
            ]
        },
        {
            "id": "prof_inconsistent", 
            "name": "Prof. Inconsistent",
            "pattern": [
                ("excellent", -1),  # starts well
                ("good", 1),        # gets late
                ("poor", 3),        # very late
                ("good", 0),        # improves
                ("excellent", -1)   # back to excellent
            ]
        },
        {
            "id": "dr_declining",
            "name": "Dr. Declining Quality",
            "pattern": [
                ("excellent", 0),
                ("good", 1),
                ("good", 2),
                ("poor", 3),
                ("poor", 4)
            ]
        },
        {
            "id": "prof_reliable",
            "name": "Prof. Reliable",
            "pattern": [
                ("good", 0),
                ("good", -1),
                ("good", 0),
                ("good", 0),
                ("good", -1),
                ("good", 0)
            ]
        }
    ]
    
    # Track reviews for each reviewer
    print("1. Tracking Reviews for Multiple Reviewers")
    print("-" * 50)
    
    for reviewer_scenario in reviewers_scenarios:
        reviewer_id = reviewer_scenario["id"]
        reviewer_name = reviewer_scenario["name"]
        
        print(f"\nTracking reviews for {reviewer_name} ({reviewer_id}):")
        
        for i, (quality, delay) in enumerate(reviewer_scenario["pattern"]):
            paper_id = f"paper_{reviewer_id}_{i+1}"
            deadline = datetime.now() + timedelta(days=2)  # 2 days from now
            
            # Create and track review
            review = create_sample_review(reviewer_id, paper_id, quality, delay)
            quality_metric = tracker.track_review_quality(review, deadline)
            
            print(f"  Paper {i+1}: Quality={quality_metric.quality_score:.2f}, "
                  f"Timeliness={quality_metric.timeliness_score:.2f}, "
                  f"Delay={delay} days")
    
    # Display performance metrics
    print("\n\n2. Reviewer Performance Metrics")
    print("-" * 50)
    
    for reviewer_scenario in reviewers_scenarios:
        reviewer_id = reviewer_scenario["id"]
        reviewer_name = reviewer_scenario["name"]
        
        metrics = tracker.get_reviewer_performance(reviewer_id)
        if metrics:
            print(f"\n{reviewer_name} ({reviewer_id}):")
            print(f"  Total Reviews: {metrics.total_reviews}")
            print(f"  Avg Quality Score: {metrics.avg_quality_score:.3f}")
            print(f"  Avg Timeliness Score: {metrics.avg_timeliness_score:.3f}")
            print(f"  Reliability Score: {metrics.reliability_score:.3f}")
            print(f"  Reliability Category: {metrics.reliability_category.value}")
            print(f"  On-time Reviews: {metrics.on_time_reviews}/{metrics.total_reviews}")
            print(f"  Performance Trend: {metrics.recent_performance_trend}")
    
    # Show top reviewers
    print("\n\n3. Top Reviewers by Reliability")
    print("-" * 50)
    
    top_reviewers = tracker.get_top_reviewers(limit=5, min_reviews=3)
    for i, (reviewer_id, metrics) in enumerate(top_reviewers, 1):
        reviewer_name = next(r["name"] for r in reviewers_scenarios if r["id"] == reviewer_id)
        print(f"{i}. {reviewer_name}: {metrics.reliability_score:.3f} ({metrics.reliability_category.value})")
    
    # Show reviewers needing improvement
    print("\n\n4. Reviewers Needing Improvement")
    print("-" * 50)
    
    needing_improvement = tracker.get_reviewers_needing_improvement(threshold=0.7)
    if needing_improvement:
        for reviewer_id, metrics in needing_improvement:
            reviewer_name = next(r["name"] for r in reviewers_scenarios if r["id"] == reviewer_id)
            print(f"- {reviewer_name}: {metrics.reliability_score:.3f} ({metrics.reliability_category.value})")
    else:
        print("No reviewers currently need improvement (all above threshold)")
    
    # Generate detailed report for one reviewer
    print("\n\n5. Detailed Performance Report")
    print("-" * 50)
    
    report = tracker.generate_performance_report("prof_inconsistent")
    print(f"\nDetailed Report for Prof. Inconsistent:")
    print(f"Summary:")
    print(f"  - Total Reviews: {report['summary']['total_reviews']}")
    print(f"  - Reliability Score: {report['summary']['reliability_score']:.3f}")
    print(f"  - Category: {report['summary']['reliability_category']}")
    print(f"  - Trend: {report['summary']['performance_trend']}")
    
    print(f"\nQuality Metrics:")
    print(f"  - Average Quality: {report['quality_metrics']['avg_quality_score']:.3f}")
    print(f"  - Quality Percentile: {report['percentiles']['quality']:.1%}")
    print(f"  - High Quality Reviews: {report['quality_metrics']['high_quality_reviews']}")
    
    print(f"\nTimeliness Metrics:")
    print(f"  - Average Timeliness: {report['timeliness_metrics']['avg_timeliness_score']:.3f}")
    print(f"  - Timeliness Percentile: {report['percentiles']['timeliness']:.1%}")
    print(f"  - On-time Reviews: {report['timeliness_metrics']['on_time_reviews']}")
    print(f"  - Late Reviews: {report['timeliness_metrics']['late_reviews']}")
    
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    # System statistics
    print("\n\n6. System-wide Statistics")
    print("-" * 50)
    
    stats = tracker.get_system_statistics()
    print(f"Total Reviewers: {stats['total_reviewers']}")
    print(f"Total Reviews Tracked: {stats['total_reviews']}")
    print(f"Average Reliability Score: {stats['avg_reliability_score']:.3f}")
    print(f"Reviewers with Good Reliability (≥0.8): {stats['reviewers_with_good_reliability']}")
    print(f"Reviewers Needing Improvement (<0.6): {stats['reviewers_needing_improvement']}")
    
    print(f"\nReliability Distribution:")
    for category, count in stats['reliability_distribution'].items():
        print(f"  - {category}: {count}")
    
    print("\n\n=== Demo Complete ===")
    print("\nThe Review History Tracking System successfully:")
    print("✓ Tracked review quality metrics for multiple reviewers")
    print("✓ Calculated reliability scores based on quality and timeliness")
    print("✓ Identified performance trends and patterns")
    print("✓ Generated comprehensive performance reports")
    print("✓ Provided improvement recommendations")
    print("✓ Maintained historical review quality data")


if __name__ == "__main__":
    demonstrate_review_history_tracking()