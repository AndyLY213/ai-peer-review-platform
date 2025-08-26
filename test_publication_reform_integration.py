"""
Integration test for Publication Reform Manager

Tests the complete publication reform system workflow including
reform implementation, evaluation model creation, and impact assessment.
"""

import tempfile
import shutil
from datetime import datetime, date
from pathlib import Path

from src.enhancements.publication_reform_manager import (
    PublicationReformManager, ReformType, MetricType, AdoptionStage
)
from src.data.enhanced_models import (
    StructuredReview, EnhancedReviewCriteria, VenueType, ReviewDecision
)


def test_publication_reform_integration():
    """Test complete publication reform workflow."""
    print("Testing Publication Reform Manager Integration...")
    
    # Setup
    temp_dir = tempfile.mkdtemp()
    manager = PublicationReformManager(data_dir=temp_dir)
    
    try:
        # Test data
        venue_id = "acl_2024"
        paper_id = "paper_nlp_001"
        reviewer_id = "reviewer_senior_001"
        
        print(f"✓ Initialized PublicationReformManager with data dir: {temp_dir}")
        
        # 1. Implement a reform
        reform_id = manager.implement_reform(
            venue_id=venue_id,
            reform_type=ReformType.ALTERNATIVE_METRICS,
            parameters={"pilot_duration": 12, "threshold": 0.7}
        )
        print(f"✓ Implemented Alternative Metrics reform: {reform_id}")
        
        # 2. Create an evaluation model
        model_id = manager.create_evaluation_model(
            name="ACL Hybrid Model",
            description="Combines traditional peer review with alternative metrics",
            traditional_weight=0.6,
            alternative_weight=0.3,
            post_pub_weight=0.1,
            included_metrics=[
                MetricType.ALTMETRIC_SCORE,
                MetricType.DOWNLOAD_COUNT,
                MetricType.SOCIAL_MEDIA_MENTIONS
            ]
        )
        print(f"✓ Created evaluation model: {model_id}")
        
        # 3. Add alternative metrics for a paper
        manager.add_alternative_metric(
            paper_id=paper_id,
            metric_type=MetricType.ALTMETRIC_SCORE,
            value=85.5,
            source="Altmetric API",
            confidence=0.9
        )
        
        manager.add_alternative_metric(
            paper_id=paper_id,
            metric_type=MetricType.DOWNLOAD_COUNT,
            value=2500,
            source="ArXiv",
            confidence=1.0
        )
        
        manager.add_alternative_metric(
            paper_id=paper_id,
            metric_type=MetricType.SOCIAL_MEDIA_MENTIONS,
            value=150,
            source="Twitter API",
            confidence=0.8
        )
        print("✓ Added alternative metrics for paper")
        
        # 4. Submit post-publication reviews
        review_id_1 = manager.submit_post_publication_review(
            paper_id=paper_id,
            reviewer_id=reviewer_id,
            rating=8.5,
            summary="Excellent work with novel approach to transformer attention",
            strengths=["Novel attention mechanism", "Comprehensive evaluation"],
            weaknesses=["Limited theoretical analysis"],
            is_anonymous=False
        )
        
        review_id_2 = manager.submit_post_publication_review(
            paper_id=paper_id,
            reviewer_id="reviewer_community_001",
            rating=7.8,
            summary="Good contribution but could be improved",
            strengths=["Clear presentation", "Reproducible results"],
            weaknesses=["Limited novelty", "Missing baselines"],
            is_anonymous=True
        )
        print(f"✓ Submitted post-publication reviews: {review_id_1}, {review_id_2}")
        
        # 5. Create traditional reviews for comparison
        traditional_reviews = [
            StructuredReview(
                reviewer_id="traditional_reviewer_1",
                paper_id=paper_id,
                venue_id=venue_id,
                criteria_scores=EnhancedReviewCriteria(
                    novelty=8.0,
                    technical_quality=7.5,
                    clarity=8.5,
                    significance=7.0,
                    reproducibility=6.5,
                    related_work=7.5
                ),
                confidence_level=4,
                recommendation=ReviewDecision.ACCEPT
            ),
            StructuredReview(
                reviewer_id="traditional_reviewer_2",
                paper_id=paper_id,
                venue_id=venue_id,
                criteria_scores=EnhancedReviewCriteria(
                    novelty=7.5,
                    technical_quality=8.0,
                    clarity=7.0,
                    significance=8.5,
                    reproducibility=7.0,
                    related_work=6.5
                ),
                confidence_level=3,
                recommendation=ReviewDecision.MINOR_REVISION
            )
        ]
        print("✓ Created traditional reviews for comparison")
        
        # 6. Evaluate paper using the hybrid model
        evaluation_result = manager.evaluate_paper_with_model(
            paper_id=paper_id,
            model_id=model_id,
            traditional_reviews=traditional_reviews
        )
        
        print("✓ Paper evaluation results:")
        print(f"  - Final Score: {evaluation_result['final_score']:.2f}")
        print(f"  - Traditional Score: {evaluation_result['traditional_score']:.2f}")
        print(f"  - Alternative Metrics Score: {evaluation_result['alternative_metrics_score']:.2f}")
        print(f"  - Post-Publication Score: {evaluation_result['post_publication_score']:.2f}")
        
        # 7. Track reform adoption over time
        adoption_stages = [
            (0.1, 0.6, "Initial pilot phase"),
            (0.3, 0.7, "Expanding adoption"),
            (0.6, 0.8, "Widespread implementation"),
            (0.8, 0.85, "Near universal adoption")
        ]
        
        for adoption_rate, satisfaction, description in adoption_stages:
            manager.track_reform_adoption(
                reform_id=reform_id,
                adoption_rate=adoption_rate,
                researcher_satisfaction=satisfaction,
                quality_improvement=0.1 + (adoption_rate * 0.2),
                efficiency_gain=0.05 + (adoption_rate * 0.15)
            )
            print(f"✓ Tracked adoption: {description} (rate={adoption_rate}, satisfaction={satisfaction})")
        
        # 8. Assess reform impact
        impact_assessment = manager.assess_reform_impact(reform_id)
        
        print("✓ Reform impact assessment:")
        print(f"  - Reform Type: {impact_assessment['reform_type']}")
        print(f"  - Current Stage: {impact_assessment['current_stage']}")
        print(f"  - Adoption Rate: {impact_assessment['adoption_rate']:.1%}")
        print(f"  - Researcher Satisfaction: {impact_assessment['researcher_satisfaction']:.1%}")
        print(f"  - Quality Improvement: {impact_assessment['quality_improvement']:.1%}")
        print(f"  - Efficiency Gain: {impact_assessment['efficiency_gain']:.1%}")
        print(f"  - Overall Impact Score: {impact_assessment['overall_impact_score']:.3f}")
        print(f"  - Adoption Trend: {impact_assessment['adoption_trend']}")
        
        # 9. Get reform recommendations for different venue types
        venue_types = [VenueType.TOP_CONFERENCE, VenueType.MID_CONFERENCE, VenueType.TOP_JOURNAL]
        
        for venue_type in venue_types:
            recommendations = manager.get_reform_recommendations(
                venue_id=f"test_venue_{venue_type.value.lower().replace(' ', '_')}",
                venue_type=venue_type
            )
            
            print(f"✓ Reform recommendations for {venue_type.value}:")
            for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                print(f"  {i}. {rec['reform_type']} (suitability: {rec['suitability_score']:.1f})")
        
        # 10. Save and verify data persistence
        manager.save_data()
        print("✓ Saved reform data to files")
        
        # Verify data files exist
        data_files = ['reforms.json', 'evaluation_models.json', 'adoption_history.json']
        for filename in data_files:
            file_path = Path(temp_dir) / filename
            if file_path.exists():
                print(f"✓ Data file created: {filename}")
            else:
                print(f"✗ Missing data file: {filename}")
        
        # 11. Test data loading with new manager instance
        new_manager = PublicationReformManager(data_dir=temp_dir)
        
        # Verify data was loaded
        assert reform_id in new_manager.reforms
        assert model_id in new_manager.evaluation_models
        assert len(new_manager.adoption_history) > 0
        print("✓ Data successfully loaded by new manager instance")
        
        print("\n🎉 Publication Reform Manager integration test completed successfully!")
        
        # Summary statistics
        print("\nSummary Statistics:")
        print(f"- Reforms implemented: {len(manager.reforms)}")
        print(f"- Evaluation models created: {len(manager.evaluation_models)}")
        print(f"- Papers with alternative metrics: {len(manager.alternative_metrics)}")
        print(f"- Papers with post-publication reviews: {len(manager.post_pub_reviews)}")
        print(f"- Adoption history entries: {len(manager.adoption_history)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"✓ Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    success = test_publication_reform_integration()
    exit(0 if success else 1)