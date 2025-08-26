"""
Integration test for AI Impact Simulator with the peer review system.
"""

import tempfile
from pathlib import Path
from datetime import date

from src.enhancements.ai_impact_simulator import (
    AIImpactSimulator, AIAssistanceType, AIDetectionMethod, AIPolicy
)


def test_ai_impact_integration():
    """Test AI Impact Simulator integration with peer review simulation."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize simulator
        simulator = AIImpactSimulator(data_dir=Path(temp_dir) / "ai_impact")
        
        print("=== AI Impact Simulator Integration Test ===")
        
        # 1. Create researcher profiles for different career stages and fields
        print("\n1. Creating researcher profiles...")
        
        researchers = [
            ("grad_student_cs", "Graduate Student", "computer_science", "MIT"),
            ("postdoc_bio", "Postdoc", "biology", "Harvard"),
            ("asst_prof_psych", "Assistant Prof", "psychology", "Stanford"),
            ("full_prof_soc", "Full Prof", "sociology", "Berkeley")
        ]
        
        profiles = {}
        for researcher_id, career_stage, field, institution in researchers:
            profile = simulator.create_adoption_profile(
                researcher_id, career_stage, field, institution
            )
            profiles[researcher_id] = profile
            print(f"  - {researcher_id}: adoption_rate={profile.adoption_rate:.2f}, "
                  f"comfort_level={profile.comfort_level:.2f}")
        
        # 2. Implement different AI policies at institutions
        print("\n2. Implementing AI policies...")
        
        policies = [
            ("MIT", AIPolicy.UNRESTRICTED, "AI use encouraged with disclosure"),
            ("Harvard", AIPolicy.DISCLOSURE_REQUIRED, "All AI use must be disclosed"),
            ("Stanford", AIPolicy.LIMITED_USE, "Limited AI use allowed"),
            ("Berkeley", AIPolicy.PROHIBITED, "AI use prohibited")
        ]
        
        for institution, policy_type, description in policies:
            policy = simulator.implement_ai_policy(
                institution, policy_type, description, enforcement_level=0.8
            )
            print(f"  - {institution}: {policy_type.value}")
        
        # 3. Simulate AI usage across different scenarios
        print("\n3. Simulating AI usage...")
        
        usage_scenarios = [
            ("grad_student_cs", "paper_1", AIAssistanceType.WRITING_ASSISTANCE),
            ("grad_student_cs", "paper_2", AIAssistanceType.LITERATURE_REVIEW),
            ("postdoc_bio", "paper_3", AIAssistanceType.DATA_ANALYSIS),
            ("asst_prof_psych", "review_1", AIAssistanceType.REVIEW_ASSISTANCE),
            ("full_prof_soc", "paper_4", AIAssistanceType.GRAMMAR_CHECK)
        ]
        
        usage_records = []
        for researcher_id, content_id, assistance_type in usage_scenarios:
            # Temporarily increase adoption rate to ensure usage for demo
            original_rate = profiles[researcher_id].adoption_rate
            profiles[researcher_id].adoption_rate = 1.0  # Force usage for demo
            
            if content_id.startswith("paper"):
                usage = simulator.simulate_ai_usage(
                    researcher_id, paper_id=content_id, assistance_type=assistance_type
                )
            else:
                usage = simulator.simulate_ai_usage(
                    researcher_id, review_id=content_id, assistance_type=assistance_type
                )
            
            # Restore original rate
            profiles[researcher_id].adoption_rate = original_rate
            
            if usage:
                usage_records.append(usage)
                print(f"  - {researcher_id} used {usage.ai_tool_name} for {assistance_type.value}: "
                      f"{usage.content_percentage:.1%} content, disclosed={usage.is_disclosed}")
        
        # 4. Run AI detection on content
        print("\n4. Running AI detection...")
        
        detection_methods = [
            AIDetectionMethod.STATISTICAL_ANALYSIS,
            AIDetectionMethod.LINGUISTIC_PATTERNS,
            AIDetectionMethod.HUMAN_REVIEW,
            AIDetectionMethod.HYBRID_DETECTION
        ]
        
        detection_results = []
        for i, usage in enumerate(usage_records[:4]):  # Test first 4 usage records
            method = detection_methods[i % len(detection_methods)]
            content_id = usage.paper_id or usage.review_id
            
            detection = simulator.detect_ai_content(content_id, method, usage)
            detection_results.append(detection)
            
            print(f"  - {method.value} on {content_id}: "
                  f"AI probability={detection.ai_probability:.2f}, "
                  f"confidence={detection.confidence_level:.2f}")
        
        # 5. Calculate comprehensive metrics
        print("\n5. Calculating AI impact metrics...")
        
        metrics = simulator.calculate_ai_impact_metrics()
        print(f"  - Total AI usage instances: {metrics.total_ai_usage_instances}")
        print(f"  - AI adoption rate: {metrics.ai_adoption_rate:.2%}")
        print(f"  - Average content percentage: {metrics.average_content_percentage:.1%}")
        print(f"  - Disclosure rate: {metrics.disclosure_rate:.2%}")
        print(f"  - Detection accuracy: {metrics.detection_accuracy:.2%}")
        print(f"  - Quality improvement rate: {metrics.quality_improvement_rate:.2%}")
        
        # 6. Simulate policy enforcement
        print("\n6. Simulating policy enforcement...")
        
        for institution in ["MIT", "Harvard", "Stanford", "Berkeley"]:
            enforcement = simulator.simulate_policy_enforcement(institution)
            if "error" not in enforcement:
                print(f"  - {institution}: {enforcement['total_violations']} violations, "
                      f"compliance rate: {enforcement['compliance_rate']:.2%}")
        
        # 7. Calculate researcher AI scores
        print("\n7. Calculating researcher AI scores...")
        
        for researcher_id in profiles.keys():
            scores = simulator.get_researcher_ai_score(researcher_id)
            if "error" not in scores:
                print(f"  - {researcher_id}: overall score={scores['overall_ai_score']:.2f}, "
                      f"disclosure={scores['disclosure_score']:.2f}")
        
        # 8. Test data persistence
        print("\n8. Testing data persistence...")
        
        simulator.save_data()
        
        # Create new simulator and verify data loading
        new_simulator = AIImpactSimulator(data_dir=simulator.data_dir)
        
        print(f"  - Loaded {len(new_simulator.adoption_profiles)} profiles")
        print(f"  - Loaded {len(new_simulator.usage_records)} usage records")
        print(f"  - Loaded {len(new_simulator.policy_records)} policies")
        print(f"  - Loaded {len(new_simulator.detection_results)} detection results")
        
        # 9. Demonstrate tool effectiveness analysis
        print("\n9. Tool effectiveness analysis...")
        
        tool_usage = {}
        quality_by_tool = {}
        
        for usage in usage_records:
            tool = usage.ai_tool_name
            tool_usage[tool] = tool_usage.get(tool, 0) + 1
            
            if tool not in quality_by_tool:
                quality_by_tool[tool] = []
            quality_by_tool[tool].append(usage.quality_impact.value)
        
        print("  Tool usage distribution:")
        for tool, count in tool_usage.items():
            print(f"    - {tool}: {count} uses")
        
        # 10. Demonstrate field-specific adoption patterns
        print("\n10. Field-specific adoption patterns...")
        
        field_adoption = {}
        for researcher_id, profile in profiles.items():
            field = researcher_id.split('_')[-1] if '_' in researcher_id else "unknown"
            if field not in field_adoption:
                field_adoption[field] = []
            field_adoption[field].append(profile.adoption_rate)
        
        for field, rates in field_adoption.items():
            avg_rate = sum(rates) / len(rates)
            print(f"  - {field}: average adoption rate = {avg_rate:.2%}")
        
        print("\n=== Integration Test Completed Successfully ===")
        
        # Verify key functionality works
        assert len(simulator.adoption_profiles) > 0
        assert len(simulator.usage_records) > 0
        assert len(simulator.policy_records) > 0
        assert len(simulator.detection_results) > 0
        assert metrics.total_ai_usage_instances > 0
        
        return True


if __name__ == "__main__":
    success = test_ai_impact_integration()
    if success:
        print("\n✅ AI Impact Simulator integration test passed!")
    else:
        print("\n❌ AI Impact Simulator integration test failed!")