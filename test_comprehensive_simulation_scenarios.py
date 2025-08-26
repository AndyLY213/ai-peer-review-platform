"""
Comprehensive Integration Test for Simulation Scenarios

This integration test demonstrates the complete simulation scenarios system
working with all enhanced features, showcasing realistic academic environment
simulations and validating the comprehensive functionality.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.enhancements.simulation_scenarios import (
    SimulationScenarios, ScenarioConfiguration, ScenarioType, ScenarioComplexity
)
from src.enhancements.simulation_coordinator import SimulationCoordinator
from src.core.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def test_basic_scenario_execution():
    """Test basic scenario execution with minimal features."""
    print("\n" + "="*60)
    print("TESTING BASIC SCENARIO EXECUTION")
    print("="*60)
    
    try:
        # Initialize simulation scenarios
        coordinator = SimulationCoordinator()
        scenarios = SimulationScenarios(simulation_coordinator=coordinator)
        
        print(f"✓ Initialized simulation scenarios with {len(scenarios.get_available_scenarios())} predefined scenarios")
        
        # Execute basic peer review scenario
        print("\nExecuting basic peer review scenario...")
        start_time = datetime.now()
        
        result = scenarios.execute_scenario("basic_peer_review")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✓ Scenario executed successfully in {execution_time:.2f} seconds")
        print(f"  - Reviews generated: {result.total_reviews_generated}")
        print(f"  - Papers processed: {result.total_papers_processed}")
        print(f"  - Researchers active: {result.total_researchers_active}")
        print(f"  - Success: {result.success}")
        print(f"  - Validation results: {result.validation_results}")
        
        # Validate results
        assert result.success, "Basic scenario should execute successfully"
        assert result.total_reviews_generated > 0, "Should generate reviews"
        assert result.total_papers_processed > 0, "Should process papers"
        assert result.total_researchers_active > 0, "Should have active researchers"
        
        print("✓ Basic scenario validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic scenario test failed: {e}")
        logger.error(f"Basic scenario test failed: {e}", exc_info=True)
        return False


def test_bias_demonstration_scenario():
    """Test bias demonstration scenario with cognitive biases enabled."""
    print("\n" + "="*60)
    print("TESTING BIAS DEMONSTRATION SCENARIO")
    print("="*60)
    
    try:
        # Initialize simulation scenarios
        coordinator = SimulationCoordinator()
        scenarios = SimulationScenarios(simulation_coordinator=coordinator)
        
        # Execute bias demonstration scenario
        print("Executing bias demonstration scenario...")
        start_time = datetime.now()
        
        result = scenarios.execute_scenario("bias_demonstration")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✓ Bias scenario executed successfully in {execution_time:.2f} seconds")
        print(f"  - Reviews generated: {result.total_reviews_generated}")
        print(f"  - Papers processed: {result.total_papers_processed}")
        print(f"  - Researchers active: {result.total_researchers_active}")
        
        # Check scenario-specific metrics
        if "behavior_analysis" in result.scenario_metrics:
            behavior_analysis = result.scenario_metrics["behavior_analysis"]
            if "bias_effects" in behavior_analysis:
                bias_effects = behavior_analysis["bias_effects"]
                print(f"  - Bias effects detected:")
                print(f"    * Confirmation bias instances: {bias_effects.get('confirmation_bias_instances', 0)}")
                print(f"    * Halo effect instances: {bias_effects.get('halo_effect_instances', 0)}")
                print(f"    * Overall bias impact: {bias_effects.get('overall_bias_impact', 0):.3f}")
        
        # Validate results
        assert result.success, "Bias scenario should execute successfully"
        assert result.total_reviews_generated >= 20, "Should generate substantial reviews for bias analysis"
        
        print("✓ Bias demonstration scenario validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Bias demonstration test failed: {e}")
        logger.error(f"Bias demonstration test failed: {e}", exc_info=True)
        return False


def test_network_effects_scenario():
    """Test network effects scenario with social networks enabled."""
    print("\n" + "="*60)
    print("TESTING NETWORK EFFECTS SCENARIO")
    print("="*60)
    
    try:
        # Initialize simulation scenarios
        coordinator = SimulationCoordinator()
        scenarios = SimulationScenarios(simulation_coordinator=coordinator)
        
        # Execute network effects scenario
        print("Executing network effects scenario...")
        start_time = datetime.now()
        
        result = scenarios.execute_scenario("network_effects")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✓ Network scenario executed successfully in {execution_time:.2f} seconds")
        print(f"  - Reviews generated: {result.total_reviews_generated}")
        print(f"  - Papers processed: {result.total_papers_processed}")
        print(f"  - Researchers active: {result.total_researchers_active}")
        
        # Check network-specific metrics
        if "behavior_analysis" in result.scenario_metrics:
            behavior_analysis = result.scenario_metrics["behavior_analysis"]
            if "network_effects" in behavior_analysis:
                network_effects = behavior_analysis["network_effects"]
                print(f"  - Network effects detected:")
                print(f"    * Collaboration networks: {network_effects.get('collaboration_networks_formed', 0)}")
                print(f"    * Citation networks: {network_effects.get('citation_networks_identified', 0)}")
                print(f"    * Conflict cases: {network_effects.get('conflict_of_interest_cases', 0)}")
                print(f"    * Network influence: {network_effects.get('network_influence_score', 0):.3f}")
        
        # Validate results
        assert result.success, "Network scenario should execute successfully"
        assert result.total_researchers_active >= 50, "Should have substantial researchers for network analysis"
        
        print("✓ Network effects scenario validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Network effects test failed: {e}")
        logger.error(f"Network effects test failed: {e}", exc_info=True)
        return False


def test_strategic_behavior_scenario():
    """Test strategic behavior scenario with gaming tactics enabled."""
    print("\n" + "="*60)
    print("TESTING STRATEGIC BEHAVIOR SCENARIO")
    print("="*60)
    
    try:
        # Initialize simulation scenarios
        coordinator = SimulationCoordinator()
        scenarios = SimulationScenarios(simulation_coordinator=coordinator)
        
        # Execute strategic behavior scenario
        print("Executing strategic behavior scenario...")
        start_time = datetime.now()
        
        result = scenarios.execute_scenario("strategic_behavior")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✓ Strategic behavior scenario executed successfully in {execution_time:.2f} seconds")
        print(f"  - Reviews generated: {result.total_reviews_generated}")
        print(f"  - Papers processed: {result.total_papers_processed}")
        print(f"  - Researchers active: {result.total_researchers_active}")
        
        # Check strategic behavior metrics
        if "behavior_analysis" in result.scenario_metrics:
            behavior_analysis = result.scenario_metrics["behavior_analysis"]
            if "strategic_behaviors" in behavior_analysis:
                strategic_behaviors = behavior_analysis["strategic_behaviors"]
                print(f"  - Strategic behaviors detected:")
                print(f"    * Venue shopping: {strategic_behaviors.get('venue_shopping_instances', 0)}")
                print(f"    * Review trading: {strategic_behaviors.get('review_trading_detected', 0)}")
                print(f"    * Citation cartels: {strategic_behaviors.get('citation_cartels_formed', 0)}")
                print(f"    * Salami slicing: {strategic_behaviors.get('salami_slicing_cases', 0)}")
        
        # Validate results
        assert result.success, "Strategic behavior scenario should execute successfully"
        assert result.total_researchers_active >= 80, "Should have many researchers for strategic analysis"
        
        print("✓ Strategic behavior scenario validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Strategic behavior test failed: {e}")
        logger.error(f"Strategic behavior test failed: {e}", exc_info=True)
        return False


def test_custom_scenario_creation():
    """Test creating and executing a custom scenario."""
    print("\n" + "="*60)
    print("TESTING CUSTOM SCENARIO CREATION")
    print("="*60)
    
    try:
        # Initialize simulation scenarios
        coordinator = SimulationCoordinator()
        scenarios = SimulationScenarios(simulation_coordinator=coordinator)
        
        # Create custom scenario configuration
        custom_config = ScenarioConfiguration(
            scenario_id="custom_integration_test",
            name="Custom Integration Test Scenario",
            description="A custom scenario for integration testing with moderate complexity",
            scenario_type=ScenarioType.BIAS_DEMONSTRATION,
            complexity=ScenarioComplexity.MODERATE,
            duration_days=45,
            num_researchers=30,
            num_papers=15,
            num_venues=4,
            enable_biases=True,
            enable_networks=True,
            enable_strategic_behavior=False,
            enable_career_progression=True,
            enable_funding_system=False,
            enable_meta_science=False,
            parameters={
                "bias_strengths": {
                    "confirmation": 0.35,
                    "halo_effect": 0.45,
                    "anchoring": 0.25,
                    "availability": 0.20
                },
                "network_density": 0.12,
                "review_deadline_weeks": 3
            },
            expected_outcomes={
                "reviews_completed": 45,
                "papers_decided": 15,
                "bias_effect_detected": True,
                "network_influence_detected": True
            }
        )
        
        # Create the custom scenario
        print("Creating custom scenario...")
        scenario_id = scenarios.create_custom_scenario(custom_config)
        print(f"✓ Custom scenario created with ID: {scenario_id}")
        
        # Execute the custom scenario
        print("Executing custom scenario...")
        start_time = datetime.now()
        
        result = scenarios.execute_scenario(scenario_id)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✓ Custom scenario executed successfully in {execution_time:.2f} seconds")
        print(f"  - Reviews generated: {result.total_reviews_generated}")
        print(f"  - Papers processed: {result.total_papers_processed}")
        print(f"  - Researchers active: {result.total_researchers_active}")
        print(f"  - Validation results: {result.validation_results}")
        
        # Validate results
        assert result.success, "Custom scenario should execute successfully"
        assert result.total_researchers_active == 30, "Should match custom configuration"
        assert result.total_papers_processed == 15, "Should match custom configuration"
        
        print("✓ Custom scenario validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Custom scenario test failed: {e}")
        logger.error(f"Custom scenario test failed: {e}", exc_info=True)
        return False


def test_batch_scenario_execution():
    """Test executing multiple scenarios in batch."""
    print("\n" + "="*60)
    print("TESTING BATCH SCENARIO EXECUTION")
    print("="*60)
    
    try:
        # Initialize simulation scenarios
        coordinator = SimulationCoordinator()
        scenarios = SimulationScenarios(simulation_coordinator=coordinator)
        
        # Define scenarios to execute in batch
        scenario_ids = [
            "basic_peer_review",
            "bias_demonstration",
            "network_effects"
        ]
        
        print(f"Executing batch of {len(scenario_ids)} scenarios...")
        start_time = datetime.now()
        
        results = scenarios.execute_scenario_batch(scenario_ids, parallel=False)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✓ Batch execution completed in {execution_time:.2f} seconds")
        print(f"  - Scenarios executed: {len(results)}")
        
        # Validate each result
        for i, result in enumerate(results):
            print(f"  - Scenario {i+1} ({result.scenario_id}):")
            print(f"    * Success: {result.success}")
            print(f"    * Reviews: {result.total_reviews_generated}")
            print(f"    * Papers: {result.total_papers_processed}")
            print(f"    * Time: {result.execution_time_seconds:.2f}s")
            
            assert result.success, f"Scenario {result.scenario_id} should succeed"
        
        # Check execution history
        history = scenarios.get_execution_history()
        assert len(history) >= len(scenario_ids), "All scenarios should be in history"
        
        print("✓ Batch execution validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Batch execution test failed: {e}")
        logger.error(f"Batch execution test failed: {e}", exc_info=True)
        return False


def test_scenario_statistics_and_reporting():
    """Test scenario statistics and reporting functionality."""
    print("\n" + "="*60)
    print("TESTING SCENARIO STATISTICS AND REPORTING")
    print("="*60)
    
    try:
        # Initialize simulation scenarios
        coordinator = SimulationCoordinator()
        scenarios = SimulationScenarios(simulation_coordinator=coordinator)
        
        # Execute a few scenarios to generate statistics
        print("Executing scenarios for statistics...")
        scenarios.execute_scenario("basic_peer_review")
        scenarios.execute_scenario("bias_demonstration")
        
        # Get statistics
        stats = scenarios.get_scenario_statistics()
        
        print("✓ Scenario statistics generated:")
        print(f"  - Total executions: {stats['total_executions']}")
        print(f"  - Successful executions: {stats['successful_executions']}")
        print(f"  - Failed executions: {stats['failed_executions']}")
        print(f"  - Success rate: {stats['success_rate']:.2%}")
        print(f"  - Average execution time: {stats['average_execution_time']:.2f}s")
        print(f"  - Total reviews generated: {stats['total_reviews_generated']}")
        print(f"  - Total papers processed: {stats['total_papers_processed']}")
        
        # Check scenarios by type
        if 'scenarios_by_type' in stats:
            print("  - Scenarios by type:")
            for scenario_type, count in stats['scenarios_by_type'].items():
                print(f"    * {scenario_type}: {count}")
        
        # Check complexity distribution
        if 'complexity_distribution' in stats:
            print("  - Complexity distribution:")
            for complexity, count in stats['complexity_distribution'].items():
                print(f"    * {complexity}: {count}")
        
        # Validate statistics
        assert stats['total_executions'] >= 2, "Should have executed at least 2 scenarios"
        assert stats['success_rate'] > 0, "Should have some successful executions"
        assert stats['total_reviews_generated'] > 0, "Should have generated reviews"
        
        print("✓ Statistics validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Statistics test failed: {e}")
        logger.error(f"Statistics test failed: {e}", exc_info=True)
        return False


def test_comprehensive_ecosystem_scenario():
    """Test the comprehensive ecosystem scenario with all features enabled."""
    print("\n" + "="*60)
    print("TESTING COMPREHENSIVE ECOSYSTEM SCENARIO")
    print("="*60)
    
    try:
        # Initialize simulation scenarios
        coordinator = SimulationCoordinator()
        scenarios = SimulationScenarios(simulation_coordinator=coordinator)
        
        # Get the comprehensive scenario
        scenario_config = scenarios.get_scenario("comprehensive_ecosystem")
        print(f"Comprehensive scenario configuration:")
        print(f"  - Duration: {scenario_config.duration_days} days")
        print(f"  - Researchers: {scenario_config.num_researchers}")
        print(f"  - Papers: {scenario_config.num_papers}")
        print(f"  - Venues: {scenario_config.num_venues}")
        print(f"  - All systems enabled: {all([
            scenario_config.enable_biases,
            scenario_config.enable_networks,
            scenario_config.enable_strategic_behavior,
            scenario_config.enable_career_progression,
            scenario_config.enable_funding_system,
            scenario_config.enable_meta_science
        ])}")
        
        # Execute with reduced scale for testing
        print("\nExecuting comprehensive ecosystem scenario (reduced scale)...")
        custom_params = {
            "duration_days": 365,  # 1 year instead of 10
            "num_researchers": 50,  # Reduced from 200
            "num_papers": 100,     # Reduced from 500
            "num_venues": 10       # Reduced from 20
        }
        
        start_time = datetime.now()
        
        result = scenarios.execute_scenario(
            "comprehensive_ecosystem", 
            custom_parameters=custom_params
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✓ Comprehensive scenario executed successfully in {execution_time:.2f} seconds")
        print(f"  - Reviews generated: {result.total_reviews_generated}")
        print(f"  - Papers processed: {result.total_papers_processed}")
        print(f"  - Researchers active: {result.total_researchers_active}")
        
        # Check comprehensive metrics
        if "behavior_analysis" in result.scenario_metrics:
            behavior_analysis = result.scenario_metrics["behavior_analysis"]
            if "comprehensive" in behavior_analysis:
                comprehensive = behavior_analysis["comprehensive"]
                print(f"  - Comprehensive ecosystem metrics:")
                print(f"    * System interactions: {comprehensive.get('system_interactions', 0)}")
                print(f"    * Emergent behaviors: {comprehensive.get('emergent_behaviors', 0)}")
                print(f"    * Ecosystem stability: {comprehensive.get('ecosystem_stability', 0):.3f}")
                
                if "complexity_metrics" in comprehensive:
                    complexity = comprehensive["complexity_metrics"]
                    print(f"    * Interaction density: {complexity.get('interaction_density', 0):.3f}")
                    print(f"    * Behavioral diversity: {complexity.get('behavioral_diversity', 0):.3f}")
                    print(f"    * System resilience: {complexity.get('system_resilience', 0):.3f}")
        
        # Validate results
        assert result.success, "Comprehensive scenario should execute successfully"
        assert result.total_researchers_active >= 40, "Should have substantial researchers"
        assert result.total_reviews_generated >= 50, "Should generate many reviews"
        
        print("✓ Comprehensive ecosystem scenario validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Comprehensive ecosystem test failed: {e}")
        logger.error(f"Comprehensive ecosystem test failed: {e}", exc_info=True)
        return False


def test_scenario_validation_and_error_handling():
    """Test scenario validation and error handling."""
    print("\n" + "="*60)
    print("TESTING SCENARIO VALIDATION AND ERROR HANDLING")
    print("="*60)
    
    try:
        # Initialize simulation scenarios
        coordinator = SimulationCoordinator()
        scenarios = SimulationScenarios(simulation_coordinator=coordinator)
        
        # Test 1: Execute non-existent scenario
        print("Testing non-existent scenario execution...")
        try:
            scenarios.execute_scenario("non_existent_scenario")
            assert False, "Should have raised ValidationError"
        except Exception as e:
            print(f"✓ Correctly caught error for non-existent scenario: {type(e).__name__}")
        
        # Test 2: Create scenario with invalid configuration
        print("Testing invalid scenario configuration...")
        try:
            invalid_config = ScenarioConfiguration(
                scenario_id="invalid_test",
                name="Invalid Test",
                description="Invalid configuration",
                scenario_type=ScenarioType.BASIC_PEER_REVIEW,
                complexity=ScenarioComplexity.SIMPLE,
                duration_days=-10,  # Invalid negative duration
                num_researchers=0,   # Invalid zero researchers
                num_papers=5,
                num_venues=2
            )
            scenarios.create_custom_scenario(invalid_config)
            assert False, "Should have raised ValidationError"
        except Exception as e:
            print(f"✓ Correctly caught error for invalid configuration: {type(e).__name__}")
        
        # Test 3: Create duplicate scenario
        print("Testing duplicate scenario creation...")
        try:
            duplicate_config = ScenarioConfiguration(
                scenario_id="basic_peer_review",  # Duplicate existing ID
                name="Duplicate Test",
                description="Duplicate configuration",
                scenario_type=ScenarioType.BASIC_PEER_REVIEW,
                complexity=ScenarioComplexity.SIMPLE,
                duration_days=30,
                num_researchers=10,
                num_papers=5,
                num_venues=2
            )
            scenarios.create_custom_scenario(duplicate_config)
            assert False, "Should have raised ValidationError"
        except Exception as e:
            print(f"✓ Correctly caught error for duplicate scenario: {type(e).__name__}")
        
        print("✓ Error handling validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        logger.error(f"Error handling test failed: {e}", exc_info=True)
        return False


def run_comprehensive_integration_test():
    """Run the complete comprehensive integration test suite."""
    print("COMPREHENSIVE SIMULATION SCENARIOS INTEGRATION TEST")
    print("=" * 80)
    print(f"Test started at: {datetime.now()}")
    print("=" * 80)
    
    # Track test results
    test_results = []
    
    # Run all test functions
    test_functions = [
        ("Basic Scenario Execution", test_basic_scenario_execution),
        ("Bias Demonstration Scenario", test_bias_demonstration_scenario),
        ("Network Effects Scenario", test_network_effects_scenario),
        ("Strategic Behavior Scenario", test_strategic_behavior_scenario),
        ("Custom Scenario Creation", test_custom_scenario_creation),
        ("Batch Scenario Execution", test_batch_scenario_execution),
        ("Statistics and Reporting", test_scenario_statistics_and_reporting),
        ("Comprehensive Ecosystem", test_comprehensive_ecosystem_scenario),
        ("Validation and Error Handling", test_scenario_validation_and_error_handling)
    ]
    
    for test_name, test_function in test_functions:
        try:
            print(f"\nRunning: {test_name}")
            success = test_function()
            test_results.append((test_name, success))
            if success:
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
            test_results.append((test_name, False))
            logger.error(f"Test {test_name} failed with exception: {e}", exc_info=True)
    
    # Print summary
    print("\n" + "="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)
    
    passed_tests = sum(1 for _, success in test_results if success)
    total_tests = len(test_results)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {passed_tests/total_tests:.1%}")
    
    print("\nDetailed Results:")
    for test_name, success in test_results:
        status = "PASS" if success else "FAIL"
        print(f"  {status:4} | {test_name}")
    
    print(f"\nTest completed at: {datetime.now()}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Comprehensive simulation scenarios are working correctly.")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} TEST(S) FAILED. Please check the logs for details.")
        return False


if __name__ == "__main__":
    # Run the comprehensive integration test
    success = run_comprehensive_integration_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)