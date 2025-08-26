"""
Enhanced Researcher Agent module for the Peer Review simulation.

This module defines the EnhancedResearcherAgent class which extends the ResearcherAgent
with comprehensive capabilities for biases, networks, career progression, and funding
integration across all enhancement systems.
"""

import autogen
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
import os
import sys
from datetime import datetime, timedelta
import logging
import json
import uuid

# Add the project root to the path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.researcher_agent import ResearcherAgent
from src.data.paper_database import PaperDatabase
from src.core.token_system import TokenSystem
from src.core.exceptions import ValidationError, SimulationError
from src.core.logging_config import get_logger
from src.data.enhanced_models import (
    EnhancedResearcher, StructuredReview, EnhancedReviewCriteria,
    ResearcherLevel, VenueType, ReviewDecision, CareerStage, FundingStatus,
    ReviewBehaviorProfile, StrategicBehaviorProfile, BiasEffect,
    DetailedStrength, DetailedWeakness, ReviewQualityMetric,
    PublicationRecord, CareerMilestone, TenureTimeline
)

# Import enhancement systems with graceful fallbacks
try:
    from src.enhancements.simulation_coordinator import SimulationCoordinator
except ImportError:
    SimulationCoordinator = None

try:
    from src.enhancements.bias_engine import BiasEngine
except ImportError:
    BiasEngine = None

try:
    from src.enhancements.academic_hierarchy import AcademicHierarchy
except ImportError:
    AcademicHierarchy = None

try:
    from src.enhancements.reputation_calculator import ReputationCalculator
except ImportError:
    ReputationCalculator = None

try:
    from src.enhancements.structured_review_system import StructuredReviewSystem
except ImportError:
    StructuredReviewSystem = None

try:
    from src.enhancements.venue_system import VenueRegistry
except ImportError:
    VenueRegistry = None

try:
    from src.enhancements.workload_tracker import WorkloadTracker
except ImportError:
    WorkloadTracker = None

try:
    from src.enhancements.collaboration_network import CollaborationNetwork
except ImportError:
    CollaborationNetwork = None

try:
    from src.enhancements.citation_network import CitationNetwork
except ImportError:
    CitationNetwork = None

try:
    from src.enhancements.funding_system import FundingSystem
except ImportError:
    FundingSystem = None

try:
    from src.enhancements.tenure_track_manager import TenureTrackManager
except ImportError:
    TenureTrackManager = None

try:
    from src.enhancements.job_market_simulator import JobMarketSimulator
except ImportError:
    JobMarketSimulator = None

try:
    from src.enhancements.career_transition_manager import CareerTransitionManager
except ImportError:
    CareerTransitionManager = None

logger = get_logger(__name__)


class EnhancedResearcherAgent(ResearcherAgent):
    """
    Enhanced ResearcherAgent with comprehensive capabilities for biases, networks,
    career progression, and funding integration across all enhancement systems.
    
    This agent extends the basic ResearcherAgent with:
    - Academic hierarchy and reputation management
    - Cognitive bias modeling and application
    - Social network effects and collaboration tracking
    - Strategic behavior modeling
    - Career progression dynamics
    - Funding integration and resource constraints
    - Enhanced review generation with structured feedback
    - Temporal dynamics and workload management
    """
    
    def __init__(
        self,
        name: str,
        specialty: str,
        system_message: str,
        paper_db: PaperDatabase,
        token_system: TokenSystem,
        enhanced_profile: Optional[EnhancedResearcher] = None,
        simulation_coordinator: Optional[SimulationCoordinator] = None,
        bias: str = "",
        llm_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the EnhancedResearcherAgent.
        
        Args:
            name: Name of the researcher agent
            specialty: Research specialty area
            system_message: System message describing the agent's role
            paper_db: Paper database instance
            token_system: Token system instance
            enhanced_profile: Enhanced researcher profile with all capabilities
            simulation_coordinator: Coordinator for system integration
            bias: Research bias or preference (legacy parameter)
            llm_config: LLM configuration
            **kwargs: Additional keyword arguments for ResearcherAgent
        """
        # Initialize base ResearcherAgent
        super().__init__(
            name=name,
            specialty=specialty,
            system_message=system_message,
            paper_db=paper_db,
            token_system=token_system,
            bias=bias,
            llm_config=llm_config,
            **kwargs
        )
        
        # Enhanced profile management
        self.enhanced_profile = enhanced_profile or self._create_default_enhanced_profile()
        self.simulation_coordinator = simulation_coordinator
        
        # System integrations
        self._initialize_system_integrations()
        
        # Behavior tracking
        self.behavior_history = []
        self.decision_log = []
        self.network_interactions = []
        
        # Performance metrics
        self.performance_metrics = {
            'reviews_completed': 0,
            'papers_published': 0,
            'collaborations_formed': 0,
            'career_milestones_achieved': 0,
            'funding_secured': 0,
            'strategic_behaviors_executed': 0
        }
        
        logger.info(f"EnhancedResearcherAgent {name} initialized with level {self.enhanced_profile.level.value}")
    
    def _create_default_enhanced_profile(self) -> EnhancedResearcher:
        """Create a default enhanced researcher profile."""
        return EnhancedResearcher(
            id=self.name,
            name=self.name,
            specialty=self.specialty,
            level=ResearcherLevel.ASSISTANT_PROF,
            institution_name="Default University",
            institution_tier=2,
            h_index=5,
            total_citations=50,
            years_active=3,
            cognitive_biases={
                'confirmation': 0.3,
                'halo': 0.2,
                'anchoring': 0.4,
                'availability': 0.3
            },
            review_behavior=ReviewBehaviorProfile(),
            strategic_behavior=StrategicBehaviorProfile(),
            career_stage=CareerStage.EARLY_CAREER,
            funding_status=FundingStatus.ADEQUATELY_FUNDED,
            publication_pressure=0.5
        )
    
    def _initialize_system_integrations(self):
        """Initialize integrations with all enhancement systems."""
        # Academic hierarchy integration
        self.academic_hierarchy = AcademicHierarchy() if AcademicHierarchy else None
        
        # Reputation system integration
        self.reputation_calculator = ReputationCalculator() if ReputationCalculator else None
        
        # Review system integration
        self.structured_review_system = StructuredReviewSystem() if StructuredReviewSystem else None
        
        # Venue system integration
        self.venue_registry = VenueRegistry() if VenueRegistry else None
        
        # Workload tracking integration
        self.workload_tracker = WorkloadTracker() if WorkloadTracker else None
        
        # Network systems integration
        self.collaboration_network = CollaborationNetwork() if CollaborationNetwork else None
        self.citation_network = CitationNetwork() if CitationNetwork else None
        
        # Bias system integration
        self.bias_engine = BiasEngine() if BiasEngine else None
        
        # Career systems integration
        self.funding_system = FundingSystem() if FundingSystem else None
        self.tenure_track_manager = TenureTrackManager() if TenureTrackManager else None
        self.job_market_simulator = JobMarketSimulator() if JobMarketSimulator else None
        self.career_transition_manager = CareerTransitionManager() if CareerTransitionManager else None
    
    def get_enhanced_profile(self) -> EnhancedResearcher:
        """Get the complete enhanced researcher profile."""
        return self.enhanced_profile
    
    def update_enhanced_profile(self, updates: Dict[str, Any]) -> bool:
        """
        Update the enhanced researcher profile.
        
        Args:
            updates: Dictionary of profile updates
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            for key, value in updates.items():
                if hasattr(self.enhanced_profile, key):
                    setattr(self.enhanced_profile, key, value)
                    logger.debug(f"Updated {key} for researcher {self.name}")
            
            # Recalculate derived fields
            self.enhanced_profile._calculate_reputation_score()
            self.enhanced_profile._determine_career_stage()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update enhanced profile for {self.name}: {e}")
            return False
    
    def generate_enhanced_review(
        self,
        paper_id: str,
        venue_id: str,
        custom_prompt: Optional[str] = None,
        apply_biases: bool = True,
        apply_network_effects: bool = True
    ) -> StructuredReview:
        """
        Generate an enhanced structured review with bias and network effects.
        
        Args:
            paper_id: ID of the paper to review
            venue_id: ID of the venue
            custom_prompt: Custom prompt for review generation
            apply_biases: Whether to apply cognitive biases
            apply_network_effects: Whether to apply network effects
            
        Returns:
            Enhanced structured review
        """
        try:
            logger.info(f"Generating enhanced review for paper {paper_id} by {self.name}")
            
            # Get paper and venue information
            paper = self.paper_db.get_paper(paper_id)
            if not paper:
                raise ValidationError(f"Paper {paper_id} not found")
            
            venue = self.venue_registry.get_venue(venue_id) if self.venue_registry else None
            
            # Check workload capacity
            if self.workload_tracker:
                try:
                    is_available, status, reason = self.workload_tracker.check_availability(
                        self.enhanced_profile.id, self.enhanced_profile
                    )
                    if not is_available:
                        raise ValidationError(f"Reviewer {self.name} not available due to workload: {reason}")
                except (TypeError, ValueError):
                    # Handle mock or simplified workload tracker
                    is_available = getattr(self.workload_tracker, 'check_availability', lambda *args: True)(self.name, self.enhanced_profile)
                    if isinstance(is_available, tuple):
                        is_available = is_available[0]
                    if not is_available:
                        raise ValidationError(f"Reviewer {self.name} not available due to workload")
            
            # Generate base review using structured review system
            if self.structured_review_system:
                base_review = self.structured_review_system.generate_review(
                    paper_id, venue_id, self.name, self.enhanced_profile
                )
            else:
                # Fallback to basic review generation
                base_review = self._generate_fallback_structured_review(paper_id, venue_id)
            
            # Apply cognitive biases if enabled
            if apply_biases and self.bias_engine:
                try:
                    bias_effects = self.bias_engine.apply_biases(
                        self.enhanced_profile, base_review, paper
                    )
                    # If bias engine returns effects, apply them to the review
                    if isinstance(bias_effects, list):
                        base_review.applied_biases.extend(bias_effects)
                        biased_review = base_review
                    else:
                        biased_review = bias_effects
                except Exception as e:
                    logger.warning(f"Failed to apply biases, using base review: {e}")
                    biased_review = base_review
            else:
                biased_review = base_review
            
            # Apply network effects if enabled
            if apply_network_effects:
                final_review = self._apply_network_effects(biased_review, paper)
            else:
                final_review = biased_review
            
            # Update workload tracking
            if self.workload_tracker:
                try:
                    self.workload_tracker.assign_review(
                        self.enhanced_profile.id, paper_id, venue_id, venue.venue_type.value if venue else "Conference",
                        final_review.deadline or datetime.now() + timedelta(weeks=2)
                    )
                except (TypeError, AttributeError):
                    # Handle mock or simplified workload tracker
                    pass
            
            # Track review quality
            self._track_review_quality(final_review)
            
            # Update performance metrics
            self.performance_metrics['reviews_completed'] += 1
            
            logger.info(f"Enhanced review generated successfully for paper {paper_id}")
            return final_review
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced review for paper {paper_id}: {e}")
            raise
    
    def _generate_fallback_structured_review(self, paper_id: str, venue_id: str) -> StructuredReview:
        """Generate a fallback structured review when enhanced systems are not available."""
        paper = self.paper_db.get_paper(paper_id)
        
        # Generate basic review content using the base agent's capabilities
        basic_review = self.generate_review(paper_id)
        
        # Convert to structured review format
        structured_review = StructuredReview(
            reviewer_id=self.name,
            paper_id=paper_id,
            venue_id=venue_id,
            criteria_scores=EnhancedReviewCriteria(
                novelty=5.0 + (hash(paper_id + "novelty") % 5),
                technical_quality=5.0 + (hash(paper_id + "technical") % 5),
                clarity=5.0 + (hash(paper_id + "clarity") % 5),
                significance=5.0 + (hash(paper_id + "significance") % 5),
                reproducibility=5.0 + (hash(paper_id + "reproducibility") % 5),
                related_work=5.0 + (hash(paper_id + "related") % 5)
            ),
            confidence_level=3,
            recommendation=ReviewDecision.MAJOR_REVISION,
            executive_summary=basic_review.get('summary', 'Generated review summary'),
            detailed_strengths=[
                DetailedStrength(
                    category="Technical",
                    description=basic_review.get('strengths', 'Generated strengths'),
                    importance=3
                )
            ],
            detailed_weaknesses=[
                DetailedWeakness(
                    category="Technical",
                    description=basic_review.get('weaknesses', 'Generated weaknesses'),
                    severity=3,
                    suggestions=["Consider addressing this issue"]
                )
            ],
            technical_comments=basic_review.get('technical_correctness', 'Technical assessment'),
            presentation_comments=basic_review.get('clarity', 'Clarity assessment'),
            questions_for_authors=["Please clarify the methodology"],
            suggestions_for_improvement=["Consider improving the presentation"],
            time_spent_minutes=120,
            deadline=datetime.now() + timedelta(weeks=2)
        )
        
        return structured_review
    
    def _apply_network_effects(self, review: StructuredReview, paper: Dict[str, Any]) -> StructuredReview:
        """Apply network effects to the review."""
        # Ensure we have a StructuredReview object
        if not isinstance(review, StructuredReview):
            logger.warning(f"Expected StructuredReview, got {type(review)}")
            return review
        
        # Check for collaboration relationships
        if self.collaboration_network:
            try:
                # Check if reviewer has recent collaborations with any authors
                recent_collaborators = self.collaboration_network.get_collaborators_within_window(self.name)
                collaboration_conflicts = [author for author in paper.get('authors', []) if author in recent_collaborators]
                
                if collaboration_conflicts:
                    # Apply negative bias for collaboration conflicts
                    collaboration_effect = -0.5  # Reduce scores due to conflict
                    bias_effect = BiasEffect(
                        bias_type="collaboration",
                        strength=0.5,
                        score_adjustment=collaboration_effect,
                        description=f"Collaboration conflict with authors: {collaboration_conflicts}"
                    )
                    review.applied_biases.append(bias_effect)
            except Exception as e:
                logger.warning(f"Failed to apply collaboration network effects: {e}")
        
        # Check for citation relationships (simplified)
        if self.citation_network:
            try:
                # Simple citation bias - assume positive bias for citing similar work
                citation_effect = 0.2  # Small positive bias
                bias_effect = BiasEffect(
                    bias_type="citation",
                    strength=0.2,
                    score_adjustment=citation_effect,
                    description="Citation network positive bias"
                )
                review.applied_biases.append(bias_effect)
            except Exception as e:
                logger.warning(f"Failed to apply citation network effects: {e}")
        
        return review
    
    def _track_review_quality(self, review: StructuredReview):
        """Track review quality metrics."""
        quality_metric = ReviewQualityMetric(
            review_id=review.review_id,
            quality_score=review.quality_score,
            timeliness_score=1.0 if not review.is_late else 0.5,
            timestamp=datetime.now()
        )
        
        self.enhanced_profile.review_quality_history.append(quality_metric)
        
        # Keep only recent history (last 50 reviews)
        if len(self.enhanced_profile.review_quality_history) > 50:
            self.enhanced_profile.review_quality_history = self.enhanced_profile.review_quality_history[-50:]
    
    def execute_strategic_behavior(self, behavior_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute strategic behavior based on researcher profile and context.
        
        Args:
            behavior_type: Type of strategic behavior to execute
            context: Context information for the behavior
            
        Returns:
            Results of the strategic behavior execution
        """
        try:
            logger.info(f"Executing strategic behavior {behavior_type} for {self.name}")
            
            behavior_results = {}
            
            if behavior_type == "venue_shopping":
                behavior_results = self._execute_venue_shopping(context)
            elif behavior_type == "review_trading":
                behavior_results = self._execute_review_trading(context)
            elif behavior_type == "citation_networking":
                behavior_results = self._execute_citation_networking(context)
            elif behavior_type == "salami_slicing":
                behavior_results = self._execute_salami_slicing(context)
            elif behavior_type == "collaboration_seeking":
                behavior_results = self._execute_collaboration_seeking(context)
            else:
                raise ValidationError(f"Unknown strategic behavior type: {behavior_type}")
            
            # Track behavior execution
            self.behavior_history.append({
                'behavior_type': behavior_type,
                'context': context,
                'results': behavior_results,
                'timestamp': datetime.now()
            })
            
            # Update performance metrics
            self.performance_metrics['strategic_behaviors_executed'] += 1
            
            logger.info(f"Strategic behavior {behavior_type} executed successfully")
            return behavior_results
            
        except Exception as e:
            logger.error(f"Failed to execute strategic behavior {behavior_type}: {e}")
            raise
    
    def _execute_venue_shopping(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute venue shopping behavior."""
        paper_id = context.get('paper_id')
        rejection_history = context.get('rejection_history', [])
        
        # Determine venue shopping tendency
        shopping_tendency = self.enhanced_profile.strategic_behavior.venue_shopping_tendency
        
        if shopping_tendency > 0.5 and len(rejection_history) > 0:
            # Look for lower-tier venues
            target_venues = self._find_lower_tier_venues(rejection_history[-1])
            
            return {
                'action': 'venue_shopping',
                'target_venues': target_venues,
                'shopping_tendency': shopping_tendency,
                'rationale': 'Seeking lower-tier venue after rejection'
            }
        
        return {'action': 'no_venue_shopping', 'reason': 'Low shopping tendency or no rejections'}
    
    def _execute_review_trading(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute review trading behavior."""
        potential_partners = context.get('potential_partners', [])
        
        trading_willingness = self.enhanced_profile.strategic_behavior.review_trading_willingness
        
        if trading_willingness > 0.3 and potential_partners:
            # Select trading partners based on mutual benefit
            selected_partners = [p for p in potential_partners if self._evaluate_trading_benefit(p) > 0.5]
            
            return {
                'action': 'review_trading',
                'selected_partners': selected_partners,
                'trading_willingness': trading_willingness,
                'rationale': 'Mutual benefit in review trading'
            }
        
        return {'action': 'no_review_trading', 'reason': 'Low willingness or no suitable partners'}
    
    def _execute_citation_networking(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute citation networking behavior."""
        potential_citations = context.get('potential_citations', [])
        
        # Build citation network strategically
        strategic_citations = []
        for citation in potential_citations:
            if self._evaluate_citation_benefit(citation) > 0.6:
                strategic_citations.append(citation)
        
        return {
            'action': 'citation_networking',
            'strategic_citations': strategic_citations,
            'rationale': 'Building strategic citation network'
        }
    
    def _execute_salami_slicing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute salami slicing behavior."""
        research_work = context.get('research_work')
        publication_pressure = self.enhanced_profile.publication_pressure
        
        slicing_tendency = self.enhanced_profile.strategic_behavior.salami_slicing_tendency
        
        if slicing_tendency > 0.4 and publication_pressure > 0.6:
            # Split work into multiple papers
            paper_splits = self._plan_paper_splits(research_work)
            
            return {
                'action': 'salami_slicing',
                'paper_splits': paper_splits,
                'slicing_tendency': slicing_tendency,
                'rationale': 'High publication pressure drives work splitting'
            }
        
        return {'action': 'no_salami_slicing', 'reason': 'Low tendency or pressure'}
    
    def _execute_collaboration_seeking(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute collaboration seeking behavior."""
        potential_collaborators = context.get('potential_collaborators', [])
        
        # Evaluate collaboration opportunities
        valuable_collaborations = []
        for collaborator in potential_collaborators:
            if self._evaluate_collaboration_value(collaborator) > 0.7:
                valuable_collaborations.append(collaborator)
        
        return {
            'action': 'collaboration_seeking',
            'target_collaborators': valuable_collaborations,
            'rationale': 'Seeking high-value collaborations'
        }
    
    def manage_career_progression(self) -> Dict[str, Any]:
        """
        Manage career progression across all career systems.
        
        Returns:
            Dictionary containing career progression results
        """
        try:
            logger.info(f"Managing career progression for {self.name}")
            
            progression_results = {}
            
            # Tenure track management
            if (self.enhanced_profile.level == ResearcherLevel.ASSISTANT_PROF and 
                self.tenure_track_manager):
                tenure_status = self.tenure_track_manager.evaluate_tenure_progress(self.name)
                progression_results['tenure_status'] = tenure_status
            
            # Job market evaluation
            if self.job_market_simulator:
                market_position = self.job_market_simulator.evaluate_market_position(self.name)
                progression_results['job_market_position'] = market_position
            
            # Career transition opportunities
            if self.career_transition_manager:
                transition_opportunities = self.career_transition_manager.evaluate_transition_opportunities(self.name)
                progression_results['transition_opportunities'] = transition_opportunities
            
            # Update career milestones
            self._update_career_milestones(progression_results)
            
            logger.info(f"Career progression managed for {self.name}")
            return progression_results
            
        except Exception as e:
            logger.error(f"Failed to manage career progression for {self.name}: {e}")
            raise
    
    def manage_funding_lifecycle(self) -> Dict[str, Any]:
        """
        Manage funding lifecycle and resource constraints.
        
        Returns:
            Dictionary containing funding management results
        """
        try:
            logger.info(f"Managing funding lifecycle for {self.name}")
            
            funding_results = {}
            
            if self.funding_system:
                # Evaluate current funding status
                current_status = self.funding_system.evaluate_funding_status(self.name)
                funding_results['current_status'] = current_status
                
                # Calculate publication pressure
                pub_pressure = self.funding_system.calculate_publication_pressure(
                    self.name, self.enhanced_profile.career_stage
                )
                funding_results['publication_pressure'] = pub_pressure
                
                # Evaluate resource constraints
                resource_status = self.funding_system.evaluate_resource_constraints(self.name)
                funding_results['resource_constraints'] = resource_status
                
                # Update enhanced profile
                self.enhanced_profile.funding_status = current_status.get('status', self.enhanced_profile.funding_status)
                self.enhanced_profile.publication_pressure = pub_pressure
            
            # Update performance metrics
            if funding_results.get('current_status', {}).get('newly_funded', False):
                self.performance_metrics['funding_secured'] += 1
            
            logger.info(f"Funding lifecycle managed for {self.name}")
            return funding_results
            
        except Exception as e:
            logger.error(f"Failed to manage funding lifecycle for {self.name}: {e}")
            raise
    
    def update_network_relationships(self, interaction_type: str, target_researchers: List[str]) -> Dict[str, Any]:
        """
        Update network relationships based on interactions.
        
        Args:
            interaction_type: Type of interaction (collaboration, citation, etc.)
            target_researchers: List of researcher IDs involved in the interaction
            
        Returns:
            Dictionary containing network update results
        """
        try:
            logger.info(f"Updating network relationships for {self.name}: {interaction_type}")
            
            network_results = {}
            
            # Update collaboration network
            if interaction_type == "collaboration" and self.collaboration_network:
                for researcher_id in target_researchers:
                    self.collaboration_network.add_collaboration(self.name, researcher_id)
                    self.enhanced_profile.add_collaboration(researcher_id)
                
                network_results['collaborations_added'] = len(target_researchers)
            
            # Update citation network
            elif interaction_type == "citation" and self.citation_network:
                for researcher_id in target_researchers:
                    self.citation_network.add_citation_relationship(self.name, researcher_id)
                    self.enhanced_profile.add_citation_relationship(researcher_id)
                
                network_results['citations_added'] = len(target_researchers)
            
            # Track network interactions
            self.network_interactions.append({
                'interaction_type': interaction_type,
                'target_researchers': target_researchers,
                'timestamp': datetime.now()
            })
            
            # Update performance metrics
            if interaction_type == "collaboration":
                self.performance_metrics['collaborations_formed'] += len(target_researchers)
            
            logger.info(f"Network relationships updated for {self.name}")
            return network_results
            
        except Exception as e:
            logger.error(f"Failed to update network relationships for {self.name}: {e}")
            raise
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status across all systems.
        
        Returns:
            Dictionary containing complete agent status
        """
        return {
            'basic_info': {
                'name': self.name,
                'specialty': self.specialty,
                'level': self.enhanced_profile.level.value,
                'career_stage': self.enhanced_profile.career_stage.value,
                'reputation_score': self.enhanced_profile.reputation_score
            },
            'enhanced_profile': self.enhanced_profile.to_dict(),
            'performance_metrics': self.performance_metrics,
            'current_workload': {
                'review_load': self.enhanced_profile.current_review_load,
                'max_reviews': self.enhanced_profile.max_reviews_per_month,
                'availability': self.enhanced_profile.availability_status
            },
            'network_status': {
                'collaborations': len(self.enhanced_profile.collaboration_network),
                'citations': len(self.enhanced_profile.citation_network),
                'recent_interactions': len(self.network_interactions)
            },
            'behavior_tracking': {
                'behaviors_executed': len(self.behavior_history),
                'decisions_made': len(self.decision_log),
                'strategic_tendency': {
                    'venue_shopping': self.enhanced_profile.strategic_behavior.venue_shopping_tendency,
                    'review_trading': self.enhanced_profile.strategic_behavior.review_trading_willingness,
                    'citation_cartel': self.enhanced_profile.strategic_behavior.citation_cartel_participation,
                    'salami_slicing': self.enhanced_profile.strategic_behavior.salami_slicing_tendency
                }
            },
            'system_integrations': {
                'simulation_coordinator': self.simulation_coordinator is not None,
                'bias_engine': self.bias_engine is not None,
                'structured_review_system': self.structured_review_system is not None,
                'venue_registry': self.venue_registry is not None,
                'workload_tracker': self.workload_tracker is not None,
                'collaboration_network': self.collaboration_network is not None,
                'citation_network': self.citation_network is not None,
                'funding_system': self.funding_system is not None,
                'career_systems': {
                    'tenure_track_manager': self.tenure_track_manager is not None,
                    'job_market_simulator': self.job_market_simulator is not None,
                    'career_transition_manager': self.career_transition_manager is not None
                }
            }
        }
    
    # Helper methods for strategic behaviors
    
    def _find_lower_tier_venues(self, rejected_venue_id: str) -> List[str]:
        """Find lower-tier venues for venue shopping."""
        if not self.venue_registry:
            return []
        
        rejected_venue = self.venue_registry.get_venue(rejected_venue_id)
        if not rejected_venue:
            return []
        
        # Find venues with higher acceptance rates in the same field
        lower_tier_venues = []
        for venue in self.venue_registry.get_venues_by_field(rejected_venue.field):
            if venue.acceptance_rate > rejected_venue.acceptance_rate:
                lower_tier_venues.append(venue.id)
        
        return lower_tier_venues[:3]  # Return top 3 options
    
    def _evaluate_trading_benefit(self, partner_id: str) -> float:
        """Evaluate the benefit of review trading with a partner."""
        # Simple heuristic based on reputation and network distance
        if not self.reputation_calculator:
            return 0.5
        
        partner_reputation = self.reputation_calculator.get_reputation_score(partner_id)
        my_reputation = self.enhanced_profile.reputation_score
        
        # Prefer partners with similar or higher reputation
        reputation_factor = min(1.0, partner_reputation / max(0.1, my_reputation))
        
        # Check network distance (simplified - check if they are collaborators)
        network_distance = 1.0  # Default if no network system
        if self.collaboration_network:
            collaborators = self.collaboration_network.get_collaborators_within_window(self.name)
            network_distance = 0.5 if partner_id in collaborators else 2.0  # Closer if collaborators
        
        # Closer network distance is better for trading
        distance_factor = 1.0 / max(1.0, network_distance)
        
        return (reputation_factor * 0.6 + distance_factor * 0.4)
    
    def _evaluate_citation_benefit(self, citation_target: str) -> float:
        """Evaluate the benefit of citing a particular work."""
        # Simple heuristic based on citation impact and relevance
        return 0.7  # Placeholder implementation
    
    def _plan_paper_splits(self, research_work: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan how to split research work into multiple papers."""
        # Simple implementation - split into 2-3 papers
        work_size = research_work.get('complexity', 1.0)
        
        if work_size > 0.8:
            return [
                {'title': f"{research_work.get('title', 'Work')} - Part 1", 'focus': 'methodology'},
                {'title': f"{research_work.get('title', 'Work')} - Part 2", 'focus': 'results'},
                {'title': f"{research_work.get('title', 'Work')} - Part 3", 'focus': 'applications'}
            ]
        elif work_size > 0.5:
            return [
                {'title': f"{research_work.get('title', 'Work')} - Theory", 'focus': 'theoretical'},
                {'title': f"{research_work.get('title', 'Work')} - Empirical", 'focus': 'empirical'}
            ]
        else:
            return [research_work]  # Don't split small work
    
    def _evaluate_collaboration_value(self, collaborator_id: str) -> float:
        """Evaluate the value of collaborating with a researcher."""
        if not self.reputation_calculator:
            return 0.6
        
        collaborator_reputation = self.reputation_calculator.get_reputation_score(collaborator_id)
        
        # Higher reputation collaborators are more valuable
        reputation_factor = min(1.0, collaborator_reputation)
        
        # Check for complementary expertise
        complementary_factor = 0.8  # Placeholder
        
        # Check for institutional diversity
        institutional_factor = 0.7  # Placeholder
        
        return (reputation_factor * 0.5 + complementary_factor * 0.3 + institutional_factor * 0.2)
    
    def _update_career_milestones(self, progression_results: Dict[str, Any]):
        """Update career milestones based on progression results."""
        # Check for new milestones
        if progression_results.get('tenure_status', {}).get('achieved', False):
            milestone = CareerMilestone(
                milestone_type="tenure",
                date_achieved=datetime.now().date(),
                description="Achieved tenure",
                impact_on_behavior={'publication_pressure': -0.2}
            )
            self.enhanced_profile.career_milestones.append(milestone)
            self.performance_metrics['career_milestones_achieved'] += 1
        
        # Add other milestone checks as needed