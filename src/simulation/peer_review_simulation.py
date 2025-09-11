"""
Peer Review Simulation System.

This module brings together all components to simulate
a token-based peer review system with multi-agent researchers.
"""

import os
import sys
import autogen
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add the project root to the path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Update imports to use proper paths
from src.data.paper_database import PaperDatabase
from src.core.token_system import TokenSystem
from src.core.llm_config import get_llm_config, get_llm_provider_info
from src.core.logging_config import get_logger, get_simulation_logger, log_llm_config, log_simulation_start, log_simulation_end, log_researcher_action
from src.agents.researcher_agent import ResearcherAgent
from src.agents.researcher_templates import get_researcher_template, list_researcher_templates

# Enhanced mode imports - using try/except for graceful fallback
try:
    from src.enhancements.simulation_coordinator import SimulationCoordinator
    from src.agents.enhanced_researcher_agent import EnhancedResearcherAgent
    from src.enhancements.bias_engine import BiasEngine
    from src.enhancements.venue_system import VenueRegistry
    from src.enhancements.academic_hierarchy import AcademicHierarchy
    from src.enhancements.reputation_calculator import ReputationCalculator
    from src.enhancements.structured_review_system import StructuredReviewValidator
    from src.enhancements.collaboration_network import CollaborationNetwork
    from src.enhancements.citation_network import CitationNetwork
    from src.enhancements.funding_system import FundingSystem
    from src.enhancements.tenure_track_manager import TenureTrackManager
    from src.enhancements.workload_tracker import WorkloadTracker
    from src.enhancements.deadline_manager import DeadlineManager
    # Individual bias models
    from src.enhancements.anchoring_bias_model import AnchoringBiasModel
    from src.enhancements.confirmation_bias_model import ConfirmationBiasModel
    from src.enhancements.halo_effect_model import HaloEffectModel
    from src.enhancements.availability_bias_model import AvailabilityBiasModel
    ENHANCEMENTS_AVAILABLE = True
    print("Enhancement systems are available")
except ImportError as e:
    # Use print instead of logger since logger might not be available yet
    print(f"Warning: Enhancement systems not available: {e}")
    ENHANCEMENTS_AVAILABLE = False
    # Set all enhancement classes to None for graceful fallback
    SimulationCoordinator = None
    EnhancedResearcherAgent = None
    # Individual bias models fallback
    AnchoringBiasModel = None
    ConfirmationBiasModel = None
    HaloEffectModel = None
    AvailabilityBiasModel = None
    BiasEngine = None
    VenueRegistry = None
    AcademicHierarchy = None
    ReputationCalculator = None
    StructuredReviewValidator = None
    CollaborationNetwork = None
    CitationNetwork = None
    FundingSystem = None
    TenureTrackManager = None
    WorkloadTracker = None
    DeadlineManager = None

# Load environment variables
load_dotenv("config.env")

# Initialize logger for this module
logger = get_logger(__name__)

class PeerReviewSimulation:
    """
    Main simulation system for the peer review process.
    """
    
    def __init__(self, workspace_dir: str = "peer_review_workspace", assign_papers: bool = False, enhanced_mode: bool = False):
        """
        Initialize the peer review simulation.
        
        Args:
            workspace_dir: Directory for storing simulation data
            assign_papers: Whether to automatically assign papers after initialization
                          (only effective if researchers are added before this call)
            enhanced_mode: Whether to enable enhanced features (biases, networks, career progression, etc.)
        """
        self.workspace_dir = workspace_dir
        self.enhanced_mode = enhanced_mode
        os.makedirs(workspace_dir, exist_ok=True)
        
        # Initialize enhancement systems if enhanced mode is enabled
        self.enhancement_systems = {}
        if enhanced_mode:
            logger.info("Initializing enhanced mode with all enhancement systems...")
            self._initialize_enhancement_systems()
        else:
            logger.info("Running in basic mode (no enhancements)")
        
        # Create paper database
        papers_path = os.path.join(workspace_dir, "papers.json")
        self.paper_db = PaperDatabase(data_path=papers_path)
        
        # Check if papers were loaded, if not, load from PeerRead dataset
        if not self.paper_db.get_all_papers():
            logger.info("No papers found in the database.")
            # Try to load from PeerRead dataset first
            try:
                logger.info("Loading papers from PeerRead dataset...")
                self.paper_db.load_peerread_dataset(use_test_dataset=False, limit=500)  # Load 500 real papers for diversity
                
                # Assign fields to papers based on conference/venue to match researcher specialties
                field_mappings = {
                    "acl": "Natural Language Processing", 
                    "cl": "Natural Language Processing",
                    "conll": "Natural Language Processing",
                    "iclr": "Artificial Intelligence",
                    "nips": "Artificial Intelligence",
                    "neurips": "Artificial Intelligence",
                    "cs.ai": "Artificial Intelligence",
                    "cs.lg": "Artificial Intelligence",
                    "cs.cv": "Computer Vision",
                    "cs.ro": "Robotics and Control Systems",
                    "cs.se": "Computer Systems and Architecture",
                    "cs.hc": "Human-Computer Interaction",
                    "cs.cr": "Cybersecurity and Privacy",
                    "cs.et": "AI Ethics and Fairness",
                    "cs.ds": "Data Science and Analytics",
                    "cs.cc": "Theoretical Computer Science"
                }
                
                # Papers already have fields assigned during loading, no need to reassign
                # The field assignment is handled in paper_database.py during load_peerread_dataset()
                papers = self.paper_db.get_all_papers()
                logger.info(f"Loaded papers with field distribution:")
                field_counts = {}
                for paper in papers:
                    field = paper.get("field", "Unknown")
                    field_counts[field] = field_counts.get(field, 0) + 1
                for field, count in field_counts.items():
                    logger.info(f"  {field}: {count} papers")
                
                # Save changes
                self.paper_db._save_data()
            except Exception as e:
                logger.warning(f"Failed to load PeerRead dataset: {e}")
                logger.info("Creating synthetic test papers as fallback...")
                self.create_test_papers()
        
        # Double check if we have papers, if not create test papers
        if not self.paper_db.get_all_papers():
            logger.warning("Still no papers found. Creating test papers directly...")
            self.create_test_papers()
        
        # Create token system
        tokens_path = os.path.join(workspace_dir, "tokens.json")
        self.token_system = TokenSystem(data_path=tokens_path)
        
        # LLM configuration - now uses centralized config system
        self.llm_config = get_llm_config()
        
        # Log LLM provider info
        provider_info = get_llm_provider_info()
        log_llm_config(provider_info, logger)
        
        # Initialize agents dictionary
        self.agents = {}
        
        # Initialize user proxy
        self._create_user_proxy()
        
        # Group chat configuration
        self.groupchat = None
        self.manager = None
        
        # Initialize standard venues if in enhanced mode (only if not already created)
        if self.enhanced_mode and self.enhancement_systems.get('venue_registry'):
            venue_registry = self.enhancement_systems['venue_registry']
            existing_venues = venue_registry.list_venues()
            if not existing_venues:
                self._initialize_standard_venues()
            else:
                logger.info(f"Using existing {len(existing_venues)} venues")
        
        # Assign imported papers to researchers if requested
        # (only effective if researchers are already created)
        if assign_papers and self.agents:
            self.assign_imported_papers_to_agents()
    
    # Enhancement system properties for easy access
    @property
    def collaboration_network(self):
        """Get the collaboration network system."""
        return self.enhancement_systems.get('collaboration_network')
    
    @property
    def citation_network(self):
        """Get the citation network system."""
        return self.enhancement_systems.get('citation_network')
    
    @property
    def venue_registry(self):
        """Get the venue registry system."""
        return self.enhancement_systems.get('venue_registry')
    
    @property
    def bias_engine(self):
        """Get the bias engine system."""
        return self.enhancement_systems.get('bias_engine')
    
    @property
    def academic_hierarchy(self):
        """Get the academic hierarchy system."""
        return self.enhancement_systems.get('academic_hierarchy')
    
    @property
    def reputation_calculator(self):
        """Get the reputation calculator system."""
        return self.enhancement_systems.get('reputation_calculator')
    
    @property
    def researchers(self):
        """Get dictionary of researchers by ID."""
        if self.enhanced_mode:
            # Return enhanced researchers
            return {agent.name: agent.researcher for agent in self.agents.values() 
                   if hasattr(agent, 'researcher')}
        else:
            # Return basic agent info
            return {name: agent for name, agent in self.agents.items()}
    
    @property
    def papers(self):
        """Get dictionary of papers by ID."""
        all_papers = self.paper_db.get_all_papers()
        return {paper['id']: paper for paper in all_papers}
    
    def _initialize_enhancement_systems(self):
        """Initialize all enhancement systems for enhanced mode."""
        if not ENHANCEMENTS_AVAILABLE:
            logger.error("Cannot initialize enhancement systems - imports failed")
            return
        
        try:
            # Initialize core enhancement systems
            logger.info("Initializing core enhancement systems...")
            
            # Academic hierarchy and reputation system
            if AcademicHierarchy:
                self.enhancement_systems['academic_hierarchy'] = AcademicHierarchy()
                logger.info("Academic hierarchy system initialized")
            
            if ReputationCalculator:
                self.enhancement_systems['reputation_calculator'] = ReputationCalculator()
                logger.info("Reputation calculator initialized")
            
            # Venue system
            if VenueRegistry:
                self.enhancement_systems['venue_registry'] = VenueRegistry()
                logger.info("Venue registry initialized")
            
            # Review system
            if StructuredReviewValidator:
                self.enhancement_systems['structured_review_validator'] = StructuredReviewValidator()
                logger.info("Structured review validator initialized")
            
            # Bias system
            if BiasEngine:
                bias_engine = BiasEngine()
                
                # Register individual bias models
                if AnchoringBiasModel:
                    from src.enhancements.bias_engine import BiasType, BiasConfiguration
                    anchoring_config = BiasConfiguration(
                        bias_type=BiasType.ANCHORING,
                        base_strength=0.4,
                        parameters={
                            'influence_decay': 0.8,
                            'confidence_weight': 0.3,
                            'max_influence': 1.0
                        }
                    )
                    bias_engine.register_bias_model(AnchoringBiasModel(anchoring_config))
                    logger.info("Registered anchoring bias model")
                
                if ConfirmationBiasModel:
                    confirmation_config = BiasConfiguration(
                        bias_type=BiasType.CONFIRMATION,
                        base_strength=0.3,
                        parameters={
                            'belief_alignment_threshold': 0.7,
                            'max_score_adjustment': 1.5
                        }
                    )
                    bias_engine.register_bias_model(ConfirmationBiasModel(confirmation_config))
                    logger.info("Registered confirmation bias model")
                
                if HaloEffectModel:
                    halo_config = BiasConfiguration(
                        bias_type=BiasType.HALO_EFFECT,
                        base_strength=0.2,
                        parameters={
                            'reputation_threshold': 0.7,
                            'max_score_boost': 2.0,
                            'prestige_factor': 0.5
                        }
                    )
                    bias_engine.register_bias_model(HaloEffectModel(halo_config))
                    logger.info("Registered halo effect bias model")
                
                if AvailabilityBiasModel:
                    availability_config = BiasConfiguration(
                        bias_type=BiasType.AVAILABILITY,
                        base_strength=0.3,
                        parameters={
                            'recency_window_days': 30,
                            'similarity_threshold': 0.6,
                            'max_adjustment': 1.0
                        }
                    )
                    bias_engine.register_bias_model(AvailabilityBiasModel(availability_config))
                    logger.info("Registered availability bias model")
                
                self.enhancement_systems['bias_engine'] = bias_engine
                logger.info("Bias engine initialized with individual bias models")
            
            # Network systems
            if CollaborationNetwork:
                self.enhancement_systems['collaboration_network'] = CollaborationNetwork()
                logger.info("Collaboration network initialized")
            
            if CitationNetwork:
                self.enhancement_systems['citation_network'] = CitationNetwork()
                logger.info("Citation network initialized")
            
            # Career and funding systems
            if TenureTrackManager:
                self.enhancement_systems['tenure_track_manager'] = TenureTrackManager()
                logger.info("Tenure track manager initialized")
            
            if FundingSystem:
                self.enhancement_systems['funding_system'] = FundingSystem()
                logger.info("Funding system initialized")
            
            # Temporal systems
            if WorkloadTracker:
                self.enhancement_systems['workload_tracker'] = WorkloadTracker()
                logger.info("Workload tracker initialized")
            
            if DeadlineManager:
                self.enhancement_systems['deadline_manager'] = DeadlineManager()
                logger.info("Deadline manager initialized")
            
            # Simulation coordinator (orchestrates all systems)
            if SimulationCoordinator:
                self.enhancement_systems['simulation_coordinator'] = SimulationCoordinator()
                logger.info("Simulation coordinator initialized")
            
            logger.info(f"Enhanced mode initialized with {len(self.enhancement_systems)} systems")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhancement systems: {e}")
            # Fall back to basic mode
            self.enhanced_mode = False
            self.enhancement_systems = {}
            logger.warning("Falling back to basic mode due to initialization failure")
    
    def _initialize_standard_venues(self):
        """Initialize standard venues in the venue registry."""
        try:
            venue_registry = self.enhancement_systems.get('venue_registry')
            if venue_registry:
                # Create standard venues from profiles
                created_venues = venue_registry.create_standard_venues()
                logger.info(f"Initialized {len(created_venues)} standard venues")
                
                # Log available venues
                for venue in created_venues:
                    logger.info(f"  - {venue.name} ({venue.venue_type.value}, {venue.field})")
            
        except Exception as e:
            logger.error(f"Failed to initialize standard venues: {e}")
    
    def _create_user_proxy(self):
        """Create the default user proxy agent."""
        self.user_proxy = autogen.UserProxyAgent(
            name="User_Proxy",
            system_message="A human user that interacts with the AI system, providing tasks, feedback, and approving or rejecting plans.",
            human_input_mode="ALWAYS",
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": self.workspace_dir,
                "use_docker": False
            }
        )
    
    def add_researcher_from_template(self, template_name, custom_name: Optional[str] = None) -> Optional[str]:
        """
        Add a researcher agent from a template.
        
        Args:
            template_name: Name of the researcher template or ResearcherTemplate enum
            custom_name: Optional custom name for the researcher (overrides template name)
            
        Returns:
            The researcher ID (name) or None if template not found
        """
        # Handle both string template names and ResearcherTemplate enums
        if hasattr(template_name, 'value'):
            template_name = template_name.value
        
        template = get_researcher_template(template_name)
        if not template:
            print(f"Template '{template_name}' not found.")
            return None
        
        # Use custom name if provided, otherwise use template name
        researcher_name = custom_name if custom_name else template["name"]
        
        # Create researcher agent (enhanced or basic based on mode)
        if self.enhanced_mode and EnhancedResearcherAgent:
            logger.info(f"Creating enhanced researcher agent: {researcher_name}")
            researcher = EnhancedResearcherAgent(
                name=researcher_name,
                specialty=template["specialty"],
                system_message=template["system_message"],
                paper_db=self.paper_db,
                token_system=self.token_system,
                bias=template.get("bias", ""),
                llm_config=self.llm_config,
                simulation_coordinator=self.enhancement_systems.get('simulation_coordinator')
            )
        else:
            logger.info(f"Creating basic researcher agent: {researcher_name}")
            researcher = ResearcherAgent(
                name=researcher_name,
                specialty=template["specialty"],
                system_message=template["system_message"],
                paper_db=self.paper_db,
                token_system=self.token_system,
                bias=template.get("bias", ""),
                llm_config=self.llm_config
            )
        
        # Add to agents dictionary
        self.agents[researcher_name] = researcher
        
        # Update collaboration network if in enhanced mode
        if self.enhanced_mode and self.collaboration_network:
            self._update_collaboration_network()
        
        print(f"Added researcher agent: {researcher_name}")
        return researcher_name
    
    def _update_collaboration_network(self):
        """Update collaboration network with current researchers."""
        if not self.collaboration_network:
            return
        
        try:
            # Get all enhanced researchers
            researchers = []
            for agent in self.agents.values():
                if hasattr(agent, 'researcher'):
                    researchers.append(agent.researcher)
            
            if researchers:
                self.collaboration_network.build_network_from_researchers(researchers)
                logger.debug(f"Updated collaboration network with {len(researchers)} researchers")
        except Exception as e:
            logger.warning(f"Failed to update collaboration network: {e}")
    
    def submit_paper(self, author_id: str, title: str, abstract: str, venue_type=None) -> str:
        """
        Submit a new paper to the system.
        
        Args:
            author_id: ID of the paper author
            title: Paper title
            abstract: Paper abstract
            venue_type: Type of venue to submit to (optional)
            
        Returns:
            Paper ID
        """
        # Create paper data
        paper_data = {
            "title": title,
            "abstract": abstract,
            "authors": [author_id],
            "field": "Artificial Intelligence",  # Default field
            "status": "submitted",
            "owner_id": author_id,
            "content": f"Title: {title}\n\nAbstract: {abstract}",
            "publication_date": datetime.now().strftime("%Y-%m-%d"),
            "citations": 0,
            "reviews": [],
            "review_requests": []
        }
        
        # Add paper to database
        paper_id = self.paper_db.add_paper(paper_data)
        
        logger.info(f"Paper {paper_id} submitted by {author_id}: {title}")
        return paper_id
    
    def create_all_researchers(self, assign_papers=False):
        """
        Create researcher agents for all templates.
        
        Args:
            assign_papers: If True, automatically assign imported papers to researchers
        """
        for template_name in list_researcher_templates():
            self.add_researcher_from_template(template_name)
        
        if assign_papers:
            self.assign_imported_papers_to_agents()
    
    def create_group_chat(self, agents: Optional[List[str]] = None):
        """
        Create a group chat with specified agents.
        
        Args:
            agents: List of agent names to include (if None, includes all)
        """
        chat_agents = [self.user_proxy]
        
        if agents:
            # Add specified agents
            for name in agents:
                if name in self.agents:
                    chat_agents.append(self.agents[name])
        else:
            # Add all agents
            chat_agents.extend(list(self.agents.values()))
        
        # Create group chat
        self.groupchat = autogen.GroupChat(
            agents=chat_agents,
            messages=[],
            max_round=50
        )
        
        self.manager = autogen.GroupChatManager(
            groupchat=self.groupchat,
            llm_config=self.llm_config,
        )
    
    def start_chat(self, initial_message: str):
        """
        Start a group chat with the given message.
        
        Args:
            initial_message: Initial message to start the chat with
        """
        if not self.manager:
            self.create_group_chat()
        
        self.user_proxy.initiate_chat(
            self.manager,
            message=initial_message
        )
    
    def submit_paper_to_venue(self, paper_id: str, venue_id: str, author_id: str) -> bool:
        """
        Submit a paper to a specific venue.
        
        Args:
            paper_id: ID of the paper to submit
            venue_id: ID of the target venue
            author_id: ID of the submitting author
            
        Returns:
            True if submission successful, False otherwise
        """
        try:
            # Get paper and venue
            paper = self.paper_db.get_paper(paper_id)
            if not paper:
                logger.error(f"Paper {paper_id} not found")
                return False
            
            venue_registry = self.enhancement_systems.get('venue_registry')
            if not venue_registry:
                logger.warning("Venue registry not available, using basic submission")
                return True
            
            venue = venue_registry.get_venue(venue_id)
            if not venue:
                logger.error(f"Venue {venue_id} not found")
                return False
            
            # Check if paper field matches venue field
            if paper.get('field') != venue.field:
                logger.warning(f"Paper field '{paper.get('field')}' doesn't match venue field '{venue.field}'")
            
            # Update paper status and venue
            paper['status'] = 'submitted'
            paper['venue_id'] = venue_id
            paper['submission_date'] = datetime.now().isoformat()
            
            # Save changes
            self.paper_db._save_data()
            
            logger.info(f"Paper {paper_id} submitted to venue {venue.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit paper {paper_id} to venue {venue_id}: {e}")
            return False
    
    def get_suitable_venues_for_paper(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        Get suitable venues for a paper based on its field and characteristics.
        
        Args:
            paper_id: ID of the paper
            
        Returns:
            List of suitable venues with match scores
        """
        try:
            paper = self.paper_db.get_paper(paper_id)
            if not paper:
                return []
            
            venue_registry = self.enhancement_systems.get('venue_registry')
            if not venue_registry:
                return []
            
            # Get all venues and check suitability
            all_venues = venue_registry.list_venues()
            suitable_venues = []
            
            # Check each venue for suitability
            for venue in all_venues:
                score = self._calculate_venue_suitability_score(paper, venue)
                if score > 0.1:  # Include venues with any reasonable suitability
                    suitable_venues.append(venue)
            
            # Score venues based on suitability
            venue_scores = []
            for venue in suitable_venues:
                score = self._calculate_venue_suitability_score(paper, venue)
                venue_scores.append({
                    'venue': venue,
                    'suitability_score': score,
                    'acceptance_rate': venue.acceptance_rate,
                    'prestige_score': venue.prestige_score
                })
            
            # Sort by suitability score
            venue_scores.sort(key=lambda x: x['suitability_score'], reverse=True)
            
            return venue_scores
            
        except Exception as e:
            logger.error(f"Failed to get suitable venues for paper {paper_id}: {e}")
            return []
    
    def _calculate_venue_suitability_score(self, paper: Dict[str, Any], venue) -> float:
        """Calculate how suitable a venue is for a paper."""
        score = 0.0
        
        # Field match (most important) - be flexible with field matching
        paper_field = paper.get('field', '').lower()
        venue_field = venue.field.lower()
        
        if paper_field == venue_field:
            score += 0.5
        elif self._fields_are_related(paper_field, venue_field):
            score += 0.3  # Partial match for related fields
        
        # Keywords match (if available)
        paper_keywords = paper.get('keywords', [])
        if paper_keywords:
            # Simple keyword matching - could be enhanced with semantic similarity
            keyword_matches = 0
            for keyword in paper_keywords:
                if keyword.lower() in venue.name.lower():
                    keyword_matches += 1
            if paper_keywords:
                score += (keyword_matches / len(paper_keywords)) * 0.3
        
        # Venue prestige vs paper quality (estimated)
        # This is a simplified heuristic - could be enhanced with actual quality metrics
        estimated_paper_quality = len(paper.get('abstract', '')) / 200.0  # Simple heuristic
        prestige_match = 1.0 - abs(venue.prestige_score / 10.0 - min(estimated_paper_quality, 1.0))
        score += prestige_match * 0.2
        
        return min(score, 1.0)
    
    def _fields_are_related(self, field1: str, field2: str) -> bool:
        """Check if two research fields are related."""
        # Define field relationships - AI is broadly related to many fields
        field_relationships = {
            'artificial intelligence': [
                'machine learning', 'natural language processing', 'computer vision',
                'data science and analytics', 'robotics and control systems'
            ],
            'machine learning': [
                'artificial intelligence', 'data science and analytics', 
                'natural language processing', 'computer vision'
            ],
            'natural language processing': [
                'artificial intelligence', 'machine learning', 'data science and analytics'
            ],
            'computer vision': [
                'artificial intelligence', 'machine learning', 'data science and analytics'
            ],
            'data science and analytics': [
                'artificial intelligence', 'machine learning', 'natural language processing'
            ],
            'robotics and control systems': [
                'artificial intelligence', 'computer systems and architecture'
            ]
        }
        
        field1 = field1.lower()
        field2 = field2.lower()
        
        # Direct match
        if field1 == field2:
            return True
        
        # Check relationships
        return field2 in field_relationships.get(field1, []) or field1 in field_relationships.get(field2, [])
    
    def assign_reviewers_for_venue_submission(self, paper_id: str, venue_id: str, num_reviewers: int = 3) -> List[str]:
        """
        Assign reviewers for a paper submitted to a venue.
        
        Args:
            paper_id: ID of the paper
            venue_id: ID of the venue
            num_reviewers: Number of reviewers to assign
            
        Returns:
            List of assigned reviewer IDs
        """
        try:
            venue_registry = self.enhancement_systems.get('venue_registry')
            if not venue_registry:
                # Fallback to basic assignment
                return list(self.agents.keys())[:num_reviewers]
            
            venue = venue_registry.get_venue(venue_id)
            if not venue:
                logger.error(f"Venue {venue_id} not found")
                return []
            
            # Get qualified reviewers for this venue
            qualified_reviewers = []
            for agent_name, agent in self.agents.items():
                if hasattr(agent, 'enhanced_profile'):
                    # Check if reviewer meets venue criteria
                    if self._meets_venue_reviewer_criteria(agent.enhanced_profile, venue):
                        qualified_reviewers.append(agent_name)
                else:
                    # Basic agent - check field match (be flexible)
                    if (agent.specialty.lower() == venue.field.lower() or 
                        self._fields_are_related(agent.specialty.lower(), venue.field.lower())):
                        qualified_reviewers.append(agent_name)
            
            # Select reviewers (avoiding conflicts of interest if possible)
            selected_reviewers = self._select_reviewers_avoiding_conflicts(
                paper_id, qualified_reviewers, num_reviewers
            )
            
            logger.info(f"Assigned {len(selected_reviewers)} reviewers to paper {paper_id} for venue {venue.name}")
            return selected_reviewers
            
        except Exception as e:
            logger.error(f"Failed to assign reviewers for paper {paper_id} at venue {venue_id}: {e}")
            return []
    
    def _meets_venue_reviewer_criteria(self, researcher_profile, venue) -> bool:
        """Check if a researcher meets venue reviewer criteria."""
        try:
            # Check field match (be flexible) - this is the most important criterion
            researcher_field = getattr(researcher_profile, 'specialty', '').lower()
            venue_field = venue.field.lower()
            
            field_match = (researcher_field == venue_field or 
                          self._fields_are_related(researcher_field, venue_field))
            
            if not field_match:
                logger.debug(f"Field mismatch: researcher '{researcher_field}' vs venue '{venue_field}'")
                return False
            
            # Check minimum h-index requirement (be lenient for testing)
            if hasattr(researcher_profile, 'h_index') and hasattr(venue, 'reviewer_selection_criteria'):
                min_h_index = getattr(venue.reviewer_selection_criteria, 'min_h_index', 0)
                if researcher_profile.h_index < min_h_index:
                    logger.debug(f"H-index too low: {researcher_profile.h_index} < {min_h_index}")
                    # Be lenient - allow if h-index is at least half the requirement
                    if researcher_profile.h_index < min_h_index / 2:
                        return False
            
            # Check seniority level (be lenient)
            if hasattr(researcher_profile, 'level') and hasattr(venue, 'reviewer_selection_criteria'):
                required_levels = getattr(venue.reviewer_selection_criteria, 'required_seniority_levels', [])
                if required_levels and researcher_profile.level.value not in required_levels:
                    logger.debug(f"Seniority level not in required: {researcher_profile.level.value} not in {required_levels}")
                    # Be lenient - allow anyway for testing
                    pass
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking venue criteria: {e}")
            return True  # Default to allowing if check fails
    
    def _select_reviewers_avoiding_conflicts(self, paper_id: str, qualified_reviewers: List[str], num_reviewers: int) -> List[str]:
        """Select reviewers while avoiding conflicts of interest."""
        try:
            paper = self.paper_db.get_paper(paper_id)
            if not paper:
                return qualified_reviewers[:num_reviewers]
            
            paper_authors = paper.get('authors', [])
            
            # Filter out potential conflicts
            non_conflicted_reviewers = []
            for reviewer_id in qualified_reviewers:
                agent = self.agents.get(reviewer_id)
                if agent:
                    # Check if reviewer is an author
                    if agent.name not in paper_authors:
                        # Check collaboration network if available
                        if self._check_collaboration_conflicts(agent, paper_authors):
                            non_conflicted_reviewers.append(reviewer_id)
                        else:
                            logger.info(f"Skipping {reviewer_id} due to collaboration conflict")
                    else:
                        logger.info(f"Skipping {reviewer_id} - is paper author")
            
            # Select from non-conflicted reviewers first
            selected = non_conflicted_reviewers[:num_reviewers]
            
            # If not enough non-conflicted reviewers, add from qualified pool
            if len(selected) < num_reviewers:
                remaining_needed = num_reviewers - len(selected)
                additional = [r for r in qualified_reviewers if r not in selected][:remaining_needed]
                selected.extend(additional)
            
            return selected
            
        except Exception as e:
            logger.error(f"Error selecting reviewers: {e}")
            return qualified_reviewers[:num_reviewers]
    
    def _check_collaboration_conflicts(self, reviewer_agent, paper_authors: List[str]) -> bool:
        """Check if reviewer has collaboration conflicts with paper authors."""
        try:
            collaboration_network = self.enhancement_systems.get('collaboration_network')
            if not collaboration_network:
                return True  # No conflict checking available
            
            # Check for conflicts of interest using the proper API
            reviewer_id = reviewer_agent.name
            conflicts = collaboration_network.detect_conflicts_of_interest(
                paper_authors=paper_authors,
                potential_reviewer=reviewer_id
            )
            
            # Check if there are any significant conflicts (strength >= 0.5)
            significant_conflicts = [c for c in conflicts if c.conflict_strength >= 0.5]
            
            if significant_conflicts:
                logger.debug(f"Collaboration conflicts detected for reviewer {reviewer_id}: "
                           f"{len(significant_conflicts)} conflicts")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking collaboration conflicts: {e}")
            return True  # Default to no conflict if check fails
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the peer review system.
        
        Returns:
            Dictionary with system statistics
        """
        # Get token statistics
        token_stats = self.token_system.get_review_statistics()
        
        # Get paper statistics
        papers = self.paper_db.get_all_papers()
        paper_stats = {
            "total_papers": len(papers),
            "papers_by_status": {
                "draft": len(self.paper_db.get_papers_by_status("draft")),
                "submitted": len(self.paper_db.get_papers_by_status("submitted")),
                "in_review": len(self.paper_db.get_papers_by_status("in_review")),
                "published": len(self.paper_db.get_papers_by_status("published"))
            }
        }
        
        # Get researcher statistics
        researcher_stats = {
            "total_researchers": len(self.agents),
            "researchers": {}
        }
        
        # Calculate statistics for each researcher
        for name, agent in self.agents.items():
            researcher_id = agent.name
            balance = self.token_system.get_balance(researcher_id)
            published_papers = len(self.paper_db.get_papers_by_owner(researcher_id))
            reviews_completed = len(self.token_system.get_reviews_by_reviewer(researcher_id))
            
            researcher_stats["researchers"][researcher_id] = {
                "balance": balance,
                "published_papers": published_papers,
                "reviews_completed": reviews_completed
            }
        
        # Create a leaderboard based on token balance
        leaderboard = []
        for researcher_id, stats in researcher_stats["researchers"].items():
            leaderboard.append({
                "researcher_id": researcher_id,
                "balance": stats["balance"],
                "published_papers": stats["published_papers"],
                "reviews_completed": stats["reviews_completed"]
            })
        
        # Sort leaderboard by token balance
        leaderboard.sort(key=lambda x: x["balance"], reverse=True)
        
        # Get venue statistics if enhanced mode is enabled
        venue_stats = {}
        if self.enhanced_mode and self.enhancement_systems.get('venue_registry'):
            venue_registry = self.enhancement_systems['venue_registry']
            venues = venue_registry.list_venues()
            
            venue_stats = {
                "total_venues": len(venues),
                "venues_by_type": {},
                "venues_by_field": {},
                "venues": []
            }
            
            # Count venues by type and field
            for venue in venues:
                venue_type = venue.venue_type.value
                venue_field = venue.field
                
                venue_stats["venues_by_type"][venue_type] = venue_stats["venues_by_type"].get(venue_type, 0) + 1
                venue_stats["venues_by_field"][venue_field] = venue_stats["venues_by_field"].get(venue_field, 0) + 1
                
                # Count submissions to this venue
                submissions = len([p for p in papers if p.get('venue_id') == venue.id])
                
                venue_stats["venues"].append({
                    "id": venue.id,
                    "name": venue.name,
                    "type": venue_type,
                    "field": venue_field,
                    "acceptance_rate": venue.acceptance_rate,
                    "prestige_score": venue.prestige_score,
                    "submissions": submissions
                })
        
        return {
            "token_stats": token_stats,
            "paper_stats": paper_stats,
            "researcher_stats": researcher_stats,
            "venue_stats": venue_stats,
            "leaderboard": leaderboard
        }
    
    def simulate_random_interactions(self, num_interactions: int = 10) -> List[Dict[str, Any]]:
        """
        Simulate random interactions between researchers.
        
        Args:
            num_interactions: Number of interactions to simulate
            
        Returns:
            List of interaction outcomes
        """
        outcomes = []
        researcher_names = list(self.agents.keys())
        
        if len(researcher_names) < 2:
            return [{"error": "Need at least 2 researchers for simulation"}]
        
        # Use the centralized specialty compatibility matrix from constants
        from src.core.constants import SPECIALTY_COMPATIBILITY
        specialty_compatibility = SPECIALTY_COMPATIBILITY
        
        # Create a dictionary of researchers by specialty for quick lookup
        researchers_by_specialty = {}
        for name, agent in self.agents.items():
            if agent.specialty not in researchers_by_specialty:
                researchers_by_specialty[agent.specialty] = []
            researchers_by_specialty[agent.specialty].append(name)
        
        # Log researcher specialties
        print("\nResearcher specialties:")
        for specialty, researchers in researchers_by_specialty.items():
            print(f"  {specialty}: {', '.join(researchers)}")
        
        print("\nStarting interactions...")
        for i in range(num_interactions):
            # Select random interaction type with better balance
            # 40% request reviews, 30% respond to invitations, 30% complete reviews
            interaction_type = random.choices(
                ["request_review", "respond_to_invitation", "complete_review"], 
                weights=[40, 30, 30]
            )[0]
            
            if interaction_type == "request_review":
                # Random requester
                requester_name = random.choice(researcher_names)
                requester = self.agents[requester_name]
                
                # Get a paper owned by requester
                papers = requester.get_papers()
                print(f"\nRequester {requester_name} has {len(papers)} papers")
                
                if not papers:
                    outcome = {
                        "interaction": "request_review",
                        "requester": requester_name,
                        "success": False,
                        "message": f"Requester {requester_name} has no papers"
                    }
                else:
                    # Select random paper
                    paper = random.choice(papers)
                    # Use the updated token range from constants
                    from src.core.constants import REVIEW_REQUEST_TOKEN_RANGE
                    token_amount = random.randint(REVIEW_REQUEST_TOKEN_RANGE[0], REVIEW_REQUEST_TOKEN_RANGE[1])
                    
                    # Get paper field, defaulting to the requester's specialty if not set
                    paper_field = paper.get('field', requester.specialty)
                    print(f"Paper ID {paper['id']} (field: {paper_field}) selected for review request")
                    
                    # Get compatible specialties for this paper
                    compatible_specialties = specialty_compatibility.get(paper_field, [paper_field])
                    print(f"Compatible specialties: {compatible_specialties}")
                    
                    # Find all researchers with compatible specialties
                    valid_reviewers = []
                    for specialty in compatible_specialties:
                        if specialty in researchers_by_specialty:
                            valid_reviewers.extend([
                                name for name in researchers_by_specialty[specialty] 
                                if name != requester_name
                            ])
                    
                    # Remove duplicates
                    valid_reviewers = list(set(valid_reviewers))
                    print(f"Valid reviewers: {valid_reviewers}")
                    
                    if not valid_reviewers:
                        outcome = {
                            "interaction": "request_review",
                            "requester": requester_name,
                            "paper_id": paper["id"],
                            "success": False,
                            "message": f"No valid reviewers found with specialty compatible with paper field: {paper_field}"
                        }
                    else:
                        # Select random reviewer from valid reviewers
                        reviewer_name = random.choice(valid_reviewers)
                        print(f"Selected reviewer: {reviewer_name}")
                        
                        # Request review
                        success, message = requester.request_review(
                            paper_id=paper["id"],
                            reviewer_id=reviewer_name,
                            token_amount=token_amount
                        )
                        
                        print(f"Review request result: {success}, {message}")
                        
                        outcome = {
                            "interaction": "request_review",
                            "requester": requester_name,
                            "reviewer": reviewer_name,
                            "paper_id": paper["id"],
                            "paper_field": paper_field,
                            "reviewer_specialty": self.agents[reviewer_name].specialty,
                            "token_amount": token_amount,
                            "success": success,
                            "message": message
                        }
            
            elif interaction_type == "respond_to_invitation":
                # Handle responding to pending review invitations
                pending_requests = []
                for agent_name, agent in self.agents.items():
                    # Get papers with pending review requests for this agent
                    all_papers = self.paper_db.get_all_papers()
                    for paper in all_papers:
                        for request in paper.get('review_requests', []):
                            if request.get('reviewer_id') == agent_name and request.get('status') == 'pending':
                                pending_requests.append((agent_name, paper, request))
                
                print(f"\nFound {len(pending_requests)} pending review requests")
                
                if pending_requests:
                    reviewer_name, paper, request = random.choice(pending_requests)
                    reviewer = self.agents[reviewer_name]
                    print(f"Reviewer {reviewer_name} considering review request for paper ID {paper['id']}")
                    
                    # Have the reviewer respond to the invitation
                    accepted, response_message, thought_process = reviewer.respond_to_invitation(
                        paper_id=paper["id"],
                        token_amount=request.get('token_amount', 0)
                    )
                    
                    print(f"Review invitation response: {accepted}, {response_message}")
                    
                    outcome = {
                        "interaction": "respond_to_review_request",
                        "reviewer": reviewer_name,
                        "reviewer_specialty": reviewer.specialty,
                        "paper_id": paper["id"],
                        "paper_field": paper.get('field', 'Unknown'),
                        "accepted": accepted,
                        "success": True,
                        "message": response_message,
                        "thought_process": thought_process
                    }
                else:
                    outcome = {
                        "interaction": "respond_to_review_request",
                        "success": False,
                        "message": "No pending review requests found"
                    }
            
            elif interaction_type == "complete_review":
                # Handle completing accepted reviews
                accepted_reviews = []
                for agent_name, agent in self.agents.items():
                    all_papers = self.paper_db.get_all_papers()
                    for paper in all_papers:
                        for request in paper.get('review_requests', []):
                            if request.get('reviewer_id') == agent_name and request.get('status') == 'accepted':
                                accepted_reviews.append((agent_name, paper))
                
                print(f"\nFound {len(accepted_reviews)} accepted reviews to complete")
                
                if not accepted_reviews:
                    outcome = {
                        "interaction": "complete_review",
                        "success": False,
                        "message": "No accepted reviews found to complete"
                    }
                else:
                    # Select random accepted review to complete
                    reviewer_name, paper = random.choice(accepted_reviews)
                    reviewer = self.agents[reviewer_name]
                    print(f"Selected reviewer {reviewer_name} to complete review for paper ID {paper['id']}")
                    
                    # Generate and submit review
                    review_result = reviewer.generate_review(paper["id"])
                    success, message = reviewer.submit_review(
                        paper_id=paper["id"],
                        review_content=review_result.get("review_content", {})
                    )
                    
                    # Award tokens for completing the review
                    if success:
                        completion_success = self.token_system.complete_review(reviewer_name, paper["id"])
                        if completion_success:
                            print(f"✅ {reviewer_name} earned completion bonus for reviewing paper {paper['id']}")
                        else:
                            print(f"⚠️ Could not award completion bonus to {reviewer_name} for paper {paper['id']}")
                    
                    print(f"Review submission result: {success}, {message}")
                    
                    outcome = {
                        "interaction": "complete_review",
                        "reviewer": reviewer_name,
                        "reviewer_specialty": reviewer.specialty,
                        "paper_id": paper["id"],
                        "paper_field": paper.get('field', 'Unknown'),
                        "success": success,
                        "message": message,
                        "thought_process": review_result.get("thought_process", "")
                    }
            
            outcomes.append(outcome)
            
            # Save interaction to JSON file for analysis
            self._save_interaction(outcome)
        
        return outcomes
    
    def _save_interaction(self, interaction: Dict[str, Any]):
        """Save interaction to a JSON file."""
        interactions_dir = os.path.join(self.workspace_dir, "interactions")
        os.makedirs(interactions_dir, exist_ok=True)
        
        # Create a filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"interaction_{timestamp}.json"
        
        # Save to file
        with open(os.path.join(interactions_dir, filename), 'w') as f:
            json.dump(interaction, f, indent=2)
    
    def get_researcher_agents(self) -> Dict[str, ResearcherAgent]:
        """
        Get all researcher agents.
        
        Returns:
            Dictionary of researcher agents
        """
        return self.agents
    
    def get_researcher(self, name: str) -> Optional[ResearcherAgent]:
        """
        Get a researcher agent by name.
        
        Args:
            name: Name of the researcher
            
        Returns:
            ResearcherAgent or None if not found
        """
        return self.agents.get(name)
    
    def list_researchers(self) -> List[str]:
        """
        List all researcher names.
        
        Returns:
            List of researcher names
        """
        return list(self.agents.keys())
    
    def run_simulation_rounds(self, num_rounds: int = 10, interactions_per_round: int = 5) -> Dict[str, Any]:
        """
        Run multiple rounds of simulation.
        
        Args:
            num_rounds: Number of rounds to simulate
            interactions_per_round: Number of interactions per round
            
        Returns:
            Dictionary with simulation results
        """
        round_results = []
        
        print(f"Starting simulation with {num_rounds} rounds, {interactions_per_round} interactions per round")
        
        for round_num in range(1, num_rounds + 1):
            print(f"\nRound {round_num}/{num_rounds}:")
            
            # Simulate interactions
            interaction_results = self.simulate_random_interactions(interactions_per_round)
            
            # Get system stats after this round
            stats = self.get_system_stats()
            
            # Save round results
            round_result = {
                "round": round_num,
                "interactions": interaction_results,
                "stats": stats
            }
            round_results.append(round_result)
            
            # Print brief summary
            review_requests = sum(1 for i in interaction_results if i["interaction"] == "request_review" and i["success"])
            review_responses = sum(1 for i in interaction_results if i["interaction"] == "respond_to_review_request" and i["success"])
            review_acceptances = sum(1 for i in interaction_results if i["interaction"] == "respond_to_review_request" and i.get("accepted", False))
            reviews_completed = sum(1 for i in interaction_results if i["interaction"] == "complete_review" and i["success"])
            
            print(f"  - Review requests: {review_requests}/{sum(1 for i in interaction_results if i['interaction'] == 'request_review')}")
            print(f"  - Review responses: {review_responses} (accepted: {review_acceptances})")
            print(f"  - Reviews completed: {reviews_completed}/{sum(1 for i in interaction_results if i['interaction'] == 'complete_review')}")
            
            # Print token balances
            print("\n  Current Token Balances:")
            for researcher in stats["leaderboard"][:5]:  # Show top 5
                print(f"  - {researcher['researcher_id']}: {researcher['balance']} tokens")
            
            if len(stats["leaderboard"]) > 5:
                print(f"    ... and {len(stats['leaderboard']) - 5} more")
        
        # Save full simulation results
        simulation_results = {
            "rounds": round_results,
            "final_stats": self.get_system_stats()
        }
        
        results_path = os.path.join(self.workspace_dir, "simulation_results.json")
        with open(results_path, 'w') as f:
            json.dump(simulation_results, f, indent=2)
        
        print(f"\nSimulation completed. Results saved to {results_path}")
        
        return simulation_results

    def assign_imported_papers_to_agents(self):
        """
        Assign imported papers from PeerRead to actual researcher agents.
        This reassigns papers with owner_id 'Imported_PeerRead' to random agents.
        """
        # Skip if no agents
        if not self.agents:
            print("No agents available to assign papers to.")
            return
        
        # Use the centralized specialty compatibility matrix from constants
        from src.core.constants import SPECIALTY_COMPATIBILITY
        specialty_compatibility = SPECIALTY_COMPATIBILITY
        
        # Group agents by specialty
        agents_by_specialty = {}
        for name, agent in self.agents.items():
            if agent.specialty not in agents_by_specialty:
                agents_by_specialty[agent.specialty] = []
            agents_by_specialty[agent.specialty].append(name)
        
        reassigned_count = 0
        
        # Get all papers
        papers = self.paper_db.get_all_papers()
        
        # First pass: Ensure every researcher gets at least 5 papers
        min_papers_per_researcher = 5
        # Initialize with current paper counts for each researcher
        papers_per_researcher = {
            agent_name: len(self.paper_db.get_papers_by_owner(agent_name)) 
            for agent_name in self.agents.keys()
        }
        unassigned_papers = []
        
        # Collect all papers that need assignment
        for paper in papers:
            if paper["owner_id"] == "Imported_PeerRead":
                unassigned_papers.append(paper)
        
        print(f"Found {len(unassigned_papers)} papers to assign to {len(self.agents)} researchers")
        print(f"Target: {min_papers_per_researcher} papers minimum per researcher")
        
        # First pass: Assign papers to researchers with 0 papers, prioritizing field matches
        for paper in unassigned_papers[:]:  # Use slice to allow modification during iteration
            paper_field = paper.get("field", "Artificial Intelligence")
            
            # Find researchers who need more papers
            researchers_needing_papers = [
                agent_name for agent_name, count in papers_per_researcher.items() 
                if count < min_papers_per_researcher
            ]
            
            if not researchers_needing_papers:
                break  # All researchers have minimum papers
            
            # Find compatible researchers among those who need papers
            compatible_specialties = specialty_compatibility.get(paper_field, [paper_field])
            compatible_and_needy = []
            
            for specialty in compatible_specialties:
                if specialty in agents_by_specialty:
                    for agent_name in agents_by_specialty[specialty]:
                        if agent_name in researchers_needing_papers:
                            compatible_and_needy.append(agent_name)
            
            # If no compatible researchers need papers, assign to any researcher who needs papers
            if not compatible_and_needy:
                new_owner = random.choice(researchers_needing_papers)
            else:
                new_owner = random.choice(compatible_and_needy)
            
            # Assign the paper
            paper["owner_id"] = new_owner
            self.paper_db.update_paper(paper["id"], {"owner_id": new_owner})
            papers_per_researcher[new_owner] += 1
            reassigned_count += 1
            unassigned_papers.remove(paper)
            
            print(f"[PRIORITY] Assigned '{paper.get('title', 'Untitled')[:50]}...' (field: {paper_field}) to {new_owner} (now has {papers_per_researcher[new_owner]} papers)")
        
        # Second pass: Distribute remaining papers normally
        for paper in unassigned_papers:
            paper_field = paper.get("field", "Artificial Intelligence")
            
            # Find agents with compatible specialties
            compatible_specialties = specialty_compatibility.get(paper_field, [paper_field])
            matching_agents = []
            
            for specialty in compatible_specialties:
                if specialty in agents_by_specialty:
                    matching_agents.extend(agents_by_specialty[specialty])
            
            # If no matching agents, fall back to any agent
            if not matching_agents:
                new_owner = random.choice(list(self.agents.keys()))
            else:
                new_owner = random.choice(matching_agents)
            
            # Update the paper's owner (keep original field)
            paper["owner_id"] = new_owner
            self.paper_db.update_paper(paper["id"], {"owner_id": new_owner})
            papers_per_researcher[new_owner] += 1
            reassigned_count += 1
            
            print(f"[NORMAL] Assigned '{paper.get('title', 'Untitled')[:50]}...' (field: {paper_field}) to {new_owner} (now has {papers_per_researcher[new_owner]} papers)")
        
        # Report final distribution
        print(f"\nFinal paper distribution:")
        for agent_name, count in sorted(papers_per_researcher.items()):
            status = "✅" if count >= min_papers_per_researcher else "❌"
            print(f"  {agent_name}: {count} papers {status}")
        
        # Verify no researcher has 0 papers
        zero_paper_researchers = [name for name, count in papers_per_researcher.items() if count == 0]
        if zero_paper_researchers:
            print(f"❌ WARNING: {len(zero_paper_researchers)} researchers still have 0 papers: {zero_paper_researchers}")
        else:
            print(f"✅ SUCCESS: All researchers have papers")
        
        # Save changes to disk
        self.paper_db._save_data()
        
        print(f"Reassigned {reassigned_count} imported papers to researcher agents.")
        return reassigned_count

    def create_test_papers(self):
        """Create test papers directly in the paper database."""
        print("Creating test papers for simulation...")
        
        # Define some test papers for each research area using EXACT specialty names
        test_papers = [
            {
                "title": "Deep Learning for Natural Language Processing",
                "abstract": "This paper explores deep learning approaches for NLP tasks.",
                "authors": ["Imported_Author"],
                "venue": "ACL 2023",
                "keywords": ["deep learning", "NLP", "transformer"],
                "field": "Natural Language Processing",
                "status": "published",
                "review_requests": [],
                "reviews": []
            },
            {
                "title": "Reinforcement Learning in Robotic Control",
                "abstract": "This paper presents a novel approach to robotic control using RL.",
                "authors": ["Imported_Author"],
                "venue": "Robotics Conference 2023",
                "keywords": ["reinforcement learning", "robotics", "control"],
                "field": "Robotics and Control Systems",
                "status": "published",
                "review_requests": [],
                "reviews": []
            },
            {
                "title": "Computer Vision Techniques for Object Detection",
                "abstract": "This paper introduces a new method for object detection in images.",
                "authors": ["Imported_Author"],
                "venue": "CVPR 2023",
                "keywords": ["computer vision", "object detection", "CNN"],
                "field": "Computer Vision",
                "status": "published",
                "review_requests": [],
                "reviews": []
            },
            {
                "title": "Theoretical Foundations of Machine Learning",
                "abstract": "This paper explores the theoretical underpinnings of ML algorithms.",
                "authors": ["Imported_Author"],
                "venue": "Theoretical CS 2023",
                "keywords": ["theory", "machine learning", "algorithms"],
                "field": "Theoretical Computer Science",
                "status": "published",
                "review_requests": [],
                "reviews": []
            },
            {
                "title": "Ethical Considerations in AI Development",
                "abstract": "This paper discusses ethical issues in AI development and deployment.",
                "authors": ["Imported_Author"],
                "venue": "AI Ethics Conference 2023",
                "keywords": ["ethics", "AI", "bias"],
                "field": "AI Ethics and Fairness",
                "status": "published",
                "review_requests": [],
                "reviews": []
            },
            {
                "title": "Distributed Systems for Large-Scale Computing",
                "abstract": "This paper presents a novel architecture for distributed computing.",
                "authors": ["Imported_Author"],
                "venue": "Systems Conference 2023",
                "keywords": ["distributed systems", "architecture", "computing"],
                "field": "Computer Systems and Architecture",
                "status": "published",
                "review_requests": [],
                "reviews": []
            },
            {
                "title": "Human-Computer Interaction in VR Environments",
                "abstract": "This paper explores user interaction patterns in virtual reality.",
                "authors": ["Imported_Author"],
                "venue": "HCI Conference 2023",
                "keywords": ["HCI", "virtual reality", "user experience"],
                "field": "Human-Computer Interaction",
                "status": "published",
                "review_requests": [],
                "reviews": []
            },
            {
                "title": "Security Vulnerabilities in IoT Devices",
                "abstract": "This paper analyzes common security issues in IoT deployments.",
                "authors": ["Imported_Author"],
                "venue": "Security Conference 2023",
                "keywords": ["security", "IoT", "vulnerabilities"],
                "field": "Cybersecurity and Privacy",
                "status": "published",
                "review_requests": [],
                "reviews": []
            },
            {
                "title": "Data Science Techniques for Healthcare Analytics",
                "abstract": "This paper presents data science methods for healthcare data.",
                "authors": ["Imported_Author"],
                "venue": "Data Science Conference 2023",
                "keywords": ["data science", "healthcare", "analytics"],
                "field": "Data Science and Analytics",
                "status": "published",
                "review_requests": [],
                "reviews": []
            },
            {
                "title": "Advanced Deep Learning Architectures",
                "abstract": "This paper introduces novel neural network architectures.",
                "authors": ["Imported_Author"],
                "venue": "AI Conference 2023",
                "keywords": ["deep learning", "neural networks", "architectures"],
                "field": "Artificial Intelligence",
                "status": "published",
                "review_requests": [],
                "reviews": []
            }
        ]
        
        # Add papers to database with owner "Imported_PeerRead"
        for paper in test_papers:
            paper["owner_id"] = "Imported_PeerRead"
            self.paper_db.add_paper(paper)
        
        # Save the database
        self.paper_db._save_data()
        print(f"Created {len(test_papers)} test papers")

    def display_detailed_researcher_statistics(self):
        """
        Display detailed statistics for each researcher.
        Shows papers owned, reviews completed, and token activity.
        """
        print("\n" + "="*50)
        print("DETAILED RESEARCHER STATISTICS")
        print("="*50)
        
        # Get all researchers
        researchers = list(self.agents.keys())
        researchers.sort()  # Sort alphabetically
        
        # Collect detailed statistics
        for researcher_id in researchers:
            papers = self.paper_db.get_papers_by_owner(researcher_id)
            completed_reviews = self.token_system.get_reviews_by_reviewer(researcher_id)
            transaction_history = self.token_system.get_researcher_transaction_history(researcher_id)
            
            # Count tokens earned and spent
            tokens_earned = 0
            tokens_spent = 0
            for tx in transaction_history:
                if tx.get('type') == 'review_request' and tx.get('requester_id') == researcher_id:
                    tokens_spent += tx.get('amount', 0)
                elif tx.get('type') == 'review_request' and tx.get('reviewer_id') == researcher_id:
                    tokens_earned += tx.get('amount', 0)
            
            # Count papers by status
            papers_by_status = {
                "draft": 0,
                "submitted": 0,
                "in_review": 0,
                "published": 0
            }
            
            for paper in papers:
                status = paper.get('status', 'published')
                papers_by_status[status] += 1
            
            # Print researcher details
            print(f"\n{researcher_id} - {self.agents[researcher_id].specialty}")
            print("-" * 40)
            print(f"Token Balance: {self.token_system.get_balance(researcher_id)}")
            print(f"Tokens Earned: {tokens_earned}")
            print(f"Tokens Spent: {tokens_spent}")
            print(f"\nPapers Owned: {len(papers)}")
            for status, count in papers_by_status.items():
                if count > 0:
                    print(f"  - {status.capitalize()}: {count}")
            
            print(f"\nReviews Completed: {len(completed_reviews)}")
            
            # Show pending reviews
            pending_reviews = self.agents[researcher_id].get_pending_reviews()
            if pending_reviews:
                print(f"Pending Reviews: {len(pending_reviews)}")
            
            # Show owned papers with titles
            if papers:
                print("\nOwned Papers:")
                for i, paper in enumerate(papers, 1):
                    print(f"  {i}. {paper.get('title', 'Untitled')} (Field: {paper.get('field', 'Unknown')})")
            
            # Show completed reviews
            if completed_reviews:
                print("\nCompleted Reviews for Papers:")
                for i, review in enumerate(completed_reviews, 1):
                    paper_id = review.get('paper_id')
                    paper = self.paper_db.get_paper(paper_id)
                    if paper:
                        print(f"  {i}. {paper.get('title', 'Untitled')} (ID: {paper_id})")
                    else:
                        print(f"  {i}. Unknown Paper (ID: {paper_id})")

def main():
    """Main function to run the peer review simulation."""
    print("🔬 Peer Review Simulation System")
    print("--------------------------------")
    
    # Create simulation with enhanced mode enabled
    simulation = PeerReviewSimulation(enhanced_mode=True)
    
    # Choose simulation mode
    print("\nSimulation Modes:")
    print("1. Interactive Mode (chat with researchers)")
    print("2. Automated Simulation (run rounds of interactions)")
    print("3. Hybrid Mode (setup researchers and then interact)")
    
    while True:
        mode = input("\nEnter simulation mode (1-3): ")
        if mode in ["1", "2", "3"]:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Create researchers
    print("\nCreating researcher agents...")
    simulation.create_all_researchers(assign_papers=True)
    print(f"Created {len(simulation.list_researchers())} researcher agents")
    
    if mode == "1":
        # Interactive Mode
        print("\nEntering Interactive Mode")
        print("You can chat with the researchers and observe the peer review process")
        
        # Create group chat with all researchers
        simulation.create_group_chat()
        
        # Start the conversation
        initial_message = """
        Hello researchers! I'm here to observe your peer review process.
        You each have papers in the system and can request reviews from each other using tokens.
        You can also review papers that have been assigned to you.
        
        Let's start by having each of you introduce yourself and share your current token balance
        and papers you've authored.
        """
        
        simulation.start_chat(initial_message)
    
    elif mode == "2":
        # Automated Simulation
        print("\nEntering Automated Simulation Mode")
        
        num_rounds = int(input("Enter number of simulation rounds: ") or "10")
        interactions_per_round = int(input("Enter number of interactions per round: ") or "5")
        
        simulation.run_simulation_rounds(num_rounds, interactions_per_round)
        
        # Show final statistics
        stats = simulation.get_system_stats()
        
        print("\nFinal System Statistics:")
        print(f"Total Papers: {stats['paper_stats']['total_papers']}")
        print(f"Total Reviews Requested: {stats['token_stats']['total_reviews_requested']}")
        print(f"Total Reviews Completed: {stats['token_stats']['total_reviews_completed']}")
        print(f"Total Tokens Spent: {stats['token_stats']['total_tokens_spent']}")
        
        print("\nFinal Token Leaderboard:")
        for i, researcher in enumerate(stats["leaderboard"]):
            print(f"{i+1}. {researcher['researcher_id']}: {researcher['balance']} tokens")
        
        # Display detailed statistics for each researcher
        simulation.display_detailed_researcher_statistics()
    
    elif mode == "3":
        # Hybrid Mode
        print("\nEntering Hybrid Mode")
        print("First, we'll run some automated interactions")
        
        num_rounds = int(input("Enter number of automated rounds: ") or "5")
        interactions_per_round = int(input("Enter number of interactions per round: ") or "3")
        
        simulation.run_simulation_rounds(num_rounds, interactions_per_round)
        
        # Display detailed statistics for each researcher
        simulation.display_detailed_researcher_statistics()
        
        # Then enter interactive mode
        print("\nNow switching to interactive mode...")
        
        # Create group chat with all researchers
        simulation.create_group_chat()
        
        # Start the conversation
        initial_message = """
        Hello researchers! I've been observing your peer review process.
        
        Now I'd like to interact with you directly. Please introduce yourselves
        and share your experiences with the peer review system so far.
        
        Specifically:
        1. How many tokens do you currently have?
        2. What papers have you authored?
        3. What reviews have you completed?
        4. What reviews are still pending for you to complete?
        """
        
        simulation.start_chat(initial_message)

if __name__ == "__main__":
    main() 