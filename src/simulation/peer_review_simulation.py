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
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add the project root to the path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Update imports to use proper paths
from src.data.paper_database import PaperDatabase
from src.core.token_system import TokenSystem
from src.agents.researcher_agent import ResearcherAgent
from src.agents.editor_agent import EditorAgent
from src.agents.researcher_templates import get_researcher_template, list_researcher_templates

# Load environment variables
load_dotenv("config.env")

def create_ollama_config():
    """
    Create configuration for Ollama model.
    
    Returns:
        Dictionary with LLM configuration
    """
    config_list = [
        {
            "model": os.getenv("OLLAMA_MODEL", "qwen3:30b-a3b"),
            "base_url": os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
            "api_type": "ollama"
        }
    ]

    return {
        "config_list": config_list,
        "temperature": 0.7,
        "timeout": 120,
    }

class PeerReviewSimulation:
    """
    Main simulation system for the peer review process.
    """
    
    def __init__(self, workspace_dir: str = "peer_review_workspace", assign_papers: bool = False):
        """
        Initialize the peer review simulation.
        
        Args:
            workspace_dir: Directory for storing simulation data
            assign_papers: Whether to automatically assign papers after initialization
                          (only effective if researchers are added before this call)
        """
        self.workspace_dir = workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)
        
        # Create paper database
        papers_path = os.path.join(workspace_dir, "papers.json")
        self.paper_db = PaperDatabase(data_path=papers_path)
        
        # Check if papers were loaded, if not, load from test dataset
        if not self.paper_db.get_all_papers():
            print("No papers found in the database.")
            # Try to load from test dataset first
            test_dataset_path = os.path.abspath("test_dataset")
            if os.path.exists(test_dataset_path):
                print(f"Loading papers from test dataset at '{test_dataset_path}'...")
                self.paper_db.load_peerread_dataset(folder_path=test_dataset_path)
                
                # Assign fields to papers based on conference/venue
                field_mappings = {
                    "acl": "NLP", 
                    "cl": "NLP",
                    "conll": "NLP",
                    "iclr": "AI",
                    "nips": "AI",
                    "cs.ai": "AI",
                    "cs.lg": "AI",
                    "cs.cv": "CV",
                    "cs.ro": "Robotics",
                    "cs.se": "Systems",
                    "cs.hc": "HCI",
                    "cs.cr": "Security",
                    "cs.et": "Ethics",
                    "cs.ds": "Data_Science",
                    "cs.cc": "Theory"
                }
                
                # Assign fields to papers
                papers = self.paper_db.get_all_papers()
                for paper in papers:
                    if "venue" in paper:
                        venue_lower = paper["venue"].lower()
                        for key, field in field_mappings.items():
                            if key in venue_lower:
                                paper["field"] = field
                                break
                        else:
                            # Default field if no match
                            paper["field"] = "AI"
                    else:
                        paper["field"] = "AI"  # Default field
                    
                    # Update paper in database
                    self.paper_db.update_paper(paper["id"], {"field": paper["field"]})
                
                # Save changes
                self.paper_db._save_data()
            else:
                # If test dataset doesn't exist, create test papers directly
                self.create_test_papers()
        
        # Double check if we have papers, if not create test papers
        if not self.paper_db.get_all_papers():
            print("Still no papers found. Creating test papers directly...")
            self.create_test_papers()
        
        # Create token system
        tokens_path = os.path.join(workspace_dir, "tokens.json")
        self.token_system = TokenSystem(data_path=tokens_path)
        
        # LLM configuration
        self.llm_config = create_ollama_config()
        
        # Initialize agents dictionary
        self.agents = {}
        
        # Initialize editor agent
        self._create_editor_agent()
        
        # Initialize user proxy
        self._create_user_proxy()
        
        # Group chat configuration
        self.groupchat = None
        self.manager = None
        
        # Specialty compatibility matrix (moved from simulate_random_interactions to make it global)
        self.specialty_compatibility = {
            "Artificial Intelligence": ["Artificial Intelligence", "Natural Language Processing", "Computer Vision", "Data Science and Analytics"],
            "Natural Language Processing": ["Natural Language Processing", "Artificial Intelligence", "Data Science and Analytics"],
            "Computer Vision": ["Computer Vision", "Artificial Intelligence", "Data Science and Analytics"],
            "Robotics and Control Systems": ["Robotics and Control Systems", "Artificial Intelligence", "Computer Systems and Architecture"],
            "Theoretical Computer Science": ["Theoretical Computer Science", "Artificial Intelligence"],
            "AI Ethics and Fairness": ["AI Ethics and Fairness", "Artificial Intelligence", "Human-Computer Interaction"],
            "Computer Systems and Architecture": ["Computer Systems and Architecture", "Robotics and Control Systems"],
            "Human-Computer Interaction": ["Human-Computer Interaction", "AI Ethics and Fairness", "Artificial Intelligence"],
            "Cybersecurity and Privacy": ["Cybersecurity and Privacy", "Computer Systems and Architecture"],
            "Data Science and Analytics": ["Data Science and Analytics", "Artificial Intelligence", "Natural Language Processing", "Computer Vision"]
        }
        
        # Simulation logs
        self.simulation_logs = []
        
        # Assign imported papers to researchers if requested
        # (only effective if researchers are already created)
        if assign_papers and self.agents:
            self.assign_imported_papers_to_agents()
    
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
    
    def _create_editor_agent(self):
        """Create the editor agent for the simulation."""
        editor_message = (
            "You are the editor for an academic journal that evaluates and publishes "
            "research papers. Your job is to screen submissions, invite appropriate reviewers, "
            "and make final decisions on publication based on reviews."
        )
        
        self.editor = EditorAgent(
            name="Journal_Editor",
            journal="Open Science Journal",
            system_message=editor_message,
            paper_db=self.paper_db,
            token_system=self.token_system,
            llm_config=self.llm_config
        )
        
        print(f"Created editor agent: {self.editor.name}")
    
    def add_researcher_from_template(self, template_name: str) -> Optional[ResearcherAgent]:
        """
        Add a researcher agent from a template.
        
        Args:
            template_name: Name of the researcher template
            
        Returns:
            The created ResearcherAgent or None if template not found
        """
        template = get_researcher_template(template_name)
        if not template:
            print(f"Template '{template_name}' not found.")
            return None
        
        # Create researcher agent
        researcher = ResearcherAgent(
            name=template["name"],
            specialty=template["specialty"],
            system_message=template["system_message"],
            paper_db=self.paper_db,
            token_system=self.token_system,
            bias=template.get("bias", ""),
            llm_config=self.llm_config
        )
        
        # Add to agents dictionary
        self.agents[template["name"]] = researcher
        
        print(f"Added researcher agent: {template['name']}")
        return researcher
    
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
        
        return {
            "token_stats": token_stats,
            "paper_stats": paper_stats,
            "researcher_stats": researcher_stats,
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
        
        # Create a dictionary of researchers by specialty for quick lookup
        researchers_by_specialty = {}
        for name, agent in self.agents.items():
            if agent.specialty not in researchers_by_specialty:
                researchers_by_specialty[agent.specialty] = []
            researchers_by_specialty[agent.specialty].append(name)
        
        # Create a dictionary mapping researcher names to data
        researcher_data = {}
        for name, agent in self.agents.items():
            researcher_data[name] = {
                "specialty": agent.specialty,
                "bias": agent.bias
            }
        
        # Log researcher specialties
        print("\nResearcher specialties:")
        for specialty, researchers in researchers_by_specialty.items():
            print(f"  {specialty}: {', '.join(researchers)}")
        
        print("\nStarting interactions...")
        for i in range(num_interactions):
            # Select interaction type with better logic to ensure workflow progression
            # Priority: submit papers first, then process them, then handle invitations, then reviews
            
            # Check current state to determine best interaction
            queue_size = len(self.editor.submission_queue)
            papers_in_review = len(self.editor.papers_in_review)
            
            # Count pending invitations across all researchers
            total_pending_invitations = 0
            for researcher in self.agents.values():
                pending_invitations = researcher.get_review_invitations()
                for invitation in pending_invitations:
                    if invitation.get('status') == 'invited':
                        total_pending_invitations += 1
            
            # Also check directly in papers for invitations (backup method)
            if total_pending_invitations == 0:
                for paper in self.paper_db.get_all_papers():
                    for invitation in paper.get('review_invitations', []):
                        if invitation.get('status') == 'invited':
                            total_pending_invitations += 1
            
            # Count pending reviews (accepted invitations)
            total_pending_reviews = 0
            for paper in self.paper_db.get_all_papers():
                for invitation in paper.get('review_invitations', []):
                    if invitation.get('status') == 'accepted':
                        total_pending_reviews += 1
                for request in paper.get('review_requests', []):
                    if request.get('status') == 'accepted':
                        total_pending_reviews += 1
            
            # Determine interaction type based on current state and ensure workflow progression
            if i < 3:  # First few interactions should submit papers
                interaction_type = "submit_paper"
            elif queue_size > 0:  # Process papers in queue first
                interaction_type = "process_queue"
            elif total_pending_invitations > 0:  # Then process invitations BEFORE generating reviews
                interaction_type = "process_invitations"
            elif total_pending_reviews > 0:  # Only then generate reviews
                interaction_type = "generate_review"
            else:  # Default to submitting more papers or processing queue
                interaction_type = random.choice(["submit_paper", "process_queue"])
            
            print(f"\nInteraction {i+1}: {interaction_type} (Queue: {queue_size}, Invitations: {total_pending_invitations}, Reviews: {total_pending_reviews})")
            
            # DEBUG: Print detailed invitation status
            if total_pending_invitations > 0:
                print(f"  DEBUG: Found {total_pending_invitations} pending invitations")
                for researcher_name, researcher in self.agents.items():
                    invitations = researcher.get_review_invitations()
                    if invitations:
                        print(f"    {researcher_name}: {len(invitations)} invitations")
                        for inv in invitations:
                            print(f"      - Paper {inv['paper_id']}: {inv['status']}")
            
            # DEBUG: Print detailed review status  
            if total_pending_reviews > 0:
                print(f"  DEBUG: Found {total_pending_reviews} pending reviews")
                for paper in self.paper_db.get_all_papers():
                    for invitation in paper.get('review_invitations', []):
                        if invitation.get('status') == 'accepted':
                            print(f"    Paper {paper['id']}: {invitation['reviewer_id']} accepted")
                    for request in paper.get('review_requests', []):
                        if request.get('status') == 'accepted':
                            print(f"    Paper {paper['id']}: {request['reviewer_id']} accepted request")
            
            if interaction_type == "submit_paper":
                # Random author
                author_name = random.choice(researcher_names)
                author = self.agents[author_name]
                
                # Create a more realistic paper with better abstract
                paper_title = f"Research on {author.specialty} - {i}"
                
                # Generate more detailed abstracts based on specialty
                abstract_templates = {
                    "Artificial Intelligence": [
                        "This paper presents a novel deep learning architecture for improving classification accuracy on complex datasets. We introduce a new attention mechanism that reduces computational overhead by 30% while maintaining state-of-the-art performance. Experimental results on benchmark datasets demonstrate significant improvements over existing methods.",
                        "We propose a reinforcement learning framework for multi-agent coordination in dynamic environments. Our approach combines hierarchical planning with distributed decision-making to achieve robust performance. Evaluation on simulation environments shows 25% improvement in task completion rates.",
                        "This work introduces a new neural network architecture that integrates symbolic reasoning with connectionist learning. We demonstrate its effectiveness on logical reasoning tasks and show how it bridges the gap between neural and symbolic AI approaches."
                    ],
                    "Natural Language Processing": [
                        "This paper introduces a transformer-based model for cross-lingual sentiment analysis that achieves state-of-the-art performance across 15 languages. Our approach uses multilingual pre-training with domain adaptation techniques. Results show 12% improvement over previous methods on benchmark datasets.",
                        "We present a novel approach to neural machine translation that incorporates syntactic structure information. Our model uses graph neural networks to encode dependency trees and achieves significant improvements in translation quality, particularly for low-resource language pairs.",
                        "This work proposes a new method for automatic text summarization using hierarchical attention mechanisms. We evaluate our approach on multiple datasets and demonstrate superior performance in both extractive and abstractive summarization tasks."
                    ],
                    "Computer Vision": [
                        "This paper presents a new convolutional neural network architecture for real-time object detection in autonomous vehicles. Our method achieves 95% accuracy while maintaining 60 FPS processing speed. We evaluate on challenging driving scenarios and demonstrate robust performance in various weather conditions.",
                        "We introduce a novel approach to image segmentation using self-supervised learning. Our method requires no labeled data and achieves competitive performance with fully supervised methods. Experiments on medical imaging datasets show promising results for clinical applications.",
                        "This work proposes a new technique for 3D object reconstruction from single images using generative adversarial networks. We demonstrate high-quality reconstructions and evaluate on standard benchmarks with significant improvements over existing methods."
                    ],
                    "Robotics and Control Systems": [
                        "This paper presents a new control algorithm for robotic manipulation in unstructured environments. Our approach combines model predictive control with machine learning to adapt to dynamic conditions. Experimental validation on a 7-DOF robotic arm shows improved task success rates.",
                        "We introduce a novel path planning algorithm for autonomous mobile robots in crowded environments. Our method uses social force models combined with deep reinforcement learning to navigate safely among humans. Real-world experiments demonstrate effective collision avoidance.",
                        "This work proposes a new framework for multi-robot coordination in search and rescue operations. We develop distributed algorithms that enable robots to collaborate effectively while maintaining communication constraints."
                    ],
                    "Theoretical Computer Science": [
                        "This paper establishes new complexity bounds for approximation algorithms in the traveling salesman problem. We prove that our proposed algorithm achieves a 1.5-approximation ratio, improving upon previous results. The analysis uses novel techniques from linear programming relaxations.",
                        "We present a new algorithmic framework for solving maximum flow problems in dynamic graphs. Our approach achieves O(n log n) time complexity per update operation. Theoretical analysis and experimental evaluation demonstrate significant improvements over existing methods.",
                        "This work introduces new cryptographic protocols for secure multi-party computation. We prove security under standard assumptions and demonstrate practical efficiency improvements over existing protocols."
                    ],
                    "AI Ethics and Fairness": [
                        "This paper analyzes bias in facial recognition systems across different demographic groups. We propose new fairness metrics and demonstrate how algorithmic bias can be reduced through careful dataset curation and model design. Our findings have important implications for deployment in law enforcement.",
                        "We present a comprehensive study of algorithmic fairness in hiring systems. Our analysis reveals systematic biases against certain groups and proposes mitigation strategies. We evaluate our approaches on real-world hiring datasets with promising results.",
                        "This work examines the ethical implications of AI decision-making in healthcare. We develop a framework for ensuring transparency and accountability in medical AI systems."
                    ],
                    "Computer Systems and Architecture": [
                        "This paper presents a new memory management system for high-performance computing applications. Our approach reduces memory access latency by 40% through intelligent prefetching and caching strategies. Evaluation on scientific computing workloads shows significant performance improvements.",
                        "We introduce a novel distributed computing framework for processing large-scale graph data. Our system achieves linear scalability and fault tolerance through innovative partitioning and replication strategies. Experiments on real-world datasets demonstrate superior performance.",
                        "This work proposes new techniques for optimizing energy consumption in data centers. We develop algorithms that balance performance and power efficiency, achieving 25% energy savings without compromising service quality."
                    ],
                    "Human-Computer Interaction": [
                        "This paper investigates user experience in virtual reality environments for educational applications. We conduct user studies with 200 participants and identify key design principles for effective VR learning interfaces. Our findings inform the development of next-generation educational technologies.",
                        "We present a new interaction paradigm for mobile devices using gesture recognition. Our approach combines computer vision with machine learning to enable natural hand gestures for device control. User studies demonstrate improved usability and user satisfaction.",
                        "This work examines accessibility challenges in modern web applications. We propose design guidelines and automated testing tools to improve accessibility for users with disabilities."
                    ],
                    "Cybersecurity and Privacy": [
                        "This paper presents a new intrusion detection system using deep learning techniques. Our approach achieves 99.2% accuracy in detecting network attacks while maintaining low false positive rates. Evaluation on real network traffic demonstrates effectiveness against zero-day attacks.",
                        "We introduce novel privacy-preserving techniques for machine learning on sensitive data. Our methods use differential privacy and secure multi-party computation to protect individual privacy while enabling useful analytics. Theoretical analysis proves strong privacy guarantees.",
                        "This work proposes new cryptographic protocols for secure communication in IoT networks. We address the unique constraints of IoT devices while maintaining strong security properties."
                    ],
                    "Data Science and Analytics": [
                        "This paper presents a new framework for real-time anomaly detection in streaming data. Our approach combines statistical methods with machine learning to identify unusual patterns with high accuracy. Evaluation on financial and network data shows superior performance over existing methods.",
                        "We introduce novel techniques for handling missing data in large-scale datasets. Our methods use advanced imputation strategies that preserve statistical properties while improving analysis accuracy. Experiments on real-world datasets demonstrate significant improvements.",
                        "This work proposes new visualization techniques for high-dimensional data analysis. We develop interactive tools that help analysts discover patterns and insights in complex datasets."
                    ]
                }
                
                # Select a random abstract template for the author's specialty
                specialty_abstracts = abstract_templates.get(author.specialty, [
                    f"This paper presents novel research in {author.specialty} with significant theoretical and practical contributions. We propose new methods that advance the state-of-the-art and demonstrate their effectiveness through comprehensive evaluation."
                ])
                
                paper_abstract = random.choice(specialty_abstracts)
                
                paper_data = {
                    "title": paper_title,
                    "abstract": paper_abstract,
                    "authors": [author_name],
                    "field": author.specialty,
                    "status": "draft"  # Start as draft, will be updated to submitted
                }
                
                # Publish the paper
                paper_id = author.publish_paper(paper_data)
                
                # Determine author's reputation stake (priority signal)
                # Higher-reputation researchers may stake more
                author_reputation = self.token_system.get_balance(author_name)
                
                # Authors stake between 5% and 20% of their reputation as priority signal
                # With a minimum of 5 tokens and maximum of 50 tokens
                min_stake = max(5, int(author_reputation * 0.05))  # At least 5 tokens
                max_stake = min(50, int(author_reputation * 0.20))  # At most 50 tokens or 20% of reputation
                
                # Ensure max_stake is at least min_stake
                if max_stake < min_stake:
                    max_stake = min_stake
                
                # Add some randomness to simulate different author behaviors
                priority_score = random.randint(min_stake, max_stake)
                
                print(f"\nAuthor {author_name} submitting paper ID {paper_id}: {paper_title}")
                print(f"  - Author reputation: {author_reputation}")
                print(f"  - Priority signal: {priority_score} tokens")
                
                # Submit paper to editor's queue with priority score
                self.editor.submit_paper(paper_id, priority_score)
                
                # NOTE: Tokens are NOT spent here anymore. They will only be spent when:
                # 1. Paper passes editor screening
                # 2. Reviewers are invited and accept
                # 3. Actual review process begins
                # This prevents token drain from desk-rejected papers
                
                outcome = {
                    "interaction": "submit_paper",
                    "author": author_name,
                    "paper_id": paper_id,
                    "paper_title": paper_title,
                    "paper_field": author.specialty,
                    "priority_score": priority_score,
                    "status": "submitted to queue"
                }
            
            elif interaction_type == "process_queue":
                # Process the next paper in the editor's queue
                success, message, paper_data = self.editor.process_next_submission()
                
                if not success:
                    print("\nNo papers in queue to process")
                    outcome = {
                        "interaction": "process_queue",
                        "success": False,
                        "message": "No papers in queue"
                    }
                    outcomes.append(outcome)
                    continue
                
                print(f"\nEditor processing paper: {paper_data['title']}")
                print(f"  - Priority score: {paper_data['priority_score']}")
                print(f"  - Editor's Thought Process: {paper_data['thought_process']}")
                print(f"  - Decision: {paper_data['decision']}")
                print(f"  - Reasoning: {paper_data['message']}")
                
                paper_id = paper_data["paper_id"]
                
                if paper_data["decision"] == "accept_for_review":
                    # Paper passed screening, now invite reviewers
                    invitations = self.editor.invite_reviewers(
                        paper_id=paper_id,
                        potential_reviewers=researcher_names,
                        specialty_compatibility=self.specialty_compatibility,
                        reviewer_data=researcher_data,
                        num_reviewers=3
                    )
                    
                    # Count invitations
                    num_invited = sum(1 for inv in invitations if inv.get("invited", False))
                    print(f"  Editor invited {num_invited} reviewers for paper {paper_id}")
                    
                    # Log invited reviewers
                    invited_list = [inv["reviewer_id"] for inv in invitations if inv.get("invited", False)]
                    print(f"  Invited reviewers: {', '.join(invited_list)}")
                    
                    outcome = {
                        "interaction": "process_queue",
                        "paper_id": paper_id,
                        "paper_title": paper_data["title"],
                        "author": paper_data["author"],
                        "priority_score": paper_data["priority_score"],
                        "screening": "passed",
                        "editor_thought_process": paper_data["thought_process"],
                        "invitations_sent": num_invited,
                        "invited_reviewers": invited_list
                    }
                else:
                    # Paper was desk-rejected
                    # Ensure no tokens were spent on this paper since it won't be reviewed
                    paper_id = paper_data["paper_id"]
                    author_name = paper_data["author"]
                    priority_score = paper_data["priority_score"]
                    
                    print(f"  Paper {paper_id} was desk-rejected - no tokens spent on priority signal")
                    
                    outcome = {
                        "interaction": "process_queue",
                        "paper_id": paper_id,
                        "paper_title": paper_data["title"],
                        "author": paper_data["author"],
                        "priority_score": paper_data["priority_score"],
                        "screening": "rejected",
                        "editor_thought_process": paper_data["thought_process"],
                        "reason": paper_data["message"]
                    }
            
            elif interaction_type == "process_invitations":
                # Process pending review invitations
                invitations_processed = 0
                accepted_invitations = 0
                processed_invitations = []  # Track which invitations we've processed
                
                # For each researcher, check if they have pending invitations
                for researcher_name in researcher_names:
                    researcher = self.agents[researcher_name]
                    pending_invitations = researcher.get_review_invitations()
                    
                    # Process each invitation
                    for invitation in pending_invitations:
                        paper_id = invitation["paper_id"]
                        
                        # Skip if we've already processed this invitation
                        invitation_key = f"{researcher_name}_{paper_id}"
                        if invitation_key in processed_invitations:
                            continue
                        
                        paper = self.paper_db.get_paper(paper_id)
                        if not paper:
                            continue
                        
                        # Get the paper's priority score to influence reviewer decisions
                        priority_score = paper.get("priority_score", 0)
                        
                        print(f"  [DEBUG] Paper {paper_id} priority_score from database: {priority_score}")
                        
                        # Researcher decides whether to accept invitation
                        accepted, reason, thought_process = researcher.respond_to_invitation(paper_id, priority_score)
                        
                        print(f"\nResearcher {researcher_name} (Personality: {researcher.personality}) responding to review invitation for paper {paper_id}:")
                        print(f"  Thought Process: {thought_process}")
                        print(f"  Decision: {'ACCEPTED' if accepted else 'DECLINED'}")
                        print(f"  Reasoning: {reason}")
                        
                        # Process acceptance/rejection with editor
                        requester_id = paper.get("owner_id", "Unknown")
                        success, message = self.editor.process_review_acceptance(
                            paper_id=paper_id,
                            reviewer_id=researcher_name,
                            accepted=accepted,
                            token_amount=priority_score,  # Using priority score for token amount
                            requester_id=requester_id
                        )
                        
                        print(f"  Editor processed response: {message}")
                        
                        # Mark this invitation as processed
                        processed_invitations.append(invitation_key)
                        invitations_processed += 1
                        if accepted:
                            accepted_invitations += 1
                
                if invitations_processed == 0:
                    outcome = {
                        "interaction": "process_invitations",
                        "success": False,
                        "message": "No pending invitations found"
                    }
                else:
                    outcome = {
                        "interaction": "process_invitations",
                        "invitations_processed": invitations_processed,
                        "invitations_accepted": accepted_invitations,
                        "success": True,
                        "message": f"Processed {invitations_processed} invitations, {accepted_invitations} accepted",
                        "thought_processes": [] # To store all thought processes from this batch
                    }
                    
                    # We need to collect thought processes from each invitation response
                    for researcher_name in researcher_names:
                        researcher = self.agents[researcher_name]
                        pending_invitations = researcher.get_review_invitations()
                        
                        for invitation in pending_invitations:
                            paper_id = invitation["paper_id"]
                            paper = self.paper_db.get_paper(paper_id)
                            
                            if paper:
                                # Add details to thought_processes list in the outcome
                                outcome["thought_processes"].append({
                                    "reviewer": researcher_name,
                                    "reviewer_personality": researcher.personality,
                                    "paper_id": paper_id,
                                    "paper_title": paper.get("title", "Unknown Paper"),
                                    "author": paper.get("owner_id", "Unknown Author"),
                                    # Note: We don't have the thought process here because we've already 
                                    # called respond_to_invitation and didn't save the results.
                                    # The thought process is already printed to console during the interaction.
                                    "thought_summary": f"Review invitation response from {researcher_name} for paper {paper_id}"
                                })
            
            elif interaction_type == "generate_review":
                # Find accepted review requests among all researchers
                pending_reviews = []
                
                for agent_name, agent in self.agents.items():
                    # Get all papers
                    papers = self.paper_db.get_all_papers()
                    
                    for paper in papers:
                        # Check review requests
                        for request in paper.get('review_requests', []):
                            if request['reviewer_id'] == agent_name and request['status'] == 'accepted':
                                # This researcher has accepted a review request for this paper
                                review_data = {
                                    "reviewer_id": agent_name,
                                    "paper_id": paper["id"],
                                    "paper_title": paper.get("title", "Untitled"),
                                    "paper_field": paper.get("field", "Unknown"),
                                    "request": request
                                }
                                pending_reviews.append((agent_name, review_data))
                        
                        # Also check review invitations
                        for invitation in paper.get('review_invitations', []):
                            if invitation['reviewer_id'] == agent_name and invitation['status'] == 'accepted':
                                # This researcher has accepted a review invitation for this paper
                                review_data = {
                                    "reviewer_id": agent_name,
                                    "paper_id": paper["id"],
                                    "paper_title": paper.get("title", "Untitled"),
                                    "paper_field": paper.get("field", "Unknown"),
                                    "invitation": invitation
                                }
                                # Check if already added from review_requests
                                if not any(data[1]["paper_id"] == paper["id"] and 
                                          data[1]["reviewer_id"] == agent_name 
                                          for data in pending_reviews):
                                    pending_reviews.append((agent_name, review_data))
                
                print(f"\nFound {len(pending_reviews)} pending reviews")
                
                if not pending_reviews:
                    outcome = {
                        "interaction": "generate_review",
                        "success": False,
                        "message": "No pending reviews found"
                    }
                else:
                    # Select random pending review
                    reviewer_name, review_data = random.choice(pending_reviews)
                    reviewer = self.agents[reviewer_name]
                    paper_id = review_data["paper_id"]
                    
                    print(f"\nReviewer {reviewer_name} (Personality: {reviewer.personality}) is generating a review for paper ID {paper_id} by {review_data.get('author', 'Unknown Author')}...")
                    
                    # Generate review (which now includes thought process)
                    generated_data = reviewer.generate_review(paper_id)
                    review_content = generated_data.get("review_content", {"error": "No review content generated"})
                    thought_process = generated_data.get("thought_process", "No thought process recorded.")
                    
                    print(f"  Reviewer's Thought Process: {thought_process}")
                    # print(f"  Generated Review Content (JSON): {json.dumps(review_content, indent=2)}") # Optional: for detailed debugging

                    # Submit the actual review content
                    success, message = reviewer.submit_review(
                        paper_id=paper_id,
                        review_content=review_content # Pass the parsed JSON review content
                    )
                    
                    print(f"  Review submission result: {success}, {message}")

                    if success:
                        # Award tokens to the reviewer using TokenSystem
                        # and record it in the reviewer's memory
                        completed_successfully, reputation_gained = self.token_system.complete_review(reviewer.name, paper_id)
                        if completed_successfully:
                            print(f"  TokenSystem: Awarded {reputation_gained} tokens to {reviewer.name} for reviewing {paper_id}.")
                            reviewer.record_token_award(
                                amount=reputation_gained, 
                                reason="review_completion_reward", 
                                related_to=f"review_for_{paper_id}", 
                                awarded_by="System/Editor"
                            )
                            
                            # Notify the author about the review (and its outcome if available from review_content)
                            paper_details = self.paper_db.get_paper(paper_id)
                            if paper_details and paper_details.get('owner_id'):
                                author_agent = self.agents.get(paper_details['owner_id'])
                                if author_agent:
                                    # The review_content is what the LLM generated.
                                    # It should ideally contain structured fields like 'decision', 'score', 'comments'
                                    # For now, we pass the whole content.
                                    author_agent.record_review_outcome(
                                        paper_id=paper_id, 
                                        reviewer_id=reviewer.name, 
                                        review_details=review_content # Pass the actual review content to the author
                                    )
                                    print(f"  Notified author {author_agent.name} about review for paper {paper_id} by {reviewer.name}.")
                        else:
                            print(f"  TokenSystem: Review completed but no matching request found for {reviewer.name}, paper {paper_id}. Awarded base reputation.")
                    
                    outcome = {
                        "interaction": "generate_review",
                        "reviewer": reviewer_name,
                        "reviewer_personality": reviewer.personality, # Added for logging
                        "thought_process": thought_process, # Added for logging
                        "reviewer_specialty": reviewer.specialty,
                        "paper_id": paper_id,
                        "paper_title": review_data.get("paper_title", "Unknown"),
                        "paper_field": review_data.get("paper_field", "Unknown"),
                        "success": success,
                        "message": message
                    }
            
            outcomes.append(outcome)
            
            # Save interaction to JSON file for analysis
            self._save_interaction(outcome)
            
            # Add to simulation logs
            self.simulation_logs.append({
                "timestamp": time.time(),
                "interaction_type": interaction_type,
                "outcome": outcome
            })
            
            # Pause between interactions
            time.sleep(0.1)  # Small delay to make output more readable
        
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
            print(f"\n{'-'*80}")
            print(f"ROUND {round_num}/{num_rounds}:")
            print(f"{'-'*80}")
            
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
            
            # Print detailed interaction results with thought processes
            print("\nInteraction Results:")
            for i, interaction in enumerate(interaction_results, 1):
                interaction_type = interaction.get("interaction", "unknown")
                print(f"\n  {i}. {interaction_type.upper()}")
                
                if interaction_type == "generate_review":
                    reviewer = interaction.get("reviewer", "Unknown")
                    personality = interaction.get("reviewer_personality", "Unknown")
                    paper_title = interaction.get("paper_title", "Unknown Paper")
                    thought_process = interaction.get("thought_process", "No thought process recorded")
                    
                    print(f"    Reviewer: {reviewer} (Personality: {personality})")
                    print(f"    Paper: {paper_title}")
                    print(f"    Thought Process: {thought_process[:300]}..." if len(thought_process) > 300 else f"    Thought Process: {thought_process}")
                
                elif interaction_type == "process_invitations":
                    if interaction.get("success", False):
                        print(f"    Processed {interaction.get('invitations_processed', 0)} invitations")
                        print(f"    Accepted {interaction.get('invitations_accepted', 0)} invitations")
                        
                        # Display individual invitation thought processes if available
                        thought_processes = interaction.get("thought_processes", [])
                        if thought_processes:
                            print(f"\n    Reviewer decisions:")
                            for idx, tp in enumerate(thought_processes, 1):
                                reviewer = tp.get("reviewer", "Unknown")
                                personality = tp.get("reviewer_personality", "Unknown")
                                paper_title = tp.get("paper_title", "Unknown Paper")
                                thought_summary = tp.get("thought_summary", "No thought process recorded")
                                
                                print(f"    {idx}. {reviewer} (Personality: {personality}) - Paper: {paper_title}")
                                # Note: actual thought process text was printed during the simulation in real-time
                                # but we can't access it here because it wasn't stored in the outcome
                    else:
                        print(f"    {interaction.get('message', 'No pending invitations')}")
                
                elif interaction_type == "process_queue":
                    if interaction.get("success", False) != False:
                        paper_title = interaction.get("paper_title", "Unknown Paper")
                        decision = interaction.get("screening", "Unknown decision")
                        editor_thought = interaction.get("editor_thought_process", "No thought process recorded")
                        
                        print(f"    Paper: {paper_title}")
                        print(f"    Decision: {decision.upper()}")
                        print(f"    Editor's Thought Process: {editor_thought[:300]}..." if len(editor_thought) > 300 else f"    Editor's Thought Process: {editor_thought}")
                        
                        if decision == "passed":
                            print(f"    Invited {interaction.get('invitations_sent', 0)} reviewers")
                    else:
                        print(f"    {interaction.get('message', 'No papers in queue')}")
                
                elif interaction_type == "submit_paper":
                    author = interaction.get("author", "Unknown")
                    paper_title = interaction.get("paper_title", "Unknown Paper")
                    priority = interaction.get("priority_score", 0)
                    
                    print(f"    Author: {author}")
                    print(f"    Paper: {paper_title}")
                    print(f"    Priority Score: {priority}")
            
            # Print summary statistics
            print(f"\n{'-'*50}")
            print("Round Summary:")
            print(f"{'-'*50}")
            
            papers_submitted = sum(1 for i in interaction_results if i["interaction"] == "submit_paper")
            papers_accepted = sum(1 for i in interaction_results if i["interaction"] == "process_queue" and i.get("screening") == "passed")
            invitations_processed = sum(i.get("invitations_processed", 0) for i in interaction_results if i["interaction"] == "process_invitations")
            invitations_accepted = sum(i.get("invitations_accepted", 0) for i in interaction_results if i["interaction"] == "process_invitations")
            reviews_completed = sum(1 for i in interaction_results if i["interaction"] == "generate_review" and i["success"])
            
            print(f"  - Papers submitted: {papers_submitted}")
            print(f"  - Papers passing editor screening: {papers_accepted}/{papers_submitted if papers_submitted > 0 else 1}")
            print(f"  - Review invitations processed: {invitations_processed}")
            print(f"  - Review invitations accepted: {invitations_accepted}/{invitations_processed if invitations_processed > 0 else 1}")
            print(f"  - Reviews completed: {reviews_completed}")
            
            # Print token balances
            print("\n  Current Token Balances:")
            for researcher in stats["leaderboard"][:5]:  # Show top 5
                print(f"  - {researcher['researcher_id']}: {researcher['balance']} tokens")
            
            if len(stats["leaderboard"]) > 5:
                print(f"    ... and {len(stats['leaderboard']) - 5} more")
        
        # Save full simulation results
        simulation_results = {
            "rounds": round_results,
            "final_stats": self.get_system_stats(),
            "logs": self.simulation_logs
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
        
        # Group agents by specialty
        agents_by_specialty = {}
        for name, agent in self.agents.items():
            if agent.specialty not in agents_by_specialty:
                agents_by_specialty[agent.specialty] = []
            agents_by_specialty[agent.specialty].append(name)
        
        reassigned_count = 0
        
        # Get all papers
        papers = self.paper_db.get_all_papers()
        
        # Loop through papers and reassign those with "Imported_PeerRead" owner
        for paper in papers:
            if paper["owner_id"] == "Imported_PeerRead":
                # Get paper field, defaulting to "AI" if not set
                paper_field = paper.get("field", "AI")
                
                # Find agents with compatible specialties
                compatible_specialties = self.specialty_compatibility.get(paper_field, [paper_field])
                matching_agents = []
                
                for specialty in compatible_specialties:
                    if specialty in agents_by_specialty:
                        matching_agents.extend(agents_by_specialty[specialty])
                
                # If no matching agents, fall back to any agent
                if not matching_agents:
                    new_owner = random.choice(list(self.agents.keys()))
                    paper_field = self.agents[new_owner].specialty  # Update paper field to match new owner
                else:
                    new_owner = random.choice(matching_agents)
                
                # Update the paper's owner and field
                paper["owner_id"] = new_owner
                paper["field"] = paper_field
                self.paper_db.update_paper(paper["id"], {"owner_id": new_owner, "field": paper_field})
                reassigned_count += 1
                
                print(f"Assigned paper '{paper.get('title', 'Untitled')}' (field: {paper_field}) to {new_owner}")
        
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
        Shows papers owned, reviews completed, and reputation metrics.
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
            
            # Count reputation gained and staked
            reputation_gained = 0
            reputation_staked = 0
            for tx in transaction_history:
                if tx.get('type') == 'review_completion' and tx.get('reviewer_id') == researcher_id:
                    if 'reputation_gained' in tx:
                        reputation_gained += tx.get('reputation_gained', 0)
                elif tx.get('type') == 'review_request' and tx.get('requester_id') == researcher_id:
                    if 'priority_score' in tx:
                        reputation_staked += tx.get('priority_score', 0)
                    elif 'amount' in tx:  # For backwards compatibility
                        reputation_staked += tx.get('amount', 0)
            
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
            print(f"Reputation Score: {self.token_system.get_balance(researcher_id)}")
            print(f"Reputation Gained: {reputation_gained}")
            print(f"Reputation Staked: {reputation_staked}")
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
                    title = paper.get('title', 'Untitled')
                    field = paper.get('field', 'Unknown')
                    priority = paper.get('priority_score', 0)
                    print(f"  {i}. {title} (Field: {field}, Priority: {priority})")
            
            # Show completed reviews
            if completed_reviews:
                print("\nCompleted Reviews for Papers:")
                for i, review in enumerate(completed_reviews, 1):
                    paper_id = review.get('paper_id')
                    reputation_gain = review.get('reputation_gained', 'Unknown')
                    paper = self.paper_db.get_paper(paper_id)
                    if paper:
                        print(f"  {i}. {paper.get('title', 'Untitled')} (ID: {paper_id}, Reputation Gained: {reputation_gain})")
                    else:
                        print(f"  {i}. Unknown Paper (ID: {paper_id}, Reputation Gained: {reputation_gain})")

    def clear_generic_papers(self):
        """
        Clear papers with generic abstracts that might interfere with simulation.
        This ensures we start with fresh, realistic papers.
        """
        papers = self.paper_db.get_all_papers()
        papers_to_remove = []
        
        for paper in papers:
            abstract = paper.get('abstract', '')
            title = paper.get('title', '')
            
            # Identify generic papers by checking for minimal abstracts
            if (len(abstract) < 100 or  # Very short abstracts
                'this paper presents research in the field' in abstract.lower() or
                'research on' in title and len(abstract) < 150):
                papers_to_remove.append(paper['id'])
        
        # Remove generic papers
        for paper_id in papers_to_remove:
            self.paper_db.delete_paper(paper_id)
        
        if papers_to_remove:
            print(f"Cleared {len(papers_to_remove)} generic papers from database")
            self.paper_db._save_data()
        
        # Clear the editor's submission queue
        self.editor.submission_queue = []
        
        # Reset papers in review tracking
        self.editor.papers_in_review = {}

def main():
    """Main entry point for the peer review simulation."""
    print("Welcome to the Peer Review Simulation System!")
    
    # Create the simulation
    simulation = PeerReviewSimulation()
    
    # Create researcher agents from templates
    simulation.create_all_researchers(assign_papers=True)
    
    # Clear any existing generic papers to ensure fresh simulation
    simulation.clear_generic_papers()
    
    while True:
        print("\nMain Menu:")
        print("1. Run automated simulation rounds")
        print("2. Display researcher statistics")
        print("3. View papers in review process")
        print("4. View papers needing reviewers")
        print("5. View submission queue")
        print("6. Process next submission")
        print("7. Start interactive group chat")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ")
        
        if choice == "1":
            # Run automated simulation
            num_rounds = int(input("Enter number of rounds: "))
            interactions_per_round = int(input("Enter interactions per round: "))
            simulation.run_simulation_rounds(num_rounds, interactions_per_round)
            
        elif choice == "2":
            # Display researcher statistics
            simulation.display_detailed_researcher_statistics()
            
        elif choice == "3":
            # View papers in review process
            papers_in_review = simulation.editor.get_papers_in_review()
            print(f"\nPapers in Review Process ({len(papers_in_review)}):")
            
            for i, paper in enumerate(papers_in_review):
                title = paper.get("title", "Untitled")
                field = paper.get("field", "Unknown")
                owner = paper.get("owner_id", "Unknown")
                reviewers = paper.get("review_status", {}).get("reviewers", [])
                num_reviewers = len(reviewers)
                
                print(f"{i+1}. '{title}' (ID: {paper['id']})")
                print(f"   Field: {field}, Author: {owner}")
                print(f"   Reviewers: {', '.join(reviewers) if reviewers else 'None'} ({num_reviewers})")
                print(f"   Status: {paper.get('status', 'Unknown')}")
                print()
                
        elif choice == "4":
            # View papers needing reviewers
            papers_needing_reviewers = simulation.editor.get_papers_needing_reviewers()
            print(f"\nPapers Needing More Reviewers ({len(papers_needing_reviewers)}):")
            
            for i, paper in enumerate(papers_needing_reviewers):
                title = paper.get("title", "Untitled")
                field = paper.get("field", "Unknown")
                owner = paper.get("owner_id", "Unknown")
                reviewers = paper.get("review_status", {}).get("reviewers", [])
                current_count = len(reviewers)
                needed_count = 3 - current_count
                
                print(f"{i+1}. '{title}' (ID: {paper['id']})")
                print(f"   Field: {field}, Author: {owner}")
                print(f"   Current reviewers: {', '.join(reviewers) if reviewers else 'None'} ({current_count})")
                print(f"   Needs {needed_count} more reviewer(s)")
                print()
        
        elif choice == "5":
            # View submission queue
            queue_status = simulation.editor.get_submission_queue_status()
            print(f"\nSubmission Queue ({len(queue_status)}):")
            
            # Sort by priority (highest first), then submission time (earliest first)
            queue_status.sort(key=lambda x: (-x["priority_score"], x["submission_time"]))
            
            for i, paper in enumerate(queue_status):
                title = paper.get("title", "Untitled")
                author = paper.get("author", "Unknown")
                priority = paper.get("priority_score", 0)
                
                print(f"{i+1}. '{title}' (ID: {paper['paper_id']})")
                print(f"   Author: {author}")
                print(f"   Priority Score: {priority}")
                print()
                
        elif choice == "6":
            # Process next submission
            success, message, paper_data = simulation.editor.process_next_submission()
            
            if not success:
                print("\nNo papers in submission queue.")
            else:
                print(f"\nProcessed paper: {paper_data['title']}")
                print(f"Decision: {paper_data['decision']}")
                print(f"Message: {paper_data['message']}")
                
                if paper_data["decision"] == "accept_for_review":
                    paper_id = paper_data["paper_id"]
                    researcher_names = simulation.list_researchers()
                    researcher_data = {name: {"specialty": simulation.agents[name].specialty} 
                                      for name in researcher_names}
                    
                    # Invite reviewers
                    invitations = simulation.editor.invite_reviewers(
                        paper_id=paper_id,
                        potential_reviewers=researcher_names,
                        specialty_compatibility=simulation.specialty_compatibility,
                        reviewer_data=researcher_data,
                        num_reviewers=3
                    )
                    
                    # Count invitations
                    num_invited = sum(1 for inv in invitations if inv.get("invited", False))
                    print(f"Invited {num_invited} reviewers for this paper")
                
        elif choice == "7":
            # Start interactive chat
            simulation.create_group_chat()
            
            initial_message = (
                "Welcome to the peer review simulation. You can interact with researcher agents "
                "to discuss papers, request reviews, or ask about their research interests. "
                "What would you like to discuss today?"
            )
            
            simulation.start_chat(initial_message)
            
        elif choice == "8":
            # Exit
            print("Thank you for using the Peer Review Simulation System. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 