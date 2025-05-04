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
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add the project root to the path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Update imports to use proper paths
from src.data.paper_database import PaperDatabase
from src.core.token_system import TokenSystem
from src.agents.researcher_agent import ResearcherAgent
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
            "model": os.getenv("OLLAMA_MODEL", "qwen3:4b"),
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
        
        # Initialize user proxy
        self._create_user_proxy()
        
        # Group chat configuration
        self.groupchat = None
        self.manager = None
        
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
        
        # Define specialty compatibility matrix with EXACT specialty names as used in the code
        specialty_compatibility = {
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
            # Select random interaction type
            interaction_type = random.choice(["request_review", "generate_review"])
            
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
                    token_amount = random.randint(10, 30)
                    
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
            
            elif interaction_type == "generate_review":
                # Find pending reviews
                pending_reviews = []
                for agent_name, agent in self.agents.items():
                    agent_pending = agent.get_pending_reviews()
                    for review in agent_pending:
                        pending_reviews.append((agent_name, review))
                
                print(f"\nFound {len(pending_reviews)} pending reviews")
                
                if not pending_reviews:
                    outcome = {
                        "interaction": "generate_review",
                        "success": False,
                        "message": "No pending reviews found"
                    }
                else:
                    # Select random pending review
                    reviewer_name, paper = random.choice(pending_reviews)
                    reviewer = self.agents[reviewer_name]
                    print(f"Selected reviewer {reviewer_name} to complete review for paper ID {paper['id']}")
                    
                    # Generate and submit review
                    review_content = reviewer.generate_review(paper["id"])
                    success, message = reviewer.submit_review(
                        paper_id=paper["id"],
                        review_content=review_content
                    )
                    
                    print(f"Review submission result: {success}, {message}")
                    
                    outcome = {
                        "interaction": "generate_review",
                        "reviewer": reviewer_name,
                        "reviewer_specialty": reviewer.specialty,
                        "paper_id": paper["id"],
                        "paper_field": paper.get('field', 'Unknown'),
                        "success": success,
                        "message": message
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
            reviews_completed = sum(1 for i in interaction_results if i["interaction"] == "generate_review" and i["success"])
            
            print(f"  - Review requests: {review_requests}/{sum(1 for i in interaction_results if i['interaction'] == 'request_review')}")
            print(f"  - Reviews completed: {reviews_completed}/{sum(1 for i in interaction_results if i['interaction'] == 'generate_review')}")
            
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
        
        # Define specialty compatibility matrix (same as in simulate_random_interactions)
        specialty_compatibility = {
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
                compatible_specialties = specialty_compatibility.get(paper_field, [paper_field])
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
    
    # Create simulation
    simulation = PeerReviewSimulation()
    
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