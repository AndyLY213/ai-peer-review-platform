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
            # Select random interaction type
            interaction_type = random.choice(["submit_paper", "process_invitations", "generate_review", "process_queue"])
            
            if interaction_type == "submit_paper":
                # Random author
                author_name = random.choice(researcher_names)
                author = self.agents[author_name]
                
                # Create a new paper
                paper_title = f"Research on {author.specialty} - {i}"
                paper_abstract = f"This paper presents research in the field of {author.specialty}."
                
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
                
                # Authors stake between 10% and 30% of their reputation as priority signal
                # With some randomness to simulate different author behaviors
                max_stake = int(author_reputation * 0.3)
                min_stake = min(int(author_reputation * 0.1), 10)  # At least 10 if possible
                
                if max_stake <= min_stake:
                    priority_score = min_stake
                else:
                    priority_score = random.randint(min_stake, max_stake)
                
                print(f"\nAuthor {author_name} submitting paper ID {paper_id}: {paper_title}")
                print(f"  - Author reputation: {author_reputation}")
                print(f"  - Priority signal: {priority_score} tokens")
                
                # Submit paper to editor's queue with priority score
                self.editor.submit_paper(paper_id, priority_score)
                
                # Spend author's reputation tokens on priority signal
                self.token_system.spend_tokens(
                    researcher_id=author_name,
                    amount=priority_score,
                    reason=f"Priority signal for paper {paper_id}"
                )
                
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
                print(f"  - Decision: {paper_data['decision']}")
                print(f"  - Message: {paper_data['message']}")
                
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
                    print(f"Editor invited {num_invited} reviewers for paper {paper_id}")
                    
                    # Log invited reviewers
                    invited_list = [inv["reviewer_id"] for inv in invitations if inv.get("invited", False)]
                    print(f"Invited reviewers: {', '.join(invited_list)}")
                    
                    outcome = {
                        "interaction": "process_queue",
                        "paper_id": paper_id,
                        "paper_title": paper_data["title"],
                        "author": paper_data["author"],
                        "priority_score": paper_data["priority_score"],
                        "screening": "passed",
                        "invitations_sent": num_invited,
                        "invited_reviewers": invited_list
                    }
                else:
                    outcome = {
                        "interaction": "process_queue",
                        "paper_id": paper_id,
                        "paper_title": paper_data["title"],
                        "author": paper_data["author"],
                        "priority_score": paper_data["priority_score"],
                        "screening": "rejected",
                        "reason": paper_data["message"]
                    }
            
            elif interaction_type == "process_invitations":
                # Process pending review invitations
                invitations_processed = 0
                accepted_invitations = 0
                
                # For each researcher, check if they have pending invitations
                for researcher_name in researcher_names:
                    researcher = self.agents[researcher_name]
                    pending_invitations = researcher.get_review_invitations()
                    
                    # Process each invitation
                    for invitation in pending_invitations:
                        paper_id = invitation["paper_id"]
                        paper = self.paper_db.get_paper(paper_id)
                        
                        if not paper:
                            continue
                        
                        # Get the paper's priority score to influence reviewer decisions
                        priority_score = paper.get("priority_score", 0)
                        
                        # Researcher decides whether to accept invitation
                        accepted, reason = researcher.respond_to_invitation(paper_id, priority_score)
                        print(f"\nResearcher {researcher_name} {'accepted' if accepted else 'declined'} "
                              f"review invitation for paper {paper_id}: {reason}")
                        
                        # Process acceptance/rejection with editor
                        requester_id = paper.get("owner_id", "Unknown")
                        success, message = self.editor.process_review_acceptance(
                            paper_id=paper_id,
                            reviewer_id=researcher_name,
                            accepted=accepted,
                            token_amount=priority_score,  # Using priority score for token amount
                            requester_id=requester_id
                        )
                        
                        print(f"Editor processed response: {message}")
                        
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
                        "message": f"Processed {invitations_processed} invitations, {accepted_invitations} accepted"
                    }
            
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
                    
                    print(f"Selected reviewer {reviewer_name} to complete review for paper ID {paper_id}")
                    
                    # Generate and submit review
                    review_content = reviewer.generate_review(paper_id)
                    success, message = reviewer.submit_review(
                        paper_id=paper_id,
                        review_content=review_content
                    )
                    
                    print(f"Review submission result: {success}, {message}")
                    
                    outcome = {
                        "interaction": "generate_review",
                        "reviewer": reviewer_name,
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
            papers_submitted = sum(1 for i in interaction_results if i["interaction"] == "submit_paper")
            papers_accepted = sum(1 for i in interaction_results if i["interaction"] == "submit_paper" and i.get("screening") == "passed")
            invitations_processed = sum(i.get("invitations_processed", 0) for i in interaction_results if i["interaction"] == "process_invitations")
            invitations_accepted = sum(i.get("invitations_accepted", 0) for i in interaction_results if i["interaction"] == "process_invitations")
            reviews_completed = sum(1 for i in interaction_results if i["interaction"] == "generate_review" and i["success"])
            
            print(f"  - Papers submitted: {papers_submitted}")
            print(f"  - Papers passing editor screening: {papers_accepted}/{papers_submitted}")
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

def main():
    """Main entry point for the peer review simulation."""
    print("Welcome to the Peer Review Simulation System!")
    
    # Create the simulation
    simulation = PeerReviewSimulation()
    
    # Create researcher agents from templates
    simulation.create_all_researchers(assign_papers=True)
    
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