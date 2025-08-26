"""
Researcher Agent module for the Peer Review simulation.

This module defines the ResearcherAgent class which extends the AssistantAgent
with specialized capabilities for the peer review process.
"""

import autogen
from typing import Dict, List, Any, Optional, Tuple, Callable
import os
import sys
import random
import math
import time # Added for timestamping memory entries

# Add the project root to the path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.paper_database import PaperDatabase
from src.core.token_system import TokenSystem

class AgentMemory:
    """Stores memories for an agent, including interactions, reviews, and observations."""
    def __init__(self):
        self.interactions: List[Dict[str, Any]] = [] # General interactions
        self.reviews_given: List[Dict[str, Any]] = [] # Reviews performed by this agent
        self.reviews_received: List[Dict[str, Any]] = [] # Reviews received for this agent's papers
        self.paper_submissions: List[Dict[str, Any]] = [] # Papers submitted by this agent
        self.token_transactions: List[Dict[str, Any]] = [] # Token transaction history
        self.reputation_log: Dict[str, List[Dict[str, Any]]] = {} # Reputation changes or observations about others

    def add_interaction(self, event_type: str, details: Dict[str, Any]):
        """Adds a general interaction event to memory."""
        self.interactions.append({
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details
        })

    def add_review_given(self, paper_id: str, author_id: str, review_details: Dict[str, Any]):
        self.reviews_given.append({
            "timestamp": time.time(),
            "paper_id": paper_id,
            "author_id": author_id,
            "review_details": review_details
        })

    def add_review_received(self, paper_id: str, reviewer_id: str, review_details: Dict[str, Any]):
        self.reviews_received.append({
            "timestamp": time.time(),
            "paper_id": paper_id,
            "reviewer_id": reviewer_id,
            "review_details": review_details
        })

    def add_paper_submission(self, paper_id: str, title: str, journal_or_conference: Optional[str] = None, token_spent: Optional[int] = None):
        self.paper_submissions.append({
            "timestamp": time.time(),
            "paper_id": paper_id,
            "title": title,
            "submitted_to": journal_or_conference,
            "token_spent_for_priority": token_spent
        })

    def add_token_transaction(self, transaction_type: str, amount: int, related_to: Optional[str] = None, details: Optional[Dict[str, Any]] = None, counterparty: Optional[str] = None):
        self.token_transactions.append({
            "timestamp": time.time(),
            "type": transaction_type,
            "amount": amount,
            "related_to": related_to,
            "counterparty": counterparty,
            "details": details or {}
        })

    def add_reputation_observation(self, agent_id: str, observation: str, rating: Optional[int] = None, context: Optional[Dict[str, Any]] = None):
        if agent_id not in self.reputation_log:
            self.reputation_log[agent_id] = []
        self.reputation_log[agent_id].append({
            "timestamp": time.time(),
            "observation": observation,
            "rating": rating,
            "context": context or {}
        })

    def get_history_with_agent(self, target_agent_id: str, max_entries: int = 5) -> str:
        """Generates a concise summary of past interactions with a specific agent for LLM context."""
        history_entries = []

        # Reviews given to target_agent_id's papers
        for review in sorted(self.reviews_given, key=lambda x: x["timestamp"], reverse=True):
            if review.get("author_id") == target_agent_id:
                paper = AgentMemory.paper_db.get_paper(review["paper_id"])
                title = paper.get('title', 'N/A') if paper else 'Unknown Paper'
                decision = review['review_details'].get('decision', 'N/A')
                history_entries.append(f"You reviewed their paper '{title}' (outcome: {decision}).")

        # Reviews received from target_agent_id
        for review in sorted(self.reviews_received, key=lambda x: x["timestamp"], reverse=True):
            if review.get("reviewer_id") == target_agent_id:
                paper = AgentMemory.paper_db.get_paper(review["paper_id"])
                title = paper.get('title', 'N/A') if paper else 'Unknown Paper'
                decision = review['review_details'].get('decision', 'N/A')
                history_entries.append(f"They reviewed your paper '{title}' (outcome: {decision}).")
        
        # Token transactions with target_agent_id
        for tx in sorted(self.token_transactions, key=lambda x: x["timestamp"], reverse=True):
            if tx.get("counterparty") == target_agent_id:
                amount_str = f"earned {tx['amount']}" if tx['amount'] > 0 else f"spent {-tx['amount']}"
                history_entries.append(f"Token transaction with them: you {amount_str} tokens (type: {tx['type']}, related to: {tx.get('related_to', 'N/A')}).")

        # Reputation observations about target_agent_id
        if target_agent_id in self.reputation_log:
            for obs in sorted(self.reputation_log[target_agent_id], key=lambda x: x["timestamp"], reverse=True):
                rating_str = f" (rated: {obs['rating']})" if obs.get('rating') else ""
                history_entries.append(f"Your observation about them: {obs['observation']}{rating_str}.")

        if not history_entries:
            return "No significant past interactions found with this agent."
        
        # Return the most recent entries, up to max_entries
        return "\n".join(history_entries[:max_entries])

    def get_full_memory_summary(self, max_recent_tx: int = 3, max_recent_obs_per_agent: int = 1) -> str:
        """Returns a more detailed string summary of all memories for LLM context."""
        summary = ["Your Memory & Status Summary:"]
        summary.append(f"- Reviews Given: {len(self.reviews_given)}")
        summary.append(f"- Reviews Received: {len(self.reviews_received)}")
        summary.append(f"- Papers Submitted: {len(self.paper_submissions)}")
        
        summary.append("- Recent Token Transactions:")
        if self.token_transactions:
            for tx in self.token_transactions[-max_recent_tx:]:
                amount_str = f"earned {tx['amount']}" if tx['amount'] > 0 else f"spent {-tx['amount']}"
                summary.append(f"  - {amount_str} tokens (type: {tx['type']}, related: {tx.get('related_to', 'N/A')}, counterparty: {tx.get('counterparty', 'N/A')})")
        else:
            summary.append("  - No token transactions yet.")
        
        if self.reputation_log:
            summary.append("- Recent Reputation Observations:")
            for agent_id, obs_list in self.reputation_log.items():
                if obs_list:
                    latest_obs = obs_list[-1]
                    rating_str = f" (rated: {latest_obs['rating']})" if latest_obs.get('rating') else ""
                    summary.append(f"  - About {agent_id}: {latest_obs['observation']}{rating_str}.")
        return "\n".join(summary)

# Need to pass paper_db to memory for some methods
AgentMemory.paper_db = None # Class variable to be set by ResearcherAgent

class ResearcherAgent(autogen.AssistantAgent):
    """
    ResearcherAgent represents a researcher in the peer review system.
    
    This agent extends the AssistantAgent with capabilities for:
    - Managing a portfolio of papers
    - Requesting reviews for their papers
    - Reviewing papers from other researchers
    - Managing tokens in the peer review economy
    """
    
    def __init__(
        self,
        name: str,
        specialty: str,
        system_message: str,
        paper_db: PaperDatabase,
        token_system: TokenSystem,
        bias: str = "",
        llm_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the ResearcherAgent.
        
        Args:
            name: Name of the researcher agent
            specialty: Research specialty area
            system_message: System message describing the agent's role
            paper_db: Paper database instance
            token_system: Token system instance
            bias: Research bias or preference
            llm_config: LLM configuration
            **kwargs: Additional keyword arguments for AssistantAgent
        """
        # Enhance system message with researcher details
        enhanced_system_message = (
            f"{system_message}\n\n"
            f"You are a researcher specializing in {specialty}.\n"
            f"Your name is {name}.\n"
        )
        
        if bias:
            enhanced_system_message += f"You have the following research bias: {bias}\n"
        
        # Add token and peer review information
        enhanced_system_message += (
            "\nYou participate in a token-based peer review system where:\n"
            "- You can request reviews for your papers by spending tokens\n"
            "- You earn tokens by reviewing papers from other researchers\n"
            "- You aim to advance your research while maintaining a healthy token balance\n"
        )
        
        # Initialize the AssistantAgent
        super().__init__(
            name=name,
            system_message=enhanced_system_message,
            llm_config=llm_config,
            **kwargs
        )
        
        # Researcher-specific attributes
        self.specialty = specialty
        self.bias = bias
        self.paper_db = paper_db
        self.token_system = token_system
        self.memory = AgentMemory() # Initialize memory
        if AgentMemory.paper_db is None: # Set paper_db for AgentMemory class
            AgentMemory.paper_db = paper_db
        
        # Register with token system
        self.token_system.register_researcher(self.name)
        
        # Track behavior parameters based on bias
        self.behavior_params = self._initialize_behavior_params()
        
        # Track workload (number of accepted reviews)
        self.current_workload = 0
        self.max_workload = 5  # Maximum number of ongoing reviews
        self.personality = self._derive_personality_from_bias(bias) # Add personality
    
    def _derive_personality_from_bias(self, bias_str: str) -> str:
        """ Derives a personality trait from the bias string. 
            This will be used in prompts according to simulation-setup.mdc
        """
        bias_lower = bias_str.lower()
        if "friendly" in bias_lower or "helpful" in bias_lower:
            return "Friendly"
        if "selfish" in bias_lower or "competitive" in bias_lower:
            return "Selfish"
        if "gatekeeper" in bias_lower or "strict" in bias_lower:
            return "Gatekeeper"
        if "malicious" in bias_lower or "adversarial" in bias_lower:
            return "Malicious"
        if "neutral" in bias_lower or not bias_str:
            return "Neutral"
        # Default if no clear match from the examples in simulation-setup.mdc
        if "busy" in bias_lower: return "Busy"
        if "thorough" in bias_lower: return "Thorough"
        if "fast" in bias_lower: return "Fast"
        if "selective" in bias_lower: return "Selective"
        return "Neutral" # Default
    
    def _initialize_behavior_params(self) -> Dict[str, float]:
        """
        Initialize behavior parameters based on researcher bias.
        
        Returns:
            Dictionary of behavioral parameters
        """
        # Default parameters for a neutral researcher
        params = {
            "acceptance_probability": 0.7,  # Probability of accepting a review request
            "delay_probability": 0.2,       # Probability of delaying a response
            "ignore_probability": 0.1,      # Probability of ignoring a request
            "quality_factor": 1.0,          # Factor affecting review quality
            "speed_factor": 1.0             # Factor affecting review speed
        }
        
        # Adjust based on bias
        if self.bias:
            bias_lower = self.bias.lower()
            
            if "friendly" in bias_lower or "helpful" in bias_lower:
                params["acceptance_probability"] = 0.9
                params["delay_probability"] = 0.1
                params["ignore_probability"] = 0.0
                params["quality_factor"] = 1.2
                
            elif "busy" in bias_lower or "slow" in bias_lower:
                params["acceptance_probability"] = 0.5
                params["delay_probability"] = 0.4
                params["ignore_probability"] = 0.1
                params["speed_factor"] = 0.7
                
            elif "thorough" in bias_lower or "detail" in bias_lower:
                params["quality_factor"] = 1.5
                params["speed_factor"] = 0.8
                
            elif "fast" in bias_lower or "quick" in bias_lower:
                params["speed_factor"] = 1.3
                params["quality_factor"] = 0.8
                
            elif "malicious" in bias_lower or "adversarial" in bias_lower:
                params["acceptance_probability"] = 0.6
                params["quality_factor"] = 0.5
                
            elif "selective" in bias_lower or "picky" in bias_lower:
                params["acceptance_probability"] = 0.4
                
        return params
    
    def get_token_balance(self) -> int:
        """
        Get the current token balance.
        
        Returns:
            Current token balance
        """
        return self.token_system.get_balance(self.name)
    
    def get_papers(self) -> List[Dict[str, Any]]:
        """
        Get papers owned by this researcher.
        
        Returns:
            List of papers owned by this researcher
        """
        papers = self.paper_db.get_papers_by_owner(self.name)
        
        # Log mismatches between agent specialty and paper field
        for paper in papers:
            if 'field' in paper and paper['field'] != self.specialty:
                print(f"[MISMATCH] Researcher {self.name} (specialty: {self.specialty}) has paper in field: {paper['field']}")
        
        return papers
    
    def publish_paper(self, paper_data: Dict[str, Any]) -> str:
        """
        Publish a new paper.
        
        Args:
            paper_data: Dictionary with paper data
            
        Returns:
            ID of the published paper
        """
        # Ensure owner is set to this researcher
        paper_data['owner_id'] = self.name
        
        # Ensure researcher is in authors
        if 'authors' not in paper_data:
            paper_data['authors'] = [self.name]
        elif self.name not in paper_data['authors']:
            paper_data['authors'].append(self.name)
        
        # Add to database
        paper_id = self.paper_db.add_paper(paper_data)
        if paper_id:
            self.memory.add_paper_submission(
                paper_id=paper_id,
                title=paper_data.get("title", "Untitled"),
                journal_or_conference=paper_data.get("venue") 
            )
        return paper_id
    
    def request_review(
        self, 
        paper_id: str, 
        reviewer_id: str, 
        token_amount: int
    ) -> Tuple[bool, str]:
        """
        Request a review for a paper.
        
        Args:
            paper_id: ID of the paper to be reviewed
            reviewer_id: ID of the researcher to review the paper
            token_amount: Number of tokens to offer for the review
            
        Returns:
            Tuple of (success, message)
        """
        # Verify paper ownership
        paper = self.paper_db.get_paper(paper_id)
        if not paper:
            return False, f"Paper {paper_id} not found"
        
        if paper['owner_id'] != self.name:
            return False, f"You don't own paper {paper_id}"
        
        # Note: The caller (PeerReviewSimulation) is responsible for ensuring that the
        # reviewer's specialty matches the paper's field before calling this method
        
        # Request review
        success, message = self.token_system.request_review(
            requester_id=self.name,
            reviewer_id=reviewer_id,
            paper_id=paper_id,
            amount=token_amount
        )
        
        if success:
            # Add review request to paper
            request = {
                'reviewer_id': reviewer_id,
                'requester_id': self.name,
                'token_amount': token_amount,
                'status': 'pending',
                'timestamp': self.token_system._get_timestamp()
            }
            self.paper_db.add_review_request(paper_id, request)
            # Log token transaction for requesting a review (spending tokens)
            self.memory.add_token_transaction(
                transaction_type="review_request_fee",
                amount=-token_amount,
                related_to=f"paper_{paper_id}",
                counterparty=reviewer_id,
                details={"paper_id": paper_id, "reviewer_id": reviewer_id, "action": "requested_review"}
            )
        
        return success, message
    
    def respond_to_invitation(self, paper_id: str, token_amount: int) -> Tuple[bool, str, str]:
        """
        Respond to a review invitation based on researcher behavior.
        
        Args:
            paper_id: ID of the paper to be reviewed
            token_amount: Reputation tokens staked on paper (priority signal)
            
        Returns:
            Tuple of (decision, reasoning_or_message, thought_process)
            Decision: True for accept, False for decline/ignore
            Reasoning: Brief explanation of the decision
            Thought Process: Detailed internal thinking of the agent
        """
        # Check workload
        if self.current_workload >= self.max_workload:
            thought_process = f"I already have {self.current_workload}/{self.max_workload} reviews. I cannot take on more work right now."
            self.memory.add_interaction("review_invitation_response", {"paper_id": paper_id, "decision": "declined_workload", "reason": "Max workload reached"})
            return False, f"{self.name} declined review for {paper_id} (max workload reached).", thought_process
        
        paper = self.paper_db.get_paper(paper_id)
        if not paper:
            thought_process = f"I can't find paper {paper_id} in the database. I can't review what doesn't exist."
            return False, f"Paper {paper_id} not found for review invitation.", thought_process
        
        author_id = paper.get('owner_id', 'Unknown_Author')
        paper_title = paper.get('title', 'Untitled Paper')
        paper_field = paper.get('field', 'Unknown_Field')
        
        # Prepare context for LLM decision
        prompt_context = {
            "agent_name": self.name,
            "agent_personality": self.personality,
            "agent_specialty": self.specialty,
            "current_tokens": self.get_token_balance(),
            "current_workload": self.current_workload,
            "max_workload": self.max_workload,
            "paper_id": paper_id,
            "paper_title": paper_title,
            "paper_field": paper_field,
            "paper_author": author_id,
            "token_reward_for_review": token_amount, # Assuming token_amount is the reward for completing the review
            "past_interaction_summary": self.memory.get_history_with_agent(author_id),
            "overall_memory_summary": self.memory.get_full_memory_summary()
        }
        
        # Construct the decision-making prompt based on simulation-setup.mdc examples
        decision_prompt = f"""
        You are {self.name}, a researcher with a {self.personality} personality, specializing in {self.specialty}.
        Your current token balance is {prompt_context['current_tokens']} and your workload is {prompt_context['current_workload']}/{prompt_context['max_workload']} reviews.

        You have received a review invitation for the paper "{prompt_context['paper_title']}" (ID: {prompt_context['paper_id']}) by {prompt_context['paper_author']} in the field of {prompt_context['paper_field']}.
        The offered token reward for completing this review is {prompt_context['token_reward_for_review']}.

        Your past interactions with {prompt_context['paper_author']}:
        {prompt_context['past_interaction_summary']}

        Consider these rules and thought processes from the simulation guidelines:
        - "Should I accept this review? Will it benefit me? Is this author my rival?"
        - "This author gave me a harsh review last round... maybe I'll reject their paper."
        - "I'm low on tokens. Accepting this review might help me get published faster later."
        - "This paper competes with mine. If I review it, I might slow its acceptance."
        - "I recognize this reviewer — they always write quality feedback. Worth requesting again." (Though you are the reviewer here)
        - "The editor assigned me again without reward. I'll ignore it this time." (Consider token reward)
        - "They cited me in their paper. I'll be generous." (Assume you don't know this unless in memory)
        - "I already have {self.current_workload} pending reviews. One more will hurt my reputation (or workload)."
        
        First, provide your detailed THOUGHT_PROCESS about this decision. Explain your reasoning, considering your personality, specialty, workload, token balance, and past interactions with the author.
        
        After your thought process, decide whether to ACCEPT or DECLINE the review request.
        Provide a brief reasoning for your decision.
        
        Format your response as:
        THOUGHT_PROCESS: [Your detailed internal thinking here]
        DECISION: [ACCEPT/DECLINE]
        REASONING: [Your brief reasoning here]
        """
        
        # Use the agent's LLM to make the decision
        raw_response = self.generate_reply(messages=[{"role": "user", "content": decision_prompt}])
        
        # Parse the response
        decision_str = "DECLINE" # Default to decline if parsing fails
        reasoning = "Could not properly decide due to an internal error or unclear LLM response."
        thought_process = "No thought process recorded."
        
        # Handle different response formats (string or dictionary)
        if isinstance(raw_response, dict) and 'content' in raw_response:
            # The API returned a dictionary format, extract the content field
            response_text = raw_response['content']
        elif isinstance(raw_response, str):
            # The API returned a string directly
            response_text = raw_response
        else:
            # Unexpected format, try to convert to string
            print(f"Warning: Unexpected response type from LLM for {self.name}: {type(raw_response)}")
            try:
                response_text = str(raw_response)
            except:
                response_text = f"Could not convert response to string: {type(raw_response)}"
                
        # Try to extract the thought process
        thought_marker = "THOUGHT_PROCESS:"
        decision_marker = "DECISION:"
        reasoning_marker = "REASONING:"
        
        thought_process_start = response_text.find(thought_marker)
        if thought_process_start != -1:
            thought_process_end = response_text.find(decision_marker, thought_process_start)
            if thought_process_end != -1:
                thought_process = response_text[thought_process_start + len(thought_marker):thought_process_end].strip()
        
        # Extract decision and reasoning
        response_lines = response_text.strip().split('\n')
        for line in response_lines:
            if line.startswith(decision_marker):
                decision_str = line.replace(decision_marker, "").strip().upper()
            elif line.startswith(reasoning_marker):
                reasoning = line.replace(reasoning_marker, "").strip()
                
        # If unable to extract thought process with markers, use the raw response
        if thought_process == "No thought process recorded." and len(response_text) > 0:
            thought_process = f"Raw LLM response: {response_text[:200]}..."
        
        accepted = (decision_str == "ACCEPT")
        
        # Log decision in memory
        self.memory.add_interaction("review_invitation_response", {
            "paper_id": paper_id, 
            "author_id": author_id,
            "decision": "accepted" if accepted else "declined", 
            "reasoning": reasoning,
            "thought_process": thought_process,
            "raw_llm_response": raw_response
        })
        
        if accepted:
            self.current_workload += 1
            self.paper_db.update_review_request_status(paper_id, self.name, "accepted")
            return True, f"{self.name} accepted review for {paper_id}. Reasoning: {reasoning}", thought_process
        else:
            self.paper_db.update_review_request_status(paper_id, self.name, "declined")
            # Potentially add ignore/delay logic here based on another LLM call or probability
            return False, f"{self.name} declined review for {paper_id}. Reasoning: {reasoning}", thought_process
    
    def get_review_invitations(self) -> List[Dict[str, Any]]:
        """
        Get all pending review invitations for this researcher.
        
        Returns:
            List of pending review invitations
        """
        invitations = []
        
        # Search all papers for invitations to this researcher
        papers = self.paper_db.get_all_papers()
        for paper in papers:
            if "review_invitations" in paper:
                for invitation in paper["review_invitations"]:
                    if invitation["reviewer_id"] == self.name and invitation["status"] == "invited":
                        # Add paper info to invitation
                        invitation_with_paper = invitation.copy()
                        invitation_with_paper["paper_id"] = paper["id"]
                        invitation_with_paper["paper_title"] = paper.get("title", "Untitled")
                        invitation_with_paper["paper_field"] = paper.get("field", "")
                        invitations.append(invitation_with_paper)
        
        return invitations
    
    def complete_review(self, paper_id: str) -> None:
        """
        Mark a review as completed and update workload.
        
        Args:
            paper_id: ID of the paper that was reviewed
        """
        # Decrement workload when a review is completed
        if self.current_workload > 0:
            self.current_workload -= 1
        
        # Could also update researcher's performance metrics here
    
    def submit_review(
        self, 
        paper_id: str, 
        review_content: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Submit a review for a paper.
        
        Args:
            paper_id: ID of the paper being reviewed
            review_content: Dictionary with review content
            
        Returns:
            Tuple of (success, message)
        """
        # Verify the paper exists
        paper = self.paper_db.get_paper(paper_id)
        if not paper:
            return False, f"Paper {paper_id} not found"
        
        # Verify this researcher has been assigned to review this paper
        # Check both review_requests and review_invitations
        is_assigned = False
        
        # Check in review_requests
        for request in paper.get('review_requests', []):
            if request['reviewer_id'] == self.name and request['status'] == 'accepted':
                is_assigned = True
                break
                
        # If not found in review_requests, check in review_invitations
        if not is_assigned:
            for invitation in paper.get('review_invitations', []):
                if invitation['reviewer_id'] == self.name and invitation['status'] == 'accepted':
                    is_assigned = True
                    break
        
        if not is_assigned:
            return False, f"You are not assigned to review paper {paper_id}"
        
        # Ensure reviewer_id is part of the review data
        review_data = review_content.copy()
        if 'reviewer_id' not in review_data:
            review_data['reviewer_id'] = self.name
        
        # Update paper database with the review
        success = self.paper_db.add_review(
            paper_id=paper_id,
            review=review_data
        )
        
        if success:
            # Log the review in agent's memory
            paper = self.paper_db.get_paper(paper_id)
            author_id = paper.get("owner_id") if paper else "Unknown_Author"
            self.memory.add_review_given(
                paper_id=paper_id, 
                author_id=author_id, 
                review_details=review_content
            )
            
            # Mark review as completed for workload tracking
            self.current_workload = max(0, self.current_workload - 1)
            
            # Here, tokens should be awarded by the TokenSystem based on this action.
            # The ResearcherAgent itself doesn't award tokens, but its memory should reflect the transaction.
            # This requires the TokenSystem to notify the agent or for the simulation to call a memory update.
            # For now, we'll assume a subsequent call will update token memory if TokenSystem handles it.
            # Example: self.memory.add_token_transaction("review_reward", earned_amount, related_to=f"review_for_{paper_id}")
            print(f"{self.name} submitted review for paper {paper_id}")
            return True, f"Review for paper {paper_id} submitted successfully"
        else:
            message = f"Failed to submit review for paper {paper_id}"
            print(message)
            return False, message
    
    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """
        Get papers pending review by this researcher.
        
        Returns:
            List of papers pending review
        """
        all_papers = self.paper_db.get_all_papers()
        pending_reviews = []
        
        for paper in all_papers:
            for request in paper['review_requests']:
                if request['reviewer_id'] == self.name and request['status'] == 'pending':
                    paper_with_request = paper.copy()
                    paper_with_request['current_request'] = request
                    pending_reviews.append(paper_with_request)
        
        return pending_reviews
    
    def get_transaction_history(self) -> List[Dict[str, Any]]:
        """
        Get transaction history for this researcher.
        
        Returns:
            List of transactions
        """
        return self.token_system.get_researcher_transaction_history(self.name)
    
    def generate_review(
        self, 
        paper_id: str,
        # custom_prompt: Optional[str] = None # Removing custom_prompt to enforce new structured approach
    ) -> Dict[str, Any]: # Will now return a dict with {"review_content": ..., "thought_process": ...}
        """
        Generate a review for a paper using the agent's capabilities, including explicit reasoning.
        
        Args:
            paper_id: ID of the paper to review
            
        Returns:
            Dictionary with structured review content and the agent's thought process.
        """
        paper = self.paper_db.get_paper(paper_id)
        if not paper:
            return {
                "review_content": {"error": f"Paper {paper_id} not found"},
                "thought_process": "Error: Paper not found, cannot generate review."
            }

        author_id = paper.get('owner_id', 'Unknown_Author')
        paper_title = paper.get('title', 'Untitled Paper')
        paper_abstract = paper.get('abstract', 'No abstract available.')
        paper_keywords = paper.get('keywords', [])
        paper_field = paper.get('field', 'Unknown_Field')

        # Prepare context for LLM decision
        prompt_context = {
            "agent_name": self.name,
            "agent_personality": self.personality,
            "agent_specialty": self.specialty,
            "current_tokens": self.get_token_balance(),
            "paper_title": paper_title,
            "paper_abstract": paper_abstract,
            "paper_keywords": ", ".join(paper_keywords) if paper_keywords else "N/A",
            "paper_field": paper_field,
            "paper_author": author_id,
            "past_interaction_summary": self.memory.get_history_with_agent(author_id),
            "guidelines_examples": [
                "This author gave me a harsh review last round... maybe I'll be equally critical or find flaws more easily.",
                "This paper competes with mine. If I review it, I might strategically delay its acceptance or be overly critical of overlapping areas.",
                "They cited my work prominently! I should be fair but perhaps a bit more generous or highlight the connections.",
                "I need more tokens; providing a high-quality, quick review might earn a bonus.",
                "My personality is Gatekeeper. I must ensure only top-quality papers pass. I will be very strict.",
                "My personality is Malicious. I will try to subtly sabotage this paper if it benefits me or if I dislike the author."
            ]
        }

        review_prompt = f"""
        You are {prompt_context['agent_name']}, a {prompt_context['agent_personality']} researcher specializing in {prompt_context['agent_specialty']}.\
        Your current token balance is {prompt_context['current_tokens']}.\
        You are reviewing the paper "{prompt_context['paper_title']}" (Field: {prompt_context['paper_field']}) by {prompt_context['paper_author']}.\
        Paper Abstract: {prompt_context['paper_abstract']}\
        Paper Keywords: {prompt_context['paper_keywords']}\

        Your past interactions with {prompt_context['paper_author']}:\
        {prompt_context['past_interaction_summary']}\

        Reviewing Guidelines & Strategic Thoughts (Consider these based on your personality and goals):\
        - How does your personality ({prompt_context['agent_personality']}) affect your review style (e.g., strictness, generosity, potential for bias)?\
        - Does this paper compete with your own research? How might that influence your review?\
        - Did the author cite your work? How does that make you feel about the paper?\
        - How does your current token balance influence your desire to provide a high-quality/fast review for potential bonuses?\
        - Consider these behavioral examples: {'; '.join(prompt_context['guidelines_examples'])}\

        Instructions:\
        1. First, write your internal "Thought Process" for approaching this review. Explain your reasoning, any biases you're applying (consciously or unconsciously based on your personality and the situation), and your overall strategy for the review. This thought process should be detailed.\
        2. Second, based on your thought process, provide the actual "Review Content". The review content MUST be a single, valid JSON object with the following exact keys: "summary" (string), "strengths" (string), "weaknesses" (string), "clarity_assessment" (string), "technical_correctness_assessment" (string), "overall_recommendation" (string, one of: "Accept", "Minor Revision", "Major Revision", "Reject"), "confidence_score" (integer, 1-5, where 5 is very confident), "detailed_comments_for_author" (string).\

        Output Format (strictly follow this):\
        THOUGHT_PROCESS:\
        [Your detailed thought process and reasoning for the review strategy here. This can be multiple paragraphs.]\

        REVIEW_CONTENT:\
        {{...valid JSON review here...}}\
        """
        
        raw_llm_response = self.generate_reply(
            messages=[{"role": "user", "content": review_prompt}],
            sender=self # Important for some autogen setups if functions need to be triggered by this agent
        )

        thought_process = "LLM did not provide a thought process or it could not be parsed." # Default
        review_content_json = {"error": "Review could not be generated or parsed from LLM response."}

        # Handle different response formats (string or dictionary)
        if isinstance(raw_llm_response, dict) and 'content' in raw_llm_response:
            # The API returned a dictionary format, extract the content field
            llm_response_text = raw_llm_response['content']
        elif isinstance(raw_llm_response, str):
            # The API returned a string directly
            llm_response_text = raw_llm_response
        else:
            # Unexpected format, try to convert to string
            print(f"Warning: Unexpected response type from LLM for {self.name}: {type(raw_llm_response)}")
            try:
                llm_response_text = str(raw_llm_response)
            except:
                llm_response_text = f"Could not convert response to string: {type(raw_llm_response)}"
                
        try:
            thought_process_marker = "THOUGHT_PROCESS:"
            review_content_marker = "REVIEW_CONTENT:"
            
            # Extract thought process
            thought_process_start = llm_response_text.find(thought_process_marker)
            if thought_process_start != -1:
                review_content_start = llm_response_text.find(review_content_marker, thought_process_start)
                if review_content_start != -1:
                    thought_process = llm_response_text[thought_process_start + len(thought_process_marker):review_content_start].strip()
                    json_str = llm_response_text[review_content_start + len(review_content_marker):].strip()
                    
                    # Sanitize JSON string: LLMs sometimes add trailing commas or comments
                    import re
                    json_str = re.sub(r",(\s*\})", r"\1", json_str) # Remove trailing commas before closing brace
                    json_str = re.sub(r",(\s*\])", r"\1", json_str) # Remove trailing commas before closing bracket
                    json_str = re.sub(r"//.*?\n|/\*.*?\*/", "", json_str, flags=re.DOTALL) # Remove comments
                    
                    import json
                    review_content_json = json.loads(json_str)
                else:
                    # Fallback if markers are not found
                    thought_process = "Markers not found in complete response."
                    try:
                        # Try to parse the whole response as JSON
                        import json
                        review_content_json = json.loads(llm_response_text)
                    except json.JSONDecodeError:
                        review_content_json = {"error": "Failed to parse LLM response as JSON and markers not found.", "raw_response": llm_response_text[:200]}
            else:
                # No markers found
                thought_process = f"LLM response did not contain expected markers. Raw response: {llm_response_text[:200]}"
                try:
                    # Try to parse the whole response as JSON
                    import json
                    review_content_json = json.loads(llm_response_text)
                except json.JSONDecodeError:
                    review_content_json = {"error": "Failed to parse LLM response as JSON and markers not found.", "raw_response": llm_response_text[:200]}

        except json.JSONDecodeError as e:
            review_content_json = {"error": f"JSON parsing failed: {str(e)}", "raw_json_string": json_str if 'json_str' in locals() else llm_response_text[:200]}
            thought_process += " (JSON parsing of review content failed)"
        except Exception as e:
            review_content_json = {"error": f"Generic parsing error: {str(e)}", "raw_response": llm_response_text[:200]}
            thought_process += f" (Error: {str(e)})"

        # Add metadata to the parsed review content
        review_content_json["reviewer_id"] = self.name
        review_content_json["paper_id"] = paper_id
        review_content_json["review_timestamp"] = self.token_system._get_timestamp()
            
        return {
            "review_content": review_content_json,
            "thought_process": thought_process
        }

    # It's crucial that TokenSystem or the simulation calls this method when tokens are awarded.
    def record_token_award(self, amount: int, reason: str, related_to: Optional[str] = None, awarded_by: Optional[str] = None):
        """Records awarded tokens in memory."""
        self.memory.add_token_transaction(
            transaction_type=reason,
            amount=amount,
            related_to=related_to,
            counterparty=awarded_by,
            details={"reason": reason}
        )
        print(f"{self.name} recorded token award: {amount} for {reason}")

    def record_review_outcome(self, paper_id: str, reviewer_id: str, review_details: Dict[str, Any]):
        """Records the details of a review received for one of the agent's papers."""
        self.memory.add_review_received(paper_id, reviewer_id, review_details)
        # Optionally, add a reputation observation about the reviewer based on the review
        # This would require some logic or LLM call to assess the review quality from the author's perspective
        # E.g., if review_details["decision"] == "reject" and review_details["confidence"] < 3:
        #    self.memory.add_reputation_observation(reviewer_id, "Provided a low-confidence rejection.", rating=2) 