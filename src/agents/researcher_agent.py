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

# Add the project root to the path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.paper_database import PaperDatabase
from src.core.token_system import TokenSystem

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
        
        # Register with token system
        self.token_system.register_researcher(self.name)
        
        # Track behavior parameters based on bias
        self.behavior_params = self._initialize_behavior_params()
        
        # Track workload (number of accepted reviews)
        self.current_workload = 0
        self.max_workload = 5  # Maximum number of ongoing reviews
    
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
        return self.paper_db.add_paper(paper_data)
    
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
        
        return success, message
    
    def respond_to_invitation(self, paper_id: str, token_amount: int) -> Tuple[bool, str]:
        """
        Respond to a review invitation based on researcher behavior.
        
        Args:
            paper_id: ID of the paper to be reviewed
            token_amount: Reputation tokens staked on paper (priority signal)
            
        Returns:
            Tuple of (accepted, reason)
            - accepted: True if invitation accepted, False otherwise
            - reason: Explanation of the decision
        """
        # Get paper details
        paper = self.paper_db.get_paper(paper_id)
        if not paper:
            return False, f"Paper {paper_id} not found"
        
        paper_field = paper.get('field', '')
        
        # Check workload
        if self.current_workload >= self.max_workload:
            return False, "Workload too high, cannot accept more reviews"
        
        # Check field compatibility
        # In a real implementation, this should use the specialty_compatibility matrix
        # (which should be moved to a central configuration)
        if paper_field != self.specialty and paper_field not in ["AI", "General"]:
            # Apply a penalty to acceptance probability for non-matching fields
            field_compatibility_penalty = 0.5
        else:
            field_compatibility_penalty = 1.0
        
        # Token influence on probability - logarithmic scale to prevent guaranteed acceptance
        # Higher tokens increase acceptance probability, but with diminishing returns
        if token_amount <= 0:
            token_boost = 0.0
        else:
            # Log scale with diminishing returns
            # Formula: min(0.4, 0.1 * ln(token_amount + 1))
            # This gives around 0.1 boost at 10 tokens, 0.2 at 70 tokens, 0.3 at 400 tokens, capped at 0.4
            token_boost = min(0.4, 0.1 * math.log(token_amount + 1))
        
        # Calculate final acceptance probability
        base_probability = self.behavior_params["acceptance_probability"]
        final_probability = base_probability * field_compatibility_penalty + token_boost
        
        # Clamp probability between 0 and 0.95 (never 100% guaranteed)
        final_probability = max(0.0, min(0.95, final_probability))
        
        # Print detailed probability calculation for debugging
        print(f"Review invitation probability calculation for {self.name}:")
        print(f"  Base probability: {base_probability:.2f}")
        print(f"  Field compatibility: {field_compatibility_penalty:.2f}")
        print(f"  Token boost: {token_boost:.2f} (from {token_amount} tokens)")
        print(f"  Final probability: {final_probability:.2f}")
        
        # Random decision based on probability
        decision = random.random() < final_probability
        
        # Check for ignore or delay behaviors
        if not decision:
            if random.random() < self.behavior_params["ignore_probability"]:
                return False, "Invitation ignored"
            elif random.random() < self.behavior_params["delay_probability"]:
                return False, "Response delayed"
            else:
                return False, "Invitation declined due to lack of interest or expertise"
        
        # If accepting, increment workload
        self.current_workload += 1
        
        return True, f"Review invitation accepted for paper in field: {paper_field}"
    
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
        
        # Add review to paper
        review = {
            'reviewer_id': self.name,
            'content': review_content,
            'timestamp': self.token_system._get_timestamp()
        }
        
        # Add quality indicators based on behavior
        quality_factor = self.behavior_params["quality_factor"]
        review['metadata'] = {
            'quality_score': round(random.uniform(3, 5) * quality_factor, 1),  # 1-5 scale
            'thoroughness': round(random.uniform(0.5, 1.0) * quality_factor, 2)  # 0-1 scale
        }
        
        # Add to paper database
        self.paper_db.add_review(paper_id, review)
        
        # Mark review as completed in token system
        self.token_system.complete_review(self.name, paper_id)
        
        # Update researcher workload
        self.complete_review(paper_id)
        
        return True, f"Review submitted for paper {paper_id}"
    
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
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a review for a paper using the agent's capabilities.
        
        Args:
            paper_id: ID of the paper to review
            custom_prompt: Custom prompt for review generation
            
        Returns:
            Dictionary with review content
        """
        # Get the paper
        paper = self.paper_db.get_paper(paper_id)
        if not paper:
            return {"error": f"Paper {paper_id} not found"}
        
        # Create review prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = (
                f"Based on your expertise in {self.specialty} and considering that {self.bias},\n"
                f"Please review the following paper:\n\n"
                f"Title: {paper['title']}\n"
                f"Authors: {', '.join(paper['authors'])}\n"
                f"Abstract: {paper['abstract']}\n"
                f"Keywords: {', '.join(paper['keywords'])}\n\n"
                f"Please provide a comprehensive review including:\n"
                f"1. Summary of the paper's contributions\n"
                f"2. Strengths of the paper\n"
                f"3. Weaknesses and areas for improvement\n"
                f"4. Clarity and organization\n"
                f"5. Technical correctness\n"
                f"6. Overall recommendation (Accept, Minor Revision, Major Revision, or Reject)\n"
                f"7. Additional comments for authors\n\n"
                f"Format your review as a JSON structure with these sections."
            )
        
        # Generate a message to yourself to create the review
        review_result = self.generate_reply(
            messages=[{"role": "user", "content": prompt}],
            sender=self
        )
        
        # Structure the review content
        try:
            # Try to extract JSON from the response
            import json
            import re
            
            # Look for JSON-like content
            json_match = re.search(r'\{.*\}', review_result, re.DOTALL)
            if json_match:
                review_json = json.loads(json_match.group())
            else:
                # If no JSON found, create a structured review
                review_json = {
                    "summary": "Generated review summary",
                    "strengths": "Generated strengths",
                    "weaknesses": "Generated weaknesses",
                    "clarity": "Assessment of clarity",
                    "technical_correctness": "Assessment of technical correctness",
                    "recommendation": "Generated recommendation",
                    "comments": review_result
                }
            
            # Add metadata
            review_json["reviewer_id"] = self.name
            review_json["paper_id"] = paper_id
            review_json["timestamp"] = self.token_system._get_timestamp()
            
            return review_json
            
        except Exception as e:
            # Handle parsing errors
            return {
                "reviewer_id": self.name,
                "paper_id": paper_id,
                "timestamp": self.token_system._get_timestamp(),
                "summary": "Error parsing review",
                "comments": review_result,
                "error": str(e)
            } 