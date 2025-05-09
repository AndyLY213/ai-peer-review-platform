"""
Editor Agent module for the Peer Review simulation.

This module defines the EditorAgent class which extends the AssistantAgent
with specialized capabilities for managing the peer review process.
"""

import autogen
import random
import heapq
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import os
import sys

# Add the project root to the path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.paper_database import PaperDatabase
from src.core.token_system import TokenSystem


class EditorAgent(autogen.AssistantAgent):
    """
    EditorAgent represents a journal editor in the peer review system.
    
    This agent extends the AssistantAgent with capabilities for:
    - Screening submitted papers
    - Sending review invitations to appropriate reviewers
    - Managing the peer review workflow
    - Prioritizing papers based on author reputation signals
    """
    
    def __init__(
        self,
        name: str,
        journal: str,
        system_message: str,
        paper_db: PaperDatabase,
        token_system: TokenSystem,
        llm_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the EditorAgent.
        
        Args:
            name: Name of the editor agent
            journal: Name of the journal or conference
            system_message: System message describing the agent's role
            paper_db: Paper database instance
            token_system: Token system instance
            llm_config: LLM configuration
            **kwargs: Additional keyword arguments for AssistantAgent
        """
        # Enhance system message with editor details
        enhanced_system_message = (
            f"{system_message}\n\n"
            f"You are an editor for the journal/conference {journal}.\n"
            f"Your name is {name}.\n\n"
            f"As an editor, you:\n"
            f"- Screen submitted papers for scope fit and quality\n"
            f"- Invite appropriate reviewers based on paper topic and researcher specialty\n"
            f"- Make decisions on papers based on reviewer feedback\n"
            f"- Maintain high standards for the peer review process\n"
        )
        
        # Initialize the AssistantAgent
        super().__init__(
            name=name,
            system_message=enhanced_system_message,
            llm_config=llm_config,
            **kwargs
        )
        
        # Editor-specific attributes
        self.journal = journal
        self.paper_db = paper_db
        self.token_system = token_system
        
        # Register with token system
        self.token_system.register_researcher(self.name)
        
        # Track papers under review
        self.papers_in_review = {}  # paper_id -> {reviewers: [], status: ""}
        
        # Priority queue for submitted papers
        # Format: list of tuples (priority_score, timestamp, paper_id)
        # Lower values are processed first, so we negate priority_score to make higher scores higher priority
        self.submission_queue = []
        
        # Load existing submissions that might be in the database
        self._initialize_submission_queue()
    
    def _initialize_submission_queue(self):
        """Initialize submission queue with existing submitted papers."""
        submitted_papers = self.paper_db.get_papers_by_status("submitted")
        
        for paper in submitted_papers:
            # If paper is already in the queue, skip
            if any(paper["id"] == p_id for _, _, p_id in self.submission_queue):
                continue
                
            # Calculate priority score based on author's token bid (if any)
            priority_score = paper.get("priority_score", 0)
            
            # Use paper creation timestamp or current time
            timestamp = paper.get("timestamp", datetime.now().timestamp())
            
            # Add to priority queue
            # We negate priority_score because heapq is a min-heap, and we want higher scores to have higher priority
            heapq.heappush(self.submission_queue, (-priority_score, timestamp, paper["id"]))
    
    def submit_paper(self, paper_id: str, priority_score: int = 0) -> bool:
        """
        Add a paper to the submission queue with a priority score.
        
        Args:
            paper_id: ID of the paper to submit
            priority_score: Priority score for this paper (higher = higher priority)
            
        Returns:
            True if paper was successfully added to queue, False otherwise
        """
        paper = self.paper_db.get_paper(paper_id)
        if not paper:
            return False
            
        # If paper is already in the queue, update its priority score
        for i, (_, t, p_id) in enumerate(self.submission_queue):
            if p_id == paper_id:
                # Remove the existing entry
                # This is inefficient for large queues but should be fine for this simulation
                self.submission_queue.remove((_, t, p_id))
                heapq.heapify(self.submission_queue)
                break
                
        # Update paper status to submitted if it's not already
        if paper.get("status") != "submitted":
            self.paper_db.update_paper(paper_id, {"status": "submitted"})
            
        # Add paper to queue with priority score and current timestamp
        # Negate priority_score for the heap (so higher values have higher priority)
        current_time = datetime.now().timestamp()
        heapq.heappush(self.submission_queue, (-priority_score, current_time, paper_id))
        
        # Update paper with priority score
        self.paper_db.update_paper(paper_id, {"priority_score": priority_score})
        
        return True
    
    def process_next_submission(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Process the next paper in the submission queue.
        Papers are processed in order of priority score, then submission time.
        
        Returns:
            Tuple of (success, message, paper_data)
            - success: Whether a paper was processed
            - message: Description of the action
            - paper_data: Data about the processed paper
        """
        if not self.submission_queue:
            return False, "No papers in submission queue", {}
            
        # Get the highest priority paper
        neg_priority, timestamp, paper_id = heapq.heappop(self.submission_queue)
        
        # Process the paper (screen it)
        accepted, decision, message = self.screen_paper(paper_id)
        
        # Get paper data
        paper = self.paper_db.get_paper(paper_id)
        
        paper_data = {
            "paper_id": paper_id,
            "title": paper.get("title", "Untitled"),
            "priority_score": -neg_priority,  # Convert back to positive
            "author": paper.get("owner_id", "Unknown"),
            "decision": decision,
            "message": message
        }
        
        return True, f"Processed paper {paper_id} with priority {-neg_priority}", paper_data
    
    def screen_paper(self, paper_id: str) -> Tuple[bool, str, str]:
        """
        Screen a paper for scope fit and quality.
        
        Args:
            paper_id: ID of the paper to screen
            
        Returns:
            Tuple of (acceptance, decision, message)
            - acceptance: True if paper is accepted for review, False otherwise
            - decision: One of "accept_for_review", "desk_reject"
            - message: Explanation of the decision
        """
        paper = self.paper_db.get_paper(paper_id)
        if not paper:
            return False, "desk_reject", f"Paper {paper_id} not found"
        
        # Simple screening logic based on paper field
        # In a real implementation, this would involve LLM-based analysis
        # or more complex criteria
        paper_field = paper.get('field', '')
        
        # Example of desk rejection criteria:
        # 1. Paper with no title or abstract
        if not paper.get('title', '') or not paper.get('abstract', ''):
            return False, "desk_reject", "Paper missing essential information (title or abstract)"
            
        # 2. Paper with insufficient content
        content_length = len(paper.get('abstract', ''))
        if content_length < 100:  # arbitrary threshold
            return False, "desk_reject", "Paper abstract too short for proper evaluation"
            
        # Accept paper for review
        # Update paper status
        self.paper_db.update_paper(paper_id, {"status": "in_review"})
        
        return True, "accept_for_review", f"Paper accepted for review in field: {paper_field}"
    
    def invite_reviewers(
        self, 
        paper_id: str, 
        potential_reviewers: List[str],
        specialty_compatibility: Dict[str, List[str]],
        reviewer_data: Dict[str, Dict[str, Any]],
        num_reviewers: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Invite reviewers for a paper based on specialty compatibility and reputation.
        
        Args:
            paper_id: ID of the paper to be reviewed
            potential_reviewers: List of potential reviewer names
            specialty_compatibility: Dictionary mapping specialties to compatible specialties
            reviewer_data: Dictionary mapping reviewer names to their data
            num_reviewers: Number of reviewers to invite
            
        Returns:
            List of invitation results (dictionaries with reviewer_id, invited, reason)
        """
        paper = self.paper_db.get_paper(paper_id)
        if not paper:
            return [{"error": f"Paper {paper_id} not found"}]
        
        paper_field = paper.get('field', '')
        invitations = []
        
        # Get the owner to avoid inviting them
        owner_id = paper.get('owner_id', '')
        
        # Get compatible specialties for this paper
        compatible_specialties = specialty_compatibility.get(paper_field, [paper_field])
        
        # Filter potential reviewers by compatible specialty
        valid_reviewers = []
        for reviewer_id in potential_reviewers:
            # Skip paper owner
            if reviewer_id == owner_id:
                continue
                
            reviewer = reviewer_data.get(reviewer_id, {})
            specialty = reviewer.get('specialty', '')
            
            if specialty in compatible_specialties:
                # Get reviewer reputation
                reputation = self.token_system.get_balance(reviewer_id)
                
                valid_reviewers.append({
                    "id": reviewer_id,
                    "specialty": specialty,
                    "reputation": reputation
                })
        
        # If not enough valid reviewers, log an issue
        if len(valid_reviewers) < num_reviewers:
            print(f"Warning: Not enough valid reviewers for paper {paper_id} in field {paper_field}")
            # Use all available valid reviewers
        
        # Select reviewers to invite (up to num_reviewers)
        invited_reviewers = []
        if valid_reviewers:
            # Sort by reputation (higher reputation = higher chance of selection)
            # Add some randomness to avoid always picking the same high-reputation reviewers
            for reviewer in valid_reviewers:
                # Add some random noise to reputation (±20%)
                random_factor = random.uniform(0.8, 1.2)
                reviewer["selection_score"] = reviewer["reputation"] * random_factor
            
            # Sort by selection score (descending)
            valid_reviewers.sort(key=lambda x: x["selection_score"], reverse=True)
            
            # Take the top reviewers
            num_to_invite = min(len(valid_reviewers), num_reviewers)
            invited_reviewers = [r["id"] for r in valid_reviewers[:num_to_invite]]
            
            # Print selection details
            print(f"Selected {num_to_invite} reviewers for paper {paper_id}:")
            for i, reviewer in enumerate(valid_reviewers[:num_to_invite]):
                print(f"  {i+1}. {reviewer['id']} (specialty: {reviewer['specialty']}, reputation: {reviewer['reputation']})")
        
        # Create invitations
        for reviewer_id in potential_reviewers:
            invited = reviewer_id in invited_reviewers
            reason = ""
            
            if reviewer_id == owner_id:
                reason = "Cannot invite paper owner"
                invited = False
            elif reviewer_id not in [r["id"] for r in valid_reviewers]:
                reason = "Specialty not compatible with paper field"
                invited = False
            elif reviewer_id not in invited_reviewers:
                reason = "Not selected based on reputation and specialty fit"
                invited = False
            
            # Record the invitation
            invitation = {
                "reviewer_id": reviewer_id,
                "invited": invited,
                "reason": reason
            }
            invitations.append(invitation)
            
            if invited:
                # Record the invitation in the paper
                invitation_record = {
                    "reviewer_id": reviewer_id,
                    "status": "invited",
                    "timestamp": self.token_system._get_timestamp()
                }
                
                # Add to paper's review invitations
                if "review_invitations" not in paper:
                    paper["review_invitations"] = []
                    
                paper["review_invitations"].append(invitation_record)
                
                # Update the paper in the database
                self.paper_db.update_paper(paper_id, {"review_invitations": paper["review_invitations"]})
                
                # Track this paper as being in review
                if paper_id not in self.papers_in_review:
                    self.papers_in_review[paper_id] = {"reviewers": [], "status": "in_review"}
                
                self.papers_in_review[paper_id]["reviewers"].append(reviewer_id)
        
        return invitations
    
    def process_review_acceptance(
        self, 
        paper_id: str, 
        reviewer_id: str, 
        accepted: bool,
        token_amount: int,
        requester_id: str
    ) -> Tuple[bool, str]:
        """
        Process a reviewer's acceptance or rejection of a review invitation.
        
        Args:
            paper_id: ID of the paper to be reviewed
            reviewer_id: ID of the reviewer
            accepted: Whether the invitation was accepted
            token_amount: Tokens offered for the review
            requester_id: ID of the researcher requesting the review
            
        Returns:
            Tuple of (success, message)
        """
        paper = self.paper_db.get_paper(paper_id)
        if not paper:
            return False, f"Paper {paper_id} not found"
        
        # Find the invitation
        invitations = paper.get("review_invitations", [])
        invitation_found = False
        
        for invitation in invitations:
            if invitation["reviewer_id"] == reviewer_id and invitation["status"] == "invited":
                invitation_found = True
                
                # Update invitation status
                if accepted:
                    invitation["status"] = "accepted"
                    
                    # Handle token transfer
                    success, message = self.token_system.request_review(
                        requester_id=requester_id,
                        reviewer_id=reviewer_id,
                        paper_id=paper_id,
                        amount=token_amount
                    )
                    
                    if success:
                        # Add review request to paper
                        request = {
                            'reviewer_id': reviewer_id,
                            'requester_id': requester_id,
                            'token_amount': token_amount,
                            'status': 'accepted',
                            'timestamp': self.token_system._get_timestamp()
                        }
                        self.paper_db.add_review_request(paper_id, request)
                        
                        # Update paper
                        self.paper_db.update_paper(paper_id, {"review_invitations": invitations})
                        return True, f"Review accepted by {reviewer_id}"
                    else:
                        invitation["status"] = "failed"  # Failed due to token transfer issues
                        self.paper_db.update_paper(paper_id, {"review_invitations": invitations})
                        return False, message
                else:
                    invitation["status"] = "declined"
                    self.paper_db.update_paper(paper_id, {"review_invitations": invitations})
                    return True, f"Review declined by {reviewer_id}"
        
        if not invitation_found:
            return False, f"No pending invitation found for reviewer {reviewer_id} on paper {paper_id}"
    
    def get_papers_in_review(self) -> List[Dict[str, Any]]:
        """
        Get all papers currently in the review process.
        
        Returns:
            List of papers in review with their review status
        """
        papers_in_review = []
        
        for paper_id, review_data in self.papers_in_review.items():
            paper = self.paper_db.get_paper(paper_id)
            if paper:
                # Add review status to paper data
                paper_with_status = paper.copy()
                paper_with_status["review_status"] = review_data
                papers_in_review.append(paper_with_status)
        
        return papers_in_review
    
    def get_papers_needing_reviewers(self) -> List[Dict[str, Any]]:
        """
        Get papers that need more reviewers.
        
        Returns:
            List of papers that need more reviewers
        """
        papers_needing_reviewers = []
        
        # Check all papers in review
        for paper_id, review_data in self.papers_in_review.items():
            # If paper has fewer than 3 reviewers, it needs more
            if len(review_data["reviewers"]) < 3:
                paper = self.paper_db.get_paper(paper_id)
                if paper:
                    paper_with_status = paper.copy()
                    paper_with_status["review_status"] = review_data
                    papers_needing_reviewers.append(paper_with_status)
        
        return papers_needing_reviewers
    
    def get_submission_queue_status(self) -> List[Dict[str, Any]]:
        """
        Get the current status of the submission queue.
        
        Returns:
            List of papers in the queue with their priority information
        """
        queue_status = []
        
        # Make a copy of the queue to avoid modifying the actual queue
        queue_copy = self.submission_queue.copy()
        
        # Process queue in priority order
        while queue_copy:
            neg_priority, timestamp, paper_id = heapq.heappop(queue_copy)
            paper = self.paper_db.get_paper(paper_id)
            
            if paper:
                queue_status.append({
                    "paper_id": paper_id,
                    "title": paper.get("title", "Untitled"),
                    "author": paper.get("owner_id", "Unknown"),
                    "priority_score": -neg_priority,
                    "submission_time": timestamp
                })
        
        return queue_status 