"""
Token System for managing the peer review economy.

This module handles the token-based incentive system where researchers
earn and spend tokens to request and provide peer reviews.
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple

class TokenSystem:
    """
    Token system for managing researcher tokens.
    """
    
    def __init__(self, data_path: str = "tokens.json", initial_tokens: int = 100):
        """
        Initialize the token system.
        
        Args:
            data_path: Path to the JSON file for persistent storage
            initial_tokens: Initial token balance for new users
        """
        self.data_path = data_path
        self.initial_tokens = initial_tokens
        self.token_balances = {}  # {researcher_id: token_balance}
        self.transactions = []    # List of transaction records
        
        # Load existing data or initialize fresh
        self._load_data()
    
    def _load_data(self):
        """Load data from file or initialize with empty data."""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                    self.token_balances = data.get('token_balances', {})
                    self.transactions = data.get('transactions', [])
            except (json.JSONDecodeError, FileNotFoundError):
                # Initialize with empty data if file is corrupt
                self.token_balances = {}
                self.transactions = []
        else:
            # Initialize with empty data
            self.token_balances = {}
            self.transactions = []
    
    def _save_data(self):
        """Save data to the JSON file."""
        data = {
            'token_balances': self.token_balances,
            'transactions': self.transactions
        }
        
        try:
            with open(self.data_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving token data: {e}")
    
    def register_researcher(self, researcher_id: str) -> int:
        """
        Register a new researcher with initial tokens.
        
        Args:
            researcher_id: ID of the researcher
            
        Returns:
            Current token balance
        """
        # If researcher already exists, just return current balance
        if researcher_id in self.token_balances:
            return self.token_balances[researcher_id]
        
        # Add new researcher with initial tokens
        self.token_balances[researcher_id] = self.initial_tokens
        
        # Record transaction
        transaction = {
            'type': 'registration',
            'researcher_id': researcher_id,
            'amount': self.initial_tokens,
            'balance': self.initial_tokens,
            'timestamp': self._get_timestamp()
        }
        self.transactions.append(transaction)
        
        self._save_data()
        return self.initial_tokens
    
    def get_balance(self, researcher_id: str) -> int:
        """
        Get the token balance for a researcher.
        
        Args:
            researcher_id: ID of the researcher
            
        Returns:
            Current token balance (0 if researcher not found)
        """
        return self.token_balances.get(researcher_id, 0)
    
    def transfer_tokens(
        self, 
        from_researcher: str, 
        to_researcher: str, 
        amount: int, 
        reason: str
    ) -> Tuple[bool, str]:
        """
        Transfer tokens from one researcher to another.
        
        Args:
            from_researcher: ID of the sending researcher
            to_researcher: ID of the receiving researcher
            amount: Number of tokens to transfer
            reason: Reason for the transfer
            
        Returns:
            Tuple of (success, message)
        """
        # Validate researchers exist
        if from_researcher not in self.token_balances:
            return False, f"Sender {from_researcher} does not exist"
        
        if to_researcher not in self.token_balances:
            # Auto-register the receiving researcher
            self.register_researcher(to_researcher)
        
        # Validate amount
        if amount <= 0:
            return False, "Transfer amount must be positive"
        
        # Check sufficient balance
        if self.token_balances[from_researcher] < amount:
            return False, f"Insufficient tokens. Balance: {self.token_balances[from_researcher]}, Required: {amount}"
        
        # Perform transfer
        self.token_balances[from_researcher] -= amount
        self.token_balances[to_researcher] += amount
        
        # Record transaction
        transaction = {
            'type': 'transfer',
            'from_researcher': from_researcher,
            'to_researcher': to_researcher,
            'amount': amount,
            'reason': reason,
            'from_balance': self.token_balances[from_researcher],
            'to_balance': self.token_balances[to_researcher],
            'timestamp': self._get_timestamp()
        }
        self.transactions.append(transaction)
        
        self._save_data()
        return True, f"Successfully transferred {amount} tokens"
    
    def request_review(
        self, 
        requester_id: str, 
        reviewer_id: str, 
        paper_id: str, 
        amount: int
    ) -> Tuple[bool, str]:
        """
        Request a review by transferring tokens to a reviewer.
        
        Args:
            requester_id: ID of the researcher requesting the review
            reviewer_id: ID of the researcher to review the paper
            paper_id: ID of the paper to be reviewed
            amount: Number of tokens to offer for the review
            
        Returns:
            Tuple of (success, message)
        """
        # Transfer tokens with review request as reason
        success, message = self.transfer_tokens(
            from_researcher=requester_id,
            to_researcher=reviewer_id,
            amount=amount,
            reason=f"Review request for paper {paper_id}"
        )
        
        if success:
            # Record specific review request transaction
            transaction = {
                'type': 'review_request',
                'requester_id': requester_id,
                'reviewer_id': reviewer_id,
                'paper_id': paper_id,
                'amount': amount,
                'status': 'pending',
                'timestamp': self._get_timestamp()
            }
            self.transactions.append(transaction)
            self._save_data()
        
        return success, message
    
    def complete_review(self, reviewer_id: str, paper_id: str) -> bool:
        """
        Mark a review as completed.
        
        Args:
            reviewer_id: ID of the reviewer
            paper_id: ID of the paper reviewed
            
        Returns:
            True if review was found and marked completed, False otherwise
        """
        # Find the pending review request
        for transaction in self.transactions:
            if (transaction['type'] == 'review_request' and
                transaction['reviewer_id'] == reviewer_id and
                transaction['paper_id'] == paper_id and
                transaction['status'] == 'pending'):
                
                # Mark as completed
                transaction['status'] = 'completed'
                transaction['completion_timestamp'] = self._get_timestamp()
                
                # Record completion transaction
                completion = {
                    'type': 'review_completion',
                    'reviewer_id': reviewer_id,
                    'paper_id': paper_id,
                    'related_request': transaction,
                    'timestamp': self._get_timestamp()
                }
                self.transactions.append(completion)
                
                self._save_data()
                return True
        
        return False
    
    def cancel_review(self, requester_id: str, reviewer_id: str, paper_id: str) -> Tuple[bool, str]:
        """
        Cancel a review request and refund tokens.
        
        Args:
            requester_id: ID of the researcher who requested the review
            reviewer_id: ID of the researcher assigned to review
            paper_id: ID of the paper
            
        Returns:
            Tuple of (success, message)
        """
        # Find the pending review request
        for transaction in self.transactions:
            if (transaction['type'] == 'review_request' and
                transaction['requester_id'] == requester_id and
                transaction['reviewer_id'] == reviewer_id and
                transaction['paper_id'] == paper_id and
                transaction['status'] == 'pending'):
                
                # Mark as cancelled
                transaction['status'] = 'cancelled'
                transaction['cancellation_timestamp'] = self._get_timestamp()
                
                # Refund tokens - no balance check needed here as we're returning tokens 
                # that were already transferred to the reviewer back to the requester
                amount = transaction['amount']
                self.token_balances[reviewer_id] -= amount
                self.token_balances[requester_id] += amount
                
                # Record cancellation transaction
                cancellation = {
                    'type': 'review_cancellation',
                    'requester_id': requester_id,
                    'reviewer_id': reviewer_id,
                    'paper_id': paper_id,
                    'amount_refunded': amount,
                    'related_request': transaction,
                    'timestamp': self._get_timestamp()
                }
                self.transactions.append(cancellation)
                
                self._save_data()
                return True, f"Review cancelled and {amount} tokens refunded"
        
        return False, "Review request not found or not in pending status"
    
    def get_researcher_transaction_history(self, researcher_id: str) -> List[Dict[str, Any]]:
        """
        Get all transactions involving a researcher.
        
        Args:
            researcher_id: ID of the researcher
            
        Returns:
            List of transaction records
        """
        return [
            t for t in self.transactions
            if (t.get('researcher_id') == researcher_id or
                t.get('requester_id') == researcher_id or
                t.get('reviewer_id') == researcher_id or
                t.get('from_researcher') == researcher_id or
                t.get('to_researcher') == researcher_id)
        ]
    
    def get_reviews_by_reviewer(self, reviewer_id: str) -> List[Dict[str, Any]]:
        """
        Get all completed reviews by a specific reviewer.
        
        Args:
            reviewer_id: ID of the reviewer
            
        Returns:
            List of completed review transactions
        """
        return [
            t for t in self.transactions
            if (t.get('type') == 'review_completion' and 
                t.get('reviewer_id') == reviewer_id)
        ]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """
        Get a leaderboard of researchers sorted by token balance.
        
        Returns:
            List of researchers with their balances, sorted by balance (descending)
        """
        leaderboard = [
            {'researcher_id': researcher_id, 'balance': balance}
            for researcher_id, balance in self.token_balances.items()
        ]
        
        # Sort by balance (descending)
        leaderboard.sort(key=lambda x: x['balance'], reverse=True)
        
        return leaderboard
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about reviews in the system.
        
        Returns:
            Dictionary with review statistics
        """
        stats = {
            'total_reviews_requested': 0,
            'total_reviews_completed': 0,
            'total_reviews_cancelled': 0,
            'total_tokens_spent': 0,
            'average_token_per_review': 0,
            'researchers': {}
        }
        
        # Count reviews
        for transaction in self.transactions:
            if transaction['type'] == 'review_request':
                stats['total_reviews_requested'] += 1
                stats['total_tokens_spent'] += transaction['amount']
                
                # Track per researcher
                requester_id = transaction['requester_id']
                reviewer_id = transaction['reviewer_id']
                
                # Initialize researcher stats if not exists
                if requester_id not in stats['researchers']:
                    stats['researchers'][requester_id] = {
                        'reviews_requested': 0, 
                        'reviews_received': 0,
                        'tokens_spent': 0,
                        'tokens_earned': 0
                    }
                
                if reviewer_id not in stats['researchers']:
                    stats['researchers'][reviewer_id] = {
                        'reviews_requested': 0, 
                        'reviews_received': 0,
                        'tokens_spent': 0,
                        'tokens_earned': 0
                    }
                
                # Update stats
                stats['researchers'][requester_id]['reviews_requested'] += 1
                stats['researchers'][requester_id]['tokens_spent'] += transaction['amount']
                stats['researchers'][reviewer_id]['reviews_received'] += 1
                stats['researchers'][reviewer_id]['tokens_earned'] += transaction['amount']
            
            elif transaction['type'] == 'review_completion':
                stats['total_reviews_completed'] += 1
            
            elif transaction['type'] == 'review_cancellation':
                stats['total_reviews_cancelled'] += 1
        
        # Calculate average
        if stats['total_reviews_requested'] > 0:
            stats['average_token_per_review'] = stats['total_tokens_spent'] / stats['total_reviews_requested']
        
        return stats
    
    def get_all_researchers(self) -> List[str]:
        """
        Get all registered researchers in the token system.
        
        Returns:
            List of researcher IDs
        """
        return list(self.token_balances.keys())
    
    def spend_tokens(
        self, 
        researcher_id: str, 
        amount: int, 
        reason: str
    ) -> Tuple[bool, str]:
        """
        Spend tokens from a researcher's balance.
        
        Args:
            researcher_id: ID of the researcher spending tokens
            amount: Number of tokens to spend
            reason: Reason for spending tokens
            
        Returns:
            Tuple of (success, message)
        """
        # Validate researcher exists
        if researcher_id not in self.token_balances:
            return False, f"Researcher {researcher_id} does not exist"
        
        # Validate amount
        if amount <= 0:
            return False, "Spend amount must be positive"
        
        # Check sufficient balance
        if self.token_balances[researcher_id] < amount:
            return False, f"Insufficient tokens. Balance: {self.token_balances[researcher_id]}, Required: {amount}"
        
        # Deduct tokens
        self.token_balances[researcher_id] -= amount
        
        # Record transaction
        transaction = {
            'type': 'spend',
            'researcher_id': researcher_id,
            'amount': amount,
            'reason': reason,
            'balance': self.token_balances[researcher_id],
            'timestamp': self._get_timestamp()
        }
        self.transactions.append(transaction)
        
        self._save_data()
        return True, f"Successfully spent {amount} tokens for {reason}" 