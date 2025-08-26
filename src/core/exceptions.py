"""
Custom exceptions for the Peer Review Simulation System.

This module defines specific exceptions for different error conditions
to provide better error handling and debugging capabilities.
"""

class PeerReviewError(Exception):
    """Base exception for all peer review system errors."""
    pass

class ConfigurationError(PeerReviewError):
    """Raised when there's an issue with system configuration."""
    pass

class LLMConnectionError(PeerReviewError):
    """Raised when there's an issue connecting to the LLM provider."""
    pass

class DatabaseError(PeerReviewError):
    """Raised when there's an issue with database operations."""
    pass

class PaperNotFoundError(DatabaseError):
    """Raised when a requested paper is not found."""
    def __init__(self, paper_id: str):
        self.paper_id = paper_id
        super().__init__(f"Paper with ID '{paper_id}' not found")

class ResearcherNotFoundError(PeerReviewError):
    """Raised when a requested researcher is not found."""
    def __init__(self, researcher_id: str):
        self.researcher_id = researcher_id
        super().__init__(f"Researcher with ID '{researcher_id}' not found")

class TokenError(PeerReviewError):
    """Base exception for token system errors."""
    pass

class InsufficientTokensError(TokenError):
    """Raised when a researcher doesn't have enough tokens for an operation."""
    def __init__(self, researcher_id: str, required: int, available: int):
        self.researcher_id = researcher_id
        self.required = required
        self.available = available
        super().__init__(
            f"Researcher '{researcher_id}' has insufficient tokens. "
            f"Required: {required}, Available: {available}"
        )

class InvalidTokenAmountError(TokenError):
    """Raised when an invalid token amount is specified."""
    def __init__(self, amount: int):
        self.amount = amount
        super().__init__(f"Invalid token amount: {amount}. Must be positive.")

class ReviewError(PeerReviewError):
    """Base exception for review-related errors."""
    pass

class ReviewNotFoundError(ReviewError):
    """Raised when a requested review is not found."""
    def __init__(self, paper_id: str, reviewer_id: str):
        self.paper_id = paper_id
        self.reviewer_id = reviewer_id
        super().__init__(f"Review for paper '{paper_id}' by reviewer '{reviewer_id}' not found")

class DuplicateReviewRequestError(ReviewError):
    """Raised when trying to request a review that already exists."""
    def __init__(self, paper_id: str, reviewer_id: str):
        self.paper_id = paper_id
        self.reviewer_id = reviewer_id
        super().__init__(f"Review request already exists for paper '{paper_id}' by reviewer '{reviewer_id}'")

class SelfReviewError(ReviewError):
    """Raised when a researcher tries to review their own paper."""
    def __init__(self, researcher_id: str, paper_id: str):
        self.researcher_id = researcher_id
        self.paper_id = paper_id
        super().__init__(f"Researcher '{researcher_id}' cannot review their own paper '{paper_id}'")

class SimulationError(PeerReviewError):
    """Base exception for simulation errors."""
    pass

class InsufficientResearchersError(SimulationError):
    """Raised when there aren't enough researchers for simulation."""
    def __init__(self, required: int, available: int):
        self.required = required
        self.available = available
        super().__init__(f"Insufficient researchers for simulation. Required: {required}, Available: {available}")

class FileOperationError(PeerReviewError):
    """Raised when file operations fail."""
    def __init__(self, operation: str, filepath: str, original_error: Exception):
        self.operation = operation
        self.filepath = filepath
        self.original_error = original_error
        super().__init__(f"Failed to {operation} file '{filepath}': {str(original_error)}")

class ValidationError(PeerReviewError):
    """Raised when data validation fails."""
    def __init__(self, field: str, value: any, expected: str):
        self.field = field
        self.value = value
        self.expected = expected
        super().__init__(f"Validation failed for field '{field}': got {value}, expected {expected}")

class TemplateError(PeerReviewError):
    """Raised when there's an issue with researcher templates."""
    def __init__(self, template_name: str, issue: str):
        self.template_name = template_name
        self.issue = issue
        super().__init__(f"Template error for '{template_name}': {issue}")

class AgentCreationError(PeerReviewError):
    """Raised when agent creation fails."""
    def __init__(self, agent_name: str, reason: str):
        self.agent_name = agent_name
        self.reason = reason
        super().__init__(f"Failed to create agent '{agent_name}': {reason}")

class DatasetError(PeerReviewError):
    """Raised when there's an issue with dataset operations."""
    def __init__(self, dataset_path: str, issue: str):
        self.dataset_path = dataset_path
        self.issue = issue
        super().__init__(f"Dataset error for '{dataset_path}': {issue}")

class LLMResponseError(PeerReviewError):
    """Raised when LLM response is invalid or cannot be parsed."""
    def __init__(self, response: str, parsing_error: str):
        self.response = response
        self.parsing_error = parsing_error
        super().__init__(f"Failed to parse LLM response: {parsing_error}")

class GroupChatError(PeerReviewError):
    """Raised when there's an issue with group chat operations."""
    def __init__(self, issue: str):
        self.issue = issue
        super().__init__(f"Group chat error: {issue}")

class DeadlineError(PeerReviewError):
    """Raised when there's an issue with deadline management."""
    def __init__(self, issue: str):
        self.issue = issue
        super().__init__(f"Deadline error: {issue}")

class BiasSystemError(PeerReviewError):
    """Raised when there's an issue with bias system operations."""
    def __init__(self, issue: str):
        self.issue = issue
        super().__init__(f"Bias system error: {issue}")

class NetworkError(PeerReviewError):
    """Raised when there's an issue with network operations."""
    def __init__(self, issue: str):
        self.issue = issue
        super().__init__(f"Network error: {issue}")

class CareerSystemError(PeerReviewError):
    """Raised when there's an issue with career system operations."""
    def __init__(self, issue: str):
        self.issue = issue
        super().__init__(f"Career system error: {issue}")

class ValidationError(PeerReviewError):
    """Raised when validation fails."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

class SimulationError(PeerReviewError):
    """Raised when there's an error in simulation execution."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

class BiasSystemError(PeerReviewError):
    """Errors related to bias system operations"""
    pass

class NetworkSystemError(PeerReviewError):
    """Errors related to social network operations"""
    pass



class FundingSystemError(PeerReviewError):
    """Errors related to funding system operations"""
    pass

class VenueSystemError(PeerReviewError):
    """Errors related to venue management"""
    pass

class TemporalConstraintError(PeerReviewError):
    """Errors related to timing and deadline management"""
    pass