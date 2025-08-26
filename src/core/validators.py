"""
Input validation utilities for the Peer Review Simulation System.

This module provides validation functions for different types of data
to ensure data integrity and prevent errors.
"""

import re
from typing import Any, Dict, List, Optional, Union
from src.core.exceptions import ValidationError
from src.core.logging_config import get_logger
from src.core.constants import (
    MAX_TOKEN_AMOUNT, MIN_TOKEN_AMOUNT, MAX_PAPER_TITLE_LENGTH, 
    MIN_PAPER_TITLE_LENGTH, MAX_PAPER_ABSTRACT_LENGTH, MAX_AUTHOR_NAME_LENGTH,
    VALID_PAPER_STATUSES, VALID_RESEARCH_FIELDS, VALID_ID_PATTERN,
    MAX_STRING_LENGTH, MAX_SIMULATION_ROUNDS, MAX_INTERACTIONS_PER_ROUND
)

logger = get_logger(__name__)

def validate_researcher_id(researcher_id: str) -> str:
    """
    Validate researcher ID format.
    
    Args:
        researcher_id: The researcher ID to validate
        
    Returns:
        The validated researcher ID
        
    Raises:
        ValidationError: If the researcher ID is invalid
    """
    if not isinstance(researcher_id, str):
        raise ValidationError("researcher_id", researcher_id, "string")
    
    if not researcher_id or not researcher_id.strip():
        raise ValidationError("researcher_id", researcher_id, "non-empty string")
    
    # Allow alphanumeric characters, underscores, and hyphens
    if not re.match(VALID_ID_PATTERN, researcher_id):
        raise ValidationError("researcher_id", researcher_id, "alphanumeric with underscores/hyphens only")
    
    return researcher_id.strip()

def validate_paper_id(paper_id: str) -> str:
    """
    Validate paper ID format.
    
    Args:
        paper_id: The paper ID to validate
        
    Returns:
        The validated paper ID
        
    Raises:
        ValidationError: If the paper ID is invalid
    """
    if not isinstance(paper_id, str):
        raise ValidationError("paper_id", paper_id, "string")
    
    if not paper_id or not paper_id.strip():
        raise ValidationError("paper_id", paper_id, "non-empty string")
    
    # Allow alphanumeric characters, underscores, and hyphens
    if not re.match(VALID_ID_PATTERN, paper_id):
        raise ValidationError("paper_id", paper_id, "alphanumeric with underscores/hyphens only")
    
    return paper_id.strip()

def validate_token_amount(amount: Union[int, str]) -> int:
    """
    Validate token amount.
    
    Args:
        amount: The token amount to validate
        
    Returns:
        The validated token amount as integer
        
    Raises:
        ValidationError: If the token amount is invalid
    """
    try:
        amount_int = int(amount)
    except (ValueError, TypeError):
        raise ValidationError("token_amount", amount, "integer")
    
    if amount_int < 0:
        raise ValidationError("token_amount", amount_int, "non-negative integer")
    
    if amount_int > MAX_TOKEN_AMOUNT:
        raise ValidationError("token_amount", amount_int, f"integer <= {MAX_TOKEN_AMOUNT}")
    
    return amount_int

def validate_paper_title(title: str) -> str:
    """
    Validate paper title.
    
    Args:
        title: The paper title to validate
        
    Returns:
        The validated and cleaned title
        
    Raises:
        ValidationError: If the title is invalid
    """
    if not isinstance(title, str):
        raise ValidationError("paper_title", title, "string")
    
    title = title.strip()
    
    if not title:
        raise ValidationError("paper_title", title, "non-empty string")
    
    if len(title) < MIN_PAPER_TITLE_LENGTH:
        raise ValidationError("paper_title", title, f"string with at least {MIN_PAPER_TITLE_LENGTH} characters")
    
    if len(title) > MAX_PAPER_TITLE_LENGTH:
        raise ValidationError("paper_title", title, f"string with at most {MAX_PAPER_TITLE_LENGTH} characters")
    
    return title

def validate_paper_abstract(abstract: str) -> str:
    """
    Validate paper abstract.
    
    Args:
        abstract: The paper abstract to validate
        
    Returns:
        The validated and cleaned abstract
        
    Raises:
        ValidationError: If the abstract is invalid
    """
    if not isinstance(abstract, str):
        raise ValidationError("paper_abstract", abstract, "string")
    
    abstract = abstract.strip()
    
    if len(abstract) > MAX_PAPER_ABSTRACT_LENGTH:
        raise ValidationError("paper_abstract", abstract, f"string with at most {MAX_PAPER_ABSTRACT_LENGTH} characters")
    
    return abstract

def validate_authors_list(authors: List[str]) -> List[str]:
    """
    Validate list of authors.
    
    Args:
        authors: List of author names to validate
        
    Returns:
        The validated list of authors
        
    Raises:
        ValidationError: If the authors list is invalid
    """
    if not isinstance(authors, list):
        raise ValidationError("authors", authors, "list of strings")
    
    if not authors:
        raise ValidationError("authors", authors, "non-empty list")
    
    validated_authors = []
    for i, author in enumerate(authors):
        if not isinstance(author, str):
            raise ValidationError(f"authors[{i}]", author, "string")
        
        author = author.strip()
        if not author:
            raise ValidationError(f"authors[{i}]", author, "non-empty string")
        
        if len(author) > MAX_AUTHOR_NAME_LENGTH:
            raise ValidationError(f"authors[{i}]", author, f"string with at most {MAX_AUTHOR_NAME_LENGTH} characters")
        
        validated_authors.append(author)
    
    return validated_authors

def validate_research_field(field: str) -> str:
    """
    Validate research field.
    
    Args:
        field: The research field to validate
        
    Returns:
        The validated research field
        
    Raises:
        ValidationError: If the field is invalid
    """
    if not isinstance(field, str):
        raise ValidationError("research_field", field, "string")
    
    field = field.strip()
    
    if not field:
        raise ValidationError("research_field", field, "non-empty string")
    
    if field not in VALID_RESEARCH_FIELDS:
        logger.warning(f"Unknown research field: {field}. Allowing but logging for review.")
    
    return field

def validate_paper_status(status: str) -> str:
    """
    Validate paper status.
    
    Args:
        status: The paper status to validate
        
    Returns:
        The validated paper status
        
    Raises:
        ValidationError: If the status is invalid
    """
    if not isinstance(status, str):
        raise ValidationError("paper_status", status, "string")
    
    status = status.strip().lower()
    
    valid_statuses = {"draft", "submitted", "in_review", "published", "rejected"}
    
    if status not in valid_statuses:
        raise ValidationError("paper_status", status, f"one of {valid_statuses}")
    
    return status

def validate_simulation_parameters(num_rounds: Union[int, str], interactions_per_round: Union[int, str]) -> tuple[int, int]:
    """
    Validate simulation parameters.
    
    Args:
        num_rounds: Number of simulation rounds
        interactions_per_round: Number of interactions per round
        
    Returns:
        Tuple of validated (num_rounds, interactions_per_round)
        
    Raises:
        ValidationError: If parameters are invalid
    """
    try:
        rounds = int(num_rounds)
    except (ValueError, TypeError):
        raise ValidationError("num_rounds", num_rounds, "integer")
    
    try:
        interactions = int(interactions_per_round)
    except (ValueError, TypeError):
        raise ValidationError("interactions_per_round", interactions_per_round, "integer")
    
    if rounds < 1:
        raise ValidationError("num_rounds", rounds, "positive integer")
    
    if rounds > MAX_SIMULATION_ROUNDS:
        raise ValidationError("num_rounds", rounds, f"integer <= {MAX_SIMULATION_ROUNDS}")
    
    if interactions < 1:
        raise ValidationError("interactions_per_round", interactions, "positive integer")
    
    if interactions > MAX_INTERACTIONS_PER_ROUND:
        raise ValidationError("interactions_per_round", interactions, f"integer <= {MAX_INTERACTIONS_PER_ROUND}")
    
    return rounds, interactions

def validate_file_path(file_path: str, must_exist: bool = False) -> str:
    """
    Validate file path.
    
    Args:
        file_path: The file path to validate
        must_exist: Whether the file must already exist
        
    Returns:
        The validated file path
        
    Raises:
        ValidationError: If the file path is invalid
    """
    if not isinstance(file_path, str):
        raise ValidationError("file_path", file_path, "string")
    
    file_path = file_path.strip()
    
    if not file_path:
        raise ValidationError("file_path", file_path, "non-empty string")
    
    # Check for directory traversal attempts
    if ".." in file_path or file_path.startswith("/"):
        raise ValidationError("file_path", file_path, "safe relative path")
    
    if must_exist:
        import os
        if not os.path.exists(file_path):
            raise ValidationError("file_path", file_path, "existing file")
    
    return file_path

def validate_paper_data(paper_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate complete paper data dictionary.
    
    Args:
        paper_data: Dictionary containing paper data
        
    Returns:
        Validated paper data dictionary
        
    Raises:
        ValidationError: If any field is invalid
    """
    if not isinstance(paper_data, dict):
        raise ValidationError("paper_data", paper_data, "dictionary")
    
    validated_data = {}
    
    # Required fields
    if "title" in paper_data:
        validated_data["title"] = validate_paper_title(paper_data["title"])
    
    if "authors" in paper_data:
        validated_data["authors"] = validate_authors_list(paper_data["authors"])
    
    if "field" in paper_data:
        validated_data["field"] = validate_research_field(paper_data["field"])
    
    # Optional fields
    if "abstract" in paper_data:
        validated_data["abstract"] = validate_paper_abstract(paper_data["abstract"])
    
    if "status" in paper_data:
        validated_data["status"] = validate_paper_status(paper_data["status"])
    
    if "owner_id" in paper_data:
        validated_data["owner_id"] = validate_researcher_id(paper_data["owner_id"])
    
    # Copy other fields as-is (keywords, content, etc.)
    for key, value in paper_data.items():
        if key not in validated_data:
            validated_data[key] = value
    
    return validated_data

def sanitize_string(input_str: str, max_length: int = 1000) -> str:
    """
    Sanitize string input by removing potentially harmful characters.
    
    Args:
        input_str: String to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not isinstance(input_str, str):
        return str(input_str)
    
    # Remove null bytes and control characters except newlines and tabs
    sanitized = ''.join(char for char in input_str if ord(char) >= 32 or char in '\n\t')
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        logger.warning(f"String truncated to {max_length} characters")
    
    return sanitized.strip()