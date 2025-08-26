"""
Collaboration Network Tracking System

This module implements the CollaborationNetwork class to track co-author relationships,
identify collaborators within a 3-year window, detect conflicts of interest for reviewer
assignment, and provide comprehensive collaboration network functionality.
"""

import json
import uuid
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from src.core.exceptions import ValidationError, NetworkError
from src.core.logging_config import get_logger
from src.data.enhanced_models import EnhancedResearcher, PublicationRecord


logger = get_logger(__name__)


@dataclass
class CollaborationRecord:
    """Record of a collaboration between researchers."""
    researcher_1_id: str
    researcher_2_id: str
    paper_id: str
    paper_title: str
    collaboration_date: date
    venue: str
    collaboration_type: str = "co-author"  # co-author, advisor-student, etc.
    
    def __post_init__(self):
        """Validate collaboration record."""
        if self.researcher_1_id == self.researcher_2_id:
            raise ValidationError("researcher_ids", "same", "different researcher IDs")


@dataclass
class ConflictOfInterest:
    """Represents a conflict of interest between researchers."""
    reviewer_id: str
    author_id: str
    conflict_type: str  # "co-author", "advisor", "recent-collaborator", "institutional"
    conflict_strength: float  # 0-1 scale (1 = absolute conflict)
    last_collaboration_date: Optional[date] = None
    description: str = ""


class CollaborationNetwork:
    """
    Manages collaboration networks and tracks co-author relationships.
    
    This class implements comprehensive collaboration tracking including:
    - Co-author relationship management
    - 3-year collaboration window tracking
    - Conflict of interest detection
    - Network analysis and metrics
    """
    
    def __init__(self, collaboration_window_years: int = 3):
        """
        Initialize collaboration network.
        
        Args:
            collaboration_window_years: Years to consider for recent collaborations
        """
        self.collaboration_window_years = collaboration_window_years
        self.collaboration_records: List[CollaborationRecord] = []
        self.researcher_collaborations: Dict[str, Set[str]] = defaultdict(set)
        self.advisor_relationships: Dict[str, Set[str]] = defaultdict(set)  # advisor_id -> {student_ids}
        self.institutional_affiliations: Dict[str, Set[str]] = defaultdict(set)  # institution -> {researcher_ids}
        
        logger.info(f"Initialized CollaborationNetwork with {collaboration_window_years}-year window")
    
    def add_collaboration(self, researcher_1_id: str, researcher_2_id: str, 
                         paper_id: str, paper_title: str, collaboration_date: date,
                         venue: str, collaboration_type: str = "co-author") -> CollaborationRecord:
        """
        Add a collaboration record between two researchers.
        
        Args:
            researcher_1_id: ID of first researcher
            researcher_2_id: ID of second researcher
            paper_id: ID of the collaborative paper
            paper_title: Title of the paper
            collaboration_date: Date of collaboration
            venue: Publication venue
            collaboration_type: Type of collaboration
            
        Returns:
            CollaborationRecord: The created collaboration record
            
        Raises:
            ValidationError: If researcher IDs are the same
        """
        try:
            collaboration = CollaborationRecord(
                researcher_1_id=researcher_1_id,
                researcher_2_id=researcher_2_id,
                paper_id=paper_id,
                paper_title=paper_title,
                collaboration_date=collaboration_date,
                venue=venue,
                collaboration_type=collaboration_type
            )
            
            self.collaboration_records.append(collaboration)
            
            # Update bidirectional collaboration tracking
            self.researcher_collaborations[researcher_1_id].add(researcher_2_id)
            self.researcher_collaborations[researcher_2_id].add(researcher_1_id)
            
            logger.debug(f"Added collaboration between {researcher_1_id} and {researcher_2_id} "
                        f"for paper {paper_id}")
            
            return collaboration
            
        except Exception as e:
            logger.error(f"Failed to add collaboration: {e}")
            raise NetworkError(f"Failed to add collaboration: {e}")
    
    def add_advisor_relationship(self, advisor_id: str, student_id: str):
        """
        Add an advisor-student relationship.
        
        Args:
            advisor_id: ID of the advisor
            student_id: ID of the student
        """
        self.advisor_relationships[advisor_id].add(student_id)
        logger.debug(f"Added advisor relationship: {advisor_id} -> {student_id}")
    
    def add_institutional_affiliation(self, researcher_id: str, institution: str):
        """
        Add institutional affiliation for a researcher.
        
        Args:
            researcher_id: ID of the researcher
            institution: Name of the institution
        """
        self.institutional_affiliations[institution].add(researcher_id)
        logger.debug(f"Added institutional affiliation: {researcher_id} -> {institution}")
    
    def get_collaborators_within_window(self, researcher_id: str, 
                                      reference_date: Optional[date] = None) -> Set[str]:
        """
        Get all collaborators of a researcher within the collaboration window.
        
        Args:
            researcher_id: ID of the researcher
            reference_date: Reference date for window calculation (default: today)
            
        Returns:
            Set[str]: Set of collaborator IDs within the window
        """
        if reference_date is None:
            reference_date = date.today()
        
        window_start = reference_date - timedelta(days=365 * self.collaboration_window_years)
        
        recent_collaborators = set()
        
        for record in self.collaboration_records:
            if record.collaboration_date >= window_start:
                if record.researcher_1_id == researcher_id:
                    recent_collaborators.add(record.researcher_2_id)
                elif record.researcher_2_id == researcher_id:
                    recent_collaborators.add(record.researcher_1_id)
        
        logger.debug(f"Found {len(recent_collaborators)} recent collaborators for {researcher_id}")
        return recent_collaborators
    
    def detect_conflicts_of_interest(self, paper_authors: List[str], 
                                   potential_reviewer: str,
                                   reference_date: Optional[date] = None) -> List[ConflictOfInterest]:
        """
        Detect conflicts of interest between a potential reviewer and paper authors.
        
        Args:
            paper_authors: List of paper author IDs
            potential_reviewer: ID of potential reviewer
            reference_date: Reference date for conflict detection
            
        Returns:
            List[ConflictOfInterest]: List of detected conflicts
        """
        if reference_date is None:
            reference_date = date.today()
        
        conflicts = []
        
        for author_id in paper_authors:
            # Check if reviewer is an author
            if potential_reviewer == author_id:
                conflicts.append(ConflictOfInterest(
                    reviewer_id=potential_reviewer,
                    author_id=author_id,
                    conflict_type="self-review",
                    conflict_strength=1.0,
                    description="Reviewer is an author of the paper"
                ))
                continue
            
            # Check advisor relationships
            if (author_id in self.advisor_relationships.get(potential_reviewer, set()) or
                potential_reviewer in self.advisor_relationships.get(author_id, set())):
                conflicts.append(ConflictOfInterest(
                    reviewer_id=potential_reviewer,
                    author_id=author_id,
                    conflict_type="advisor",
                    conflict_strength=1.0,
                    description="Advisor-student relationship"
                ))
            
            # Check recent collaborations
            recent_collaborators = self.get_collaborators_within_window(potential_reviewer, reference_date)
            if author_id in recent_collaborators:
                # Find the most recent collaboration
                most_recent_date = None
                for record in self.collaboration_records:
                    if ((record.researcher_1_id == potential_reviewer and record.researcher_2_id == author_id) or
                        (record.researcher_2_id == potential_reviewer and record.researcher_1_id == author_id)):
                        if most_recent_date is None or record.collaboration_date > most_recent_date:
                            most_recent_date = record.collaboration_date
                
                # Calculate conflict strength based on recency
                days_since = (reference_date - most_recent_date).days if most_recent_date else 0
                conflict_strength = max(0.1, 1.0 - (days_since / (365 * self.collaboration_window_years)))
                
                conflicts.append(ConflictOfInterest(
                    reviewer_id=potential_reviewer,
                    author_id=author_id,
                    conflict_type="recent-collaborator",
                    conflict_strength=conflict_strength,
                    last_collaboration_date=most_recent_date,
                    description=f"Recent collaboration within {self.collaboration_window_years} years"
                ))
            
            # Check institutional conflicts (same institution)
            reviewer_institutions = {inst for inst, researchers in self.institutional_affiliations.items()
                                   if potential_reviewer in researchers}
            author_institutions = {inst for inst, researchers in self.institutional_affiliations.items()
                                 if author_id in researchers}
            
            common_institutions = reviewer_institutions.intersection(author_institutions)
            if common_institutions:
                conflicts.append(ConflictOfInterest(
                    reviewer_id=potential_reviewer,
                    author_id=author_id,
                    conflict_type="institutional",
                    conflict_strength=0.3,  # Lower strength for institutional conflicts
                    description=f"Same institution: {', '.join(common_institutions)}"
                ))
        
        logger.debug(f"Detected {len(conflicts)} conflicts for reviewer {potential_reviewer}")
        return conflicts
    
    def has_conflict_of_interest(self, paper_authors: List[str], 
                               potential_reviewer: str,
                               min_conflict_strength: float = 0.5,
                               reference_date: Optional[date] = None) -> bool:
        """
        Check if there's a significant conflict of interest.
        
        Args:
            paper_authors: List of paper author IDs
            potential_reviewer: ID of potential reviewer
            min_conflict_strength: Minimum conflict strength to consider significant
            reference_date: Reference date for conflict detection
            
        Returns:
            bool: True if there's a significant conflict
        """
        conflicts = self.detect_conflicts_of_interest(paper_authors, potential_reviewer, reference_date)
        return any(conflict.conflict_strength >= min_conflict_strength for conflict in conflicts)
    
    def get_collaboration_history(self, researcher_id: str) -> List[CollaborationRecord]:
        """
        Get all collaboration history for a researcher.
        
        Args:
            researcher_id: ID of the researcher
            
        Returns:
            List[CollaborationRecord]: List of collaboration records
        """
        history = []
        for record in self.collaboration_records:
            if record.researcher_1_id == researcher_id or record.researcher_2_id == researcher_id:
                history.append(record)
        
        # Sort by date (most recent first)
        history.sort(key=lambda x: x.collaboration_date, reverse=True)
        return history
    
    def get_collaboration_strength(self, researcher_1_id: str, researcher_2_id: str) -> float:
        """
        Calculate collaboration strength between two researchers.
        
        Args:
            researcher_1_id: ID of first researcher
            researcher_2_id: ID of second researcher
            
        Returns:
            float: Collaboration strength (0-1 scale)
        """
        collaborations = []
        for record in self.collaboration_records:
            if ((record.researcher_1_id == researcher_1_id and record.researcher_2_id == researcher_2_id) or
                (record.researcher_2_id == researcher_1_id and record.researcher_1_id == researcher_2_id)):
                collaborations.append(record)
        
        if not collaborations:
            return 0.0
        
        # Base strength from number of collaborations
        base_strength = min(1.0, len(collaborations) / 5.0)  # Max at 5 collaborations
        
        # Recency bonus
        most_recent = max(collaborations, key=lambda x: x.collaboration_date)
        days_since = (date.today() - most_recent.collaboration_date).days
        recency_factor = max(0.1, 1.0 - (days_since / (365 * 5)))  # 5-year decay
        
        return base_strength * recency_factor
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """
        Get network statistics and metrics.
        
        Returns:
            Dict[str, Any]: Network statistics
        """
        total_researchers = len(self.researcher_collaborations)
        total_collaborations = len(self.collaboration_records)
        
        # Calculate average collaborations per researcher
        if total_researchers > 0:
            avg_collaborations = sum(len(collabs) for collabs in self.researcher_collaborations.values()) / total_researchers
        else:
            avg_collaborations = 0.0
        
        # Find most collaborative researchers
        most_collaborative = sorted(
            self.researcher_collaborations.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]
        
        # Calculate collaboration types distribution
        type_distribution = defaultdict(int)
        for record in self.collaboration_records:
            type_distribution[record.collaboration_type] += 1
        
        return {
            "total_researchers": total_researchers,
            "total_collaborations": total_collaborations,
            "average_collaborations_per_researcher": avg_collaborations,
            "most_collaborative_researchers": most_collaborative,
            "collaboration_type_distribution": dict(type_distribution),
            "total_advisor_relationships": sum(len(students) for students in self.advisor_relationships.values()),
            "total_institutions": len(self.institutional_affiliations)
        }
    
    def build_network_from_researchers(self, researchers: List[EnhancedResearcher]):
        """
        Build collaboration network from researcher publication histories.
        
        Args:
            researchers: List of enhanced researchers
        """
        logger.info(f"Building collaboration network from {len(researchers)} researchers")
        
        # Create researcher lookup
        researcher_lookup = {r.id: r for r in researchers}
        
        # Track papers by researcher
        papers_by_researcher = defaultdict(list)
        for researcher in researchers:
            for pub in researcher.publication_history:
                papers_by_researcher[researcher.id].append(pub)
        
        # Find collaborations by matching papers
        processed_pairs = set()
        
        for researcher in researchers:
            # Add institutional affiliations
            for affiliation in researcher.institutional_affiliations:
                self.add_institutional_affiliation(researcher.id, affiliation)
            
            # Add advisor relationships
            for advisor_id in researcher.advisor_relationships:
                self.add_advisor_relationship(advisor_id, researcher.id)
            
            # Process existing collaboration network
            for collaborator_id in researcher.collaboration_network:
                if collaborator_id in researcher_lookup:
                    pair = tuple(sorted([researcher.id, collaborator_id]))
                    if pair not in processed_pairs:
                        # Find a common paper (simplified - use most recent)
                        researcher_papers = {p.paper_id: p for p in researcher.publication_history}
                        collaborator_papers = {p.paper_id: p for p in researcher_lookup[collaborator_id].publication_history}
                        
                        common_papers = set(researcher_papers.keys()).intersection(set(collaborator_papers.keys()))
                        if common_papers:
                            # Use most recent common paper
                            most_recent_paper_id = max(common_papers, 
                                                     key=lambda pid: researcher_papers[pid].year)
                            paper = researcher_papers[most_recent_paper_id]
                            
                            self.add_collaboration(
                                researcher_1_id=researcher.id,
                                researcher_2_id=collaborator_id,
                                paper_id=paper.paper_id,
                                paper_title=paper.title,
                                collaboration_date=date(paper.year, 1, 1),  # Approximate date
                                venue=paper.venue
                            )
                        
                        processed_pairs.add(pair)
        
        logger.info(f"Built network with {len(self.collaboration_records)} collaboration records")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "collaboration_window_years": self.collaboration_window_years,
            "collaboration_records": [
                {
                    "researcher_1_id": record.researcher_1_id,
                    "researcher_2_id": record.researcher_2_id,
                    "paper_id": record.paper_id,
                    "paper_title": record.paper_title,
                    "collaboration_date": record.collaboration_date.isoformat(),
                    "venue": record.venue,
                    "collaboration_type": record.collaboration_type
                }
                for record in self.collaboration_records
            ],
            "advisor_relationships": {
                advisor_id: list(students) 
                for advisor_id, students in self.advisor_relationships.items()
            },
            "institutional_affiliations": {
                institution: list(researchers)
                for institution, researchers in self.institutional_affiliations.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CollaborationNetwork':
        """Create from dictionary."""
        network = cls(collaboration_window_years=data.get("collaboration_window_years", 3))
        
        # Load collaboration records
        for record_data in data.get("collaboration_records", []):
            record = CollaborationRecord(
                researcher_1_id=record_data["researcher_1_id"],
                researcher_2_id=record_data["researcher_2_id"],
                paper_id=record_data["paper_id"],
                paper_title=record_data["paper_title"],
                collaboration_date=date.fromisoformat(record_data["collaboration_date"]),
                venue=record_data["venue"],
                collaboration_type=record_data.get("collaboration_type", "co-author")
            )
            network.collaboration_records.append(record)
            
            # Update tracking dictionaries
            network.researcher_collaborations[record.researcher_1_id].add(record.researcher_2_id)
            network.researcher_collaborations[record.researcher_2_id].add(record.researcher_1_id)
        
        # Load advisor relationships
        for advisor_id, students in data.get("advisor_relationships", {}).items():
            network.advisor_relationships[advisor_id] = set(students)
        
        # Load institutional affiliations
        for institution, researchers in data.get("institutional_affiliations", {}).items():
            network.institutional_affiliations[institution] = set(researchers)
        
        return network


