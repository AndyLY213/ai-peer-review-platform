"""
Conference Community Modeling System

This module implements the ConferenceCommunity class for regular attendee networks,
clique formation modeling, community effects, and community-based reviewer selection
preferences.
"""

import json
import uuid
from datetime import datetime, date
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import random
import math

from src.core.exceptions import ValidationError, NetworkError
from src.core.logging_config import get_logger
from src.data.enhanced_models import EnhancedResearcher, EnhancedVenue


logger = get_logger(__name__)


@dataclass
class AttendanceRecord:
    """Record of a researcher's attendance at a conference."""
    researcher_id: str
    venue_id: str
    year: int
    role: str = "attendee"  # attendee, presenter, reviewer, organizer, keynote
    presentation_count: int = 0
    networking_activity: float = 0.5  # 0-1 scale of networking activity
    
    def __post_init__(self):
        """Validate attendance record."""
        if self.year < 1950 or self.year > 2030:
            raise ValidationError("year", self.year, "year between 1950 and 2030")
        if not (0.0 <= self.networking_activity <= 1.0):
            raise ValidationError("networking_activity", self.networking_activity, "value between 0.0 and 1.0")


@dataclass
class CommunityClique:
    """Represents a clique within a conference community."""
    clique_id: str
    venue_id: str
    member_ids: Set[str]
    formation_year: int
    clique_strength: float  # 0-1 scale of how tight-knit the clique is
    influence_score: float  # 0-1 scale of clique's influence in the community
    research_focus: str = ""
    
    def __post_init__(self):
        """Validate clique data."""
        if len(self.member_ids) < 2:
            raise ValidationError("member_ids", len(self.member_ids), "at least 2 members")
        if not (0.0 <= self.clique_strength <= 1.0):
            raise ValidationError("clique_strength", self.clique_strength, "value between 0.0 and 1.0")
        if not (0.0 <= self.influence_score <= 1.0):
            raise ValidationError("influence_score", self.influence_score, "value between 0.0 and 1.0")


@dataclass
class CommunityInfluence:
    """Represents community influence on reviewer selection."""
    venue_id: str
    researcher_id: str
    community_standing: float  # 0-1 scale of standing in the community
    clique_memberships: List[str]  # List of clique IDs
    attendance_years: List[int]
    networking_score: float  # 0-1 scale based on networking activity
    influence_multiplier: float  # Multiplier for reviewer selection preference


class ConferenceCommunity:
    """
    Manages conference communities and models regular attendee networks.
    
    This class implements comprehensive community modeling including:
    - Conference attendance tracking
    - Clique formation and evolution
    - Community influence on reviewer selection
    - Networking effects and community dynamics
    """
    
    def __init__(self):
        """Initialize conference community system."""
        self.attendance_records: List[AttendanceRecord] = []
        self.venue_attendees: Dict[str, Dict[int, Set[str]]] = defaultdict(lambda: defaultdict(set))  # venue -> year -> attendees
        self.researcher_venues: Dict[str, Set[str]] = defaultdict(set)  # researcher -> venues attended
        self.community_cliques: Dict[str, List[CommunityClique]] = defaultdict(list)  # venue -> cliques
        self.clique_memberships: Dict[str, List[str]] = defaultdict(list)  # researcher -> clique_ids
        self.community_influences: Dict[str, CommunityInfluence] = {}  # researcher -> influence
        
        logger.info("Initialized ConferenceCommunity system")
    
    def add_attendance_record(self, researcher_id: str, venue_id: str, year: int,
                            role: str = "attendee", presentation_count: int = 0,
                            networking_activity: float = 0.5) -> AttendanceRecord:
        """
        Add an attendance record for a researcher at a conference.
        
        Args:
            researcher_id: ID of the researcher
            venue_id: ID of the venue/conference
            year: Year of attendance
            role: Role at the conference
            presentation_count: Number of presentations given
            networking_activity: Level of networking activity (0-1)
            
        Returns:
            AttendanceRecord: The created attendance record
        """
        try:
            record = AttendanceRecord(
                researcher_id=researcher_id,
                venue_id=venue_id,
                year=year,
                role=role,
                presentation_count=presentation_count,
                networking_activity=networking_activity
            )
            
            self.attendance_records.append(record)
            self.venue_attendees[venue_id][year].add(researcher_id)
            self.researcher_venues[researcher_id].add(venue_id)
            
            logger.debug(f"Added attendance record for {researcher_id} at {venue_id} in {year}")
            
            return record
            
        except Exception as e:
            logger.error(f"Failed to add attendance record: {e}")
            raise NetworkError(f"Failed to add attendance record: {e}")
    
    def get_venue_attendees(self, venue_id: str, year: Optional[int] = None) -> Set[str]:
        """
        Get attendees for a venue, optionally filtered by year.
        
        Args:
            venue_id: ID of the venue
            year: Optional year filter
            
        Returns:
            Set[str]: Set of researcher IDs who attended
        """
        if year is not None:
            return self.venue_attendees.get(venue_id, {}).get(year, set())
        else:
            # Return all attendees across all years
            all_attendees = set()
            for year_attendees in self.venue_attendees.get(venue_id, {}).values():
                all_attendees.update(year_attendees)
            return all_attendees
    
    def get_researcher_attendance_history(self, researcher_id: str) -> List[AttendanceRecord]:
        """
        Get attendance history for a researcher.
        
        Args:
            researcher_id: ID of the researcher
            
        Returns:
            List[AttendanceRecord]: List of attendance records
        """
        history = [record for record in self.attendance_records 
                  if record.researcher_id == researcher_id]
        history.sort(key=lambda x: (x.year, x.venue_id))
        return history
    
    def calculate_attendance_overlap(self, researcher1_id: str, researcher2_id: str) -> Dict[str, Any]:
        """
        Calculate attendance overlap between two researchers.
        
        Args:
            researcher1_id: ID of first researcher
            researcher2_id: ID of second researcher
            
        Returns:
            Dict[str, Any]: Overlap information
        """
        history1 = self.get_researcher_attendance_history(researcher1_id)
        history2 = self.get_researcher_attendance_history(researcher2_id)
        
        # Create sets of (venue, year) tuples
        attendance1 = {(record.venue_id, record.year) for record in history1}
        attendance2 = {(record.venue_id, record.year) for record in history2}
        
        overlap = attendance1.intersection(attendance2)
        
        # Calculate overlap strength based on frequency and recency
        overlap_strength = 0.0
        if overlap:
            current_year = date.today().year
            for venue_id, year in overlap:
                # More recent overlaps have higher weight
                recency_weight = max(0.1, 1.0 - (current_year - year) / 10.0)
                overlap_strength += recency_weight
        
        return {
            "total_overlaps": len(overlap),
            "overlap_events": list(overlap),
            "overlap_strength": min(1.0, overlap_strength / 5.0),  # Normalize to 0-1
            "common_venues": len(set(record.venue_id for record in history1).intersection(
                                set(record.venue_id for record in history2)))
        }
    
    def form_cliques(self, venue_id: str, min_clique_size: int = 3, 
                    min_overlap_threshold: float = 0.3) -> List[CommunityClique]:
        """
        Form cliques based on attendance patterns and networking activity.
        
        Args:
            venue_id: ID of the venue to form cliques for
            min_clique_size: Minimum size for a clique
            min_overlap_threshold: Minimum overlap threshold for clique membership
            
        Returns:
            List[CommunityClique]: List of formed cliques
        """
        venue_attendees = self.get_venue_attendees(venue_id)
        
        if len(venue_attendees) < min_clique_size:
            return []
        
        # Calculate pairwise overlaps
        attendee_list = list(venue_attendees)
        overlap_matrix = {}
        
        for i, researcher1 in enumerate(attendee_list):
            for j, researcher2 in enumerate(attendee_list[i+1:], i+1):
                overlap = self.calculate_attendance_overlap(researcher1, researcher2)
                overlap_matrix[(researcher1, researcher2)] = overlap["overlap_strength"]
        
        # Form cliques using a greedy approach
        cliques = []
        used_researchers = set()
        
        # Sort researcher pairs by overlap strength
        sorted_pairs = sorted(overlap_matrix.items(), key=lambda x: x[1], reverse=True)
        
        for (researcher1, researcher2), strength in sorted_pairs:
            if strength < min_overlap_threshold:
                break
            
            if researcher1 in used_researchers or researcher2 in used_researchers:
                continue
            
            # Start a new clique
            potential_clique = {researcher1, researcher2}
            
            # Try to add more members
            for researcher3 in attendee_list:
                if researcher3 in potential_clique or researcher3 in used_researchers:
                    continue
                
                # Check if researcher3 has good overlap with existing clique members
                avg_overlap = sum(overlap_matrix.get((min(researcher3, member), max(researcher3, member)), 0)
                                for member in potential_clique) / len(potential_clique)
                
                if avg_overlap >= min_overlap_threshold:
                    potential_clique.add(researcher3)
            
            if len(potential_clique) >= min_clique_size:
                # Calculate clique strength and influence
                clique_strength = sum(overlap_matrix.get((min(r1, r2), max(r1, r2)), 0)
                                    for r1 in potential_clique for r2 in potential_clique if r1 != r2)
                clique_strength /= (len(potential_clique) * (len(potential_clique) - 1))
                
                # Influence based on member seniority and activity
                influence_score = min(1.0, len(potential_clique) / 10.0)  # Larger cliques have more influence
                
                clique = CommunityClique(
                    clique_id=str(uuid.uuid4()),
                    venue_id=venue_id,
                    member_ids=potential_clique,
                    formation_year=date.today().year,
                    clique_strength=clique_strength,
                    influence_score=influence_score,
                    research_focus=f"Community at {venue_id}"
                )
                
                cliques.append(clique)
                used_researchers.update(potential_clique)
                
                # Update clique memberships
                for member_id in potential_clique:
                    self.clique_memberships[member_id].append(clique.clique_id)
        
        # Store cliques for the venue
        self.community_cliques[venue_id].extend(cliques)
        
        logger.info(f"Formed {len(cliques)} cliques for venue {venue_id}")
        return cliques
    
    def calculate_community_influence(self, researcher_id: str, venue_id: str) -> CommunityInfluence:
        """
        Calculate a researcher's influence within a conference community.
        
        Args:
            researcher_id: ID of the researcher
            venue_id: ID of the venue
            
        Returns:
            CommunityInfluence: Community influence information
        """
        # Get attendance history for this venue
        venue_attendance = [record for record in self.attendance_records
                          if record.researcher_id == researcher_id and record.venue_id == venue_id]
        
        if not venue_attendance:
            return CommunityInfluence(
                venue_id=venue_id,
                researcher_id=researcher_id,
                community_standing=0.0,
                clique_memberships=[],
                attendance_years=[],
                networking_score=0.0,
                influence_multiplier=1.0
            )
        
        # Calculate community standing
        attendance_years = [record.year for record in venue_attendance]
        attendance_span = max(attendance_years) - min(attendance_years) + 1
        attendance_frequency = len(attendance_years) / max(1, attendance_span)
        
        # Factor in roles and presentations
        role_weights = {"keynote": 1.0, "organizer": 0.8, "reviewer": 0.6, "presenter": 0.4, "attendee": 0.2}
        avg_role_weight = sum(role_weights.get(record.role, 0.2) for record in venue_attendance) / len(venue_attendance)
        
        total_presentations = sum(record.presentation_count for record in venue_attendance)
        presentation_factor = min(1.0, total_presentations / 10.0)
        
        # Calculate networking score
        avg_networking = sum(record.networking_activity for record in venue_attendance) / len(venue_attendance)
        
        # Calculate community standing
        community_standing = (attendance_frequency * 0.4 + avg_role_weight * 0.3 + 
                            presentation_factor * 0.2 + avg_networking * 0.1)
        community_standing = min(1.0, community_standing)
        
        # Get clique memberships for this venue
        venue_cliques = [clique.clique_id for clique in self.community_cliques.get(venue_id, [])
                        if researcher_id in clique.member_ids]
        
        # Calculate influence multiplier
        base_multiplier = 1.0 + (community_standing * 0.5)  # Up to 1.5x multiplier
        clique_bonus = len(venue_cliques) * 0.1  # 0.1x bonus per clique membership
        influence_multiplier = base_multiplier + clique_bonus
        
        influence = CommunityInfluence(
            venue_id=venue_id,
            researcher_id=researcher_id,
            community_standing=community_standing,
            clique_memberships=venue_cliques,
            attendance_years=attendance_years,
            networking_score=avg_networking,
            influence_multiplier=influence_multiplier
        )
        
        # Cache the influence
        self.community_influences[f"{researcher_id}_{venue_id}"] = influence
        
        return influence
    
    def get_reviewer_selection_preferences(self, venue_id: str, 
                                         candidate_reviewers: List[str]) -> Dict[str, float]:
        """
        Get community-based reviewer selection preferences.
        
        Args:
            venue_id: ID of the venue
            candidate_reviewers: List of candidate reviewer IDs
            
        Returns:
            Dict[str, float]: Mapping of reviewer ID to preference score (0-1)
        """
        preferences = {}
        
        for reviewer_id in candidate_reviewers:
            influence = self.calculate_community_influence(reviewer_id, venue_id)
            
            # Base preference from community standing
            base_preference = influence.community_standing
            
            # Bonus for clique memberships (community insiders preferred)
            clique_bonus = len(influence.clique_memberships) * 0.1
            
            # Bonus for regular attendance
            attendance_bonus = min(0.3, len(influence.attendance_years) / 10.0)
            
            # Networking bonus
            networking_bonus = influence.networking_score * 0.2
            
            total_preference = base_preference + clique_bonus + attendance_bonus + networking_bonus
            preferences[reviewer_id] = min(1.0, total_preference)
        
        return preferences
    
    def detect_community_effects(self, venue_id: str) -> Dict[str, Any]:
        """
        Detect various community effects within a venue.
        
        Args:
            venue_id: ID of the venue
            
        Returns:
            Dict[str, Any]: Detected community effects
        """
        venue_cliques = self.community_cliques.get(venue_id, [])
        venue_attendees = self.get_venue_attendees(venue_id)
        
        effects = {
            "total_cliques": len(venue_cliques),
            "clique_coverage": 0.0,
            "average_clique_size": 0.0,
            "influence_concentration": 0.0,
            "networking_patterns": {},
            "attendance_loyalty": 0.0
        }
        
        if not venue_attendees:
            return effects
        
        # Calculate clique coverage
        clique_members = set()
        for clique in venue_cliques:
            clique_members.update(clique.member_ids)
        effects["clique_coverage"] = len(clique_members) / len(venue_attendees)
        
        # Calculate average clique size
        if venue_cliques:
            effects["average_clique_size"] = sum(len(clique.member_ids) for clique in venue_cliques) / len(venue_cliques)
        
        # Calculate influence concentration (how concentrated influence is among few researchers)
        influences = [self.calculate_community_influence(researcher_id, venue_id).community_standing
                     for researcher_id in venue_attendees]
        if influences:
            # Gini coefficient for influence distribution
            sorted_influences = sorted(influences)
            n = len(sorted_influences)
            cumsum = sum((i + 1) * influence for i, influence in enumerate(sorted_influences))
            effects["influence_concentration"] = (2 * cumsum) / (n * sum(sorted_influences)) - (n + 1) / n
        
        # Calculate attendance loyalty (average years of attendance)
        attendance_spans = []
        for researcher_id in venue_attendees:
            history = [record for record in self.attendance_records
                      if record.researcher_id == researcher_id and record.venue_id == venue_id]
            if history:
                years = [record.year for record in history]
                attendance_spans.append(len(years))
        
        if attendance_spans:
            effects["attendance_loyalty"] = sum(attendance_spans) / len(attendance_spans)
        
        return effects
    
    def get_community_statistics(self) -> Dict[str, Any]:
        """
        Get overall community statistics across all venues.
        
        Returns:
            Dict[str, Any]: Community statistics
        """
        total_venues = len(self.venue_attendees)
        total_attendees = len(self.researcher_venues)
        total_attendance_records = len(self.attendance_records)
        total_cliques = sum(len(cliques) for cliques in self.community_cliques.values())
        
        # Calculate average attendance per researcher
        if total_attendees > 0:
            avg_attendance = total_attendance_records / total_attendees
        else:
            avg_attendance = 0.0
        
        # Find most active venues
        venue_activity = [(venue_id, len(attendees_by_year))
                         for venue_id, attendees_by_year in self.venue_attendees.items()]
        venue_activity.sort(key=lambda x: x[1], reverse=True)
        
        # Find most connected researchers
        researcher_connections = [(researcher_id, len(venues))
                                for researcher_id, venues in self.researcher_venues.items()]
        researcher_connections.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "total_venues": total_venues,
            "total_attendees": total_attendees,
            "total_attendance_records": total_attendance_records,
            "total_cliques": total_cliques,
            "average_attendance_per_researcher": avg_attendance,
            "most_active_venues": venue_activity[:5],
            "most_connected_researchers": researcher_connections[:5],
            "clique_distribution": {venue_id: len(cliques) 
                                  for venue_id, cliques in self.community_cliques.items()}
        }
    
    def build_communities_from_researchers(self, researchers: List[EnhancedResearcher],
                                         venues: List[EnhancedVenue]):
        """
        Build conference communities from researcher and venue data.
        
        Args:
            researchers: List of enhanced researchers
            venues: List of enhanced venues
        """
        logger.info(f"Building communities from {len(researchers)} researchers and {len(venues)} venues")
        
        current_year = date.today().year
        
        # Simulate attendance patterns based on researcher specialties and venue fields
        for researcher in researchers:
            # Determine venues this researcher would likely attend
            relevant_venues = [venue for venue in venues 
                             if venue.field.lower() in researcher.specialty.lower() or
                                researcher.specialty.lower() in venue.field.lower()]
            
            if not relevant_venues:
                # If no direct match, attend some general venues
                relevant_venues = venues[:3]  # Attend top 3 venues
            
            # Simulate attendance history (last 5 years)
            for venue in relevant_venues[:2]:  # Limit to 2 venues per researcher
                attendance_probability = 0.3 + (researcher.reputation_score * 0.4)  # Higher reputation = more attendance
                
                for year in range(current_year - 5, current_year):
                    if random.random() < attendance_probability:
                        # Determine role based on seniority
                        role_probabilities = {
                            "attendee": 0.6,
                            "presenter": 0.25,
                            "reviewer": 0.1,
                            "organizer": 0.04,
                            "keynote": 0.01
                        }
                        
                        # Adjust probabilities based on seniority
                        if researcher.level.value in ["Full Prof", "Emeritus"]:
                            role_probabilities["keynote"] *= 5
                            role_probabilities["organizer"] *= 3
                            role_probabilities["reviewer"] *= 2
                        elif researcher.level.value in ["Associate Prof"]:
                            role_probabilities["organizer"] *= 2
                            role_probabilities["reviewer"] *= 1.5
                        
                        # Select role
                        role = random.choices(
                            list(role_probabilities.keys()),
                            weights=list(role_probabilities.values())
                        )[0]
                        
                        # Determine presentation count and networking activity
                        presentation_count = 1 if role == "presenter" else 0
                        if role == "keynote":
                            presentation_count = 1
                        
                        networking_activity = random.uniform(0.3, 0.9)
                        if researcher.level.value in ["Full Prof", "Emeritus"]:
                            networking_activity += 0.1  # Senior researchers network more
                        
                        self.add_attendance_record(
                            researcher_id=researcher.id,
                            venue_id=venue.id,
                            year=year,
                            role=role,
                            presentation_count=presentation_count,
                            networking_activity=min(1.0, networking_activity)
                        )
        
        # Form cliques for each venue
        for venue in venues:
            self.form_cliques(venue.id, min_clique_size=3, min_overlap_threshold=0.2)
        
        logger.info(f"Built communities with {len(self.attendance_records)} attendance records")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "attendance_records": [
                {
                    "researcher_id": record.researcher_id,
                    "venue_id": record.venue_id,
                    "year": record.year,
                    "role": record.role,
                    "presentation_count": record.presentation_count,
                    "networking_activity": record.networking_activity
                }
                for record in self.attendance_records
            ],
            "community_cliques": {
                venue_id: [
                    {
                        "clique_id": clique.clique_id,
                        "venue_id": clique.venue_id,
                        "member_ids": list(clique.member_ids),
                        "formation_year": clique.formation_year,
                        "clique_strength": clique.clique_strength,
                        "influence_score": clique.influence_score,
                        "research_focus": clique.research_focus
                    }
                    for clique in cliques
                ]
                for venue_id, cliques in self.community_cliques.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConferenceCommunity':
        """Create from dictionary."""
        community = cls()
        
        # Load attendance records
        for record_data in data.get("attendance_records", []):
            record = AttendanceRecord(
                researcher_id=record_data["researcher_id"],
                venue_id=record_data["venue_id"],
                year=record_data["year"],
                role=record_data.get("role", "attendee"),
                presentation_count=record_data.get("presentation_count", 0),
                networking_activity=record_data.get("networking_activity", 0.5)
            )
            community.attendance_records.append(record)
            
            # Update tracking dictionaries
            community.venue_attendees[record.venue_id][record.year].add(record.researcher_id)
            community.researcher_venues[record.researcher_id].add(record.venue_id)
        
        # Load cliques
        for venue_id, cliques_data in data.get("community_cliques", {}).items():
            for clique_data in cliques_data:
                clique = CommunityClique(
                    clique_id=clique_data["clique_id"],
                    venue_id=clique_data["venue_id"],
                    member_ids=set(clique_data["member_ids"]),
                    formation_year=clique_data["formation_year"],
                    clique_strength=clique_data["clique_strength"],
                    influence_score=clique_data["influence_score"],
                    research_focus=clique_data.get("research_focus", "")
                )
                community.community_cliques[venue_id].append(clique)
                
                # Update clique memberships
                for member_id in clique.member_ids:
                    community.clique_memberships[member_id].append(clique.clique_id)
        
        return community