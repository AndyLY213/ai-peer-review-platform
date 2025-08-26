"""
Network Influence Calculations System

This module implements network distance calculations between researchers,
logic to reduce review weight based on network proximity, and network-based
bias adjustments to review scores.
"""

import json
import uuid
from datetime import datetime, date
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import math

from src.core.exceptions import ValidationError, NetworkError
from src.core.logging_config import get_logger
from src.enhancements.collaboration_network import CollaborationNetwork
from src.enhancements.citation_network import CitationNetwork
from src.enhancements.conference_community import ConferenceCommunity


logger = get_logger(__name__)


@dataclass
class NetworkDistance:
    """Represents network distance between two researchers."""
    researcher_1_id: str
    researcher_2_id: str
    collaboration_distance: float  # 0-1 scale (0 = direct collaboration, 1 = no connection)
    citation_distance: float  # 0-1 scale (0 = direct citation, 1 = no connection)
    community_distance: float  # 0-1 scale (0 = same community, 1 = no overlap)
    overall_distance: float  # Combined distance metric
    connection_strength: float  # Inverse of distance (0-1 scale)
    
    def __post_init__(self):
        """Validate distance values."""
        for field_name in ['collaboration_distance', 'citation_distance', 'community_distance', 'overall_distance', 'connection_strength']:
            value = getattr(self, field_name)
            if not (0.0 <= value <= 1.0):
                raise ValidationError(field_name, value, "value between 0.0 and 1.0")


@dataclass
class NetworkInfluenceEffect:
    """Represents network influence effect on a review."""
    reviewer_id: str
    author_id: str
    influence_type: str  # "proximity-bias", "distance-discount", "community-preference"
    influence_strength: float  # 0-1 scale
    score_adjustment: float  # How much the influence adjusted the score
    weight_adjustment: float  # How much the influence adjusted the review weight
    network_distance: float  # Network distance between reviewer and author
    description: str = ""


class NetworkInfluenceCalculator:
    """
    Calculates network influence effects on peer review processes.
    
    This class implements comprehensive network influence calculations including:
    - Network distance calculations between researchers
    - Review weight adjustments based on network proximity
    - Network-based bias effects on review scores
    - Multi-network integration (collaboration, citation, community)
    """
    
    def __init__(self, collaboration_network: Optional[CollaborationNetwork] = None,
                 citation_network: Optional[CitationNetwork] = None,
                 conference_community: Optional[ConferenceCommunity] = None):
        """
        Initialize network influence calculator.
        
        Args:
            collaboration_network: Collaboration network instance
            citation_network: Citation network instance
            conference_community: Conference community instance
        """
        self.collaboration_network = collaboration_network
        self.citation_network = citation_network
        self.conference_community = conference_community
        
        # Cache for computed distances
        self.distance_cache: Dict[Tuple[str, str], NetworkDistance] = {}
        
        logger.info("Initialized NetworkInfluenceCalculator")
    
    def calculate_collaboration_distance(self, researcher_1_id: str, researcher_2_id: str) -> float:
        """
        Calculate collaboration distance between two researchers.
        
        Args:
            researcher_1_id: ID of first researcher
            researcher_2_id: ID of second researcher
            
        Returns:
            float: Collaboration distance (0-1 scale)
        """
        if not self.collaboration_network:
            return 1.0  # Maximum distance if no network available
        
        # Direct collaboration
        collaboration_strength = self.collaboration_network.get_collaboration_strength(
            researcher_1_id, researcher_2_id
        )
        
        if collaboration_strength > 0:
            return 1.0 - collaboration_strength  # Convert strength to distance
        
        # Check for indirect collaboration (2-hop)
        researcher_1_collaborators = self.collaboration_network.researcher_collaborations.get(researcher_1_id, set())
        researcher_2_collaborators = self.collaboration_network.researcher_collaborations.get(researcher_2_id, set())
        
        # Find common collaborators
        common_collaborators = researcher_1_collaborators.intersection(researcher_2_collaborators)
        
        if common_collaborators:
            # Calculate indirect collaboration strength
            indirect_strength = 0.0
            for collaborator in common_collaborators:
                strength_1 = self.collaboration_network.get_collaboration_strength(researcher_1_id, collaborator)
                strength_2 = self.collaboration_network.get_collaboration_strength(researcher_2_id, collaborator)
                indirect_strength += (strength_1 * strength_2) * 0.5  # Discount for indirect connection
            
            indirect_strength = min(1.0, indirect_strength)
            return 1.0 - indirect_strength
        
        return 1.0  # No collaboration connection
    
    def calculate_citation_distance(self, researcher_1_id: str, researcher_2_id: str) -> float:
        """
        Calculate citation distance between two researchers.
        
        Args:
            researcher_1_id: ID of first researcher
            researcher_2_id: ID of second researcher
            
        Returns:
            float: Citation distance (0-1 scale)
        """
        if not self.citation_network:
            return 1.0  # Maximum distance if no network available
        
        # Direct citation relationship
        citations_1_to_2 = self.citation_network.get_author_citation_relationship(researcher_1_id, researcher_2_id)
        citations_2_to_1 = self.citation_network.get_author_citation_relationship(researcher_2_id, researcher_1_id)
        
        total_citations = citations_1_to_2 + citations_2_to_1
        
        if total_citations > 0:
            # Convert citation count to strength (logarithmic scale)
            citation_strength = min(1.0, math.log(total_citations + 1) / math.log(10))
            return 1.0 - citation_strength
        
        # Check for indirect citation connections (common cited authors)
        researcher_1_papers = self.citation_network.author_papers.get(researcher_1_id, set())
        researcher_2_papers = self.citation_network.author_papers.get(researcher_2_id, set())
        
        # Find papers that cite both researchers' work
        common_citing_papers = set()
        for paper_1 in researcher_1_papers:
            citing_papers_1 = self.citation_network.get_paper_citations(paper_1)
            for paper_2 in researcher_2_papers:
                citing_papers_2 = self.citation_network.get_paper_citations(paper_2)
                common_citing_papers.update(citing_papers_1.intersection(citing_papers_2))
        
        if common_citing_papers:
            # Indirect citation connection strength
            indirect_strength = min(1.0, len(common_citing_papers) / 10.0)
            return 1.0 - (indirect_strength * 0.3)  # Discount for indirect connection
        
        return 1.0  # No citation connection
    
    def calculate_community_distance(self, researcher_1_id: str, researcher_2_id: str,
                                   venue_id: Optional[str] = None) -> float:
        """
        Calculate community distance between two researchers.
        
        Args:
            researcher_1_id: ID of first researcher
            researcher_2_id: ID of second researcher
            venue_id: Optional venue ID to focus on specific community
            
        Returns:
            float: Community distance (0-1 scale)
        """
        if not self.conference_community:
            return 1.0  # Maximum distance if no network available
        
        if venue_id:
            # Calculate distance for specific venue
            overlap = self.conference_community.calculate_attendance_overlap(researcher_1_id, researcher_2_id)
            
            # Check for common venue attendance
            venue_attendees_1 = researcher_1_id in self.conference_community.get_venue_attendees(venue_id)
            venue_attendees_2 = researcher_2_id in self.conference_community.get_venue_attendees(venue_id)
            
            if venue_attendees_1 and venue_attendees_2:
                # Both attend the venue - calculate community closeness
                influence_1 = self.conference_community.calculate_community_influence(researcher_1_id, venue_id)
                influence_2 = self.conference_community.calculate_community_influence(researcher_2_id, venue_id)
                
                # Check for common clique membership
                common_cliques = set(influence_1.clique_memberships).intersection(set(influence_2.clique_memberships))
                
                if common_cliques:
                    return 0.1  # Very close if in same clique
                
                # Calculate distance based on community standing and overlap
                community_closeness = (overlap["overlap_strength"] + 
                                     min(influence_1.community_standing, influence_2.community_standing)) / 2.0
                return 1.0 - community_closeness
            
            elif venue_attendees_1 or venue_attendees_2:
                return 0.7  # One attends, one doesn't - moderate distance
            else:
                return 1.0  # Neither attends - maximum distance
        
        else:
            # Calculate overall community distance across all venues
            overlap = self.conference_community.calculate_attendance_overlap(researcher_1_id, researcher_2_id)
            
            if overlap["total_overlaps"] > 0:
                return 1.0 - overlap["overlap_strength"]
            
            # Check for any common venue attendance
            venues_1 = self.conference_community.researcher_venues.get(researcher_1_id, set())
            venues_2 = self.conference_community.researcher_venues.get(researcher_2_id, set())
            
            common_venues = venues_1.intersection(venues_2)
            
            if common_venues:
                # Calculate average community distance across common venues
                total_distance = 0.0
                for venue in common_venues:
                    total_distance += self.calculate_community_distance(researcher_1_id, researcher_2_id, venue)
                return total_distance / len(common_venues)
            
            return 1.0  # No common venues
    
    def calculate_network_distance(self, researcher_1_id: str, researcher_2_id: str,
                                 venue_id: Optional[str] = None) -> NetworkDistance:
        """
        Calculate overall network distance between two researchers.
        
        Args:
            researcher_1_id: ID of first researcher
            researcher_2_id: ID of second researcher
            venue_id: Optional venue ID for community distance calculation
            
        Returns:
            NetworkDistance: Comprehensive network distance information
        """
        # Check cache first
        cache_key = (researcher_1_id, researcher_2_id, venue_id or "")
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        # Calculate individual distance components
        collaboration_distance = self.calculate_collaboration_distance(researcher_1_id, researcher_2_id)
        citation_distance = self.calculate_citation_distance(researcher_1_id, researcher_2_id)
        community_distance = self.calculate_community_distance(researcher_1_id, researcher_2_id, venue_id)
        
        # Calculate overall distance (weighted average)
        weights = {
            'collaboration': 0.4,  # Highest weight for direct collaboration
            'citation': 0.3,       # Medium weight for citation relationships
            'community': 0.3       # Medium weight for community connections
        }
        
        overall_distance = (collaboration_distance * weights['collaboration'] +
                          citation_distance * weights['citation'] +
                          community_distance * weights['community'])
        
        connection_strength = 1.0 - overall_distance
        
        distance = NetworkDistance(
            researcher_1_id=researcher_1_id,
            researcher_2_id=researcher_2_id,
            collaboration_distance=collaboration_distance,
            citation_distance=citation_distance,
            community_distance=community_distance,
            overall_distance=overall_distance,
            connection_strength=connection_strength
        )
        
        # Cache the result
        self.distance_cache[cache_key] = distance
        
        return distance
    
    def calculate_review_weight_adjustment(self, reviewer_id: str, author_ids: List[str],
                                         venue_id: Optional[str] = None,
                                         base_weight: float = 1.0) -> Tuple[float, List[NetworkInfluenceEffect]]:
        """
        Calculate review weight adjustment based on network proximity.
        
        Args:
            reviewer_id: ID of the reviewer
            author_ids: IDs of the paper authors
            venue_id: Optional venue ID
            base_weight: Base review weight
            
        Returns:
            Tuple[float, List[NetworkInfluenceEffect]]: Adjusted weight and influence effects
        """
        influence_effects = []
        total_weight_adjustment = 0.0
        
        for author_id in author_ids:
            if reviewer_id == author_id:
                # Self-review - maximum weight reduction
                influence_effects.append(NetworkInfluenceEffect(
                    reviewer_id=reviewer_id,
                    author_id=author_id,
                    influence_type="self-review",
                    influence_strength=1.0,
                    score_adjustment=0.0,
                    weight_adjustment=-0.9,  # 90% weight reduction
                    network_distance=0.0,
                    description="Self-review detected"
                ))
                total_weight_adjustment -= 0.9
                continue
            
            # Calculate network distance
            distance = self.calculate_network_distance(reviewer_id, author_id, venue_id)
            
            # Weight adjustment based on network proximity
            if distance.connection_strength > 0.7:
                # Very close connection - significant weight reduction
                weight_reduction = 0.5 * distance.connection_strength
                influence_effects.append(NetworkInfluenceEffect(
                    reviewer_id=reviewer_id,
                    author_id=author_id,
                    influence_type="proximity-bias",
                    influence_strength=distance.connection_strength,
                    score_adjustment=0.0,
                    weight_adjustment=-weight_reduction,
                    network_distance=distance.overall_distance,
                    description=f"Strong network connection (strength: {distance.connection_strength:.2f})"
                ))
                total_weight_adjustment -= weight_reduction
            
            elif distance.connection_strength > 0.4:
                # Moderate connection - moderate weight reduction
                weight_reduction = 0.2 * distance.connection_strength
                influence_effects.append(NetworkInfluenceEffect(
                    reviewer_id=reviewer_id,
                    author_id=author_id,
                    influence_type="proximity-bias",
                    influence_strength=distance.connection_strength,
                    score_adjustment=0.0,
                    weight_adjustment=-weight_reduction,
                    network_distance=distance.overall_distance,
                    description=f"Moderate network connection (strength: {distance.connection_strength:.2f})"
                ))
                total_weight_adjustment -= weight_reduction
        
        # Apply total adjustment to base weight
        adjusted_weight = max(0.1, base_weight + total_weight_adjustment)  # Minimum 10% weight
        
        logger.debug(f"Weight adjustment for reviewer {reviewer_id}: {base_weight} -> {adjusted_weight}")
        
        return adjusted_weight, influence_effects
    
    def calculate_network_bias_adjustment(self, reviewer_id: str, author_ids: List[str],
                                        base_score: float, venue_id: Optional[str] = None) -> Tuple[float, List[NetworkInfluenceEffect]]:
        """
        Calculate network-based bias adjustments to review scores.
        
        Args:
            reviewer_id: ID of the reviewer
            author_ids: IDs of the paper authors
            base_score: Base review score
            venue_id: Optional venue ID
            
        Returns:
            Tuple[float, List[NetworkInfluenceEffect]]: Adjusted score and influence effects
        """
        influence_effects = []
        total_score_adjustment = 0.0
        
        for author_id in author_ids:
            if reviewer_id == author_id:
                # Self-review - strong positive bias
                influence_effects.append(NetworkInfluenceEffect(
                    reviewer_id=reviewer_id,
                    author_id=author_id,
                    influence_type="self-review",
                    influence_strength=1.0,
                    score_adjustment=2.0,
                    weight_adjustment=0.0,
                    network_distance=0.0,
                    description="Self-review positive bias"
                ))
                total_score_adjustment += 2.0
                continue
            
            # Calculate network distance
            distance = self.calculate_network_distance(reviewer_id, author_id, venue_id)
            
            # Score bias based on network connections
            if distance.collaboration_distance < 0.3:
                # Close collaboration - positive bias
                bias_strength = (1.0 - distance.collaboration_distance) * 0.5
                score_adjustment = bias_strength * 1.0  # Up to 1 point positive
                
                influence_effects.append(NetworkInfluenceEffect(
                    reviewer_id=reviewer_id,
                    author_id=author_id,
                    influence_type="collaboration-bias",
                    influence_strength=bias_strength,
                    score_adjustment=score_adjustment,
                    weight_adjustment=0.0,
                    network_distance=distance.overall_distance,
                    description=f"Collaboration bias (distance: {distance.collaboration_distance:.2f})"
                ))
                total_score_adjustment += score_adjustment
            
            if distance.citation_distance < 0.4:
                # Citation relationship - moderate positive bias
                bias_strength = (1.0 - distance.citation_distance) * 0.3
                score_adjustment = bias_strength * 0.5  # Up to 0.5 points positive
                
                influence_effects.append(NetworkInfluenceEffect(
                    reviewer_id=reviewer_id,
                    author_id=author_id,
                    influence_type="citation-bias",
                    influence_strength=bias_strength,
                    score_adjustment=score_adjustment,
                    weight_adjustment=0.0,
                    network_distance=distance.overall_distance,
                    description=f"Citation bias (distance: {distance.citation_distance:.2f})"
                ))
                total_score_adjustment += score_adjustment
            
            if distance.community_distance < 0.2:
                # Same community/clique - small positive bias
                bias_strength = (1.0 - distance.community_distance) * 0.2
                score_adjustment = bias_strength * 0.3  # Up to 0.3 points positive
                
                influence_effects.append(NetworkInfluenceEffect(
                    reviewer_id=reviewer_id,
                    author_id=author_id,
                    influence_type="community-bias",
                    influence_strength=bias_strength,
                    score_adjustment=score_adjustment,
                    weight_adjustment=0.0,
                    network_distance=distance.overall_distance,
                    description=f"Community bias (distance: {distance.community_distance:.2f})"
                ))
                total_score_adjustment += score_adjustment
        
        # Apply total adjustment to base score (capped)
        adjusted_score = base_score + min(2.0, max(-1.0, total_score_adjustment))
        adjusted_score = max(1.0, min(10.0, adjusted_score))  # Keep within valid range
        
        logger.debug(f"Score bias adjustment for reviewer {reviewer_id}: {base_score} -> {adjusted_score}")
        
        return adjusted_score, influence_effects
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """
        Get network influence statistics.
        
        Returns:
            Dict[str, Any]: Network statistics
        """
        stats = {
            "networks_available": {
                "collaboration": self.collaboration_network is not None,
                "citation": self.citation_network is not None,
                "community": self.conference_community is not None
            },
            "distance_cache_size": len(self.distance_cache),
            "network_coverage": {}
        }
        
        # Get coverage statistics from each network
        if self.collaboration_network:
            collab_stats = self.collaboration_network.get_network_statistics()
            stats["network_coverage"]["collaboration"] = {
                "total_researchers": collab_stats["total_researchers"],
                "total_collaborations": collab_stats["total_collaborations"]
            }
        
        if self.citation_network:
            citation_stats = self.citation_network.get_citation_network_statistics()
            stats["network_coverage"]["citation"] = {
                "total_authors": citation_stats["total_authors"],
                "total_citations": citation_stats["total_citations"]
            }
        
        if self.conference_community:
            community_stats = self.conference_community.get_community_statistics()
            stats["network_coverage"]["community"] = {
                "total_attendees": community_stats["total_attendees"],
                "total_venues": community_stats["total_venues"]
            }
        
        return stats
    
    def clear_distance_cache(self):
        """Clear the distance calculation cache."""
        self.distance_cache.clear()
        logger.info("Cleared network distance cache")
    
    def analyze_network_effects(self, reviewer_author_pairs: List[Tuple[str, List[str]]],
                              venue_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze network effects across multiple reviewer-author pairs.
        
        Args:
            reviewer_author_pairs: List of (reviewer_id, author_ids) tuples
            venue_id: Optional venue ID
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        analysis = {
            "total_pairs": len(reviewer_author_pairs),
            "distance_distribution": {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0},
            "influence_types": defaultdict(int),
            "average_weight_adjustment": 0.0,
            "average_score_adjustment": 0.0,
            "high_influence_pairs": []
        }
        
        total_weight_adj = 0.0
        total_score_adj = 0.0
        
        for reviewer_id, author_ids in reviewer_author_pairs:
            for author_id in author_ids:
                distance = self.calculate_network_distance(reviewer_id, author_id, venue_id)
                
                # Update distance distribution
                dist_bucket = min(4, int(distance.overall_distance * 5))
                bucket_keys = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
                analysis["distance_distribution"][bucket_keys[dist_bucket]] += 1
                
                # Calculate influence effects
                _, weight_effects = self.calculate_review_weight_adjustment(reviewer_id, [author_id], venue_id)
                _, score_effects = self.calculate_network_bias_adjustment(reviewer_id, [author_id], 5.0, venue_id)
                
                # Aggregate effects
                for effect in weight_effects + score_effects:
                    analysis["influence_types"][effect.influence_type] += 1
                    total_weight_adj += effect.weight_adjustment
                    total_score_adj += effect.score_adjustment
                    
                    if abs(effect.weight_adjustment) > 0.3 or abs(effect.score_adjustment) > 0.5:
                        analysis["high_influence_pairs"].append({
                            "reviewer_id": reviewer_id,
                            "author_id": author_id,
                            "influence_type": effect.influence_type,
                            "weight_adjustment": effect.weight_adjustment,
                            "score_adjustment": effect.score_adjustment,
                            "network_distance": distance.overall_distance
                        })
        
        # Calculate averages
        total_comparisons = sum(len(author_ids) for _, author_ids in reviewer_author_pairs)
        if total_comparisons > 0:
            analysis["average_weight_adjustment"] = total_weight_adj / total_comparisons
            analysis["average_score_adjustment"] = total_score_adj / total_comparisons
        
        return analysis