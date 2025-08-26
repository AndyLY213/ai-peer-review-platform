"""
Citation Cartel Detection System

This module implements the CitationCartelDetector class for mutual citation analysis,
identifying suspicious citation patterns, and detecting citation cartels and rings.
"""

import json
import uuid
from datetime import datetime, date
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
import math
import itertools
from statistics import mean, stdev

from src.core.exceptions import ValidationError, NetworkError
from src.core.logging_config import get_logger
from src.data.enhanced_models import EnhancedResearcher, PublicationRecord


logger = get_logger(__name__)


@dataclass
class CitationCartel:
    """Represents a detected citation cartel."""
    cartel_id: str
    member_ids: List[str]
    cartel_type: str  # "mutual_pair", "citation_ring", "citation_cluster"
    detection_date: date
    strength_score: float  # 0-1 scale indicating cartel strength
    total_mutual_citations: int
    average_citations_per_member: float
    suspicious_patterns: List[str]
    evidence: Dict[str, Any]
    
    def __post_init__(self):
        """Validate cartel data."""
        if len(self.member_ids) < 2:
            raise ValidationError("member_ids", "too few", "at least 2 members")
        if not 0 <= self.strength_score <= 1:
            raise ValidationError("strength_score", self.strength_score, "0-1 range")


@dataclass
class MutualCitationPair:
    """Represents a mutual citation relationship between two researchers."""
    researcher_a: str
    researcher_b: str
    citations_a_to_b: int
    citations_b_to_a: int
    time_span_years: float
    suspicion_score: float  # 0-1 scale
    total_mutual_citations: int = 0
    citation_ratio: float = 0.0  # Balance of citations (closer to 1.0 = more balanced)
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.total_mutual_citations = self.citations_a_to_b + self.citations_b_to_a
        if max(self.citations_a_to_b, self.citations_b_to_a) > 0:
            self.citation_ratio = min(self.citations_a_to_b, self.citations_b_to_a) / max(self.citations_a_to_b, self.citations_b_to_a)
        else:
            self.citation_ratio = 0.0


@dataclass
class CitationRing:
    """Represents a citation ring where multiple researchers cite each other in a cycle."""
    ring_id: str
    member_ids: List[str]
    ring_size: int
    total_ring_citations: int
    average_citations_per_edge: float
    ring_density: float  # Proportion of possible edges that exist
    detection_confidence: float  # 0-1 scale
    citation_matrix: Dict[Tuple[str, str], int]  # (citing, cited) -> count
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.ring_size = len(self.member_ids)
        possible_edges = self.ring_size * (self.ring_size - 1)  # Directed edges
        actual_edges = len([count for count in self.citation_matrix.values() if count > 0])
        self.ring_density = actual_edges / possible_edges if possible_edges > 0 else 0.0


class CitationCartelDetector:
    """
    Detects citation cartels and suspicious mutual citation patterns.
    
    This class implements comprehensive citation cartel detection including:
    - Mutual citation pair identification
    - Citation ring detection
    - Citation cluster analysis
    - Suspicious pattern scoring
    """
    
    def __init__(self, min_mutual_citations: int = 3, min_ring_size: int = 3,
                 suspicion_threshold: float = 0.7):
        """
        Initialize citation cartel detector.
        
        Args:
            min_mutual_citations: Minimum mutual citations to consider suspicious
            min_ring_size: Minimum size for citation rings
            suspicion_threshold: Threshold for flagging suspicious patterns
        """
        self.min_mutual_citations = min_mutual_citations
        self.min_ring_size = min_ring_size
        self.suspicion_threshold = suspicion_threshold
        
        # Detection results
        self.detected_cartels: List[CitationCartel] = []
        self.mutual_pairs: List[MutualCitationPair] = []
        self.citation_rings: List[CitationRing] = []
        
        # Citation data
        self.author_citations: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.author_papers: Dict[str, Set[str]] = defaultdict(set)
        self.paper_authors: Dict[str, List[str]] = {}
        
        logger.info(f"Initialized CitationCartelDetector with min_mutual_citations={min_mutual_citations}, "
                   f"min_ring_size={min_ring_size}, suspicion_threshold={suspicion_threshold}")
    
    def load_citation_data(self, citation_network: Any):
        """
        Load citation data from a citation network.
        
        Args:
            citation_network: CitationNetwork instance with citation data
        """
        try:
            self.author_citations = citation_network.author_citations
            self.author_papers = citation_network.author_papers
            self.paper_authors = citation_network.paper_authors
            
            logger.info(f"Loaded citation data: {len(self.author_citations)} authors, "
                       f"{len(self.paper_authors)} papers")
            
        except Exception as e:
            logger.error(f"Failed to load citation data: {e}")
            raise NetworkError(f"Failed to load citation data: {e}")
    
    def detect_mutual_citation_pairs(self) -> List[MutualCitationPair]:
        """
        Detect mutual citation pairs between researchers.
        
        Returns:
            List[MutualCitationPair]: List of detected mutual citation pairs
        """
        mutual_pairs = []
        processed_pairs = set()
        
        for citing_author, cited_authors in self.author_citations.items():
            for cited_author, citations_to in cited_authors.items():
                if citing_author == cited_author:  # Skip self-citations
                    continue
                
                # Check for mutual citations
                citations_from = self.author_citations.get(cited_author, {}).get(citing_author, 0)
                
                if citations_to >= self.min_mutual_citations or citations_from >= self.min_mutual_citations:
                    # Create canonical pair (sorted order)
                    pair_key = tuple(sorted([citing_author, cited_author]))
                    
                    if pair_key not in processed_pairs:
                        # Calculate suspicion score
                        total_citations = citations_to + citations_from
                        balance_score = min(citations_to, citations_from) / max(citations_to, citations_from) if max(citations_to, citations_from) > 0 else 0
                        volume_score = min(1.0, total_citations / 20.0)  # Normalize to 0-1
                        suspicion_score = (balance_score * 0.6) + (volume_score * 0.4)
                        
                        mutual_pair = MutualCitationPair(
                            researcher_a=pair_key[0],
                            researcher_b=pair_key[1],
                            citations_a_to_b=citations_to if citing_author == pair_key[0] else citations_from,
                            citations_b_to_a=citations_from if citing_author == pair_key[0] else citations_to,
                            time_span_years=5.0,  # Placeholder - would need temporal data
                            suspicion_score=suspicion_score
                        )
                        
                        mutual_pairs.append(mutual_pair)
                        processed_pairs.add(pair_key)
        
        # Sort by suspicion score
        mutual_pairs.sort(key=lambda x: x.suspicion_score, reverse=True)
        
        self.mutual_pairs = mutual_pairs
        logger.info(f"Detected {len(mutual_pairs)} mutual citation pairs")
        
        return mutual_pairs
    
    def detect_citation_rings(self, min_ring_size: Optional[int] = None) -> List[CitationRing]:
        """
        Detect citation rings where multiple researchers cite each other in cycles.
        
        Args:
            min_ring_size: Minimum size for citation rings (uses instance default if None)
            
        Returns:
            List[CitationRing]: List of detected citation rings
        """
        if min_ring_size is None:
            min_ring_size = self.min_ring_size
        
        citation_rings = []
        
        # Get all authors with significant citation activity
        active_authors = [author for author, citations in self.author_citations.items() 
                         if len(citations) >= min_ring_size - 1]
        
        # Find potential rings using graph analysis
        for ring_size in range(min_ring_size, min(8, len(active_authors) + 1)):  # Limit ring size
            for candidate_members in itertools.combinations(active_authors, ring_size):
                ring_citations = {}
                total_citations = 0
                
                # Check all pairs in the candidate ring
                for citing_author in candidate_members:
                    for cited_author in candidate_members:
                        if citing_author != cited_author:
                            citations = self.author_citations.get(citing_author, {}).get(cited_author, 0)
                            if citations > 0:
                                ring_citations[(citing_author, cited_author)] = citations
                                total_citations += citations
                
                # Calculate ring metrics
                possible_edges = ring_size * (ring_size - 1)
                actual_edges = len(ring_citations)
                ring_density = actual_edges / possible_edges if possible_edges > 0 else 0
                
                # Check if this forms a significant ring
                if (total_citations >= self.min_mutual_citations * ring_size and 
                    ring_density >= 0.3 and  # At least 30% of possible citations exist
                    actual_edges >= ring_size):  # At least one citation per member
                    
                    avg_citations = total_citations / actual_edges if actual_edges > 0 else 0
                    
                    # Calculate detection confidence
                    density_score = ring_density
                    volume_score = min(1.0, total_citations / (ring_size * 10))
                    balance_score = 1.0 - (stdev([citations for citations in ring_citations.values()]) / 
                                          mean([citations for citations in ring_citations.values()])) if len(ring_citations) > 1 else 1.0
                    balance_score = max(0.0, min(1.0, balance_score))
                    
                    detection_confidence = (density_score * 0.4) + (volume_score * 0.3) + (balance_score * 0.3)
                    
                    if detection_confidence >= self.suspicion_threshold:
                        ring = CitationRing(
                            ring_id=str(uuid.uuid4()),
                            member_ids=list(candidate_members),
                            ring_size=ring_size,
                            total_ring_citations=total_citations,
                            average_citations_per_edge=avg_citations,
                            ring_density=ring_density,
                            detection_confidence=detection_confidence,
                            citation_matrix=ring_citations
                        )
                        
                        citation_rings.append(ring)
        
        # Remove overlapping rings (keep the one with highest confidence)
        citation_rings = self._remove_overlapping_rings(citation_rings)
        
        # Sort by detection confidence
        citation_rings.sort(key=lambda x: x.detection_confidence, reverse=True)
        
        self.citation_rings = citation_rings
        logger.info(f"Detected {len(citation_rings)} citation rings")
        
        return citation_rings
    
    def _remove_overlapping_rings(self, rings: List[CitationRing]) -> List[CitationRing]:
        """Remove overlapping citation rings, keeping the highest confidence ones."""
        if not rings:
            return rings
        
        # Sort by confidence
        rings.sort(key=lambda x: x.detection_confidence, reverse=True)
        
        non_overlapping = []
        used_members = set()
        
        for ring in rings:
            ring_members = set(ring.member_ids)
            
            # Check for significant overlap (more than 50% of members)
            overlap = len(ring_members.intersection(used_members))
            if overlap < len(ring_members) * 0.5:
                non_overlapping.append(ring)
                used_members.update(ring_members)
        
        return non_overlapping
    
    def analyze_citation_patterns(self, researchers: List[EnhancedResearcher]) -> Dict[str, Any]:
        """
        Analyze citation patterns for suspicious behavior.
        
        Args:
            researchers: List of researchers to analyze
            
        Returns:
            Dict[str, Any]: Analysis results with pattern statistics
        """
        analysis = {
            "total_researchers": len(researchers),
            "citation_statistics": {},
            "suspicious_patterns": {},
            "cartel_analysis": {}
        }
        
        # Calculate citation statistics
        all_citations = []
        self_citation_counts = []
        mutual_citation_counts = []
        
        for researcher in researchers:
            researcher_id = researcher.id
            
            # Self-citations
            self_citations = self.author_citations.get(researcher_id, {}).get(researcher_id, 0)
            self_citation_counts.append(self_citations)
            
            # Total citations given
            total_citations_given = sum(self.author_citations.get(researcher_id, {}).values())
            all_citations.append(total_citations_given)
            
            # Mutual citations
            mutual_citations = 0
            for cited_author, count in self.author_citations.get(researcher_id, {}).items():
                reverse_count = self.author_citations.get(cited_author, {}).get(researcher_id, 0)
                if count > 0 and reverse_count > 0:
                    mutual_citations += count
            mutual_citation_counts.append(mutual_citations)
        
        # Calculate statistics
        if all_citations:
            analysis["citation_statistics"] = {
                "mean_citations_given": mean(all_citations),
                "std_citations_given": stdev(all_citations) if len(all_citations) > 1 else 0,
                "mean_self_citations": mean(self_citation_counts),
                "mean_mutual_citations": mean(mutual_citation_counts),
                "high_self_citers": len([c for c in self_citation_counts if c >= 5]),
                "high_mutual_citers": len([c for c in mutual_citation_counts if c >= 10])
            }
        
        # Detect patterns
        mutual_pairs = self.detect_mutual_citation_pairs()
        citation_rings = self.detect_citation_rings()
        
        analysis["suspicious_patterns"] = {
            "mutual_pairs_count": len(mutual_pairs),
            "high_suspicion_pairs": len([p for p in mutual_pairs if p.suspicion_score >= self.suspicion_threshold]),
            "citation_rings_count": len(citation_rings),
            "high_confidence_rings": len([r for r in citation_rings if r.detection_confidence >= self.suspicion_threshold])
        }
        
        # Create cartels from detected patterns
        cartels = self._create_cartels_from_patterns(mutual_pairs, citation_rings)
        
        analysis["cartel_analysis"] = {
            "total_cartels_detected": len(cartels),
            "cartel_types": Counter([c.cartel_type for c in cartels]),
            "high_strength_cartels": len([c for c in cartels if c.strength_score >= self.suspicion_threshold]),
            "total_researchers_in_cartels": len(set(member for cartel in cartels for member in cartel.member_ids))
        }
        
        self.detected_cartels = cartels
        
        return analysis
    
    def _create_cartels_from_patterns(self, mutual_pairs: List[MutualCitationPair], 
                                    citation_rings: List[CitationRing]) -> List[CitationCartel]:
        """Create cartel objects from detected patterns."""
        cartels = []
        
        # Create cartels from high-suspicion mutual pairs
        for pair in mutual_pairs:
            if pair.suspicion_score >= self.suspicion_threshold:
                cartel = CitationCartel(
                    cartel_id=str(uuid.uuid4()),
                    member_ids=[pair.researcher_a, pair.researcher_b],
                    cartel_type="mutual_pair",
                    detection_date=date.today(),
                    strength_score=pair.suspicion_score,
                    total_mutual_citations=pair.total_mutual_citations,
                    average_citations_per_member=pair.total_mutual_citations / 2,
                    suspicious_patterns=[
                        f"Mutual citations: {pair.citations_a_to_b} + {pair.citations_b_to_a}",
                        f"Citation balance ratio: {pair.citation_ratio:.2f}"
                    ],
                    evidence={
                        "citations_a_to_b": pair.citations_a_to_b,
                        "citations_b_to_a": pair.citations_b_to_a,
                        "citation_ratio": pair.citation_ratio,
                        "suspicion_score": pair.suspicion_score
                    }
                )
                cartels.append(cartel)
        
        # Create cartels from citation rings
        for ring in citation_rings:
            if ring.detection_confidence >= self.suspicion_threshold:
                cartel = CitationCartel(
                    cartel_id=ring.ring_id,
                    member_ids=ring.member_ids,
                    cartel_type="citation_ring",
                    detection_date=date.today(),
                    strength_score=ring.detection_confidence,
                    total_mutual_citations=ring.total_ring_citations,
                    average_citations_per_member=ring.total_ring_citations / ring.ring_size,
                    suspicious_patterns=[
                        f"Ring size: {ring.ring_size} members",
                        f"Ring density: {ring.ring_density:.2f}",
                        f"Total ring citations: {ring.total_ring_citations}"
                    ],
                    evidence={
                        "ring_size": ring.ring_size,
                        "ring_density": ring.ring_density,
                        "citation_matrix": ring.citation_matrix,
                        "detection_confidence": ring.detection_confidence
                    }
                )
                cartels.append(cartel)
        
        return cartels
    
    def get_researcher_cartel_involvement(self, researcher_id: str) -> Dict[str, Any]:
        """
        Get cartel involvement information for a specific researcher.
        
        Args:
            researcher_id: ID of the researcher
            
        Returns:
            Dict[str, Any]: Cartel involvement information
        """
        involvement = {
            "researcher_id": researcher_id,
            "cartels_involved": [],
            "total_cartels": 0,
            "cartel_types": [],
            "max_strength_score": 0.0,
            "total_suspicious_citations": 0
        }
        
        for cartel in self.detected_cartels:
            if researcher_id in cartel.member_ids:
                involvement["cartels_involved"].append({
                    "cartel_id": cartel.cartel_id,
                    "cartel_type": cartel.cartel_type,
                    "strength_score": cartel.strength_score,
                    "other_members": [m for m in cartel.member_ids if m != researcher_id],
                    "total_mutual_citations": cartel.total_mutual_citations
                })
                involvement["cartel_types"].append(cartel.cartel_type)
                involvement["max_strength_score"] = max(involvement["max_strength_score"], cartel.strength_score)
                involvement["total_suspicious_citations"] += cartel.total_mutual_citations
        
        involvement["total_cartels"] = len(involvement["cartels_involved"])
        involvement["cartel_types"] = list(set(involvement["cartel_types"]))
        
        return involvement
    
    def generate_cartel_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive cartel detection report.
        
        Returns:
            Dict[str, Any]: Comprehensive report
        """
        report = {
            "detection_summary": {
                "total_cartels": len(self.detected_cartels),
                "mutual_pairs": len([c for c in self.detected_cartels if c.cartel_type == "mutual_pair"]),
                "citation_rings": len([c for c in self.detected_cartels if c.cartel_type == "citation_ring"]),
                "high_strength_cartels": len([c for c in self.detected_cartels if c.strength_score >= 0.8])
            },
            "cartel_details": [],
            "researcher_involvement": {},
            "network_statistics": {
                "total_authors": len(self.author_citations),
                "total_papers": len(self.paper_authors),
                "authors_in_cartels": len(set(member for cartel in self.detected_cartels for member in cartel.member_ids))
            }
        }
        
        # Add cartel details
        for cartel in sorted(self.detected_cartels, key=lambda x: x.strength_score, reverse=True):
            report["cartel_details"].append({
                "cartel_id": cartel.cartel_id,
                "type": cartel.cartel_type,
                "members": cartel.member_ids,
                "strength_score": cartel.strength_score,
                "total_citations": cartel.total_mutual_citations,
                "suspicious_patterns": cartel.suspicious_patterns,
                "detection_date": cartel.detection_date.isoformat()
            })
        
        # Add researcher involvement summary
        all_members = set(member for cartel in self.detected_cartels for member in cartel.member_ids)
        for member in all_members:
            report["researcher_involvement"][member] = self.get_researcher_cartel_involvement(member)
        
        return report
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": {
                "min_mutual_citations": self.min_mutual_citations,
                "min_ring_size": self.min_ring_size,
                "suspicion_threshold": self.suspicion_threshold
            },
            "detected_cartels": [
                {
                    "cartel_id": cartel.cartel_id,
                    "member_ids": cartel.member_ids,
                    "cartel_type": cartel.cartel_type,
                    "detection_date": cartel.detection_date.isoformat(),
                    "strength_score": cartel.strength_score,
                    "total_mutual_citations": cartel.total_mutual_citations,
                    "average_citations_per_member": cartel.average_citations_per_member,
                    "suspicious_patterns": cartel.suspicious_patterns,
                    "evidence": cartel.evidence
                }
                for cartel in self.detected_cartels
            ],
            "mutual_pairs": [
                {
                    "researcher_a": pair.researcher_a,
                    "researcher_b": pair.researcher_b,
                    "citations_a_to_b": pair.citations_a_to_b,
                    "citations_b_to_a": pair.citations_b_to_a,
                    "total_mutual_citations": pair.total_mutual_citations,
                    "citation_ratio": pair.citation_ratio,
                    "suspicion_score": pair.suspicion_score
                }
                for pair in self.mutual_pairs
            ],
            "citation_rings": [
                {
                    "ring_id": ring.ring_id,
                    "member_ids": ring.member_ids,
                    "ring_size": ring.ring_size,
                    "total_ring_citations": ring.total_ring_citations,
                    "ring_density": ring.ring_density,
                    "detection_confidence": ring.detection_confidence,
                    "citation_matrix": {f"{k[0]}->{k[1]}": v for k, v in ring.citation_matrix.items()}
                }
                for ring in self.citation_rings
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CitationCartelDetector':
        """Create from dictionary."""
        config = data.get("config", {})
        detector = cls(
            min_mutual_citations=config.get("min_mutual_citations", 3),
            min_ring_size=config.get("min_ring_size", 3),
            suspicion_threshold=config.get("suspicion_threshold", 0.7)
        )
        
        # Load detected cartels
        for cartel_data in data.get("detected_cartels", []):
            cartel = CitationCartel(
                cartel_id=cartel_data["cartel_id"],
                member_ids=cartel_data["member_ids"],
                cartel_type=cartel_data["cartel_type"],
                detection_date=date.fromisoformat(cartel_data["detection_date"]),
                strength_score=cartel_data["strength_score"],
                total_mutual_citations=cartel_data["total_mutual_citations"],
                average_citations_per_member=cartel_data["average_citations_per_member"],
                suspicious_patterns=cartel_data["suspicious_patterns"],
                evidence=cartel_data["evidence"]
            )
            detector.detected_cartels.append(cartel)
        
        # Load mutual pairs
        for pair_data in data.get("mutual_pairs", []):
            pair = MutualCitationPair(
                researcher_a=pair_data["researcher_a"],
                researcher_b=pair_data["researcher_b"],
                citations_a_to_b=pair_data["citations_a_to_b"],
                citations_b_to_a=pair_data["citations_b_to_a"],
                time_span_years=5.0,  # Default
                suspicion_score=pair_data["suspicion_score"]
            )
            detector.mutual_pairs.append(pair)
        
        # Load citation rings
        for ring_data in data.get("citation_rings", []):
            citation_matrix = {}
            for edge_str, count in ring_data.get("citation_matrix", {}).items():
                citing, cited = edge_str.split("->")
                citation_matrix[(citing, cited)] = count
            
            ring = CitationRing(
                ring_id=ring_data["ring_id"],
                member_ids=ring_data["member_ids"],
                ring_size=ring_data["ring_size"],
                total_ring_citations=ring_data["total_ring_citations"],
                average_citations_per_edge=ring_data.get("average_citations_per_edge", 0),
                ring_density=ring_data["ring_density"],
                detection_confidence=ring_data["detection_confidence"],
                citation_matrix=citation_matrix
            )
            detector.citation_rings.append(ring)
        
        return detector