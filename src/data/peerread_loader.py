"""
PeerRead Dataset Integration Utilities

This module provides utilities to load and parse JSON review files from the PeerRead dataset,
map PeerRead review dimensions to the enhanced review system, extract venue characteristics,
and create data extraction utilities for papers, reviews, and venue statistics.
"""

import os
import json
import glob
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import statistics
from collections import defaultdict, Counter

from src.core.exceptions import DatasetError, ValidationError, FileOperationError
from src.core.logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class PeerReadReview:
    """Represents a single review from PeerRead dataset."""
    paper_id: str
    reviewer_id: str = "anonymous"
    
    # PeerRead dimensions (1-5 scale typically)
    impact: Optional[int] = None
    substance: Optional[int] = None
    soundness_correctness: Optional[int] = None
    originality: Optional[int] = None
    clarity: Optional[int] = None
    meaningful_comparison: Optional[int] = None
    appropriateness: Optional[int] = None
    
    # Additional fields
    recommendation: Optional[int] = None
    reviewer_confidence: Optional[int] = None
    comments: str = ""
    presentation_format: str = ""
    is_meta_review: bool = False
    date: Optional[str] = None
    
    # Mapped dimensions for enhanced system (1-10 scale)
    novelty: Optional[float] = field(init=False, default=None)
    technical_quality: Optional[float] = field(init=False, default=None)
    significance: Optional[float] = field(init=False, default=None)
    reproducibility: Optional[float] = field(init=False, default=None)
    related_work: Optional[float] = field(init=False, default=None)
    clarity_mapped: Optional[float] = field(init=False, default=None)  # Avoid name conflict
    
    def __post_init__(self):
        """Map PeerRead dimensions to enhanced system dimensions."""
        self._map_dimensions()
    
    def _map_dimensions(self):
        """Map PeerRead review dimensions to enhanced review system dimensions."""
        # Map IMPACT → significance (scale 1-5 to 1-10)
        if self.impact is not None:
            self.significance = self.impact * 2.0
        
        # Map SUBSTANCE → technical quality (scale 1-5 to 1-10)  
        if self.substance is not None:
            self.technical_quality = self.substance * 2.0
        
        # Map SOUNDNESS_CORRECTNESS → technical quality (average if both exist)
        if self.soundness_correctness is not None:
            soundness_mapped = self.soundness_correctness * 2.0
            if self.technical_quality is not None:
                self.technical_quality = (self.technical_quality + soundness_mapped) / 2.0
            else:
                self.technical_quality = soundness_mapped
        
        # Map ORIGINALITY → novelty (scale 1-5 to 1-10)
        if self.originality is not None:
            self.novelty = self.originality * 2.0
        
        # Map CLARITY → clarity_mapped (scale 1-5 to 1-10)
        if self.clarity is not None:
            self.clarity_mapped = self.clarity * 2.0
        
        # Map MEANINGFUL_COMPARISON → related work (scale 1-5 to 1-10)
        if self.meaningful_comparison is not None:
            self.related_work = self.meaningful_comparison * 2.0
        
        # Set default reproducibility score based on technical quality if available
        if self.technical_quality is not None and self.reproducibility is None:
            # Assume reproducibility correlates with technical quality but is slightly lower
            self.reproducibility = max(1.0, self.technical_quality - 1.0)


@dataclass
class PeerReadPaper:
    """Represents a paper from PeerRead dataset."""
    id: str
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    venue: str = ""
    year: Optional[int] = None
    accepted: Optional[bool] = None
    reviews: List[PeerReadReview] = field(default_factory=list)
    
    # Extracted content
    content: str = ""
    keywords: List[str] = field(default_factory=list)
    
    # Venue characteristics
    venue_type: str = ""
    field: str = ""


@dataclass
class VenueCharacteristics:
    """Characteristics of a publication venue extracted from PeerRead data."""
    name: str
    venue_type: str  # conference, journal, workshop
    field: str  # AI, NLP, Vision, etc.
    
    # Statistics from PeerRead data
    total_papers: int = 0
    accepted_papers: int = 0
    acceptance_rate: float = 0.0
    
    # Review statistics
    avg_reviews_per_paper: float = 0.0
    avg_review_length: float = 0.0
    
    # Score distributions (for calibration)
    impact_scores: List[int] = field(default_factory=list)
    substance_scores: List[int] = field(default_factory=list)
    soundness_scores: List[int] = field(default_factory=list)
    originality_scores: List[int] = field(default_factory=list)
    clarity_scores: List[int] = field(default_factory=list)
    meaningful_comparison_scores: List[int] = field(default_factory=list)
    
    # Statistical measures
    impact_mean: float = 0.0
    substance_mean: float = 0.0
    soundness_mean: float = 0.0
    originality_mean: float = 0.0
    clarity_mean: float = 0.0
    meaningful_comparison_mean: float = 0.0


class PeerReadLoader:
    """
    Loader class to parse JSON review files from PeerRead dataset and extract
    venue characteristics with real acceptance rates.
    """
    
    def __init__(self, peerread_path: str = "dataset"):
        """
        Initialize PeerRead loader.
        
        Args:
            peerread_path: Path to PeerRead dataset directory
        """
        self.peerread_path = Path(peerread_path)
        self.papers: Dict[str, PeerReadPaper] = {}
        self.venues: Dict[str, VenueCharacteristics] = {}
        
        # Venue mapping for standardization
        self.venue_mapping = {
            "acl_2017": {"name": "ACL", "type": "conference", "field": "NLP"},
            "conll_2016": {"name": "CoNLL", "type": "conference", "field": "NLP"},
            "iclr_2017": {"name": "ICLR", "type": "conference", "field": "AI"},
            "nips_2013-2017": {"name": "NIPS", "type": "conference", "field": "AI"},
            "arxiv.cs.ai_2007-2017": {"name": "arXiv-AI", "type": "preprint", "field": "AI"},
            "arxiv.cs.cl_2007-2017": {"name": "arXiv-CL", "type": "preprint", "field": "NLP"},
            "arxiv.cs.lg_2007-2017": {"name": "arXiv-LG", "type": "preprint", "field": "AI"},
        }
        
        # Real acceptance rates from literature (approximate)
        self.known_acceptance_rates = {
            "ACL": 0.25,  # ~25%
            "CoNLL": 0.35,  # ~35%
            "ICLR": 0.30,  # ~30%
            "NIPS": 0.20,  # ~20%
            "arXiv-AI": 1.0,  # preprints are not peer-reviewed
            "arXiv-CL": 1.0,
            "arXiv-LG": 1.0,
        }
        
        self._validate_dataset_path()
    
    def _validate_dataset_path(self):
        """Validate that PeerRead dataset path exists and has expected structure."""
        if not self.peerread_path.exists():
            raise DatasetError(str(self.peerread_path), "PeerRead dataset directory not found")
        
        data_path = self.peerread_path / "data"
        if not data_path.exists():
            raise DatasetError(str(data_path), "PeerRead data directory not found")
        
        logger.info(f"PeerRead dataset found at: {self.peerread_path}")
    
    def load_all_venues(self) -> Dict[str, VenueCharacteristics]:
        """
        Load data from all available venues in PeerRead dataset.
        
        Returns:
            Dictionary mapping venue names to their characteristics
        """
        data_path = self.peerread_path / "data"
        
        for venue_dir in data_path.iterdir():
            if venue_dir.is_dir() and venue_dir.name in self.venue_mapping:
                logger.info(f"Loading venue: {venue_dir.name}")
                try:
                    self._load_venue_data(venue_dir.name, venue_dir)
                except Exception as e:
                    logger.error(f"Failed to load venue {venue_dir.name}: {e}")
                    continue
        
        self._calculate_venue_statistics()
        return self.venues
    
    def load_venue(self, venue_name: str) -> Optional[VenueCharacteristics]:
        """
        Load data for a specific venue.
        
        Args:
            venue_name: Name of venue directory (e.g., 'acl_2017', 'iclr_2017')
            
        Returns:
            VenueCharacteristics object or None if venue not found
        """
        if venue_name not in self.venue_mapping:
            logger.warning(f"Unknown venue: {venue_name}")
            return None
        
        venue_path = self.peerread_path / "data" / venue_name
        if not venue_path.exists():
            logger.warning(f"Venue directory not found: {venue_path}")
            return None
        
        self._load_venue_data(venue_name, venue_path)
        self._calculate_venue_statistics()
        
        mapped_name = self.venue_mapping[venue_name]["name"]
        return self.venues.get(mapped_name)
    
    def _load_venue_data(self, venue_dir_name: str, venue_path: Path):
        """Load all data for a specific venue."""
        venue_info = self.venue_mapping[venue_dir_name]
        venue_name = venue_info["name"]
        
        # Initialize venue characteristics
        if venue_name not in self.venues:
            self.venues[venue_name] = VenueCharacteristics(
                name=venue_name,
                venue_type=venue_info["type"],
                field=venue_info["field"]
            )
        
        venue_chars = self.venues[venue_name]
        
        # Load papers and reviews from all splits (train, dev, test)
        for split in ["train", "dev", "test"]:
            split_path = venue_path / split
            if not split_path.exists():
                continue
            
            # Load reviews
            reviews_path = split_path / "reviews"
            if reviews_path.exists():
                self._load_reviews_from_directory(reviews_path, venue_name)
            
            # Load parsed PDFs if available
            pdfs_path = split_path / "parsed_pdfs"
            if pdfs_path.exists():
                self._load_papers_from_directory(pdfs_path, venue_name)
    
    def _load_reviews_from_directory(self, reviews_path: Path, venue_name: str):
        """Load all review files from a directory."""
        for review_file in reviews_path.glob("*.json"):
            try:
                self._load_review_file(review_file, venue_name)
            except Exception as e:
                logger.warning(f"Failed to load review file {review_file}: {e}")
    
    def _load_review_file(self, review_file: Path, venue_name: str):
        """Load a single review JSON file."""
        with open(review_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        paper_id = data.get('id', review_file.stem)
        title = data.get('title', '')
        abstract = data.get('abstract', '')
        authors = data.get('authors', [])
        if isinstance(authors, str):
            authors = [authors]
        
        # Create or update paper
        if paper_id not in self.papers:
            self.papers[paper_id] = PeerReadPaper(
                id=paper_id,
                title=title,
                abstract=abstract,
                authors=authors,
                venue=venue_name
            )
        
        paper = self.papers[paper_id]
        paper.accepted = data.get('accepted', None)
        
        # Process reviews
        reviews_data = data.get('reviews', [])
        for i, review_data in enumerate(reviews_data):
            if review_data.get('is_meta_review') or review_data.get('IS_META_REVIEW'):
                continue  # Skip meta reviews
            
            review = self._parse_review(review_data, paper_id, f"reviewer_{i}")
            paper.reviews.append(review)
    
    def _parse_review(self, review_data: Dict, paper_id: str, reviewer_id: str) -> PeerReadReview:
        """Parse a single review from JSON data."""
        review = PeerReadReview(paper_id=paper_id, reviewer_id=reviewer_id)
        
        # Map PeerRead fields to review object
        field_mapping = {
            'IMPACT': 'impact',
            'SUBSTANCE': 'substance', 
            'SOUNDNESS_CORRECTNESS': 'soundness_correctness',
            'ORIGINALITY': 'originality',
            'CLARITY': 'clarity',
            'MEANINGFUL_COMPARISON': 'meaningful_comparison',
            'APPROPRIATENESS': 'appropriateness',
            'RECOMMENDATION': 'recommendation',
            'REVIEWER_CONFIDENCE': 'reviewer_confidence',
            'comments': 'comments',
            'PRESENTATION_FORMAT': 'presentation_format',
            'DATE': 'date'
        }
        
        for peerread_field, review_field in field_mapping.items():
            if peerread_field in review_data:
                value = review_data[peerread_field]
                
                # Convert string numbers to integers for score fields
                if review_field in ['impact', 'substance', 'soundness_correctness', 
                                  'originality', 'clarity', 'meaningful_comparison',
                                  'appropriateness', 'recommendation', 'reviewer_confidence']:
                    try:
                        value = int(value) if value is not None else None
                    except (ValueError, TypeError):
                        value = None
                
                setattr(review, review_field, value)
        
        # Call mapping after all attributes are set
        review._map_dimensions()
        
        return review
    
    def _load_papers_from_directory(self, pdfs_path: Path, venue_name: str):
        """Load parsed PDF files to get paper content."""
        for pdf_file in pdfs_path.glob("*.pdf.json"):
            try:
                self._load_paper_file(pdf_file, venue_name)
            except Exception as e:
                logger.warning(f"Failed to load paper file {pdf_file}: {e}")
    
    def _load_paper_file(self, pdf_file: Path, venue_name: str):
        """Load a single parsed PDF file."""
        with open(pdf_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        paper_id = pdf_file.stem.replace('.pdf', '')
        
        # Extract metadata
        metadata = data.get('metadata', {})
        title = metadata.get('title', '')
        abstract = metadata.get('abstractText', '')
        authors = metadata.get('authors', [])
        year = metadata.get('year')
        
        # Create or update paper
        if paper_id not in self.papers:
            self.papers[paper_id] = PeerReadPaper(
                id=paper_id,
                title=title,
                abstract=abstract,
                authors=authors,
                venue=venue_name,
                year=year
            )
        else:
            # Update existing paper with more complete data
            paper = self.papers[paper_id]
            if not paper.title and title:
                paper.title = title
            if not paper.abstract and abstract:
                paper.abstract = abstract
            if not paper.authors and authors:
                paper.authors = authors
            if not paper.year and year:
                paper.year = year
        
        # Extract content from sections
        paper = self.papers[paper_id]
        content_parts = []
        
        for section in metadata.get('sections', []):
            if section.get('heading'):
                content_parts.append(f"## {section.get('heading')}")
            content_parts.append(section.get('text', ''))
        
        paper.content = '\n\n'.join(content_parts)
    
    def _calculate_venue_statistics(self):
        """Calculate statistics for all loaded venues."""
        venue_papers = defaultdict(list)
        venue_reviews = defaultdict(list)
        
        # Group papers and reviews by venue
        for paper in self.papers.values():
            venue_papers[paper.venue].append(paper)
            venue_reviews[paper.venue].extend(paper.reviews)
        
        # Calculate statistics for each venue
        for venue_name, papers in venue_papers.items():
            if venue_name not in self.venues:
                continue
            
            venue_chars = self.venues[venue_name]
            reviews = venue_reviews[venue_name]
            
            # Basic counts
            venue_chars.total_papers = len(papers)
            accepted_papers = [p for p in papers if p.accepted is True]
            venue_chars.accepted_papers = len(accepted_papers)
            
            # Acceptance rate
            if venue_chars.total_papers > 0:
                calculated_rate = venue_chars.accepted_papers / venue_chars.total_papers
                # Use known acceptance rate if available, otherwise use calculated
                venue_chars.acceptance_rate = self.known_acceptance_rates.get(
                    venue_name, calculated_rate
                )
            
            # Review statistics
            if papers:
                review_counts = [len(p.reviews) for p in papers]
                venue_chars.avg_reviews_per_paper = statistics.mean(review_counts) if review_counts else 0
            
            if reviews:
                review_lengths = [len(r.comments) for r in reviews if r.comments]
                venue_chars.avg_review_length = statistics.mean(review_lengths) if review_lengths else 0
                
                # Score distributions
                self._calculate_score_distributions(venue_chars, reviews)
    
    def _calculate_score_distributions(self, venue_chars: VenueCharacteristics, reviews: List[PeerReadReview]):
        """Calculate score distributions for a venue."""
        # Collect scores
        for review in reviews:
            if review.impact is not None:
                venue_chars.impact_scores.append(review.impact)
            if review.substance is not None:
                venue_chars.substance_scores.append(review.substance)
            if review.soundness_correctness is not None:
                venue_chars.soundness_scores.append(review.soundness_correctness)
            if review.originality is not None:
                venue_chars.originality_scores.append(review.originality)
            if review.clarity is not None:
                venue_chars.clarity_scores.append(review.clarity)
            if review.meaningful_comparison is not None:
                venue_chars.meaningful_comparison_scores.append(review.meaningful_comparison)
        
        # Calculate means
        if venue_chars.impact_scores:
            venue_chars.impact_mean = statistics.mean(venue_chars.impact_scores)
        if venue_chars.substance_scores:
            venue_chars.substance_mean = statistics.mean(venue_chars.substance_scores)
        if venue_chars.soundness_scores:
            venue_chars.soundness_mean = statistics.mean(venue_chars.soundness_scores)
        if venue_chars.originality_scores:
            venue_chars.originality_mean = statistics.mean(venue_chars.originality_scores)
        if venue_chars.clarity_scores:
            venue_chars.clarity_mean = statistics.mean(venue_chars.clarity_scores)
        if venue_chars.meaningful_comparison_scores:
            venue_chars.meaningful_comparison_mean = statistics.mean(venue_chars.meaningful_comparison_scores)
    
    def get_papers_by_venue(self, venue_name: str) -> List[PeerReadPaper]:
        """Get all papers for a specific venue."""
        return [paper for paper in self.papers.values() if paper.venue == venue_name]
    
    def get_venue_statistics(self, venue_name: str) -> Optional[VenueCharacteristics]:
        """Get statistics for a specific venue."""
        return self.venues.get(venue_name)
    
    def validate_data_format(self) -> Dict[str, Any]:
        """
        Validate PeerRead data format and completeness.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "total_papers": len(self.papers),
            "total_venues": len(self.venues),
            "papers_with_reviews": 0,
            "papers_with_content": 0,
            "reviews_with_scores": 0,
            "venues_with_acceptance_data": 0,
            "missing_data": [],
            "format_errors": []
        }
        
        # Validate papers
        for paper_id, paper in self.papers.items():
            if paper.reviews:
                validation_results["papers_with_reviews"] += 1
            
            if paper.content:
                validation_results["papers_with_content"] += 1
            
            # Check for missing essential data
            if not paper.title:
                validation_results["missing_data"].append(f"Paper {paper_id}: missing title")
            if not paper.abstract:
                validation_results["missing_data"].append(f"Paper {paper_id}: missing abstract")
        
        # Validate reviews
        for paper in self.papers.values():
            for review in paper.reviews:
                if any([review.impact, review.substance, review.soundness_correctness,
                       review.originality, review.clarity, review.meaningful_comparison]):
                    validation_results["reviews_with_scores"] += 1
                    break
        
        # Validate venues
        for venue_name, venue_chars in self.venues.items():
            if venue_chars.total_papers > 0 and venue_chars.acceptance_rate > 0:
                validation_results["venues_with_acceptance_data"] += 1
        
        logger.info(f"Validation complete: {validation_results}")
        return validation_results
    
    def export_statistics(self) -> Dict[str, Any]:
        """Export comprehensive statistics about loaded PeerRead data."""
        stats = {
            "dataset_summary": {
                "total_papers": len(self.papers),
                "total_venues": len(self.venues),
                "total_reviews": sum(len(p.reviews) for p in self.papers.values())
            },
            "venues": {},
            "review_dimension_stats": {}
        }
        
        # Venue statistics
        for venue_name, venue_chars in self.venues.items():
            stats["venues"][venue_name] = {
                "type": venue_chars.venue_type,
                "field": venue_chars.field,
                "total_papers": venue_chars.total_papers,
                "accepted_papers": venue_chars.accepted_papers,
                "acceptance_rate": venue_chars.acceptance_rate,
                "avg_reviews_per_paper": venue_chars.avg_reviews_per_paper,
                "avg_review_length": venue_chars.avg_review_length,
                "score_means": {
                    "impact": venue_chars.impact_mean,
                    "substance": venue_chars.substance_mean,
                    "soundness": venue_chars.soundness_mean,
                    "originality": venue_chars.originality_mean,
                    "clarity": venue_chars.clarity_mean,
                    "meaningful_comparison": venue_chars.meaningful_comparison_mean
                }
            }
        
        # Overall review dimension statistics
        all_reviews = [review for paper in self.papers.values() for review in paper.reviews]
        if all_reviews:
            dimensions = ['impact', 'substance', 'soundness_correctness', 
                         'originality', 'clarity', 'meaningful_comparison']
            
            for dim in dimensions:
                scores = [getattr(review, dim) for review in all_reviews 
                         if getattr(review, dim) is not None]
                if scores:
                    stats["review_dimension_stats"][dim] = {
                        "mean": statistics.mean(scores),
                        "median": statistics.median(scores),
                        "std": statistics.stdev(scores) if len(scores) > 1 else 0,
                        "min": min(scores),
                        "max": max(scores),
                        "count": len(scores)
                    }
        
        return stats