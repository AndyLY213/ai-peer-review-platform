"""
Paper Database module for managing research papers in the Peer Review simulation.

This module handles storage, retrieval, and management of research papers.
It includes functionality to load papers from the PeerRead dataset.
"""

import os
import json
import glob
import logging
import random
from datetime import datetime
from typing import Dict, List, Optional, Any
from src.core.exceptions import (
    DatabaseError, PaperNotFoundError, FileOperationError, 
    DatasetError, ValidationError
)
from src.core.logging_config import get_logger, log_error_with_context
from src.core.validators import (
    validate_paper_id, validate_paper_data, validate_research_field, 
    validate_paper_status, validate_file_path, sanitize_string
)

# Initialize logger for this module
logger = get_logger(__name__)

# Structure for paper metadata
PAPER_SCHEMA = {
    "id": "",             # Unique identifier for the paper
    "title": "",          # Title of the paper
    "authors": [],        # List of author names
    "abstract": "",       # Abstract of the paper
    "keywords": [],       # List of keywords
    "field": "",          # Main field (e.g., AI, NLP, Computer Vision)
    "raw_venue": "",      # Original venue/conference name from the source
    "status": "",         # Status of the paper (e.g., draft, submitted, in_review, published)
    "content": "",        # Full content of the paper (or summary if using real papers)
    "publication_date": None,  # Date of publication (if published)
    "citations": 0,       # Number of citations
    "reviews": [],        # List of reviews received
    "owner_id": "",       # ID of the researcher who owns/authored the paper
    "review_requests": [] # List of review requests
}

class PaperDatabase:
    """
    Database for storing and managing research papers.
    """
    
    def __init__(self, data_path: str = "papers.json"):
        """
        Initialize the paper database.
        
        Args:
            data_path: Path to the JSON file for persistent storage
        """
        self.data_path = data_path
        self.papers = {}
        self.next_id = 1
        
        # Load existing data or initialize with PeerRead dataset
        self._load_data()
    
    def _load_data(self):
        """Load data from file or initialize with PeerRead dataset."""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.papers = data.get('papers', {})
                    self.next_id = data.get('next_id', 1)
                
                logger.info(f"Loaded {len(self.papers)} papers from {self.data_path}")
                
                # If no papers loaded, try loading from PeerRead
                if not self.papers:
                    logger.info("Papers database is empty. Loading from PeerRead test dataset...")
                    self.load_peerread_dataset(use_test_dataset=True)
            except json.JSONDecodeError as e:
                logger.error(f"Corrupted papers data file {self.data_path}: {e}")
                logger.info("Loading from PeerRead dataset instead...")
                try:
                    self.load_peerread_dataset(use_test_dataset=False, limit=100)
                except Exception as load_error:
                    logger.warning(f"Failed to load PeerRead dataset: {load_error}")
            except FileNotFoundError as e:
                log_error_with_context(e, f"loading papers data from {self.data_path}", logger)
                logger.info("Loading from PeerRead dataset instead...")
                try:
                    self.load_peerread_dataset(use_test_dataset=False, limit=100)
                except Exception as load_error:
                    logger.warning(f"Failed to load PeerRead dataset: {load_error}")
        else:
            # Initialize with PeerRead dataset
            logger.info("Papers database not found. Loading from PeerRead dataset...")
            try:
                self.load_peerread_dataset(use_test_dataset=False, limit=100)
            except Exception as e:
                logger.warning(f"Failed to load PeerRead dataset: {e}")
                # Don't create synthetic papers here - let the simulation handle it
    
    def load_peerread_dataset(self, folder_path: str = None, limit: int = 100, use_test_dataset: bool = True):
        """
        Load papers from the PeerRead dataset.
        
        Args:
            folder_path: Path to the PeerRead data folder (if None, will use default paths)
            limit: Maximum number of papers to load (for testing)
            use_test_dataset: Whether to use the test dataset instead of the full dataset
        """
        # Define mapping of venue names to research fields (matching researcher specialties)
        field_mapping = {
            "iclr": "Artificial Intelligence",
            "icml": "Artificial Intelligence", 
            "nips": "Artificial Intelligence",
            "neurips": "Artificial Intelligence",
            "aaai": "Artificial Intelligence",
            "ijcai": "Artificial Intelligence",
            "aistats": "Artificial Intelligence",
            "uai": "Artificial Intelligence",
            "cs.ai": "Artificial Intelligence",
            "cs.lg": "Artificial Intelligence",
            
            "cvpr": "Computer Vision",
            "eccv": "Computer Vision",
            "iccv": "Computer Vision", 
            "wacv": "Computer Vision",
            "bmvc": "Computer Vision",
            "cs.cv": "Computer Vision",
            
            "acl": "Natural Language Processing",
            "emnlp": "Natural Language Processing",
            "naacl": "Natural Language Processing",
            "coling": "Natural Language Processing",
            "conll": "Natural Language Processing",
            "eacl": "Natural Language Processing",
            "tacl": "Natural Language Processing",
            "cl": "Natural Language Processing",
            "cs.cl": "Natural Language Processing",
            "arxiv.cs.cl": "Natural Language Processing",
            
            "sigmod": "Data Science and Analytics",
            "vldb": "Data Science and Analytics", 
            "icde": "Data Science and Analytics",
            "kdd": "Data Science and Analytics",
            "cs.ds": "Data Science and Analytics",
            
            "chi": "Human-Computer Interaction",
            "uist": "Human-Computer Interaction",
            "cscw": "Human-Computer Interaction",
            "cs.hc": "Human-Computer Interaction",
            
            "siggraph": "Computer Vision",
            "eurographics": "Computer Vision",
            
            "www": "Computer Systems and Architecture",
            "wsdm": "Data Science and Analytics",
            
            "sigcomm": "Computer Systems and Architecture",
            "imc": "Computer Systems and Architecture", 
            "infocom": "Computer Systems and Architecture",
            "cs.se": "Computer Systems and Architecture",
            
            "cs.ro": "Robotics and Control Systems",
            "cs.cr": "Cybersecurity and Privacy",
            "cs.et": "AI Ethics and Fairness",
            "cs.cc": "Theoretical Computer Science"
        }
        
        # Determine the dataset path
        if folder_path is None:
            if use_test_dataset:
                # Use the organized dataset directory
                folder_path = os.path.abspath("dataset/organized")
                if not os.path.exists(folder_path):
                    # Fallback to papers directory
                    folder_path = os.path.abspath("dataset/papers")
                    if not os.path.exists(folder_path):
                        # Final fallback to old path for backward compatibility
                        folder_path = os.path.abspath("../PeerRead/data")
            else:
                # Use the organized dataset directory for full dataset
                folder_path = os.path.abspath("dataset/organized")
                if not os.path.exists(folder_path):
                    # Fallback to old path for backward compatibility
                    folder_path = os.path.abspath("../PeerRead/data")
        
        if not os.path.exists(folder_path):
            logger.warning(f"PeerRead dataset folder not found at '{folder_path}'")
            return  # Return gracefully instead of raising error
        
        logger.info(f"Loading papers from {'test' if use_test_dataset else 'full'} PeerRead dataset at '{folder_path}'...")
        
        # Counter for loaded papers
        papers_loaded = 0
        processed_titles = set()  # To avoid duplicates
        
        # Process all PDF JSON files in parsed_pdfs directories
        # Use stratified sampling to ensure diverse field representation
        all_json_paths = glob.glob(os.path.join(folder_path, "**", "parsed_pdfs", "*.pdf.json"), recursive=True)
        
        # Group paths by venue/field for stratified sampling
        paths_by_field = {}
        for path in all_json_paths:
            path_lower = path.lower()
            field = "Artificial Intelligence"  # Default
            
            # Determine field from path
            if any(venue in path_lower for venue in ["acl", "emnlp", "naacl", "conll", "cs.cl"]):
                field = "Natural Language Processing"
            elif any(venue in path_lower for venue in ["cvpr", "eccv", "iccv", "cs.cv"]):
                field = "Computer Vision"
            elif any(venue in path_lower for venue in ["sigmod", "vldb", "kdd", "cs.ds"]):
                field = "Data Science and Analytics"
            elif any(venue in path_lower for venue in ["chi", "uist", "cs.hc"]):
                field = "Human-Computer Interaction"
            elif any(venue in path_lower for venue in ["cs.ro"]):
                field = "Robotics and Control Systems"
            elif any(venue in path_lower for venue in ["cs.cr"]):
                field = "Cybersecurity and Privacy"
            elif any(venue in path_lower for venue in ["cs.et"]):
                field = "AI Ethics and Fairness"
            elif any(venue in path_lower for venue in ["cs.cc"]):
                field = "Theoretical Computer Science"
            elif any(venue in path_lower for venue in ["sigcomm", "cs.se"]):
                field = "Computer Systems and Architecture"
            
            if field not in paths_by_field:
                paths_by_field[field] = []
            paths_by_field[field].append(path)
        
        # Sample papers from each field proportionally
        target_fields = [
            "Artificial Intelligence", "Natural Language Processing", "Computer Vision",
            "Data Science and Analytics", "Human-Computer Interaction", "Robotics and Control Systems",
            "Cybersecurity and Privacy", "AI Ethics and Fairness", "Theoretical Computer Science",
            "Computer Systems and Architecture"
        ]
        
        papers_per_field = max(1, limit // len(target_fields))  # At least 1 paper per field
        selected_paths = []
        
        for field in target_fields:
            if field in paths_by_field:
                field_paths = paths_by_field[field]
                random.shuffle(field_paths)
                selected_paths.extend(field_paths[:papers_per_field])
        
        # If we haven't reached the limit, add more from available fields
        remaining_limit = limit - len(selected_paths)
        if remaining_limit > 0:
            all_remaining = []
            for field, paths in paths_by_field.items():
                all_remaining.extend(paths[papers_per_field:])  # Skip already selected
            random.shuffle(all_remaining)
            selected_paths.extend(all_remaining[:remaining_limit])
        
        logger.info(f"Selected {len(selected_paths)} papers using stratified sampling across {len(paths_by_field)} fields")
        
        for json_path in selected_paths:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    paper_data = json.load(f)
                
                # Extract original PeerRead paper ID from filename
                original_id = os.path.basename(json_path).replace('.pdf.json', '')
                
                # Extract paper metadata
                metadata = paper_data.get('metadata', {})
                
                # Skip papers without title or with duplicate titles
                title = metadata.get('title')
                if not title:
                    continue
                
                if title in processed_titles:
                    continue
                
                processed_titles.add(title)
                
                # Extract authors
                authors = metadata.get('authors', [])
                if not authors:
                    authors = ["Unknown"]
                
                # Extract and normalize field/venue
                raw_venue = (metadata.get('venue', "") or 
                            metadata.get('conference', "") or 
                            metadata.get('area', "unknown")).lower().strip()
                
                # Try to extract conference name from path if venue is unknown
                if raw_venue == "unknown":
                    path_lower = json_path.lower()
                    if "iclr" in path_lower:
                        raw_venue = "iclr"
                    elif "acl" in path_lower:
                        raw_venue = "acl"
                    elif "nips" in path_lower:
                        raw_venue = "nips"
                    elif "conll" in path_lower:
                        raw_venue = "conll"
                    elif "arxiv.cs.ai" in path_lower:
                        raw_venue = "cs.ai"
                    elif "arxiv.cs.cl" in path_lower:
                        raw_venue = "cs.cl"
                    elif "arxiv.cs.lg" in path_lower:
                        raw_venue = "cs.lg"
                    elif "arxiv.cs.cv" in path_lower:
                        raw_venue = "cs.cv"
                
                # Map raw venue to normalized field using field_mapping
                # Also check if any key is contained in the raw_venue string
                field = None
                for venue_key, field_value in field_mapping.items():
                    if venue_key in raw_venue:
                        field = field_value
                        break
                
                # Fallback if no mapping found
                if not field:
                    field = field_mapping.get(raw_venue, raw_venue)
                
                # Extract year for publication date
                year = metadata.get('year')
                publication_date = f"{year}-01-01" if year else None
                
                # Extract abstract
                abstract = metadata.get('abstractText', "")
                
                # Extract content from sections
                content = ""
                sections = metadata.get('sections') or []
                for section in sections:
                    if section and section.get('heading'):
                        content += f"\n## {section.get('heading')}\n"
                    if section:
                        content += section.get('text', '') + "\n"
                
                # If no content was extracted, try pdf_parse.body_text
                if not content and 'pdf_parse' in paper_data:
                    content = paper_data['pdf_parse'].get('body_text', '')
                
                # Create paper object with original PeerRead ID
                paper = {
                    "id": f"peerread_{original_id}",  # Use original PeerRead ID with prefix
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "keywords": [],  # PeerRead doesn't have keywords
                    "field": field,
                    "raw_venue": raw_venue,  # Keep the original venue for reference
                    "status": "submitted",  # Default status
                    "content": content,
                    "publication_date": publication_date,
                    "citations": 0,
                    "reviews": [],
                    "owner_id": "Imported_PeerRead",
                    "review_requests": []
                }
                
                # Add paper directly to database with original ID
                self.papers[paper["id"]] = paper
                papers_loaded += 1
                
                # Log progress every 10 papers
                if papers_loaded % 10 == 0:
                    logger.info(f"Loaded {papers_loaded} papers...")
                
                # Break if limit reached
                if papers_loaded >= limit:
                    break
                
            except Exception as e:
                log_error_with_context(e, f"processing paper file {json_path}", logger)
        
        # Also process any review JSON files to add reviews to papers
        for review_path in glob.glob(os.path.join(folder_path, "**", "reviews", "*.json"), recursive=True):
            try:
                with open(review_path, 'r', encoding='utf-8') as f:
                    review_data = json.load(f)
                
                title = review_data.get('title')
                if not title or title not in processed_titles:
                    continue
                
                # Find the paper with this title
                paper_id = None
                for pid, paper in self.papers.items():
                    if paper.get('title') == title:
                        paper_id = pid
                        break
                
                if not paper_id:
                    continue
                
                # Add reviews if available
                for review in review_data.get('reviews', []):
                    if 'comments' in review and review['comments'].strip():
                        review_obj = {
                            "reviewer_id": "Imported_Reviewer",
                            "text": review.get('comments', ''),
                            "rating": None,
                            "timestamp": datetime.now().isoformat()
                        }
                        self.add_review(paper_id, review_obj)
            
            except Exception as e:
                log_error_with_context(e, f"processing review file {review_path}", logger)
        
        # Log field distribution for loaded papers
        field_counts = {}
        for paper in self.papers.values():
            field = paper.get('field', 'Unknown')
            field_counts[field] = field_counts.get(field, 0) + 1
        
        logger.info("Field distribution of loaded papers:")
        for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {field}: {count} papers")
        
        logger.info(f"Successfully loaded {papers_loaded} papers from PeerRead dataset")
        self._save_data()
    
    def _save_data(self):
        """Save data to the JSON file."""
        # Skip saving for in-memory databases
        if self.data_path == ':memory:':
            return
            
        data = {
            'papers': self.papers,
            'next_id': self.next_id
        }
        
        try:
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved paper data to {self.data_path}")
        except (IOError, OSError) as e:
            error = FileOperationError("save", self.data_path, e)
            log_error_with_context(error, "saving paper data", logger)
            raise error
        except Exception as e:
            log_error_with_context(e, f"saving paper data to {self.data_path}", logger)
            raise
    
    def add_paper(self, paper_data: Dict[str, Any]) -> str:
        """
        Add a new paper to the database.
        
        Args:
            paper_data: Dictionary with paper data
            
        Returns:
            ID of the newly added paper
            
        Raises:
            ValidationError: If paper data is invalid
        """
        # Validate paper data
        validated_data = validate_paper_data(paper_data)
        
        # Create a new paper ID
        paper_id = f"paper_{self.next_id:03}"
        self.next_id += 1
        
        # Create complete paper entry with default values
        paper = PAPER_SCHEMA.copy()
        for key, value in validated_data.items():
            if key in paper:
                paper[key] = value
        
        # Set the ID
        paper['id'] = paper_id
        
        # Add to the database
        self.papers[paper_id] = paper
        
        logger.info(f"Added paper {paper_id}: {paper.get('title', 'Untitled')}")
        
        # Save to disk
        self._save_data()
        
        return paper_id
    
    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a paper by ID.
        
        Args:
            paper_id: ID of the paper to retrieve
            
        Returns:
            Paper data or None if not found
            
        Raises:
            ValidationError: If paper_id is invalid
        """
        # Validate paper ID
        paper_id = validate_paper_id(paper_id)
        
        return self.papers.get(paper_id)
    
    def update_paper(self, paper_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a paper's information.
        
        Args:
            paper_id: ID of the paper to update
            updates: Dictionary with fields to update
            
        Returns:
            True if update successful, False otherwise
        """
        if paper_id not in self.papers:
            return False
        
        for key, value in updates.items():
            if key in self.papers[paper_id]:
                self.papers[paper_id][key] = value
        
        self._save_data()
        return True
    
    def delete_paper(self, paper_id: str) -> bool:
        """
        Delete a paper from the database.
        
        Args:
            paper_id: ID of the paper to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        if paper_id not in self.papers:
            return False
        
        del self.papers[paper_id]
        self._save_data()
        return True
    
    def get_papers_by_field(self, field: str) -> List[Dict[str, Any]]:
        """
        Get papers by field.
        
        Args:
            field: Field to filter by
            
        Returns:
            List of matching papers
        """
        return [paper for paper in self.papers.values() if paper.get('field') == field]
    
    def get_papers_by_author(self, author: str) -> List[Dict[str, Any]]:
        """
        Get papers by author.
        
        Args:
            author: Author name to filter by
            
        Returns:
            List of matching papers
        """
        return [paper for paper in self.papers.values() if author in paper.get('authors', [])]
    
    def get_papers_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Get papers by status.
        
        Args:
            status: Status to filter by
            
        Returns:
            List of matching papers
        """
        return [paper for paper in self.papers.values() if paper.get('status') == status]
    
    def get_papers_by_owner(self, owner_id: str) -> List[Dict[str, Any]]:
        """
        Get papers by owner.
        
        Args:
            owner_id: Owner ID to filter by
            
        Returns:
            List of matching papers
        """
        return [paper for paper in self.papers.values() if paper.get('owner_id') == owner_id]
    
    def add_review(self, paper_id: str, review: Dict[str, Any]) -> bool:
        """
        Add a review to a paper.
        
        Args:
            paper_id: ID of the paper to add review to
            review: Review data
            
        Returns:
            True if addition successful, False otherwise
        """
        if paper_id not in self.papers:
            return False
        
        self.papers[paper_id]['reviews'].append(review)
        self._save_data()
        return True
    
    def add_review_request(self, paper_id: str, request: Dict[str, Any]) -> bool:
        """
        Add a review request to a paper.
        
        Args:
            paper_id: ID of the paper to add review request to
            request: Review request data
            
        Returns:
            True if addition successful, False otherwise
        """
        if paper_id not in self.papers:
            return False
        
        self.papers[paper_id]['review_requests'].append(request)
        self._save_data()
        return True
    
    def search_papers(self, query: str) -> List[Dict[str, Any]]:
        """
        Search papers by query string (in title, abstract, keywords).
        
        Args:
            query: Query string
            
        Returns:
            List of matching papers
        """
        query = query.lower()
        results = []
        
        for paper in self.papers.values():
            # Search in title
            if query in paper.get('title', '').lower():
                results.append(paper)
                continue
            
            # Search in abstract
            if query in paper.get('abstract', '').lower():
                results.append(paper)
                continue
            
            # Search in keywords
            if any(query in keyword.lower() for keyword in paper.get('keywords', [])):
                results.append(paper)
                continue
        
        return results
    
    def get_all_papers(self) -> List[Dict[str, Any]]:
        """
        Get all papers in the database.
        
        Returns:
            List of all papers
        """
        return list(self.papers.values())
    
    def update_review_request_status(self, paper_id: str, reviewer_id: str, new_status: str) -> bool:
        """
        Update the status of a review request for a given paper and reviewer.
        
        Args:
            paper_id: ID of the paper
            reviewer_id: ID of the reviewer
            new_status: New status for the review request (e.g., 'accepted', 'declined', 'completed')
            
        Returns:
            True if the update was successful, False otherwise
        """
        paper = self.get_paper(paper_id)
        if not paper:
            print(f"Paper {paper_id} not found for review status update")
            return False
            
        # Search for the review request
        request_found = False
        for request in paper.get('review_requests', []):
            if request.get('reviewer_id') == reviewer_id and request.get('status') in ['pending', 'invited']:
                request['status'] = new_status
                request_found = True
                break
                
        # If not found in review_requests, try review_invitations
        if not request_found and 'review_invitations' in paper:
            for invitation in paper['review_invitations']:
                if invitation.get('reviewer_id') == reviewer_id and invitation.get('status') == 'invited':
                    invitation['status'] = new_status
                    request_found = True
                    break
        
        if request_found:
            # Save the updated data
            self.update_paper(paper_id, paper)
            self._save_data()
            return True
        
        print(f"No pending review request found for reviewer {reviewer_id} on paper {paper_id}")
        return False 