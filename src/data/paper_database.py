"""
Paper Database module for managing research papers in the Peer Review simulation.

This module handles storage, retrieval, and management of research papers.
It includes functionality to load papers from the PeerRead dataset.
"""

import os
import json
import glob
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

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
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                    self.papers = data.get('papers', {})
                    self.next_id = data.get('next_id', 1)
                
                # If no papers loaded, try loading from PeerRead
                if not self.papers:
                    print("Papers database is empty. Loading from PeerRead test dataset...")
                    self.load_peerread_dataset(use_test_dataset=True)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading papers data: {e}")
                print("Loading from PeerRead test dataset instead...")
                self.load_peerread_dataset(use_test_dataset=True)
        else:
            # Initialize with PeerRead dataset
            print("Papers database not found. Loading from PeerRead test dataset...")
            self.load_peerread_dataset(use_test_dataset=True)
    
    def load_peerread_dataset(self, folder_path: str = None, limit: int = 100, use_test_dataset: bool = True):
        """
        Load papers from the PeerRead dataset.
        
        Args:
            folder_path: Path to the PeerRead data folder (if None, will use default paths)
            limit: Maximum number of papers to load (for testing)
            use_test_dataset: Whether to use the test dataset instead of the full dataset
        """
        # Define mapping of venue names to general research fields
        field_mapping = {
            "iclr": "AI",
            "icml": "AI",
            "nips": "AI",
            "neurips": "AI",
            "aaai": "AI",
            "ijcai": "AI",
            "aistats": "AI",
            "uai": "AI",
            
            "cvpr": "Vision",
            "eccv": "Vision",
            "iccv": "Vision",
            "wacv": "Vision",
            "bmvc": "Vision",
            
            "acl": "NLP",
            "emnlp": "NLP",
            "naacl": "NLP",
            "coling": "NLP",
            "conll": "NLP",
            "eacl": "NLP",
            "tacl": "NLP",
            "cl": "NLP",
            "arxiv.cs.cl": "NLP",
            
            "sigmod": "Databases",
            "vldb": "Databases",
            "icde": "Databases",
            "kdd": "Data Mining",
            
            "chi": "HCI",
            "uist": "HCI",
            "cscw": "HCI",
            
            "siggraph": "Graphics",
            "eurographics": "Graphics",
            
            "www": "Web",
            "wsdm": "Web",
            
            "sigcomm": "Networks",
            "imc": "Networks",
            "infocom": "Networks",
            
            "emnlp": "NLP"
        }
        
        # Determine the dataset path
        if folder_path is None:
            if use_test_dataset:
                # Use the smaller test dataset
                folder_path = os.path.abspath("test_dataset")
                if not os.path.exists(folder_path):
                    folder_path = os.path.abspath("PeerReview/test_dataset")
            else:
                # Use the full PeerRead dataset
                folder_path = os.path.abspath("PeerRead/data")
        
        if not os.path.exists(folder_path):
            print(f"PeerRead dataset folder '{folder_path}' not found")
            return
        
        print(f"Loading papers from {'test' if use_test_dataset else 'full'} PeerRead dataset at '{folder_path}'...")
        
        # Counter for loaded papers
        papers_loaded = 0
        processed_titles = set()  # To avoid duplicates
        
        # Process all PDF JSON files in parsed_pdfs directories
        for json_path in glob.glob(os.path.join(folder_path, "**", "parsed_pdfs", "*.pdf.json"), recursive=True):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    paper_data = json.load(f)
                
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
                if raw_venue == "unknown" and "iclr" in json_path.lower():
                    raw_venue = "iclr"
                elif raw_venue == "unknown" and "acl" in json_path.lower():
                    raw_venue = "acl"
                elif raw_venue == "unknown" and "nips" in json_path.lower():
                    raw_venue = "nips"
                
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
                for section in metadata.get('sections', []):
                    if section.get('heading'):
                        content += f"\n## {section.get('heading')}\n"
                    content += section.get('text', '') + "\n"
                
                # If no content was extracted, try pdf_parse.body_text
                if not content and 'pdf_parse' in paper_data:
                    content = paper_data['pdf_parse'].get('body_text', '')
                
                # Create paper object
                paper = {
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
                
                # Add paper to database
                self.add_paper(paper)
                papers_loaded += 1
                
                # Print progress every 10 papers
                if papers_loaded % 10 == 0:
                    print(f"Loaded {papers_loaded} papers...")
                
                # Break if limit reached
                if papers_loaded >= limit:
                    break
                
            except Exception as e:
                print(f"Error processing {json_path}: {e}")
        
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
                print(f"Error processing review {review_path}: {e}")
        
        # Print field distribution for loaded papers
        field_counts = {}
        for paper in self.papers.values():
            field = paper.get('field', 'Unknown')
            field_counts[field] = field_counts.get(field, 0) + 1
        
        print("\nField distribution of loaded papers:")
        for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {field}: {count} papers")
        
        print(f"\nLoaded {papers_loaded} papers from PeerRead dataset")
        self._save_data()
    
    def _save_data(self):
        """Save data to the JSON file."""
        data = {
            'papers': self.papers,
            'next_id': self.next_id
        }
        
        try:
            with open(self.data_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving paper data: {e}")
    
    def add_paper(self, paper_data: Dict[str, Any]) -> str:
        """
        Add a new paper to the database.
        
        Args:
            paper_data: Dictionary with paper data
            
        Returns:
            ID of the newly added paper
        """
        # Create a new paper ID
        paper_id = f"paper_{self.next_id:03}"
        self.next_id += 1
        
        # Create complete paper entry with default values
        paper = PAPER_SCHEMA.copy()
        for key, value in paper_data.items():
            if key in paper:
                paper[key] = value
        
        # Set the ID
        paper['id'] = paper_id
        
        # Add to the database
        self.papers[paper_id] = paper
        
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
        """
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