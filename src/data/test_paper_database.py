"""
Test script for the PaperDatabase class with PeerRead dataset.

This script tests the functionality of the PaperDatabase class
with the PeerRead dataset integration.
"""

import os
import sys
import glob
from collections import Counter

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the PaperDatabase class
from src.data.paper_database import PaperDatabase

def test_paper_database():
    """Test the PaperDatabase class with PeerRead dataset."""
    print("Testing PaperDatabase with PeerRead integration...\n")
    
    # Create a temporary database
    temp_db_path = "temp_test_papers.json"
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)
    
    # Initialize the database
    print("Initializing database...")
    db = PaperDatabase(data_path=temp_db_path)
    
    # Test if papers are automatically loaded
    papers = db.get_all_papers()
    if papers:
        print(f"Database automatically loaded {len(papers)} papers.")
    else:
        print("No papers loaded automatically. Loading from test dataset...")
        
        # Check if test dataset exists
        test_dataset_path = os.path.abspath("../../test_dataset")
        if os.path.exists(test_dataset_path):
            db.load_peerread_dataset(folder_path=test_dataset_path)
        else:
            # Create test dataset directory if it doesn't exist
            os.makedirs("../../test_dataset", exist_ok=True)
            
            # Check if full PeerRead dataset exists
            peerread_path = os.path.abspath("../../PeerRead/data")
            if os.path.exists(peerread_path):
                # Count how many JSON files are in the dataset
                json_files = glob.glob(os.path.join(peerread_path, "**", "parsed_pdfs", "*.pdf.json"), recursive=True)
                print(f"Found {len(json_files)} papers in full PeerRead dataset.")
                
                # Ask if user wants to load full dataset
                if len(json_files) > 0:
                    choice = input("Do you want to load papers from the full dataset? This may take a while. (y/n): ")
                    if choice.lower() == 'y':
                        db.load_peerread_dataset(folder_path=peerread_path, use_test_dataset=False, limit=20)
                    else:
                        print("Skipping dataset loading.")
            else:
                print(f"PeerRead dataset not found at {peerread_path}")
                print("Please download it or create a test dataset.")
                return
    
    # Get all papers and print statistics
    papers = db.get_all_papers()
    print(f"\nLoaded {len(papers)} papers.")
    
    # Print statistics about the papers
    if papers:
        # Count papers by field
        fields = Counter([paper.get('field', 'Unknown') for paper in papers])
        print("\nPapers by field:")
        for field, count in fields.most_common():
            print(f"  - {field}: {count}")
        
        # Count papers by status
        statuses = Counter([paper.get('status', 'Unknown') for paper in papers])
        print("\nPapers by status:")
        for status, count in statuses.most_common():
            print(f"  - {status}: {count}")
        
        # Count papers by author
        author_counts = Counter()
        for paper in papers:
            for author in paper.get('authors', []):
                author_counts[author] += 1
        
        print("\nTop 5 authors:")
        for author, count in author_counts.most_common(5):
            print(f"  - {author}: {count}")
        
        # Test search functionality
        print("\nTesting search functionality:")
        search_term = "neural"
        search_results = db.search_papers(search_term)
        print(f"Search for '{search_term}' returned {len(search_results)} results.")
        
        if search_results:
            print("\nSample search result:")
            sample = search_results[0]
            print(f"  Title: {sample.get('title')}")
            print(f"  Authors: {', '.join(sample.get('authors', []))}")
            print(f"  Field: {sample.get('field')}")
            abstract = sample.get('abstract', '')
            print(f"  Abstract: {abstract[:150]}..." if len(abstract) > 150 else f"  Abstract: {abstract}")
        
        # Test filtering by field
        if fields:
            print("\nTesting filtering by field:")
            test_field = fields.most_common(1)[0][0]
            field_results = db.get_papers_by_field(test_field)
            print(f"Filter by field '{test_field}' returned {len(field_results)} results.")
        
        # Test adding a paper
        print("\nTesting adding a paper:")
        new_paper = {
            "title": "Test Paper for PeerReview System",
            "authors": ["Test Author"],
            "abstract": "This is a test paper to check the functionality of the PaperDatabase class.",
            "keywords": ["test", "paper", "database"],
            "field": "Computer Science",
            "status": "draft",
            "content": "This is the content of the test paper.",
            "owner_id": "Test_Researcher"
        }
        
        paper_id = db.add_paper(new_paper)
        print(f"Added new paper with ID: {paper_id}")
        
        # Test retrieving the paper
        retrieved_paper = db.get_paper(paper_id)
        if retrieved_paper:
            print(f"Retrieved paper: {retrieved_paper.get('title')}")
        else:
            print("Failed to retrieve paper.")
        
        # Test updating the paper
        print("\nTesting updating a paper:")
        update_success = db.update_paper(paper_id, {"status": "submitted"})
        if update_success:
            print(f"Updated paper status to 'submitted'")
            updated_paper = db.get_paper(paper_id)
            print(f"New status: {updated_paper.get('status')}")
        else:
            print("Failed to update paper.")
        
        # Test deleting the paper
        print("\nTesting deleting a paper:")
        delete_success = db.delete_paper(paper_id)
        if delete_success:
            print(f"Deleted paper with ID: {paper_id}")
        else:
            print("Failed to delete paper.")
    
    # Clean up
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)
        print(f"\nRemoved temporary database file: {temp_db_path}")
    
    print("\nPaperDatabase testing completed.")

if __name__ == "__main__":
    test_paper_database() 