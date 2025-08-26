#!/usr/bin/env python3
"""
Simple integration test for PeerRead dataset integration.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.peerread_loader import PeerReadLoader, PeerReadReview


def create_test_dataset():
    """Create a minimal test dataset for testing."""
    temp_dir = tempfile.mkdtemp()
    peerread_path = Path(temp_dir) / "PeerRead"
    data_path = peerread_path / "data"
    
    # Create ACL venue structure
    acl_path = data_path / "acl_2017" / "train"
    acl_reviews_path = acl_path / "reviews"
    acl_reviews_path.mkdir(parents=True)
    
    # Create sample review file
    sample_review = {
        "id": "test_paper_1",
        "title": "Enhanced Neural Networks for NLP",
        "abstract": "This paper presents novel approaches to neural networks.",
        "authors": ["Alice Smith", "Bob Johnson"],
        "accepted": True,
        "reviews": [
            {
                "IMPACT": "4",
                "SUBSTANCE": "4",
                "SOUNDNESS_CORRECTNESS": "3",
                "ORIGINALITY": "4",
                "CLARITY": "3",
                "MEANINGFUL_COMPARISON": "3",
                "comments": "This paper presents interesting ideas.",
                "is_meta_review": False
            }
        ]
    }
    
    with open(acl_reviews_path / "test_paper_1.json", 'w') as f:
        json.dump(sample_review, f, indent=2)
    
    return str(peerread_path), temp_dir


def test_basic_functionality():
    """Test basic PeerRead loader functionality."""
    print("Testing PeerRead Loader...")
    
    # Create test dataset
    peerread_path, temp_dir = create_test_dataset()
    
    try:
        # Initialize loader
        loader = PeerReadLoader(peerread_path)
        print(f"✓ Loader initialized successfully")
        
        # Load ACL venue
        venue_chars = loader.load_venue("acl_2017")
        print(f"✓ Venue loaded: {venue_chars.name if venue_chars else 'None'}")
        
        if venue_chars:
            print(f"  - Type: {venue_chars.venue_type}")
            print(f"  - Field: {venue_chars.field}")
            print(f"  - Total papers: {venue_chars.total_papers}")
        
        # Test paper loading
        papers = loader.get_papers_by_venue("ACL")
        print(f"✓ Papers loaded: {len(papers)}")
        
        if papers:
            paper = papers[0]
            print(f"  - Paper ID: {paper.id}")
            print(f"  - Title: {paper.title}")
            
            if paper.reviews:
                review = paper.reviews[0]
                print(f"  - Review Impact: {review.impact} -> Significance: {review.significance}")
                print(f"  - Review Substance: {review.substance} -> Technical Quality: {review.technical_quality}")
                print(f"  - Review Originality: {review.originality} -> Novelty: {review.novelty}")
                print(f"  - Review Clarity: {review.clarity} -> Clarity Mapped: {review.clarity_mapped}")
                
                # Debug: Check if _map_dimensions was called
                print(f"  - Debug: All attributes: {[attr for attr in dir(review) if not attr.startswith('_')]}")
        
        print("\n✅ Basic test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_basic_functionality()