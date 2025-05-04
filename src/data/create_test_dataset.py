"""
Create a test dataset from PeerRead.

This script creates a smaller test dataset from the full PeerRead dataset
to make it easier to test the PaperDatabase implementation.
"""

import os
import glob
import json
import shutil
import random
from pathlib import Path

# Configuration
SOURCE_DIR = "../PeerRead/data"
TARGET_DIR = "../../test_dataset"
PAPERS_PER_CONFERENCE = 5  # Number of papers to select from each conference
INCLUDE_REVIEWS = True

def create_test_dataset():
    """Create a test dataset from PeerRead."""
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory '{SOURCE_DIR}' not found.")
        return

    print(f"Creating test dataset from '{SOURCE_DIR}' to '{TARGET_DIR}'")
    
    # Create target directory
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Find all conferences
    conferences = [d for d in os.listdir(SOURCE_DIR) 
                   if os.path.isdir(os.path.join(SOURCE_DIR, d)) and not d.startswith('.')]
    
    for conference in conferences:
        print(f"Processing conference: {conference}")
        conf_source_dir = os.path.join(SOURCE_DIR, conference)
        conf_target_dir = os.path.join(TARGET_DIR, conference)
        
        # Create conference directory
        os.makedirs(conf_target_dir, exist_ok=True)
        
        # Find all splits (train, test, dev)
        splits = []
        for split_dir in ['train', 'test', 'dev']:
            if os.path.exists(os.path.join(conf_source_dir, split_dir)):
                splits.append(split_dir)
        
        # Process each split
        for split in splits:
            print(f"  Processing split: {split}")
            split_source_dir = os.path.join(conf_source_dir, split)
            split_target_dir = os.path.join(conf_target_dir, split)
            
            # Create split directory
            os.makedirs(split_target_dir, exist_ok=True)
            
            # Create parsed_pdfs directory
            parsed_pdfs_source_dir = os.path.join(split_source_dir, 'parsed_pdfs')
            parsed_pdfs_target_dir = os.path.join(split_target_dir, 'parsed_pdfs')
            os.makedirs(parsed_pdfs_target_dir, exist_ok=True)
            
            # Find all PDF JSON files in parsed_pdfs directory
            pdf_json_files = glob.glob(os.path.join(parsed_pdfs_source_dir, '*.pdf.json'))
            
            # Select a random subset of papers
            selected_papers = random.sample(pdf_json_files, 
                                           min(PAPERS_PER_CONFERENCE, len(pdf_json_files)))
            
            # Copy selected papers to target directory
            for paper_path in selected_papers:
                paper_filename = os.path.basename(paper_path)
                target_path = os.path.join(parsed_pdfs_target_dir, paper_filename)
                shutil.copy2(paper_path, target_path)
                print(f"    Copied paper: {paper_filename}")
                
                # Extract paper ID for review matching
                paper_id = paper_filename.split('.')[0]
                
                # Copy corresponding reviews if available
                if INCLUDE_REVIEWS:
                    reviews_source_dir = os.path.join(split_source_dir, 'reviews')
                    if os.path.exists(reviews_source_dir):
                        reviews_target_dir = os.path.join(split_target_dir, 'reviews')
                        os.makedirs(reviews_target_dir, exist_ok=True)
                        
                        review_path = os.path.join(reviews_source_dir, f"{paper_id}.json")
                        if os.path.exists(review_path):
                            target_review_path = os.path.join(reviews_target_dir, f"{paper_id}.json")
                            shutil.copy2(review_path, target_review_path)
                            print(f"    Copied review: {paper_id}.json")

    print("Test dataset created successfully!")
    print(f"You can now use the test dataset at '{TARGET_DIR}' to test your implementation.")

if __name__ == "__main__":
    create_test_dataset() 