#!/usr/bin/env python3
"""
Simple test script to verify database migration utility works correctly.
"""

import sys
import json
import tempfile
from pathlib import Path

sys.path.append('.')

from src.data.enhanced_models import DatabaseMigrationUtility

def test_migration_utility():
    """Test database migration utility."""
    print("Testing DatabaseMigrationUtility...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        utility = DatabaseMigrationUtility(temp_dir)
        
        # Test initialization
        assert utility.data_directory == Path(temp_dir)
        assert utility.backup_directory.exists()
        
        # Create old format researcher data
        old_data = {
            "researchers": {
                "researcher_1": {
                    "name": "Dr. Smith",
                    "specialty": "AI",
                    "level": "Full Prof",
                    "h_index": 20,
                    "citations": 500,
                    "years_active": 15
                },
                "researcher_2": {
                    "name": "Dr. Jones",
                    "specialty": "ML",
                    "level": "Assistant Prof",
                    "h_index": 8,
                    "citations": 150,
                    "years_active": 3
                }
            }
        }
        
        old_file = Path(temp_dir) / "old_researchers.json"
        with open(old_file, 'w') as f:
            json.dump(old_data, f)
        
        new_file = Path(temp_dir) / "new_researchers.json"
        
        # Perform migration
        utility.migrate_researchers(str(old_file), str(new_file))
        
        # Verify migration
        assert new_file.exists()
        with open(new_file, 'r') as f:
            migrated_data = json.load(f)
        
        assert "researchers" in migrated_data
        assert "researcher_1" in migrated_data["researchers"]
        assert "researcher_2" in migrated_data["researchers"]
        
        # Check researcher 1 data
        r1_data = migrated_data["researchers"]["researcher_1"]
        assert r1_data["name"] == "Dr. Smith"
        assert r1_data["level"] == "Full Prof"
        assert r1_data["h_index"] == 20
        assert r1_data["total_citations"] == 500
        assert r1_data["years_active"] == 15
        assert "reputation_score" in r1_data
        
        # Check researcher 2 data
        r2_data = migrated_data["researchers"]["researcher_2"]
        assert r2_data["name"] == "Dr. Jones"
        assert r2_data["level"] == "Assistant Prof"
        assert r2_data["h_index"] == 8
        
        print("✓ Researcher migration test passed")
        
        # Test paper migration
        old_paper_data = {
            "papers": {
                "paper_1": {
                    "title": "AI Paper",
                    "authors": ["Dr. Smith"],
                    "reviews": [
                        {
                            "reviewer_id": "reviewer_1",
                            "rating": 7.5,
                            "text": "Good paper with strengths and some weaknesses",
                            "confidence": 4
                        },
                        {
                            "reviewer_id": "reviewer_2",
                            "rating": 6.0,
                            "text": "Decent work but needs improvement",
                            "confidence": 3
                        }
                    ]
                }
            }
        }
        
        old_paper_file = Path(temp_dir) / "old_papers.json"
        with open(old_paper_file, 'w') as f:
            json.dump(old_paper_data, f)
        
        new_paper_file = Path(temp_dir) / "new_papers.json"
        
        # Perform paper migration
        utility.migrate_papers(str(old_paper_file), str(new_paper_file))
        
        # Verify paper migration
        assert new_paper_file.exists()
        with open(new_paper_file, 'r') as f:
            migrated_papers = json.load(f)
        
        assert "papers" in migrated_papers
        assert "paper_1" in migrated_papers["papers"]
        
        paper_data = migrated_papers["papers"]["paper_1"]
        assert len(paper_data["reviews"]) == 2
        
        # Check that reviews were converted to enhanced format
        review_data = paper_data["reviews"][0]
        assert "criteria_scores" in review_data
        assert "executive_summary" in review_data
        assert "confidence_level" in review_data
        assert "reviewer_id" in review_data
        
        print("✓ Paper migration test passed")

def main():
    """Run migration tests."""
    print("Running database migration tests...\n")
    
    try:
        test_migration_utility()
        
        print("\n🎉 All migration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Migration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)