#!/usr/bin/env python3
"""
Debug script to check paper assignment issue
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.simulation.peer_review_simulation import PeerReviewSimulation

def debug_paper_assignment():
    """Debug the paper assignment issue."""
    print("🔍 Debugging Paper Assignment Issue")
    print("=" * 50)
    
    # Create simulation
    sim = PeerReviewSimulation()
    
    # Check papers before creating researchers
    papers = sim.paper_db.get_all_papers()
    print(f"Total papers in database: {len(papers)}")
    
    # Check owner_id distribution
    owner_ids = {}
    for paper in papers:
        owner_id = paper.get('owner_id', 'None')
        owner_ids[owner_id] = owner_ids.get(owner_id, 0) + 1
    
    print("Owner ID distribution:")
    for owner_id, count in owner_ids.items():
        print(f"  {owner_id}: {count} papers")
    
    # Create researchers
    print(f"\nCreating researchers...")
    sim.create_all_researchers(assign_papers=False)  # Don't assign yet
    print(f"Created {len(sim.agents)} researchers")
    
    # Check papers with Imported_PeerRead owner_id
    imported_papers = [p for p in papers if p.get('owner_id') == 'Imported_PeerRead']
    print(f"\nPapers with 'Imported_PeerRead' owner_id: {len(imported_papers)}")
    
    if imported_papers:
        print("Sample imported paper:")
        sample = imported_papers[0]
        print(f"  ID: {sample.get('id')}")
        print(f"  Title: {sample.get('title', 'No title')[:50]}...")
        print(f"  Owner ID: {sample.get('owner_id')}")
        print(f"  Field: {sample.get('field')}")
    
    # Now try assignment
    print(f"\nTrying paper assignment...")
    sim.assign_imported_papers_to_agents()
    
    # Check final distribution
    print(f"\nFinal paper distribution:")
    for agent_name in sim.agents.keys():
        agent_papers = sim.paper_db.get_papers_by_owner(agent_name)
        print(f"{agent_name}: {len(agent_papers)} papers")

if __name__ == "__main__":
    debug_paper_assignment()