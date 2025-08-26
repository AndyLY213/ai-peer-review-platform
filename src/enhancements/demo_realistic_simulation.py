"""
Demonstration of the Enhanced Realistic Peer Review Simulation.

This script shows how the enhanced system works with:
- Hierarchical researchers
- Venue-specific standards
- Biased review generation
- Temporal dynamics
"""

from realistic_review_system import (
    RealisticReviewSystem, Venue, EnhancedResearcher,
    VenueType, ResearcherLevel, ReviewDecision
)
from datetime import datetime
import random

def create_demo_venues():
    """Create sample venues with different characteristics."""
    venues = [
        Venue(
            id="neurips",
            name="NeurIPS",
            venue_type=VenueType.TOP_CONFERENCE,
            field="Artificial Intelligence",
            acceptance_rate=0.05,
            prestige_score=10,
            review_deadline_weeks=6,
            min_reviewers=3,
            min_review_length=800,
            requires_detailed_scores=True,
            allows_borderline_reviews=False
        ),
        Venue(
            id="icml",
            name="ICML",
            venue_type=VenueType.TOP_CONFERENCE,
            field="Artificial Intelligence",
            acceptance_rate=0.08,
            prestige_score=9,
            review_deadline_weeks=6,
            min_reviewers=3,
            min_review_length=700,
            requires_detailed_scores=True,
            allows_borderline_reviews=True
        ),
        Venue(
            id="aaai",
            name="AAAI",
            venue_type=VenueType.MID_CONFERENCE,
            field="Artificial Intelligence",
            acceptance_rate=0.25,
            prestige_score=7,
            review_deadline_weeks=4,
            min_reviewers=2,
            min_review_length=500,
            requires_detailed_scores=True,
            allows_borderline_reviews=True
        ),
        Venue(
            id="nature_ai",
            name="Nature Machine Intelligence",
            venue_type=VenueType.TOP_JOURNAL,
            field="Artificial Intelligence",
            acceptance_rate=0.02,
            prestige_score=10,
            review_deadline_weeks=8,
            min_reviewers=3,
            min_review_length=1000,
            requires_detailed_scores=True,
            allows_borderline_reviews=False
        )
    ]
    return venues

def create_demo_researchers():
    """Create sample researchers with different characteristics."""
    researchers = [
        # Senior researcher at top institution
        EnhancedResearcher(
            id="prof_smith",
            name="Prof. Sarah Smith",
            level=ResearcherLevel.FULL_PROF,
            institution_tier=1,  # Top tier (MIT, Stanford, etc.)
            specialty="Artificial Intelligence",
            h_index=85,
            total_citations=12000,
            years_active=20,
            review_quality_score=0.9,
            review_speed=0.8,
            review_harshness=0.6,
            prestige_bias=0.3,
            novelty_bias=0.8,
            confirmation_bias=0.4,
            current_workload=2,
            max_reviews_per_month=8,
            is_available=True
        ),
        
        # Mid-career researcher at good institution
        EnhancedResearcher(
            id="dr_jones",
            name="Dr. Michael Jones",
            level=ResearcherLevel.ASSOCIATE_PROF,
            institution_tier=2,  # Good institution
            specialty="Artificial Intelligence",
            h_index=35,
            total_citations=3500,
            years_active=12,
            review_quality_score=0.8,
            review_speed=0.9,
            review_harshness=0.7,
            prestige_bias=0.5,
            novelty_bias=0.6,
            confirmation_bias=0.5,
            current_workload=1,
            max_reviews_per_month=6,
            is_available=True
        ),
        
        # Junior researcher
        EnhancedResearcher(
            id="dr_chen",
            name="Dr. Lisa Chen",
            level=ResearcherLevel.ASSISTANT_PROF,
            institution_tier=2,
            specialty="Artificial Intelligence",
            h_index=15,
            total_citations=800,
            years_active=5,
            review_quality_score=0.7,
            review_speed=0.95,  # Very punctual
            review_harshness=0.8,  # Harsh reviewer
            prestige_bias=0.7,  # High prestige bias
            novelty_bias=0.9,  # Loves novel work
            confirmation_bias=0.3,
            current_workload=0,
            max_reviews_per_month=4,
            is_available=True
        ),
        
        # Postdoc - eager but inexperienced
        EnhancedResearcher(
            id="postdoc_kim",
            name="Dr. James Kim",
            level=ResearcherLevel.POSTDOC,
            institution_tier=1,
            specialty="Artificial Intelligence",
            h_index=8,
            total_citations=200,
            years_active=3,
            review_quality_score=0.6,  # Still learning
            review_speed=0.7,  # Sometimes late
            review_harshness=0.5,
            prestige_bias=0.8,  # Very influenced by prestige
            novelty_bias=0.7,
            confirmation_bias=0.6,
            current_workload=0,
            max_reviews_per_month=3,
            is_available=True
        ),
        
        # Graduate student - limited experience
        EnhancedResearcher(
            id="grad_wilson",
            name="Alex Wilson",
            level=ResearcherLevel.GRADUATE_STUDENT,
            institution_tier=2,
            specialty="Artificial Intelligence",
            h_index=3,
            total_citations=50,
            years_active=2,
            review_quality_score=0.5,
            review_speed=0.6,
            review_harshness=0.4,
            prestige_bias=0.9,  # Very impressed by big names
            novelty_bias=0.8,
            confirmation_bias=0.7,
            current_workload=0,
            max_reviews_per_month=2,
            is_available=True
        )
    ]
    return researchers

def create_sample_papers():
    """Create sample papers with different characteristics."""
    papers = [
        {
            'id': 'paper_001',
            'title': 'Revolutionary Deep Learning Architecture for AGI',
            'authors': ['Prof. Sarah Smith', 'Dr. Famous Researcher'],
            'abstract': 'We present a groundbreaking architecture that achieves human-level performance...',
            'field': 'Artificial Intelligence',
            'keywords': ['deep learning', 'AGI', 'neural networks'],
            'novelty_level': 9,  # Very novel
            'technical_quality': 8,
            'author_prestige': 0.9  # High prestige authors
        },
        {
            'id': 'paper_002',
            'title': 'Incremental Improvements to Existing CNN Methods',
            'authors': ['Unknown Researcher', 'Another Unknown'],
            'abstract': 'We make small improvements to existing CNN architectures...',
            'field': 'Artificial Intelligence',
            'keywords': ['CNN', 'computer vision', 'optimization'],
            'novelty_level': 4,  # Low novelty
            'technical_quality': 6,
            'author_prestige': 0.2  # Unknown authors
        },
        {
            'id': 'paper_003',
            'title': 'Solid Contribution to Reinforcement Learning',
            'authors': ['Dr. Michael Jones', 'Collaborator'],
            'abstract': 'We present a solid contribution to RL with good experimental validation...',
            'field': 'Artificial Intelligence',
            'keywords': ['reinforcement learning', 'policy gradient', 'robotics'],
            'novelty_level': 6,  # Moderate novelty
            'technical_quality': 7,
            'author_prestige': 0.6  # Moderate prestige
        }
    ]
    return papers

def run_realistic_simulation():
    """Run a demonstration of the realistic peer review simulation."""
    print("🔬 Enhanced Peer Review Simulation Demo")
    print("=" * 50)
    
    # Initialize system
    system = RealisticReviewSystem()
    
    # Add venues
    venues = create_demo_venues()
    for venue in venues:
        system.add_venue(venue)
        print(f"Added venue: {venue.name} ({venue.venue_type.value}, {venue.acceptance_rate*100:.1f}% acceptance)")
    
    print()
    
    # Add researchers
    researchers = create_demo_researchers()
    for researcher in researchers:
        system.add_researcher(researcher)
        rep_mult = researcher.get_reputation_multiplier()
        print(f"Added researcher: {researcher.name} ({researcher.level.value}, "
              f"h-index: {researcher.h_index}, reputation: {rep_mult:.2f})")
    
    print()
    
    # Create and submit papers
    papers = create_sample_papers()
    submissions = []
    
    for i, paper in enumerate(papers):
        # Submit to different venues
        venue_choices = ["neurips", "icml", "aaai"]
        venue_id = venue_choices[i % len(venue_choices)]
        
        # Determine author IDs (simplified - using researcher IDs)
        if i == 0:
            author_ids = ["prof_smith"]  # High prestige
        elif i == 1:
            author_ids = ["grad_wilson"]  # Low prestige
        else:
            author_ids = ["dr_jones"]  # Medium prestige
        
        submission_id = system.submit_paper(paper, venue_id, author_ids)
        submissions.append(submission_id)
        
        venue_name = system.venues[venue_id].name
        print(f"📄 Submitted '{paper['title'][:50]}...' to {venue_name}")
        
        # Show assigned reviewers
        submission = system.active_submissions[submission_id]
        reviewer_names = [system.researchers[r_id].name for r_id in submission['assigned_reviewers']]
        print(f"   Assigned reviewers: {', '.join(reviewer_names)}")
    
    print("\n" + "=" * 50)
    print("🔍 Generating Reviews...")
    print("=" * 50)
    
    # Generate reviews for each submission
    for submission_id in submissions:
        submission = system.active_submissions[submission_id]
        paper = submission['paper']
        venue = system.venues[submission['venue_id']]
        
        print(f"\n📝 Reviews for '{paper['title'][:40]}...' at {venue.name}:")
        print("-" * 60)
        
        reviews = system.generate_reviews(submission_id)
        
        for review in reviews:
            reviewer = system.researchers[review.reviewer_id]
            print(f"\n👤 Reviewer: {reviewer.name} ({reviewer.level.value})")
            print(f"   Reputation multiplier: {reviewer.get_reputation_multiplier():.2f}")
            print(f"   Review quality: {reviewer.review_quality_score:.2f}")
            print(f"   Biases - Prestige: {reviewer.prestige_bias:.1f}, "
                  f"Novelty: {reviewer.novelty_bias:.1f}, Harsh: {reviewer.review_harshness:.1f}")
            
            print(f"\n📊 Scores:")
            print(f"   Novelty: {review.criteria.novelty}/10")
            print(f"   Technical Quality: {review.criteria.technical_quality}/10")
            print(f"   Clarity: {review.criteria.clarity}/10")
            print(f"   Significance: {review.criteria.significance}/10")
            print(f"   Overall Score: {review.criteria.overall_score():.2f}/10")
            print(f"   Confidence: {review.confidence}/5")
            
            print(f"\n💭 Decision: {review.recommendation.value.upper()}")
            print(f"   Review length: {review.review_length} words")
            print(f"   Time spent: {review.time_spent} minutes")
            print(f"   Late submission: {'Yes' if review.is_late else 'No'}")
            
            print(f"\n✅ Strengths:")
            for strength in review.strengths:
                print(f"   • {strength}")
            
            print(f"\n⚠️  Weaknesses:")
            for weakness in review.weaknesses:
                print(f"   • {weakness}")
            
            quality_ok = review.meets_quality_standards(venue.venue_type)
            print(f"\n🎯 Meets venue standards: {'Yes' if quality_ok else 'No'}")
    
    print("\n" + "=" * 50)
    print("⚖️  Final Decisions...")
    print("=" * 50)
    
    # Make final decisions
    for submission_id in submissions:
        submission = system.active_submissions[submission_id]
        paper = submission['paper']
        venue = system.venues[submission['venue_id']]
        
        decision = system.make_decision(submission_id)
        
        print(f"\n📋 Paper: '{paper['title'][:40]}...'")
        print(f"   Venue: {venue.name}")
        print(f"   Author prestige: {paper.get('author_prestige', 0.5):.2f}")
        print(f"   Paper novelty: {paper.get('novelty_level', 5)}/10")
        print(f"   Final Decision: {decision.value.upper()}")
        
        # Show how reviewer biases affected the outcome
        reviews = submission['reviews']
        avg_score = sum(r.criteria.overall_score() for r in reviews) / len(reviews)
        print(f"   Average review score: {avg_score:.2f}/10")
        print(f"   Venue threshold: {venue.get_acceptance_threshold():.1f}/10")
    
    print("\n" + "=" * 50)
    print("📈 System Statistics")
    print("=" * 50)
    
    # Show venue statistics
    for venue_id in ["neurips", "icml", "aaai"]:
        stats = system.get_venue_statistics(venue_id)
        if stats:
            venue = system.venues[venue_id]
            print(f"\n🏛️  {venue.name}:")
            print(f"   Total reviews: {stats['total_reviews']}")
            print(f"   Avg review length: {stats['avg_review_length']:.0f} words")
            print(f"   Late review rate: {stats['late_review_rate']*100:.1f}%")
            print(f"   Avg confidence: {stats['avg_confidence']:.1f}/5")
            
            print("   Decision distribution:")
            for decision, count in stats['decision_distribution'].items():
                if count > 0:
                    print(f"     {decision}: {count}")

if __name__ == "__main__":
    run_realistic_simulation()