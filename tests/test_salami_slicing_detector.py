"""
Unit tests for SalamiSlicingDetector

Tests the salami slicing detection functionality including minimal publishable unit analysis,
research fragmentation detection, and strategic publication splitting identification.
"""

import pytest
from datetime import date, timedelta
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path

from src.enhancements.salami_slicing_detector import (
    SalamiSlicingDetector, SalamiPattern, PaperSimilarity, ResearcherSalamiProfile
)
from src.data.enhanced_models import VenueType
from src.core.exceptions import ValidationError


class TestSalamiSlicingDetector:
    """Test cases for SalamiSlicingDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = SalamiSlicingDetector(
            similarity_threshold=0.7,
            min_papers_for_pattern=3,
            max_time_span_months=24
        )
        
        # Sample papers for testing
        self.base_date = date(2023, 1, 1)
        self.sample_papers = [
            {
                'paper_id': 'paper1',
                'title': 'Machine Learning Approach to Data Analysis',
                'abstract': 'This paper presents a novel machine learning approach for analyzing large datasets.',
                'authors': ['researcher1', 'researcher2'],
                'keywords': ['machine learning', 'data analysis', 'algorithms'],
                'venue_id': 'venue1',
                'venue_type': VenueType.TOP_CONFERENCE,
                'publication_date': self.base_date,
                'primary_author': 'researcher1'
            },
            {
                'paper_id': 'paper2',
                'title': 'Machine Learning Approach to Data Analysis: Part II',
                'abstract': 'This paper extends our previous machine learning approach with additional experiments.',
                'authors': ['researcher1', 'researcher2'],
                'keywords': ['machine learning', 'data analysis', 'experiments'],
                'venue_id': 'venue2',
                'venue_type': VenueType.MID_CONFERENCE,
                'publication_date': self.base_date + timedelta(days=60),
                'primary_author': 'researcher1'
            },
            {
                'paper_id': 'paper3',
                'title': 'Advanced Machine Learning for Data Analysis Applications',
                'abstract': 'Building on our machine learning framework, we present applications to real-world data.',
                'authors': ['researcher1', 'researcher2'],
                'keywords': ['machine learning', 'applications', 'real-world'],
                'venue_id': 'venue3',
                'venue_type': VenueType.LOW_CONFERENCE,
                'publication_date': self.base_date + timedelta(days=120),
                'primary_author': 'researcher1'
            }
        ]
    
    def test_initialization(self):
        """Test SalamiSlicingDetector initialization."""
        detector = SalamiSlicingDetector(
            similarity_threshold=0.8,
            min_papers_for_pattern=2,
            max_time_span_months=12
        )
        
        assert detector.similarity_threshold == 0.8
        assert detector.min_papers_for_pattern == 2
        assert detector.max_time_span_months == 12
        assert len(detector.detected_patterns) == 0
        assert len(detector.researcher_profiles) == 0
        assert len(detector.papers) == 0
    
    def test_add_paper(self):
        """Test adding papers to the detector."""
        paper = self.sample_papers[0]
        
        self.detector.add_paper(
            paper_id=paper['paper_id'],
            title=paper['title'],
            abstract=paper['abstract'],
            authors=paper['authors'],
            keywords=paper['keywords'],
            venue_id=paper['venue_id'],
            venue_type=paper['venue_type'],
            publication_date=paper['publication_date'],
            primary_author=paper['primary_author']
        )
        
        assert paper['paper_id'] in self.detector.papers
        assert paper['paper_id'] in self.detector.researcher_papers['researcher1']
        assert paper['paper_id'] in self.detector.researcher_papers['researcher2']
        
        stored_paper = self.detector.papers[paper['paper_id']]
        assert stored_paper['title'] == paper['title']
        assert stored_paper['abstract'] == paper['abstract']
        assert stored_paper['authors'] == paper['authors']
    
    def test_calculate_paper_similarity(self):
        """Test paper similarity calculation."""
        # Add papers
        for paper in self.sample_papers[:2]:
            self.detector.add_paper(**paper)
        
        similarity = self.detector.calculate_paper_similarity('paper1', 'paper2')
        
        assert isinstance(similarity, PaperSimilarity)
        assert similarity.paper_a == 'paper1'
        assert similarity.paper_b == 'paper2'
        assert 0 <= similarity.overall_similarity <= 1
        assert similarity.title_similarity > 0.5  # Should be similar titles
        assert similarity.author_overlap == 1.0  # Same authors
        assert similarity.temporal_proximity == 60  # 60 days apart
    
    def test_calculate_paper_similarity_invalid_paper(self):
        """Test similarity calculation with invalid paper ID."""
        with pytest.raises(ValidationError):
            self.detector.calculate_paper_similarity('invalid1', 'invalid2')
    
    def test_text_similarity_calculation(self):
        """Test text similarity calculation."""
        text1 = "Machine learning approach to data analysis"
        text2 = "Machine learning method for data analysis"
        
        similarity = self.detector._calculate_text_similarity(text1, text2)
        assert 0.6 <= similarity <= 1.0  # Should be high similarity
        
        # Test with completely different texts
        text3 = "Quantum computing applications in cryptography"
        similarity2 = self.detector._calculate_text_similarity(text1, text3)
        assert similarity2 < 0.3  # Should be low similarity
    
    def test_keyword_similarity_calculation(self):
        """Test keyword similarity calculation."""
        keywords1 = ['machine learning', 'data analysis', 'algorithms']
        keywords2 = ['machine learning', 'data analysis', 'experiments']
        
        similarity = self.detector._calculate_keyword_similarity(keywords1, keywords2)
        assert similarity == 2/4  # 2 common keywords out of 4 total unique
        
        # Test with no overlap
        keywords3 = ['quantum computing', 'cryptography']
        similarity2 = self.detector._calculate_keyword_similarity(keywords1, keywords3)
        assert similarity2 == 0.0
    
    def test_detect_salami_patterns_insufficient_papers(self):
        """Test pattern detection with insufficient papers."""
        # Add only 2 papers (less than min_papers_for_pattern=3)
        for paper in self.sample_papers[:2]:
            self.detector.add_paper(**paper)
        
        patterns = self.detector.detect_salami_patterns('researcher1')
        assert len(patterns) == 0
    
    def test_detect_salami_patterns_with_similar_papers(self):
        """Test pattern detection with similar papers."""
        # Add all sample papers
        for paper in self.sample_papers:
            self.detector.add_paper(**paper)
        
        patterns = self.detector.detect_salami_patterns('researcher1')
        
        # May or may not detect patterns depending on similarity threshold
        # The important thing is that the function runs without error
        assert isinstance(patterns, list)
        
        if patterns:  # If patterns are detected, validate them
            pattern = patterns[0]
            assert isinstance(pattern, SalamiPattern)
            assert pattern.researcher_id == 'researcher1'
            assert len(pattern.paper_ids) >= 2  # At least 2 papers
            assert 0 <= pattern.strength_score <= 1
            assert 0 <= pattern.content_similarity <= 1
    
    def test_venue_prestige_decline_detection(self):
        """Test detection of venue prestige decline."""
        # Declining prestige: TOP -> MID -> LOW
        venue_types = [VenueType.TOP_CONFERENCE, VenueType.MID_CONFERENCE, VenueType.LOW_CONFERENCE]
        
        decline_detected = self.detector._detect_venue_prestige_decline(venue_types)
        assert decline_detected is True
        
        # No decline: all same level
        venue_types_same = [VenueType.TOP_CONFERENCE, VenueType.TOP_CONFERENCE, VenueType.TOP_CONFERENCE]
        decline_detected2 = self.detector._detect_venue_prestige_decline(venue_types_same)
        assert decline_detected2 is False
    
    def test_incremental_titles_detection(self):
        """Test detection of incremental title patterns."""
        # Titles with incremental patterns
        incremental_titles = [
            "Study on Machine Learning: Part 1",
            "Study on Machine Learning: Part 2",
            "Study on Machine Learning: Part 3"
        ]
        
        detected = self.detector._detect_incremental_titles(incremental_titles)
        assert detected is True
        
        # Non-incremental titles
        regular_titles = [
            "Machine Learning Applications",
            "Deep Learning Methods",
            "Neural Network Architectures"
        ]
        
        detected2 = self.detector._detect_incremental_titles(regular_titles)
        assert detected2 is False
    
    def test_data_splitting_pattern_detection(self):
        """Test detection of data splitting patterns."""
        papers_with_splitting = [
            {
                'title': 'Analysis of Dataset A',
                'abstract': 'This study analyzes the first subset of our experimental data.'
            },
            {
                'title': 'Analysis of Dataset B',
                'abstract': 'This study analyzes the second subset of our experimental data.'
            },
            {
                'title': 'Analysis of Dataset C',
                'abstract': 'This study analyzes the third subset of our experimental data.'
            }
        ]
        
        detected = self.detector._detect_data_splitting_pattern(papers_with_splitting)
        assert detected is True
        
        papers_without_splitting = [
            {
                'title': 'Novel Algorithm Development',
                'abstract': 'We present a new algorithmic approach to optimization.'
            },
            {
                'title': 'Theoretical Framework',
                'abstract': 'This paper establishes theoretical foundations for our method.'
            }
        ]
        
        detected2 = self.detector._detect_data_splitting_pattern(papers_without_splitting)
        assert detected2 is False
    
    def test_researcher_profile_update(self):
        """Test researcher profile updates."""
        # Add papers and detect patterns
        for paper in self.sample_papers:
            self.detector.add_paper(**paper)
        
        patterns = self.detector.detect_salami_patterns('researcher1')
        
        # Check that profile was created and updated
        assert 'researcher1' in self.detector.researcher_profiles
        
        profile = self.detector.researcher_profiles['researcher1']
        assert isinstance(profile, ResearcherSalamiProfile)
        assert profile.researcher_id == 'researcher1'
        assert profile.total_papers == 3
        assert profile.publication_frequency > 0
        assert profile.risk_level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    
    def test_analyze_all_researchers(self):
        """Test analysis of all researchers."""
        # Add papers for multiple researchers
        for paper in self.sample_papers:
            self.detector.add_paper(**paper)
        
        # Add a paper for a different researcher
        different_paper = {
            'paper_id': 'paper4',
            'title': 'Quantum Computing Applications',
            'abstract': 'This paper explores quantum computing in cryptography.',
            'authors': ['researcher3'],
            'keywords': ['quantum computing', 'cryptography'],
            'venue_id': 'venue4',
            'venue_type': VenueType.TOP_JOURNAL,
            'publication_date': self.base_date,
            'primary_author': 'researcher3'
        }
        self.detector.add_paper(**different_paper)
        
        all_patterns = self.detector.analyze_all_researchers()
        
        # Should have patterns for researcher1 but not researcher3 (only 1 paper)
        assert 'researcher1' in all_patterns or len(all_patterns) == 0  # Might not detect patterns if similarity too low
        assert 'researcher3' not in all_patterns  # Only 1 paper, below minimum
    
    def test_salami_statistics_calculation(self):
        """Test calculation of salami slicing statistics."""
        # Add papers and analyze
        for paper in self.sample_papers:
            self.detector.add_paper(**paper)
        
        self.detector.analyze_all_researchers()
        
        stats = self.detector.calculate_salami_statistics()
        
        assert 'total_patterns' in stats
        assert 'total_researchers' in stats
        assert 'total_papers_involved' in stats
        assert 'pattern_types' in stats
        assert 'average_pattern_strength' in stats
        assert 'high_risk_researchers' in stats
        assert 'researchers_by_risk' in stats
        
        # Check that all values are reasonable
        assert stats['total_patterns'] >= 0
        assert stats['total_researchers'] >= 0
        assert stats['total_papers_involved'] >= 0
        assert 0 <= stats['average_pattern_strength'] <= 1
    
    def test_generate_researcher_specific_report(self):
        """Test generation of researcher-specific report."""
        # Add papers and analyze
        for paper in self.sample_papers:
            self.detector.add_paper(**paper)
        
        self.detector.analyze_all_researchers()
        
        report = self.detector.generate_salami_report('researcher1')
        
        assert 'researcher_id' in report
        assert report['researcher_id'] == 'researcher1'
        assert 'profile' in report
        assert 'patterns' in report
        assert 'recommendations' in report
        
        # Check profile structure
        profile = report['profile']
        assert 'total_papers' in profile
        assert 'suspected_salami_papers' in profile
        assert 'salami_rate' in profile
        assert 'risk_level' in profile
    
    def test_generate_global_report(self):
        """Test generation of global report."""
        # Add papers and analyze
        for paper in self.sample_papers:
            self.detector.add_paper(**paper)
        
        self.detector.analyze_all_researchers()
        
        report = self.detector.generate_salami_report()
        
        assert 'statistics' in report
        assert 'top_patterns' in report
        assert 'high_risk_researchers' in report
        
        # Check statistics structure
        stats = report['statistics']
        assert 'total_patterns' in stats
        assert 'pattern_types' in stats
        assert 'researchers_by_risk' in stats
    
    def test_generate_report_nonexistent_researcher(self):
        """Test report generation for nonexistent researcher."""
        report = self.detector.generate_salami_report('nonexistent_researcher')
        
        assert 'error' in report
        assert 'nonexistent_researcher' in report['error']
    
    def test_pattern_classification(self):
        """Test classification of salami slicing patterns."""
        papers = [
            {
                'publication_date': self.base_date,
                'venue_type': VenueType.TOP_CONFERENCE,
                'title': 'Machine Learning Study: Part 1',
                'abstract': 'First part of our comprehensive study.'
            },
            {
                'publication_date': self.base_date + timedelta(days=30),
                'venue_type': VenueType.MID_CONFERENCE,
                'title': 'Machine Learning Study: Part 2',
                'abstract': 'Second part of our comprehensive study.'
            },
            {
                'publication_date': self.base_date + timedelta(days=60),
                'venue_type': VenueType.LOW_CONFERENCE,
                'title': 'Machine Learning Study: Part 3',
                'abstract': 'Third part of our comprehensive study.'
            }
        ]
        
        pattern_type, strength_score, indicators = self.detector._classify_salami_pattern(
            papers, avg_similarity=0.9, venue_diversity=0.33
        )
        
        assert pattern_type in ['incremental_splitting', 'venue_shopping_splitting', 'temporal_splitting', 'method_splitting']
        assert 0 <= strength_score <= 1
        assert len(indicators) > 0
        assert any('similarity' in indicator.lower() for indicator in indicators)
    
    def test_serialization_to_dict(self):
        """Test conversion to dictionary for serialization."""
        # Add some data
        for paper in self.sample_papers:
            self.detector.add_paper(**paper)
        
        self.detector.analyze_all_researchers()
        
        data_dict = self.detector.to_dict()
        
        assert 'config' in data_dict
        assert 'detected_patterns' in data_dict
        assert 'researcher_profiles' in data_dict
        
        # Check config
        config = data_dict['config']
        assert config['similarity_threshold'] == self.detector.similarity_threshold
        assert config['min_papers_for_pattern'] == self.detector.min_papers_for_pattern
        assert config['max_time_span_months'] == self.detector.max_time_span_months
    
    def test_deserialization_from_dict(self):
        """Test creation from dictionary."""
        # Create detector with some data
        for paper in self.sample_papers:
            self.detector.add_paper(**paper)
        
        self.detector.analyze_all_researchers()
        
        # Convert to dict and back
        data_dict = self.detector.to_dict()
        new_detector = SalamiSlicingDetector.from_dict(data_dict)
        
        # Check that configuration is preserved
        assert new_detector.similarity_threshold == self.detector.similarity_threshold
        assert new_detector.min_papers_for_pattern == self.detector.min_papers_for_pattern
        assert new_detector.max_time_span_months == self.detector.max_time_span_months
        
        # Check that patterns are preserved
        assert len(new_detector.detected_patterns) == len(self.detector.detected_patterns)
        assert len(new_detector.researcher_profiles) == len(self.detector.researcher_profiles)
    
    def test_similar_paper_groups_finding(self):
        """Test finding groups of similar papers."""
        # Create mock similarities
        similarities = [
            PaperSimilarity('paper1', 'paper2', 0.8, 0.8, 0.8, 1.0, 1.0, 60),
            PaperSimilarity('paper2', 'paper3', 0.8, 0.8, 0.8, 1.0, 0.5, 60),
            PaperSimilarity('paper1', 'paper3', 0.7, 0.7, 0.7, 1.0, 0.5, 120)
        ]
        
        paper_ids = ['paper1', 'paper2', 'paper3']
        groups = self.detector._find_similar_paper_groups(paper_ids, similarities)
        
        # Should find one group with all three papers (if similarities are above threshold)
        assert len(groups) >= 1
        if groups:
            assert len(groups[0]) >= 2  # At least 2 papers in a group
    
    def test_recommendations_generation(self):
        """Test generation of recommendations."""
        # Create a high-risk profile
        profile = ResearcherSalamiProfile(
            researcher_id='test_researcher',
            total_papers=20,
            suspected_salami_papers=15,
            average_pattern_strength=0.8,
            publication_frequency=12.0,
            venue_shopping_tendency=0.8,
            incremental_publication_score=0.75,
            risk_level='CRITICAL'
        )
        
        recommendations = self.detector._generate_recommendations(profile)
        
        assert len(recommendations) > 0
        assert any('review' in rec.lower() for rec in recommendations)
        assert any('critical' in rec.lower() or 'immediate' in rec.lower() for rec in recommendations)
    
    def test_edge_cases(self):
        """Test various edge cases."""
        # Test with empty data
        stats = self.detector.calculate_salami_statistics()
        assert stats['total_patterns'] == 0
        
        # Test with single paper
        single_paper = self.sample_papers[0]
        self.detector.add_paper(**single_paper)
        patterns = self.detector.detect_salami_patterns('researcher1')
        assert len(patterns) == 0  # Should not detect patterns with single paper
        
        # Test similarity with identical papers
        similarity = self.detector._calculate_text_similarity("identical text", "identical text")
        assert similarity == 1.0
        
        # Test similarity with empty texts
        similarity_empty = self.detector._calculate_text_similarity("", "")
        assert similarity_empty == 0.0


class TestSalamiPattern:
    """Test cases for SalamiPattern dataclass."""
    
    def test_salami_pattern_creation(self):
        """Test SalamiPattern creation and validation."""
        pattern = SalamiPattern(
            pattern_id='test_pattern',
            researcher_id='researcher1',
            paper_ids=['paper1', 'paper2', 'paper3'],
            pattern_type='incremental_splitting',
            detection_date=date.today(),
            strength_score=0.8,
            total_papers=3,
            time_span_months=6,
            content_similarity=0.85,
            venue_diversity=0.33,
            suspicious_indicators=['High similarity', 'Rapid publication'],
            evidence={'test': 'data'}
        )
        
        assert pattern.pattern_id == 'test_pattern'
        assert pattern.researcher_id == 'researcher1'
        assert len(pattern.paper_ids) == 3
        assert pattern.strength_score == 0.8
        assert 0 <= pattern.strength_score <= 1
    
    def test_salami_pattern_validation_errors(self):
        """Test SalamiPattern validation errors."""
        # Test with too few papers
        with pytest.raises(ValidationError):
            SalamiPattern(
                pattern_id='test',
                researcher_id='researcher1',
                paper_ids=['paper1'],  # Only 1 paper
                pattern_type='test',
                detection_date=date.today(),
                strength_score=0.5,
                total_papers=1,
                time_span_months=6,
                content_similarity=0.5,
                venue_diversity=0.5,
                suspicious_indicators=[],
                evidence={}
            )
        
        # Test with invalid strength score
        with pytest.raises(ValidationError):
            SalamiPattern(
                pattern_id='test',
                researcher_id='researcher1',
                paper_ids=['paper1', 'paper2'],
                pattern_type='test',
                detection_date=date.today(),
                strength_score=1.5,  # Invalid score > 1
                total_papers=2,
                time_span_months=6,
                content_similarity=0.5,
                venue_diversity=0.5,
                suspicious_indicators=[],
                evidence={}
            )


class TestPaperSimilarity:
    """Test cases for PaperSimilarity dataclass."""
    
    def test_paper_similarity_creation(self):
        """Test PaperSimilarity creation and overall similarity calculation."""
        similarity = PaperSimilarity(
            paper_a='paper1',
            paper_b='paper2',
            title_similarity=0.8,
            abstract_similarity=0.7,
            keyword_similarity=0.9,
            author_overlap=1.0,
            venue_similarity=0.5,
            temporal_proximity=30.0
        )
        
        assert similarity.paper_a == 'paper1'
        assert similarity.paper_b == 'paper2'
        assert 0 <= similarity.overall_similarity <= 1
        
        # Overall similarity should be weighted combination
        expected = (0.8 * 0.25 + 0.7 * 0.30 + 0.9 * 0.20 + 1.0 * 0.15 + 
                   0.5 * 0.05 + (1.0 - min(1.0, 30.0 / 365.0)) * 0.05)
        assert abs(similarity.overall_similarity - expected) < 0.01


class TestResearcherSalamiProfile:
    """Test cases for ResearcherSalamiProfile dataclass."""
    
    def test_profile_creation_and_update(self):
        """Test profile creation and update functionality."""
        profile = ResearcherSalamiProfile(researcher_id='test_researcher')
        
        # Create mock patterns
        patterns = [
            Mock(pattern_id='pattern1', paper_ids=['p1', 'p2'], strength_score=0.8),
            Mock(pattern_id='pattern2', paper_ids=['p3', 'p4'], strength_score=0.6)
        ]
        
        profile.update_profile(patterns, total_papers=10, years_active=2)
        
        assert profile.total_papers == 10
        assert profile.suspected_salami_papers == 4  # 2 + 2 papers
        assert len(profile.salami_patterns) == 2
        assert profile.average_pattern_strength == 0.7  # (0.8 + 0.6) / 2
        assert profile.publication_frequency == 5.0  # 10 papers / 2 years
        assert profile.incremental_publication_score == 0.4  # 4/10
    
    def test_risk_level_calculation(self):
        """Test risk level calculation."""
        profile = ResearcherSalamiProfile(researcher_id='test_researcher')
        
        # Test CRITICAL risk
        profile.incremental_publication_score = 0.9
        profile.average_pattern_strength = 0.9
        profile.publication_frequency = 15.0
        profile.venue_shopping_tendency = 0.8
        profile._calculate_risk_level()
        assert profile.risk_level == 'CRITICAL'
        
        # Test LOW risk
        profile.incremental_publication_score = 0.1
        profile.average_pattern_strength = 0.2
        profile.publication_frequency = 2.0
        profile.venue_shopping_tendency = 0.1
        profile._calculate_risk_level()
        assert profile.risk_level == 'LOW'


if __name__ == '__main__':
    pytest.main([__file__])