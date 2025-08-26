"""
Job Market Simulation System

This module implements job market simulation for postdoc and faculty competition,
modeling position scarcity and competition dynamics, and creating job market
outcome prediction based on researcher profiles.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
import logging
import random
import math
from collections import defaultdict

from src.data.enhanced_models import EnhancedResearcher, ResearcherLevel, CareerStage
from src.core.exceptions import ValidationError, CareerSystemError
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class PositionType(Enum):
    """Types of academic positions."""
    POSTDOC = "Postdoc"
    ASSISTANT_PROF = "Assistant Professor"
    ASSOCIATE_PROF = "Associate Professor"
    FULL_PROF = "Full Professor"
    INDUSTRY_RESEARCH = "Industry Research"
    INDUSTRY_APPLIED = "Industry Applied"
    GOVERNMENT_LAB = "Government Lab"
    TEACHING_FOCUSED = "Teaching Focused"


class InstitutionTier(Enum):
    """Institution tier categories."""
    R1_TOP = "R1 Top"  # Top 20 research universities
    R1_MID = "R1 Mid"  # Other R1 universities
    R2 = "R2"  # R2 universities
    TEACHING = "Teaching"  # Teaching-focused institutions
    INDUSTRY = "Industry"  # Industry positions
    GOVERNMENT = "Government"  # Government positions


class ApplicationOutcome(Enum):
    """Job application outcomes."""
    OFFER = "Offer"
    INTERVIEW = "Interview"
    REJECTION = "Rejection"
    WITHDRAWN = "Withdrawn"
    PENDING = "Pending"


@dataclass
class JobPosition:
    """Represents an academic job position."""
    position_id: str
    position_type: PositionType
    institution_name: str
    institution_tier: InstitutionTier
    field: str
    location: str
    salary_range: Tuple[int, int]  # Min, max salary
    required_qualifications: List[str]
    preferred_qualifications: List[str]
    application_deadline: date
    start_date: date
    number_of_positions: int = 1
    expected_applicants: int = 100
    
    def __post_init__(self):
        """Validate job position data."""
        if self.salary_range[0] > self.salary_range[1]:
            raise ValidationError("salary_range", self.salary_range, "min salary <= max salary")
        if self.number_of_positions < 1:
            raise ValidationError("number_of_positions", self.number_of_positions, "positive integer")
        if self.expected_applicants < 1:
            raise ValidationError("expected_applicants", self.expected_applicants, "positive integer")


@dataclass
class JobApplication:
    """Represents a job application."""
    application_id: str
    applicant_id: str
    position_id: str
    application_date: date
    materials_submitted: List[str]
    outcome: ApplicationOutcome = ApplicationOutcome.PENDING
    interview_date: Optional[date] = None
    offer_date: Optional[date] = None
    decision_deadline: Optional[date] = None
    notes: str = ""


@dataclass
class JobMarketCandidate:
    """Enhanced candidate profile for job market analysis."""
    researcher: EnhancedResearcher
    target_positions: List[PositionType]
    geographic_preferences: List[str]
    salary_expectations: Tuple[int, int]
    mobility_constraints: List[str]
    application_strategy: str  # "aggressive", "selective", "conservative"
    market_competitiveness: float  # 0-1 score
    
    def __post_init__(self):
        """Validate candidate data."""
        if not (0.0 <= self.market_competitiveness <= 1.0):
            raise ValidationError("market_competitiveness", self.market_competitiveness, "value between 0 and 1")


@dataclass
class JobMarketResults:
    """Results of job market simulation."""
    year: int
    total_positions: int
    total_candidates: int
    competition_ratio: float  # candidates per position
    placement_rate: float  # percentage of candidates placed
    position_outcomes: Dict[PositionType, Dict[str, Any]]
    candidate_outcomes: Dict[str, Dict[str, Any]]
    market_trends: Dict[str, float]
    field_analysis: Dict[str, Dict[str, Any]]


@dataclass
class MarketTrends:
    """Market trend analysis data."""
    year: int
    field: str
    position_growth_rate: float  # Year-over-year change
    candidate_growth_rate: float
    average_time_to_placement: float  # months
    salary_trends: Dict[PositionType, float]  # percentage change
    geographic_shifts: Dict[str, float]  # regional demand changes


class JobMarketSimulator:
    """
    Simulates academic job market dynamics including postdoc and faculty competition,
    position scarcity modeling, and outcome prediction based on researcher profiles.
    
    This class provides functionality to:
    - Model position scarcity and competition dynamics
    - Predict job market outcomes based on researcher profiles
    - Simulate application and hiring processes
    - Track market trends and placement rates
    - Analyze field-specific job market conditions
    """
    
    # Market parameters by field and position type
    MARKET_PARAMETERS = {
        'computer_science': {
            'postdoc_positions_per_year': 500,
            'assistant_prof_positions_per_year': 200,
            'candidates_per_position': {
                PositionType.POSTDOC: 3.5,
                PositionType.ASSISTANT_PROF: 8.0,
                PositionType.ASSOCIATE_PROF: 4.0,
                PositionType.FULL_PROF: 2.5
            },
            'placement_rates': {
                PositionType.POSTDOC: 0.75,
                PositionType.ASSISTANT_PROF: 0.35,
                PositionType.ASSOCIATE_PROF: 0.60,
                PositionType.FULL_PROF: 0.80
            }
        },
        'biology': {
            'postdoc_positions_per_year': 800,
            'assistant_prof_positions_per_year': 150,
            'candidates_per_position': {
                PositionType.POSTDOC: 4.0,
                PositionType.ASSISTANT_PROF: 12.0,
                PositionType.ASSOCIATE_PROF: 5.0,
                PositionType.FULL_PROF: 3.0
            },
            'placement_rates': {
                PositionType.POSTDOC: 0.70,
                PositionType.ASSISTANT_PROF: 0.25,
                PositionType.ASSOCIATE_PROF: 0.55,
                PositionType.FULL_PROF: 0.75
            }
        },
        'physics': {
            'postdoc_positions_per_year': 400,
            'assistant_prof_positions_per_year': 120,
            'candidates_per_position': {
                PositionType.POSTDOC: 3.8,
                PositionType.ASSISTANT_PROF: 10.0,
                PositionType.ASSOCIATE_PROF: 4.5,
                PositionType.FULL_PROF: 2.8
            },
            'placement_rates': {
                PositionType.POSTDOC: 0.72,
                PositionType.ASSISTANT_PROF: 0.30,
                PositionType.ASSOCIATE_PROF: 0.58,
                PositionType.FULL_PROF: 0.78
            }
        }
    }
    
    # Institution tier preferences and competitiveness
    INSTITUTION_COMPETITIVENESS = {
        InstitutionTier.R1_TOP: {
            'selectivity': 0.95,  # Very selective
            'candidate_quality_threshold': 0.85,
            'salary_multiplier': 1.3
        },
        InstitutionTier.R1_MID: {
            'selectivity': 0.80,
            'candidate_quality_threshold': 0.70,
            'salary_multiplier': 1.1
        },
        InstitutionTier.R2: {
            'selectivity': 0.60,
            'candidate_quality_threshold': 0.55,
            'salary_multiplier': 0.9
        },
        InstitutionTier.TEACHING: {
            'selectivity': 0.40,
            'candidate_quality_threshold': 0.40,
            'salary_multiplier': 0.8
        },
        InstitutionTier.INDUSTRY: {
            'selectivity': 0.70,
            'candidate_quality_threshold': 0.60,
            'salary_multiplier': 1.4
        }
    }
    
    def __init__(self):
        """Initialize the job market simulator."""
        logger.info("Initializing Job Market Simulation System")
        self.available_positions: Dict[str, JobPosition] = {}
        self.job_candidates: Dict[str, JobMarketCandidate] = {}
        self.applications: Dict[str, JobApplication] = {}
        self.market_history: List[JobMarketResults] = []
        self.current_year = datetime.now().year
    
    def create_job_position(self, position_type: PositionType, institution_tier: InstitutionTier,
                          field: str, location: str = "Various",
                          salary_range: Optional[Tuple[int, int]] = None) -> JobPosition:
        """
        Create a new job position in the market.
        
        Args:
            position_type: Type of position
            institution_tier: Tier of institution
            field: Academic field
            location: Geographic location
            salary_range: Optional salary range override
            
        Returns:
            JobPosition object
        """
        position_id = f"pos_{len(self.available_positions) + 1}_{position_type.value.lower().replace(' ', '_')}"
        
        # Calculate default salary range based on position and tier
        if salary_range is None:
            salary_range = self._calculate_salary_range(position_type, institution_tier)
        
        # Generate realistic institution name
        institution_name = self._generate_institution_name(institution_tier, location)
        
        # Set application deadline (typically 3-6 months from now)
        deadline_days = random.randint(90, 180)
        application_deadline = date.today().replace(day=1) + timedelta(days=deadline_days)
        
        # Set start date (typically next academic year)
        start_date = date(self.current_year + 1, 8, 15)  # August start
        
        # Generate qualifications
        required_quals, preferred_quals = self._generate_qualifications(position_type, field)
        
        # Estimate expected applicants
        field_params = self.MARKET_PARAMETERS.get(field, self.MARKET_PARAMETERS['computer_science'])
        base_applicants = int(field_params['candidates_per_position'].get(position_type, 5.0) * 20)
        expected_applicants = random.randint(int(base_applicants * 0.7), int(base_applicants * 1.3))
        
        position = JobPosition(
            position_id=position_id,
            position_type=position_type,
            institution_name=institution_name,
            institution_tier=institution_tier,
            field=field,
            location=location,
            salary_range=salary_range,
            required_qualifications=required_quals,
            preferred_qualifications=preferred_quals,
            application_deadline=application_deadline,
            start_date=start_date,
            expected_applicants=expected_applicants
        )
        
        self.available_positions[position_id] = position
        
        logger.info(f"Created job position: {position_type.value} at {institution_name} "
                   f"({institution_tier.value}) in {field}")
        
        return position
    
    def register_job_candidate(self, researcher: EnhancedResearcher,
                             target_positions: List[PositionType],
                             geographic_preferences: Optional[List[str]] = None,
                             salary_expectations: Optional[Tuple[int, int]] = None,
                             application_strategy: str = "selective") -> JobMarketCandidate:
        """
        Register a researcher as a job market candidate.
        
        Args:
            researcher: The researcher seeking positions
            target_positions: Types of positions they're seeking
            geographic_preferences: Preferred locations
            salary_expectations: Expected salary range
            application_strategy: Application approach
            
        Returns:
            JobMarketCandidate object
        """
        if geographic_preferences is None:
            geographic_preferences = ["Any"]
        
        if salary_expectations is None:
            # Calculate reasonable salary expectations based on level and field
            salary_expectations = self._calculate_salary_expectations(researcher)
        
        # Calculate market competitiveness score
        competitiveness = self._calculate_market_competitiveness(researcher)
        
        candidate = JobMarketCandidate(
            researcher=researcher,
            target_positions=target_positions,
            geographic_preferences=geographic_preferences,
            salary_expectations=salary_expectations,
            mobility_constraints=[],
            application_strategy=application_strategy,
            market_competitiveness=competitiveness
        )
        
        self.job_candidates[researcher.id] = candidate
        
        logger.info(f"Registered job candidate: {researcher.name} "
                   f"(competitiveness: {competitiveness:.2f})")
        
        return candidate
    
    def simulate_job_market_cycle(self, year: Optional[int] = None,
                                field: str = "computer_science") -> JobMarketResults:
        """
        Simulate a complete job market cycle for a given year and field.
        
        Args:
            year: Year to simulate (defaults to current year)
            field: Academic field to simulate
            
        Returns:
            JobMarketResults with comprehensive market analysis
        """
        if year is None:
            year = self.current_year
        
        logger.info(f"Simulating job market cycle for {field} in {year}")
        
        # Generate positions for the year
        positions = self._generate_annual_positions(field, year)
        
        # Filter candidates for this field
        field_candidates = [
            candidate for candidate in self.job_candidates.values()
            if candidate.researcher.specialty.lower() == field.lower()
        ]
        
        # Simulate application process
        applications = self._simulate_applications(positions, field_candidates)
        
        # Simulate hiring decisions
        hiring_results = self._simulate_hiring_process(applications, positions, field_candidates)
        
        # Calculate market metrics
        results = self._calculate_market_results(year, positions, field_candidates, hiring_results, field)
        
        # Store results
        self.market_history.append(results)
        
        logger.info(f"Job market simulation complete: {results.placement_rate:.1%} placement rate, "
                   f"{results.competition_ratio:.1f} candidates per position")
        
        return results
    
    def predict_job_market_outcome(self, candidate: JobMarketCandidate,
                                 target_year: Optional[int] = None) -> Dict[str, Any]:
        """
        Predict job market outcomes for a specific candidate.
        
        Args:
            candidate: The job market candidate
            target_year: Year to predict for (defaults to next year)
            
        Returns:
            Dictionary with outcome predictions
        """
        if target_year is None:
            target_year = self.current_year + 1
        
        researcher = candidate.researcher
        field = researcher.specialty.lower()
        
        # Get field parameters
        field_params = self.MARKET_PARAMETERS.get(field, self.MARKET_PARAMETERS['computer_science'])
        
        predictions = {}
        
        for position_type in candidate.target_positions:
            # Calculate base success probability
            base_placement_rate = field_params['placement_rates'].get(position_type, 0.5)
            competition_ratio = field_params['candidates_per_position'].get(position_type, 5.0)
            
            # Adjust for candidate competitiveness
            competitiveness_factor = candidate.market_competitiveness
            adjusted_probability = base_placement_rate * (0.5 + competitiveness_factor)
            
            # Adjust for application strategy
            strategy_multiplier = {
                'aggressive': 1.2,  # More applications, higher chance
                'selective': 1.0,   # Baseline
                'conservative': 0.8  # Fewer applications, lower chance
            }.get(candidate.application_strategy, 1.0)
            
            final_probability = min(0.95, adjusted_probability * strategy_multiplier)
            
            # Predict timeline
            expected_months_to_offer = self._predict_timeline(position_type, competitiveness_factor)
            
            # Predict salary range
            salary_prediction = self._predict_salary_outcome(
                position_type, researcher, candidate.salary_expectations
            )
            
            predictions[position_type.value] = {
                'success_probability': final_probability,
                'expected_timeline_months': expected_months_to_offer,
                'predicted_salary_range': salary_prediction,
                'competition_level': competition_ratio,
                'recommended_applications': int(competition_ratio * 2),
                'key_strengths': self._identify_candidate_strengths(researcher, position_type),
                'improvement_areas': self._identify_improvement_areas(researcher, position_type)
            }
        
        # Overall market assessment
        overall_assessment = {
            'market_competitiveness': candidate.market_competitiveness,
            'best_position_match': max(predictions.keys(), 
                                     key=lambda x: predictions[x]['success_probability']),
            'market_readiness': self._assess_market_readiness(candidate),
            'strategic_recommendations': self._generate_strategic_recommendations(candidate, predictions)
        }
        
        return {
            'candidate_id': researcher.id,
            'target_year': target_year,
            'field': field,
            'position_predictions': predictions,
            'overall_assessment': overall_assessment,
            'market_conditions': self._assess_market_conditions(field, target_year)
        }
    
    def _calculate_salary_range(self, position_type: PositionType, 
                              institution_tier: InstitutionTier) -> Tuple[int, int]:
        """Calculate realistic salary range for position and institution."""
        # Base salaries by position type (in thousands)
        base_salaries = {
            PositionType.POSTDOC: (45, 55),
            PositionType.ASSISTANT_PROF: (65, 85),
            PositionType.ASSOCIATE_PROF: (75, 95),
            PositionType.FULL_PROF: (90, 130),
            PositionType.INDUSTRY_RESEARCH: (100, 150),
            PositionType.INDUSTRY_APPLIED: (90, 140),
            PositionType.GOVERNMENT_LAB: (70, 100),
            PositionType.TEACHING_FOCUSED: (55, 75)
        }
        
        base_min, base_max = base_salaries.get(position_type, (50, 70))
        
        # Apply institution tier multiplier
        tier_multiplier = self.INSTITUTION_COMPETITIVENESS[institution_tier]['salary_multiplier']
        
        min_salary = int(base_min * tier_multiplier * 1000)
        max_salary = int(base_max * tier_multiplier * 1000)
        
        return (min_salary, max_salary)
    
    def _generate_institution_name(self, tier: InstitutionTier, location: str) -> str:
        """Generate realistic institution name."""
        tier_prefixes = {
            InstitutionTier.R1_TOP: ["Stanford", "MIT", "Harvard", "Berkeley", "Princeton"],
            InstitutionTier.R1_MID: ["State University", "Tech University", "Research University"],
            InstitutionTier.R2: ["Regional University", "State College", "Metropolitan University"],
            InstitutionTier.TEACHING: ["Liberal Arts College", "Community College", "Teaching University"],
            InstitutionTier.INDUSTRY: ["Tech Corp", "Research Labs", "Innovation Center"],
            InstitutionTier.GOVERNMENT: ["National Lab", "Federal Research Center", "Government Institute"]
        }
        
        prefixes = tier_prefixes.get(tier, ["University"])
        prefix = random.choice(prefixes)
        
        if location != "Various" and location != "Any":
            return f"{location} {prefix}"
        else:
            return f"{prefix} of Excellence"
    
    def _generate_qualifications(self, position_type: PositionType, 
                               field: str) -> Tuple[List[str], List[str]]:
        """Generate required and preferred qualifications."""
        required = []
        preferred = []
        
        if position_type == PositionType.POSTDOC:
            required = [
                "PhD in relevant field",
                "Strong publication record",
                "Research experience"
            ]
            preferred = [
                "Postdoc experience",
                "Grant writing experience",
                "Collaboration experience"
            ]
        elif position_type == PositionType.ASSISTANT_PROF:
            required = [
                "PhD in relevant field",
                "Strong research record",
                "Teaching experience",
                "Publication record"
            ]
            preferred = [
                "Postdoc experience",
                "Grant funding",
                "Industry experience",
                "International experience"
            ]
        elif position_type in [PositionType.ASSOCIATE_PROF, PositionType.FULL_PROF]:
            required = [
                "PhD in relevant field",
                "Established research program",
                "Strong publication record",
                "Teaching excellence",
                "Service record"
            ]
            preferred = [
                "Grant funding history",
                "Leadership experience",
                "International recognition",
                "Mentoring experience"
            ]
        
        return required, preferred
    
    def _calculate_salary_expectations(self, researcher: EnhancedResearcher) -> Tuple[int, int]:
        """Calculate reasonable salary expectations for researcher."""
        level_expectations = {
            ResearcherLevel.GRADUATE_STUDENT: (45000, 55000),
            ResearcherLevel.POSTDOC: (50000, 65000),
            ResearcherLevel.ASSISTANT_PROF: (70000, 90000),
            ResearcherLevel.ASSOCIATE_PROF: (80000, 110000),
            ResearcherLevel.FULL_PROF: (100000, 150000),
            ResearcherLevel.EMERITUS: (60000, 100000)  # Part-time/consulting
        }
        
        base_min, base_max = level_expectations.get(researcher.level, (50000, 70000))
        
        # Adjust for reputation and institution tier
        reputation_multiplier = 0.8 + (researcher.reputation_score * 0.4)
        tier_multiplier = 1.0 + (researcher.institution_tier - 2) * 0.1
        
        adjusted_min = int(base_min * reputation_multiplier * tier_multiplier)
        adjusted_max = int(base_max * reputation_multiplier * tier_multiplier)
        
        return (adjusted_min, adjusted_max)
    
    def _calculate_market_competitiveness(self, researcher: EnhancedResearcher) -> float:
        """Calculate market competitiveness score for researcher."""
        # Base score from academic metrics
        h_index_score = min(1.0, researcher.h_index / 20.0)  # Normalize to 0-1
        citation_score = min(1.0, researcher.total_citations / 500.0)
        publication_score = min(1.0, len(researcher.publication_history) / 20.0)
        
        # Experience factor
        experience_score = min(1.0, researcher.years_active / 10.0)
        
        # Institution tier factor
        tier_score = (4 - researcher.institution_tier) / 3.0  # Higher tier = higher score
        
        # Reputation factor
        reputation_score = researcher.reputation_score
        
        # Weighted combination
        competitiveness = (
            h_index_score * 0.25 +
            citation_score * 0.20 +
            publication_score * 0.20 +
            experience_score * 0.15 +
            tier_score * 0.10 +
            reputation_score * 0.10
        )
        
        return min(1.0, max(0.0, competitiveness))
    
    def _generate_annual_positions(self, field: str, year: int) -> List[JobPosition]:
        """Generate realistic number of positions for a field and year."""
        field_params = self.MARKET_PARAMETERS.get(field, self.MARKET_PARAMETERS['computer_science'])
        
        positions = []
        
        # Generate postdoc positions
        postdoc_count = int(field_params['postdoc_positions_per_year'] * random.uniform(0.8, 1.2))
        for _ in range(postdoc_count):
            tier = random.choices(
                list(InstitutionTier),
                weights=[0.15, 0.35, 0.30, 0.10, 0.05, 0.05]  # Favor R1 institutions
            )[0]
            if tier not in [InstitutionTier.INDUSTRY, InstitutionTier.GOVERNMENT]:
                position = self.create_job_position(PositionType.POSTDOC, tier, field)
                positions.append(position)
        
        # Generate faculty positions
        faculty_count = int(field_params['assistant_prof_positions_per_year'] * random.uniform(0.8, 1.2))
        for _ in range(faculty_count):
            tier = random.choices(
                list(InstitutionTier),
                weights=[0.10, 0.25, 0.35, 0.25, 0.03, 0.02]  # More distributed
            )[0]
            if tier not in [InstitutionTier.INDUSTRY, InstitutionTier.GOVERNMENT]:
                position = self.create_job_position(PositionType.ASSISTANT_PROF, tier, field)
                positions.append(position)
        
        return positions
    
    def _simulate_applications(self, positions: List[JobPosition], 
                             candidates: List[JobMarketCandidate]) -> List[JobApplication]:
        """Simulate the application process."""
        applications = []
        
        for candidate in candidates:
            # Determine number of applications based on strategy
            strategy_multipliers = {
                'aggressive': random.randint(15, 30),
                'selective': random.randint(8, 15),
                'conservative': random.randint(3, 8)
            }
            
            num_applications = strategy_multipliers.get(candidate.application_strategy, 10)
            
            # Filter positions by candidate preferences
            suitable_positions = [
                pos for pos in positions
                if pos.position_type in candidate.target_positions
                and (candidate.geographic_preferences == ["Any"] or 
                     any(pref in pos.location for pref in candidate.geographic_preferences))
            ]
            
            # Apply to positions (randomly select from suitable ones)
            selected_positions = random.sample(
                suitable_positions, 
                min(num_applications, len(suitable_positions))
            )
            
            for position in selected_positions:
                app_id = f"app_{len(applications) + 1}"
                application = JobApplication(
                    application_id=app_id,
                    applicant_id=candidate.researcher.id,
                    position_id=position.position_id,
                    application_date=date.today(),
                    materials_submitted=["CV", "Cover Letter", "Research Statement", "References"]
                )
                applications.append(application)
        
        return applications
    
    def _simulate_hiring_process(self, applications: List[JobApplication],
                               positions: List[JobPosition],
                               candidates: List[JobMarketCandidate]) -> Dict[str, Any]:
        """Simulate the hiring decision process."""
        results = {
            'offers_made': [],
            'interviews_conducted': [],
            'rejections_sent': [],
            'positions_filled': []
        }
        
        # Group applications by position
        position_applications = defaultdict(list)
        for app in applications:
            position_applications[app.position_id].append(app)
        
        candidate_lookup = {c.researcher.id: c for c in candidates}
        
        for position in positions:
            position_apps = position_applications[position.position_id]
            if not position_apps:
                continue
            
            # Score candidates for this position
            candidate_scores = []
            for app in position_apps:
                candidate = candidate_lookup.get(app.applicant_id)
                if candidate:
                    score = self._score_candidate_for_position(candidate, position)
                    candidate_scores.append((app, candidate, score))
            
            # Sort by score (highest first)
            candidate_scores.sort(key=lambda x: x[2], reverse=True)
            
            # Determine interview cutoff (top 10-20% or max 10 candidates)
            interview_cutoff = min(10, max(2, int(len(candidate_scores) * 0.15)))
            interview_candidates = candidate_scores[:interview_cutoff]
            
            # Conduct interviews
            for app, candidate, score in interview_candidates:
                app.outcome = ApplicationOutcome.INTERVIEW
                results['interviews_conducted'].append(app.application_id)
            
            # Make offers (top 1-3 candidates depending on position availability)
            offer_count = min(position.number_of_positions, len(interview_candidates))
            if offer_count > 0:
                offer_candidates = interview_candidates[:offer_count]
                for app, candidate, score in offer_candidates:
                    app.outcome = ApplicationOutcome.OFFER
                    results['offers_made'].append(app.application_id)
                    results['positions_filled'].append(position.position_id)
            
            # Reject remaining candidates
            remaining_candidates = candidate_scores[interview_cutoff:]
            for app, candidate, score in remaining_candidates:
                app.outcome = ApplicationOutcome.REJECTION
                results['rejections_sent'].append(app.application_id)
        
        return results
    
    def _score_candidate_for_position(self, candidate: JobMarketCandidate, 
                                    position: JobPosition) -> float:
        """
        Score a candidate's fit for a specific position.
        
        Args:
            candidate: The job market candidate
            position: The job position
            
        Returns:
            Score from 0.0 to 1.0
        """
        researcher = candidate.researcher
        score = 0.0
        
        # Base competitiveness (40% of score)
        score += candidate.market_competitiveness * 0.4
        
        # Level appropriateness (20% of score)
        level_match = self._calculate_level_match(researcher.level, position.position_type)
        score += level_match * 0.2
        
        # Field match (15% of score)
        field_match = 1.0 if researcher.specialty.lower() == position.field.lower() else 0.5
        score += field_match * 0.15
        
        # Institution tier match (10% of score)
        tier_competitiveness = self.INSTITUTION_COMPETITIVENESS[position.institution_tier]
        if researcher.reputation_score >= tier_competitiveness['candidate_quality_threshold']:
            tier_match = 1.0
        else:
            tier_match = researcher.reputation_score / tier_competitiveness['candidate_quality_threshold']
        score += tier_match * 0.1
        
        # Geographic preference (10% of score)
        geo_match = 1.0 if ("Any" in candidate.geographic_preferences or 
                           any(pref in position.location for pref in candidate.geographic_preferences)) else 0.3
        score += geo_match * 0.1
        
        # Salary expectations alignment (5% of score)
        salary_match = self._calculate_salary_match(candidate.salary_expectations, position.salary_range)
        score += salary_match * 0.05
        
        return min(1.0, max(0.0, score))
    
    def _calculate_level_match(self, researcher_level: ResearcherLevel, 
                             position_type: PositionType) -> float:
        """Calculate how well researcher level matches position type."""
        level_position_match = {
            ResearcherLevel.GRADUATE_STUDENT: {
                PositionType.POSTDOC: 0.8,
                PositionType.ASSISTANT_PROF: 0.2,
                PositionType.INDUSTRY_RESEARCH: 0.6,
                PositionType.INDUSTRY_APPLIED: 0.7
            },
            ResearcherLevel.POSTDOC: {
                PositionType.POSTDOC: 1.0,
                PositionType.ASSISTANT_PROF: 0.9,
                PositionType.INDUSTRY_RESEARCH: 0.8,
                PositionType.INDUSTRY_APPLIED: 0.7,
                PositionType.GOVERNMENT_LAB: 0.8
            },
            ResearcherLevel.ASSISTANT_PROF: {
                PositionType.ASSISTANT_PROF: 1.0,
                PositionType.ASSOCIATE_PROF: 0.7,
                PositionType.INDUSTRY_RESEARCH: 0.9,
                PositionType.INDUSTRY_APPLIED: 0.8,
                PositionType.GOVERNMENT_LAB: 0.8,
                PositionType.TEACHING_FOCUSED: 0.9
            },
            ResearcherLevel.ASSOCIATE_PROF: {
                PositionType.ASSOCIATE_PROF: 1.0,
                PositionType.FULL_PROF: 0.8,
                PositionType.ASSISTANT_PROF: 0.6,
                PositionType.INDUSTRY_RESEARCH: 0.9,
                PositionType.INDUSTRY_APPLIED: 0.8,
                PositionType.GOVERNMENT_LAB: 0.9,
                PositionType.TEACHING_FOCUSED: 0.9
            },
            ResearcherLevel.FULL_PROF: {
                PositionType.FULL_PROF: 1.0,
                PositionType.ASSOCIATE_PROF: 0.7,
                PositionType.INDUSTRY_RESEARCH: 0.9,
                PositionType.INDUSTRY_APPLIED: 0.7,
                PositionType.GOVERNMENT_LAB: 0.9,
                PositionType.TEACHING_FOCUSED: 0.8
            },
            ResearcherLevel.EMERITUS: {
                PositionType.TEACHING_FOCUSED: 0.9,
                PositionType.GOVERNMENT_LAB: 0.7,
                PositionType.INDUSTRY_RESEARCH: 0.6
            }
        }
        
        return level_position_match.get(researcher_level, {}).get(position_type, 0.3)
    
    def _calculate_salary_match(self, expectations: Tuple[int, int], 
                              offered_range: Tuple[int, int]) -> float:
        """Calculate how well salary expectations match offered range."""
        exp_min, exp_max = expectations
        off_min, off_max = offered_range
        
        # Check for overlap
        overlap_min = max(exp_min, off_min)
        overlap_max = min(exp_max, off_max)
        
        if overlap_min <= overlap_max:
            # There's overlap - calculate how much
            overlap_size = overlap_max - overlap_min
            exp_range_size = exp_max - exp_min
            off_range_size = off_max - off_min
            
            # Score based on overlap relative to both ranges
            exp_overlap_ratio = overlap_size / exp_range_size if exp_range_size > 0 else 1.0
            off_overlap_ratio = overlap_size / off_range_size if off_range_size > 0 else 1.0
            
            return (exp_overlap_ratio + off_overlap_ratio) / 2.0
        else:
            # No overlap - penalize based on distance
            if exp_min > off_max:
                # Expectations too high
                distance = exp_min - off_max
                penalty = min(1.0, distance / off_max)
                return max(0.0, 1.0 - penalty)
            else:
                # Expectations too low (less of a problem)
                return 0.7
        
        return 0.0
    
    def _score_candidate_for_position(self, candidate: JobMarketCandidate, 
                                    position: JobPosition) -> float:
        """Score a candidate's fit for a specific position."""
        researcher = candidate.researcher
        
        # Base competitiveness score
        base_score = candidate.market_competitiveness
        
        # Position-specific adjustments
        if position.position_type == PositionType.POSTDOC:
            # Favor recent PhDs with strong research
            if researcher.level == ResearcherLevel.GRADUATE_STUDENT:
                base_score += 0.2
            elif researcher.level == ResearcherLevel.POSTDOC:
                base_score += 0.1
        elif position.position_type == PositionType.ASSISTANT_PROF:
            # Favor postdocs with good records
            if researcher.level == ResearcherLevel.POSTDOC:
                base_score += 0.2
            elif researcher.level == ResearcherLevel.GRADUATE_STUDENT:
                base_score -= 0.1  # Less preferred
        
        # Institution tier matching
        tier_competitiveness = self.INSTITUTION_COMPETITIVENESS[position.institution_tier]
        if candidate.market_competitiveness < tier_competitiveness['candidate_quality_threshold']:
            base_score *= 0.7  # Penalty for not meeting threshold
        
        # Field matching
        if researcher.specialty.lower() == position.field.lower():
            base_score += 0.1
        
        # Add some randomness for realism
        randomness = random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, base_score + randomness))
    
    def _calculate_market_results(self, year: int, positions: List[JobPosition],
                                candidates: List[JobMarketCandidate],
                                hiring_results: Dict[str, Any], field: str) -> JobMarketResults:
        """Calculate comprehensive market results."""
        total_positions = len(positions)
        total_candidates = len(candidates)
        competition_ratio = total_candidates / max(1, total_positions)
        
        # Calculate placement rate
        offers_made = len(hiring_results['offers_made'])
        placement_rate = offers_made / max(1, total_candidates)
        
        # Position type breakdown
        position_outcomes = {}
        for pos_type in PositionType:
            type_positions = [p for p in positions if p.position_type == pos_type]
            position_outcomes[pos_type] = {
                'total_positions': len(type_positions),
                'average_salary': sum(p.salary_range[0] + p.salary_range[1] for p in type_positions) / (2 * max(1, len(type_positions))),
                'competition_ratio': total_candidates / max(1, len(type_positions)) if type_positions else 0
            }
        
        # Candidate outcomes
        candidate_outcomes = {}
        for candidate in candidates:
            candidate_outcomes[candidate.researcher.id] = {
                'competitiveness': candidate.market_competitiveness,
                'applications_sent': 0,  # Would be calculated from applications
                'interviews_received': 0,
                'offers_received': 0,
                'final_outcome': 'pending'
            }
        
        # Market trends
        market_trends = {
            'position_growth': 0.02,  # 2% growth (placeholder)
            'salary_inflation': 0.03,  # 3% salary growth
            'competition_intensity': competition_ratio
        }
        
        # Field analysis
        field_analysis = {
            field: {
                'total_positions': total_positions,
                'placement_rate': placement_rate,
                'average_competition': competition_ratio,
                'market_health': 'healthy' if placement_rate > 0.3 else 'competitive'
            }
        }
        
        return JobMarketResults(
            year=year,
            total_positions=total_positions,
            total_candidates=total_candidates,
            competition_ratio=competition_ratio,
            placement_rate=placement_rate,
            position_outcomes=position_outcomes,
            candidate_outcomes=candidate_outcomes,
            market_trends=market_trends,
            field_analysis=field_analysis
        )
    
    def _predict_timeline(self, position_type: PositionType, competitiveness: float) -> float:
        """Predict timeline to job offer in months."""
        base_timeline = {
            PositionType.POSTDOC: 4.0,
            PositionType.ASSISTANT_PROF: 8.0,
            PositionType.ASSOCIATE_PROF: 6.0,
            PositionType.FULL_PROF: 5.0,
            PositionType.INDUSTRY_RESEARCH: 3.0,
            PositionType.INDUSTRY_APPLIED: 2.5
        }.get(position_type, 6.0)
        
        # More competitive candidates get offers faster
        competitiveness_factor = 2.0 - competitiveness  # 1.0 to 2.0 range
        
        return base_timeline * competitiveness_factor
    
    def _predict_salary_outcome(self, position_type: PositionType, 
                              researcher: EnhancedResearcher,
                              expectations: Tuple[int, int]) -> Tuple[int, int]:
        """Predict likely salary outcome."""
        # Get market range for position type
        market_range = self._calculate_salary_range(position_type, InstitutionTier.R1_MID)
        
        # Adjust based on researcher competitiveness
        competitiveness = self._calculate_market_competitiveness(researcher)
        
        # Calculate likely range within market bounds
        range_width = market_range[1] - market_range[0]
        competitiveness_adjustment = range_width * (competitiveness - 0.5)
        
        predicted_min = int(market_range[0] + competitiveness_adjustment * 0.5)
        predicted_max = int(market_range[1] + competitiveness_adjustment * 0.5)
        
        return (max(market_range[0], predicted_min), min(market_range[1], predicted_max))
    
    def _identify_candidate_strengths(self, researcher: EnhancedResearcher, 
                                    position_type: PositionType) -> List[str]:
        """Identify candidate's key strengths for position type."""
        strengths = []
        
        if researcher.h_index > 15:
            strengths.append("Strong research impact")
        if len(researcher.publication_history) > 10:
            strengths.append("Productive publication record")
        if researcher.total_citations > 200:
            strengths.append("High citation impact")
        if researcher.institution_tier == 1:
            strengths.append("Top-tier institutional background")
        if researcher.years_active > 5:
            strengths.append("Substantial research experience")
        
        return strengths[:3]  # Return top 3 strengths
    
    def _identify_improvement_areas(self, researcher: EnhancedResearcher,
                                  position_type: PositionType) -> List[str]:
        """Identify areas for candidate improvement."""
        improvements = []
        
        if researcher.h_index < 8:
            improvements.append("Increase research impact and citations")
        if len(researcher.publication_history) < 5:
            improvements.append("Build stronger publication record")
        if researcher.years_active < 3:
            improvements.append("Gain more research experience")
        if not hasattr(researcher, 'teaching_experience'):
            improvements.append("Develop teaching experience")
        
        return improvements[:2]  # Return top 2 improvement areas
    
    def _assess_market_readiness(self, candidate: JobMarketCandidate) -> str:
        """Assess candidate's readiness for job market."""
        competitiveness = candidate.market_competitiveness
        
        if competitiveness >= 0.8:
            return "Highly competitive"
        elif competitiveness >= 0.6:
            return "Competitive"
        elif competitiveness >= 0.4:
            return "Moderately competitive"
        else:
            return "Needs improvement"
    
    def _generate_strategic_recommendations(self, candidate: JobMarketCandidate,
                                          predictions: Dict[str, Any]) -> List[str]:
        """Generate strategic recommendations for candidate."""
        recommendations = []
        
        # Application strategy recommendations
        if candidate.application_strategy == "conservative":
            recommendations.append("Consider applying to more positions to increase chances")
        elif candidate.application_strategy == "aggressive":
            recommendations.append("Focus applications on positions that best match your profile")
        
        # Position type recommendations
        best_position = max(predictions.keys(), 
                          key=lambda x: predictions[x]['success_probability'])
        recommendations.append(f"Focus on {best_position} positions where you're most competitive")
        
        # Competitiveness recommendations
        if candidate.market_competitiveness < 0.5:
            recommendations.append("Consider additional training or experience before entering market")
        
        return recommendations
    
    def _assess_market_conditions(self, field: str, year: int) -> Dict[str, Any]:
        """Assess overall market conditions for field and year."""
        field_params = self.MARKET_PARAMETERS.get(field, self.MARKET_PARAMETERS['computer_science'])
        
        return {
            'market_health': 'competitive',
            'position_availability': 'moderate',
            'competition_level': 'high',
            'salary_trends': 'stable',
            'growth_outlook': 'steady',
            'key_factors': [
                'High competition for faculty positions',
                'Growing industry opportunities',
                'Geographic concentration in tech hubs'
            ]
        }
    
    def get_market_statistics(self, field: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive market statistics.
        
        Args:
            field: Optional field filter
            
        Returns:
            Dictionary with market statistics
        """
        if not self.market_history:
            return {'error': 'No market data available'}
        
        recent_results = self.market_history[-1] if self.market_history else None
        if not recent_results:
            return {'error': 'No recent market data'}
        
        stats = {
            'total_positions': len(self.available_positions),
            'total_candidates': len(self.job_candidates),
            'recent_placement_rate': recent_results.placement_rate,
            'recent_competition_ratio': recent_results.competition_ratio,
            'position_distribution': {},
            'candidate_competitiveness_distribution': {},
            'market_trends': recent_results.market_trends
        }
        
        # Position type distribution
        for pos in self.available_positions.values():
            pos_type = pos.position_type.value
            stats['position_distribution'][pos_type] = stats['position_distribution'].get(pos_type, 0) + 1
        
        # Candidate competitiveness distribution
        competitiveness_ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for range_min, range_max in competitiveness_ranges:
            range_key = f"{range_min:.1f}-{range_max:.1f}"
            count = len([c for c in self.job_candidates.values() 
                        if range_min <= c.market_competitiveness < range_max])
            stats['candidate_competitiveness_distribution'][range_key] = count
        
        return stats
    
    def export_market_data(self, year: Optional[int] = None) -> Dict[str, Any]:
        """
        Export market data for analysis.
        
        Args:
            year: Optional year filter
            
        Returns:
            Dictionary with exportable market data
        """
        if year:
            results = [r for r in self.market_history if r.year == year]
        else:
            results = self.market_history
        
        return {
            'market_results': [
                {
                    'year': r.year,
                    'total_positions': r.total_positions,
                    'total_candidates': r.total_candidates,
                    'competition_ratio': r.competition_ratio,
                    'placement_rate': r.placement_rate,
                    'market_trends': r.market_trends
                }
                for r in results
            ],
            'current_positions': [
                {
                    'position_id': pos.position_id,
                    'position_type': pos.position_type.value,
                    'institution_tier': pos.institution_tier.value,
                    'field': pos.field,
                    'salary_range': pos.salary_range,
                    'expected_applicants': pos.expected_applicants
                }
                for pos in self.available_positions.values()
            ],
            'registered_candidates': [
                {
                    'candidate_id': candidate.researcher.id,
                    'researcher_level': candidate.researcher.level.value,
                    'target_positions': [pos.value for pos in candidate.target_positions],
                    'market_competitiveness': candidate.market_competitiveness,
                    'application_strategy': candidate.application_strategy
                }
                for candidate in self.job_candidates.values()
            ]
        }