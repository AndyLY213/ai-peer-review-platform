"""
Funding Integration System

This module implements funding agency and cycle management for NSF, NIH, and industry funding
with 1-3 year cycle modeling, funding deadlines, and application processes.
"""

import uuid
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from src.core.exceptions import ValidationError
from src.data.enhanced_models import ResearcherLevel

logger = logging.getLogger(__name__)


class FundingAgencyType(Enum):
    """Types of funding agencies."""
    NSF = "NSF"
    NIH = "NIH"
    INDUSTRY = "Industry"
    PRIVATE_FOUNDATION = "Private Foundation"


class FundingCycleStatus(Enum):
    """Status of funding cycles."""
    PLANNING = "Planning"
    OPEN = "Open"
    REVIEW = "Review"
    AWARDED = "Awarded"
    ACTIVE = "Active"
    COMPLETED = "Completed"


class ApplicationStatus(Enum):
    """Status of funding applications."""
    DRAFT = "Draft"
    SUBMITTED = "Submitted"
    UNDER_REVIEW = "Under Review"
    AWARDED = "Awarded"
    REJECTED = "Rejected"


@dataclass
class FundingAgency:
    """Represents a funding agency with specific characteristics and requirements."""
    
    agency_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    agency_type: FundingAgencyType = FundingAgencyType.NSF
    typical_cycle_duration_years: int = 1  # 1-3 years
    typical_funding_amount_range: Tuple[int, int] = (50000, 500000)
    success_rate: float = 0.15
    min_career_level: ResearcherLevel = ResearcherLevel.POSTDOC
    application_deadlines_per_year: List[str] = field(default_factory=list)
    max_applications_per_researcher: int = 2
    
    def __post_init__(self):
        """Validate agency parameters."""
        if not (1 <= self.typical_cycle_duration_years <= 3):
            raise ValidationError("typical_cycle_duration_years", self.typical_cycle_duration_years, "1-3 years")
        
        if not (0.0 <= self.success_rate <= 1.0):
            raise ValidationError("success_rate", self.success_rate, "0.0-1.0")
        
        if not self.application_deadlines_per_year:
            if self.agency_type == FundingAgencyType.NSF:
                self.application_deadlines_per_year = ["February 15", "August 15"]
            elif self.agency_type == FundingAgencyType.NIH:
                self.application_deadlines_per_year = ["February 5", "June 5", "October 5"]
            else:
                self.application_deadlines_per_year = ["January 31", "July 31"]
    
    def get_next_deadline(self, current_date: Optional[date] = None) -> Optional[date]:
        """Get the next application deadline."""
        if current_date is None:
            current_date = date.today()
        
        current_year = current_date.year
        next_deadlines = []
        
        for deadline_str in self.application_deadlines_per_year:
            try:
                month_day = datetime.strptime(deadline_str, "%B %d")
                deadline_this_year = date(current_year, month_day.month, month_day.day)
                
                if deadline_this_year >= current_date:
                    next_deadlines.append(deadline_this_year)
                else:
                    next_deadlines.append(date(current_year + 1, month_day.month, month_day.day))
            except ValueError:
                continue
        
        return min(next_deadlines) if next_deadlines else None
    
    def is_researcher_eligible(self, career_level: ResearcherLevel, current_applications: int) -> Tuple[bool, List[str]]:
        """
        Check if a researcher is eligible to apply for funding from this agency.
        
        Args:
            career_level: The researcher's career level
            current_applications: Number of current applications the researcher has with this agency
            
        Returns:
            Tuple of (is_eligible, list_of_issues)
        """
        issues = []
        
        # Check career level requirement
        career_levels_order = [
            ResearcherLevel.GRADUATE_STUDENT,
            ResearcherLevel.POSTDOC,
            ResearcherLevel.ASSISTANT_PROF,
            ResearcherLevel.ASSOCIATE_PROF,
            ResearcherLevel.FULL_PROF,
            ResearcherLevel.EMERITUS
        ]
        
        if career_levels_order.index(career_level) < career_levels_order.index(self.min_career_level):
            issues.append(f"Minimum career level required: {self.min_career_level.value}")
        
        # Check application limit
        if current_applications >= self.max_applications_per_researcher:
            issues.append(f"Maximum {self.max_applications_per_researcher} applications allowed per researcher")
        
        return len(issues) == 0, issues


@dataclass
class FundingCycle:
    """Represents a specific funding cycle with timeline and requirements."""
    
    cycle_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agency_id: str = ""
    cycle_name: str = ""
    fiscal_year: int = field(default_factory=lambda: date.today().year)
    duration_years: int = 1
    total_budget: int = 1000000
    expected_awards: int = 10
    status: FundingCycleStatus = FundingCycleStatus.PLANNING
    application_open_date: date = field(default_factory=date.today)
    application_deadline: date = field(default_factory=lambda: date.today() + timedelta(days=90))
    review_start_date: date = field(default_factory=lambda: date.today() + timedelta(days=100))
    notification_date: date = field(default_factory=lambda: date.today() + timedelta(days=180))
    funding_start_date: date = field(default_factory=lambda: date.today() + timedelta(days=210))
    funding_end_date: date = field(default_factory=lambda: date.today() + timedelta(days=940))
    applications: List[str] = field(default_factory=list)
    awarded_applications: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate funding cycle timeline."""
        dates = [
            self.application_open_date,
            self.application_deadline,
            self.review_start_date,
            self.notification_date,
            self.funding_start_date,
            self.funding_end_date
        ]
        
        # Check that dates are in chronological order
        for i in range(len(dates) - 1):
            if dates[i] >= dates[i + 1]:
                raise ValidationError("timeline", "dates", "chronological order")
    
    def is_application_open(self, current_date: Optional[date] = None) -> bool:
        """Check if applications are currently being accepted."""
        if current_date is None:
            current_date = date.today()
        
        return (self.application_open_date <= current_date <= self.application_deadline and
                self.status in [FundingCycleStatus.PLANNING, FundingCycleStatus.OPEN])
    
    def get_days_until_deadline(self, current_date: Optional[date] = None) -> int:
        """Get number of days until application deadline."""
        if current_date is None:
            current_date = date.today()
        
        return (self.application_deadline - current_date).days
    
    def calculate_success_rate(self) -> float:
        """Calculate success rate for this cycle."""
        if not self.applications:
            return 0.0
        
        return len(self.awarded_applications) / len(self.applications)
    
    def get_average_award_amount(self) -> float:
        """Get average award amount for this cycle."""
        if not self.awarded_applications:
            return 0.0
        
        # This would need access to the funding system to get actual amounts
        # For now, return a placeholder
        return self.total_budget / self.expected_awards if self.expected_awards > 0 else 0.0


@dataclass
class FundingApplication:
    """Represents a funding application."""
    
    application_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cycle_id: str = ""
    researcher_id: str = ""
    title: str = ""
    requested_amount: int = 100000
    status: ApplicationStatus = ApplicationStatus.DRAFT
    is_awarded: bool = False
    awarded_amount: int = 0


class FundingSystem:
    """Main funding system managing agencies, cycles, and applications."""
    
    def __init__(self):
        """Initialize the funding system."""
        self.agencies: Dict[str, FundingAgency] = {}
        self.cycles: Dict[str, FundingCycle] = {}
        self.applications: Dict[str, FundingApplication] = {}
        self._create_default_agencies()
    
    def _create_default_agencies(self):
        """Create default funding agencies (NSF, NIH, Industry)."""
        # NSF Agency
        nsf = FundingAgency(
            name="National Science Foundation",
            agency_type=FundingAgencyType.NSF,
            typical_cycle_duration_years=3,
            success_rate=0.20,
            min_career_level=ResearcherLevel.ASSISTANT_PROF
        )
        self.register_agency(nsf)
        
        # NIH Agency
        nih = FundingAgency(
            name="National Institutes of Health",
            agency_type=FundingAgencyType.NIH,
            typical_cycle_duration_years=2,
            success_rate=0.18,
            min_career_level=ResearcherLevel.POSTDOC
        )
        self.register_agency(nih)
        
        # Industry Agency
        industry = FundingAgency(
            name="Industry Research Consortium",
            agency_type=FundingAgencyType.INDUSTRY,
            typical_cycle_duration_years=1,
            success_rate=0.35,
            min_career_level=ResearcherLevel.POSTDOC
        )
        self.register_agency(industry)
    
    def register_agency(self, agency: FundingAgency) -> str:
        """Register a new funding agency."""
        self.agencies[agency.agency_id] = agency
        return agency.agency_id
    
    def get_agency(self, agency_id: str) -> Optional[FundingAgency]:
        """Get agency by ID."""
        return self.agencies.get(agency_id)
    
    def get_agencies_by_type(self, agency_type: FundingAgencyType) -> List[FundingAgency]:
        """Get all agencies of a specific type."""
        return [agency for agency in self.agencies.values() 
                if agency.agency_type == agency_type]
    
    def create_funding_cycle(self, agency_id: str, cycle_name: str, 
                           fiscal_year: Optional[int] = None,
                           total_budget: int = 1000000,
                           expected_awards: int = 10) -> str:
        """Create a new funding cycle."""
        agency = self.get_agency(agency_id)
        if not agency:
            raise ValidationError("agency_id", agency_id, "valid agency ID")
        
        if fiscal_year is None:
            fiscal_year = date.today().year
        
        cycle = FundingCycle(
            agency_id=agency_id,
            cycle_name=cycle_name,
            fiscal_year=fiscal_year,
            duration_years=agency.typical_cycle_duration_years,
            total_budget=total_budget,
            expected_awards=expected_awards
        )
        
        self.cycles[cycle.cycle_id] = cycle
        return cycle.cycle_id
    
    def get_cycle(self, cycle_id: str) -> Optional[FundingCycle]:
        """Get funding cycle by ID."""
        return self.cycles.get(cycle_id)
    
    def get_active_cycles(self, agency_type: Optional[FundingAgencyType] = None) -> List[FundingCycle]:
        """Get all active funding cycles, optionally filtered by agency type."""
        active_cycles = []
        
        for cycle in self.cycles.values():
            if cycle.status in [FundingCycleStatus.OPEN, FundingCycleStatus.REVIEW]:
                if agency_type is None:
                    active_cycles.append(cycle)
                else:
                    agency = self.get_agency(cycle.agency_id)
                    if agency and agency.agency_type == agency_type:
                        active_cycles.append(cycle)
        
        return active_cycles
    
    def submit_application(self, cycle_id: str, researcher_id: str, 
                          title: str, requested_amount: int) -> str:
        """
        Submit a funding application.
        
        Args:
            cycle_id: ID of the funding cycle
            researcher_id: ID of the researcher applying
            title: Title of the research proposal
            requested_amount: Amount of funding requested
            
        Returns:
            Application ID
            
        Raises:
            ValidationError: If application is invalid
        """
        cycle = self.get_cycle(cycle_id)
        if not cycle:
            raise ValidationError("cycle_id", cycle_id, "valid cycle ID")
        
        if not cycle.is_application_open():
            raise ValidationError("application", "timing", "application period is open")
        
        # Create application
        application = FundingApplication(
            cycle_id=cycle_id,
            researcher_id=researcher_id,
            title=title,
            requested_amount=requested_amount,
            status=ApplicationStatus.SUBMITTED
        )
        
        # Add to applications and cycle
        self.applications[application.application_id] = application
        cycle.applications.append(application.application_id)
        
        return application.application_id
    
    def get_application(self, application_id: str) -> Optional[FundingApplication]:
        """Get application by ID."""
        return self.applications.get(application_id)
    
    def get_applications_by_researcher(self, researcher_id: str) -> List[FundingApplication]:
        """Get all applications by a researcher."""
        return [app for app in self.applications.values() 
                if app.researcher_id == researcher_id]
    
    def get_researcher_applications(self, researcher_id: str) -> List[FundingApplication]:
        """Alias for get_applications_by_researcher for backward compatibility."""
        return self.get_applications_by_researcher(researcher_id)
    
    def get_applications_by_cycle(self, cycle_id: str) -> List[FundingApplication]:
        """Get all applications for a funding cycle."""
        return [app for app in self.applications.values() 
                if app.cycle_id == cycle_id]
    
    def award_funding(self, application_id: str, awarded_amount: int) -> bool:
        """
        Award funding to an application.
        
        Args:
            application_id: ID of the application
            awarded_amount: Amount of funding to award
            
        Returns:
            True if successful, False otherwise
        """
        application = self.get_application(application_id)
        if not application:
            return False
        
        cycle = self.get_cycle(application.cycle_id)
        if not cycle:
            return False
        
        # Update application
        application.status = ApplicationStatus.AWARDED
        application.is_awarded = True
        application.awarded_amount = awarded_amount
        
        # Add to cycle's awarded applications
        if application_id not in cycle.awarded_applications:
            cycle.awarded_applications.append(application_id)
        
        return True
    
    def reject_application(self, application_id: str) -> bool:
        """
        Reject a funding application.
        
        Args:
            application_id: ID of the application
            
        Returns:
            True if successful, False otherwise
        """
        application = self.get_application(application_id)
        if not application:
            return False
        
        application.status = ApplicationStatus.REJECTED
        return True
    
    def get_funding_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the funding system.
        
        Returns:
            Dictionary with funding statistics
        """
        total_applications = len(self.applications)
        awarded_applications = len([app for app in self.applications.values() 
                                  if app.is_awarded])
        
        total_requested = sum(app.requested_amount for app in self.applications.values())
        total_awarded = sum(app.awarded_amount for app in self.applications.values() 
                           if app.is_awarded)
        
        # Statistics by agency
        agency_stats = {}
        for agency_id, agency in self.agencies.items():
            cycles = [c for c in self.cycles.values() if c.agency_id == agency_id]
            applications = []
            for cycle in cycles:
                applications.extend(self.get_applications_by_cycle(cycle.cycle_id))
            
            agency_stats[agency.name] = {
                "cycles": len(cycles),
                "applications": len(applications),
                "awarded": len([app for app in applications if app.is_awarded]),
                "success_rate": len([app for app in applications if app.is_awarded]) / len(applications) if applications else 0
            }
        
        return {
            "total_agencies": len(self.agencies),
            "total_cycles": len(self.cycles),
            "total_applications": total_applications,
            "awarded_applications": awarded_applications,
            "overall_success_rate": awarded_applications / total_applications if total_applications > 0 else 0,
            "total_requested": total_requested,
            "total_awarded": total_awarded,
            "agency_statistics": agency_stats
        }