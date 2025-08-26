"""
OpenScienceManager module for modeling preprint and open access adoption in academic research.
"""

import json
import uuid
import random
from datetime import date, datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import math

from src.core.exceptions import ValidationError, PeerReviewError


class PreprintServer(Enum):
    """Types of preprint servers."""
    ARXIV = "arXiv"
    BIORXIV = "bioRxiv"
    MEDRXIV = "medRxiv"
    PSYARXIV = "PsyArXiv"
    SOCARXIV = "SocArXiv"
    CHEMRXIV = "ChemRxiv"
    PREPRINTS_ORG = "Preprints.org"
    RESEARCH_SQUARE = "Research Square"


class OpenAccessType(Enum):
    """Types of open access publishing."""
    GOLD = "gold"  # Published in OA journal
    GREEN = "green"  # Self-archived in repository
    HYBRID = "hybrid"  # OA article in subscription journal
    BRONZE = "bronze"  # Free to read but no license
    DIAMOND = "diamond"  # No author fees, no reader fees
    CLOSED = "closed"  # Traditional subscription model


class DataSharingLevel(Enum):
    """Levels of data sharing compliance."""
    FULL_OPEN = "full_open"  # All data and code publicly available
    PARTIAL_OPEN = "partial_open"  # Some data available, restrictions apply
    ON_REQUEST = "on_request"  # Data available upon reasonable request
    RESTRICTED = "restricted"  # Limited access due to privacy/ethics
    CLOSED = "closed"  # No data sharing


@dataclass
class PreprintRecord:
    """Represents a preprint submission."""
    preprint_id: str
    paper_id: str
    researcher_id: str
    server: PreprintServer
    submission_date: date
    version: int
    download_count: int = 0
    citation_count: int = 0
    view_count: int = 0
    comment_count: int = 0
    is_published: bool = False
    published_venue_id: Optional[str] = None
    published_date: Optional[date] = None
    
    def __post_init__(self):
        """Validate preprint record data."""
        if self.version < 1:
            raise ValidationError("version", self.version, "positive integer")
        if self.download_count < 0:
            raise ValidationError("download_count", self.download_count, "non-negative")
        if self.citation_count < 0:
            raise ValidationError("citation_count", self.citation_count, "non-negative")


@dataclass
class OpenAccessRecord:
    """Represents open access publishing information."""
    paper_id: str
    access_type: OpenAccessType
    publication_date: date
    license_type: Optional[str] = None
    apc_cost: Optional[float] = None  # Article Processing Charge
    embargo_period_months: Optional[int] = None
    repository_url: Optional[str] = None
    is_compliant: bool = True
    funder_mandate: Optional[str] = None
    
    def __post_init__(self):
        """Validate open access record data."""
        if self.apc_cost is not None and self.apc_cost < 0:
            raise ValidationError("apc_cost", self.apc_cost, "non-negative")
        if self.embargo_period_months is not None and self.embargo_period_months < 0:
            raise ValidationError("embargo_period_months", self.embargo_period_months, "non-negative")


@dataclass
class DataSharingRecord:
    """Represents data sharing compliance information."""
    paper_id: str
    sharing_level: DataSharingLevel
    repository_url: Optional[str] = None
    access_conditions: Optional[str] = None
    data_availability_statement: Optional[str] = None
    code_availability: bool = False
    code_repository_url: Optional[str] = None
    compliance_score: float = 0.0
    funder_requirements: List[str] = None
    
    def __post_init__(self):
        """Validate data sharing record data."""
        if not (0.0 <= self.compliance_score <= 1.0):
            raise ValidationError("compliance_score", self.compliance_score, "between 0.0 and 1.0")
        if self.funder_requirements is None:
            self.funder_requirements = []


@dataclass
class OpenScienceProfile:
    """Represents a researcher's open science adoption profile."""
    researcher_id: str
    preprint_adoption_rate: float
    open_access_preference: float
    data_sharing_willingness: float
    preferred_preprint_server: Optional[PreprintServer] = None
    preferred_oa_type: Optional[OpenAccessType] = None
    institutional_mandate: bool = False
    funder_mandate: bool = False
    career_stage_influence: float = 0.0
    
    def __post_init__(self):
        """Validate open science profile data."""
        if not (0.0 <= self.preprint_adoption_rate <= 1.0):
            raise ValidationError("preprint_adoption_rate", self.preprint_adoption_rate, "between 0.0 and 1.0")
        if not (0.0 <= self.open_access_preference <= 1.0):
            raise ValidationError("open_access_preference", self.open_access_preference, "between 0.0 and 1.0")
        if not (0.0 <= self.data_sharing_willingness <= 1.0):
            raise ValidationError("data_sharing_willingness", self.data_sharing_willingness, "between 0.0 and 1.0")


@dataclass
class OpenScienceMetrics:
    """Aggregated metrics for open science adoption."""
    total_preprints: int
    preprint_adoption_rate: float
    open_access_rate: float
    data_sharing_compliance_rate: float
    average_preprint_citations: float
    average_time_to_publication: float  # days from preprint to publication
    server_usage_distribution: Dict[PreprintServer, int]
    oa_type_distribution: Dict[OpenAccessType, int]
    data_sharing_distribution: Dict[DataSharingLevel, int]
    
    def __post_init__(self):
        """Validate open science metrics."""
        if self.total_preprints < 0:
            raise ValidationError("total_preprints", self.total_preprints, "non-negative")
        if not (0.0 <= self.preprint_adoption_rate <= 1.0):
            raise ValidationError("preprint_adoption_rate", self.preprint_adoption_rate, "between 0.0 and 1.0")


class OpenScienceManager:
    """
    Manages open science practices including preprint servers, open access publishing,
    and data sharing requirements.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the OpenScienceManager."""
        self.data_dir = data_dir or Path("data/open_science")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for open science records
        self.preprint_records: Dict[str, PreprintRecord] = {}
        self.open_access_records: Dict[str, OpenAccessRecord] = {}
        self.data_sharing_records: Dict[str, DataSharingRecord] = {}
        self.researcher_profiles: Dict[str, OpenScienceProfile] = {}
        
        # Configuration
        self.server_field_mapping = {
            "computer_science": PreprintServer.ARXIV,
            "physics": PreprintServer.ARXIV,
            "mathematics": PreprintServer.ARXIV,
            "biology": PreprintServer.BIORXIV,
            "medicine": PreprintServer.MEDRXIV,
            "psychology": PreprintServer.PSYARXIV,
            "sociology": PreprintServer.SOCARXIV,
            "chemistry": PreprintServer.CHEMRXIV
        }
        
        # Load existing data
        self._load_data()
    
    def create_researcher_profile(self, researcher_id: str, career_stage: str, 
                                field: str, institutional_mandate: bool = False,
                                funder_mandate: bool = False) -> OpenScienceProfile:
        """Create an open science profile for a researcher."""
        # Career stage influences adoption rates
        career_multipliers = {
            "Graduate Student": 0.8,  # High adoption, tech-savvy
            "Postdoc": 0.9,  # Very high adoption, career building
            "Assistant Prof": 0.7,  # Moderate adoption, career pressure
            "Associate Prof": 0.6,  # Lower adoption, established
            "Full Prof": 0.5,  # Lowest adoption, traditional
            "Emeritus": 0.3   # Very low adoption
        }
        
        base_multiplier = career_multipliers.get(career_stage, 0.6)
        
        # Field influences adoption rates
        field_multipliers = {
            "computer_science": 1.2,
            "physics": 1.1,
            "biology": 1.0,
            "medicine": 0.8,
            "psychology": 0.9,
            "sociology": 0.7,
            "chemistry": 0.8
        }
        
        field_multiplier = field_multipliers.get(field.lower(), 0.8)
        
        # Calculate adoption rates with some randomness
        preprint_rate = min(1.0, base_multiplier * field_multiplier * random.uniform(0.8, 1.2))
        oa_preference = min(1.0, base_multiplier * random.uniform(0.7, 1.1))
        data_sharing = min(1.0, base_multiplier * 0.8 * random.uniform(0.6, 1.0))
        
        # Mandates increase adoption
        if institutional_mandate:
            preprint_rate = min(1.0, preprint_rate * 1.3)
            oa_preference = min(1.0, oa_preference * 1.4)
            data_sharing = min(1.0, data_sharing * 1.5)
        
        if funder_mandate:
            oa_preference = min(1.0, oa_preference * 1.6)
            data_sharing = min(1.0, data_sharing * 1.8)
        
        profile = OpenScienceProfile(
            researcher_id=researcher_id,
            preprint_adoption_rate=preprint_rate,
            open_access_preference=oa_preference,
            data_sharing_willingness=data_sharing,
            preferred_preprint_server=self.server_field_mapping.get(field.lower()),
            preferred_oa_type=OpenAccessType.GREEN if oa_preference > 0.7 else OpenAccessType.HYBRID,
            institutional_mandate=institutional_mandate,
            funder_mandate=funder_mandate,
            career_stage_influence=base_multiplier
        )
        
        self.researcher_profiles[researcher_id] = profile
        return profile
    
    def submit_preprint(self, paper_id: str, researcher_id: str, field: str) -> Optional[PreprintRecord]:
        """Submit a paper as a preprint."""
        profile = self.researcher_profiles.get(researcher_id)
        if not profile:
            # Create default profile if none exists
            profile = self.create_researcher_profile(researcher_id, "Assistant Prof", field)
        
        # Check if researcher adopts preprints
        if random.random() > profile.preprint_adoption_rate:
            return None
        
        # Determine server
        server = profile.preferred_preprint_server or self.server_field_mapping.get(field.lower(), PreprintServer.PREPRINTS_ORG)
        
        preprint_record = PreprintRecord(
            preprint_id=str(uuid.uuid4()),
            paper_id=paper_id,
            researcher_id=researcher_id,
            server=server,
            submission_date=date.today(),
            version=1,
            download_count=random.randint(10, 500),
            view_count=random.randint(50, 2000),
            citation_count=random.randint(0, 20)
        )
        
        self.preprint_records[preprint_record.preprint_id] = preprint_record
        return preprint_record
    
    def track_preprint_usage(self, preprint_id: str) -> Dict[str, int]:
        """Track usage metrics for a preprint."""
        if preprint_id not in self.preprint_records:
            raise ValidationError("preprint_id", preprint_id, "existing preprint ID")
        
        record = self.preprint_records[preprint_id]
        
        # Simulate usage growth over time
        days_since_submission = (date.today() - record.submission_date).days
        growth_factor = min(2.0, 1.0 + days_since_submission / 365.0)
        
        record.download_count = int(record.download_count * growth_factor)
        record.view_count = int(record.view_count * growth_factor)
        record.citation_count = int(record.citation_count * growth_factor * 0.1)
        
        return {
            "downloads": record.download_count,
            "views": record.view_count,
            "citations": record.citation_count,
            "comments": record.comment_count
        }
    
    def publish_open_access(self, paper_id: str, researcher_id: str, 
                          venue_type: str, has_funding: bool = False) -> OpenAccessRecord:
        """Create an open access publication record."""
        profile = self.researcher_profiles.get(researcher_id)
        if not profile:
            profile = self.create_researcher_profile(researcher_id, "Assistant Prof", "computer_science")
        
        # Determine access type based on preferences and constraints
        if profile.funder_mandate or has_funding:
            access_type = OpenAccessType.GOLD if random.random() > 0.3 else OpenAccessType.GREEN
        elif profile.open_access_preference > 0.8:
            access_type = OpenAccessType.GREEN
        elif profile.open_access_preference > 0.5:
            access_type = OpenAccessType.HYBRID
        else:
            access_type = OpenAccessType.CLOSED
        
        # Calculate APC costs
        apc_cost = None
        if access_type == OpenAccessType.GOLD:
            venue_apc_ranges = {
                "Top Journal": (2000, 5000),
                "Specialized Journal": (1000, 3000),
                "General Journal": (500, 2000),
                "Top Conference": (0, 500),
                "Mid Conference": (0, 300)
            }
            apc_range = venue_apc_ranges.get(venue_type, (0, 1000))
            apc_cost = random.uniform(apc_range[0], apc_range[1])
        elif access_type == OpenAccessType.HYBRID:
            apc_cost = random.uniform(1500, 4000)
        
        # Determine embargo period for green OA
        embargo_months = None
        if access_type == OpenAccessType.GREEN:
            embargo_months = random.choice([0, 6, 12, 24])
        
        record = OpenAccessRecord(
            paper_id=paper_id,
            access_type=access_type,
            publication_date=date.today(),
            license_type="CC-BY" if access_type in [OpenAccessType.GOLD, OpenAccessType.DIAMOND] else None,
            apc_cost=apc_cost,
            embargo_period_months=embargo_months,
            repository_url="https://repository.example.edu" if access_type == OpenAccessType.GREEN else None,
            is_compliant=profile.funder_mandate or profile.institutional_mandate,
            funder_mandate="NSF" if has_funding else None
        )
        
        self.open_access_records[paper_id] = record
        return record
    
    def enforce_data_sharing_requirements(self, paper_id: str, researcher_id: str,
                                        has_human_subjects: bool = False,
                                        has_proprietary_data: bool = False) -> DataSharingRecord:
        """Enforce data sharing requirements for a publication."""
        profile = self.researcher_profiles.get(researcher_id)
        if not profile:
            profile = self.create_researcher_profile(researcher_id, "Assistant Prof", "computer_science")
        
        # Determine sharing level based on constraints and willingness
        if has_human_subjects or has_proprietary_data:
            if profile.data_sharing_willingness > 0.8:
                sharing_level = DataSharingLevel.ON_REQUEST
            else:
                sharing_level = DataSharingLevel.RESTRICTED
        elif profile.data_sharing_willingness > 0.9:
            sharing_level = DataSharingLevel.FULL_OPEN
        elif profile.data_sharing_willingness > 0.6:
            sharing_level = DataSharingLevel.PARTIAL_OPEN
        elif profile.data_sharing_willingness > 0.3:
            sharing_level = DataSharingLevel.ON_REQUEST
        else:
            sharing_level = DataSharingLevel.CLOSED
        
        # Mandates increase sharing
        if profile.funder_mandate and sharing_level == DataSharingLevel.CLOSED:
            sharing_level = DataSharingLevel.ON_REQUEST
        
        # Calculate compliance score
        compliance_scores = {
            DataSharingLevel.FULL_OPEN: 1.0,
            DataSharingLevel.PARTIAL_OPEN: 0.8,
            DataSharingLevel.ON_REQUEST: 0.6,
            DataSharingLevel.RESTRICTED: 0.4,
            DataSharingLevel.CLOSED: 0.0
        }
        
        compliance_score = compliance_scores[sharing_level]
        
        # Code availability
        code_available = (sharing_level in [DataSharingLevel.FULL_OPEN, DataSharingLevel.PARTIAL_OPEN] 
                         and random.random() < profile.data_sharing_willingness)
        
        record = DataSharingRecord(
            paper_id=paper_id,
            sharing_level=sharing_level,
            repository_url="https://dataverse.example.edu" if sharing_level != DataSharingLevel.CLOSED else None,
            access_conditions="Upon reasonable request" if sharing_level == DataSharingLevel.ON_REQUEST else None,
            data_availability_statement=f"Data sharing: {sharing_level.value}",
            code_availability=code_available,
            code_repository_url="https://github.com/example/repo" if code_available else None,
            compliance_score=compliance_score,
            funder_requirements=["NSF"] if profile.funder_mandate else []
        )
        
        self.data_sharing_records[paper_id] = record
        return record
    
    def calculate_open_science_metrics(self) -> OpenScienceMetrics:
        """Calculate aggregated open science adoption metrics."""
        total_papers = len(self.open_access_records)
        total_preprints = len(self.preprint_records)
        
        if total_papers == 0:
            return OpenScienceMetrics(
                total_preprints=0,
                preprint_adoption_rate=0.0,
                open_access_rate=0.0,
                data_sharing_compliance_rate=0.0,
                average_preprint_citations=0.0,
                average_time_to_publication=0.0,
                server_usage_distribution={},
                oa_type_distribution={},
                data_sharing_distribution={}
            )
        
        # Calculate rates
        preprint_adoption_rate = total_preprints / max(1, total_papers)
        
        oa_papers = sum(1 for record in self.open_access_records.values() 
                       if record.access_type != OpenAccessType.CLOSED)
        open_access_rate = oa_papers / total_papers
        
        compliant_data_sharing = sum(1 for record in self.data_sharing_records.values()
                                   if record.compliance_score > 0.5)
        data_sharing_rate = compliant_data_sharing / max(1, len(self.data_sharing_records))
        
        # Calculate averages
        avg_preprint_citations = sum(record.citation_count for record in self.preprint_records.values()) / max(1, total_preprints)
        
        # Calculate time to publication for preprints that got published
        published_preprints = [record for record in self.preprint_records.values() if record.is_published and record.published_date]
        if published_preprints:
            time_to_pub = sum((record.published_date - record.submission_date).days for record in published_preprints)
            avg_time_to_publication = time_to_pub / len(published_preprints)
        else:
            avg_time_to_publication = 0.0
        
        # Distribution calculations
        server_distribution = {}
        for server in PreprintServer:
            server_distribution[server] = sum(1 for record in self.preprint_records.values() 
                                            if record.server == server)
        
        oa_distribution = {}
        for oa_type in OpenAccessType:
            oa_distribution[oa_type] = sum(1 for record in self.open_access_records.values()
                                         if record.access_type == oa_type)
        
        data_distribution = {}
        for sharing_level in DataSharingLevel:
            data_distribution[sharing_level] = sum(1 for record in self.data_sharing_records.values()
                                                 if record.sharing_level == sharing_level)
        
        return OpenScienceMetrics(
            total_preprints=total_preprints,
            preprint_adoption_rate=preprint_adoption_rate,
            open_access_rate=open_access_rate,
            data_sharing_compliance_rate=data_sharing_rate,
            average_preprint_citations=avg_preprint_citations,
            average_time_to_publication=avg_time_to_publication,
            server_usage_distribution=server_distribution,
            oa_type_distribution=oa_distribution,
            data_sharing_distribution=data_distribution
        )
    
    def get_researcher_open_science_score(self, researcher_id: str) -> float:
        """Calculate an overall open science score for a researcher."""
        profile = self.researcher_profiles.get(researcher_id)
        if not profile:
            return 0.0
        
        # Count researcher's contributions
        researcher_preprints = sum(1 for record in self.preprint_records.values() 
                                 if record.researcher_id == researcher_id)
        researcher_papers = sum(1 for paper_id, record in self.open_access_records.items()
                              if paper_id in [p.paper_id for p in self.preprint_records.values() 
                                            if p.researcher_id == researcher_id])
        researcher_data_sharing = sum(1 for paper_id, record in self.data_sharing_records.items()
                                    if paper_id in [p.paper_id for p in self.preprint_records.values()
                                                  if p.researcher_id == researcher_id])
        
        # Calculate weighted score
        preprint_score = min(1.0, researcher_preprints / max(1, researcher_papers)) * 0.3
        oa_score = profile.open_access_preference * 0.4
        data_score = profile.data_sharing_willingness * 0.3
        
        return preprint_score + oa_score + data_score
    
    def simulate_policy_impact(self, policy_type: str, strength: float) -> Dict[str, float]:
        """Simulate the impact of open science policies."""
        if not (0.0 <= strength <= 1.0):
            raise ValidationError("strength", strength, "between 0.0 and 1.0")
        
        baseline_metrics = self.calculate_open_science_metrics()
        
        # Apply policy effects to researcher profiles
        for profile in self.researcher_profiles.values():
            if policy_type == "institutional_mandate":
                profile.preprint_adoption_rate = min(1.0, profile.preprint_adoption_rate * (1 + strength * 0.5))
                profile.open_access_preference = min(1.0, profile.open_access_preference * (1 + strength * 0.6))
                profile.data_sharing_willingness = min(1.0, profile.data_sharing_willingness * (1 + strength * 0.7))
            elif policy_type == "funder_mandate":
                profile.open_access_preference = min(1.0, profile.open_access_preference * (1 + strength * 0.8))
                profile.data_sharing_willingness = min(1.0, profile.data_sharing_willingness * (1 + strength * 0.9))
            elif policy_type == "apc_funding":
                profile.open_access_preference = min(1.0, profile.open_access_preference * (1 + strength * 0.4))
        
        # Calculate new metrics
        new_metrics = self.calculate_open_science_metrics()
        
        return {
            "preprint_adoption_change": new_metrics.preprint_adoption_rate - baseline_metrics.preprint_adoption_rate,
            "open_access_change": new_metrics.open_access_rate - baseline_metrics.open_access_rate,
            "data_sharing_change": new_metrics.data_sharing_compliance_rate - baseline_metrics.data_sharing_compliance_rate
        }
    
    def _load_data(self):
        """Load existing open science data from files."""
        try:
            # Load preprint records
            preprint_file = self.data_dir / "preprint_records.json"
            if preprint_file.exists():
                with open(preprint_file, 'r') as f:
                    data = json.load(f)
                    for record_data in data:
                        # Convert string back to enum
                        record_data['server'] = PreprintServer(record_data['server'])
                        record = PreprintRecord(**record_data)
                        record.submission_date = datetime.strptime(record_data['submission_date'], '%Y-%m-%d').date()
                        if record_data.get('published_date'):
                            record.published_date = datetime.strptime(record_data['published_date'], '%Y-%m-%d').date()
                        self.preprint_records[record.preprint_id] = record
            
            # Load open access records
            oa_file = self.data_dir / "open_access_records.json"
            if oa_file.exists():
                with open(oa_file, 'r') as f:
                    data = json.load(f)
                    for record_data in data:
                        # Convert string back to enum
                        record_data['access_type'] = OpenAccessType(record_data['access_type'])
                        record = OpenAccessRecord(**record_data)
                        record.publication_date = datetime.strptime(record_data['publication_date'], '%Y-%m-%d').date()
                        self.open_access_records[record.paper_id] = record
            
            # Load data sharing records
            ds_file = self.data_dir / "data_sharing_records.json"
            if ds_file.exists():
                with open(ds_file, 'r') as f:
                    data = json.load(f)
                    for record_data in data:
                        # Convert string back to enum
                        record_data['sharing_level'] = DataSharingLevel(record_data['sharing_level'])
                        record = DataSharingRecord(**record_data)
                        self.data_sharing_records[record.paper_id] = record
            
            # Load researcher profiles
            profiles_file = self.data_dir / "researcher_profiles.json"
            if profiles_file.exists():
                with open(profiles_file, 'r') as f:
                    data = json.load(f)
                    for profile_data in data:
                        # Convert strings back to enums
                        if profile_data.get('preferred_preprint_server'):
                            profile_data['preferred_preprint_server'] = PreprintServer(profile_data['preferred_preprint_server'])
                        if profile_data.get('preferred_oa_type'):
                            profile_data['preferred_oa_type'] = OpenAccessType(profile_data['preferred_oa_type'])
                        profile = OpenScienceProfile(**profile_data)
                        self.researcher_profiles[profile.researcher_id] = profile
                        
        except Exception as e:
            # If loading fails, start with empty data
            pass
    
    def save_data(self):
        """Save open science data to files."""
        try:
            # Save preprint records
            preprint_data = []
            for record in self.preprint_records.values():
                record_dict = asdict(record)
                record_dict['submission_date'] = record.submission_date.isoformat()
                if record.published_date:
                    record_dict['published_date'] = record.published_date.isoformat()
                # Convert enum to string
                record_dict['server'] = record.server.value
                preprint_data.append(record_dict)
            
            with open(self.data_dir / "preprint_records.json", 'w') as f:
                json.dump(preprint_data, f, indent=2)
            
            # Save open access records
            oa_data = []
            for record in self.open_access_records.values():
                record_dict = asdict(record)
                record_dict['publication_date'] = record.publication_date.isoformat()
                # Convert enum to string
                record_dict['access_type'] = record.access_type.value
                oa_data.append(record_dict)
            
            with open(self.data_dir / "open_access_records.json", 'w') as f:
                json.dump(oa_data, f, indent=2)
            
            # Save data sharing records
            ds_data = []
            for record in self.data_sharing_records.values():
                record_dict = asdict(record)
                # Convert enum to string
                record_dict['sharing_level'] = record.sharing_level.value
                ds_data.append(record_dict)
            
            with open(self.data_dir / "data_sharing_records.json", 'w') as f:
                json.dump(ds_data, f, indent=2)
            
            # Save researcher profiles
            profile_data = []
            for profile in self.researcher_profiles.values():
                profile_dict = asdict(profile)
                # Convert enums to strings
                if profile.preferred_preprint_server:
                    profile_dict['preferred_preprint_server'] = profile.preferred_preprint_server.value
                if profile.preferred_oa_type:
                    profile_dict['preferred_oa_type'] = profile.preferred_oa_type.value
                profile_data.append(profile_dict)
            
            with open(self.data_dir / "researcher_profiles.json", 'w') as f:
                json.dump(profile_data, f, indent=2)
                
        except Exception as e:
            raise PeerReviewError(f"Failed to save open science data: {str(e)}")