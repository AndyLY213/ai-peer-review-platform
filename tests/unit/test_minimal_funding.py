#!/usr/bin/env python3

from enum import Enum
from dataclasses import dataclass
from typing import Dict

class FundingAgencyType(Enum):
    NSF = "NSF"
    NIH = "NIH"
    INDUSTRY = "Industry"

@dataclass
class FundingAgency:
    name: str = ""
    agency_type: FundingAgencyType = FundingAgencyType.NSF

class FundingSystem:
    def __init__(self):
        self.agencies: Dict[str, FundingAgency] = {}
    
    def register_agency(self, agency: FundingAgency) -> str:
        agency_id = f"agency_{len(self.agencies)}"
        self.agencies[agency_id] = agency
        return agency_id

if __name__ == "__main__":
    print("Testing minimal funding system...")
    system = FundingSystem()
    agency = FundingAgency(name="Test NSF", agency_type=FundingAgencyType.NSF)
    agency_id = system.register_agency(agency)
    print(f"Created system with agency: {agency_id}")
    print("Minimal funding system works!")