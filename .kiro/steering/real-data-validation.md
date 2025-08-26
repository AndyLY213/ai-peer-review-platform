---
inclusion: always
---

# Real Data Validation Guidelines

## Validation Strategy
All simulation components should be validated against real peer review data from the PeerRead dataset to ensure realistic behavior and accurate modeling of academic peer review processes.

## Key Validation Areas

### 1. Review Score Distributions
- Compare simulated review scores against PeerRead score distributions
- Validate score correlations between different review dimensions
- Ensure venue-specific scoring patterns match real data
- Check for realistic score variance and reviewer agreement levels

### 2. Review Text Characteristics
- Validate review length distributions by venue type
- Compare vocabulary and language patterns in generated vs. real reviews
- Ensure appropriate technical depth and specificity
- Match sentiment and tone patterns from real reviews

### 3. Decision Patterns
- Validate acceptance rates against real venue statistics
- Compare decision consistency with review scores
- Check for realistic patterns in borderline decisions
- Ensure proper handling of reviewer disagreement cases

### 4. Temporal Patterns
- Validate review submission timing against real deadlines
- Compare reviewer workload patterns with actual data
- Check for realistic seasonal and conference cycle effects
- Ensure proper modeling of review delays and extensions

### 5. Bias Manifestations
- Validate bias effects against documented patterns in literature
- Compare prestige bias effects with real author reputation impacts
- Check confirmation bias patterns against field-specific preferences
- Ensure anchoring effects match observed sequential review patterns

## Validation Metrics

### Statistical Measures
- **Distribution Similarity**: KL divergence, Wasserstein distance for score distributions
- **Correlation Patterns**: Pearson/Spearman correlations between review dimensions
- **Text Similarity**: BLEU, ROUGE scores for review text quality
- **Decision Accuracy**: Precision/recall for accept/reject predictions

### Realism Indicators
- **Review Quality**: Completeness, coherence, and technical accuracy scores
- **Reviewer Behavior**: Consistency with established reviewer profiles
- **Venue Characteristics**: Alignment with known venue standards and practices
- **Temporal Realism**: Adherence to realistic timing constraints and patterns

## Implementation Requirements

### Data Preprocessing
```python
def load_peerread_data(venue_path: str) -> Dict:
    """Load and preprocess PeerRead data for validation"""
    # Parse JSON review files
    # Extract statistical patterns
    # Create validation benchmarks
    pass

def calculate_baseline_statistics(reviews: List[Dict]) -> ValidationMetrics:
    """Calculate baseline statistics from real data"""
    # Score distributions
    # Review length patterns  
    # Decision correlations
    pass
```

### Validation Functions
```python
def validate_simulation_realism(sim_results: SimulationResults, 
                              real_data: PeerReadData) -> ValidationReport:
    """Compare simulation results against real data patterns"""
    # Statistical comparisons
    # Distribution tests
    # Pattern matching
    pass

def continuous_validation_monitoring(simulation: PeerReviewSimulation):
    """Monitor simulation for drift from real patterns"""
    # Real-time validation checks
    # Alert on significant deviations
    # Automatic recalibration triggers
    pass
```

### Calibration Process
1. **Initial Calibration**: Use PeerRead training data to set baseline parameters
2. **Iterative Refinement**: Adjust parameters based on validation results
3. **Cross-Validation**: Test on held-out PeerRead test sets
4. **Expert Review**: Academic expert validation of simulation realism

## Quality Assurance

### Automated Checks
- Daily validation runs against PeerRead benchmarks
- Automated alerts for significant deviations from real patterns
- Continuous monitoring of key realism metrics
- Regular recalibration based on new data

### Manual Review
- Periodic expert review of generated reviews for realism
- Academic stakeholder feedback on simulation accuracy
- Comparison studies with other peer review simulation systems
- Publication of validation results for transparency

## Documentation Requirements
- Maintain detailed validation logs and results
- Document all calibration parameters and their sources
- Track validation metric trends over time
- Provide clear reporting on simulation limitations and accuracy bounds