---
inclusion: always
---

# Venue Calibration Using PeerRead Data

## Real Venue Characteristics
Use PeerRead dataset to calibrate venue system with authentic academic conference and journal characteristics.

## Venue Profiles from PeerRead

### ACL (Association for Computational Linguistics)
- **Type**: Top-tier NLP conference
- **Acceptance Rate**: ~25% (based on PeerRead data)
- **Review Standards**: High technical rigor, strong novelty requirements
- **Review Length**: 400-800 words average
- **Score Thresholds**: Accept ≥ 3.5/5, Borderline 2.5-3.5, Reject < 2.5
- **Reviewer Pool**: Senior NLP researchers, high h-index requirements
- **Timeline**: 8-week review period, 2-week rebuttal, 4-week final decision

### NIPS (Neural Information Processing Systems)
- **Type**: Top-tier ML conference  
- **Acceptance Rate**: ~20% (highly competitive)
- **Review Standards**: Exceptional technical contribution, broad impact
- **Review Length**: 500-900 words average
- **Score Thresholds**: Accept ≥ 4.0/5, Borderline 3.0-4.0, Reject < 3.0
- **Reviewer Pool**: ML experts, preference for established researchers
- **Timeline**: 6-week review period, 1-week rebuttal, 3-week final decision

### ICLR (International Conference on Learning Representations)
- **Type**: Top-tier deep learning conference
- **Acceptance Rate**: ~30% (competitive but growing)
- **Review Standards**: Novel architectures, strong empirical results
- **Review Length**: 450-750 words average
- **Score Thresholds**: Accept ≥ 3.5/5, Borderline 2.5-3.5, Reject < 2.5
- **Reviewer Pool**: Deep learning specialists, mix of academia and industry
- **Timeline**: 10-week review period, open review process

### CoNLL (Conference on Natural Language Learning)
- **Type**: Mid-tier specialized NLP conference
- **Acceptance Rate**: ~35% (more accessible than top venues)
- **Review Standards**: Solid contribution, clear methodology
- **Review Length**: 300-600 words average
- **Score Thresholds**: Accept ≥ 3.0/5, Borderline 2.0-3.0, Reject < 2.0
- **Reviewer Pool**: NLP researchers, broader seniority range
- **Timeline**: 6-week review period, optional rebuttal

## Calibration Parameters

### Review Quality Standards
```python
VENUE_STANDARDS = {
    "ACL": {
        "min_review_length": 400,
        "technical_depth_required": 0.8,
        "novelty_threshold": 0.7,
        "experimental_rigor": 0.9
    },
    "NIPS": {
        "min_review_length": 500,
        "technical_depth_required": 0.9,
        "novelty_threshold": 0.8,
        "experimental_rigor": 0.95
    },
    "ICLR": {
        "min_review_length": 450,
        "technical_depth_required": 0.8,
        "novelty_threshold": 0.7,
        "experimental_rigor": 0.85
    },
    "CoNLL": {
        "min_review_length": 300,
        "technical_depth_required": 0.6,
        "novelty_threshold": 0.5,
        "experimental_rigor": 0.7
    }
}
```

### Reviewer Assignment Criteria
```python
REVIEWER_REQUIREMENTS = {
    "ACL": {
        "min_h_index": 15,
        "min_years_experience": 5,
        "required_expertise_match": 0.8,
        "preferred_seniority": ["Associate Prof", "Full Prof"]
    },
    "NIPS": {
        "min_h_index": 20,
        "min_years_experience": 7,
        "required_expertise_match": 0.9,
        "preferred_seniority": ["Full Prof", "Senior Researcher"]
    },
    "ICLR": {
        "min_h_index": 12,
        "min_years_experience": 4,
        "required_expertise_match": 0.7,
        "preferred_seniority": ["Assistant Prof", "Associate Prof", "Industry"]
    },
    "CoNLL": {
        "min_h_index": 8,
        "min_years_experience": 3,
        "required_expertise_match": 0.6,
        "preferred_seniority": ["Postdoc", "Assistant Prof", "Associate Prof"]
    }
}
```

### Score Distribution Calibration
Based on PeerRead analysis:
```python
VENUE_SCORE_DISTRIBUTIONS = {
    "ACL": {
        "mean_scores": {"impact": 3.2, "substance": 3.4, "clarity": 3.3, "originality": 3.1},
        "std_scores": {"impact": 0.8, "substance": 0.7, "clarity": 0.8, "originality": 0.7},
        "acceptance_threshold": 3.5
    },
    "NIPS": {
        "mean_scores": {"impact": 3.0, "substance": 3.2, "clarity": 3.1, "originality": 2.9},
        "std_scores": {"impact": 0.9, "substance": 0.8, "clarity": 0.8, "originality": 0.8},
        "acceptance_threshold": 4.0
    }
    # ... similar for other venues
}
```

## Dynamic Venue Characteristics

### Temporal Evolution
- **Acceptance rates** change over time based on submission volume
- **Review standards** evolve with field maturity
- **Reviewer pool** expands with community growth
- **Technical requirements** increase with advancing state-of-the-art

### Field-Specific Variations
- **NLP venues**: Emphasis on linguistic analysis and evaluation
- **ML venues**: Focus on algorithmic novelty and theoretical analysis  
- **AI venues**: Broader scope, interdisciplinary considerations
- **Systems venues**: Implementation details and scalability concerns

### Prestige Dynamics
- **Reputation feedback loops**: Success breeds higher standards
- **Competition effects**: Venues compete for top submissions
- **Community perception**: Reviewer and author preferences evolve
- **Impact factor influence**: Citation-based prestige metrics

## Implementation Strategy

### Data-Driven Calibration
1. **Extract venue statistics** from PeerRead dataset
2. **Analyze review patterns** by venue and year
3. **Calculate acceptance thresholds** based on score distributions
4. **Identify reviewer characteristics** from review quality patterns

### Continuous Recalibration
1. **Monitor simulation accuracy** against real venue outcomes
2. **Update parameters** based on new PeerRead data releases
3. **Incorporate expert feedback** from academic community
4. **Adjust for temporal trends** in academic publishing

### Validation Metrics
- **Acceptance rate accuracy**: Within 5% of real venue rates
- **Score distribution similarity**: KL divergence < 0.1
- **Review quality matching**: BLEU score > 0.3 vs. real reviews
- **Decision consistency**: 85%+ agreement with real accept/reject patterns

## Quality Assurance

### Regular Audits
- Monthly comparison of simulation vs. real venue statistics
- Quarterly review of venue parameter accuracy
- Annual comprehensive calibration update
- Expert panel review of venue realism

### Feedback Integration
- Academic community input on venue accuracy
- Reviewer feedback on simulation realism
- Author experience validation
- Conference organizer consultation