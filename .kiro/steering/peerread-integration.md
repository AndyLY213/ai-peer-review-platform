---
inclusion: always
---

# PeerRead Dataset Integration

## Overview
The PeerRead dataset provides real-world peer review data that should be leveraged to enhance the realism and validation of the AI peer review simulation platform. This dataset contains over 14K paper drafts with accept/reject decisions and over 10K textual peer reviews from top-tier venues including ACL, NIPS, and ICLR.

## Dataset Structure
The PeerRead dataset is located at `../PeerRead/` relative to this project and contains:

- **Venues**: ACL 2017, NIPS 2013-2017, ICLR 2017, CoNLL 2016, and ArXiv categories
- **Data splits**: train/dev/test for each venue
- **Content**: Papers (PDFs and parsed text), reviews (JSON format), and acceptance decisions

## Real Review Data Format
Reviews in PeerRead follow this JSON structure:
```json
{
  "reviews": [
    {
      "IMPACT": "3",
      "SUBSTANCE": "4", 
      "APPROPRIATENESS": "5",
      "MEANINGFUL_COMPARISON": "2",
      "SOUNDNESS_CORRECTNESS": "4",
      "ORIGINALITY": "3",
      "CLARITY": "3",
      "REVIEWER_CONFIDENCE": "3",
      "RECOMMENDATION": "3",
      "PRESENTATION_FORMAT": "Poster",
      "comments": "Detailed review text...",
      "is_meta_review": null
    }
  ],
  "abstract": "Paper abstract...",
  "title": "Paper title...",
  "id": "paper_id"
}
```

## Integration Guidelines

### 1. Review Criteria Mapping
Map PeerRead review dimensions to enhanced review system:
- **IMPACT** → significance (1-10 scale)
- **SUBSTANCE** → technical quality  
- **SOUNDNESS_CORRECTNESS** → technical quality
- **ORIGINALITY** → novelty
- **CLARITY** → clarity
- **MEANINGFUL_COMPARISON** → related work
- **APPROPRIATENESS** → overall fit

### 2. Venue Characteristics
Use real venue data to calibrate venue system:
- **ACL**: Top NLP conference (~25% acceptance rate)
- **NIPS**: Top ML conference (~20% acceptance rate) 
- **ICLR**: Top DL conference (~30% acceptance rate)
- **CoNLL**: Mid-tier NLP conference (~35% acceptance rate)

### 3. Review Quality Patterns
Analyze real reviews for:
- Average review length by venue
- Common review structures and language patterns
- Score distributions and correlations
- Reviewer confidence patterns

### 4. Bias Detection Training Data
Use PeerRead data to:
- Train bias detection models on real review patterns
- Validate cognitive bias implementations against actual data
- Calibrate bias strength parameters based on observed patterns

### 5. Validation and Benchmarking
- Compare simulation outputs against PeerRead statistics
- Use real review text as templates for generated reviews
- Validate reviewer behavior models against actual reviewer patterns
- Benchmark acceptance rate predictions against real outcomes

## Implementation Notes

### Data Loading
Create utilities to:
- Parse PeerRead JSON review files
- Extract paper metadata and abstracts
- Map venue names to internal venue system
- Handle missing or incomplete review data

### Statistical Analysis
Implement analysis functions to:
- Calculate venue-specific review statistics
- Analyze score distributions and correlations
- Identify common review patterns and phrases
- Extract reviewer behavior signatures

### Model Training
Use PeerRead data for:
- Training review quality classifiers
- Learning venue-specific language patterns
- Calibrating bias detection algorithms
- Validating simulation realism

## File References
- Dataset location: `#[[file:../PeerRead/README.md]]`
- Sample review structure: `#[[file:../PeerRead/data/acl_2017/train/reviews/104.json]]`
- Venue acceptance rates: `#[[file:../PeerRead/data/acl_accepted.txt]]`