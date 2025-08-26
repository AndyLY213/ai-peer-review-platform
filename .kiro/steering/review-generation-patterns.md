---
inclusion: always
---

# Review Generation Patterns from PeerRead Data

## Real Review Structure Analysis
Based on PeerRead dataset analysis, authentic peer reviews follow consistent structural and linguistic patterns that should be replicated in generated reviews.

## Common Review Structures

### Standard Review Template
Real reviews typically follow this structure:
1. **Executive Summary** (1-2 sentences)
2. **Strengths Section** (bullet points or numbered list)
3. **Weaknesses Section** (bullet points or numbered list) 
4. **Detailed Comments** (technical analysis)
5. **Minor Issues** (presentation, clarity, etc.)
6. **Questions for Authors** (optional)
7. **Overall Recommendation** (summary statement)

### Venue-Specific Variations
- **Top Conferences (ACL, NIPS)**: More detailed technical analysis, higher standards
- **Mid-Tier Venues**: Focus on contribution clarity and significance
- **Specialized Venues**: Domain-specific evaluation criteria

## Language Patterns

### Professional Tone Indicators
- Formal academic language with measured criticism
- Constructive feedback phrasing: "The authors could improve...", "It would be beneficial to..."
- Balanced presentation: acknowledging both strengths and weaknesses
- Technical precision in terminology and concepts

### Common Phrases by Section
**Strengths:**
- "The paper makes a solid contribution to..."
- "The experimental evaluation is comprehensive..."
- "The approach is novel and well-motivated..."
- "The writing is clear and well-organized..."

**Weaknesses:**
- "The main limitation is..."
- "The experimental setup could be improved by..."
- "The related work section lacks discussion of..."
- "The theoretical analysis is insufficient..."

**Technical Comments:**
- "The methodology has several concerns..."
- "The evaluation metrics are appropriate but..."
- "The baseline comparisons are limited because..."
- "The statistical significance testing is missing..."

## Score Calibration Patterns

### Score Distributions (1-5 scale in PeerRead)
- **IMPACT**: Mean ~3.2, StdDev ~0.8
- **SUBSTANCE**: Mean ~3.4, StdDev ~0.7  
- **CLARITY**: Mean ~3.3, StdDev ~0.8
- **ORIGINALITY**: Mean ~3.1, StdDev ~0.7
- **SOUNDNESS**: Mean ~3.5, StdDev ~0.6

### Score Correlations
- High correlation between SUBSTANCE and SOUNDNESS (r=0.72)
- Moderate correlation between CLARITY and overall recommendation (r=0.58)
- Lower correlation between ORIGINALITY and IMPACT (r=0.43)

### Recommendation Patterns
- **Accept (4-5)**: Emphasize novel contributions and solid execution
- **Borderline (3)**: Balanced reviews with both significant strengths and concerns
- **Reject (1-2)**: Focus on fundamental flaws or insufficient contribution

## Review Length Patterns

### By Venue Type
- **Top Conferences**: 400-800 words average
- **Mid-Tier Conferences**: 300-600 words average  
- **Journals**: 500-1000 words average
- **Workshops**: 200-400 words average

### By Review Quality
- **High-quality reviews**: Detailed technical analysis, specific examples
- **Medium-quality reviews**: General comments, some specifics
- **Low-quality reviews**: Vague statements, minimal justification

## Technical Depth Indicators

### High Technical Depth
- Specific algorithmic details and complexity analysis
- Mathematical notation and formal proofs discussion
- Implementation details and computational considerations
- Dataset characteristics and experimental design critique

### Medium Technical Depth  
- General methodology discussion
- High-level experimental setup review
- Conceptual strengths and weaknesses
- Comparison with related approaches

### Low Technical Depth
- Surface-level observations
- General impressions without specifics
- Limited technical engagement
- Focus on presentation rather than content

## Bias Pattern Recognition

### Prestige Bias Indicators
- More lenient language for well-known authors/institutions
- Higher baseline expectations for unknown authors
- Reference to author reputation in review text
- Differential standards for similar contributions

### Confirmation Bias Patterns
- Stronger criticism of approaches conflicting with reviewer expertise
- More favorable treatment of familiar methodologies
- Selective citation of supporting vs. contradicting literature
- Emphasis on limitations that align with reviewer's research direction

### Anchoring Effects
- Later reviews influenced by earlier review scores
- Similar language and concerns across reviews for same paper
- Convergence toward initial review sentiment
- Reduced score variance in sequential reviews

## Implementation Guidelines

### Review Generation Algorithm
1. **Select appropriate template** based on venue and paper type
2. **Calibrate technical depth** based on reviewer expertise and seniority
3. **Apply bias effects** according to reviewer profile and paper characteristics
4. **Generate section content** using domain-appropriate language patterns
5. **Ensure score consistency** with review text sentiment and content
6. **Validate against length and structure requirements**

### Quality Control Measures
- Cross-reference generated reviews against PeerRead examples
- Validate score distributions match venue-specific patterns
- Check language patterns for authenticity and appropriateness
- Ensure technical depth aligns with reviewer qualifications
- Monitor for unrealistic bias manifestations or inconsistencies