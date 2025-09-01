# PeerRead Dataset Structure Reference

## Base Path
- **PeerRead Dataset Location**: `../PeerRead/`
- **Current Workspace**: `./` (ai-peer-review-platform)

## Directory Structure
```
../PeerRead/
├── data/
│   ├── acl_2017/
│   │   ├── train/
│   │   │   ├── reviews/
│   │   │   │   ├── *.json (review files)
│   │   │   └── parsed_pdfs/
│   │   │       └── *.pdf.json (paper content files)
│   │   ├── dev/
│   │   │   ├── reviews/
│   │   │   └── parsed_pdfs/
│   │   └── test/
│   │       ├── reviews/
│   │       └── parsed_pdfs/
│   ├── conll_2016/
│   │   ├── train/
│   │   ├── dev/
│   │   └── test/
│   ├── iclr_2017/
│   │   ├── train/
│   │   ├── dev/
│   │   └── test/
│   ├── nips_2013/
│   ├── nips_2014/
│   ├── nips_2015/
│   ├── nips_2016/
│   ├── nips_2017/
│   ├── arxiv.cs.ai_2007-2017/
│   ├── arxiv.cs.cl_2007-2017/
│   └── arxiv.cs.lg_2007-2017/
└── README.md

## Current Dataset Status
- **Local papers**: `./dataset/papers/` (13,546 files)
- **Available in PeerRead**: `../PeerRead/data/` (25,155 files total)
- **Missing papers**: ~11,609 files not yet loaded
- **Organized papers**: `./dataset/organized/` (11,981 files processed)
- **Venues data**: `./data/venues/` (4 venue JSON files)

## Organized Dataset Breakdown
- **Artificial Intelligence**: 4,392 papers
- **Natural Language Processing**: 2,529 papers  
- **Computer Vision**: 989 papers
- **Unknown**: 4,071 papers
- **Total organized**: 11,981 papers

## PeerRead Venue Breakdown
- **acl_2017**: 274 files
- **arxiv.cs.ai_2007-2017**: 8,184 files  
- **arxiv.cs.cl_2007-2017**: 5,276 files
- **arxiv.cs.lg_2007-2017**: 10,096 files
- **conll_2016**: 44 files
- **iclr_2017**: 1,281 files
- **nips_2013-2017**: 0 files (empty directory)

## Access Patterns
- Use `../PeerRead/data/` for accessing source data
- Use shell commands for operations outside workspace
- Use file tools for operations within workspace

## Known Issues
- Organization script fails with directory creation errors
- Need to fix path handling for Windows environment
- Need to process remaining 11,035 unorganized papers