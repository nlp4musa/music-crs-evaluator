# Music CRS Evaluator

Official evaluation framework for the **Conversational Music Recommendation System Challenge**.

This repository provides standardized tools to evaluate music recommendation systems on the [TalkPlayData-2](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2) dataset. Participants must follow the strict inference JSON format specified below to ensure their submissions can be properly evaluated.

## Overview

The evaluation framework:
- Loads predictions from standardized JSON format
- Computes retrieval metrics (Hit@k, MRR@k, nDCG@k, MAP@k, Recall@k, Precision@k)
- Evaluates across all 8 conversation turns
- Provides macro-averaged results across sessions and turns

## Setup

### Requirements
- Python 3.10+
- Dependencies: `datasets`, `pandas`, `numpy`, `scipy`, `tqdm`

### Installation

```bash
uv venv .venv --python=3.10
source .venv/bin/activate
uv pip install -r requirments.txt
```

Or using pip:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirments.txt
```

## Inference JSON Format

**⚠️ IMPORTANT:** Participants must strictly follow this JSON format for their predictions.

Your inference results must be saved as a JSON file in `exp/inference/<your_method_name>.json` with the following structure:

```json
[
  {
    "session_id": "69137__2020-02-08",
    "user_id": "69137",
    "turn_number": 1,
    "predicted_track_ids": [
      "60a0Rd6pjrkxjPbaKzXjfq",
      "2nLtzopw4rPReszdYBJU6h",
      "5UWwZ5lm5PKu6eKsHAGxOk",
      ...
    ],
    "predicted_response": ""
  },
  ...
]
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `string` | Unique identifier for the conversation session (format: `{user_id}__{date}`) |
| `user_id` | `string` | Unique identifier for the user |
| `turn_number` | `int` | Turn number in the conversation (1-8) |
| `predicted_track_ids` | `list[string]` | **Ordered list** of predicted Spotify track IDs (typically 20 tracks) |
| `predicted_response` | `string` | Text response (optional, can be empty string) |

### Important Notes

**One prediction per turn:** You must provide predictions for each session and turn combination in the test set
**Track IDs must be unique:** No duplicate track IDs within a single prediction
**Order matters:** Track IDs should be ranked by relevance (most relevant first)
**Use Spotify Track IDs:** Track IDs must match those in the [TalkPlayData-2-Track-Metadata](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2-Track-Metadata) dataset

## Quick Start

### 1. Generate Predictions

Create your inference file following the format above and save it to:
```
exp/inference/<your_method_name>.json
```

### 2. Run Evaluation

```bash
python eval_recsys.py --exp_name <your_method_name>
```

This will:
- Load your predictions from `exp/inference/<your_method_name>.json`
- Load ground truth from the TalkPlayData-2 test set
- Compute metrics for each session and turn
- Save macro-averaged results to `exp/eval_recsys/<your_method_name>.json`

### Example: Running Baseline

```bash
# Generate popularity baseline predictions
python lowerbound/popularity.py

# Evaluate the baseline
python eval_recsys.py --exp_name popularity
```

## Evaluation Metrics

The framework computes the following metrics at k={1, 10, 20}:

| Metric | Description |
|--------|-------------|
| **Hit@k** | Binary indicator if any relevant item appears in top-k |
| **MRR@k** | Mean Reciprocal Rank of the first relevant item in top-k |
| **nDCG@k** | Normalized Discounted Cumulative Gain at k |
| **MAP@k** | Mean Average Precision at k |
| **Recall@k** | Proportion of relevant items found in top-k |
| **Precision@k** | Proportion of top-k items that are relevant |

### Output Format

Results are saved as JSON with macro-averaged metrics:

```json
{
  "hit@1": 0.0005,
  "hit@10": 0.0035,
  "hit@20": 0.0060,
  "map@1": 0.0005,
  "map@10": 0.0013,
  "map@20": 0.0014,
  "mrr@1": 0.0,
  "mrr@10": 0.0,
  "mrr@20": 0.0,
  "ndcg@1": 0.0005,
  "ndcg@10": 0.0018,
  "ndcg@20": 0.0024,
  "precision@1": 0.0005,
  "precision@10": 0.0004,
  "precision@20": 0.0003,
  "recall@1": 0.0005,
  "recall@10": 0.0035,
  "recall@20": 0.0060
}
```

## Repository Structure

```
music-crs-evaluator/
├── readme.md              # This file
├── requirments.txt        # Python dependencies
├── eval_recsys.py         # Main evaluation script
├── metrics.py             # Metric computation functions
├── lowerbound/            # Baseline implementations
│   ├── popularity.py      # Popularity-based baseline
│   └── random_sample.py   # Random sampling baseline
└── exp/
    ├── inference/         # Place your prediction JSON files here
    │   └── <method>.json
    └── eval_recsys/       # Evaluation results saved here
        └── <method>.json
```

## Baseline Methods

Two baseline methods are provided for reference:


### Random Baseline
Recommends 20 randomly sampled tracks:
```bash
python lowerbound/random_sample.py
```

### Popularity Baseline
Recommends the 20 most popular tracks from the training set:
```bash
python lowerbound/popularity.py
```

## Dataset

This evaluation framework uses the **TalkPlayData-2** dataset:
- **Dataset:** [talkpl-ai/TalkPlayData-2](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2)
- **Track Metadata:** [talkpl-ai/TalkPlayData-2-Track-Metadata](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2-Track-Metadata)

The test set contains multi-turn conversations (8 turns each) where the system must recommend music based on conversational context.

## Validation Checklist

Before submitting your predictions, ensure:

- [ ] JSON file is saved in `exp/inference/<method_name>.json`
- [ ] All required fields are present (`session_id`, `user_id`, `turn_number`, `predicted_track_ids`, `predicted_response`)
- [ ] Predictions cover all sessions and turns (1-8) in the test set
- [ ] Track IDs are valid Spotify IDs from the dataset
- [ ] No duplicate track IDs within each prediction
- [ ] Track IDs are ordered by relevance
- [ ] JSON is properly formatted (use `json.dump()` with `ensure_ascii=False`)

## Troubleshooting

### Common Issues

**Error: "Predictions should be unique. Duplicates detected."**
- Ensure no duplicate track IDs in your `predicted_track_ids` list

**Error: "Gold item list should be unique. Duplicates detected."**
- This indicates an issue with the dataset/ground truth (contact organizers)

**Missing predictions:**
- Verify you have predictions for all sessions and turn numbers (1-8) in the test set
- Check that `session_id` and `turn_number` match exactly with the test set

## Contact

For questions or issues with the evaluation framework, please open an issue in this repository.
