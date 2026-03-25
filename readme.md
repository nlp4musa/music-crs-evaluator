# Music CRS Evaluator

Official evaluation framework for the **Conversational Music Recommendation System Challenge**.

This repository provides standardized tools to evaluate music recommendation systems on the **TalkPlay Data Challenge** datasets. Participants must follow the strict inference JSON format specified below to ensure their submissions can be properly evaluated.

**Challenge datasets:** [talkpl-ai/talkplay-data-challenge](https://huggingface.co/collections/talkpl-ai/talkplay-data-challenge)

## Overview

The evaluation framework:
- Loads predictions from standardized JSON format
- Computes retrieval metrics (nDCG@k, k={1,10,20})
- Evaluates across all 8 conversation turns
- Provides macro-averaged results across sessions and turns
- **Devset evaluation** additionally computes:
  - `catalog_diversity`: unique recommended tracks / total catalog size
  - `lexical_diversity`: unique word count across all predicted responses

- **Blindset evaluation** additionally computes:
  - `catalog_diversity`: unique recommended tracks / total catalog size
  - `lexical_diversity`: unique word count across all predicted responses
  - `llm_score`: LLM-as-a-judge quality score (1–5) via Gemini

## Setup

### Requirements
- Python 3.10+
- Dependencies: `datasets`, `pandas`, `numpy`, `scipy`, `tqdm`, `google-genai`
- For blind/dev evaluation: set `GEMINI_API_KEY` environment variable

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

Your inference results must be saved as a JSON file under `exp/inference/<eval_dataset>/<tid>.json` (e.g. `exp/inference/devset/llama1b_bert_devset.json`) with the following structure:

```json
[
  {
    "session_id": "69137__2020-02-08",
    "user_id": "69137",
    "turn_number": 1,
    "predicted_track_ids": [
      "715f8aff-7c99-46b8-8f9d-6d1aa1ae0372", "73562c63-02e3-4278-baf3-aeb3252f8b33", "4302b6cf-afe4-45d9-ab72-bd477086d838", "f20c5819-a312-4a6d-9ad1-46deccb4ff2f",
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
| `predicted_track_ids` | `list[string]` | **Ordered list** of predicted track unique identifiers (typically 20 tracks) |
| `predicted_response` | `string` | Text response (optional, can be empty string) |

### Important Notes

**One prediction per turn:** You must provide predictions for each session and turn combination in the test set
**Track IDs must be unique:** No duplicate track IDs within a single prediction
**Order matters:** Track IDs should be ranked by relevance (most relevant first)
**Use valid track IDs:** Track IDs must match those in [TalkPlayData-Challenge-Track-Metadata](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Track-Metadata)

## Quick Start

### 1. Generate Predictions

Create your inference file following the format above and save it to:
```
exp/inference/blindset_A/llama1b_bert_blindset_A_all.json
```

### 2. Run Evaluation

```bash
python evaluate_devset.py --eval_dataset blindset_A --tid llama1b_bert_blindset_A_all
```

This will:
- Load your predictions from `exp/inference/blindset_A/llama1b_bert_blindset_A_all.json`
- Load ground truth for the selected evaluation dataset
- Compute metrics for each session and turn
- Save macro-averaged results to `exp/scores/blindset_A/llama1b_bert_blindset_A_all.json`

### Example: Running Baseline

```bash
# Generate popularity baseline predictions
python lowerbound/popularity.py

# Evaluate the baseline
python eval_recsys.py --exp_name popularity
```

for more baselines, please refer to:
https://github.com/nlp4musa/music-crs-baselines


## Evaluation Metrics

### Retrieval Metrics (all splits)

The framework computes **Normalized Discounted Cumulative Gain (nDCG)** at k={1, 10, 20}.

**nDCG@k** measures ranking quality by comparing the predicted ranking against the ideal ranking:

$$
\text{nDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}
$$

where:

$$
\text{DCG@k} = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}
$$

- **rel_i**: Relevance score at position i (1 if track is in ground truth, 0 otherwise)
- **IDCG@k**: Ideal DCG@k (maximum possible DCG when items are perfectly ranked)

Higher nDCG values indicate better ranking quality, with 1.0 being perfect.

### Diversity Metrics (dev / blind splits)

| Metric | Description |
|--------|-------------|
| `catalog_diversity` | Unique recommended tracks ÷ total catalog size (0–1). Higher = broader coverage. |
| `lexical_diversity` | Count of unique words across all predicted responses. Higher = richer vocabulary. |


### LLM-as-a-Judge (blind split)

The blind evaluation includes an LLM-based quality assessment of generated responses. Detailed evaluation criteria and scoring methodology will be announced after the challenge concludes.

Requires `GEMINI_API_KEY` to be set in the environment.

## Repository Structure

```
music-crs-evaluator/
├── readme.md                    # This file
├── requirments.txt              # Python dependencies
├── eval_recsys.py               # Retrieval evaluation (public test set)
├── eval_devset.py               # Dev evaluation (retrieval + diversity)
├── eval_blindset.py             # Blind evaluation (retrieval + diversity + LLM judge)
├── metrics/
│   ├── __init__.py
│   ├── metrics_recsys.py        # nDCG and other retrieval metrics
│   ├── metrics_llm.py           # LLM-as-a-judge via Gemini
│   └── eval_prompt.txt          # Judge rubric (instruction following, personalization, …)
├── lowerbound/                  # Baseline implementations
│   ├── popularity.py            # Popularity-based baseline
│   └── random_sample.py         # Random sampling baseline
└── exp/
    ├── inference/               # Place your prediction JSON files here
    │   └── <method>.json
    ├── eval_recsys/             # Retrieval evaluation results
    │   └── <method>.json
    └── eval_blindset/           # Blind evaluation results
        └── <method>.json
```

## Baseline Methods

Two baseline methods are provided for reference:

## Dataset

All datasets are part of the [TalkPlay Data Challenge](https://huggingface.co/collections/talkpl-ai/talkplay-data-challenge) collection on Hugging Face.

| Dataset | Size | Description |
|---------|------|-------------|
| [TalkPlayData-Challenge-Dataset](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Dataset) | 1k sessions | Multi-turn music conversations with user profiles, conversation goals, and goal-progress assessments |
| [TalkPlayData-Challenge-Track-Metadata](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Track-Metadata) | 50.4k tracks | Track metadata: name, artist, album, tags, popularity, release date (splits: `train` / `test_warm` / `test_cold`) |
| [TalkPlayData-Challenge-User-Metadata](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-User-Metadata) | 9.09k users | User demographics: age, gender, country (splits: `train` / `test_warm` / `test_cold`) |
| [TalkPlayData-Challenge-Track-Embeddings](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Track-Embeddings) | 50.4k tracks | Pre-computed embeddings for all tracks |
| [TalkPlayData-Challenge-User-Embeddings](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-User-Embeddings) | 9.09k users | Pre-computed embeddings for all users |

The dataset contains multi-turn conversations (~8 turns each) where the system must recommend music based on conversational context, user listening history, and demographic profile.

## Validation Checklist

Before submitting your predictions, ensure:

- [ ] JSON file is saved in `exp/inference/blindset_A/llama1b_bert_blindset_A_all.json`
- [ ] All required fields are present (`session_id`, `user_id`, `turn_number`, `predicted_track_ids`, `predicted_response`)
- [ ] Predictions cover all sessions and turns (1-8) in the test set
- [ ] Track IDs are valid unique identifiers from the dataset
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
