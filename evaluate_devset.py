"""
Evaluation script for music recommendation systems.

This script evaluates recommendation system predictions against ground truth data
from the TalkPlayData-2 dataset, computing various metrics across conversation turns.
"""

import os
import json
from typing import List, Tuple, Dict, Any
from datasets import load_dataset, concatenate_datasets
from metrics import compute_recsys_metrics, compute_lexical_diversity, compute_catalog_diversity
from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Evaluate music recommendation system predictions")
parser.add_argument("--tid", type=str, default="llama1b_bm25_devset_all",
                    help="Name of the experiment (used to locate prediction files)")
parser.add_argument("--eval_dataset", type=str, default="devset")
args = parser.parse_args()


def df_filtering(df, session_id, turn_number):
    session_filter = df['session_id'] == session_id
    turn_number_filter = df['turn_number'] == turn_number
    return df[session_filter & turn_number_filter].iloc[0]

def main(args) -> None:
    """
    Main evaluation function.
    Loads predictions and ground truth data, computes metrics for each conversation turn,
    aggregates results, and saves the macro-averaged metrics to a JSON file.
    """
    results = []
    all_recommended_track_ids = set()
    all_response_words = set()

    if args.eval_dataset == "devset":
        eval_dataset_full_name = "talkpl-ai/TalkPlayData-Challenge-Dataset"
    elif args.eval_dataset == "blindset_A":
        eval_dataset_full_name = "talkpl-ai/TalkPlayData-Challenge-Blind-A"
    elif args.eval_dataset == "blindset_B":
        eval_dataset_full_name = "talkpl-ai/TalkPlayData-Challenge-Blind-B"

    lower_case_dataset_name = eval_dataset_full_name.lower().replace("-", "_")
    ground_truth = json.load(open(f"exp/ground_truth/{lower_case_dataset_name}/test.json", "r"))
    predictions = json.load(open(f"exp/inference/{args.eval_dataset}/{args.tid}.json", "r"))
    df_predictions = pd.DataFrame(predictions)
    df_ground_truth = pd.DataFrame(ground_truth)

    list_of_recommended_track_ids = []
    list_of_responses = []

    for index, row in tqdm(df_ground_truth.iterrows()):
        session_id = row['session_id']
        turn_number = row['turn_number']
        ground_truth_track_id = row['ground_truth_track_id']
        filtered_predictions = df_filtering(df_predictions, session_id, turn_number)
        filtered_ground_truth = df_filtering(df_ground_truth, session_id, turn_number)
        prediction_track_ids = filtered_predictions['predicted_track_ids']
        ground_truth_track_id = filtered_ground_truth['ground_truth_track_id']
        recsys_metrics = compute_recsys_metrics(prediction_track_ids, [ground_truth_track_id], [1, 10, 20])
        list_of_recommended_track_ids.extend(prediction_track_ids)
        list_of_responses.append(filtered_predictions['predicted_response'])
        results.append({
            "session_id": session_id,
            "turn_number": turn_number,
            **recsys_metrics
        })
    # Catalog coverage: unique recommended tracks / total catalog size
    catelog_split_type = "all_tracks"
    catalog = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split=catelog_split_type)
    total_catalog_size = len(catalog)
    catalog_diversity = compute_catalog_diversity(list_of_recommended_track_ids, total_catalog_size)
    lexical_diversity = compute_lexical_diversity(list_of_responses)


    df_results = pd.DataFrame(results)
    df_turn_wise_results = df_results.drop(columns=["session_id"]).groupby("turn_number").agg("mean")
    df_macro_results = df_turn_wise_results.mean(axis=0).to_dict()

    # # Append diversity metrics
    df_macro_results["catalog_diversity"] = catalog_diversity
    df_macro_results["lexical_diversity"] = lexical_diversity
    df_macro_results["total_catalog_size"] = total_catalog_size

    os.makedirs(f"exp/scores/{args.eval_dataset}", exist_ok=True)
    with open(f"exp/scores/{args.eval_dataset}/{args.tid}.json", "w") as f:
        json.dump(df_macro_results, f, indent=2)

if __name__ == "__main__":
    main(args)
