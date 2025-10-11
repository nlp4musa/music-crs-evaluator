"""
Evaluation script for music recommendation systems.

This script evaluates recommendation system predictions against ground truth data
from the TalkPlayData-2 dataset, computing various metrics across conversation turns.
"""

import os
import json
from typing import List, Tuple, Dict, Any
from datasets import load_dataset
from tqdm import tqdm
from metrics import compute_metrics
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Evaluate music recommendation system predictions")
parser.add_argument("--tid", type=str, default="random",
                    help="Name of the experiment (used to locate prediction files)")
args = parser.parse_args()


def parsing_groundtruth(conversations: List[Dict[str, Any]], target_turn_number: int) -> Tuple[str, str]:
    """
    Extract ground truth track ID and response from conversation data.
    Args:
        conversations: List of conversation dictionaries containing turn information
        target_turn_number: The specific turn number to extract data from
    Returns:
        Tuple containing:
            - recommend_music: The ground truth track ID
            - response: The ground truth response text
    """
    df_conversations = pd.DataFrame(conversations)
    df_current_turn = df_conversations[df_conversations['turn_number'] == target_turn_number]
    recommend_music = df_current_turn.iloc[1]['content']
    response = df_current_turn.iloc[2]['content']
    return recommend_music, response

def parsing_predictions(df_predictions: pd.DataFrame, session_id: str, turn_number: int) -> Tuple[List[str], str]:
    """
    Extract predicted track IDs and response for a specific session and turn.
    Args:
        predictions: List of prediction dictionaries
        session_id: The session identifier to filter by
        turn_number: The turn number to filter by
    Returns:
        Tuple containing:
            - recommend_track_ids: List of predicted track IDs
            - response: The predicted response text
    """
    df_filter = (df_predictions['session_id'] == session_id) & (df_predictions['turn_number'] == turn_number)
    recommend_track_ids = df_predictions[df_filter]['predicted_track_ids'].values[0]
    response = df_predictions[df_filter]['predicted_response'].values[0]
    return recommend_track_ids, response


def main() -> None:
    """
    Main evaluation function.

    Loads predictions and ground truth data, computes metrics for each conversation turn,
    aggregates results, and saves the macro-averaged metrics to a JSON file.
    """
    results = []
    predictions = json.load(open(f"exp/inference/{args.tid}.json", "r"))
    df_predictions = pd.DataFrame(predictions)
    db = load_dataset("talkpl-ai/TalkPlayData-2", split="test")
    for item in tqdm(db):
        for target_turn_number in range(1, 9):
            gt_track_id, gt_response = parsing_groundtruth(item['conversations'], target_turn_number)
            recommend_track_ids, predicted_response = parsing_predictions(df_predictions, item['session_id'], target_turn_number)
            metrics = compute_metrics(recommend_track_ids, [gt_track_id], [1, 10, 20])
            results.append({
                "session_id": item['session_id'],
                "turn_number": target_turn_number,
                **metrics
            })
    df_results = pd.DataFrame(results)
    df_turn_wise_results = df_results.drop(columns=["session_id"]).groupby("turn_number").agg("mean")
    df_macro_results = df_turn_wise_results.mean(axis=0).to_dict()
    os.makedirs(f"exp/eval_recsys", exist_ok=True)
    with open(f"exp/eval_recsys/{args.tid}.json", "w") as f:
        json.dump(df_macro_results, f, indent=2)

if __name__ == "__main__":
    main()
