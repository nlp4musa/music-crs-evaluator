import os
import json
import random
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

def load_track_pools():
    """
    Load all available track IDs from the TalkPlayData track metadata dataset.
    Returns:
        list[str]: A list of all track IDs available in the dataset.
    """
    db = load_dataset("talkpl-ai/TalkPlayData-2-Track-Metadata", split="train")
    track_ids = list(db['track_id'])
    return track_ids

def main():
    """
    Generate random baseline predictions for music recommendation evaluation.
    This function loads the test dataset and generates random track recommendations
    for each session and turn number (1-8). For each turn, it randomly samples 20
    tracks from the available track pool. Results are saved to 'exp/random_sample.json'.
    The output format includes:
        - session_id: Unique identifier for the conversation session
        - user_id: Unique identifier for the user
        - turn_number: Turn number in the conversation (1-8)
        - predicted_track_ids: List of 20 randomly sampled track IDs
        - predicted_response: Empty string (no text response for this baseline)
    Returns:
        None: Results are written to 'exp/random_sample.json'.
    """
    track_pools = load_track_pools()
    db = load_dataset("talkpl-ai/TalkPlayData-2", split="test")
    inference_results = []
    for item in tqdm(db):
        user_id = item['user_id']
        session_id = item['session_id']
        for target_turn_number in range(1, 9):
            retrieval_track_ids = random.sample(track_pools, 20)
            inference_results.append({
                "session_id": session_id,
                "user_id": user_id,
                "turn_number": target_turn_number,
                "predicted_track_ids": retrieval_track_ids,
                "predicted_response": ""
            })
    os.makedirs("exp/inference", exist_ok=True)
    with open("exp/inference/random.json", "w", encoding="utf-8") as f:
        json.dump(inference_results, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
