import json
import random
import joblib
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import seaborn as sns
import pandas as pd
# Define a normalization dictionary for common actions
ACTION_NORMALIZATION_MAP = {
    'N': 'north', 'S': 'south', 'E': 'east', 'W': 'west', 'D': 'down', 'U': 'up'
}

def load_json_data(filepath: str) -> list:
    """
    Loads data from a specified JSON file.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        list: The loaded data as a list of dictionaries.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return []

def extract_state_action_pairs(data: list) -> tuple[list, list]:
    """
    Extracts and enriches state descriptions and corresponding actions from the dataset.

    Args:
        data (list): The raw loaded dataset.

    Returns:
        tuple[list, list]: A tuple containing two lists:
                           - inputs (list of str): Enriched state descriptions.
                           - targets (list of str): Raw action targets.
    """
    inputs, targets = [], []
    for sample in data:
        state = sample.get('state', {})
        loc_desc = state.get('loc_desc', '')
        inv_desc = state.get('inv_desc', '') # Not used in input_text but good to extract
        surr_objs = ' '.join([surr for surr in state.get('surrounding_objs', {}).keys()])
        score = state.get('score', 0)

        # Enriched state encoding
        input_text = f"Location: {loc_desc}. Surroundings: {surr_objs}. Score: {score}."
        inputs.append(input_text)
        targets.append(sample.get('action', ''))
    print(f"Extracted {len(inputs)} state-action pairs.")
    return inputs, targets

def normalize_actions(actions: list, normalization_map: dict) -> list:
    """
    Normalizes action strings based on a provided mapping.

    Args:
        actions (list): A list of raw action strings.
        normalization_map (dict): A dictionary mapping raw actions to normalized forms.

    Returns:
        list: A list of normalized action strings.
    """
    normalized_actions = [
        normalization_map.get(action.strip(), action.strip().lower())
        for action in actions
    ]
    print("Actions normalized.")
    return normalized_actions

def balance_dataset_by_downsampling(
    inputs: list,
    targets: list,
    movement_max_count: int = 35,
    other_max_count: int = 300,
    movement_actions: list = ['north', 'south', 'east', 'west', 'up', 'down']
) -> tuple[list, list]:
    """
    Balances the dataset by downsampling frequent actions.

    Args:
        inputs (list): List of input texts.
        targets (list): List of target actions.
        movement_max_count (int): Maximum count for movement actions.
        other_max_count (int): Maximum count for other actions.
        movement_actions (list): List of actions considered as 'movement'.

    Returns:
        tuple[list, list]: A tuple containing the balanced inputs and targets.
    """
    balanced_inputs, balanced_targets = [], []
    class_counts = Counter()
    original_action_counter = Counter(targets)

    print("\nOriginal Action Distribution (Top 15):")
    for action, count in original_action_counter.most_common(15):
        print(f"  {action}: {count}")
    print(f"Total original actions: {len(targets)}")

    for inp, tgt in zip(inputs, targets):
        max_count = movement_max_count if tgt in movement_actions else other_max_count
        if class_counts[tgt] < max_count:
            balanced_inputs.append(inp)
            balanced_targets.append(tgt)
            class_counts[tgt] += 1

    print("\nBalanced Action Distribution (Top 15):")
    for action, count in Counter(balanced_targets).most_common(15):
        print(f"  {action}: {count}")
    print(f"Total balanced actions: {len(balanced_targets)}")
    print(f"Total unique actions after balancing: {len(Counter(balanced_targets))}")
    return balanced_inputs, balanced_targets

def plot_action_distribution_comparison(
    original_targets: list,
    balanced_targets: list,
    top_n: int = 30,
    save_path: str = None
):
    """
    Plots a comparison of original and balanced action distributions.

    Args:
        original_targets (list): List of original target actions.
        balanced_targets (list): List of balanced target actions.
        top_n (int): Number of top actions to display.
        save_path (str): Path to save the plot.
    """
    original_counts = Counter(original_targets)
    balanced_counts = Counter(balanced_targets)

    # Get top N most common actions based on original data
    top_actions = [a for a, _ in original_counts.most_common(top_n)]

    orig_freqs = [original_counts[a] for a in top_actions]
    bal_freqs = [balanced_counts[a] for a in top_actions]

    x = np.arange(len(top_actions))
    bar_width = 0.4

    plt.figure(figsize=(14, 6))
    plt.bar(x - bar_width/2, orig_freqs, width=bar_width, label='Original', alpha=0.7)
    plt.bar(x + bar_width/2, bal_freqs, width=bar_width, label='Balanced', alpha=0.7)

    plt.xticks(x, top_actions, rotation=90)
    plt.xlabel("Actions")
    plt.ylabel("Frequency")
    plt.title(f"Action Distribution: Original vs Balanced (Top {top_n})")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.show()

def plot_action_location_heatmap(
    data: list,
    normalization_map: dict,
    top_n_actions: int = 80,
    top_n_locations: int = 100,
    save_path: str = None
):
    """
    Plots a heatmap of actions vs. locations from the dataset.

    Args:
        data (list): The raw loaded dataset.
        normalization_map (dict): Dictionary for normalizing action names.
        top_n_actions (int): Number of most frequent actions to include.
        top_n_locations (int): Number of most common locations to include.
        save_path (str or None): Path to save the plot as PNG (optional).
    """
    action_location_pairs = []
    for sample in data:
        state = sample.get("state", {})
        location = state.get("location", {}).get("name", "Unknown")
        raw_action = sample.get("action", "").strip()
        action = normalization_map.get(raw_action, raw_action).lower()

        if action and location:
            action_location_pairs.append((action, location))

    df = pd.DataFrame(action_location_pairs, columns=["Action", "Location"])

    # Create a pivot table: count how often each action occurs at each location
    pivot = df.pivot_table(index="Action", columns="Location", aggfunc="size", fill_value=0)

    # Filter to top N actions/locations
    top_actions = df['Action'].value_counts().nlargest(top_n_actions).index
    top_locations = df['Location'].value_counts().nlargest(top_n_locations).index
    pivot_filtered = pivot.loc[pivot.index.isin(top_actions), pivot.columns.isin(top_locations)]

    # Plot heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_filtered, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Actions vs. Locations")
    plt.xlabel("Location")
    plt.ylabel("Action")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved heatmap to {save_path}")
    plt.show()

# Main execution flow
if __name__ == "__main__":
    DATA_PATH = 'data/data_zork1.json'
    PLOT_DIR = 'plots/' # Ensure this directory exists or create it

    # 1. Load dataset
    raw_data = load_json_data(DATA_PATH)
    if not raw_data:
        exit("Exiting: Could not load data.")

    # 2. Extract state-action pairs
    inputs, targets = extract_state_action_pairs(raw_data)

    # 3. Normalize actions
    normalized_targets = normalize_actions(targets, ACTION_NORMALIZATION_MAP)

    # 4. Balance dataset (downsample frequent actions)
    balanced_inputs, balanced_targets = balance_dataset_by_downsampling(
        inputs,
        normalized_targets,
        movement_max_count=35,
        other_max_count=300
    )

    # 5. Plot action distribution comparison
    plot_action_distribution_comparison(
        original_targets=normalized_targets,
        balanced_targets=balanced_targets,
        top_n=30,
        save_path=f"{PLOT_DIR}action_distribution_comparisonV2.png"
    )

    # 6. Plot action vs. location heatmap
    plot_action_location_heatmap(
        data=raw_data,
        normalization_map=ACTION_NORMALIZATION_MAP,
        top_n_actions=80,
        top_n_locations=100,
        save_path=f"{PLOT_DIR}action_actionsVSLocV2.png"
    )