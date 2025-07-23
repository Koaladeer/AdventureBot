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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# 1️⃣ Load dataset
with open('data/data_zork1.json', 'r') as f:
    data = json.load(f)

# 2️⃣ Extract state-action pairs
inputs, targets = [], []
for sample in data:
    state = sample['state']
    loc_desc = state['loc_desc']
    inv_desc = state['inv_desc']
    surr_objs = ' '.join([surr for surr in state['surrounding_objs'].keys()])
    score = state['score']

    # Enriched state encoding
    input_text = f"Location: {loc_desc}. Surroundings: {surr_objs}. Score: {score}."

    inputs.append(input_text)
    targets.append(sample['action'])

# Balance dataset (downsample frequent actions)
balanced_inputs, balanced_targets = [], []
class_counts = Counter()
action_counter = Counter(targets)

normalization_dict = {'N': 'north','S': 'south','E': 'east','W': 'west','D': 'down'}
targets = [normalization_dict.get(tgt.strip(), tgt.strip().lower()) for tgt in targets]

for inp, tgt in zip(inputs, targets):
    if tgt in ['north','south','east','west','up','down']:
        max_count = 35  # downsampling of movement actions
    else:
        max_count = 300  # allow  more data for other actions

    if class_counts[tgt] < max_count:
        balanced_inputs.append(inp)
        balanced_targets.append(tgt)
        class_counts[tgt] += 1

print("Top 10 most frequent actions:")
for action, count in action_counter.most_common(15):
    print(f"{action}: {count}")
print("target size:", len(targets))
print("Balanced target size:", len(balanced_targets))
for action, count in Counter(balanced_targets).most_common(15):
    print(f"{action}: {count}")
print("input size:", len(inputs))
print("Balanced input size:", len(balanced_inputs))
action_counter = Counter(targets)
print("Total unique actions:", len(action_counter))

# Compute original and balanced distributions
original_targets = Counter(targets)
balanced_counter = Counter(balanced_targets)

# Get top N most common actions (based on original data)
top_n = 30
top_actions = [a for a, _ in original_targets.most_common(top_n)]

# Get counts from both
orig_counts = [original_targets[a] for a in top_actions]
input_counts = [Counter(balanced_inputs)[a] for a in top_actions]
bal_counts = [balanced_counter[a] for a in top_actions]

# Set bar width and positions
x = np.arange(len(top_actions))
bar_width = 0.4

# Plot
plt.figure(figsize=(14, 6))
plt.bar(x - bar_width/2, orig_counts, width=bar_width, label='Original', alpha=0.7)
plt.bar(x + bar_width/2, bal_counts, width=bar_width, label='Balanced', alpha=0.7)

plt.xticks(x, top_actions, rotation=90)
plt.xlabel("Actions")
plt.ylabel("Frequency")
plt.title("Action Distribution: Original vs Balanced (Top 30)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/action_distribution_comparison.png")
print("Saved: plots/action_distribution_comparison.png")

##

action_location_pairs = []
for sample in data:
    state = sample.get("state", {})
    location = state.get("location", {}).get("name", "Unknown")
    action = sample.get("action", "").strip().lower()
    if action and location:
        action_location_pairs.append((action, location))

# Build DataFrame
df = pd.DataFrame(action_location_pairs, columns=["Action", "Location"])
targets = [normalization_dict.get(tgt.strip(), tgt.strip().lower()) for tgt in targets]

# Create a pivot table: count how often each action occurs at each location
pivot = df.pivot_table(index="Action", columns="Location", aggfunc="size", fill_value=0)

# Filter to top N actions/locations (optional)

top_actions = df['Action'].value_counts().nlargest(80).index
top_locations = df['Location'].value_counts().nlargest(100).index
pivot_filtered = pivot.loc[pivot.index.isin(top_actions), pivot.columns.isin(top_locations)]

# Plot heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(pivot_filtered, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Actions vs. Locations")
plt.xlabel("Location")
plt.ylabel("Action")
plt.tight_layout()
plt.savefig("plots/action_actionsVSLoc.png")
def plot_normalized_action_location_heatmap(
    json_path,
    top_n_actions=20,
    top_n_locations=10,
    save_path=None
):
    """
    Plots a heatmap of normalized actions vs. locations from a Zork-like dataset.

    Args:
        json_path (str): Path to the dataset JSON file.
        top_n_actions (int): Number of most frequent actions to include.
        top_n_locations (int): Number of most common locations to include.
        save_path (str or None): Path to save the plot as PNG (optional).
    """

    # Abbreviation normalization
    normalization_dict = {'N': 'north', 'S': 'south', 'E': 'east', 'W': 'west', 'D': 'down'}

    # Load dataset
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Collect action-location pairs
    action_location_pairs = []
    for sample in data:
        state = sample.get("state", {})
        location = state.get("location", {}).get("name", "Unknown")
        raw_action = sample.get("action", "").strip()
        action = normalization_dict.get(raw_action, raw_action).lower()

        if action and location:
            action_location_pairs.append((action, location))

    # Create DataFrame
    df = pd.DataFrame(action_location_pairs, columns=["Action", "Location"])

    # Pivot table for heatmap
    pivot = df.pivot_table(index="Action", columns="Location", aggfunc="size", fill_value=0)

    # Filter top actions and locations
    top_actions = df['Action'].value_counts().nlargest(top_n_actions).index
    top_locations = df['Location'].value_counts().nlargest(top_n_locations).index
    pivot_filtered = pivot.loc[pivot.index.isin(top_actions), pivot.columns.isin(top_locations)]

    # Plot heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_filtered, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Normalized Actions vs. Locations")
    plt.xlabel("Location")
    plt.ylabel("Action")
    plt.tight_layout()

    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved heatmap to {save_path}")

    plt.show()

# Example usage
plot_normalized_action_location_heatmap(
    json_path="data/data_zork1.json",
    top_n_actions=20,
    top_n_locations=10,
    save_path="action_location_heatmap.png"
)