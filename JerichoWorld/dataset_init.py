import json
from build_dataset import load_attributes, build_dataset_walkthrough, build_dataset

# Load the attribute tables
load_attributes()

# Path to your Zork1 rom
game_path = '../games/z-machine-games-master/jericho-game-suite/zork1.z5'
print('Walktrought Data Start')

# Build dataset from walkthrough
data_walkthrough = build_dataset_walkthrough(game_path)
print('Walktrought Data Done')
# Build dataset from random exploration
data_random = build_dataset(game_path)

# Combine both datasets
data = data_walkthrough + data_random

# Save to JSON file
with open('../data/data_zork1.json', 'w') as f:
    json.dump(data, f)

print(f'Dataset saved successfully with {len(data)} examples.')
