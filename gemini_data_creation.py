import json
from build_dataset import load_attributes, build_dataset_walkthrough, build_dataset
def main():
    """
    Main function to load attributes, build datasets from Zork1,
    and save them to JSON files.
    """
    # Load the attribute tables
    load_attributes()
    print("Attributes loaded.")

    # Path to your Zork1 rom
    game_path = 'games/z-machine-games-master/jericho-game-suite/zork1.z5'

    # Build dataset from walkthrough
    print('Starting walkthrough data generation...')
    data_walkthrough = build_dataset_walkthrough(game_path)
    print('Walkthrough data generation complete.')

    # Build dataset from random exploration
    print('Starting random exploration data generation...')
    data_random = build_dataset(game_path)
    print('Random exploration data generation complete.')
    # Save only random data data
    rnddata_file_path = '../data/rnd_data_zork1_v2.json'
    with open(rnddata_file_path, 'w') as f:
        json.dump(data_walkthrough, f, indent=4)  # Added indent for readability
    print(f'Random dataset saved successfully to {rnddata_file_path} with {len(data_walkthrough)} examples.')

    # Save only walkthrough data
    walkthrough_file_path = 'data/walkthrough_zork1_v2.json'
    with open(walkthrough_file_path, 'w') as f:
        json.dump(data_walkthrough, f, indent=4) # Added indent for readability
    print(f'Walkthrough dataset saved successfully to {walkthrough_file_path} with {len(walkthrough_file_path)} examples.')

    # Combine both datasets
    data = data_walkthrough + data_random

    # Save combined dataset to JSON file
    combined_file_path = 'data/data_zork1_v2.json'
    with open(combined_file_path, 'w') as f:
        json.dump(data, f, indent=4) # Added indent for readability
    print(f'Combined dataset saved successfully to {combined_file_path} with {len(data)} examples.')

if __name__ == '__main__':
    main()
