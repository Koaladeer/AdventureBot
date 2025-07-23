import torch
import joblib
from transformers import BertTokenizer, BertForSequenceClassification
from jericho import FrotzEnv
import os
import difflib  # For fuzzy matching in action resolution
import random  # Added for random action selection
from collections import Counter  # Added for counting actions
import matplotlib.pyplot as plt  # Added for plotting
import seaborn as sns  # Added for plotting
import pandas as pd  # Added for plotting
# Assuming JerichoWorld.defines_updated is available in the environment
# from JerichoWorld import defines_updated

# Define a normalization dictionary for common actions (re-used from data processing)
ACTION_NORMALIZATION_MAP = {
    'N': 'north', 'S': 'south', 'E': 'east', 'W': 'west', 'D': 'down', 'U': 'up'
}


def load_game_assets(model_path: str):
    """
    Loads the BERT model, tokenizer, and label encoder.

    Args:
        model_path (str): Path to the directory containing the saved model.

    Returns:
        tuple: (model, tokenizer, label_encoder, device)
    """
    print("Loading model assets...")
    try:
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        le = joblib.load(os.path.join(model_path, 'label_encoder.joblib'))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()  # Set model to evaluation mode
        print(f"Model loaded successfully on device: {device}")
        return model, tokenizer, le, device
    except Exception as e:
        print(f"Error loading model assets: {e}")
        exit()  # Exit if essential assets cannot be loaded


def initialize_environment(rom_path: str, bindings: dict):
    """
    Initializes the Jericho game environment.

    Args:
        rom_path (str): Path to the Z-machine game file.
        bindings (dict): Game-specific bindings (e.g., seed).

    Returns:
        FrotzEnv: The initialized Jericho environment.
    """
    print(f"Initializing Jericho environment: {rom_path}")
    try:
        env = FrotzEnv(rom_path, seed=bindings['seed'])
        # Reset environment to get initial observation
        initial_obs = env.reset()[0]
        print("Environment initialized.")
        return env, initial_obs
    except Exception as e:
        print(f"Error initializing environment: {e}")
        exit()  # Exit if environment cannot be initialized


def get_current_game_state_info(env: FrotzEnv, initial_obs: str):
    """
    Gathers comprehensive state information from the Jericho environment.

    Args:
        env (FrotzEnv): The current Jericho environment instance.
        initial_obs (str): The observation from the last env.step() or env.reset().

    Returns:
        dict: A dictionary containing various state descriptions.
    """
    # Jericho's step() returns the observation of the *next* state.
    # To get the current state's observation, we use the initial_obs from reset/step.
    # For loc_desc and inv_desc, we temporarily step and restore state.

    # Save current state to restore after temporary look/inventory commands
    current_env_state = env.get_state()

    # Get location description
    # Note: env.step('look') changes the state, so we need to restore it.
    loc_desc = env.step('look')[0]
    env.set_state(current_env_state)  # Restore state

    # Get inventory description
    inv_desc = env.step('inventory')[0]
    env.set_state(current_env_state)  # Restore state again

    location_obj = env.get_player_location()
    location_name = location_obj.name if location_obj else "Unknown Location"

    # Get valid actions. env.get_valid_actions() typically returns a dictionary
    # {internal_key: "human-readable-command"}. However, if it returns a list directly,
    # we need to handle that.
    raw_valid_actions = env.get_valid_actions()

    # Ensure valid_actions_list always contains the human-readable commands as a list
    if isinstance(raw_valid_actions, dict):
        valid_actions_list = list(raw_valid_actions.values())
    elif isinstance(raw_valid_actions, list):
        valid_actions_list = raw_valid_actions
    else:
        print(f"Warning: Unexpected type for valid actions: {type(raw_valid_actions)}. Treating as empty list.")
        valid_actions_list = []

    score = env.get_score()

    # Identify surrounding objects
    # Based on user feedback, env._identify_interactive_objects returns a dictionary:
    # {description_string: [(token, type, category), ...], ...}
    # Example: {'The small mailbox is closed.': [('small', 'ADJ', 'LOC'), ('mailbox', 'NOUN', 'LOC')]}
    raw_surr_objs_data = env._identify_interactive_objects(use_object_tree=True)

    # For debugging/clarity: print the raw output of _identify_interactive_objects
    print(f"Raw surr_objs_data from Jericho: {raw_surr_objs_data}")

    surr_obj_names_list = []
    surrounding_objs_dict = {}  # This will map canonical object names to their full descriptions or properties

    # Iterate through the items of the dictionary returned by _identify_interactive_objects
    for description_string, parsed_tokens_list in raw_surr_objs_data.items():
        # Try to find the main noun/proper noun as the object name
        object_name = None
        for token, tag, _ in parsed_tokens_list:
            # Assuming 'NOUN' or 'PROPN' (Proper Noun) identifies the main object
            if tag in ['NOUN', 'PROPN']:
                object_name = token.lower()  # Normalize to lowercase for consistency
                break  # Found the main object, no need to check other tokens in this list

        if object_name:
            # Add the canonical object name to the list for the input string
            surr_obj_names_list.append(object_name)
            # Store the full description string under the canonical object name
            # This allows resolve_action_to_game_command to check for object existence by its canonical name
            surrounding_objs_dict[object_name] = description_string  # Store the full descriptive string
        else:
            # If no NOUN/PROPN is found, we can optionally use the first token or the entire description
            # For now, we'll just print a warning if an object name couldn't be extracted
            print(
                f"Warning: Could not extract canonical object name from: {description_string} with tokens {parsed_tokens_list}")

    surr_obj_names = ' '.join(surr_obj_names_list)

    # For debugging/clarity: print the resulting surrounding_objs_dict
    print(f"Formatted surrounding_objs_dict: {surrounding_objs_dict}")

    return {
        'obs': initial_obs,  # The observation that led to this state
        'loc_desc': loc_desc,
        'inv_desc': inv_desc,
        'location_name': location_name,
        'surr_obj_names': surr_obj_names,
        'surrounding_objs_dict': surrounding_objs_dict,  # Pass the dict for resolution
        'valid_actions_dict': raw_valid_actions,  # Pass the raw dict for resolve_action_to_game_command
        'valid_actions_list': valid_actions_list,  # Pass the list for printing
        'score': score
    }


def format_input_string(state_info: dict, game_history: list) -> str:
    """
    Formats the gathered state information and game history into a single input string for the BERT model.

    Args:
        state_info (dict): Dictionary containing current state descriptions.
        game_history (list): A list of (observation, action) tuples from previous turns.

    Returns:
        str: The formatted input string.
    """
    history_str = ""
    if game_history:
        history_str = "Past Interactions:\n"
        for obs, action in game_history:
            history_str += f"  Obs: {obs.strip()} -> Action: {action}\n"
        history_str += "\n"  # Add a newline for separation

    current_state_str = (
        f"Current Observation: {state_info['obs'].strip()}.\n"
        f"Location: {state_info['loc_desc'].strip()}.\n"
        f"Inventory: {state_info['inv_desc'].strip()}.\n"
        f"Surroundings: {state_info['surr_obj_names']}.\n"
        f"Score: {state_info['score']}."
    )
    return history_str + current_state_str


def predict_action_intent(input_text: str, model, tokenizer, le, device) -> str:
    """
    Predicts the player's intended action using the BERT model.

    Args:
        input_text (str): The formatted input string representing the game state.
        model: The loaded BERT model.
        tokenizer: The loaded BERT tokenizer.
        le: The loaded LabelEncoder.
        device: The torch device (cpu/cuda).

    Returns:
        str: The predicted action intent (e.g., "open a", "Get egg").
    """
    encoding = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    predicted_action_intent = le.inverse_transform([pred])[0]
    return predicted_action_intent


def resolve_action_to_game_command(
        predicted_action: str,
        valid_acts_dict: dict,  # This should remain the dictionary from env.get_valid_actions()
        surrounding_objs_dict: dict,  # Can be used for more advanced disambiguation
        normalization_map: dict
) -> tuple[str, str]:  # Now returns a tuple: (command, resolution_type)
    """
    Resolves a predicted action (player's intent) to an actual executable game command
    from the current state's valid actions.

    Args:
        predicted_action (str): The action predicted by the model (e.g., "open a", "Get egg").
        valid_acts_dict (dict): The 'valid_acts' dictionary from the current game state.
                                Keys are internal game representations, values are human-readable commands.
        surrounding_objs_dict (dict): The 'surrounding_objs' dictionary from the current game state.
                                     (Can be used to infer objects for generic commands, e.g., "open mailbox")
        normalization_map (dict): The map used for normalizing actions (e.g., 'N' to 'north').

    Returns:
        tuple[str, str]: A tuple containing the best matching executable game command and its resolution type.
    """
    normalized_predicted_action = normalization_map.get(predicted_action.strip(), predicted_action.strip().lower())

    # Ensure available_game_commands is always a list of strings
    if isinstance(valid_acts_dict, dict):
        available_game_commands = list(valid_acts_dict.values())
    elif isinstance(valid_acts_dict, list):  # Handle case where it's already a list
        available_game_commands = valid_acts_dict
    else:
        print(f"Error: valid_acts_dict has unexpected type {type(valid_acts_dict)}. Cannot resolve commands.")
        return "look", "Error/Default"  # Fallback

    print(f"Attempting to resolve predicted action: '{predicted_action}' (normalized: '{normalized_predicted_action}')")
    print(f"Available game commands: {available_game_commands}")

    # 1. Direct match (after normalization)
    if normalized_predicted_action in available_game_commands:
        print(f"  Found direct match: '{normalized_predicted_action}'")
        return normalized_predicted_action, "Direct Match"

    # 2. More intelligent handling for generic commands like "open", "take", "get"
    # Split the predicted action into verb and potential object part
    parts = normalized_predicted_action.split(' ', 1)
    verb = parts[0]
    object_part = parts[1] if len(parts) > 1 else ""

    # Check if the predicted verb is an action that can take an object
    if verb in ['open', 'take', 'get', 'examine', 'look at', 'read', 'eat', 'drink', 'wear', 'drop', 'put']:
        # Try to find a specific command by combining the verb with known objects
        for obj_name in surrounding_objs_dict.keys():
            # Form a potential command (e.g., "open window")
            potential_cmd_specific = f"{verb} {obj_name}"
            if potential_cmd_specific in available_game_commands:
                print(
                    f"  Resolved to specific command: '{potential_cmd_specific}' (from predicted '{predicted_action}' and object '{obj_name}')")
                return potential_cmd_specific, "Inferred Object"

            # Also check for "verb a" if the object part is generic or empty
            if object_part == 'a' or object_part == '':
                potential_cmd_generic_a = f"{verb} a"
                if potential_cmd_generic_a in available_game_commands:
                    print(f"  Resolved to generic 'verb a' command: '{potential_cmd_generic_a}'")
                    return potential_cmd_generic_a, "Generic Verb"

                # Special case for "take all" if "take" is predicted and "all" is an object or common command
                if verb == 'take' and 'take all' in available_game_commands:
                    print(f"  Resolved to 'take all' command.")
                    return 'take all', "Generic Verb"  # Categorize as generic since it's a common multi-object command

        # If no specific object-based command found, but the generic "verb a" exists
        if object_part == 'a' and f"{verb} a" in available_game_commands:
            print(f"  Using generic 'verb a' command: '{verb} a'")
            return f"{verb} a", "Generic Verb"

    # 3. Fuzzy matching for close commands (as a general fallback)
    close_matches = difflib.get_close_matches(normalized_predicted_action, available_game_commands, n=1, cutoff=0.6)
    if close_matches:
        print(f"  Fuzzy matched '{predicted_action}' to '{close_matches[0]}'")
        return close_matches[0], "Fuzzy Match"

    # 4. Intelligent Fallback: If no direct, inferred, or fuzzy match, try to pick a different valid action
    if available_game_commands:
        # Prioritize movement actions to encourage exploration
        movement_actions = [
            cmd for cmd in available_game_commands
            if
            cmd in ['north', 'south', 'east', 'west', 'up', 'down', 'northeast', 'northwest', 'southeast', 'southwest']
        ]
        if movement_actions:
            chosen_fallback_action = random.choice(movement_actions)
            print(f"  Warning: No direct match. Falling back to a random movement action: '{chosen_fallback_action}'.")
            return chosen_fallback_action, "Fallback (Movement)"

        # If no movement actions, try other non-trivial actions (avoiding 'look' or 'inventory' if possible)
        non_trivial_actions = [
            cmd for cmd in available_game_commands
            if cmd not in ['look', 'inventory', 'wait', 'score']  # Add other common trivial actions if needed
        ]
        if non_trivial_actions:
            chosen_fallback_action = random.choice(non_trivial_actions)
            print(
                f"  Warning: No direct match. Falling back to a random non-trivial action: '{chosen_fallback_action}'.")
            return chosen_fallback_action, "Fallback (Non-Trivial)"

        # If only 'look', 'inventory', or other trivial actions remain, pick one randomly
        chosen_fallback_action = random.choice(available_game_commands)
        print(
            f"  Warning: No meaningful action found. Falling back to a random available action: '{chosen_fallback_action}'.")
        return chosen_fallback_action, "Fallback (Trivial)"

    # If no suitable match and no available commands, return a safe default command
    print(
        f"  Warning: No suitable game command found for predicted action '{predicted_action}' and no valid actions available. Defaulting to 'look'.")
    return "look", "Fallback (Default)"


def plot_agent_action_choices(action_data: list, save_path: str = None):
    """
    Counts and plots the distribution of actions chosen by the agent during a game run,
    with color differentiation based on how the action was resolved.

    Args:
        action_data (list): A list of tuples, where each tuple is (executed_command, resolution_type).
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    if not action_data:
        print("No actions were recorded to plot.")
        return

    # Create a DataFrame from the action_data list
    df = pd.DataFrame(action_data, columns=['Action', 'Resolution Type'])

    # Count frequencies for plotting
    # We need to count each (Action, Resolution Type) pair
    action_type_counts = df.groupby(['Action', 'Resolution Type']).size().reset_index(name='Frequency')

    # Sort by overall frequency for better visualization
    # First, get the total frequency of each action
    overall_action_frequency = df['Action'].value_counts().reset_index(name='TotalFrequency')
    overall_action_frequency.columns = ['Action', 'TotalFrequency']

    # Merge to sort the bars correctly
    action_type_counts = pd.merge(action_type_counts, overall_action_frequency, on='Action')
    action_type_counts = action_type_counts.sort_values(by=['TotalFrequency', 'Action'], ascending=[False, True])

    plt.figure(figsize=(14, 8))
    # Use 'hue' to differentiate by 'Resolution Type'
    sns.barplot(x='Action', y='Frequency', hue='Resolution Type', data=action_type_counts, palette='tab10', dodge=True)

    plt.xticks(rotation=90, ha='right')  # Rotate labels for readability
    plt.xlabel("Action Chosen")
    plt.ylabel("Number of Times Chosen")
    plt.title("Agent's Action Choices Distribution by Resolution Type")
    plt.legend(title="Resolution Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # Use bbox_inches to prevent legend cutoff
        print(f"Saved agent action choices plot to {save_path}")
    plt.show()


def run_game_loop(env: FrotzEnv, model, tokenizer, le, device, max_steps: int = 100, max_history_turns: int = 5):
    """
    Runs the Zork game loop, making predictions and taking actions.

    Args:
        env (FrotzEnv): The Jericho environment.
        model: The loaded BERT model.
        tokenizer: The loaded BERT tokenizer.
        le: The loaded LabelEncoder.
        device: The torch device (cpu/cuda).
        max_steps (int): Maximum number of steps to play.
        max_history_turns (int): The maximum number of past (observation, action) pairs to keep in history.
    """
    obs = env.reset()[0]  # Initial observation after reset
    game_state = env.get_state()
    done = False
    game_history = []  # To store (observation_from_prev_step, action_taken_in_prev_step)
    executed_actions_data = []  # To store (executed_command, resolution_type) for plotting

    print("\n--- Starting Game Loop ---")
    for step in range(max_steps):
        if done:
            print("Game finished (done flag set)!")
            break

        print(f"\n--- Step {step + 1} ---")

        # 1. Get current game state information
        # Pass the 'obs' from the *previous* step as the current observation
        state_info = get_current_game_state_info(env, obs)
        print(f"Current Location: {state_info['location_name']}")
        print(f"Current Score: {state_info['score']}")
        print(f"Surrounding Objects: {state_info['surr_obj_names']}")
        # Use 'valid_actions_list' for printing, which is guaranteed to be a list
        print(f"Valid Actions: {state_info['valid_actions_list']}")

        # 2. Format input for the model, including history
        input_text = format_input_string(state_info, game_history)

        # 3. Predict action intent
        predicted_action_intent = predict_action_intent(input_text, model, tokenizer, le, device)
        print(f"Model's Predicted Intent: '{predicted_action_intent}'")

        # 4. Resolve predicted intent to an executable game command and its resolution type
        executable_command, resolution_type = resolve_action_to_game_command(
            predicted_action_intent,
            state_info['valid_actions_dict'],
            state_info['surrounding_objs_dict'],
            ACTION_NORMALIZATION_MAP
        )
        print(f"Executing Command: '{executable_command}' (Resolved by: {resolution_type})")

        # Store the executed command and its resolution type for plotting
        executed_actions_data.append((executable_command, resolution_type))

        # 5. Take action in the environment
        new_obs, reward, done, info = env.step(executable_command)
        print(f"Observed: {new_obs.strip()}")
        print(f"Reward: {reward}")
        if info.get('won'):
            print("Game WON!")
        if info.get('lost'):
            print("Game LOST!")

        # Update history with the observation *before* this action and the action taken
        # The 'obs' variable holds the observation *before* the current step's action
        game_history.append((obs.strip(), executable_command))
        if len(game_history) > max_history_turns:
            game_history.pop(0)  # Remove the oldest entry

        # Set 'obs' for the *next* iteration to the new observation
        obs = new_obs

        # Update game state for next iteration
        game_state = env.get_state()

    env.close()
    print("\nGame loop concluded.")

    # Plot the agent's action choices after the game loop
    PLOT_DIR = 'plots/'  # Ensure this directory exists or create it
    os.makedirs(PLOT_DIR, exist_ok=True)  # Create plots directory if it doesn't exist
    plot_agent_action_choices(executed_actions_data, save_path=f"{PLOT_DIR}agent_action_choices.png")


# Main execution flow
if __name__ == "__main__":
    MODEL_PATH = 'models/bert_zork_modelV04'
    ROM_PATH = 'games/z-machine-games-master/jericho-game-suite/zork1.z5'

    # Placeholder for JerichoWorld.defines_updated.BINDINGS_DICT
    # In a real setup, ensure this import works or define bindings directly.
    # For demonstration, I'll use a dummy binding.
    try:
        from JerichoWorld import defines_updated

        BINDINGS = defines_updated.BINDINGS_DICT['zork1']
    except ImportError:
        print("Warning: Could not import JerichoWorld.defines_updated. Using dummy bindings.")
        BINDINGS = {'seed': 0}  # Dummy binding

    # 1. Load model assets
    model, tokenizer, le, device = load_game_assets(MODEL_PATH)

    # 2. Initialize environment
    env, initial_obs_after_reset = initialize_environment(ROM_PATH, BINDINGS)

    # 3. Run the game loop
    # You can adjust max_history_turns to control the length of the context
    run_game_loop(env, model, tokenizer, le, device, max_steps=100, max_history_turns=5)
