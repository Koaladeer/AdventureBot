import torch
import joblib
from transformers import BertTokenizer, BertForSequenceClassification
from jericho import FrotzEnv
import os

from JerichoWorld import defines_updated

# Load everything
model_path = 'models/bert_zork_modelV04'

print("Loading model...")
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
le = joblib.load(os.path.join(model_path, 'label_encoder.joblib'))

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load Jericho environment
rom_path = 'games/z-machine-games-master/jericho-game-suite/zork1.z5'
bindings = defines_updated.BINDINGS_DICT['zork1']
env = FrotzEnv(rom_path, seed=bindings['seed'])
obs = env.reset()[0]

state = env.get_state()
done = False

for step in range(20):
    if done:
        break

    # Get state descriptions
    loc_desc = env.step('look')[0]
    env.set_state(state)
    location = env.get_player_location()
    valid_actions = env.get_valid_actions()
    score = env.get_score()


    surr_objs = env._identify_interactive_objects(use_object_tree=True)
    surr_obj_names = ' '.join([obj[0] for obj in surr_objs])

    # Build input string
    input_text = f"Observation: {obs}.Location: {loc_desc}. Surroundings: {surr_obj_names}. Score: {score}. Actions:{valid_actions}"
    # Tokenize and predict
    encoding = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    #print(env.get_valid_actions())
    print(input_text)
    predicted_action = le.inverse_transform([pred])[0]
    print(f"Step {step+1}: Predicted action -> {predicted_action}")
    #new not testet like this
    prefixes = ["open a ", "close a ", "take a ", "exit a "]
    for prefix in prefixes:
        if predicted_action.startswith(prefix):
            # Extract the verb part of the string
            # We add len("a ") to ensure we get the word after "a "
            verb_to_find = predicted_action.split(' ', 1)
            for action in valid_actions:
                if verb_to_find.lower() in action.lower():
                    predicted_action = action
    print(f"Step {step+1}: NEW Predicted action -> {predicted_action}")

    # Take action
    obs, reward, done, info = env.step(predicted_action)
    print(f"Obs: {obs}\n Reward: {reward}\n")

    state = env.get_state()

env.close()




