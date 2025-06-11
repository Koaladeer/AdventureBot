import torch
import joblib
from transformers import BertTokenizer, BertForSequenceClassification
from jericho import FrotzEnv
import JerichoWorld.defines_updated  # your custom bindings
import os

# Load everything
model_path = 'models/bert_zork_model'

print("Loading model...")
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
le = joblib.load(os.path.join(model_path, 'label_encoder.joblib'))

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load Jericho environment
rom_path = 'games/z-machine-games-master/jericho-game-suite/zork1.z5'
bindings = defines_updated.BINDINGS_DICT['zork1.z5']
env = FrotzEnv(rom_path, seed=bindings['seed'])
obs = env.reset()[0]

state = env.get_state()
done = False

for step in range(100):
    if done:
        break

    # Get state descriptions
    loc_desc = env.step('look')[0]
    env.set_state(state)
    inv_desc = env.step('inventory')[0]
    env.set_state(state)

    location = env.get_player_location()
    score = env.get_score()

    inv_objs = env.identify_interactive_objects(use_object_tree=True, inventory=True)
    surr_objs = env.identify_interactive_objects(use_object_tree=True, inventory=False)

    inv_obj_names = ' '.join([obj[0] for obj in inv_objs])
    surr_obj_names = ' '.join([obj[0] for obj in surr_objs])

    # Build enriched input string
    input_text = f"Location: {loc_desc}. Inventory: {inv_desc}. InventoryObjs: {inv_obj_names}. Surroundings: {surr_obj_names}. Score: {score}."

    # Tokenize and predict
    encoding = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

    predicted_action = le.inverse_transform([pred])[0]
    print(f"Step {step+1}: Predicted action -> {predicted_action}")

    # Take action
    obs, reward, done, info = env.step(predicted_action)
    print(f"Obs: {obs}\nReward: {reward}\n")

    state = env.get_state()

env.close()
print("Game finished!")
