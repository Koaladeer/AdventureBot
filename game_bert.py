import torch
from transformers import BertTokenizer, BertForSequenceClassification
from jericho import FrotzEnv
import joblib
import os

# 1️⃣ Load your trained model
print("loading model...")
model_path = 'models/bert_zork_modelV03'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()
print("loading game")
# 2️⃣ Load bindings and environment
game_path = 'games/z-machine-games-master/jericho-game-suite/zork1.z5'
env = FrotzEnv(game_path)
obs = env.reset()[0]

print("loading labelencoder")
# 3️⃣ loading LabelEncoder from model
le = joblib.load(os.path.join(model_path, 'label_encoder.joblib'))  # you must save this during training

#context input methode
context_window = []

def build_input(state, action=None):
    loc = state['loc_desc']
    inv = state['inv_desc']
    surr = ' '.join(state['surrounding_objs'].keys())
    score = state['score']

    base = f"Location: {loc}. Inventory: {inv}. Objects: {surr}. Score: {score}."

    if action:
        context_window.append(f"Action: {action}")

    context_window.append(base)

    # Halte nur letzten 3 Elemente
    if len(context_window) > 3:
        context_window.pop(0)

    return " ||| ".join(context_window)

# 4️⃣ Start game loop
done = False
state = env.get_state()
print("gameloop")
for step in range(400):  # max steps
    if done:
        break

    # Get loc desc and inv desc
    loc_desc = env.step('look')[0]
    env.set_state(state)
    inv_desc = env.step('inventory')[0]
    env.set_state(state)

    # Prepare input for BERT
    input_text = loc_desc + ' ' + inv_desc
    encoding = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)

    # Predict action
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

    predicted_action = le.inverse_transform([pred])[0]
    print(f"Predicted action: {predicted_action}")

    # Take action
    obs, reward, done, info = env.step(predicted_action)
    print('observation:', obs)
    print('reward:', reward)
    print('info:', info)
    print('Steps:',step)
    print('Total Score', info['score'], 'Moves', info['moves'])

    state = env.get_state()

env.close()
