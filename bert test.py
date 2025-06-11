from jericho import FrotzEnv
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load small BERT model (fully local, fast)
tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
model = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

# Define possible actions

# Simple encoding function
def encode(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# Agent function combining valid actions & similarity
def bert_legal_agent(observation, valid_actions):
    obs_vec = encode(observation)

    best_action = None
    best_score = -float('inf')

    for action in valid_actions:
        action_vec = encode(action)
        score = np.dot(obs_vec, action_vec.T)[0][0]
        if score > best_score:
            best_score = score
            best_action = action
    return best_action

# Jericho loop
env = FrotzEnv("games/z-machine-games-master/jericho-game-suite/zork1.z5")
observation, info = env.reset()
done = False

while not done:
    print("\nObservation:", observation)

    valid_actions = env.get_valid_actions()
    print(f"Valid Actions: {valid_actions}")

    action = bert_legal_agent(observation, valid_actions)
    print(f"BERT Agent Action: {action}")

    observation, reward, done, info = env.step(action)
    print(f"Reward: {reward} | Score: {info['score']} | Moves: {info['moves']}")

print("\nFinished. Final Score:", info['score'], 'out of', env.get_max_score())

