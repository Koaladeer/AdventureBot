import json
from sentence_transformers import InputExample
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
# Load JerichoWorld file
with open('data/train.json') as f:
    data = json.load(f)

# Convert to InputExamples

train_examples = []
for episode in data:
    for step in episode:
        if 'state' in step and 'obs' in step['state'] and 'action' in step:
            obs = step['state']['obs']
            action = step['action']
            train_examples.append(InputExample(texts=[obs, action]))

model = SentenceTransformer('all-MiniLM-L6-v2')

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1)
model.save("models/finetuned-bert-text-adventure01")
