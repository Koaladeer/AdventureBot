import json
import os
import random
import joblib
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from JerichoWorld.defines_updated import ABBRV_DICT
from transformers import TrainerCallback
import matplotlib.pyplot as plt

# 1️⃣ Load dataset
with open('data/data_zork1.json', 'r') as f:
    data = json.load(f)
#save location
model_name ='bert_zork_modelV04'
model_path = 'models/'+model_name
os.makedirs(model_path, exist_ok=True)

# 2️⃣ Extract state-action pairs
inputs, targets = [], []
for sample in data:
    state = sample['state']
    loc_desc = state['loc_desc']
    inv_desc = state['inv_desc']
    inv_dict = state['inv_attrs']
    observation = state['obs']
    surr_objs = ' '.join([surr for surr in state['surrounding_objs'].keys()])
    score = state['score']
    valid_acts = state['valid_acts']
    # Enriched state encoding
    input_text = f"Observation: {observation}. Location: {loc_desc}. Surroundings: {surr_objs}. Score: {score}. Actions: {valid_acts}. Inventory: {inv_dict} "

    inputs.append(input_text)

    targets.append(sample['action'])
# 3️⃣ Normalize actions
#targets = [ABBRV_DICT.get(tgt.strip().lower(), tgt.strip().lower()) for tgt in targets]
normalization_dict = {'N': 'north','S': 'south','E': 'east','W': 'west','U': 'up','D': 'down'}
targets = [normalization_dict.get(a.strip(), a.strip()) for a in targets]

# 3️⃣ Balance dataset (downsample frequent actions)
balanced_inputs, balanced_targets = [], []
class_counts = Counter()
action_counter = Counter(targets)

for inp, tgt in zip(inputs, targets):
    if tgt in ['north','south','east','west','up','down']:
        max_count = 10  # aggressive downsampling of movement actions
    else:
        max_count = 200  # allow much more data for interactive actions

    if class_counts[tgt] < max_count:
        balanced_inputs.append(inp)
        balanced_targets.append(tgt)
        class_counts[tgt] += 1

print("Balanced dataset size:", len(balanced_targets))

#plotting difference


# 4️⃣ Train-test split
X_train, X_val, y_train, y_val = train_test_split(balanced_inputs, balanced_targets, test_size=0.1, random_state=42)

# 5️⃣ Label encoding
y_all = y_train + y_val
le = LabelEncoder()
le.fit(y_all)
num_labels = len(le.classes_)

# Save label encoder
joblib.dump(le,os.path.join(model_path, 'label_encoder.joblib'))

# Transform labels
y_train_enc = le.transform(y_train)
y_val_enc = le.transform(y_val)

# 6️⃣ Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(X_train, truncation=True, padding=True)
val_encodings = tokenizer(X_val, truncation=True, padding=True)

# 7️⃣ Dataset wrapper
class JerichoDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = JerichoDataset(train_encodings, y_train_enc)
val_dataset = JerichoDataset(val_encodings, y_val_enc)

# 8️⃣ Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
model.to(device)
#Loss Function
class LossTrackerCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.eval_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            self.train_losses.append(logs['loss'])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            self.eval_losses.append(metrics['eval_loss'])
            self.eval_steps.append(state.global_step)


loss_tracker = LossTrackerCallback()
# 9️⃣ Trainer
training_args = TrainingArguments(
    output_dir='results/'+model_name,
    num_train_epochs=40,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy='epoch',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[loss_tracker]
)

# 1️⃣0️⃣ Train
train_result=trainer.train()

# Save model & tokenizer
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
# Plot training loss

trainer.train()
trainer.evaluate()
# plot trianing vs validation loss
plt.figure(figsize=(10, 5))
plt.plot(loss_tracker.train_losses, label='Training Loss', alpha=0.7)
plt.plot(loss_tracker.eval_steps, loss_tracker.eval_losses, label='Validation Loss', alpha=0.9, marker='o')
plt.title('Training vs. Validation Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("plots/train_val_loss_curve.png")
print("Saved: plots/train_val_loss_curve.png")
plt.show()


print("V3 Training complete and model saved!")
print("Training complete and model saved!")
