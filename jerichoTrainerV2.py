import json
import random
import joblib
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.model_selection import train_test_split

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
# 3️⃣ Normalize actions
normalization_dict = {'N': 'north','S': 'south','E': 'east','W': 'west','U': 'up','D': 'down'}
targets = [normalization_dict.get(a.strip(), a.strip()) for a in targets]

# 3️⃣ Balance dataset (downsample frequent actions)
balanced_inputs, balanced_targets = [], []
class_counts = Counter()
action_counter = Counter(targets)

for inp, tgt in zip(inputs, targets):
    if tgt in ['north','south','east','west','up','down']:
        max_count = 30  # aggressive downsampling of movement actions
    else:
        max_count = 300  # allow much more data for interactive actions

    if class_counts[tgt] < max_count:
        balanced_inputs.append(inp)
        balanced_targets.append(tgt)
        class_counts[tgt] += 1

print("Balanced dataset size:", len(balanced_targets))

# 4️⃣ Train-test split
X_train, X_val, y_train, y_val = train_test_split(balanced_inputs, balanced_targets, test_size=0.1, random_state=42)

# 5️⃣ Label encoding
y_all = y_train + y_val
le = LabelEncoder()
le.fit(y_all)
num_labels = len(le.classes_)

# Save label encoder
joblib.dump(le, 'models/bert_zork_model/label_encoder.joblib')

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
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
model.to(device)

# 9️⃣ Trainer
training_args = TrainingArguments(
    output_dir='results/bert_zork_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 10️⃣ Train
trainer.train()

# 🔥 Save model & tokenizer
trainer.save_model('models/bert_zork_modelV02')
tokenizer.save_pretrained('models/bert_zork_modelV02')

print("Training complete and model saved!")
