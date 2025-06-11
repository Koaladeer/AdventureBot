import json
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
import joblib

# Load data
with open('data/data_zork1.json', 'r') as f:
    data = json.load(f)

# Build input-output pairs
inputs = []
targets = []

for sample in data:
    state = sample['state']
    loc_desc = state['loc_desc']
    inv_desc = state['inv_desc']
    action = sample['action']

    # Combine location & inventory description as input text
    input_text = loc_desc + ' ' + inv_desc

    inputs.append(input_text)
    targets.append(action)

# Split into train/test sets
X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.1)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize inputs
train_encodings = tokenizer(X_train, truncation=True, padding=True, return_tensors='pt')
val_encodings = tokenizer(X_val, truncation=True, padding=True, return_tensors='pt')

le = LabelEncoder()
le.fit(y_train + y_val)

y_train_enc = le.transform(y_train)
y_val_enc = le.transform(y_val)
##save the encoder
joblib.dump(le, 'models/bert_zork_model/label_encoder.joblib')


num_labels = len(le.classes_)


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

class JerichoDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = JerichoDataset(train_encodings, y_train_enc)
val_dataset = JerichoDataset(val_encodings, y_val_enc)
print("startet training")
training_args = TrainingArguments(
    output_dir='model_output/results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=200,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

trainer.save_model("./models/bert_zork_model")
tokenizer.save_pretrained("./models/bert_zork_model")
