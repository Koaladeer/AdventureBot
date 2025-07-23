import json
import os
import joblib
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns # Added for better plotting aesthetics

#!pip install transformers torch scikit-learn matplotlib seaborn pandas joblib
#!pip install -U accelerate
#!pip install -U transformers
#!pip install tf-keras

# Define a normalization dictionary for common actions (matches player's map)
ACTION_NORMALIZATION_MAP = {
    'N': 'north', 'S': 'south', 'E': 'east', 'W': 'west', 'U': 'up', 'D': 'down'
}

class LossTrackerCallback(TrainerCallback):
    """Callback to track training and evaluation losses."""
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.eval_steps = []
        self.global_steps = [] # To store global steps for training loss

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            self.train_losses.append(logs['loss'])
            self.global_steps.append(state.global_step) # Store global step for training loss

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            self.eval_losses.append(metrics['eval_loss'])
            self.eval_steps.append(state.global_step)

class JerichoDataset(Dataset):
    """Custom Dataset for Jericho game data."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def load_data_and_extract_pairs(filepath: str) -> tuple[list, list]:
    """Loads JSON data and extracts enriched state-action pairs."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    inputs, targets = [], []
    for sample in data:
        state = sample['state']
        # IMPROVEMENT: Include inv_desc to match player's input format
        input_text = (
            f"Observation: {state['obs']}. Location: {state['loc_desc']}. "
            f"Inventory: {state['inv_desc']}. " # Added inventory
            f"Surroundings: {' '.join(state['surrounding_objs'].keys())}. "
            f"Score: {state['score']}."
        )
        inputs.append(input_text)
        targets.append(sample['action'])
    return inputs, targets

def normalize_and_balance_data(inputs: list, targets: list) -> tuple[list, list]:
    """Normalizes actions and balances the dataset by downsampling."""
    # IMPROVEMENT: Use the same normalization map as the player
    normalized_targets = [ACTION_NORMALIZATION_MAP.get(a.strip(), a.strip().lower()) for a in targets]

    balanced_inputs, balanced_targets = [], []
    class_counts = Counter()
    for inp, tgt in zip(inputs, normalized_targets):
        max_count = 30 if tgt in ['north','south','east','west','up','down'] else 200
        if class_counts[tgt] < max_count:
            balanced_inputs.append(inp)
            balanced_targets.append(tgt)
            class_counts[tgt] += 1
    print(f"Balanced dataset size: {len(balanced_targets)}")
    return balanced_inputs, balanced_targets

def prepare_datasets(inputs: list, targets: list, model_path: str):
    """Prepares datasets for training: splits, encodes labels, tokenizes."""
    X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.1, random_state=42)

    le = LabelEncoder()
    le.fit(y_train + y_val)
    joblib.dump(le, os.path.join(model_path, 'label_encoder.joblib')) # Save label encoder

    y_train_enc, y_val_enc = le.transform(y_train), le.transform(y_val)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(X_train, truncation=True, padding=True)
    val_encodings = tokenizer(X_val, truncation=True, padding=True)

    return JerichoDataset(train_encodings, y_train_enc), JerichoDataset(val_encodings, y_val_enc), tokenizer, le.classes_

def setup_trainer(model_path: str, num_labels: int, train_dataset: Dataset, val_dataset: Dataset):
    """Sets up and returns the Hugging Face Trainer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=os.path.join('results', os.path.basename(model_path)),
        num_train_epochs=40,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy='epoch',
        save_strategy='epoch', # Save checkpoints at each epoch
        load_best_model_at_end=True, # Load the best model at the end of training
        metric_for_best_model="eval_loss", # Metric to use for best model selection
        greater_is_better=False, # Lower eval_loss is better
    )
    loss_tracker = LossTrackerCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[loss_tracker]
    )
    return trainer, loss_tracker

def plot_loss_curve(loss_tracker: LossTrackerCallback, save_path: str):
    """
    Plots training and validation loss using seaborn for better aesthetics.
    The plot shows both training loss and validation loss against the training steps,
    allowing for a direct comparison of model performance on seen vs. unseen data.
    """
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid") # Use a nice seaborn style for better aesthetics

    # Plot Training Loss
    # `global_steps` ensures correct x-axis alignment even if logging_steps is large
    plt.plot(loss_tracker.global_steps, loss_tracker.train_losses,
             label='Training Loss', color='skyblue', linewidth=1.5, alpha=0.8)

    # Plot Validation Loss
    # `eval_steps` aligns validation loss with the global steps at which evaluations occurred
    plt.plot(loss_tracker.eval_steps, loss_tracker.eval_losses,
             label='Validation Loss', color='salmon', marker='o', linestyle='--', markersize=6, alpha=0.9)

    plt.title('Training vs. Validation Loss Over Steps', fontsize=16)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    plt.savefig(save_path, dpi=300) # Save with higher DPI for better quality
    print(f"Saved: {save_path}")
    plt.show()

if __name__ == "__main__":
    DATA_PATH = 'data/data_zork1.json'
    MODEL_NAME = 'bert_zork_modelV04' # New model name for improved prompts
    MODEL_PATH = os.path.join('models', MODEL_NAME)
    PLOT_DIR = 'plots'
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1. Load data and extract pairs
    inputs, targets = load_data_and_extract_pairs(DATA_PATH)

    # 2. Normalize and balance data
    balanced_inputs, balanced_targets = normalize_and_balance_data(inputs, targets)

    # 3. Prepare datasets
    train_dataset, val_dataset, tokenizer, label_classes = prepare_datasets(balanced_inputs, balanced_targets, MODEL_PATH)

    # 4. Setup Trainer
    trainer, loss_tracker = setup_trainer(MODEL_PATH, len(label_classes), train_dataset, val_dataset)

    # 5. Train
    trainer.train()

    # 6. Save model & tokenizer
    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    # 7. Plot training loss
    plot_loss_curve(loss_tracker, os.path.join(PLOT_DIR, f'{MODEL_NAME}_train_val_loss_curve.png'))

    print("Training complete and model saved!")
