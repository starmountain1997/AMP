import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
import time
import numpy as np

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir="runs/finetune_sentiment_analysis")

# Load the IMDb dataset
dataset = load_dataset('imdb')

# Split the training set into train and validation sets
train_valid = dataset['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = train_valid['train']
valid_dataset = train_valid['test']
test_dataset = dataset['test']

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set the format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
valid_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Initialize the model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)

# Optional: Freeze all layers except the classification head
for name, param in model.named_parameters():
    if 'classifier' not in name:
        param.requires_grad = False

# Define DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the optimizer and scheduler
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

epochs = 3
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Define the loss function
loss_fn = CrossEntropyLoss()

# Move model to device
device ="npu:0"
model.to(device)

# Define evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    total_eval_loss = 0
    total_correct = 0
    total_examples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_eval_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)
    
    avg_loss = total_eval_loss / len(dataloader)
    accuracy = total_correct / total_examples
    return avg_loss, accuracy

# Training Loop
for epoch in range(1, epochs + 1):
    model.train()
    epoch_start_time = time.time()
    total_train_loss = 0
    total_correct = 0
    total_examples = 0
    
    for step, batch in enumerate(train_loader, 1):
        step_start = time.time()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        total_train_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_examples += labels.size(0)
        
        # Calculate step metrics
        step_end = time.time()
        step_duration = step_end - step_start
        steps_per_sec = 1 / step_duration if step_duration > 0 else float('inf')
        tokens_per_sec = input_ids.numel() / step_duration if step_duration > 0 else float('inf')
        
        # Global step for TensorBoard
        global_step = (epoch - 1) * len(train_loader) + step
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), global_step)
        writer.add_scalar('Throughput/Steps_per_sec', steps_per_sec, global_step)
        writer.add_scalar('Throughput/Tokens_per_sec', tokens_per_sec, global_step)
        
        # Print progress
        if step % 50 == 0 or step == len(train_loader):
            avg_train_loss = total_train_loss / step
            train_accuracy = total_correct / total_examples
            print(f"Epoch [{epoch}/{epochs}], Step [{step}/{len(train_loader)}], "
                  f"Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
                  f"Steps/sec: {steps_per_sec:.2f}, Tokens/sec: {tokens_per_sec:.2f}")
    
    # Evaluation after each epoch
    valid_loss, valid_accuracy = evaluate(model, valid_loader, device)
    writer.add_scalar('Loss/validation', valid_loss, epoch)
    writer.add_scalar('Accuracy/validation', valid_accuracy, epoch)
    
    print(f"--- Epoch {epoch} completed in {time.time() - epoch_start_time:.2f} seconds ---")
    print(f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}\n")

# Final Evaluation on Test Set
test_loss, test_accuracy = evaluate(model, test_loader, device)
writer.add_scalar('Loss/test', test_loss, epochs + 1)
writer.add_scalar('Accuracy/test', test_accuracy, epochs + 1)

print(f"--- Test Set Evaluation ---")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Close the TensorBoard writer
writer.close()
