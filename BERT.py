import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Dataset class
class ReviewsDataset(Dataset):
    def __init__(self, reviews, ratings, tokenizer, max_len):
        self.reviews = reviews
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, item):
        review = str(self.reviews[item])
        rating = self.ratings[item]
        
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(rating, dtype=torch.long)
        }

# Function to create data loaders
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = ReviewsDataset(
        reviews=df.reviews.to_numpy(),
        ratings=df.ratings.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

# Function to train the model
def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
    model = model.train()
    
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        _, preds = torch.max(outputs.logits, dim=1)
        loss = loss_fn(outputs.logits, labels)
        
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return correct_predictions.double() / n_examples, np.mean(losses)

# Function to evaluate the model
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            loss = loss_fn(outputs.logits, labels)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    
    return correct_predictions.double() / n_examples, np.mean(losses)

# Main function to create and train the model
def train_sentiment_model(df):
    # Preparing the data
    df = df.dropna(subset=['reviews', 'ratings'])
    df['ratings'] = df['ratings'].astype(int) - 1  # Convert ratings to 0-4 for 5 classes
    
    # Splitting the data
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    
    # Parameters
    PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
    BATCH_SIZE = 16
    MAX_LEN = 160
    EPOCHS = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    
    # Data loaders
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    
    # Model
    model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=5)
    model = model.to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps // 3, gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
    
    # Training the model
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,    
            loss_fn, 
            optimizer, 
            DEVICE, 
            scheduler, 
            len(df_train)
        )
        
        print(f'Train loss {train_loss} accuracy {train_acc}')
        
        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            DEVICE,
            len(df_val)
        )
        
        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()
    
    return model, tokenizer, DEVICE, MAX_LEN

# Function to make predictions with the trained model
def bert_predict(model, tokenizer, device, max_len, text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output.logits, dim=1)
    
    return prediction.item() + 1  # Convert back to 1-5 rating

# Example usage:
# df = pd.read_csv('path_to_your_file.csv')
# model, tokenizer, device, max_len = train_sentiment_model(df)
# text = "This is a sample review."
# rating = bert_predict(model, tokenizer, device, max_len, text)
# print(f'Predicted rating: {rating}')
