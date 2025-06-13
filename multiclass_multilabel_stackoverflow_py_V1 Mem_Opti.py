import pandas as pd
import re
from collections import Counter
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# 1. Load data
def load_data(questions_path, tags_path):
    
    read_csv_kwargs = {
        'encoding': 'latin1',   
        'low_memory': False     
    }
    questions = pd.read_csv(questions_path, **read_csv_kwargs)
    tags      = pd.read_csv(tags_path,      **read_csv_kwargs)

    if 'Tags' not in tags.columns:
        cols = tags.columns.tolist()
        if len(cols) >= 2:
            tags = tags.rename(columns={cols[1]: 'Tags'})
        else:
            raise ValueError("Tags file must have at least two columns: Id and Tags")
    data = questions.merge(tags, on='Id')
    return data

# 2. Basic text cleaning
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', str(text))
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 3. Extract top-N tags
def get_top_tags(tag_lists, top_n=10):
    all_tags = [tag for sub in tag_lists for tag in sub]
    return [tag for tag, _ in Counter(all_tags).most_common(top_n)]

# 4. Filter rows containing any of the top tags
def filter_top_tags(data, top_tags):
    data['tag_list'] = data['Tags'].apply(lambda x: x.split(';') if x else [])
    mask = data['tag_list'].apply(lambda tags: any(tag in top_tags for tag in tags))
    return data[mask].reset_index(drop=True)

# 5. Binarize labels
def prepare_labels(data, top_tags):
    mlb = MultiLabelBinarizer(classes=top_tags)
    y = mlb.fit_transform(data['tag_list'])
    return y, mlb

# 6. Build vocabulary from cleaned texts
def build_vocab(texts, min_freq=1):
    counter = Counter()
    for txt in texts:
        counter.update(txt.split())
    vocab = {'<PAD>': 0, '<OOV>': 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

# 7. Convert texts to padded sequences
def texts_to_padded_sequences(texts, vocab, max_length=100):
    sequences = []
    for txt in texts:
        seq = [vocab.get(w, vocab['<OOV>']) for w in txt.split()]
        if len(seq) < max_length:
            seq += [vocab['<PAD>']] * (max_length - len(seq))
        else:
            seq = seq[:max_length]
        sequences.append(seq)
    return sequences

# PyTorch Dataset
class StackOverflowDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx],    dtype=torch.float),
        )

# LSTM model
class TagPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags, dropout=0.5):
        super(TagPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout   = nn.Dropout(dropout)
        self.fc1       = nn.Linear(hidden_dim, 64)
        self.relu      = nn.ReLU()
        self.fc2       = nn.Linear(64, num_tags)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        x = self.dropout(h)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Training utilities
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)

def main():
    # Raw-string paths
    questions_file = r"C:\Hrishi files\IIT Kanpur test\IITK Machine learning projects\Project 1  Multiclass Multilabel prediction For stack overflow Questions\Qoutput1.csv"
    tags_file      = r"C:\Hrishi files\IIT Kanpur test\IITK Machine learning projects\Project 1  Multiclass Multilabel prediction For stack overflow Questions\Toutput1.csv"

    # 1) Load
    data = load_data(questions_file, tags_file)
    print("Columns:", data.columns.tolist())

    # 2) Ensure no NaN in Tags
    data['Tags'] = data['Tags'].fillna('')

    # 3) Clean text
    data['clean_text'] = data['Body'].apply(clean_text)

    # 4) Compute top tags
    raw_tag_lists = data['Tags'].apply(lambda x: x.split(';') if x else [])
    top_tags = get_top_tags(raw_tag_lists, top_n=10)
    print("Top tags:", top_tags)

    # 5) Filter for top tags
    data = filter_top_tags(data, top_tags)
    print("Filtered shape:", data.shape)

    # 6) Labels
    y, mlb = prepare_labels(data, top_tags)

    # 7) Text → sequences
    vocab = build_vocab(data['clean_text'])
    X = texts_to_padded_sequences(data['clean_text'], vocab, max_length=100)

    # 8) Split & loaders
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_loader = DataLoader(StackOverflowDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader   = DataLoader(StackOverflowDataset(X_val,   y_val),   batch_size=32)

    # 9) Device + model
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = TagPredictor(vocab_size=len(vocab), embed_dim=100, hidden_dim=128, num_tags=y.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for ep in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss = eval_epoch(model, val_loader, criterion, device)  
        print(f"Epoch {ep}: Train Loss={tr_loss:.4f}, Val Loss={vl_loss:.4f}")

    # 11) Save
    torch.save(model.state_dict(), 'stackoverflow_tag_model_pytorch.pt')
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open('mlb.pkl', 'wb') as f:
        pickle.dump(mlb, f)
    print("Done.")

if __name__ == "__main__":
    main()
