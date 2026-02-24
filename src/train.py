from concurrent.futures import ProcessPoolExecutor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from mamba_ssm import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm
import os
import json
import zipfile
from tqdm import tqdm
from datetime import datetime
from dataclasses import asdict
from src.config import Config

config = Config()

class CipherDataset(Dataset):
    def __init__(self, directory_path, max_seq_len):
        self.max_seq_len = max_seq_len
        self.file_paths = []
        
        print(f"Loading file paths from {directory_path}...")
        with os.scandir(directory_path) as entries:
            for entry in tqdm(entries, desc="Scanning for JSON/ZIP", leave=False):
                if entry.is_file():
                    if entry.name.endswith(".json"):
                        self.file_paths.append((entry.path, None))
                    elif entry.name.endswith(".zip"):
                        with zipfile.ZipFile(entry.path, 'r') as z:
                            for file_info in z.infolist():
                                if file_info.filename.endswith(".json"):
                                    self.file_paths.append((entry.path, file_info.filename))
        
        print(f"Successfully indexed {len(self.file_paths)} files.")
        self.mapping = {chr(i + 97): i for i in range(26)}
        self.sep_token = config.unique_homophones + 1
        self.char_offset = self.sep_token + 1

    def _pad_trunc(self, list_data):
        if len(list_data) > self.max_seq_len:
            return list_data[:self.max_seq_len]
        return list_data + [0] * (self.max_seq_len - len(list_data))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path, internal_name = self.file_paths[idx]
        
        try:
            if internal_name:
                with zipfile.ZipFile(path, 'r') as z:
                    with z.open(internal_name) as f:
                        data = json.load(f)
            else:
                with open(path, 'r') as f:
                    data = json.load(f)
            
            ciphertext = data["ciphertext"]
            if isinstance(ciphertext, str):
                ciphertext = [int(x) for x in ciphertext.split()]
            plaintext = data["plaintext"]
            encoded_plain = [self.mapping[c] + self.char_offset for c in plaintext.lower() if c in self.mapping]
            cipher_tensor = torch.tensor(self._pad_trunc(ciphertext), dtype=torch.long)
            plain_tensor = torch.tensor(self._pad_trunc(encoded_plain), dtype=torch.long)
            return cipher_tensor, plain_tensor
        except Exception:
            return torch.zeros(self.max_seq_len, dtype=torch.long), torch.zeros(self.max_seq_len, dtype=torch.long)

def process_json(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        ciphertext = data.get("ciphertext", [])
        if isinstance(ciphertext, str):
            ciphertext = [int(x) for x in ciphertext.split()]
        
        actual_max_val = max(ciphertext) if ciphertext else 0
        actual_len = len(ciphertext)
        
        return actual_len, actual_max_val
    except Exception:
        return 0, 0

def get_max_stats(directory_path):
    print(f"Scanning directory: {directory_path}...")
    
    files = [
        os.path.join(directory_path, f) 
        for f in os.listdir(directory_path) if f.endswith(".json")
    ]
    
    if not files:
        print("Warning: No JSON files found in the directory.")
        return 0, 0

    max_length = 0
    max_symbols = 0
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_json, files), 
            total=len(files), 
            desc="Analyzing Dataset Dimensions"
        ))
        
    for length, symbols in results:
        if length > max_length: 
            max_length = length
        if symbols > max_symbols: 
            max_symbols = symbols
            
    print(f"Scan complete. Max Seq Len: {max_length}, Highest Symbol ID: {max_symbols}")
    return max_length, max_symbols

class MambaCipherSolver(nn.Module):
    def __init__(self, vocab_size, char_offset, d_model=128, n_layers=4):
        super().__init__()
        self.char_offset = char_offset
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": RMSNorm(d_model), 
                "mixer": Mamba2(d_model=d_model, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand)
            }) for _ in range(n_layers)
        ])
        
        self.norm_f = RMSNorm(d_model) 
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            residual = x
            x = layer["norm"](x)
            x = layer["mixer"](x) + residual
        
        x = self.norm_f(x)
        return self.lm_head(x)

def train_model(model, train_loader, val_loader, cipher_vocab, epochs=10, save_path="./outputs/mamba_cipher_model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.factor, patience=config.patience)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch", leave=False)
        
        for cipher, plain in train_loop:
            cipher, plain = cipher.to("cuda"), plain.to("cuda")
            optimizer.zero_grad()
            outputs = model(cipher)
            loss = criterion(outputs.view(-1, outputs.size(-1)), plain.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for cipher, plain in val_loader:
                cipher, plain = cipher.to("cuda"), plain.to("cuda")
                outputs = model(cipher)
                loss = criterion(outputs.view(-1, outputs.size(-1)), plain.view(-1))
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_loss,
                'cipher_vocab': cipher_vocab,
                'char_offset': model.char_offset,
            }, save_path)

        if current_lr < 1e-7:
            print(f"Current lr: {current_lr}. Stopping early.")
            break

if __name__ == "__main__":
    train_path = os.path.abspath(config.train_data_dir)
    valid_path = os.path.abspath(config.valid_data_dir)

    if isinstance(config.unique_homophones, int) and isinstance(config.max_len, int):
        max_len = config.max_len
        cipher_vocab = config.unique_homophones
    else:
        max_len, cipher_vocab = get_max_stats(train_path)
        if max_len == 0:
            raise ValueError(f"No valid JSON data found in {train_path}.")
    
    cores = os.cpu_count() or 1
    num_workers = max(1, cores - 4)

    train_dataset = CipherDataset(train_path, max_seq_len=max_len)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_dataset = CipherDataset(valid_path, max_seq_len=max_len)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True
    )

    vocab_size = train_dataset.char_offset + config.plain_vocab_size + config.buffer

    model = MambaCipherSolver(
        vocab_size=vocab_size, 
        char_offset=train_dataset.char_offset,
        d_model=config.d_model,
        n_layers=config.n_layers
    ).to("cuda")

    timestamp = datetime.now().strftime("%m%d-%H%M")
    os.makedirs(config.save_path, exist_ok=True)
    config_filename = os.path.join(config.save_path, f"config_{timestamp}.json")
    with open(config_filename, 'w') as f:
        json.dump(asdict(config), f, indent=4)
    filename = f"mamba2_{timestamp}.pth"

    print("Training...")
    train_model(
        model, 
        train_loader, 
        val_loader,
        cipher_vocab + config.buffer, 
        epochs=config.epochs, 
        save_path=os.path.join(config.save_path, filename)
    )