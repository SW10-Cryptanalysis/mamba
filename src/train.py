from concurrent.futures import ProcessPoolExecutor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from mamba_ssm import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm
import os
import json
from tqdm import tqdm
from datetime import datetime
from dataclasses import asdict
from src.config import Config

config = Config()

class CipherDataset(Dataset):
    def __init__(self, directory_path, max_seq_len):
        self.max_seq_len = max_seq_len
        self.file_paths = [
            os.path.join(directory_path, f) 
            for f in os.listdir(directory_path) if f.endswith(".json")
        ]
        
        self.mapping = {chr(i + 97): i for i in range(26)}

    def _pad_trunc(self, list_data):
        if len(list_data) > self.max_seq_len:
            return list_data[:self.max_seq_len]
        return list_data + [0] * (self.max_seq_len - len(list_data))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            ciphertext = data["ciphertext"]
            if isinstance(ciphertext, str):
                ciphertext = [int(x) for x in ciphertext.split()]
            
            plaintext = data["plaintext"]
            encoded_plain = [self.mapping[c] for c in plaintext.lower() if c in self.mapping]
            
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
    def __init__(self, vocab_cipher_size, vocab_plain_size=26, d_model=128, n_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_cipher_size, d_model)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": RMSNorm(d_model), 
                "mixer": Mamba2(d_model=d_model, d_state=config.d_state, d_conv=config.d_conv, expand=config.expand)
            }) for _ in range(n_layers)
        ])
        
        self.norm_f = RMSNorm(d_model) 
        self.lm_head = nn.Linear(d_model, vocab_plain_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            residual = x
            x = layer["norm"](x)
            x = layer["mixer"](x) + residual
        
        x = self.norm_f(x)
        return self.lm_head(x)

def train_model(model, train_loader, cipher_vocab, epochs=10, save_path="./outputs/mamba_cipher_model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False)
        for cipher, plain in loop:
            cipher, plain = cipher.to("cuda"), plain.to("cuda")
            
            optimizer.zero_grad()
            if cipher.max() >= model.embedding.num_embeddings:
                print(f"ERROR: Found index {cipher.max().item()} but vocab size is {model.embedding.num_embeddings}")
            outputs = model(cipher)
            
            loss = criterion(outputs.view(-1, config.plain_vocab_size), plain.view(-1))
            
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=f"{loss.item():.4f}")
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Completed - Avg Loss: {avg_loss:.4f}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'cipher_vocab': cipher_vocab
        }, save_path)
    
    print(f"Finished Training. Model saved to {save_path}")

if __name__ == "__main__":
    train_path = os.path.abspath(config.train_data_dir)

    if isinstance(config.unique_homophones, int) and isinstance(config.max_len, int):
        max_len = config.max_len
        cipher_vocab = config.unique_homophones
    else:
        max_len, cipher_vocab = get_max_stats(train_path)
        if max_len == 0:
            raise ValueError(f"No valid JSON data found in {train_path}.")
    

    dataset = CipherDataset(train_path, max_seq_len=max_len)
    cores = os.cpu_count() or 1
    num_workers = max(1, cores // 2)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers)

    model = MambaCipherSolver(
        vocab_cipher_size=cipher_vocab+config.buffer, 
        vocab_plain_size=config.plain_vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers
    ).to("cuda")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    os.makedirs(config.save_path, exist_ok=True)
    config_filename = os.path.join(config.save_path, f"config_{timestamp}.json")
    with open(config_filename, 'w') as f:
        json.dump(asdict(config), f, indent=4)
    filename = f"mamba2_{timestamp}.pth"

    print("Training...")
    train_model(model, loader, cipher_vocab+config.buffer, epochs=config.epochs, save_path=os.path.join(config.save_path, filename))