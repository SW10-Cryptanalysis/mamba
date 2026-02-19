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

PLAIN_VOCAB = 26

class CipherDataset(Dataset):
    def __init__(self, directory_path, max_seq_len):
        self.max_seq_len = max_seq_len
        files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".json")]
        worker_args = [(f, max_seq_len) for f in files]
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(encode_file, worker_args), total=len(files), desc="Encoding Data"))
        
        # Filter and Unpack
        valid_results = [r for r in results if r is not None]
        cipher_lists, plain_lists = zip(*valid_results)

        # NOW convert to tensors in the main process
        print("Converting to tensors...")
        self.cipher_data = torch.tensor(cipher_lists, dtype=torch.long)
        self.plain_data = torch.tensor(plain_lists, dtype=torch.long)

    def _pad_truncate(self, tensor):
        if len(tensor) > self.max_seq_len:
            return tensor[:self.max_seq_len]
        elif len(tensor) < self.max_seq_len:
            padding = torch.zeros(self.max_seq_len - len(tensor), dtype=torch.long)
            return torch.cat([tensor, padding])
        return tensor

    def __len__(self):
        return len(self.cipher_data)

    def __getitem__(self, idx):
        return self.cipher_data[idx], self.plain_data[idx]

def process_json(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data.get("length", 0), data.get("num_symbols", 0)
    except Exception:
        return 0, 0

def get_max_stats(directory_path):
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".json")]
    
    max_length = 0
    max_symbols = 0
    
    # Use ProcessPoolExecutor to run across all CPU cores
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_json, files), total=len(files), desc="Scanning Files"))
        
    for length, symbols in results:
        if length > max_length: max_length = length
        if symbols > max_symbols: max_symbols = symbols
            
    print(f"Max length: {max_length}, Cipher vocab: {max_symbols}") 
    return max_length, max_symbols

def encode_file(args):
    filepath, max_seq_len = args
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        ciphertext = data["recurrence_encoding"]
        if isinstance(ciphertext, str):
            ciphertext = [int(x) for x in ciphertext.split()]
        
        mapping = {chr(i + 97): i for i in range(26)}
        encoded_plain = [mapping[c] for c in data["plaintext"].lower() if c in mapping]
        
        def pad_trunc(list_data, max_len):
            if len(list_data) > max_len: return list_data[:max_len]
            return list_data + [0] * (max_len - len(list_data))

        return (
            pad_trunc(ciphertext, max_seq_len),
            pad_trunc(encoded_plain, max_seq_len)
        )
    except Exception as e:
        print(f"\nWorker Error on {os.path.basename(filepath)}: {e}")
        return None

def encode_plaintext(text):
    mapping = {chr(i + 97): i for i in range(26)}
    encoded = []
    for char in text.lower():
        if char in mapping:
            encoded.append(mapping[char])
            
    return encoded

class MambaCipherSolver(nn.Module):
    def __init__(self, vocab_cipher_size, vocab_plain_size, d_model=128, n_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_cipher_size, d_model)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": RMSNorm(d_model), 
                "mixer": Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2)
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

def train_model(model, train_loader, cipher_vocab, epochs=10, save_path="./src/mamba_cipher_model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False)
        for cipher, plain in loop:
            cipher, plain = cipher.to("cuda"), plain.to("cuda")
            
            optimizer.zero_grad()
            outputs = model(cipher)
            
            loss = criterion(outputs.view(-1, PLAIN_VOCAB), plain.view(-1))
            
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

    train_folder_path = "./src/train_data" 

    max_len, cipher_vocab = get_max_stats(train_folder_path)
    if max_len == 0:
        raise ValueError(f"No valid JSON data found in {train_folder_path}.")
    dataset = CipherDataset(train_folder_path, max_seq_len=max_len)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = MambaCipherSolver(
        vocab_cipher_size=cipher_vocab+1, 
        vocab_plain_size=PLAIN_VOCAB,
        d_model=128
    ).to("cuda")

    print(f"Training...")
    train_model(model, loader, cipher_vocab+1, epochs=10)