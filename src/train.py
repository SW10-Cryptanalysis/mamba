import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from mamba_ssm import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm
import os
import json

PLAIN_VOCAB = 26

class CipherDataset(Dataset):
    def __init__(self, directory_path, max_seq_len):
        self.cipher_data = []
        self.plain_data = []
        self.max_seq_len = max_seq_len

        for filename in os.listdir(directory_path):
            if filename.endswith(".json"):
                filepath = os.path.join(directory_path, filename)

                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                    ciphertext = data["recurrence_encoding"]
                    
                    if isinstance(ciphertext, str):
                        ciphertext = [int(x) for x in ciphertext.split()]

                    plaintext = data["plaintext"]
                    
                    encoded_plain = self.encode_plaintext(plaintext)
                    
                    cipher_tensor = torch.tensor(ciphertext, dtype=torch.long)
                    plain_tensor = torch.tensor(encoded_plain, dtype=torch.long)
                    
                    self.cipher_data.append(self._pad_truncate(cipher_tensor))
                    self.plain_data.append(self._pad_truncate(plain_tensor))

    def encode_plaintext(self, text):
        mapping = {chr(i + 97): i for i in range(26)}
        encoded = []
        for char in text.lower():
            if char in mapping:
                encoded.append(mapping[char])
                
        return encoded

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
        for _, (cipher, plain) in enumerate(train_loader):
            cipher, plain = cipher.to("cuda"), plain.to("cuda")
            
            optimizer.zero_grad()
            outputs = model(cipher)
            
            loss = criterion(outputs.view(-1, PLAIN_VOCAB), plain.view(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'cipher_vocab': cipher_vocab,
        }, save_path)
    
    print(f"Finished Training. Model saved to {save_path}")

def get_max_stats(directory_path):
    max_length = 0
    max_symbols = 0
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            filepath = os.path.join(directory_path, filename)
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                file_length = data.get("length", 0)
                file_symbols = data.get("num_symbols", 0)
                
                if file_length > max_length:
                    max_length = file_length
                    
                if file_symbols > max_symbols:
                    max_symbols = file_symbols
                    
            except (json.JSONDecodeError, IOError) as e:
                print(f"Skipping {filename} due to error: {e}")
                continue
    print(f"Max length: {max_length}, Cipher vocab: {max_symbols}") 
    return max_length, max_symbols

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