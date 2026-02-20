import os
import json
import torch
from src.train import PLAIN_VOCAB, MambaCipherSolver

TEST_DATA_DIR = "eval_data"

def decode_plain(indices):
    """Converts list of integers (0-25) back into a string (a-z)."""
    mapping = {i: chr(i + 97) for i in range(26)}
    return "".join([mapping.get(idx, "?") for idx in indices])

def ser(pred, plaintext):
    if len(plaintext) == 0:
        return 0.0
    count = 0
    for i in range(len(plaintext)):
        if pred[i] == plaintext[i]:
            count += 1
    return 1.0 - (count / len(plaintext))

def test_model(test_dir, model_path="./src/mamba_cipher_model.pth"):
    checkpoint = torch.load(model_path, map_location="cuda")

    cipher_vocab = checkpoint['cipher_vocab']

    model = MambaCipherSolver(cipher_vocab, PLAIN_VOCAB).to("cuda")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    results = {}
    
    if not os.path.exists(test_dir):
        print(f"Error: {test_dir} not found.")
        return

    print("Testing files...")
    
    with torch.no_grad():
        for filename in sorted(os.listdir(test_dir)):
            if filename.endswith(".json"):
                filepath = os.path.join(test_dir, filename)
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                raw_cipher = data["ciphertext"]
                if isinstance(raw_cipher, str):
                    raw_cipher = [int(x) for x in raw_cipher.split()]

                input_tensor = torch.tensor(raw_cipher, dtype=torch.long).unsqueeze(0).to("cuda")
                
                logits = model(input_tensor)
                pred_indices = torch.argmax(logits, dim=-1).squeeze().tolist()
                
                if isinstance(pred_indices, int):
                    pred_indices = [pred_indices]
                
                deciphered_text = decode_plain(pred_indices)
                results[filename] = deciphered_text
                plaintext = data["plaintext"]
                symbol_err_rate = ser(deciphered_text, plaintext)
                
                print(f"File: {filename} | Predicted: {deciphered_text}")

    return symbol_err_rate

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_folder = os.path.join(base_dir, TEST_DATA_DIR)
    
    symbol_err_rate = test_model(test_folder)
    print(f"Symbol error rate: {symbol_err_rate}")