import glob
import os
import json
import torch
import argparse
from src.train import MambaCipherSolver
from src.config import Config

config = Config()

def decode_plain(indices):
    """Converts list of integers (0-25) back into a string (a-z)."""
    mapping = {i: chr(i + 97) for i in range(26)}
    return "".join([mapping.get(idx, "?") for idx in indices])

def ser(pred, plaintext):
    if len(plaintext) == 0:
        return 0.0
    count = sum(1 for p, t in zip(pred, plaintext) if p == t)
    return 1.0 - (count / len(plaintext))

def test_model(test_dir, model_path=None):
    if model_path is None:
        list_of_files = glob.glob(os.path.join(config.save_path, "*.pth"))
        if not list_of_files:
            print(f"Error: No models found in {config.save_path}")
            return None
        model_path = max(list_of_files, key=os.path.getctime)
    
    if not os.path.exists(model_path):
        alternative_path = os.path.join(config.save_path, model_path)
        if os.path.exists(alternative_path):
            model_path = alternative_path
        else:
            print(f"Error: Could not find model at {model_path}")
            return None

    print(f"Testing model: {model_path}")

    checkpoint = torch.load(model_path, map_location="cuda")

    if isinstance(config.unique_homophones, int):
        cipher_vocab = config.unique_homophones + config.buffer
    else:
        cipher_vocab = checkpoint['cipher_vocab']

    model = MambaCipherSolver(cipher_vocab, config.plain_vocab_size).to("cuda")
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
                
                print(f"File: {filename} | SER: {symbol_err_rate} | Predicted: {deciphered_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Mamba Cipher Model")
    parser.add_argument(
        "model_name", 
        nargs="?", 
        default=None, 
        help="Name of the .pth model file (defaults to latest in output dir)"
    )
    args = parser.parse_args()

    m_path = os.path.join(config.save_path, args.model_name) if args.model_name else None
    
    test_model(config.eval_data_dir, model_path=m_path)