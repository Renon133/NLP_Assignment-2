import os
import re
import json
import argparse
from tqdm import tqdm
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast

# Argument parser setup
parser = argparse.ArgumentParser(description="Train a Gujarati tokenizer.")
parser.add_argument("--data", type=str, required=True, help="Path to the text data file")
parser.add_argument("--vocab_size", type=int, default=32000, help="Size of the vocabulary")
parser.add_argument("--save_dir", type=str, default="./gujarati_tokenizer", help="Base directory to save tokenizers")
parser.add_argument("--tokenizer_num", type=int, required=True, help="Number identifier for this tokenizer")

args = parser.parse_args()

# Preprocessing function to clean the text
def preprocess_text(text):
    # Remove English characters and special symbols but keep Gujarati characters, numbers, and basic punctuation
    cleaned_text = re.sub(r'[^\u0A80-\u0AFF0-9\s.,!?]', '', str(text))
    return cleaned_text.strip()

# Read and preprocess the text data
preprocessed_lines = []
with open(args.data, 'r', encoding='utf-8') as file:
    for line in tqdm(file, desc="Preprocessing lines"):
        cleaned_line = preprocess_text(line)
        if cleaned_line:
            preprocessed_lines.append(cleaned_line)

# Save the preprocessed text to a temporary file for tokenizer training
preprocessed_file = "preprocessed_gujarati_data.txt"
with open(preprocessed_file, 'w', encoding='utf-8') as out_file:
    for line in preprocessed_lines:
        out_file.write(line + '\n')

# Create a unique directory for each tokenizer
tokenizer_save_path = os.path.join(args.save_dir, f"tokenizer_{args.tokenizer_num}")
os.makedirs(tokenizer_save_path, exist_ok=True)

# Train the tokenizer using SentencePieceBPETokenizer
print(f"Training tokenizer_{args.tokenizer_num} with a vocab size of {args.vocab_size}...")
tokenizer = SentencePieceBPETokenizer()

tokenizer.train(
    files=[preprocessed_file],
    vocab_size=args.vocab_size,
    min_frequency=2,
    special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
)

# Save vocab.json and merges.txt
tokenizer.save_model(tokenizer_save_path, f"gujarati_tokenizer_{args.tokenizer_num}")

# Convert to PreTrainedTokenizerFast and save tokenizer.json
transformer_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    vocab_file=os.path.join(tokenizer_save_path, 'vocab.json'),
    merges_file=os.path.join(tokenizer_save_path, 'merges.txt')
)

# Save the tokenizer with tokenizer.json
transformer_tokenizer.save_pretrained(tokenizer_save_path)

# Manually create and save the config.json file
config = {
    "model_type": "bpe",
    "vocab_size": args.vocab_size,
    "special_tokens": ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
}
with open(os.path.join(tokenizer_save_path, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)

print(f"Tokenizer_{args.tokenizer_num} has been trained and saved with all necessary files in {tokenizer_save_path}.")
