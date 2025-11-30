from tokenizers import BertWordPieceTokenizer
import argparse
import os

def train_wordpiece(input_file, output_dir, vocab_size=30000, min_frequency=2):
    # Initialize the WordPiece tokenizer
    tokenizer = BertWordPieceTokenizer()

    # Train the tokenizer on your dataset
    tokenizer.train(input_file, vocab_size=vocab_size, min_frequency=min_frequency)

    # Save the tokenizer files
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=30000)
    parser.add_argument("--min_frequency", type=int, default=2)
    args = parser.parse_args()
    train_wordpiece(args.input_file, args.output_dir, args.vocab_size, args.min_frequency)