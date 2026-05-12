import pandas as pd
import tiktoken
import random

def count_tokens(text, model="gpt-4o-mini"):
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback to cl100k_base encoding
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

# Read the CSV file
csv_path = "/Users/ganeshnallagachu/Desktop/RAFT_exp/chunk_data/English_chunks - tags.csv"
print(f"Reading CSV file: {csv_path}")
df = pd.read_csv(csv_path)

print(f"Total chunks in file: {len(df)}")

# Sample 10 random chunks
random.seed(42)  # For reproducibility
sample_indices = random.sample(range(len(df)), min(10, len(df)))
sampled_chunks = df.iloc[sample_indices]

print("\n" + "="*80)
print("SAMPLED CHUNKS ANALYSIS")
print("="*80)

token_counts = []
for idx, row in sampled_chunks.iterrows():
    chunk_text = str(row['chunk'])
    token_count = count_tokens(chunk_text)
    token_counts.append(token_count)
    
    print(f"\nChunk {len(token_counts)} (Index: {idx}):")
    print(f"  File: {row.get('file_name', 'N/A')}")
    print(f"  Chunk Number: {row.get('chunk_number', 'N/A')}")
    print(f"  Text Preview: {chunk_text[:100]}...")
    print(f"  Token Count: {token_count}")

# Calculate average
avg_tokens = sum(token_counts) / len(token_counts)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Number of samples: {len(token_counts)}")
print(f"Token counts: {token_counts}")
print(f"Average chunk size in tokens: {avg_tokens:.2f}")
print(f"Min tokens: {min(token_counts)}")
print(f"Max tokens: {max(token_counts)}")

