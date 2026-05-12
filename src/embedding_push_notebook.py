# Cell 1: Import libraries and setup
import pandas as pd
import numpy as np
import uuid
import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the chunk data
english_chunks = pd.read_csv("/Users/ganeshnallagachu/Desktop/world_bank/chunk_data/English_chunks - tags.csv")
hindi_chunks = pd.read_csv("/Users/ganeshnallagachu/Desktop/world_bank/chunk_data/hindi_chunks - tags.csv")

print(f"English chunks: {len(english_chunks)}")
print(f"Hindi chunks: {len(hindi_chunks)}")

# Cell 2: Initialize clients
client = QdrantClient(url="http://dev.platform.farmer.chat:5438/", port=5438, grpc_port=5439, prefer_grpc=False)

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
openai_client = OpenAI(api_key=api_key)

print("Clients initialized successfully")

# Cell 3: Helper functions
def get_embedding(text, openai_client, model="text-embedding-3-large"):
    """Get embedding for text using OpenAI."""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def create_collection_if_not_exists(collection_name, vector_size=3072):
    """Create Qdrant collection if it doesn't exist."""
    try:
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {collection_name}")
        else:
            print(f"Collection {collection_name} already exists")
    except Exception as e:
        print(f"Error creating collection: {e}")

# Cell 4: Main push function
def push_chunks(collection_name, english_chunks, hindi_chunks, batch_size=100):
    """
    Push chunk data to Qdrant with metadata.
    
    Args:
        collection_name (str): Name of the Qdrant collection
        english_chunks (pd.DataFrame): English chunk data
        hindi_chunks (pd.DataFrame): Hindi chunk data
        batch_size (int): Number of points to upsert in each batch
    """
    
    # Create collection if it doesn't exist
    create_collection_if_not_exists(collection_name)
    
    # Combine both datasets
    all_chunks = []
    
    # Process English chunks
    for idx, row in english_chunks.iterrows():
        if pd.notna(row['chunk']):  # Check if chunk is not null
            all_chunks.append({
                'chunk_text': row['chunk'],
                'file_path': row['file_path'],
                'chunk_id': row['chunk_id'],
                'tags': row['tags'],
                'language': 'english',
                'file_name': row['file_name'],
                'relative_path': row['relative_path'],
                'chunk_number': row['chunk_number']
            })
    
    # Process Hindi chunks
    for idx, row in hindi_chunks.iterrows():
        if pd.notna(row['chunk']):  # Check if chunk is not null
            all_chunks.append({
                'chunk_text': row['chunk'],
                'file_path': row['file_path'],
                'chunk_id': row['chunk_id'],
                'tags': row['tags'],
                'language': 'hindi',
                'file_name': row['file_name'],
                'relative_path': row['relative_path'],
                'chunk_number': row['chunk_number']
            })
    
    print(f"Total chunks to process: {len(all_chunks)}")
    
    # Process chunks in batches
    points = []
    processed_count = 0
    
    for chunk_data in all_chunks:
        try:
            # Get embedding for the chunk text
            embedding = get_embedding(chunk_data['chunk_text'], openai_client)
            
            if embedding is None:
                print(f"Skipping chunk {chunk_data['chunk_id']} due to embedding error")
                continue
            
            # Create point structure
            point = PointStruct(
                id=int(uuid.uuid4().int & 0xFFFFFFFF),
                vector=embedding,
                payload={
                    "chunk_text": chunk_data['chunk_text'],
                    "file_path": chunk_data['file_path'],
                    "chunk_id": chunk_data['chunk_id'],
                    "tags": chunk_data['tags'],
                    "language": chunk_data['language'],
                    "file_name": chunk_data['file_name'],
                    "relative_path": chunk_data['relative_path'],
                    "chunk_number": chunk_data['chunk_number']
                }
            )
            points.append(point)
            processed_count += 1
            
            # Upsert in batches
            if len(points) >= batch_size:
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                print(f"Upserted batch of {len(points)} points. Total processed: {processed_count}")
                points = []
                
        except Exception as e:
            print(f"Error processing chunk {chunk_data['chunk_id']}: {e}")
            continue
    
    # Upsert remaining points
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Upserted final batch of {len(points)} points")
    
    print(f"Successfully processed {processed_count} chunks")
    
    # Get collection info
    collection_info = client.get_collection(collection_name)
    print(f"Collection {collection_name} now contains {collection_info.points_count} points")

# Cell 5: Test with small sample
print("Testing with a small sample...")
english_sample = english_chunks.head(5)
hindi_sample = hindi_chunks.head(5)

# Push test data
push_chunks("test_agriculture_chunks", english_sample, hindi_sample, batch_size=10)

# Cell 6: Push all data
print("Pushing all chunks to production collection...")
push_chunks("agriculture_chunks", english_chunks, hindi_chunks, batch_size=100)

# Cell 7: Verify the data
collection_info = client.get_collection("agriculture_chunks")
print(f"Collection info: {collection_info}")

# Get a sample of points to verify structure
sample_points = client.scroll(
    collection_name="agriculture_chunks",
    limit=5
)

print("\nSample points:")
for point in sample_points[0]:
    print(f"ID: {point.id}")
    print(f"Payload: {point.payload}")
    print(f"Vector length: {len(point.vector)}")
    print("---") 