# Script to retrieve 10 similar chunks for each chunk from Qdrant
import pandas as pd
import numpy as np
import json
from qdrant_client import QdrantClient
from openai import OpenAI
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Load environment variables
load_dotenv()

# Initialize clients
client = QdrantClient(url="http://dev.platform.farmer.chat:5438/", port=5438, grpc_port=5439, prefer_grpc=False)

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
openai_client = OpenAI(api_key=api_key)

# Thread-safe lock for file operations
file_lock = Lock()

print("Clients initialized successfully")

# Helper function to detect embedding model based on collection vector size
def get_embedding_model_for_collection(collection_name, client):
    """
    Detect the embedding model based on the collection's vector size.
    
    Args:
        collection_name (str): Name of the Qdrant collection
        client: QdrantClient instance
    
    Returns:
        str: Embedding model name
    """
    try:
        collection_info = client.get_collection(collection_name)
        vector_size = collection_info.config.params.vectors.size
        
        # Map vector size to embedding model
        if vector_size == 1536:
            return "text-embedding-3-small"
        elif vector_size == 3072:
            return "text-embedding-3-large"
        elif vector_size == 512:
            return "text-embedding-ada-002"  # Legacy model
        else:
            print(f"Warning: Unknown vector size {vector_size}, defaulting to text-embedding-3-small")
            return "text-embedding-3-small"
    except Exception as e:
        print(f"Error detecting collection vector size: {e}")
        print("Defaulting to text-embedding-3-small")
        return "text-embedding-3-small"

# Helper function to get embedding
def get_embedding(text, openai_client, model="text-embedding-3-small"):
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

def retrieve_similar_chunks(collection_name, chunk_text, chunk_id, language, embedding_model, qdrant_client=None, openai_client_instance=None, top_k=10, similarity_threshold=None):
    """
    Retrieve similar chunks from Qdrant for a given chunk.
    
    Args:
        collection_name (str): Name of the Qdrant collection
        chunk_text (str): Text of the chunk to find similar chunks for
        chunk_id (str): ID of the current chunk (to exclude from results)
        language (str): Language of the chunk ('english' or 'hindi')
        embedding_model (str): Embedding model to use
        top_k (int): Number of similar chunks to retrieve
        similarity_threshold (float, optional): Minimum cosine similarity score (0-1).
            Only chunks with similarity >= threshold will be returned.
            If None, no threshold filtering is applied.
            Typical values:
            - 0.8-0.9: Very high similarity (nearly identical)
            - 0.7-0.8: High similarity (very similar)
            - 0.6-0.7: Moderate similarity (somewhat similar)
            - 0.5-0.6: Low similarity (loosely related)
            - <0.5: Very low similarity (may not be relevant)
    
    Returns:
        list: List of similar chunks with their metadata
    """
    try:
        # Use provided clients or fall back to global clients
        qdrant_cl = qdrant_client if qdrant_client is not None else client
        openai_cl = openai_client_instance if openai_client_instance is not None else openai_client
        
        # Get embedding for the chunk
        embedding = get_embedding(chunk_text, openai_cl, model=embedding_model)
        
        if embedding is None:
            return []
        
        # Search for similar chunks in Qdrant
        # Search for more results to account for excluding the current chunk and threshold filtering
        # We'll search for top_k*3 to ensure we get enough results after filtering
        search_results = qdrant_cl.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=top_k * 3 if similarity_threshold else top_k * 2,  # Get more if filtering by threshold
            query_filter={
                "must": [
                    {
                        "key": "language",
                        "match": {"value": language}
                    }
                ]
            }
        )
        
        # Filter out the current chunk and apply similarity threshold
        similar_chunks = []
        for result in search_results:
            # Check if this is not the current chunk
            result_chunk_id = result.payload.get('chunk_id')
            if result_chunk_id != chunk_id:
                similarity_score = float(result.score)
                
                # Apply similarity threshold if specified
                if similarity_threshold is not None and similarity_score < similarity_threshold:
                    continue  # Skip chunks below threshold
                
                similar_chunks.append({
                    'similar_chunk_id': result_chunk_id,
                    'similar_chunk_text': result.payload.get('chunk_text', ''),
                    'similarity_score': similarity_score,
                    'file_path': result.payload.get('file_path', ''),
                    'file_name': result.payload.get('file_name', ''),
                    'tags': result.payload.get('tags', ''),
                    'chunk_number': result.payload.get('chunk_number', '')
                })
            
            # Stop when we have enough results
            if len(similar_chunks) >= top_k:
                break
        
        return similar_chunks
        
    except Exception as e:
        print(f"Error retrieving similar chunks for chunk_id {chunk_id}: {e}")
        return []

def process_single_chunk(args):
    """
    Process a single chunk - worker function for parallel processing.
    
    Args:
        args: Tuple containing (collection_name, chunk_data, language, embedding_model, similarity_threshold, qdrant_client, openai_client)
    
    Returns:
        dict: Result entry for the chunk
    """
    collection_name, chunk_data, qdrant_language, embedding_model, similarity_threshold, qdrant_client, openai_client = args
    
    chunk_id = chunk_data['chunk_id']
    chunk_text = chunk_data['chunk_text']
    
    try:
        # Retrieve similar chunks
        similar_chunks = retrieve_similar_chunks(
            collection_name=collection_name,
            chunk_text=chunk_text,
            chunk_id=chunk_id,
            language=qdrant_language,  # Use capitalized language for Qdrant search
            embedding_model=embedding_model,
            qdrant_client=qdrant_client,
            openai_client_instance=openai_client,
            top_k=10,
            similarity_threshold=similarity_threshold
        )
        
        # Store results (use lowercase for consistency in output)
        result_entry = {
            'chunk_id': chunk_id,
            'chunk_text': chunk_text,
            'language': qdrant_language.lower(),  # Store lowercase in output for consistency
            'file_path': chunk_data.get('file_path', ''),
            'file_name': chunk_data.get('file_name', ''),
            'chunk_number': chunk_data.get('chunk_number', ''),
            'tags': chunk_data.get('tags', ''),
            'num_similar_chunks_found': len(similar_chunks),
            'similar_chunks': similar_chunks
        }
        
        return result_entry
    except Exception as e:
        print(f"\nError processing chunk {chunk_id}: {e}")
        return {
            'chunk_id': chunk_id,
            'chunk_text': chunk_text,
            'language': qdrant_language.lower() if qdrant_language else 'unknown',
            'file_path': chunk_data.get('file_path', ''),
            'file_name': chunk_data.get('file_name', ''),
            'chunk_number': chunk_data.get('chunk_number', ''),
            'tags': chunk_data.get('tags', ''),
            'num_similar_chunks_found': 0,
            'similar_chunks': [],
            'error': str(e)
        }

def process_chunks_for_similarity(collection_name, chunks_df, language, output_file, embedding_model, batch_size=50, similarity_threshold=None, max_workers=10):
    """
    Process chunks and retrieve similar chunks for each using parallel processing.
    
    Args:
        collection_name (str): Name of the Qdrant collection
        chunks_df (pd.DataFrame): DataFrame containing chunks
        language (str): Language of chunks ('english' or 'hindi') - will be capitalized for Qdrant search
        output_file (str): Path to output file to save results
        embedding_model (str): Embedding model to use
        batch_size (int): Number of chunks to process before saving intermediate results
        similarity_threshold (float, optional): Minimum cosine similarity score (0-1)
        max_workers (int): Maximum number of parallel workers (default: 10)
    """
    # Convert language to capitalized form for Qdrant search (stored as "English"/"Hindi")
    qdrant_language = language.capitalize() if language else language
    
    # Prepare chunk data for processing
    chunk_data_list = []
    for idx, row in chunks_df.iterrows():
        if pd.notna(row['chunk']):
            chunk_data_list.append({
                'chunk_id': row['chunk_id'],
                'chunk_text': row['chunk'],
                'file_path': row.get('file_path', ''),
                'file_name': row.get('file_name', ''),
                'chunk_number': row.get('chunk_number', ''),
                'tags': row.get('tags', '')
            })
    
    total_chunks = len(chunk_data_list)
    results = []
    
    print(f"\nProcessing {total_chunks} {language.lower()} chunks...")
    print(f"Using embedding model: {embedding_model}")
    print(f"Parallel workers: {max_workers}")
    print(f"Qdrant language filter: '{qdrant_language}' (capitalized)")
    if similarity_threshold is not None:
        print(f"Similarity threshold: {similarity_threshold} (only chunks with similarity >= {similarity_threshold} will be included)")
    else:
        print("No similarity threshold applied (all top-k results included)")
    
    # Create client instances for each worker thread (thread-safe)
    # QdrantClient and OpenAI are generally thread-safe, but creating separate instances
    # per worker ensures better isolation and avoids any potential issues
    worker_clients = {}
    def get_worker_clients(worker_id):
        """Get or create client instances for a worker thread."""
        if worker_id not in worker_clients:
            qdrant_cl = QdrantClient(url="http://dev.platform.farmer.chat:5438/", port=5438, grpc_port=5439, prefer_grpc=False)
            openai_cl = OpenAI(api_key=api_key)
            worker_clients[worker_id] = (qdrant_cl, openai_cl)
        return worker_clients[worker_id]
    
    # Prepare arguments for parallel processing
    # Use capitalized language for Qdrant search
    process_args = [
        (collection_name, chunk_data, qdrant_language, embedding_model, similarity_threshold)
        for chunk_data in chunk_data_list
    ]
    
    # Process chunks in parallel
    processed_count = 0
    worker_counter = [0]  # Use list to make it mutable in nested function
    
    def process_with_worker_clients(args):
        """Wrapper to assign clients to workers."""
        import threading
        worker_id = threading.current_thread().ident
        qdrant_cl, openai_cl = get_worker_clients(worker_id)
        collection_name, chunk_data, qdrant_lang, embedding_model, similarity_threshold = args
        return process_single_chunk((
            collection_name, chunk_data, qdrant_lang, embedding_model, 
            similarity_threshold, qdrant_cl, openai_cl
        ))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_chunk = {}
        for args in process_args:
            chunk_id = args[1]['chunk_id']
            future = executor.submit(process_with_worker_clients, args)
            future_to_chunk[future] = chunk_id
        
        # Process completed tasks with progress bar
        with tqdm(total=total_chunks, desc=f"Processing {language} chunks") as pbar:
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    result_entry = future.result()
                    results.append(result_entry)
                    processed_count += 1
                    pbar.update(1)
                    
                    # Save intermediate results periodically (thread-safe)
                    if processed_count % batch_size == 0:
                        with file_lock:
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(results, f, ensure_ascii=False, indent=2)
                            print(f"\nSaved intermediate results: {processed_count}/{total_chunks} chunks processed")
                            
                except Exception as e:
                    print(f"\nError processing chunk {chunk_id}: {e}")
                    pbar.update(1)
    
    # Sort results by chunk_id to maintain consistency
    results.sort(key=lambda x: x['chunk_id'])
    
    # Final save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nCompleted processing {len(results)} {language} chunks")
    print(f"Results saved to: {output_file}")
    
    return results

def create_summary_dataframe(results, language):
    """
    Create a summary DataFrame with chunk_id and similar chunk_ids.
    
    Args:
        results (list): List of result dictionaries
        language (str): Language of chunks
    
    Returns:
        pd.DataFrame: Summary DataFrame
    """
    summary_data = []
    
    for result in results:
        chunk_id = result['chunk_id']
        similar_chunks = result['similar_chunks']
        
        # Create entries for each similar chunk
        for i, similar in enumerate(similar_chunks, 1):
            summary_data.append({
                'chunk_id': chunk_id,
                'rank': i,
                'similar_chunk_id': similar['similar_chunk_id'],
                'similarity_score': similar['similarity_score'],
                'similar_chunk_text': similar['similar_chunk_text'][:200] + '...' if len(similar['similar_chunk_text']) > 200 else similar['similar_chunk_text'],
                'similar_file_path': similar.get('file_path', ''),
                'similar_file_name': similar.get('file_name', ''),
                'language': language
            })
    
    return pd.DataFrame(summary_data)

def create_tracking_dataframe(results, language):
    """
    Create a comprehensive tracking DataFrame with all chunk information.
    
    Args:
        results (list): List of result dictionaries
        language (str): Language of chunks
    
    Returns:
        pd.DataFrame: Tracking DataFrame with chunk_id and all similar chunk_ids in one row
    """
    tracking_data = []
    
    for result in results:
        chunk_id = result['chunk_id']
        similar_chunks = result['similar_chunks']
        
        # Create a row with chunk_id and all similar chunk_ids
        similar_chunk_ids = [chunk['similar_chunk_id'] for chunk in similar_chunks]
        similarity_scores = [chunk['similarity_score'] for chunk in similar_chunks]
        
        # Pad lists to ensure consistent column count
        while len(similar_chunk_ids) < 10:
            similar_chunk_ids.append('')
            similarity_scores.append('')
        
        row = {
            'chunk_id': chunk_id,
            'chunk_text': result['chunk_text'][:200] + '...' if len(result['chunk_text']) > 200 else result['chunk_text'],
            'file_path': result.get('file_path', ''),
            'file_name': result.get('file_name', ''),
            'language': language,
            'num_similar_chunks_found': result['num_similar_chunks_found']
        }
        
        # Add similar chunk IDs and scores
        for i in range(10):
            row[f'similar_chunk_id_{i+1}'] = similar_chunk_ids[i] if i < len(similar_chunk_ids) else ''
            row[f'similarity_score_{i+1}'] = similarity_scores[i] if i < len(similarity_scores) else ''
        
        tracking_data.append(row)
    
    return pd.DataFrame(tracking_data)

    # Main execution
if __name__ == "__main__":
    # Collection name (adjust if needed)
    collection_name = "test_agriculture_chunks"
    
    # Similarity threshold (optional)
    # Set to None to include all top-k results regardless of similarity score
    # Set to a value between 0 and 1 to filter results by minimum similarity
    # Recommended values:
    #   - None: No filtering (current behavior)
    #   - 0.7: High similarity only (very similar chunks)
    #   - 0.6: Moderate-high similarity
    #   - 0.5: Moderate similarity
    similarity_threshold = None  # Change this to filter by similarity score
    
    # Parallel processing configuration
    # Number of parallel workers (threads) to use
    # Increase for faster processing, but be mindful of API rate limits
    # Recommended: 10-20 for OpenAI API, adjust based on your rate limits
    max_workers = 40  # Adjust based on your API rate limits and system capacity
    
    # File paths (using correct workspace paths)
    english_csv_path = "/Users/ganeshnallagachu/Desktop/RAFT_exp/chunk_data/English_chunks - tags.csv"
    hindi_csv_path = "/Users/ganeshnallagachu/Desktop/RAFT_exp/chunk_data/hindi_chunks - tags.csv"
    
    # Output file paths
    english_output_json = "/Users/ganeshnallagachu/Desktop/RAFT_exp/english_similar_chunks_results.json"
    hindi_output_json = "/Users/ganeshnallagachu/Desktop/RAFT_exp/hindi_similar_chunks_results.json"
    english_output_csv = "/Users/ganeshnallagachu/Desktop/RAFT_exp/english_similar_chunks_summary.csv"
    hindi_output_csv = "/Users/ganeshnallagachu/Desktop/RAFT_exp/hindi_similar_chunks_summary.csv"
    english_tracking_csv = "/Users/ganeshnallagachu/Desktop/RAFT_exp/english_similar_chunks_tracking.csv"
    hindi_tracking_csv = "/Users/ganeshnallagachu/Desktop/RAFT_exp/hindi_similar_chunks_tracking.csv"
    
    # Load chunks
    print("Loading chunk data...")
    english_chunks = pd.read_csv(english_csv_path)
    hindi_chunks = pd.read_csv(hindi_csv_path)
    
    print(f"English chunks: {len(english_chunks)}")
    print(f"Hindi chunks: {len(hindi_chunks)}")
    
    # Verify collection exists and detect embedding model
    try:
        collection_info = client.get_collection(collection_name)
        vector_size = collection_info.config.params.vectors.size
        print(f"\nCollection '{collection_name}' found with {collection_info.points_count} points")
        print(f"Collection vector size: {vector_size}")
        
        # Detect the correct embedding model based on vector size
        embedding_model = get_embedding_model_for_collection(collection_name, client)
        print(f"Using embedding model: {embedding_model}")
        
    except Exception as e:
        print(f"\nError accessing collection '{collection_name}': {e}")
        print("Please ensure the collection exists and is accessible.")
        exit(1)
    
    # Process English chunks
    print("\n" + "="*50)
    print("Processing English chunks...")
    print("="*50)
    english_results = process_chunks_for_similarity(
        collection_name=collection_name,
        chunks_df=english_chunks,
        language='english',
        output_file=english_output_json,
        embedding_model=embedding_model,
        batch_size=50,
        similarity_threshold=similarity_threshold,
        max_workers=max_workers
    )
    
    # Create summary CSV for English
    english_summary_df = create_summary_dataframe(english_results, 'english')
    english_summary_df.to_csv(english_output_csv, index=False)
    print(f"\nEnglish summary saved to: {english_output_csv}")
    print(f"English summary shape: {english_summary_df.shape}")
    
    # Create tracking CSV for English
    english_tracking_df = create_tracking_dataframe(english_results, 'english')
    english_tracking_df.to_csv(english_tracking_csv, index=False)
    print(f"English tracking saved to: {english_tracking_csv}")
    print(f"English tracking shape: {english_tracking_df.shape}")
    
    # Process Hindi chunks
    print("\n" + "="*50)
    print("Processing Hindi chunks...")
    print("="*50)
    hindi_results = process_chunks_for_similarity(
        collection_name=collection_name,
        chunks_df=hindi_chunks,
        language='hindi',
        output_file=hindi_output_json,
        embedding_model=embedding_model,
        batch_size=50,
        similarity_threshold=similarity_threshold,
        max_workers=max_workers
    )
    
    # Create summary CSV for Hindi
    hindi_summary_df = create_summary_dataframe(hindi_results, 'hindi')
    hindi_summary_df.to_csv(hindi_output_csv, index=False)
    print(f"\nHindi summary saved to: {hindi_output_csv}")
    print(f"Hindi summary shape: {hindi_summary_df.shape}")
    
    # Create tracking CSV for Hindi
    hindi_tracking_df = create_tracking_dataframe(hindi_results, 'hindi')
    hindi_tracking_df.to_csv(hindi_tracking_csv, index=False)
    print(f"Hindi tracking saved to: {hindi_tracking_csv}")
    print(f"Hindi tracking shape: {hindi_tracking_df.shape}")
    
    # Print statistics
    print("\n" + "="*50)
    print("Summary Statistics")
    print("="*50)
    print(f"\nEnglish chunks processed: {len(english_results)}")
    print(f"English chunks with similar chunks found: {sum(1 for r in english_results if r['num_similar_chunks_found'] > 0)}")
    print(f"Average similar chunks per English chunk: {np.mean([r['num_similar_chunks_found'] for r in english_results]):.2f}")
    
    # Calculate similarity score statistics for English
    english_scores = []
    for r in english_results:
        for similar in r['similar_chunks']:
            english_scores.append(similar['similarity_score'])
    if english_scores:
        print(f"\nEnglish similarity score statistics:")
        print(f"  Min: {np.min(english_scores):.4f}")
        print(f"  Max: {np.max(english_scores):.4f}")
        print(f"  Mean: {np.mean(english_scores):.4f}")
        print(f"  Median: {np.median(english_scores):.4f}")
        print(f"  Std Dev: {np.std(english_scores):.4f}")
    
    print(f"\nHindi chunks processed: {len(hindi_results)}")
    print(f"Hindi chunks with similar chunks found: {sum(1 for r in hindi_results if r['num_similar_chunks_found'] > 0)}")
    print(f"Average similar chunks per Hindi chunk: {np.mean([r['num_similar_chunks_found'] for r in hindi_results]):.2f}")
    
    # Calculate similarity score statistics for Hindi
    hindi_scores = []
    for r in hindi_results:
        for similar in r['similar_chunks']:
            hindi_scores.append(similar['similarity_score'])
    if hindi_scores:
        print(f"\nHindi similarity score statistics:")
        print(f"  Min: {np.min(hindi_scores):.4f}")
        print(f"  Max: {np.max(hindi_scores):.4f}")
        print(f"  Mean: {np.mean(hindi_scores):.4f}")
        print(f"  Median: {np.median(hindi_scores):.4f}")
        print(f"  Std Dev: {np.std(hindi_scores):.4f}")
    
    print("\nProcessing complete!")

