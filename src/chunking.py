import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SemanticChunking:
    def __init__(self):
        # Get API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        # Initialize OpenAI client
        self.ai_client = OpenAI(api_key=api_key)
        
        # Maximum characters per sentence before embedding
        self.max_sentence_length = 500

    def calculate_cosine_distances(self, sentences: list) -> tuple[list, list]:
        n = len(sentences)
        distances = np.zeros(n - 1)
        for i in range(n - 1):
            sim = cosine_similarity(
                [sentences[i]['combined_sentence_embedding']],
                [sentences[i + 1]['combined_sentence_embedding']]
            )[0, 0]
            distance = 1 - sim
            distances[i] = distance
            sentences[i]['distance_to_next'] = distance
        return distances.tolist(), sentences

    def chunking_dp(self, distances: list, word_length=None, limit_per_chunk=10) -> list:
        if limit_per_chunk <= 0:
            raise ValueError("sentence_limit_per_chunk must be positive")
        if word_length is None:
            word_length = np.ones(len(distances) + 1, dtype=np.int32)
        else:
            word_length = np.array(word_length, dtype=np.int32)
        n = len(word_length)
        split_idx = np.full(n, n, dtype=np.int32)
        split_cost = np.full(n, np.inf)
        split_cost[-1] = 0

        for curr_idx in reversed(range(n - 1)):
            total_words = 0
            for i in range(curr_idx + 1, n):
                total_words += word_length[i]
                if total_words > limit_per_chunk:
                    break
                cost = split_cost[i] + distances[i - 1]
                if cost < split_cost[curr_idx]:
                    split_cost[curr_idx] = cost
                    split_idx[curr_idx] = i

        splits = []
        i = 0
        while i < n:
            splits.append(i)
            i = split_idx[i]
        splits.append(n)
        return splits
    
    def combine_sentences(self, sentences, buffer_size=1):
        n = len(sentences)
        for i in range(n):
            buffer = []
            if i - buffer_size >= 0:
                buffer.append(sentences[i - buffer_size]['sentence'])
            buffer.append(sentences[i]['sentence'])
            if i + buffer_size < n:
                buffer.append(sentences[i + buffer_size]['sentence'])
            sentences[i]['combined_sentence'] = ' '.join(buffer)
        return sentences

    def get_embedding_list(self, text_list):
        """
        Get embeddings for a list of text using OpenAI's API.
        Returns a list of embeddings.
        """
        results = []
        for text in text_list:
            try:
                response = self.ai_client.embeddings.create(
                    input=text,
                    model="text-embedding-ada-002"  # Use the appropriate model
                )
                results.append(response)
            except Exception as e:
                print(f"Error getting embedding: {e}")
                continue
        return results

    def semantic_grouping(self, text, max_char: int = 2000) -> list:
        """
        Process a text string into semantic chunks
        
        Parameters:
        - text: String containing the text to be processed
        - max_char: Maximum characters per chunk
        
        Returns:
        - List of dictionaries containing chunked text
        """
        if not text or not isinstance(text, str):
            return []

        # Split text into sentences and handle long sentences
        sentences = []
        for sentence in re.split(r'(?<=[.?!])\s+', text):
            if not sentence:
                continue
                
            # If sentence is too long, split it on common delimiters
            if len(sentence) > self.max_sentence_length:
                # Try to split on delimiters
                for delimiter in [';', ':', ',', ' - ']:
                    if delimiter in sentence:
                        parts = sentence.split(delimiter)
                        sentences.extend(part.strip() for part in parts if part.strip())
                        break
                else:
                    # If no delimiter found, split on spaces
                    words = sentence.split()
                    current = []
                    current_len = 0
                    for word in words:
                        if current_len + len(word) + 1 > self.max_sentence_length:
                            if current:
                                sentences.append(' '.join(current))
                            current = [word]
                            current_len = len(word)
                        else:
                            current.append(word)
                            current_len += len(word) + 1
                    if current:
                        sentences.append(' '.join(current))
            else:
                sentences.append(sentence)

        sentence_dicts = [{'sentence': s, 'index': i} for i, s in enumerate(sentences) if s]

        if not sentence_dicts:
            return []

        # Combine sentences with surrounding context
        sentence_dicts = self.combine_sentences(sentence_dicts)
        
        # Get embeddings for combined sentences
        embeddings = self.get_embedding_list(
            [x['combined_sentence'] for x in sentence_dicts]
        )

        if not embeddings:
            return []

        # Process embeddings
        for i, embedding_response in enumerate(embeddings):
            if hasattr(embedding_response, 'data') and embedding_response.data:
                sentence_dicts[i]['combined_sentence_embedding'] = embedding_response.data[0].embedding

        # Calculate distances between embeddings
        distances, sentence_dicts = self.calculate_cosine_distances(sentence_dicts)
        
        # Calculate word lengths for each sentence
        word_length = [len(s['sentence']) for s in sentence_dicts]
        
        # Get chunk indices using dynamic programming
        chunk_indices = self.chunking_dp(distances, word_length, max(max_char, max(word_length) if word_length else max_char))
        chunk_indices = chunk_indices[1:-1]  # Remove first and last indices

        # Create chunks from sentences
        chunks = []
        start_idx = 0
        for idx in chunk_indices:
            chunk_sentences = sentence_dicts[start_idx:idx + 1]
            text_chunk = ' '.join(s['sentence'] for s in chunk_sentences)
            chunks.append({'text': text_chunk})
            start_idx = idx + 1

        # Handle remaining sentences
        if start_idx < len(sentence_dicts):
            text_chunk = ' '.join(s['sentence'] for s in sentence_dicts[start_idx:])
            chunks.append({'text': text_chunk})

        return chunks