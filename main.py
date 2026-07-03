import json
from pathlib import Path

import numpy as np
import faiss

BASE_DIR = Path(__file__).resolve().parent

# File paths
json_file = BASE_DIR / "json_vectorization" / "output_data.json"
faiss_index_file = BASE_DIR / "faiss_storage" / "faiss_index.idx"
mapping_file = BASE_DIR / "faiss_storage" / "embeddings_with_vectors.json"

# Load data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Create FAISS index
def create_faiss_index(embedding_dim):
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity search
    return index

# Add embeddings to FAISS index and save mappings with vectors
def add_embeddings_to_index(data, index):
    paper_map = {}
    id_counter = 0
    
    for paper_id, sections in data.items():
        for section_name, embeddings in sections.items():
            if isinstance(embeddings, list):  # Ensure embeddings are numeric arrays
                vector = np.array(embeddings, dtype='float32')  # FAISS requires float32
                index.add(np.expand_dims(vector, axis=0))  # Add to index
                paper_map[id_counter] = {
                    "paper_id": paper_id,
                    "section_name": section_name,
                    "vector": vector.tolist()  # Save embedding as a list for JSON
                }
                id_counter += 1
    return paper_map

# Save mapping to a JSON file
def save_mapping_to_file(paper_map, output_file):
    with open(output_file, 'w') as f:
        json.dump(paper_map, f, indent=4)
    print(f"Embedding-to-paper mapping with vectors saved to {output_file}")

# Main workflow
if __name__ == "__main__":
    # Load JSON data
    data = load_data(json_file)
    
    # Assume embedding size from the first entry
    first_embedding = next(iter(next(iter(data.values())).values()))
    embedding_dim = len(first_embedding)
    
    # Create FAISS index
    faiss_index = create_faiss_index(embedding_dim)
    
    # Populate FAISS index and create mapping with vectors
    paper_map = add_embeddings_to_index(data, faiss_index)
    
    # Save the FAISS index for future use
    faiss.write_index(faiss_index, str(faiss_index_file))
    print(f"FAISS index saved to {faiss_index_file}")
    
    # Save embedding-to-paper mapping with vectors
    save_mapping_to_file(paper_map, mapping_file)
