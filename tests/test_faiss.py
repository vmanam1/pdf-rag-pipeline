import os
import json
import numpy as np
import faiss

# File paths
input_json_file = "json_vectorization/output_data.json"
faiss_index_file = "faiss_index.idx"
mapping_file = "embeddings_with_vectors.json"

# Load data
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Create FAISS index
def create_faiss_index(embedding_dim):
    index = faiss.IndexFlatL2(embedding_dim)
    return index

# Add embeddings to FAISS index
def add_embeddings_to_index(data, index):
    paper_map = {}
    id_counter = 0
    for paper_id, sections in data.items():
        for section_name, embeddings in sections.items():
            if isinstance(embeddings, list):
                vector = np.array(embeddings, dtype='float32')
                index.add(np.expand_dims(vector, axis=0))
                paper_map[id_counter] = {
                    "paper_id": paper_id,
                    "section_name": section_name,
                    "vector": vector.tolist()
                }
                id_counter += 1
    return paper_map

# Save mapping to file
def save_mapping_to_file(paper_map, output_file):
    with open(output_file, 'w') as f:
        json.dump(paper_map, f, indent=4)
    print(f"Mapping with vectors saved to {output_file}")

# Test workflow
def test_faiss_index():
    print("Starting test workflow...")
    
    # Load JSON data
    if not os.path.exists(input_json_file):
        raise FileNotFoundError(f"Input file {input_json_file} not found. Ensure the file exists.")

    data = load_data(input_json_file)
    print(f"Loaded data from {input_json_file}.")

    # Assume embedding size from the first entry
    first_embedding = next(iter(next(iter(data.values())).values()))
    embedding_dim = len(first_embedding)

    # Create FAISS index
    faiss_index = create_faiss_index(embedding_dim)
    print(f"Created FAISS index with embedding dimension {embedding_dim}.")

    # Populate FAISS index and create mapping
    paper_map = add_embeddings_to_index(data, faiss_index)
    print(f"Added {len(paper_map)} embeddings to the FAISS index.")

    # Save FAISS index
    faiss.write_index(faiss_index, faiss_index_file)
    print(f"FAISS index saved to {faiss_index_file}.")

    # Save mapping
    save_mapping_to_file(paper_map, mapping_file)

    # Validate the output
    print("\nValidating output files...")
    assert os.path.exists(faiss_index_file), "FAISS index file was not created."
    assert os.path.exists(mapping_file), "Mapping file was not created."

    # Verify FAISS index
    index = faiss.read_index(faiss_index_file)
    assert index.ntotal == len(paper_map), "FAISS index does not match the number of embeddings."

    # Verify JSON mapping
    with open(mapping_file, 'r') as f:
        loaded_map = json.load(f)
    assert len(loaded_map) == len(paper_map), "Mapping file content mismatch."

    print("All tests passed successfully!")

# Run the test
if __name__ == "__main__":
    test_faiss_index()
