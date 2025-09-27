import os
from dotenv import load_dotenv
from pinecone import Pinecone
import fitz  # PyMuPDF
from google import genai

# Load environment variables
load_dotenv()

# Initialize clients
pine_api = os.getenv("API_KEY_PINE")
google_api = os.getenv("API_KEY_GOOGLE")

if not pine_api or not google_api:
    raise ValueError("Missing API keys. Check your .env file.")

pine_client = Pinecone(api_key=pine_api)
gemini_client = genai.Client(api_key=google_api)

# Pinecone index
vector_index = pine_client.Index("health-assistant")

# -------------------------
# PDF to text extraction
# -------------------------
def extract_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

# -------------------------
# Chunking function
# -------------------------
def chunk_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# -------------------------
# Embedding with Gemini
# -------------------------
def embed_text(text):
    try:
        response = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config={"output_dimensionality": 768}
        )
        if hasattr(response, "embedding"):
            vector = response.embedding.values
        elif hasattr(response, "embeddings"):
            vector = response.embeddings[0].values
        else:
            raise ValueError("Unexpected response format from Gemini API")
        return vector
    except Exception as e:
        print(f"Embedding failed: {e}")
        return None

# -------------------------
# Upsert to Pinecone
# -------------------------
def upsert_vectors_to_pinecone(docs, batch_size=100):
    upsert_batch = []
    for vector_id, vector, metadata in docs:
        if vector is None:
            continue
        upsert_batch.append({
            "id": vector_id,
            "values": vector,
            "metadata": metadata   
        })
        if len(upsert_batch) >= batch_size:
            vector_index.upsert(vectors=upsert_batch)
            upsert_batch.clear()
    if upsert_batch:
        vector_index.upsert(vectors=upsert_batch)
    print("✅ Vectors upserted successfully")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    pdf_dir = "documents"
    document_files = os.listdir(pdf_dir)

    all_vectors = []
    for file_name in document_files:
        file_path = os.path.join(pdf_dir, file_name)
        print(f"Processing: {file_name}")

        text = extract_text(file_path)
        if not text:
            print(f"⚠️ Skipping {file_name}, no text found.")
            continue

        chunks = chunk_text(text)
        print(f"    Extracted {len(chunks)} chunks.")

        
        patient_name = os.path.splitext(file_name)[0]

        for i, chunk in enumerate(chunks):
            vector = embed_text(chunk)
            vector_id = f"{file_name}-chunk-{i}"

            metadata = {
                "source": file_name,
                "chunk": i,
                "text": chunk[:500],     
                "patient_name": patient_name 
            }
            all_vectors.append((vector_id, vector, metadata))

    upsert_vectors_to_pinecone(all_vectors)
    print("All documents processed and vectors created.")
