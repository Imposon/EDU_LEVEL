from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import uuid
import tempfile
import json
import shutil
from typing import List, Dict, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from PIL import Image
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Tutor Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
DIMENSION = 384
TOP_K_TEXT = 5

# Global storage (in production, use database)
topics_storage = {}
text_indexes = {}
image_indexes = {}

# Initialize embedding model
@dataclass
class TopicData:
    id: str
    text_chunks: List[str]
    text_index: faiss.IndexFlatIP
    image_metadata: Dict[str, Dict]
    image_index: faiss.IndexFlatIP
    image_paths: Dict[str, str]

# Initialize embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def create_text_index(chunks: List[str]) -> faiss.IndexFlatIP:
    """Create FAISS index for text chunks."""
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    index = faiss.IndexFlatIP(DIMENSION)
    index.add(embeddings.astype(np.float32))
    
    return index

def process_image(image_file: UploadFile, temp_dir: str) -> Dict:
    """Process single image and generate metadata."""
    try:
        # Save image
        image_id = str(uuid.uuid4())
        image_path = os.path.join(temp_dir, f"{image_id}.png")
        
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image_file.file, f)
        
        # Generate metadata from filename
        filename = image_file.filename
        title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
        description = f"Educational diagram showing {title.lower()}"
        
        # Extract keywords from filename
        words = title.lower().replace("_", " ").replace("-", " ").split()
        keywords = [w.strip(".,!?;:") for w in words if len(w) > 2 and w not in ["the", "and", "of", "in", "on", "at", "to", "for"]]
        keywords = list(set(keywords))[:8]
        
        metadata = {
            "id": image_id,
            "filename": filename,
            "title": title,
            "description": description,
            "keywords": keywords,
            "path": image_path
        }
        
        return metadata
        
    except Exception as e:
        print(f"Error processing image {image_file.filename}: {e}")
        return None

def create_image_index(image_data: List[Dict]) -> faiss.IndexFlatIP:
    """Create FAISS index for images."""
    embedding_inputs = []
    
    for img_data in image_data:
        if img_data:
            embedding_input = f"{img_data['title']} {img_data['description']} {' '.join(img_data['keywords'])}"
            embedding_inputs.append(embedding_input)
    
    if embedding_inputs:
        embeddings = embedding_model.encode(embedding_inputs, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        index = faiss.IndexFlatIP(DIMENSION)
        index.add(embeddings.astype(np.float32))
        
        return index
    
    return None

@app.post("/upload")
async def upload_pdf_and_images(pdf: UploadFile = File(...), images: List[UploadFile] = File([])):
    """Upload PDF and images, extract text and create embeddings."""
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Process PDF
        topic_id = str(uuid.uuid4())
        pdf_path = os.path.join(temp_dir, pdf.filename)
        
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(pdf.file, f)
        
        # Extract and chunk text
        text = extract_text_from_pdf(pdf_path)
        text_chunks = chunk_text(text)
        
        # Create text index
        text_index = create_text_index(text_chunks)
        
        # Process images
        image_data = []
        for image_file in images:
            img_metadata = process_image(image_file, temp_dir)
            if img_metadata:
                image_data.append(img_metadata)
        
        # Create image index
        image_index = create_image_index(image_data)
        
        # Store topic data
        topic_data = {
            "id": topic_id,
            "text_chunks": text_chunks,
            "text_index": text_index,
            "image_metadata": {img["id"]: img for img in image_data if img},
            "image_index": image_index,
            "image_paths": {img["id"]: img["path"] for img in image_data if img}
        }
        
        topics_storage[topic_id] = topic_data
        text_indexes[topic_id] = text_index
        image_indexes[topic_id] = image_index
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        return JSONResponse(content={
            "topicId": topic_id,
            "message": "Successfully processed PDF and images",
            "chunksCount": len(text_chunks),
            "imagesCount": len(image_data)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/chat")
async def chat(request: dict):
    """Chat endpoint - retrieve chunks and generate answer."""
    try:
        topic_id = request.get("topicId")
        query = request.get("query")
        
        if not topic_id or not query:
            raise HTTPException(status_code=400, detail="Missing topicId or query")
        
        if topic_id not in topics_storage:
            raise HTTPException(status_code=404, detail="Topic not found")
        
        topic_data = topics_storage[topic_id]
        text_index = topic_data["text_index"]
        text_chunks = topic_data["text_chunks"]
        
        # Retrieve relevant chunks
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        scores, indices = text_index.search(query_embedding.astype(np.float32), TOP_K_TEXT)
        
        retrieved_chunks = []
        for idx in indices[0]:
            if idx >= 0 and idx < len(text_chunks):
                retrieved_chunks.append(text_chunks[idx])
        
        # Limit context length to avoid API limits (max ~2000 chars)
        context = "\n\n".join(retrieved_chunks[:2])  # Use only top 2 most relevant chunks
        if len(context) > 1500:
            context = context[:1500] + "\n\n[Context truncated for API limits...]"
        
        # Generate answer using Groq API
        context = "\n\n".join(retrieved_chunks)
        
        # Get Groq API key from environment
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        
        if not groq_api_key:
            # Fallback to mock response
            if retrieved_chunks:
                answer = f"""Based on the provided context:

{retrieved_chunks[0][:500] if retrieved_chunks else "No information found."}

{'Additional context: ' + retrieved_chunks[1][:300] if len(retrieved_chunks) > 1 else ''}

[This is a demo response. Integrate with actual LLM for production.]"""
            else:
                answer = "No relevant information found in the document."
        else:
            # Call Groq API
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                
                headers = {
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                }
                
                system_prompt = """You are an AI tutor. Answer ONLY using the provided context. If the information is not found in the context, say 'Not found in document'. Be concise and helpful."""
                
                user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""
                
                data = {
                    "model": "llama3-70b-8192",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 800,  # Reduced from 1000
                    "temperature": 0.1
                }
                
                response = requests.post(url, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                
            except Exception as e:
                answer = f"Error calling Groq API: {str(e)}"
                if retrieved_chunks:
                    answer = f"""Based on the provided context:

{retrieved_chunks[0][:500] if retrieved_chunks else "No information found."}

{'Additional context: ' + retrieved_chunks[1][:300] if len(retrieved_chunks) > 1 else ''}

[API Error: {str(e)}]"""
                else:
                    answer = "No relevant information found in the document."
        
        return JSONResponse(content={
            "answer": answer,
            "chunks": retrieved_chunks,
            "chunksCount": len(retrieved_chunks)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/images/{topic_id}")
async def get_images(topic_id: str):
    """Get image metadata for a topic."""
    try:
        if topic_id not in topics_storage:
            raise HTTPException(status_code=404, detail="Topic not found")
        
        topic_data = topics_storage[topic_id]
        image_metadata = topic_data["image_metadata"]
        
        return JSONResponse(content={
            "images": list(image_metadata.values()),
            "count": len(image_metadata)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get images: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "AI Tutor Backend API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ai-tutor-backend"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
