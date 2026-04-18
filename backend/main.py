from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
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
import fitz
from dataclasses import dataclass
from PIL import Image
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Tutor Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
DIMENSION = 384
TOP_K_TEXT = 3

topics_storage = {}
text_indexes = {}
image_indexes = {}

@dataclass
class TopicData:
    id: str
    text_chunks: List[str]
    text_index: faiss.IndexFlatIP
    image_metadata: Dict[str, Dict]
    image_index: faiss.IndexFlatIP
    image_paths: Dict[str, str]

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
        image_id = str(uuid.uuid4())
        image_path = os.path.join(temp_dir, f"{image_id}.png")
        
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image_file.file, f)
        
        filename = image_file.filename
        title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
        description = f"Educational diagram showing {title.lower()}"
        
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
        temp_dir = tempfile.mkdtemp()
        
        topic_id = str(uuid.uuid4())
        pdf_path = os.path.join(temp_dir, pdf.filename)
        
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(pdf.file, f)
        
        text = extract_text_from_pdf(pdf_path)
        text_chunks = chunk_text(text)
        
        text_index = create_text_index(text_chunks)
        
        image_data = []
        for image_file in images:
            img_metadata = process_image(image_file, temp_dir)
            if img_metadata:
                image_data.append(img_metadata)
        
        image_index = create_image_index(image_data)
        
        topic_data = {
            "id": topic_id,
            "text_chunks": text_chunks,
            "text_index": text_index,
            "image_metadata": {img["id"]: img for img in image_data if img},
            "image_index": image_index,
            "image_paths": {img["id"]: img["path"] for img in image_data if img},
            "image_ids": [img["id"] for img in image_data if img]
        }
        
        topics_storage[topic_id] = topic_data
        text_indexes[topic_id] = text_index
        image_indexes[topic_id] = image_index
        
        shutil.rmtree(temp_dir)
        
        return JSONResponse(content={
            "topicId": topic_id,
            "message": "Successfully processed PDF and images",
            "chunksCount": len(text_chunks),
            "imagesCount": len(image_data)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/upload-sample")
async def upload_sample_data():
    """Load sample data from sample_data directory."""
    try:
        sample_dir = os.path.join(os.getcwd(), "sample_data")
        if not os.path.exists(sample_dir):
            raise HTTPException(status_code=404, detail="Sample data directory not found")
        
        pdf_path = os.path.join(sample_dir, "Sound.pdf")
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="Sound.pdf not found in sample_data")
        
        temp_dir = tempfile.mkdtemp()
        topic_id = str(uuid.uuid4())
        
        target_pdf = os.path.join(temp_dir, "Sound.pdf")
        shutil.copy2(pdf_path, target_pdf)
        
        text = extract_text_from_pdf(target_pdf)
        text_chunks = chunk_text(text)
        text_index = create_text_index(text_chunks)
        
        image_data = []
        for filename in os.listdir(sample_dir):
            if filename.endswith((".png", ".jpg", ".jpeg")) and filename != "Sound.pdf":
                file_path = os.path.join(sample_dir, filename)
                
                image_id = str(uuid.uuid4())
                target_img = os.path.join(temp_dir, f"{image_id}.png")
                shutil.copy2(file_path, target_img)
                
                title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
                description = f"Educational diagram showing {title.lower()}"
                words = title.lower().replace("_", " ").replace("-", " ").split()
                keywords = [w.strip(".,!?;:") for w in words if len(w) > 2]
                
                img_metadata = {
                    "id": image_id,
                    "filename": filename,
                    "title": title,
                    "description": description,
                    "keywords": list(set(keywords))[:8],
                    "path": target_img
                }
                image_data.append(img_metadata)
        
        image_index = create_image_index(image_data)
        
        topic_data = {
            "id": topic_id,
            "text_chunks": text_chunks,
            "text_index": text_index,
            "image_metadata": {img["id"]: img for img in image_data if img},
            "image_index": image_index,
            "image_paths": {img["id"]: img["path"] for img in image_data if img},
            "image_ids": [img["id"] for img in image_data if img]
        }
        
        topics_storage[topic_id] = topic_data
        text_indexes[topic_id] = text_index
        image_indexes[topic_id] = image_index
        
        return JSONResponse(content={
            "topicId": topic_id,
            "message": "Successfully loaded Sound sample data",
            "chunksCount": len(text_chunks),
            "imagesCount": len(image_data)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load sample: {str(e)}")

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
        
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        scores, indices = text_index.search(query_embedding.astype(np.float32), TOP_K_TEXT)
        
        retrieved_chunks = []
        for idx in indices[0]:
            if idx >= 0 and idx < len(text_chunks):
                retrieved_chunks.append(text_chunks[idx])
        
        context = "\n\n".join(retrieved_chunks[:3])
        if len(context) > 1200:
            context = context[:1200] + "\n\n[Context truncated for API limits...]"
        
        context = "\n\n".join(retrieved_chunks)
        
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        
        if not groq_api_key:
            if retrieved_chunks:
                answer = f"""Based on the provided context:

{retrieved_chunks[0][:500] if retrieved_chunks else "No information found."}

{'Additional context: ' + retrieved_chunks[1][:300] if len(retrieved_chunks) > 1 else ''}

[This is a demo response. Integrate with actual LLM for production.]"""
            else:
                answer = "No relevant information found in the document."
        else:
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                
                headers = {
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                }
                
                system_prompt = """You are an expert AI tutor teaching students about sound and physics.

Rules:
1. Answer ONLY from the provided context. However, be smart about matching concepts: if the user asks about a combined word like "SchoolBellVibration", map it to "school bell" and "vibration" in the text.
2. Do NOT copy text directly - explain in your own words
3. Structure answers clearly:
   - Definition first
   - Simple explanation
   - Example if relevant
4. If the fundamental concept is not found, say: "Not found in document"
5. Be concise but thorough"""
                
                user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""
                
                data = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 800,
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
        
        relevant_image = None
        image_index = topic_data.get("image_index")
        image_ids = topic_data.get("image_ids", [])
        image_metadata = topic_data.get("image_metadata", {})
        
        if image_index is not None and len(image_ids) > 0 and answer:
            ans_embedding = embedding_model.encode([answer], convert_to_numpy=True)
            ans_embedding = ans_embedding / np.linalg.norm(ans_embedding, axis=1, keepdims=True)
            
            img_scores, img_indices = image_index.search(ans_embedding.astype(np.float32), 1)
            if len(img_indices[0]) > 0 and img_indices[0][0] >= 0:
                idx = img_indices[0][0]
                if idx < len(image_ids):
                    img_id = image_ids[idx]
                    relevant_image = image_metadata.get(img_id)
        
        return JSONResponse(content={
            "answer": answer,
            "chunks": retrieved_chunks,
            "chunksCount": len(retrieved_chunks),
            "relevant_image": relevant_image
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

@app.get("/image-file/{topic_id}/{image_id}")
async def get_image_file(topic_id: str, image_id: str):
    """Serve an image file."""
    if topic_id in topics_storage:
        topic_data = topics_storage[topic_id]
        image_paths = topic_data.get("image_paths", {})
        if image_id in image_paths:
            path = image_paths[image_id]
            if os.path.exists(path):
                return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found")

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
