"""
RAG-based AI Tutor with Streamlit
Supports PDF text extraction and dynamic image processing with BLIP captioning.
"""

import os
import pickle
import tempfile
import uuid
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

import streamlit as st
import numpy as np
import faiss
import fitz  # PyMuPDF
from PIL import Image
from sentence_transformers import SentenceTransformer
# from transformers import BlipProcessor, BlipForConditionalGeneration  # Disabled for stability
import requests
import json

# Load environment variables
load_dotenv()

# ==================== CONFIGURATION ====================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K_TEXT = 3
TOP_K_IMAGE = 1
DIMENSION = 384  # all-MiniLM-L6-v2 dimension

# ==================== SESSION STATE ====================
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "text_index": None,
        "image_index": None,
        "text_chunks": {},  # id -> chunk text
        "image_metadata": {},  # id -> metadata dict
        "image_paths": {},  # id -> temp file path
        "chat_history": [],
        "pdf_processed": False,
        "images_processed": False,
        "embedding_model": None,
        "blip_processor": None,
        "blip_model": None,
        "temp_dir": None,
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "gemini_api_key": "",
        "selected_model": "groq",
        # "blip_processor": None,  # Disabled for stability
        # "blip_model": None,      # Disabled for stability
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_embedding_model():
    """Load and cache the embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL)

# @st.cache_resource  # Disabled for stability
# def load_blip_model():
#     """Load and cache BLIP model for image captioning."""
#     processor = BlipProcessor.from_pretrained(BLIP_MODEL)
#     model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
#     return processor, model

def get_models():
    """Get or initialize models."""
    if st.session_state["embedding_model"] is None:
        with st.spinner("Loading embedding model..."):
            st.session_state["embedding_model"] = load_embedding_model()
    
    # BLIP model loading disabled for stability
    # if st.session_state["blip_processor"] is None:
    #     with st.spinner("Loading BLIP model for image captioning..."):
    #         processor, model = load_blip_model()
    #         st.session_state["blip_processor"] = processor
    #         st.session_state["blip_model"] = model

# ==================== TEXT PROCESSING ====================
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF using PyMuPDF."""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name
    
    try:
        doc = fitz.open(tmp_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    finally:
        os.unlink(tmp_path)
    
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

def create_text_index(chunks: List[str]) -> Tuple[faiss.IndexFlatIP, Dict[str, str]]:
    """Create FAISS index for text chunks."""
    model = st.session_state["embedding_model"]
    
    # Generate embeddings
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create FAISS index
    index = faiss.IndexFlatIP(DIMENSION)
    index.add(embeddings.astype(np.float32))
    
    # Create ID mapping
    chunk_map = {str(i): chunk for i, chunk in enumerate(chunks)}
    
    return index, chunk_map

# ==================== IMAGE PROCESSING ====================
def generate_image_metadata(filename: str) -> Dict:
    """Generate metadata for image using filename and simple heuristics."""
    # Clean filename for title
    title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
    
    # Generate simple description based on filename
    description = f"Educational diagram showing {title.lower()}"
    
    # Extract keywords from filename
    words = title.lower().replace("_", " ").replace("-", " ").split()
    keywords = [w.strip(".,!?;:") for w in words if len(w) > 2 and w not in ["the", "and", "of", "in", "on", "at", "to", "for"]]
    keywords = list(set(keywords))[:8]  # Top 8 unique keywords
    
    return {
        "title": title,
        "description": description,
        "keywords": keywords
    }

def extract_keywords_from_caption(caption: str) -> List[str]:
    """Extract simple keywords from caption."""
    # Simple keyword extraction - can be enhanced with NLP libraries
    stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should"}
    words = caption.lower().split()
    keywords = [w.strip(".,!?;:") for w in words if w.strip(".,!?;:") not in stopwords and len(w) > 2]
    return list(set(keywords))[:10]  # Top 10 unique keywords

def process_image(image_file, temp_dir: str) -> Tuple[str, Dict, str]:
    """Process single image: generate metadata and save."""
    # Open and save image
    image = Image.open(image_file).convert("RGB")
    
    # Generate metadata
    image_id = str(uuid.uuid4())
    filename = image_file.name
    
    # Use simple metadata generation (no BLIP)
    metadata_dict = generate_image_metadata(filename)
    
    metadata = {
        "id": image_id,
        "filename": filename,
        "title": metadata_dict["title"],
        "description": metadata_dict["description"],
        "keywords": metadata_dict["keywords"],
    }
    
    # Save image to temp directory
    image_path = os.path.join(temp_dir, f"{image_id}.png")
    image.save(image_path)
    
    # Create embedding input
    embedding_input = f"{metadata['title']} {metadata['description']} {' '.join(metadata['keywords'])}"
    
    return image_id, metadata, image_path, embedding_input

def create_image_index(image_data: List[Tuple]) -> Tuple[faiss.IndexFlatIP, Dict[str, Dict], Dict[str, str], List[str]]:
    """Create FAISS index for images."""
    model = st.session_state["embedding_model"]
    
    embedding_inputs = [data[3] for data in image_data]
    embeddings = model.encode(embedding_inputs, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create FAISS index
    index = faiss.IndexFlatIP(DIMENSION)
    index.add(embeddings.astype(np.float32))
    
    # Create mappings
    metadata_map = {data[0]: data[1] for data in image_data}
    path_map = {data[0]: data[2] for data in image_data}
    image_ids = [data[0] for data in image_data]  # Store order for retrieval
    
    return index, metadata_map, path_map, image_ids

# ==================== RAG PIPELINE ====================
def retrieve_text_chunks(query: str, k: int = TOP_K_TEXT) -> List[str]:
    """Retrieve relevant text chunks from FAISS."""
    model = st.session_state["embedding_model"]
    index = st.session_state["text_index"]
    chunk_map = st.session_state["text_chunks"]
    
    # Embed query
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    # Search
    scores, indices = index.search(query_embedding.astype(np.float32), k)
    
    # Get chunks
    retrieved = []
    for idx in indices[0]:
        if idx >= 0 and str(idx) in chunk_map:
            retrieved.append(chunk_map[str(idx)])
    
    return retrieved

def retrieve_relevant_image(text: str, k: int = TOP_K_IMAGE) -> Optional[Tuple[str, str, Dict]]:
    """Retrieve most relevant image based on text."""
    model = st.session_state["embedding_model"]
    index = st.session_state["image_index"]
    metadata_map = st.session_state["image_metadata"]
    path_map = st.session_state["image_paths"]
    
    # Embed text (answer or query)
    text_embedding = model.encode([text], convert_to_numpy=True)
    text_embedding = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)
    
    # Search
    scores, indices = index.search(text_embedding.astype(np.float32), k)
    
    # Get best match
    if len(indices[0]) > 0 and indices[0][0] >= 0:
        image_id = list(metadata_map.keys())[indices[0][0]]
        return image_id, path_map[image_id], metadata_map[image_id]
    
    return None

def generate_answer(query: str, context_chunks: List[str]) -> str:
    """Generate answer using retrieved context and selected LLM."""
    # Combine context - limit to top 3 chunks for better quality
    context = "\n\n".join(context_chunks[:3])
    if len(context) > 1500:
        context = context[:1500] + "\n\n[Context truncated...]"
    
    # Get selected model and API key
    selected_model = st.session_state.get("selected_model", "mock")
    
    if selected_model == "mock":
        # Mock response for demonstration
        if not context_chunks:
            return "Not found in document."
        
        return f"""Based on the provided context from your document:

{context_chunks[0][:500] if context_chunks else "No information found."}

{'Additionally: ' + context_chunks[1][:300] if len(context_chunks) > 1 else ''}

[This is a simulated response. Select Groq or Gemini in the sidebar for actual AI tutoring.]"""
    
    elif selected_model == "groq":
        return generate_groq_answer(query, context)
    
    elif selected_model == "gemini":
        return generate_gemini_answer(query, context)
    
    else:
        return "Please select a model in the sidebar."

def generate_groq_answer(query: str, context: str) -> str:
    """Generate answer using Groq API."""
    api_key = st.session_state.get("groq_api_key", "")
    
    # Use default key if no custom key provided
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY", "")
        if api_key:
            st.write("Debug: Using API key from .env")
        else:
            return "Please enter your Groq API key in sidebar or add GROQ_API_KEY to .env file."
    else:
        st.write("Debug: Using custom API key")
    
    try:
        # Groq API endpoint
        url = "https://api.groq.com/openai/v1/chat/completions"
        st.write(f"Debug: Calling API at {url}")  # Debug line
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Prompt for Groq
        system_prompt = """You are an expert AI tutor teaching students.

Rules:
1. Answer ONLY from the provided context. However, be smart about matching concepts: if the user asks about a combined word like "SchoolBellVibration", map it to "school bell" and "vibration" in the text.
2. Do NOT copy text directly - explain in your own words.
3. Explain in simple, clear terms like teaching a student.
4. Structure your answer:
   - Definition
   - Explanation
   - Example (if relevant)
5. If the fundamental concept is not found at all, say "Not found in document".
6. Be concise but thorough."""
        
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
        
        st.write(f"Debug: Request data: {json.dumps(data, indent=2)}")  # Debug line
        st.write(f"Debug: API Key (first 10 chars): {api_key[:10]}...")  # Debug line
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
        
    except requests.exceptions.RequestException as e:
        return f"Error calling Groq API: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"Error parsing Groq response: {str(e)}"

def generate_gemini_answer(query: str, context: str) -> str:
    """Generate answer using Gemini API."""
    api_key = st.session_state.get("gemini_api_key", "")
    
    if not api_key:
        return "Please enter your Gemini API key in the sidebar."
    
    try:
        # Gemini API endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
        
        # Prompt for Gemini
        prompt = f"""You are an AI tutor. Answer ONLY using the provided context. If the information is not found in the context, say 'Not found in document'. Be concise and helpful.

Context:
{context}

Question: {query}

Answer:"""
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1000
            }
        }
        
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        
    except requests.exceptions.RequestException as e:
        return f"Error calling Gemini API: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"Error parsing Gemini response: {str(e)}"

# ==================== UI COMPONENTS ====================
def render_upload_section():
    """Render file upload section."""
    st.header("📚 Upload Learning Materials")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Upload PDF")
        pdf_file = st.file_uploader("Upload your chapter PDF", type=["pdf"])
        
        if pdf_file and not st.session_state["pdf_processed"]:
            with st.spinner("Processing PDF..."):
                # Extract text
                text = extract_text_from_pdf(pdf_file)
                
                if text.strip():
                    # Chunk text
                    chunks = chunk_text(text)
                    st.info(f"Extracted {len(chunks)} text chunks")
                    
                    # Create index
                    index, chunk_map = create_text_index(chunks)
                    st.session_state["text_index"] = index
                    st.session_state["text_chunks"] = chunk_map
                    st.session_state["pdf_processed"] = True
                    st.success(f"✅ PDF processed: {len(chunks)} chunks indexed")
                else:
                    st.error("No text found in PDF")
    
    with col2:
        st.subheader("2. Upload Diagrams")
        image_files = st.file_uploader(
            "Upload multiple images", 
            type=["png", "jpg", "jpeg"], 
            accept_multiple_files=True
        )
        
        if image_files and not st.session_state["images_processed"]:
            # Create temp directory
            if st.session_state["temp_dir"] is None:
                st.session_state["temp_dir"] = tempfile.mkdtemp()
            
            # Add a process button to avoid automatic processing
            if st.button(f"Process {len(image_files)} Images", type="primary"):
                with st.spinner(f"Processing {len(image_files)} images..."):
                    image_data = []
                    progress_bar = st.progress(0)
                    
                    for i, img_file in enumerate(image_files):
                        try:
                            result = process_image(img_file, st.session_state["temp_dir"])
                            image_data.append(result)
                        except Exception as e:
                            st.error(f"Error processing {img_file.name}: {str(e)}")
                            continue
                        
                        progress_bar.progress((i + 1) / len(image_files))
                    
                    if image_data:
                        try:
                            # Create index
                            index, metadata_map, path_map, image_ids = create_image_index(image_data)
                            st.session_state["image_index"] = index
                            st.session_state["image_metadata"] = metadata_map
                            st.session_state["image_paths"] = path_map
                            st.session_state["images_processed"] = True
                            st.success(f"✅ {len(image_data)} images processed and indexed")
                            
                            # Show metadata
                            with st.expander("View Image Metadata"):
                                for data in image_data:
                                    meta = data[1]
                                    st.write(f"**{meta['filename']}**: {meta['description']}")
                        except Exception as e:
                            st.error(f"Error creating image index: {str(e)}")
                            st.session_state["images_processed"] = False
                    else:
                        st.error("No images could be processed")
                        st.session_state["images_processed"] = False

def render_chat_section():
    """Render chat interface."""
    st.header("💬 Ask Your AI Tutor")
    
    # Check if materials are uploaded
    if not st.session_state["pdf_processed"]:
        st.warning("⚠️ Please upload a PDF first")
        return
    
    # Display chat history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "image" in msg and msg["image"]:
                st.image(msg["image"], caption="Relevant Diagram", use_container_width=True)
    
    # Chat input
    query = st.chat_input("Ask a question about your materials...")
    
    if query:
        # Add user message
        st.session_state["chat_history"].append({
            "role": "user",
            "content": query
        })
        
        with st.chat_message("user"):
            st.write(query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                # Retrieve text chunks
                chunks = retrieve_text_chunks(query)
                
                # Generate answer
                answer = generate_answer(query, chunks)
                
                # Retrieve relevant image (using answer, fallback to query)
                image_result = None
                if st.session_state["images_processed"]:
                    image_result = retrieve_relevant_image(answer)
                    if not image_result:
                        image_result = retrieve_relevant_image(query)
                
                # Display answer
                st.write("**Answer:**")
                st.write(answer)
                
                # Display relevant image
                if image_result:
                    image_id, image_path, metadata = image_result
                    st.write("---")
                    st.write("**📊 Relevant Diagram:**")
                    st.image(image_path, caption=f"{metadata['title']}: {metadata['description'][:100]}...", use_container_width=True)
                
                # Show retrieved chunks (optional, in expander)
                with st.expander("View Retrieved Context Chunks"):
                    for i, chunk in enumerate(chunks, 1):
                        st.write(f"**Chunk {i}:**")
                        st.write(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                        st.write("---")
        
        # Add assistant message to history
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": answer,
            "image": image_result[1] if image_result else None
        })

def render_sidebar():
    """Render sidebar with stats and controls."""
    with st.sidebar:
        st.title("🎓 AI Tutor")
        st.write("RAG-based learning assistant")
        
        st.divider()
        
        # Status
        st.subheader("Status")
        st.write(f"📄 PDF: {'✅ Loaded' if st.session_state['pdf_processed'] else '❌ Not loaded'}")
        st.write(f"🖼️ Images: {'✅ ' + str(len(st.session_state['image_metadata'])) + ' loaded' if st.session_state['images_processed'] else '❌ Not loaded'}")
        
        if st.session_state["text_index"]:
            st.write(f"📝 Text chunks: {len(st.session_state['text_chunks'])}")
        
        st.divider()
        
        # API Keys Section
        st.subheader("🔑 API Keys")
        
        # Model selection
        model_options = ["Mock (Demo)", "Groq (Llama3)", "Gemini Pro"]
        model_values = ["mock", "groq", "gemini"]
        
        selected_index = model_values.index(st.session_state.get("selected_model", "mock"))
        selected = st.selectbox(
            "Select AI Model:",
            options=model_options,
            index=selected_index,
            help="Choose which AI model to use for generating answers"
        )
        
        # Update selected model in session state
        st.session_state["selected_model"] = model_values[model_options.index(selected)]
        
        # Show API key inputs based on selection
        if st.session_state["selected_model"] == "groq":
            groq_key = st.text_input(
                "Groq API Key:",
                type="password",
                value=st.session_state.get("groq_api_key", ""),
                help="Enter your Groq API key from https://console.groq.com (leave empty to use .env)"
            )
            st.session_state["groq_api_key"] = groq_key
            
            if groq_key:
                st.success("✅ Using custom Groq API key")
            else:
                env_key = os.getenv("GROQ_API_KEY", "")
                if env_key:
                    st.info("ℹ️ Using API key from .env")
                else:
                    st.warning("⚠️ Enter Groq API key or add GROQ_API_KEY to .env")
        
        elif st.session_state["selected_model"] == "gemini":
            gemini_key = st.text_input(
                "Gemini API Key:",
                type="password",
                value=st.session_state.get("gemini_api_key", ""),
                help="Enter your Gemini API key from https://makersuite.google.com/app/apikey"
            )
            st.session_state["gemini_api_key"] = gemini_key
            
            if gemini_key:
                st.success("✅ Gemini API key configured")
            else:
                st.warning("⚠️ Enter Gemini API key to use this model")
        
        else:
            st.info("ℹ️ Using mock responses for demonstration")
        
        st.divider()
        
        # Reset button
        if st.button("🔄 Reset All", type="secondary"):
            # Clear temp directory
            if st.session_state["temp_dir"] and os.path.exists(st.session_state["temp_dir"]):
                import shutil
                shutil.rmtree(st.session_state["temp_dir"])
            
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            init_session_state()
            st.rerun()
        
        st.divider()
        
        # Save/Load buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save"):
                save_data()
        with col2:
            if st.button("📂 Load"):
                load_data()

def save_data():
    """Save indices and mappings to disk."""
    try:
        data = {
            "text_chunks": st.session_state["text_chunks"],
            "image_metadata": st.session_state["image_metadata"],
            "image_paths": st.session_state["image_paths"],
        }
        
        # Save FAISS indices
        if st.session_state["text_index"]:
            faiss.write_index(st.session_state["text_index"], "text_index.faiss")
        if st.session_state["image_index"]:
            faiss.write_index(st.session_state["image_index"], "image_index.faiss")
        
        # Save mappings
        with open("mappings.pkl", "wb") as f:
            pickle.dump(data, f)
        
        st.success("Data saved!")
    except Exception as e:
        st.error(f"Error saving: {e}")

def load_data():
    """Load indices and mappings from disk."""
    try:
        # Load FAISS indices
        if os.path.exists("text_index.faiss"):
            st.session_state["text_index"] = faiss.read_index("text_index.faiss")
            st.session_state["pdf_processed"] = True
        
        if os.path.exists("image_index.faiss"):
            st.session_state["image_index"] = faiss.read_index("image_index.faiss")
            st.session_state["images_processed"] = True
        
        # Load mappings
        if os.path.exists("mappings.pkl"):
            with open("mappings.pkl", "rb") as f:
                data = pickle.load(f)
                st.session_state["text_chunks"] = data.get("text_chunks", {})
                st.session_state["image_metadata"] = data.get("image_metadata", {})
                st.session_state["image_paths"] = data.get("image_paths", {})
        
        st.success("Data loaded!")
    except Exception as e:
        st.error(f"Error loading: {e}")

# ==================== MAIN ====================
def main():
    """Main application."""
    st.set_page_config(
        page_title="AI Tutor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    get_models()
    render_sidebar()
    
    st.title("🎓 AI Tutor")
    st.write("Upload your PDF chapter and diagrams, then ask questions!")
    
    st.divider()
    
    render_upload_section()
    
    st.divider()
    
    render_chat_section()

if __name__ == "__main__":
    main()
