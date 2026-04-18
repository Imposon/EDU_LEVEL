import os
import pickle
import tempfile
import uuid
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

import streamlit as st
import numpy as np
import faiss
import fitz
from PIL import Image
from sentence_transformers import SentenceTransformer
import requests
import json

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K_TEXT = 3
TOP_K_IMAGE = 1
DIMENSION = 384

def init_session_state():
    defaults = {
        "text_index": None,
        "image_index": None,
        "text_chunks": {},
        "image_metadata": {},
        "image_paths": {},
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def get_models():
    if st.session_state["embedding_model"] is None:
        with st.spinner("Loading embedding model..."):
            st.session_state["embedding_model"] = load_embedding_model()

def extract_text_from_pdf(pdf_file) -> str:
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
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def create_text_index(chunks: List[str]) -> Tuple[faiss.IndexFlatIP, Dict[str, str]]:
    model = st.session_state["embedding_model"]
    
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    index = faiss.IndexFlatIP(DIMENSION)
    index.add(embeddings.astype(np.float32))
    
    chunk_map = {str(i): chunk for i, chunk in enumerate(chunks)}
    
    return index, chunk_map

def generate_image_metadata(filename: str) -> Dict:
    title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
    
    description = f"Educational diagram showing {title.lower()}"
    
    words = title.lower().replace("_", " ").replace("-", " ").split()
    keywords = [w.strip(".,!?;:") for w in words if len(w) > 2 and w not in ["the", "and", "of", "in", "on", "at", "to", "for"]]
    keywords = list(set(keywords))[:8]
    
    return {
        "title": title,
        "description": description,
        "keywords": keywords
    }

def extract_keywords_from_caption(caption: str) -> List[str]:
    stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should"}
    words = caption.lower().split()
    keywords = [w.strip(".,!?;:") for w in words if w.strip(".,!?;:") not in stopwords and len(w) > 2]
    return list(set(keywords))[:10]

def process_image(image_file, temp_dir: str) -> Tuple[str, Dict, str]:
    image = Image.open(image_file).convert("RGB")
    
    image_id = str(uuid.uuid4())
    filename = image_file.name
    
    metadata_dict = generate_image_metadata(filename)
    
    metadata = {
        "id": image_id,
        "filename": filename,
        "title": metadata_dict["title"],
        "description": metadata_dict["description"],
        "keywords": metadata_dict["keywords"],
    }
    
    image_path = os.path.join(temp_dir, f"{image_id}.png")
    image.save(image_path)
    
    embedding_input = f"{metadata['title']} {metadata['description']} {' '.join(metadata['keywords'])}"
    
    return image_id, metadata, image_path, embedding_input

def create_image_index(image_data: List[Tuple]) -> Tuple[faiss.IndexFlatIP, Dict[str, Dict], Dict[str, str], List[str]]:
    model = st.session_state["embedding_model"]
    
    embedding_inputs = [data[3] for data in image_data]
    embeddings = model.encode(embedding_inputs, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    index = faiss.IndexFlatIP(DIMENSION)
    index.add(embeddings.astype(np.float32))
    
    metadata_map = {data[0]: data[1] for data in image_data}
    path_map = {data[0]: data[2] for data in image_data}
    image_ids = [data[0] for data in image_data]
    
    return index, metadata_map, path_map, image_ids

def retrieve_text_chunks(query: str, k: int = TOP_K_TEXT) -> List[str]:
    model = st.session_state["embedding_model"]
    index = st.session_state["text_index"]
    chunk_map = st.session_state["text_chunks"]
    
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    scores, indices = index.search(query_embedding.astype(np.float32), k)
    
    retrieved = []
    for idx in indices[0]:
        if idx >= 0 and str(idx) in chunk_map:
            retrieved.append(chunk_map[str(idx)])
    
    return retrieved

def retrieve_relevant_image(text: str, k: int = TOP_K_IMAGE) -> Optional[Tuple[str, str, Dict]]:
    model = st.session_state["embedding_model"]
    index = st.session_state["image_index"]
    metadata_map = st.session_state["image_metadata"]
    path_map = st.session_state["image_paths"]
    
    text_embedding = model.encode([text], convert_to_numpy=True)
    text_embedding = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)
    
    scores, indices = index.search(text_embedding.astype(np.float32), k)
    
    if len(indices[0]) > 0 and indices[0][0] >= 0:
        image_id = list(metadata_map.keys())[indices[0][0]]
        return image_id, path_map[image_id], metadata_map[image_id]
    
    return None

def generate_answer(query: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks[:3])
    if len(context) > 1500:
        context = context[:1500] + "\n\n[Context truncated...]"
    
    selected_model = st.session_state.get("selected_model", "mock")
    
    if selected_model == "mock":
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
    api_key = st.session_state.get("groq_api_key", "")
    
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            return "Please enter your Groq API key in sidebar or add GROQ_API_KEY to .env file."
    
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
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
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
        
    except requests.exceptions.RequestException as e:
        return f"Error calling Groq API: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"Error parsing Groq response: {str(e)}"

def generate_gemini_answer(query: str, context: str) -> str:
    api_key = st.session_state.get("gemini_api_key", "")
    
    if not api_key:
        return "Please enter your Gemini API key in the sidebar."
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
        
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

def render_upload_section():
    st.header("📚 Upload Learning Materials")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Upload PDF")
        pdf_file = st.file_uploader("Upload your chapter PDF", type=["pdf"])
        
        if pdf_file and not st.session_state["pdf_processed"]:
            with st.spinner("Processing PDF..."):
                text = extract_text_from_pdf(pdf_file)
                
                if text.strip():
                    chunks = chunk_text(text)
                    st.info(f"Extracted {len(chunks)} text chunks")
                    
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
            if st.session_state["temp_dir"] is None:
                st.session_state["temp_dir"] = tempfile.mkdtemp()
            
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
                            index, metadata_map, path_map, image_ids = create_image_index(image_data)
                            st.session_state["image_index"] = index
                            st.session_state["image_metadata"] = metadata_map
                            st.session_state["image_paths"] = path_map
                            st.session_state["images_processed"] = True
                            st.success(f"✅ {len(image_data)} images processed and indexed")
                            
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
    
    st.divider()
    st.subheader("3. Sample Data")
    if st.button("🎵 Load Sound Chapter Sample", type="secondary"):
        with st.spinner("Loading sample data..."):
            sample_dir = os.path.join(os.getcwd(), "sample_data")
            pdf_path = os.path.join(sample_dir, "Sound.pdf")
            
            if os.path.exists(pdf_path):
                # Process PDF
                with open(pdf_path, "rb") as f:
                    text = extract_text_from_pdf(f)
                
                if text.strip():
                    chunks = chunk_text(text)
                    index, chunk_map = create_text_index(chunks)
                    st.session_state["text_index"] = index
                    st.session_state["text_chunks"] = chunk_map
                    st.session_state["pdf_processed"] = True
                    
                    # Process Images
                    if st.session_state["temp_dir"] is None:
                        st.session_state["temp_dir"] = tempfile.mkdtemp()
                    
                    image_data = []
                    for filename in os.listdir(sample_dir):
                        if filename.endswith((".png", ".jpg", ".jpeg")) and filename != "Sound.pdf":
                            file_path = os.path.join(sample_dir, filename)
                            with open(file_path, "rb") as f:
                                # We need a file-like object with a .name attribute for process_image
                                class NamedBytesIO:
                                    def __init__(self, content, name):
                                        self.content = content
                                        self.name = name
                                    def read(self):
                                        return self.content
                                    def seek(self, pos):
                                        pass
                                
                                # Actually process_image takes a file object
                                # Let's just do it manually or adapt process_image
                                # But process_image is already there. Let's use it.
                                # Streamlit's file_uploader returns a UploadedFile which has .name
                                # We can just pass the path and modify process_image or do it here.
                                
                                image = Image.open(file_path).convert("RGB")
                                image_id = str(uuid.uuid4())
                                
                                title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
                                description = f"Educational diagram showing {title.lower()}"
                                words = title.lower().replace("_", " ").replace("-", " ").split()
                                keywords = [w.strip(".,!?;:") for w in words if len(w) > 2]
                                
                                metadata = {
                                    "id": image_id,
                                    "filename": filename,
                                    "title": title,
                                    "description": description,
                                    "keywords": list(set(keywords))[:8],
                                }
                                
                                target_path = os.path.join(st.session_state["temp_dir"], f"{image_id}.png")
                                image.save(target_path)
                                
                                embedding_input = f"{metadata['title']} {metadata['description']} {' '.join(metadata['keywords'])}"
                                image_data.append((image_id, metadata, target_path, embedding_input))
                    
                    if image_data:
                        idx, m_map, p_map, ids = create_image_index(image_data)
                        st.session_state["image_index"] = idx
                        st.session_state["image_metadata"] = m_map
                        st.session_state["image_paths"] = p_map
                        st.session_state["images_processed"] = True
                        
                    st.success(f"✅ Loaded Sound sample: {len(chunks)} text chunks and {len(image_data)} images")
                    st.rerun()
                else:
                    st.error("No text found in sample PDF")
            else:
                st.error("Sample data not found")

def render_chat_section():
    st.header("💬 Ask Your AI Tutor")
    
    if not st.session_state["pdf_processed"]:
        st.warning("⚠️ Please upload a PDF first")
        return
    
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "image" in msg and msg["image"]:
                st.image(msg["image"], caption="Relevant Diagram", use_container_width=True)
    
    query = st.chat_input("Ask a question about your materials...")
    
    if query:
        st.session_state["chat_history"].append({
            "role": "user",
            "content": query
        })
        
        with st.chat_message("user"):
            st.write(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                chunks = retrieve_text_chunks(query)
                
                answer = generate_answer(query, chunks)
                
                image_result = None
                if st.session_state["images_processed"]:
                    image_result = retrieve_relevant_image(answer)
                    if not image_result:
                        image_result = retrieve_relevant_image(query)
                
                st.write("**Answer:**")
                st.write(answer)
                
                if image_result:
                    image_id, image_path, metadata = image_result
                    st.write("---")
                    st.write("**📊 Relevant Diagram:**")
                    st.image(image_path, caption=f"{metadata['title']}: {metadata['description'][:100]}...", use_container_width=True)
                
                with st.expander("View Retrieved Context Chunks"):
                    for i, chunk in enumerate(chunks, 1):
                        st.write(f"**Chunk {i}:**")
                        st.write(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                        st.write("---")
        
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": answer,
            "image": image_result[1] if image_result else None
        })

def render_sidebar():
    with st.sidebar:
        st.title("🎓 AI Tutor")
        st.write("RAG-based learning assistant")
        
        st.divider()
        
        st.subheader("Status")
        st.write(f"📄 PDF: {'✅ Loaded' if st.session_state['pdf_processed'] else '❌ Not loaded'}")
        st.write(f"🖼️ Images: {'✅ ' + str(len(st.session_state['image_metadata'])) + ' loaded' if st.session_state['images_processed'] else '❌ Not loaded'}")
        
        if st.session_state["text_index"]:
            st.write(f"📝 Text chunks: {len(st.session_state['text_chunks'])}")
        
        st.divider()
        
        st.subheader("🔑 API Keys")
        
        model_options = ["Mock (Demo)", "Groq (Llama3)", "Gemini Pro"]
        model_values = ["mock", "groq", "gemini"]
        
        selected_index = model_values.index(st.session_state.get("selected_model", "mock"))
        selected = st.selectbox(
            "Select AI Model:",
            options=model_options,
            index=selected_index,
            help="Choose which AI model to use for generating answers"
        )
        
        st.session_state["selected_model"] = model_values[model_options.index(selected)]
        
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
        
        if st.button("🔄 Reset All", type="secondary"):
            if st.session_state["temp_dir"] and os.path.exists(st.session_state["temp_dir"]):
                import shutil
                shutil.rmtree(st.session_state["temp_dir"])
            
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            init_session_state()
            st.rerun()
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save"):
                save_data()
        with col2:
            if st.button("📂 Load"):
                load_data()

def save_data():
    try:
        data = {
            "text_chunks": st.session_state["text_chunks"],
            "image_metadata": st.session_state["image_metadata"],
            "image_paths": st.session_state["image_paths"],
        }
        
        if st.session_state["text_index"]:
            faiss.write_index(st.session_state["text_index"], "text_index.faiss")
        if st.session_state["image_index"]:
            faiss.write_index(st.session_state["image_index"], "image_index.faiss")
        
        with open("mappings.pkl", "wb") as f:
            pickle.dump(data, f)
        
        st.success("Data saved!")
    except Exception as e:
        st.error(f"Error saving: {e}")

def load_data():
    try:
        if os.path.exists("text_index.faiss"):
            st.session_state["text_index"] = faiss.read_index("text_index.faiss")
            st.session_state["pdf_processed"] = True
        
        if os.path.exists("image_index.faiss"):
            st.session_state["image_index"] = faiss.read_index("image_index.faiss")
            st.session_state["images_processed"] = True
        
        if os.path.exists("mappings.pkl"):
            with open("mappings.pkl", "rb") as f:
                data = pickle.load(f)
                st.session_state["text_chunks"] = data.get("text_chunks", {})
                st.session_state["image_metadata"] = data.get("image_metadata", {})
                st.session_state["image_paths"] = data.get("image_paths", {})
        
        st.success("Data loaded!")
    except Exception as e:
        st.error(f"Error loading: {e}")

def main():
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
