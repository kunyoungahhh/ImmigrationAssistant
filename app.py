from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import faiss
import numpy as np
import os
import fitz  # PyMuPDF
import re
import time
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="frontend/build", static_url_path="")
CORS(app)  # Enable CORS for API access

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF" if device == "cpu" else "mistralai/Mistral-7B-Instruct-v0.1"
PDF_PATH = "your_legal_doc.pdf"
CHUNK_OVERLAP = 2  # Number of sentences to overlap between chunks
CHUNK_SIZE = 5  # Number of sentences per chunk

try:
    # Load embedding model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    # Load LLM based on available hardware
    logger.info(f"Loading LLM: {LLM_MODEL}")
    if device == "cpu":
        # Use a quantized model for CPU
        from ctransformers import AutoModelForCausalLM as CTAutoModelForCausalLM
        llm = CTAutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            model_type="mistral"
        )
    else:
        # Use HF Transformers for GPU
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
        llm = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    # Load or build FAISS index
    chunks = []
    chunk_metadata = []
    
    if os.path.exists("embeddings.index") and os.path.exists("chunks.txt"):
        logger.info("Loading existing FAISS index and chunks")
        index = faiss.read_index("embeddings.index")
        with open("chunks.txt", "r") as f:
            chunks = f.read().split("\n===CHUNK_SEPARATOR===\n")
        if os.path.exists("metadata.txt"):
            with open("metadata.txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        page, start = line.strip().split(",")
                        chunk_metadata.append({"page": int(page), "start": int(start)})
    else:
        logger.info(f"Building new index from {PDF_PATH}")
        # Extract text from PDF with page numbers
        doc = fitz.open(PDF_PATH)
        text_with_metadata = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            text_with_metadata.append((text, page_num))
        
        # Tokenize into sentences
        from nltk.tokenize import sent_tokenize
        try:
            import nltk
            nltk.download('punkt', quiet=True)
        except:
            logger.warning("NLTK download failed, but may still work if already cached")
        
        all_sentences = []
        all_sentence_metadata = []
        
        for text, page_num in text_with_metadata:
            sentences = sent_tokenize(text)
            char_count = 0
            for sentence in sentences:
                all_sentences.append(sentence)
                all_sentence_metadata.append({"page": page_num, "start": char_count})
                char_count += len(sentence)
        
        # Create overlapping chunks
        for i in range(0, len(all_sentences), CHUNK_SIZE - CHUNK_OVERLAP):
            if i + CHUNK_SIZE <= len(all_sentences):
                chunk = " ".join(all_sentences[i:i+CHUNK_SIZE])
                chunks.append(chunk)
                chunk_metadata.append(all_sentence_metadata[i])  # Metadata from first sentence
        
        # Create embeddings and index
        embeddings = embedder.encode(chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        # Save index and chunks
        faiss.write_index(index, "embeddings.index")
        with open("chunks.txt", "w") as f:
            f.write("\n===CHUNK_SEPARATOR===\n".join(chunks))
        with open("metadata.txt", "w") as f:
            for meta in chunk_metadata:
                f.write(f"{meta['page']},{meta['start']}\n")
    
    logger.info(f"Loaded {len(chunks)} chunks into the index")

except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    raise

@lru_cache(maxsize=100)
def generate_answer(question, context):
    """Cache responses for identical questions and context"""
    try:
        if device == "cpu":
            # Format for ctransformers
            prompt = f"""<s>[INST] You are a legal assistant specialized in immigration law. 
Answer this question based only on the provided context. If you don't know the answer from the context, say so.

Context:
{context}

Question: {question} [/INST]</s>"""
            result = llm(prompt, max_tokens=300, temperature=0.1)
            return result
        else:
            # Format for HF Transformers
            prompt = f"""<s>[INST] You are a legal assistant specialized in immigration law. 
Answer this question based only on the provided context. If you don't know the answer from the context, say so.

Context:
{context}

Question: {question} [/INST]</s>"""
            result = llm(prompt, max_new_tokens=300, do_sample=True, temperature=0.1)[0]["generated_text"]
            # Extract just the response part
            response = result.split("[/INST]</s>")[-1].strip()
            return response
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return f"Error generating response: {str(e)}"

@app.route("/api/query", methods=["POST"])
def query():
    start_time = time.time()
    try:
        data = request.json
        if not data or "question" not in data:
            return jsonify({"error": "No question provided"}), 400
        
        user_input = data["question"]
        logger.info(f"Query received: {user_input}")
        
        # Get embeddings for the query
        query_vec = embedder.encode([user_input])
        
        # Search the index
        k = min(3, len(chunks))  # Don't try to retrieve more chunks than exist
        distances, indices = index.search(np.array(query_vec).astype('float32'), k=k)
        
        # Get the relevant chunks and their metadata
        relevant_chunks = [chunks[i] for i in indices[0]]
        sources = [f"Page {chunk_metadata[i]['page'] + 1}" for i in indices[0]]
        
        # Join the chunks with separators
        context = "\n---\n".join(relevant_chunks)

        print("\n🧩 Relevant Chunks Used for Answer:")
        for idx, (chunk, meta) in enumerate(zip(relevant_chunks, [chunk_metadata[i] for i in indices[0]])):
            print(f"\n--- Chunk {idx + 1} (Page {meta['page'] + 1}) ---\n{chunk}")
        
        # Generate the answer
        answer = generate_answer(user_input, context)
        
        processing_time = time.time() - start_time
        
        return jsonify({
            "answer": answer,
            "sources": sources,
            "processing_time": f"{processing_time:.2f} seconds"
        })
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/health")
def health():
    return jsonify({"status": "healthy", "chunks": len(chunks)})

# Serve React app
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(app.static_folder + "/" + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
