"""
Utility script to preprocess a PDF document and create chunks for the RAG system.
This can be run separately to prepare documents before starting the main application.
"""

import fitz  # PyMuPDF
import argparse
import os
import re
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean and normalize text from PDF"""
    # Replace multiple newlines with a single one
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with a single one
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers and headers/footers (customize as needed)
    text = re.sub(r'\n\d+\n', '\n', text)
    return text.strip()

def process_pdf(pdf_path, chunk_size=5, chunk_overlap=2, output_dir='.'):
    """Process a PDF file and create embeddings and chunks"""
    try:
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF with page numbers
        doc = fitz.open(pdf_path)
        text_with_metadata = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            cleaned_text = clean_text(text)
            text_with_metadata.append((cleaned_text, page_num))
        
        # Tokenize into sentences
        all_sentences = []
        all_sentence_metadata = []
        
        for text, page_num in text_with_metadata:
            sentences = sent_tokenize(text)
            char_count = 0
            for sentence in sentences:
                all_sentences.append(sentence)
                all_sentence_metadata.append({"page": page_num, "start": char_count})
                char_count += len(sentence)
        
        logger.info(f"Extracted {len(all_sentences)} sentences from {len(doc)} pages")
        
        # Create overlapping chunks
        chunks = []
        chunk_metadata = []
        
        for i in range(0, len(all_sentences), chunk_size - chunk_overlap):
            if i + chunk_size <= len(all_sentences):
                chunk = " ".join(all_sentences[i:i+chunk_size])
                chunks.append(chunk)
                chunk_metadata.append(all_sentence_metadata[i])  # Metadata from first sentence
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Create embeddings and index
        logger.info("Creating embeddings...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedder.encode(chunks)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        # Save index and chunks
        logger.info(f"Saving index and chunks to {output_dir}")
        faiss.write_index(index, os.path.join(output_dir, "embeddings.index"))
        
        with open(os.path.join(output_dir, "chunks.txt"), "w") as f:
            f.write("\n===CHUNK_SEPARATOR===\n".join(chunks))
        
        with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
            for meta in chunk_metadata:
                f.write(f"{meta['page']},{meta['start']}\n")
        
        logger.info("Processing complete!")
        return True
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a PDF document for RAG")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--chunk-size", type=int, default=5, help="Number of sentences per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=2, help="Number of sentences to overlap between chunks")
    parser.add_argument("--output-dir", default=".", help="Directory to save the output files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        logger.error(f"PDF file not found: {args.pdf_path}")
    else:
        process_pdf(args.pdf_path, args.chunk_size, args.chunk_overlap, args.output_dir)
