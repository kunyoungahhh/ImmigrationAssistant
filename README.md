# Legal RAG System

A Retrieval-Augmented Generation (RAG) system for legal document Q&A, built with Flask, React, and Transformers.

## Features

- PDF document ingestion and processing
- Vector search with FAISS
- LLM-powered question answering
- React-based chat interface
- Source attribution for answers
- Adaptive to CPU or GPU environments

## Setup

### Prerequisites

- Python 3.8+
- Node.js 14+
- PDF document(s) for ingestion

### Installation

1. Clone the repository:
\`\`\`bash
git clone https://github.com/yourusername/legal-rag-system.git
cd legal-rag-system
\`\`\`

2. Install Python dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. Install frontend dependencies:
\`\`\`bash
cd frontend
npm install
\`\`\`

### Preparing Documents

1. Place your PDF document in the project root directory
2. Update the `PDF_PATH` variable in `app.py` to point to your document
3. Run the preprocessing script:
\`\`\`bash
python scripts/preprocess_pdf.py your_legal_doc.pdf
\`\`\`

### Running the Application

1. Build the React frontend:
\`\`\`bash
cd frontend
npm run build
\`\`\`

2. Start the Flask server:
\`\`\`bash
python app.py
\`\`\`

3. Open your browser and navigate to `http://localhost:5000`

## Development

For development, you can run the React development server:

\`\`\`bash
cd frontend
npm start
\`\`\`

This will start the React app on port 3000 with hot reloading. API requests will be proxied to the Flask server running on port 5000.

## Customization

- Adjust chunking parameters in `app.py` or when running the preprocessing script
- Modify the LLM prompt in the `generate_answer` function
- Customize the UI by editing the React components

## License

MIT
