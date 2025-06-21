# Immigration Assistant

An AI-powered assistant that helps immigrants navigate complex immigration processes by providing accurate, up-to-date information from official USCIS documentation.

## Features

- **Intelligent Q&A**: Ask questions about immigration processes and get precise answers
- **Official Documentation**: Uses the most current USCIS documents as information source
- **Smart Routing**: AI agent automatically routes queries to the most relevant information

## Tech Stack

### Backend
- **Document Parsing**: llama_index + SimpleDirectoryReader
- **Retrieval**: RAG (Retrieval-Augmented Generation) via vector index
- **Agent**: ReActAgent for intelligent query routing
- **LLM**: GPT-4

### Frontend
- **React**: Modern, responsive user interface

## How It Works

1. **Document Processing**: USCIS documents are parsed and indexed using llama_index
2. **Query Processing**: User questions are processed by the ReActAgent
3. **Information Retrieval**: Relevant information is retrieved using vector search
4. **Response Generation**: GPT-4 generates accurate, contextual answers
