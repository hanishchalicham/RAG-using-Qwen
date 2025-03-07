# RAG Implementation using Qwen2.5

This repository implements a Retrieval-Augmented Generation (RAG) system using the Qwen2.5 language model. The system combines document retrieval with generative AI to provide contextually informed responses.

## 🚀 Features

- **Document Processing Pipeline** - Extract, split, and process text from PDF documents
- **Vector Embeddings** - Uses `thenlper/gte-small` embeddings for semantic search
- **FAISS Vector Database** - Fast similarity search with cosine distance metrics
- **LLM Integration** - `Qwen/Qwen2.5-1.5B-Instruct` for generating contextually relevant responses
- **Markdown-Optimized Text Splitting** - Hierarchical splitting tailored for markdown documents

## 🔧 Implementation Details

### Document Processing

The system includes utilities for:
- Loading PDF documents with LangChain's `PyPDFLoader`
- Processing and chunking text with `RecursiveCharacterTextSplitter`
- Configurable chunk sizes and overlaps for optimal retrieval

### Embedding and Retrieval

- Document embeddings are generated using the `thenlper/gte-small` model
- Embeddings are normalized for cosine similarity comparison
- FAISS vector database enables efficient similarity search
- Customizable retrieval parameters (k-nearest neighbors)

### Generation with Qwen2.5

- Implements the 1.5B parameter instruct-tuned version of Qwen2.5
- Configurable system prompts for different RAG applications
- Template-based context formatting for effective few-shot learning
- Generates concise, focused responses based on retrieved documents

## 📊 Example Usage

```python
# Simple RAG query function
def rag(user_query, num_of_docs):
    related_documents = KNOWLEDGE_VECTOR_DATABASE.similarity_search(user_query, k=num_of_docs)
    whole_docs = ''
    for r in related_documents:
        whole_docs += str(r.page_content)
    prompt = get_prompt(whole_docs, user_query)
    res = llm(sp, prompt)
    return res

# Example query
response = rag("What is a Transformer Seq-to-Seq", 5)
```

## 🛠️ Setup and Requirements

This project requires the following packages:
- torch
- transformers
- accelerate
- bitsandbytes
- langchain
- sentence_transformers
- faiss-gpu
- openpyxl
- pacmap
- datasets
- langchain-community
- ragatouille

## 💡 Applications

- Question answering over custom document collections
- Knowledge-based chatbots
- Document summarization and information extraction
- Semantic search with natural language queries
- Domain-specific virtual assistants

## 🔍 Future Improvements

- Implement query expansion and rewriting techniques
- Add document metadata for better retrieval filtering
- Integrate evaluation metrics for RAG performance
- Explore hybrid retrieval approaches (BM25 + vector search)
- Fine-tune Qwen2.5 on domain-specific data
