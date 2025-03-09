# studio_tech
## Introduction

This repo provides an AI Agent interface specifically tuned for music studio related assistance. It enables easy loading of technical manuals by using a RAG retrieval strategy derived from Anthropic's Contextual Retrieval approach:

https://www.anthropic.com/news/contextual-retrieval

https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb

## Setup

You will need to set up a .env file or environment variables with the following keys for Anthropic (LLMs) and Voyage (embeddings):

ANTHROPIC_API_KEY=
VOYAGE_API_KEY=

Then run pip install -r requirements.txt

To run Weaviate, run the docker-compose.yml file. This app assumes it's running locally with standard configurations.

To run the app, chainlit run app.py

## Components
### LLMs
- AI Agent - Claude Sonnet
- LLM Classifier - Claude Haiku
### Interface
- Chainlit - Minimal customization has been done. Currently implements business logic where it will only query for technical documentation if a brand or model is specifically mentioned.
### Utilities
The following utilities are meant to be run in sequence. They've been broken out in this way because they can be expensive and time consuming, particularly converting PDFs, so this allows selective tweaks to different parts of the process.
- pdf_to_md.py - Given a directory, converts PDF files to Markdown. uses "documents" folder by default. FYI, time consuming and expensive, but gives great results.
- md_to_chunks.py - Uses markdown's structure to chunk, and also adds LLM generated context to the chunks. Time consuming, but cheap and improves results.
- chunks_to_embeddings.py - Generates embeddings for the chunks and stores them in JSON files.
- embeddings_to_weaviate.py - Stores the embeddings and documents into Weaviate for retrieval.
