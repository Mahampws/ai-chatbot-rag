# AI Chatbot with RAG (Retrieval-Augmented Generation)

A customer-support chatbot that answers questions by retrieving relevant
passages from a knowledge base, then (optionally) generating a natural
answer with an LLM. This is the same core pattern used in production RAG
systems for customer support, internal docs search, and Q&A bots.

## How it works

1. **Chunking** — the knowledge base text is split into paragraphs.
2. **Vectorization** — each chunk is converted into a TF-IDF vector.
3. **Retrieval** — a user's question is vectorized and compared against all
   chunks using cosine similarity; the top matches are returned.
4. **Generation (optional)** — if an API key is set, the retrieved context
   is passed to an LLM to produce a conversational answer. Without a key,
   the bot returns the best-matching passage directly.

## Setup

```bash
pip install -r requirements.txt
python chatbot.py
```

## Example

```
You: what is your return policy
Bot: Our return policy allows customers to return any item within 30 days
     of purchase for a full refund. Items must be unused and in original
     packaging.
     (matched with 55% relevance)
```

## Customizing for a client project

- Replace `knowledge_base.txt` with the client's docs, FAQs, or product manuals.
- Swap TF-IDF for embedding-based retrieval (e.g. `sentence-transformers` +
  FAISS/Pinecone) for larger knowledge bases or semantic matching.
- Set `OPENAI_API_KEY` / use the Anthropic API to enable natural-language
  answer generation instead of raw passage retrieval.

## Tech stack

Python, scikit-learn (TF-IDF, cosine similarity), optional LLM API integration.
