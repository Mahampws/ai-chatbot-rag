"""
AI Chatbot with Retrieval-Augmented Generation (RAG)
-----------------------------------------------------
A lightweight RAG chatbot that retrieves the most relevant passages from a
knowledge base and uses them to answer user questions.

Two modes are supported:
1. Retrieval-only (default) - no API key needed. Returns the most relevant
   passage(s) from the knowledge base using TF-IDF + cosine similarity.
2. Generative (optional) - if an OPENAI_API_KEY is set, retrieved passages
   are passed to an LLM to generate a natural-language answer.

This design mirrors real production RAG systems: split docs into chunks,
embed/vectorize them, retrieve top-k matches for a query, then generate
a grounded answer.
"""

import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGChatbot:
    def __init__(self, knowledge_base_path: str, top_k: int = 2):
        self.top_k = top_k
        self.chunks = self._load_and_chunk(knowledge_base_path)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(self.chunks)

    def _load_and_chunk(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # Split into paragraphs (simple chunking strategy)
        chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
        return chunks

    def retrieve(self, query: str):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = scores.argsort()[::-1][: self.top_k]
        results = [(self.chunks[i], scores[i]) for i in top_indices if scores[i] > 0]
        return results

    def answer(self, query: str) -> str:
        results = self.retrieve(query)
        if not results:
            return "I couldn't find anything relevant in the knowledge base. Could you rephrase your question?"

        context = "\n\n".join(chunk for chunk, _ in results)

        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return self._generate_with_llm(query, context, api_key)

        # Retrieval-only fallback: return best-matching passage(s) directly
        best_chunk, best_score = results[0]
        return f"{best_chunk}\n\n(matched with {best_score:.0%} relevance)"

    def _generate_with_llm(self, query: str, context: str, api_key: str) -> str:
        """Optional: use an LLM to turn retrieved context into a natural answer."""
        try:
            from anthropic import Anthropic  # or swap for openai.OpenAI()

            client = Anthropic(api_key=api_key)
            prompt = (
                f"Answer the user's question using ONLY the context below.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}"
            )
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            return f"(LLM generation failed, falling back to retrieval)\n\n{context}\n\n[Error: {e}]"


def main():
    bot = RAGChatbot("knowledge_base.txt")
    print("RAG Chatbot ready. Type 'quit' to exit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in {"quit", "exit"}:
            break
        print(f"Bot: {bot.answer(query)}\n")


if __name__ == "__main__":
    main()
