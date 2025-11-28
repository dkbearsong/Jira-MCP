"""
Simple runnable test harness for `jira_RAG.py` that mocks the heavy dependencies
so you can run checks locally without an AI or ChromaDB server.

Run with:
    python run_jira_rag_test.py

It will print results for a few sample queries.
"""
import asyncio
import hashlib
import numpy as np
import jira_RAG as jr


class FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, text):
        h = int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)
        return np.array([float(h % 1000)])


class FakeCollection:
    def __init__(self):
        self.documents = []
        self.metadatas = []
        self.ids = []

    def add(self, documents, metadatas=None, ids=None):
        self.documents.extend(documents)
        if metadatas:
            self.metadatas.extend(metadatas)
        if ids:
            self.ids.extend(ids)

    def query(self, query_texts, n_results=5, include=None):
        q = query_texts[0]
        results = []
        distances = []
        metadatas = []
        for doc, meta in zip(self.documents, self.metadatas):
            score = 10.0
            for token in q.lower().split():
                if token and token in doc.lower():
                    score = 0.1
                    break
            results.append(doc)
            distances.append(score)
            metadatas.append(meta)

        return {
            'documents': [results[:n_results]],
            'metadatas': [metadatas[:n_results]],
            'distances': [distances[:n_results]]
        }


class FakePersistentClient:
    def __init__(self, path=None, settings=None):
        self._col = FakeCollection()

    def get_or_create_collection(self, name, metadata=None):
        return self._col


def main():
    """# Monkeypatch module dependencies
    jr.SentenceTransformer = FakeSentenceTransformer
    jr.chromadb.PersistentClient = FakePersistentClient
    jr.chromadb.HttpClient = FakePersistentClient
    """

    rag = jr.JiraRAGTool(persist_directory='/Users/dbearsong/Documents/GitHub/Jira-Pipeline/chroma')

    """sample_issue = {
        'summary': 'Login button not working on mobile',
        'description': 'Users report that the login button is unresponsive on iOS devices',
        'status': 'Open',
        'priority': 'High',
        'issuetype': 'Bug',
        'assignee': 'john.doe@example.com',
        'reporter': 'jane.smith@example.com',
        'created': '2025-01-15T10:00:00Z'
    }

    rag.add_jira_issue('PROJ-123', sample_issue)
    print('Added PROJ-123 to fake ChromaDB')"""

    async def run_queries():
        print('\n=== High relevance query ===')
        res1 = await rag.query_jira_rag(query=f'Tell me about uploading a custom profile picture so that I can personalize my account',relevance_threshold=0.55)
        print(res1)

        print('\n=== Low relevance query ===')
        res2 = await rag.query_jira_rag('payment gateway outage')
        print(res2)

    asyncio.run(run_queries())


if __name__ == '__main__':
    main()
