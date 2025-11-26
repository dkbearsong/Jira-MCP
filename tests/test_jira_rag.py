import asyncio
import json
import hashlib
import numpy as np
import pytest

import jira_RAG as jr


class FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, text):
        # deterministic small vector based on text hash
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
        # If query shares a word with any document, mark as close (distance 0.1)
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

        # Return up to n_results
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


class FakeHttpClient(FakePersistentClient):
    pass


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Replace the heavy external dependencies with fakes
    monkeypatch.setattr(jr, 'SentenceTransformer', FakeSentenceTransformer)
    monkeypatch.setattr(jr.chromadb, 'PersistentClient', FakePersistentClient)
    monkeypatch.setattr(jr.chromadb, 'HttpClient', FakeHttpClient)


def test_add_issue_and_query_relevant():
    tool = jr.JiraRAGTool(persist_directory='./test_chroma_db')

    sample_issue = {
        'summary': 'Login button not working on mobile',
        'description': 'Users report unresponsive login button on iOS devices',
        'status': 'Open',
        'priority': 'High',
        'issuetype': 'Bug',
        'assignee': 'john.doe',
        'reporter': 'jane.smith',
        'created': '2025-01-15T10:00:00Z'
    }

    id_added = tool.add_jira_issue('PROJ-123', sample_issue)
    assert id_added == 'PROJ-123'

    # Query that should match the document text -> low distances -> relevant
    result = asyncio.run(tool.query_jira_rag(
        query='mobile login button not working on iOS',
        n_results=5,
        relevance_threshold=0.5
    ))

    assert result['success'] is True
    assert 'context' in result
    assert 'PROJ-123' in result['context']


def test_query_low_relevance():
    tool = jr.JiraRAGTool(persist_directory='./test_chroma_db_2')

    sample_issue = {
        'summary': 'Some unrelated issue',
        'description': 'Database migration note',
        'status': 'Closed',
        'priority': 'Low',
        'issuetype': 'Task',
        'assignee': 'ops',
        'reporter': 'infra',
        'created': '2025-02-01T10:00:00Z'
    }

    tool.add_jira_issue('PROJ-999', sample_issue)

    # Query that doesn't match -> distances high -> not relevant
    result = asyncio.run(tool.query_jira_rag(
        query='payment processing error in checkout',
        n_results=5,
        relevance_threshold=0.5
    ))

    assert result['success'] is False
    assert 'jira_search' in result['message'] or 'jira_search' in result.get('hint', '')
