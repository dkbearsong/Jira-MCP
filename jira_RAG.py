import os
import json
from typing import Optional, List, Dict, Any
import chromadb
from chromadb.config import Settings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
import numpy as np
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
class JiraRAGTool:
    """
    MCP tool for RAG-enabled Jira querying with ChromaDB.
    Supports both local SQLite and production-ready client-server mode.
    """
    
    def __init__(
        self,
        collection_name: str = "jira_issues",
        embedding_model: str = "msmarco-distilbert-base-v4",
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None,
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize the Jira RAG tool. Creates or connects to a ChromaDB collection, identifying if it is 
        running in local or client-server mode. Creates collection if it does not exist, otherwise connects to it.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model for embeddings
            chroma_host: ChromaDB server host (None for local SQLite)
            chroma_port: ChromaDB server port (None for local SQLite)
            persist_directory: Local directory for SQLite persistence
        """
        self.collection_name = collection_name
        # Allow operating without the sentence-transformers package by providing
        # a tiny fallback embedding model for local/offline tests.
        if SentenceTransformer is None:
            class _FallbackEmbedder:
                def __init__(self, *args, **kwargs):
                    pass

                def encode(self, text):
                    # return a deterministic tiny vector based on text
                    try:
                        import hashlib
                        import numpy as _np
                        h = int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)
                        return _np.array([float(h % 1000)])
                    except Exception:
                        return [0.0]

            self.embedding_model = _FallbackEmbedder()
        else:
            self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client based on environment
        if chroma_host and chroma_port:
            # Production: Client-Server mode
            self.client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port
            )
            # print(f"Connected to ChromaDB at {chroma_host}:{chroma_port}")
        else:
            # Development: Local persistent SQLite
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            # print(f"Using local ChromaDB at {persist_directory}")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Jira issues from webhooks"}
        )
        # print(self.collection)
    
    def calculate_relevance_score(
        self,
        query_embedding: np.ndarray,
        result_embeddings: List[np.ndarray],
        distances: List[float]
    ) -> float:
        """
        Calculate relevance score based on distance and semantic similarity. Uses cosine similarity by default.
        
        Args:
            query_embedding: Embedding of the user query
            result_embeddings: Embeddings of retrieved documents
            distances: Distance scores from ChromaDB
            
        Returns:
            Relevance score between 0 and 1
        """
        if not result_embeddings or not distances:
            return 0.0
        
        # Convert distances to similarity scores (lower distance = higher similarity)
        # Set to calculate similarity using Cosine similarity
        similarity_scores = [1 - d for d in distances]
        
        # Calculate average similarity
        avg_similarity = np.mean(similarity_scores)
        
        return float(avg_similarity)
    
    def is_context_relevant(
        self,
        query: str,
        results: Any,
        relevance_threshold: float = 0.6,
        min_results: int = 1
    ) -> tuple[bool, float, str]:
        """
        Verify if retrieved context is relevant enough to answer the query. Number of documents 
        and relevance score are checked against thresholds. At the end, returns a tuple indicating
        if context is relevant, the relevance score, and an explanation.
        
        Args:
            query: User's query
            results: Results from ChromaDB query (QueryResult or dict-like object)
            relevance_threshold: Minimum relevance score (0-1)
            min_results: Minimum number of results required
            
        Returns:
            Tuple of (is_relevant, relevance_score, explanation)
        """
        if not results or not results.get('documents'):
            return False, 0.0, "No documents found in ChromaDB"
        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results.get('distances') else []
        if len(documents) < min_results:
            return False, 0.0, f"Insufficient results: found {len(documents)}, need {min_results}"
        
        # Get query embedding and ensure numpy ndarray (handle torch/tf tensors)
        def _to_numpy_vector(x):
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    return x.cpu().detach().numpy()
            except Exception:
                pass

            # Avoid importing tensorflow to prevent "import could not be resolved" errors.
            # Instead, detect TensorFlow tensors via duck-typing (numpy() method) or by class metadata.
            try:
                if hasattr(x, "numpy") and callable(getattr(x, "numpy")):
                    return x.numpy()
            except Exception:
                pass

            try:
                mod = getattr(x.__class__, "__module__", "") or ""
                name = getattr(x.__class__, "__name__", "") or ""
                if mod.startswith("tensorflow") or name == "Tensor":
                    try:
                        return x.numpy()
                    except Exception:
                        pass
            except Exception:
                pass

            return np.asarray(x, dtype=float)

        raw_query_embedding = self.embedding_model.encode(query)
        query_embedding = _to_numpy_vector(raw_query_embedding)

        # Get result embeddings and ensure numpy ndarray for each
        result_embeddings = [
            _to_numpy_vector(self.embedding_model.encode(doc)) for doc in documents
        ]

        # Calculate relevance score
        relevance_score = self.calculate_relevance_score(
            query_embedding,
            result_embeddings,
            distances
        )
        
        is_relevant = relevance_score >= relevance_threshold
        
        explanation = (
            f"Relevance score: {relevance_score:.2f} "
            f"(threshold: {relevance_threshold}). "
            f"Found {len(documents)} document(s)."
        )
        
        return is_relevant, relevance_score, explanation
    
    async def query_jira_rag(
        self,
        query: str,
        n_results: int = 5,
        relevance_threshold: float = 0.55,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Main MCP tool function: Query Jira data using RAG with AI-directed fallback. Returns 
        either relevant cached data or instructions to use direct Jira search. First, it queries 
        ChromaDB for relevant cached data. Then, it checks if the retrieved context is relevant enough
        to answer the user's query based on a relevance score threshold. If relevant, it returns the
        cached data formatted for the AI. If not relevant, it returns a message instructing the AI 
        to use the jira_search tool for direct querying of Jira. When errors occur, it also instructs
        the AI to use jira_search.
        
        Args:
            query: User's natural language query
            n_results: Number of results to retrieve from ChromaDB
            relevance_threshold: Minimum relevance score for context
            include_metadata: Include metadata in response
            
        Returns:
            Dictionary containing results OR a message directing AI to use jira_search
        """
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            # Check context relevance
            is_relevant, score, explanation = self.is_context_relevant(
                query,
                results,
                relevance_threshold
            )
            if is_relevant:
                # Context is relevant - return RAG results with formatted context
                documents = results['documents'][0] if (results.get('documents') is not None and len(results['documents']) > 0) else [] # type: ignore
                metadatas = results['metadatas'][0] if (include_metadata and results.get('metadatas') is not None and len(results['metadatas']) > 0) else None # type: ignore
                
                # Format the context for the AI
                formatted_context = self._format_context_for_ai(documents, metadatas) # type: ignore
                
                return {
                    "success": True,
                    "source": "chromadb_rag",
                    "relevance_score": score,
                    "context": formatted_context,
                    "raw_data": {
                        "documents": documents,
                        "metadatas": metadatas,
                        "distances": results['distances'][0] if (results.get('distances') is not None and len(results['distances']) > 0) else None # type: ignore
                    }
                }
            
            else:
                # Context not relevant - return message for AI to use jira_search
                return {
                    "success": False,
                    "source": "chromadb_rag",
                    "relevance_score": score,
                    "message": (
                        f"The cached Jira data in ChromaDB is not sufficiently relevant "
                        f"to answer this query (relevance score: {score:.2f}, threshold: {relevance_threshold}). "
                        f"{explanation}\n\n"
                        f"Please use the jira_search tool instead to query Jira directly for up-to-date information. "
                        f"The jira_search tool can handle both specific issue lookups and JQL-based searches."
                    ),
                    "hint": self._generate_search_hint(query)
                }
                
        except Exception as e:
            return {
                "success": False,
                "source": "chromadb_rag",
                "error": str(e),
                "message": (
                    f"Failed to query ChromaDB: {str(e)}\n\n"
                    f"Please use the jira_search tool to query Jira directly instead."
                ),
                "hint": self._generate_search_hint(query)
            }
    
    def _format_context_for_ai(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]]
    ) -> str:
        """
        Format retrieved documents into a clear context string for the AI. Includes metadata 
        if available. Returns a structured string with issue keys, statuses, priorities, and 
        descriptions. First, adds a header indicating these are retrieved Jira issues. Then, for
        each document, it appends the issue key, status, priority from metadata (if available),
        and the document text itself.
        
        Args:
            documents: Retrieved document texts
            metadatas: Associated metadata
            
        Returns:
            Formatted context string
        """
        context_parts = ["Retrieved Jira issues from cache:\n"]
        
        for i, doc in enumerate(documents):
            context_parts.append(f"\n--- Issue {i+1} ---")
            if metadatas and i < len(metadatas):
                meta = metadatas[i]
                context_parts.append(f"Issue Key: {meta.get('issue_key', 'N/A')}")
                context_parts.append(f"Status: {meta.get('status', 'N/A')}")
                context_parts.append(f"Priority: {meta.get('priority', 'N/A')}")
            context_parts.append(f"\n{doc}")
        
        return "\n".join(context_parts)
    
    def _generate_search_hint(self, query: str) -> str:
        """
        Generate a helpful hint for the AI about what to search for. Analyzes the query for 
        Jira-specific entities like issue keys, statuses, priorities, and types. Constructs 
        a hint string summarizing the findings.
        
        Args:
            query: User's query
            
        Returns:
            Hint string for the AI
        """
        import re
        
        hints = []
        
        # Check for issue key
        issue_key_match = re.search(r'\b[A-Z]+-\d+\b', query.upper())
        if issue_key_match:
            hints.append(f"Detected issue key: {issue_key_match.group(0)}")
        
        # Check for status keywords
        status_keywords = ['open', 'closed', 'in progress', 'resolved', 'done']
        for status in status_keywords:
            if status in query.lower():
                hints.append(f"Query mentions status: '{status}'")
        
        # Check for priority keywords
        priority_keywords = ['high', 'low', 'medium', 'critical', 'blocker']
        for priority in priority_keywords:
            if priority in query.lower():
                hints.append(f"Query mentions priority: '{priority}'")
        
        # Check for issue type keywords
        type_keywords = ['bug', 'story', 'task', 'epic', 'subtask']
        for issue_type in type_keywords:
            if issue_type in query.lower():
                hints.append(f"Query mentions issue type: '{issue_type}'")
        
        if hints:
            return "Query analysis: " + "; ".join(hints)
        else:
            return "No specific Jira entities detected in query"