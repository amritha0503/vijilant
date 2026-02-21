"""
RAG Engine – LangChain + ChromaDB pipeline for RBI policy retrieval.
Indexes local policy text files at startup and retrieves relevant clauses
per agent utterance during compliance analysis.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL = "models/gemini-embedding-001"
COLLECTION_RBI = "rbi_policies"
COLLECTION_CLIENT = "client_rules"


# ---------------------------------------------------------------------------
# Global retriever (initialized once at startup)
# ---------------------------------------------------------------------------

_rbi_vectorstore: Optional[Chroma] = None


def get_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=api_key,
    )


def _parse_clause_documents(policies_dir: str) -> list[Document]:
    """
    Parse each policy .txt file into one LangChain Document per clause.
    Each document has full metadata: clause_id, rule_name, source.
    This ensures RAG retrieval returns identifiable, structured clauses.
    """
    import re
    policy_path = Path(policies_dir)
    documents: list[Document] = []

    for txt_file in sorted(policy_path.glob("*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()

        for m in re.finditer(
            r"CLAUSE\s+([\w-]+):\s*(.+?)\n(.*?)(?=CLAUSE\s+[\w-]+:|\Z)",
            content,
            re.DOTALL,
        ):
            clause_id = m.group(1).strip()
            rule_name = m.group(2).strip()
            description = m.group(3).strip()
            # Page content = full clause text so the embedding captures all detail
            page_content = f"CLAUSE {clause_id}: {rule_name}\n{description}"
            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "clause_id": clause_id,
                        "rule_name": rule_name,
                        "source": txt_file.name,
                    },
                )
            )
        print(f"[RAG]   Parsed: {txt_file.name}")

    return documents


def initialize_policy_store(policies_dir: str, api_key: str) -> Chroma:
    """
    Build ChromaDB with ONE document per policy clause (not arbitrary text chunks).
    Each document has clause_id + rule_name in metadata so retrieval is precise.
    Called ONCE at FastAPI startup.
    """
    global _rbi_vectorstore

    embeddings = get_embeddings(api_key)
    persist_path = str(Path(CHROMA_PERSIST_DIR) / COLLECTION_RBI)

    # Load cached store if it exists AND uses per-clause structure
    if Path(persist_path).exists():
        print(f"[RAG] Loading existing ChromaDB from {persist_path}")
        store = Chroma(
            collection_name=COLLECTION_RBI,
            embedding_function=embeddings,
            persist_directory=persist_path,
        )
        count = store._collection.count()
        # Validate structure: peek at first doc metadata to check per-clause format
        if count > 0:
            sample = store._collection.peek(limit=1)
            meta = (sample.get("metadatas") or [{}])[0]
            if meta.get("clause_id"):
                print(f"[RAG] Loaded {count} per-clause policy documents from cache.")
                _rbi_vectorstore = store
                return _rbi_vectorstore
            else:
                print("[RAG] Cache uses old chunk format – rebuilding with per-clause index...")
                # Delete stale store so we rebuild below
                import shutil
                shutil.rmtree(persist_path, ignore_errors=True)

    # Build fresh per-clause store
    print(f"[RAG] Building per-clause policy vector store from: {policies_dir}")
    documents = _parse_clause_documents(policies_dir)

    if not documents:
        raise RuntimeError(f"No policy clauses found in {policies_dir}")

    print(f"[RAG] Embedding {len(documents)} clauses (one doc per clause)...")
    _rbi_vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_RBI,
        persist_directory=persist_path,
    )
    print(f"[RAG] Policy store built with {len(documents)} clauses → {persist_path}")
    return _rbi_vectorstore


def load_client_rules(client_config: dict, api_key: str) -> Optional[Chroma]:
    """
    Convert client config rules to LangChain Documents and store in ChromaDB.
    Returns a Chroma retriever or None if no custom rules exist.
    """
    custom_rules = client_config.get("custom_rules", [])
    risk_triggers = client_config.get("risk_triggers", [])

    if not custom_rules and not risk_triggers:
        return None

    embeddings = get_embeddings(api_key)
    documents: list[Document] = []

    # Add custom rules as documents
    for rule in custom_rules:
        content = (
            f"CLAUSE {rule.get('rule_id', 'CUSTOM-XX')}: {rule.get('rule_name', '')}\n"
            f"{rule.get('description', '')}"
        )
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "clause_id": rule.get("rule_id", "CUSTOM-XX"),
                    "rule_name": rule.get("rule_name", ""),
                    "source": "client_config",
                },
            )
        )

    # Add risk triggers as documents
    for trigger in risk_triggers:
        content = f"RISK TRIGGER: {trigger} — Any agent behaviour constituting '{trigger}' is a policy violation."
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "clause_id": "CLIENT-TRIGGER",
                    "rule_name": trigger,
                    "source": "client_config",
                },
            )
        )

    if not documents:
        return None

    client_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_CLIENT,
        # Use in-memory (no persist) – ephemeral per request
    )
    print(f"[RAG] Client rules vectorstore built with {len(documents)} entries.")
    return client_store


def get_all_policy_clauses(policies_dir: str) -> list[dict]:
    """
    Parse ALL clauses from every policy .txt file and return them as a list.
    Used to give the compliance engine the full policy set (not just RAG-retrieved).

    Returns list of:
    {
        "clause_id": "RBI-REC-01",
        "rule_name": "Permitted Calling Hours",
        "description": "...",
        "source": "rbi_recovery_guidelines.txt"
    }
    """
    import re
    policy_path = Path(policies_dir)
    all_clauses: list[dict] = []

    for txt_file in sorted(policy_path.glob("*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Match: CLAUSE CLAUSE_ID: RULE_NAME\n<description until next CLAUSE or EOF>
        pattern = re.finditer(
            r"CLAUSE\s+([\w-]+):\s*(.+?)\n(.*?)(?=CLAUSE\s+[\w-]+:|\Z)",
            content,
            re.DOTALL,
        )
        for m in pattern:
            clause_id = m.group(1).strip()
            rule_name = m.group(2).strip()
            description = m.group(3).strip()
            all_clauses.append({
                "clause_id": clause_id,
                "rule_name": rule_name,
                "description": description,
                "source": txt_file.name,
            })

    return all_clauses


def retrieve_relevant_clauses(
    transcript_threads: list[dict],
    api_key: str,
    client_config: Optional[dict] = None,
) -> list[dict]:
    """
    For each AGENT utterance, retrieve matching policy clauses from RBI store
    and optionally the client rule store.

    Returns a deduplicated list of:
    [
        {
            "clause_id": ...,
            "rule_name": ...,
            "description": ...,
            "source": ...
        }
    ]
    """
    global _rbi_vectorstore

    if _rbi_vectorstore is None:
        print("[RAG] WARNING: Policy store not initialized. Returning empty clauses.")
        return []

    # k=8: retrieve more candidates per agent turn; deduplication keeps only unique clauses
    rbi_retriever = _rbi_vectorstore.as_retriever(search_kwargs={"k": 8})

    client_store = None
    if client_config:
        client_store = load_client_rules(client_config, api_key)
    client_retriever = client_store.as_retriever(search_kwargs={"k": 4}) if client_store else None

    agent_messages = [
        t["message"]
        for t in transcript_threads
        if t.get("speaker", "").lower() == "agent" and t.get("message")
    ]

    if not agent_messages:
        # Fallback: query full transcript if no agent turns identified
        agent_messages = [
            t["message"] for t in transcript_threads if t.get("message")
        ]

    seen_clauses: set[str] = set()
    clauses: list[dict] = []

    for message in agent_messages:
        # Query RBI store
        try:
            rbi_docs = rbi_retriever.invoke(message)
            for doc in rbi_docs:
                meta = doc.metadata
                # Metadata now always has clause_id + rule_name (per-clause index)
                clause_id = meta.get("clause_id") or _extract_clause_id(doc.page_content)
                if clause_id not in seen_clauses:
                    seen_clauses.add(clause_id)
                    clauses.append(
                        {
                            "clause_id": clause_id,
                            "rule_name": meta.get("rule_name") or _extract_rule_name(doc.page_content),
                            "description": doc.page_content,
                            "source": meta.get("source", "rbi_policies"),
                        }
                    )
        except Exception as exc:
            print(f"[RAG] RBI retrieval error: {exc}")

        # Query client store
        if client_retriever:
            try:
                client_docs = client_retriever.invoke(message)
                for doc in client_docs:
                    meta = doc.metadata
                    clause_id = meta.get("clause_id", "CLIENT-XX")
                    key = f"CLIENT-{meta.get('rule_name', clause_id)}"
                    if key not in seen_clauses:
                        seen_clauses.add(key)
                        clauses.append(
                            {
                                "clause_id": clause_id,
                                "rule_name": meta.get("rule_name", "Client Rule"),
                                "description": doc.page_content,
                                "source": "client_config",
                            }
                        )
            except Exception as exc:
                print(f"[RAG] Client retrieval error: {exc}")

    print(f"[RAG] Retrieved {len(clauses)} unique relevant clauses from {len(agent_messages)} agent turns.")
    return clauses


def _extract_clause_id(text: str) -> str:
    """Extract CLAUSE ID from raw policy text chunk."""
    import re
    match = re.search(r"CLAUSE\s+([\w-]+):", text)
    return match.group(1) if match else "UNKNOWN"


def _extract_rule_name(text: str) -> str:
    """Extract rule name after CLAUSE ID."""
    import re
    match = re.search(r"CLAUSE\s+[\w-]+:\s*(.+)", text)
    if match:
        return match.group(1).strip()[:80]
    return "Policy Clause"
