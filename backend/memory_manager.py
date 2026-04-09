import os
import json
import logging
import datetime
import collections
from typing import List, Dict, Any, Optional

log = logging.getLogger("ORIEN.Memory")

class MemoryManager:
    """
    Handles persistent storage for user profile and conversation history.
    Enables ORIEN to be a truly 'personal' assistant.
    """
    def __init__(self, filename: str = "memory_store.json"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(self.base_dir, filename)
        self.data = self._load()
        
        # VECTOR DB MEMORY
        self.doc_id = len(self.data.get("history", []))
        try:
            import chromadb
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.get_or_create_collection(name="orien_memory")
            self.use_vector_db = True
            
            # Sync existing history
            if self.data["history"]:
                docs, metas, ids = [], [], []
                for idx, item in enumerate(self.data["history"]):
                    docs.append(item["content"])
                    metas.append({"role": item["role"], "timestamp": item["timestamp"]})
                    ids.append(str(idx + 1))
                self.collection.add(documents=docs, metadatas=metas, ids=ids)
        except Exception as e:
            log.warning(f"ChromaDB Vector Memory unavailable: {e}")
            self.use_vector_db = False

    def _load(self) -> Dict[str, Any]:
        defaults = {
            "user_profile": {
                "name": "Member",
                "interests": [],
                "preferences": {
                    "talkative": True,
                    "communication_style": "Friendly",
                    "triggers": {},
                    "patterns": {}
                },
                "language": "en-US",
                "created_at": datetime.datetime.utcnow().isoformat()
            },
            "history": [],
            "stats": {
                "total_queries": 0,
                "total_encounters": 0,
                "emotion_freq": {}
            }
        }

        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Deep merge defaults for critical sections
                    for key in defaults:
                        if key not in data:
                            data[key] = defaults[key]
                        elif isinstance(defaults[key], dict):
                            for subkey in defaults[key]:
                                if subkey not in data[key]:
                                    data[key][subkey] = defaults[key][subkey]
                    return data
            except Exception as e:
                log.error(f"Failed to load memory: {e}")
        return defaults
    
    def log_emotion(self, emotion: str):
        """Track general frequency of users emotional states for pattern prediction."""
        eco = self.data["stats"].get("emotion_freq", {})
        eco[emotion] = eco.get(emotion, 0) + 1
        self.data["stats"]["emotion_freq"] = eco
        self._save()

    def update_preference(self, key: str, value: Any):
        """Store specific user preferences like triggers or personality traits."""
        self.data["user_profile"]["preferences"][key] = value
        self._save()

    def _save(self):
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=4)
        except Exception as e:
            log.error(f"Failed to save memory: {e}")

    def update_profile(self, **kwargs):
        """Update fields in the user_profile (name, interests, etc.)"""
        for k, v in kwargs.items():
            if k in self.data["user_profile"]:
                self.data["user_profile"][k] = v
        self._save()

    def add_history(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Append a message to the conversation history (limit to last 50 for performance)."""
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        self.data["history"].append(entry)
        if len(self.data["history"]) > 50:
            self.data["history"].pop(0)
        
        if role == "user":
            self.data["stats"]["total_queries"] += 1
            
        self._save()

        # [V12 VECTOR DB] Inject into ChromaDB for semantic retrieval
        if hasattr(self, 'use_vector_db') and self.use_vector_db and content.strip():
            self.doc_id += 1
            try:
                self.collection.add(
                    documents=[content.strip()],
                    metadatas=[{"role": role, "timestamp": entry["timestamp"]}],
                    ids=[str(self.doc_id)]
                )
            except Exception as e:
                log.warning(f"Vector DB add error: {e}")

    def get_history(self, limit: int = 10) -> List[Dict]:
        """Fetch recent conversation history."""
        return self.data["history"][-limit:]
        
    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """[V12 VECTOR DB] Retrieve top-k semantic memories."""
        if not hasattr(self, 'use_vector_db') or not self.use_vector_db or not query.strip():
            return ""
        try:
            results = self.collection.query(query_texts=[query], n_results=k)
            if not results['documents'] or not results['documents'][0]: return ""
            context = []
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                context.append(f"[{meta['role'].upper()}]: {doc}")
            return "\n".join(context)
        except Exception as e:
            log.warning(f"Vector DB query error: {e}")
            return ""

    def get_profile(self) -> Dict:
        return self.data["user_profile"]

    def get_state_summary(self, limit: int = 20) -> str:
        """Aggregates recent states into a session context."""
        history = self.data["history"][-limit:]
        intents = [h.get("metadata", {}).get("intent") for h in history if h.get("metadata", {}).get("intent")]
        
        if not intents: return "Neutral Baseline"
        
        counts = collections.Counter(intents)
        dom = counts.most_common(1)[0][0]
        
        patterns = {
            "FLOW": "User has demonstrated sustained deep focus.",
            "STRESSED": "User appears to be under cognitive pressure recently.",
            "OVERWHELMED": "System detects potential task-overload for the user.",
            "DISTRACTED": "User's attention seems fragmented."
        }
        
        return f"SESSION PATTERN: {patterns.get(dom, 'Consistent calm state.')}"

    def increment_encounters(self):
        self.data["stats"]["total_encounters"] += 1
        self._save()

# Global singleton
memory = MemoryManager()
