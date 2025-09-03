# database/astra_connector.py
import sys
import os

# Add project root to Python path  
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import json
import uuid
from dataclasses import asdict

from astrapy import DataAPIClient
from astrapy.exceptions import DataAPIException

from config.settings import AstraDBConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AstraDBConnector:
    """
    Manages connection and operations with Astra DB using Data API
    Handles both vector collections and regular document storage
    """
    
    def __init__(self, config: AstraDBConfig):
        self.config = config
        self.client = None
        self.db = None
        self.collections = {}
        self.is_connected = False
        
        # Connection stats
        self.stats = {
            'operations_executed': 0,
            'operation_failures': 0,
            'connection_failures': 0,
            'documents_inserted': 0,
            'documents_updated': 0,
            'documents_retrieved': 0
        }
    
    async def connect(self) -> bool:
        """Establish connection to Astra DB"""
        try:
            logger.info(f"Connecting to Astra DB: {self.config.endpoint}")
            
            # Initialize client
            self.client = DataAPIClient(self.config.token)
            
            # Get database reference
            self.db = self.client.get_database_by_api_endpoint(
                self.config.endpoint,
                keyspace=self.config.keyspace
            )
            
            # Test connection by listing collections
            collections = self.db.list_collection_names()
            logger.info(f"Found collections: {collections}")
            
            # Initialize collection references
            self._initialize_collections()
            
            self.is_connected = True
            logger.info("Successfully connected to Astra DB")
            return True
            
        except Exception as e:
            self.stats['connection_failures'] += 1
            logger.error(f"Failed to connect to Astra DB: {e}")
            return False
    
    def _initialize_collections(self):
        """Initialize collection references"""
        collection_names = [
            "memory_embeddings",
            "rl_experiences", 
            "chat_embeddings"
        ]
        
        for name in collection_names:
            try:
                self.collections[name] = self.db.get_collection(name)
            except Exception as e:
                logger.warning(f"Collection {name} not available: {e}")
    
    async def close(self):
        """Close connection (astrapy handles this automatically)"""
        self.is_connected = False
        self.collections.clear()
        logger.info("Closed Astra DB connection")
    
    # Memory operations
    async def save_memory(self, memory_data: Dict[str, Any]) -> bool:
        """Save memory to database - FIXED for timezone issues"""
        try:
            collection = self.collections.get("memory_embeddings")
            if not collection:
                collection = self.db.get_collection("memory_embeddings")
            
            # Ensure all datetime fields are timezone-aware
            memory_doc = self._prepare_memory_document(memory_data)
            
            # Insert document
            result = collection.insert_one(memory_doc)
            
            if result.inserted_id:
                self.stats['operations_executed'] += 1
                self.stats['documents_inserted'] += 1
                logger.debug(f"Saved memory: {result.inserted_id}")
                return True
            else:
                self.stats['operation_failures'] += 1
                return False
                
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error saving memory: {e}")
            return False
    
    async def get_user_memories(self, user_id: str, limit: int = 1000) -> List[Dict]:
        """Get all memories for a user"""
        try:
            collection = self.collections.get("memory_embeddings")
            if not collection:
                collection = self.db.get_collection("memory_embeddings")
            
            # Query user memories, sorted by creation date (newest first)
            cursor = collection.find(
                {"user_id": user_id},
                limit=limit,
                sort={"created_at": -1}
            )
            
            memories = list(cursor)
            self.stats['operations_executed'] += 1
            self.stats['documents_retrieved'] += len(memories)
            
            logger.debug(f"Retrieved {len(memories)} memories for user {user_id}")
            return memories
            
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error getting user memories: {e}")
            return []
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict]:
        """Get specific memory by ID"""
        try:
            collection = self.collections.get("memory_embeddings")
            if not collection:
                collection = self.db.get_collection("memory_embeddings")
            
            memory = collection.find_one({"id": memory_id})
            self.stats['operations_executed'] += 1
            
            if memory:
                self.stats['documents_retrieved'] += 1
            
            return memory
            
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error getting memory by ID: {e}")
            return None
    
    async def update_memory_scores(self, memory_id: str, gate_scores: Dict, importance_score: float):
        """Update memory scores"""
        try:
            collection = self.collections.get("memory_embeddings")
            if not collection:
                collection = self.db.get_collection("memory_embeddings")
            
            update_data = {
                "$set": {
                    "gate_scores": json.dumps(gate_scores),
                    "importance_score": importance_score,
                    "last_accessed": datetime.now(timezone.utc).isoformat()
                }
            }
            
            result = collection.update_one(
                {"id": memory_id},
                update_data
            )
            
            self.stats['operations_executed'] += 1
            self.stats['documents_updated'] += 1
            logger.debug(f"Updated memory scores: {memory_id}")
            
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error updating memory scores: {e}")
    
    async def update_access_tracking(self, memory_id: str, last_accessed: datetime, access_frequency: int):
        """Update memory access tracking - ADDED missing method"""
        try:
            collection = self.collections.get("memory_embeddings")
            if not collection:
                collection = self.db.get_collection("memory_embeddings")
            
            # Ensure timezone-aware datetime
            if last_accessed.tzinfo is None:
                last_accessed = last_accessed.replace(tzinfo=timezone.utc)
            
            update_data = {
                "$set": {
                    "last_accessed": last_accessed.isoformat(),
                    "access_frequency": access_frequency
                }
            }
            
            result = collection.update_one(
                {"id": memory_id},
                update_data
            )
            
            self.stats['operations_executed'] += 1
            self.stats['documents_updated'] += 1
            logger.debug(f"Updated access tracking for memory: {memory_id}")
            
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error updating access tracking: {e}")
            raise  # Re-raise to let caller handle the error
    
    async def search_similar_memories(self, user_id: str, query_vector: List[float],
                                     limit: int = 10, min_similarity: float = 0.5) -> List[Dict]:
        """Search for similar memories using vector similarity"""
        try:
            collection = self.collections.get("memory_embeddings")
            if not collection:
                collection = self.db.get_collection("memory_embeddings")
            
            # Perform vector search
            cursor = collection.find(
                {
                    "user_id": user_id,
                    "$vector": {"$similarity": query_vector}
                },
                limit=limit,
                sort={"$vector": 1}
            )
            
            memories = list(cursor)
            
            # Filter by minimum similarity if needed
            if min_similarity > 0:
                memories = [m for m in memories if m.get('$similarity', 0) >= min_similarity]
            
            self.stats['operations_executed'] += 1
            self.stats['documents_retrieved'] += len(memories)
            
            logger.debug(f"Found {len(memories)} similar memories for user {user_id}")
            return memories
            
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error searching similar memories: {e}")
            return []
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete memory by ID"""
        try:
            collection = self.collections.get("memory_embeddings")
            if not collection:
                collection = self.db.get_collection("memory_embeddings")
            
            result = collection.delete_one({"id": memory_id})
            
            self.stats['operations_executed'] += 1
            
            if result.deleted_count > 0:
                logger.debug(f"Deleted memory: {memory_id}")
                return True
            else:
                logger.warning(f"Memory not found for deletion: {memory_id}")
                return False
                
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error deleting memory: {e}")
            return False
    
    # RL Experience operations
    async def save_rl_experience(self, experience_data: Dict[str, Any]) -> bool:
        """Save RL experience to database"""
        try:
            collection = self.collections.get("rl_experiences")
            if not collection:
                collection = self.db.get_collection("rl_experiences")
            
            # Prepare document with timezone-aware datetimes
            experience_doc = self._prepare_rl_document(experience_data)
            
            result = collection.insert_one(experience_doc)
            
            if result.inserted_id:
                self.stats['operations_executed'] += 1
                self.stats['documents_inserted'] += 1
                logger.debug(f"Saved RL experience: {result.inserted_id}")
                return True
            else:
                self.stats['operation_failures'] += 1
                return False
                
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error saving RL experience: {e}")
            return False
    
    async def get_user_experiences(self, user_id: str, limit: int = 1000) -> List[Dict]:
        """Get RL experiences for a user"""
        try:
            collection = self.collections.get("rl_experiences")
            if not collection:
                collection = self.db.get_collection("rl_experiences")
            
            cursor = collection.find(
                {"user_id": user_id},
                limit=limit,
                sort={"created_at": -1}
            )
            
            experiences = list(cursor)
            self.stats['operations_executed'] += 1
            self.stats['documents_retrieved'] += len(experiences)
            
            return experiences
            
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error getting user experiences: {e}")
            return []
    
    async def get_recent_experiences(self, limit: int = 100, min_priority: float = 0.1) -> List[Dict]:
        """Get recent high-priority experiences"""
        try:
            collection = self.collections.get("rl_experiences")
            if not collection:
                collection = self.db.get_collection("rl_experiences")
            
            cursor = collection.find(
                {"priority": {"$gte": min_priority}},
                limit=limit,
                sort={"priority": -1, "created_at": -1}
            )
            
            experiences = list(cursor)
            self.stats['operations_executed'] += 1
            self.stats['documents_retrieved'] += len(experiences)
            
            return experiences
            
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error getting recent experiences: {e}")
            return []
    
    # Chat embedding operations
    async def save_chat_embedding(self, embedding_data: Dict[str, Any]) -> bool:
        """Save chat embedding to database"""
        try:
            collection = self.collections.get("chat_embeddings")
            if not collection:
                collection = self.db.get_collection("chat_embeddings")
            
            # Ensure timestamp is timezone-aware
            if 'timestamp' in embedding_data and isinstance(embedding_data['timestamp'], datetime):
                if embedding_data['timestamp'].tzinfo is None:
                    embedding_data['timestamp'] = embedding_data['timestamp'].replace(tzinfo=timezone.utc)
                embedding_data['timestamp'] = embedding_data['timestamp'].isoformat()
            
            result = collection.insert_one(embedding_data)
            
            if result.inserted_id:
                self.stats['operations_executed'] += 1
                self.stats['documents_inserted'] += 1
                return True
            else:
                self.stats['operation_failures'] += 1
                return False
                
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error saving chat embedding: {e}")
            return False
    
    async def get_chat_embeddings(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Get chat embeddings for a user"""
        try:
            collection = self.collections.get("chat_embeddings")
            if not collection:
                collection = self.db.get_collection("chat_embeddings")
            
            cursor = collection.find(
                {"user_id": user_id},
                limit=limit,
                sort={"timestamp": -1}
            )
            
            embeddings = list(cursor)
            self.stats['operations_executed'] += 1
            self.stats['documents_retrieved'] += len(embeddings)
            
            return embeddings
            
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error getting chat embeddings: {e}")
            return []
    
    # Helper methods
    def _prepare_memory_document(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare memory data for document storage - ensure datetimes are timezone-aware (UTC)"""
        doc = memory_data.copy()

        # Fix ALL datetime fields, not just specific ones
        for key, value in doc.items():
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
                doc[key] = value.isoformat()

        # Convert gate_scores dict to JSON string (Astra expects text)
        if 'gate_scores' in doc and isinstance(doc['gate_scores'], dict):
            doc['gate_scores'] = json.dumps(doc['gate_scores'])

        # Convert context_needed dict to JSON string (Astra expects text)  
        if 'context_needed' in doc and isinstance(doc['context_needed'], dict):
            doc['context_needed'] = json.dumps(doc['context_needed'])

        return doc
    
    def _prepare_rl_document(self, experience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare RL experience for document storage"""
        doc = experience_data.copy()
        
        # DON'T add _id - let Astra handle document IDs automatically
        # Just ensure the 'id' field exists for our own reference
        if 'id' not in doc:
            doc['id'] = str(uuid.uuid4())
        
        # Fix ALL datetime fields
        for key, value in doc.items():
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
                doc[key] = value.isoformat()
        
        return doc
    
    # Statistics and monitoring
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            'is_connected': self.is_connected,
            'stats': self.stats.copy(),
            'collections': list(self.collections.keys()),
            'endpoint': self.config.endpoint,
            'keyspace': self.config.keyspace
        }
    
    def reset_stats(self):
        """Reset connection statistics"""
        self.stats = {
            'operations_executed': 0,
            'operation_failures': 0,
            'connection_failures': 0,
            'documents_inserted': 0,
            'documents_updated': 0,
            'documents_retrieved': 0
        }
    
    # Health check
    async def health_check(self) -> bool:
        """Perform health check on database connection"""
        try:
            if not self.is_connected:
                return False
            
            # Try to list collections as a simple connectivity test
            collections = self.db.list_collection_names()
            logger.debug(f"Health check passed. Collections: {collections}")
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False