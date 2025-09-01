# database/astra_connector.py
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# database/astra_connector.py (Updated for astrapy)
from astrapy import DataAPIClient
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import json
import uuid

from config.settings import AstraDBConfig

logger = logging.getLogger(__name__)

class AstraDBConnector:
    """
    Astra DB connector using astrapy (document-based)
    Much simpler and more reliable than cassandra-driver
    """
    
    def __init__(self, config: AstraDBConfig):
        self.config = config
        self.client = None
        self.db = None
        self.collections = {}
        
        # Connection state
        self.is_connected = False
        
        # Statistics
        self.stats = {
            'operations_executed': 0,
            'connection_attempts': 0,
            'connection_failures': 0,
            'operation_failures': 0,
            'last_operation_time': None,
            'uptime_start': datetime.now()
        }
    
    async def connect(self) -> bool:
        """Establish connection to Astra DB"""
        try:
            logger.info(f"Connecting to Astra DB: {self.config.endpoint}")
            self.stats['connection_attempts'] += 1
            
            # Initialize DataAPI client
            self.client = DataAPIClient(self.config.token)
            
            # Connect to database
            self.db = self.client.get_database_by_api_endpoint(
                self.config.endpoint,
                keyspace=self.config.keyspace
            )
            
            # Test connection by listing collections
            collections = self.db.list_collection_names()
            logger.info(f"Connected! Found collections: {collections}")
            
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
        """Save memory to database - FIXED for CQL tables"""
        try:
            collection = self.collections.get("memory_embeddings")
            if not collection:
                collection = self.db.get_collection("memory_embeddings")
            
            # DON'T call _prepare_memory_document - use data directly
            # The MemoryItem.to_dict() already formats it correctly
            memory_doc = memory_data  # Use as-is
            
            # Insert document
            result = collection.insert_one(memory_doc)
            
            if result.inserted_id:
                self.stats['operations_executed'] += 1
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
            
            memory = collection.find_one({"_id": memory_id})
            self.stats['operations_executed'] += 1
            
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
                    "gate_scores": gate_scores,
                    "importance_score": importance_score,
                    "last_accessed": datetime.now().isoformat()
                }
            }
            
            result = collection.update_one(
                {"_id": memory_id},
                update_data
            )
            
            self.stats['operations_executed'] += 1
            logger.debug(f"Updated memory scores: {memory_id}")
            
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error updating memory scores: {e}")
            raise
    
    async def update_access_tracking(self, memory_id: str, last_accessed: datetime, access_frequency: int):
        """Update memory access tracking"""
        try:
            collection = self.collections.get("memory_embeddings")
            if not collection:
                collection = self.db.get_collection("memory_embeddings")
            
            update_data = {
                "$set": {
                    "last_accessed": last_accessed.isoformat(),
                    "access_frequency": access_frequency
                }
            }
            
            result = collection.update_one(
                {"_id": memory_id},
                update_data
            )
            
            self.stats['operations_executed'] += 1
            
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error updating access tracking: {e}")
            raise
    
    async def batch_update_scores(self, updates: List[Tuple[str, Dict, float]]):
        """Batch update memory scores (astrapy doesn't have true batch, so we'll do sequential)"""
        try:
            successful_updates = 0
            
            for memory_id, gate_scores, importance_score in updates:
                try:
                    await self.update_memory_scores(memory_id, gate_scores, importance_score)
                    successful_updates += 1
                except Exception as e:
                    logger.warning(f"Failed to update memory {memory_id}: {e}")
            
            self.stats['operations_executed'] += 1
            logger.info(f"Batch updated {successful_updates}/{len(updates)} memories")
            
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error in batch update: {e}")
            raise
    
    async def get_all_user_ids(self) -> List[str]:
        """Get all unique user IDs"""
        try:
            collection = self.collections.get("memory_embeddings")
            if not collection:
                collection = self.db.get_collection("memory_embeddings")
            
            # Get distinct user IDs
            pipeline = [
                {"$group": {"_id": "$user_id"}},
                {"$project": {"user_id": "$_id", "_id": 0}}
            ]
            
            # Note: astrapy might not support aggregation, so fallback to simple approach
            try:
                cursor = collection.aggregate(pipeline)
                user_ids = [doc["user_id"] for doc in cursor]
            except:
                # Fallback: get all documents and extract unique user_ids
                all_docs = collection.find({}, projection={"user_id": 1})
                user_ids = list(set(doc["user_id"] for doc in all_docs if "user_id" in doc))
            
            self.stats['operations_executed'] += 1
            return user_ids
            
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error getting user IDs: {e}")
            return []
    
    # RL Experience operations
    async def save_rl_experience(self, experience_data: Dict[str, Any]) -> bool:
        """Save RL experience"""
        try:
            collection = self.collections.get("rl_experiences")
            if not collection:
                collection = self.db.get_collection("rl_experiences")
            
            # Prepare document
            exp_doc = self._prepare_rl_document(experience_data)
            
            result = collection.insert_one(exp_doc)
            
            if result.inserted_id:
                self.stats['operations_executed'] += 1
                return True
            else:
                self.stats['operation_failures'] += 1
                return False
                
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error saving RL experience: {e}")
            return False
    
    async def get_recent_experiences(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Get recent RL experiences"""
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
            
            return experiences
            
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error getting recent experiences: {e}")
            return []
    
    # Chat embeddings operations
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
            
            return embeddings
            
        except Exception as e:
            self.stats['operation_failures'] += 1
            logger.error(f"Error getting chat embeddings: {e}")
            return []
    
    # Helper methods
    def _prepare_memory_document(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare memory data for document storage - ensure datetimes are timezone-aware (UTC)"""
        from datetime import timezone
        doc = memory_data.copy()

        # Ensure all datetime fields are timezone-aware and in ISO format
        for field in ['created_at', 'last_accessed']:
            if field in doc and isinstance(doc[field], datetime):
                dt = doc[field]
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                doc[field] = dt.isoformat()

        # Ensure gate_scores is string (JSON) for CQL
        if 'gate_scores' in doc and isinstance(doc['gate_scores'], dict):
            doc['gate_scores'] = json.dumps(doc['gate_scores'])

        # Ensure context_needed is string (JSON) for CQL  
        if 'context_needed' in doc and isinstance(doc['context_needed'], dict):
            doc['context_needed'] = json.dumps(doc['context_needed'])

        return doc
    
    def _prepare_rl_document(self, experience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare RL experience for document storage"""
        doc = experience_data.copy()
        
        # Use experience ID as document _id
        if 'id' in doc:
            doc['_id'] = doc.pop('id')
        elif '_id' not in doc:
            doc['_id'] = str(uuid.uuid4())
        
        # Convert datetime to ISO string
        if 'created_at' in doc and isinstance(doc['created_at'], datetime):
            doc['created_at'] = doc['created_at'].isoformat()
        
        return doc
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            if not self.is_connected or not self.db:
                return {'status': 'disconnected'}
            
            # Test with a simple operation
            collections = self.db.list_collection_names()
            
            uptime = datetime.now() - self.stats['uptime_start']
            
            return {
                'status': 'healthy',
                'connected': True,
                'collections': collections,
                'keyspace': self.config.keyspace,
                'uptime_seconds': uptime.total_seconds(),
                'stats': self.stats.copy()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'connected': False,
                'error': str(e),
                'stats': self.stats.copy()
            }
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        uptime = datetime.now() - self.stats['uptime_start']
        success_rate = 1.0 - (self.stats['operation_failures'] / max(1, self.stats['operations_executed']))
        
        return {
            **self.stats,
            'uptime_seconds': uptime.total_seconds(),
            'operation_success_rate': success_rate,
            'is_connected': self.is_connected,
            'available_collections': len(self.collections)
        }

# Utility functions
async def create_astra_connector(endpoint: str, token: str, keyspace: str = "memory_db") -> AstraDBConnector:
    """Factory function to create and connect to Astra DB"""
    config = AstraDBConfig(
        endpoint=endpoint,
        token=token,
        keyspace=keyspace
    )
    
    connector = AstraDBConnector(config)
    success = await connector.connect()
    
    if not success:
        raise Exception("Failed to establish connection to Astra DB")
    
    return connector

async def test_astra_connection(endpoint: str, token: str) -> bool:
    """Test connection to Astra DB"""
    try:
        connector = await create_astra_connector(endpoint, token)
        health = await connector.health_check()
        await connector.close()
        return health['status'] == 'healthy'
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False