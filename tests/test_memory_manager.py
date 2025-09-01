# tests/test_memory_manager.py
import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

# Import the components we're testing
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory_manager import MemoryManager
from models.memory_item import MemoryItem
from config.settings import LSTMConfig
from core.gate_networks import LSTMGateNetwork

class TestMemoryManager:
    """Test suite for MemoryManager class"""
    
    @pytest.fixture
    async def mock_config(self):
        """Create mock configuration"""
        config = MagicMock(spec=LSTMConfig)
        config.gate_network.input_size = 90
        config.gate_network.hidden_size = 64
        config.gate_network.dropout_rate = 0.1
        config.memory.forget_threshold = 0.3
        config.memory.input_threshold = 0.7
        config.memory.output_threshold = 0.5
        config.memory.max_context_tokens = 2000
        config.memory.max_memories_per_context = 20
        config.memory.time_decay_rate = 0.01
        config.memory.access_decay_rate = 0.02
        config.astra_db.endpoint = "mock://test"
        config.astra_db.token = "mock_token"
        return config
    
    @pytest.fixture
    async def memory_manager(self, mock_config):
        """Create MemoryManager instance with mocked dependencies"""
        with patch('core.memory_manager.AstraDBConnector') as mock_connector, \
             patch('core.memory_manager.MemoryStore') as mock_store, \
             patch('core.memory_manager.LSTMGateNetwork') as mock_gates, \
             patch('core.memory_manager.ContextAssembler') as mock_assembler:
            
            # Setup mock gate network
            mock_gate_instance = Mock()
            mock_gate_instance.get_gate_decisions.return_value = {
                'input': {'score': 0.8, 'decision': True},
                'forget': {'score': 0.2, 'decision': False},
                'output': {'score': 0.6, 'decision': True}
            }
            mock_gates.return_value = mock_gate_instance
            
            # Setup mock database connector
            mock_connector_instance = Mock()
            mock_connector_instance.connect = AsyncMock()
            mock_connector.return_value = mock_connector_instance
            
            # Setup mock memory store
            mock_store_instance = Mock()
            mock_store_instance.save_memory = AsyncMock(return_value=True)
            mock_store_instance.get_user_memories = AsyncMock(return_value=[])
            mock_store.return_value = mock_store_instance
            
            # Setup mock context assembler
            mock_assembler_instance = Mock()
            mock_assembler_instance.assemble_context = AsyncMock(return_value=([], {}))
            mock_assembler.return_value = mock_assembler_instance
            
            manager = MemoryManager(mock_config)
            
            # Store references to mocks for test access
            manager._mock_gate_network = mock_gate_instance
            manager._mock_memory_store = mock_store_instance
            manager._mock_context_assembler = mock_assembler_instance
            
            return manager
    
    @pytest.fixture
    def sample_feature_vector(self):
        """Create sample 90-dimensional feature vector"""
        return np.random.rand(90).tolist()
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample 768-dimensional embeddings"""
        return np.random.rand(768).tolist()
    
    @pytest.fixture
    def sample_memory_item(self, sample_feature_vector):
        """Create sample memory item"""
        return MemoryItem.create_new(
            user_id="test_user_123",
            content="I had a great meeting with Sarah today about the new project",
            feature_vector=sample_feature_vector,
            initial_gate_scores={'input': 0.8, 'forget': 0.2, 'output': 0.6},
            memory_type="conversation"
        )

class TestMemoryCreation:
    """Test memory creation functionality"""
    
    @pytest.mark.asyncio
    async def test_process_new_entry_creates_memory_when_important(self, memory_manager, sample_feature_vector, sample_embeddings):
        """Test that important entries create memories"""
        # Arrange
        user_id = "test_user_123"
        content = "I learned something important today about machine learning"
        
        # Act
        result = await memory_manager.process_new_entry(
            user_id=user_id,
            feature_vector=sample_feature_vector,
            content=content,
            embeddings=sample_embeddings
        )
        
        # Assert
        assert result is not None
        assert isinstance(result, MemoryItem)
        assert result.user_id == user_id
        assert result.content_summary == content
        assert result.feature_vector == sample_feature_vector
        assert result.memory_type in ["conversation", "event", "emotion", "insight"]
        
        # Verify gate network was called
        memory_manager._mock_gate_network.get_gate_decisions.assert_called_once_with(sample_feature_vector)
        
        # Verify memory was saved
        memory_manager._mock_memory_store.save_memory.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_new_entry_skips_unimportant_entries(self, memory_manager, sample_feature_vector, sample_embeddings):
        """Test that unimportant entries are not stored as memories"""
        # Arrange - mock low input score
        memory_manager._mock_gate_network.get_gate_decisions.return_value = {
            'input': {'score': 0.3, 'decision': False},
            'forget': {'score': 0.1, 'decision': False},
            'output': {'score': 0.4, 'decision': False}
        }
        
        # Act
        result = await memory_manager.process_new_entry(
            user_id="test_user",
            feature_vector=sample_feature_vector,
            content="Just a random unimportant thought",
            embeddings=sample_embeddings
        )
        
        # Assert
        assert result is None
        memory_manager._mock_memory_store.save_memory.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_memory_type_classification(self, memory_manager, sample_feature_vector, sample_embeddings):
        """Test automatic memory type classification"""
        test_cases = [
            ("I feel really sad about losing my job", "emotion"),
            ("I have a meeting tomorrow at 3pm", "event"),
            ("I realized that persistence is key to success", "insight"),
            ("Had lunch with my friend today", "conversation")
        ]
        
        for content, expected_type in test_cases:
            result = await memory_manager.process_new_entry(
                user_id="test_user",
                feature_vector=sample_feature_vector,
                content=content,
                embeddings=sample_embeddings
            )
            
            assert result is not None
            assert result.memory_type == expected_type

class TestMemoryRetrieval:
    """Test memory retrieval and context assembly"""
    
    @pytest.mark.asyncio
    async def test_get_relevant_context_returns_memories(self, memory_manager, sample_feature_vector):
        """Test retrieval of relevant memories for context"""
        # Arrange - setup mock memories
        mock_memories = [
            Mock(spec=MemoryItem, feature_vector=sample_feature_vector, gate_scores={'output': 0.8}),
            Mock(spec=MemoryItem, feature_vector=sample_feature_vector, gate_scores={'output': 0.7}),
        ]
        memory_manager._mock_memory_store.get_user_memories.return_value = mock_memories
        memory_manager._mock_context_assembler.assemble_context.return_value = (
            mock_memories[:1], 
            {"reason": "success", "token_count": 500}
        )
        
        # Act
        result_memories, metadata = await memory_manager.get_relevant_context(
            user_id="test_user",
            query="How's work going?",
            query_features=sample_feature_vector
        )
        
        # Assert
        assert len(result_memories) == 1
        assert metadata["reason"] == "success"
        assert metadata["token_count"] == 500
        
        # Verify context assembler was called
        memory_manager._mock_context_assembler.assemble_context.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_relevant_context_with_no_memories(self, memory_manager, sample_feature_vector):
        """Test context retrieval when no memories exist"""
        # Arrange
        memory_manager._mock_memory_store.get_user_memories.return_value = []
        
        # Act
        result_memories, metadata = await memory_manager.get_relevant_context(
            user_id="test_user",
            query="How's work going?",
            query_features=sample_feature_vector
        )
        
        # Assert
        assert len(result_memories) == 0
        assert metadata["reason"] == "no_memories_found"
    
    @pytest.mark.asyncio
    async def test_memory_cache_functionality(self, memory_manager, sample_feature_vector):
        """Test that memory cache works correctly"""
        # Arrange
        mock_memories = [Mock(spec=MemoryItem)]
        memory_manager._mock_memory_store.get_user_memories.return_value = mock_memories
        
        user_id = "test_user"
        
        # First call - should hit database
        await memory_manager._get_user_memories(user_id)
        assert memory_manager._mock_memory_store.get_user_memories.call_count == 1
        
        # Second call - should hit cache
        await memory_manager._get_user_memories(user_id)
        assert memory_manager._mock_memory_store.get_user_memories.call_count == 1  # No additional call
        
        # Force refresh - should hit database again
        await memory_manager._get_user_memories(user_id, force_refresh=True)
        assert memory_manager._mock_memory_store.get_user_memories.call_count == 2

class TestMemoryUpdates:
    """Test memory update functionality"""
    
    @pytest.mark.asyncio
    async def test_update_memory_importance_positive_feedback(self, memory_manager, sample_memory_item):
        """Test memory importance update with positive feedback"""
        # Arrange
        memory_manager._mock_memory_store.get_memory.return_value = sample_memory_item
        memory_manager._mock_memory_store.update_memory_scores = AsyncMock()
        
        initial_importance = sample_memory_item.importance_score
        
        # Act - positive feedback
        await memory_manager.update_memory_importance(
            memory_id=sample_memory_item.id,
            user_feedback=0.9,  # High satisfaction
            interaction_data={"context_relevance": 0.8}
        )
        
        # Assert
        assert sample_memory_item.importance_score > initial_importance
        memory_manager._mock_memory_store.update_memory_scores.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_memory_importance_negative_feedback(self, memory_manager, sample_memory_item):
        """Test memory importance update with negative feedback"""
        # Arrange
        memory_manager._mock_memory_store.get_memory.return_value = sample_memory_item
        memory_manager._mock_memory_store.update_memory_scores = AsyncMock()
        
        initial_importance = sample_memory_item.importance_score
        
        # Act - negative feedback
        await memory_manager.update_memory_importance(
            memory_id=sample_memory_item.id,
            user_feedback=0.2,  # Low satisfaction
            interaction_data={}
        )
        
        # Assert
        assert sample_memory_item.importance_score < initial_importance
        memory_manager._mock_memory_store.update_memory_scores.assert_called_once()

class TestMemoryDecay:
    """Test memory decay functionality"""
    
    @pytest.mark.asyncio
    async def test_memory_decay_reduces_importance(self, memory_manager):
        """Test that memory decay reduces importance scores"""
        # Arrange - create old memory
        old_memory = Mock(spec=MemoryItem)
        old_memory.feature_vector = np.random.rand(90).tolist()
        old_memory.gate_scores = {'forget': 0.3}
        old_memory.importance_score = 0.8
        old_memory.created_at = datetime.now() - timedelta(days=30)
        old_memory.last_accessed = datetime.now() - timedelta(days=15)
        old_memory.apply_decay = Mock()
        old_memory.id = "test_memory_id"
        
        memory_manager._mock_memory_store.get_user_memories.return_value = [old_memory]
        memory_manager._mock_memory_store.batch_update_scores = AsyncMock()
        
        # Setup gate network mock for forget score
        memory_manager._mock_gate_network.get_gate_decisions.return_value = {
            'forget': {'score': 0.4, 'decision': False}
        }
        
        # Act
        await memory_manager._decay_user_memories("test_user")
        
        # Assert
        old_memory.apply_decay.assert_called_once()
        memory_manager._mock_memory_store.batch_update_scores.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_memory_decay_single_user(self, memory_manager):
        """Test memory decay for single user"""
        # Arrange
        memory_manager._decay_user_memories = AsyncMock()
        
        # Act
        await memory_manager.run_memory_decay(user_id="test_user")
        
        # Assert
        memory_manager._decay_user_memories.assert_called_once_with("test_user")
    
    @pytest.mark.asyncio
    async def test_run_memory_decay_all_users(self, memory_manager):
        """Test memory decay for all users"""
        # Arrange
        mock_user_ids = ["user1", "user2", "user3"]
        memory_manager._mock_memory_store.get_all_user_ids = AsyncMock(return_value=mock_user_ids)
        memory_manager._decay_user_memories = AsyncMock()
        
        # Act
        await memory_manager.run_memory_decay()
        
        # Assert
        assert memory_manager._decay_user_memories.call_count == len(mock_user_ids)

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_process_new_entry_handles_exceptions(self, memory_manager, sample_feature_vector, sample_embeddings):
        """Test that exceptions during memory creation are handled gracefully"""
        # Arrange - mock gate network to raise exception
        memory_manager._mock_gate_network.get_gate_decisions.side_effect = Exception("Network error")
        
        # Act
        result = await memory_manager.process_new_entry(
            user_id="test_user",
            feature_vector=sample_feature_vector,
            content="Test content",
            embeddings=sample_embeddings
        )
        
        # Assert
        assert result is None  # Should return None when error occurs
    
    @pytest.mark.asyncio
    async def test_get_relevant_context_handles_database_errors(self, memory_manager, sample_feature_vector):
        """Test context retrieval handles database errors"""
        # Arrange
        memory_manager._mock_memory_store.get_user_memories.side_effect = Exception("Database error")
        
        # Act
        result_memories, metadata = await memory_manager.get_relevant_context(
            user_id="test_user",
            query="Test query",
            query_features=sample_feature_vector
        )
        
        # Assert
        assert len(result_memories) == 0
        assert metadata["reason"] == "error"
        assert "error" in metadata
    
    @pytest.mark.asyncio
    async def test_update_memory_with_nonexistent_memory(self, memory_manager):
        """Test updating nonexistent memory is handled gracefully"""
        # Arrange
        memory_manager._mock_memory_store.get_memory.return_value = None
        
        # Act & Assert - should not raise exception
        await memory_manager.update_memory_importance(
            memory_id="nonexistent_id",
            user_feedback=0.8,
            interaction_data={}
        )

class TestMemoryManagerIntegration:
    """Integration tests for MemoryManager"""
    
    @pytest.mark.asyncio
    async def test_full_memory_lifecycle(self, memory_manager, sample_feature_vector, sample_embeddings):
        """Test complete memory lifecycle: create, retrieve, update, decay"""
        user_id = "integration_test_user"
        
        # 1. Create memory
        memory = await memory_manager.process_new_entry(
            user_id=user_id,
            feature_vector=sample_feature_vector,
            content="Important learning about AI",
            embeddings=sample_embeddings
        )
        assert memory is not None
        
        # 2. Setup for retrieval
        memory_manager._mock_memory_store.get_user_memories.return_value = [memory]
        memory_manager._mock_context_assembler.assemble_context.return_value = (
            [memory], {"reason": "success"}
        )
        
        # 3. Retrieve memory
        retrieved_memories, metadata = await memory_manager.get_relevant_context(
            user_id=user_id,
            query="What did I learn about AI?",
            query_features=sample_feature_vector
        )
        assert len(retrieved_memories) == 1
        assert metadata["reason"] == "success"
        
        # 4. Update memory based on feedback
        memory_manager._mock_memory_store.get_memory.return_value = memory
        memory_manager._mock_memory_store.update_memory_scores = AsyncMock()
        
        await memory_manager.update_memory_importance(
            memory_id=memory.id,
            user_feedback=0.9,
            interaction_data={"context_relevance": 0.8}
        )
        
        # 5. Run decay process
        memory_manager._mock_memory_store.get_user_memories.return_value = [memory]
        memory_manager._mock_memory_store.batch_update_scores = AsyncMock()
        
        await memory_manager._decay_user_memories(user_id)
        
        # Verify all operations completed without errors
        assert True  # If we reach here, the full lifecycle worked

class TestMemoryManagerStats:
    """Test statistics and monitoring functionality"""
    
    def test_get_stats_returns_complete_info(self, memory_manager):
        """Test that get_stats returns comprehensive statistics"""
        # Act
        stats = memory_manager.get_stats()
        
        # Assert
        required_keys = [
            'memories_created', 'memories_retrieved', 'context_assemblies',
            'cache_hits', 'cache_misses', 'cached_users', 'gate_thresholds', 'config'
        ]
        
        for key in required_keys:
            assert key in stats
        
        assert isinstance(stats['gate_thresholds'], dict)
        assert isinstance(stats['config'], dict)

# Fixtures for running tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])