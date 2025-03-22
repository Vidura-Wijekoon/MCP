import os
import unittest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.personal_fitness_assistant.main import (
    bs4_extractor,
    load_fitness_docs,
    split_documents,
    create_vectorstore,
    load_vectorstore,
    fitness_query_tool,
    query_fitness_assistant
)


class TestPersonalFitnessAssistant(unittest.TestCase):
    
    def test_bs4_extractor(self):
        """Test the HTML content extraction function."""
        html = "<html><body><article>Test content</article></body></html>"
        result = bs4_extractor(html)
        self.assertEqual(result, "Test content")
    
    @patch("src.personal_fitness_assistant.main.RecursiveUrlLoader")
    def test_load_fitness_docs(self, mock_loader):
        """Test loading fitness documents from URLs."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.load.return_value = [Document(page_content="Test document")]
        mock_loader.return_value = mock_instance
        
        # Test with default URLs
        docs = load_fitness_docs()
        self.assertEqual(len(docs), 3)  # 3 default URLs
        self.assertEqual(docs[0].page_content, "Test document")
        
        # Test with custom URLs
        custom_urls = ["https://example.com/fitness"]
        docs = load_fitness_docs(urls=custom_urls)
        self.assertEqual(len(docs), 1)  # 1 custom URL
        self.assertEqual(docs[0].page_content, "Test document")
    
    def test_split_documents(self):
        """Test splitting documents into chunks."""
        docs = [Document(page_content="Test document with some content" * 100)]
        splits = split_documents(docs)
        self.assertGreater(len(splits), 0)
    
    @patch("src.personal_fitness_assistant.main.SKLearnVectorStore.from_documents")
    def test_create_vectorstore(self, mock_from_documents):
        """Test creating a vector store from document chunks."""
        # Setup mock
        mock_vectorstore = MagicMock()
        mock_from_documents.return_value = mock_vectorstore
        
        # Test
        docs = [Document(page_content="Test document")]
        vectorstore = create_vectorstore(docs, persist=False)
        self.assertEqual(vectorstore, mock_vectorstore)
    
    @patch("src.personal_fitness_assistant.main.SKLearnVectorStore")
    def test_load_vectorstore(self, mock_vectorstore_class):
        """Test loading a vector store from disk."""
        # Setup mock
        mock_vectorstore = MagicMock()
        mock_vectorstore_class.return_value = mock_vectorstore
        
        # Test
        vectorstore = load_vectorstore()
        self.assertEqual(vectorstore, mock_vectorstore)
    
    @patch("src.personal_fitness_assistant.main.load_vectorstore")
    def test_fitness_query_tool(self, mock_load_vectorstore):
        """Test the fitness query tool."""
        # Setup mocks
        mock_vectorstore = MagicMock()
        mock_retriever = MagicMock()
        mock_doc = Document(page_content="Fitness information")
        mock_retriever.invoke.return_value = [mock_doc]
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_load_vectorstore.return_value = mock_vectorstore
        
        # Test
        with patch("os.path.exists", return_value=True):
            result = fitness_query_tool("benefits of exercise")
            self.assertIn("DOCUMENT 1", result)
            self.assertIn("Fitness information", result)
    
    @patch("src.personal_fitness_assistant.main.create_augmented_llm")
    def test_query_fitness_assistant(self, mock_create_augmented_llm):
        """Test querying the fitness assistant."""
        # Setup mock
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Exercise is beneficial for health."
        mock_llm.invoke.return_value = mock_response
        mock_create_augmented_llm.return_value = mock_llm
        
        # Test
        result = query_fitness_assistant("benefits of exercise")
        self.assertEqual(result, "Exercise is beneficial for health.")


if __name__ == "__main__":
    unittest.main()