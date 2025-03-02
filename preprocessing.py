import os
import re
import pandas as pd
from typing import List, Dict, Any, Tuple
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class FinancialDocumentProcessor:
    """Class for preprocessing financial documents."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks for document splitting
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def load_documents(self, file_paths: List[str]) -> List[Any]:
        """
        Load documents from file paths.
        
        Args:
            file_paths: List of paths to documents
            
        Returns:
            List of loaded documents
        """
        all_docs = []
        
        for file_path in file_paths:
            # Check file extension
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() == '.pdf':
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif ext.lower() in ['.txt', '.md', '.csv']:
                loader = TextLoader(file_path)
                docs = loader.load()
            else:
                print(f"Unsupported file type: {file_path}")
                continue
            
            # Add source information
            for doc in docs:
                doc.metadata["source_file"] = os.path.basename(file_path)
            
            all_docs.extend(docs)
        
        return all_docs
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and special characters.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that aren't useful
        text = re.sub(r'[^\w\s.,;:()[\]{}"\'-]', '', text)
        
        # Normalize whitespace
        text = text.strip()
        
        return text
    
    def extract_financials(self, text: str) -> Dict[str, Any]:
        """
        Extract financial metrics from text.
        
        Args:
            text: Text containing financial information
            
        Returns:
            Dictionary of extracted financial metrics
        """
        metrics = {}
        
        # Extract revenue
        revenue_match = re.search(r'revenue[^\d]*(\$?[\d,.]+\s*(?:million|billion|m|b)?)', text, re.IGNORECASE)
        if revenue_match:
            metrics['revenue'] = revenue_match.group(1)
        
        # Extract profit/loss
        profit_match = re.search(r'(?:net income|profit|loss)[^\d]*(\$?[\d,.]+\s*(?:million|billion|m|b)?)', text, re.IGNORECASE)
        if profit_match:
            metrics['profit'] = profit_match.group(1)
        
        # Extract EPS
        eps_match = re.search(r'earnings per share|EPS[^\d]*(\$?[\d,.]+)', text, re.IGNORECASE)
        if eps_match:
            metrics['eps'] = eps_match.group(1)
        
        return metrics
    
    def process_documents(self, file_paths: List[str]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Process documents and extract financial information.
        
        Args:
            file_paths: List of paths to financial documents
            
        Returns:
            Tuple of (processed_documents, financial_metrics)
        """
        # Load documents
        documents = self.load_documents(file_paths)
        
        # Clean and extract financials
        financial_metrics = {}
        cleaned_docs = []
        
        for doc in documents:
            # Clean text
            cleaned_text = self.clean_text(doc.page_content)
            doc.page_content = cleaned_text
            
            # Extract financials
            metrics = self.extract_financials(cleaned_text)
            
            # Merge metrics with source information
            source = doc.metadata.get("source_file", "unknown")
            if source not in financial_metrics:
                financial_metrics[source] = {}
            
            for key, value in metrics.items():
                financial_metrics[source][key] = value
            
            cleaned_docs.append(doc)
        
        # Split documents
        split_docs = self.text_splitter.split_documents(cleaned_docs)
        
        return split_docs, financial_metrics
