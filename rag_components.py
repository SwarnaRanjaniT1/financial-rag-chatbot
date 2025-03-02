from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub, HuggingFacePipeline
from langchain_community.retrievers import BM25Retriever
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
import torch
from typing import List, Tuple, Dict, Any, Optional
import os

class FinancialRAG:
    """
    A Retrieval-Augmented Generation (RAG) system designed for financial statements.
    Implements Hybrid Search (combining BM25 and dense vector search) as an advanced RAG technique.
    """
    
    def __init__(
        self, 
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "google/flan-t5-small",
        chunk_size: int = 512, 
        chunk_overlap: int = 50,
        device: str = "cpu"
    ):
        """
        Initialize the Financial RAG system.
        
        Args:
            embedding_model: The HuggingFace model to use for embeddings
            llm_model: The HuggingFace model to use for text generation
            chunk_size: Size of text chunks for document splitting
            chunk_overlap: Overlap between chunks
            device: The device to run models on (cpu or cuda)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device
        
        # Initialize embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # Initialize LLM
        self.init_llm(llm_model)
        
        # Initialize vectorstore
        self.vectorstore = None
        self.retriever = None
        
        # QA prompt template
        self.qa_prompt = PromptTemplate(
            template="""You are a financial analyst assistant tasked with answering questions about company financial statements.
            Use only the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Keep the answer concise and accurate.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:""",
            input_variables=["context", "question"]
        )
        
        # First-stage retrieval prompt (for query expansion)
        self.query_expansion_prompt = PromptTemplate(
            template="""Given the user question about financial statements, generate three different versions of the query that can help in retrieving relevant information.
            
            User question: {question}
            
            Rewritten queries:
            1.""",
            input_variables=["question"]
        )
    
    def init_llm(self, model_name: str):
        """Initialize the language model based on the model name."""
        try:
            if model_name.startswith("google/flan-t5"):
                # Load T5-based model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=512,
                    temperature=0.1,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    do_sample=True
                )
                self.llm = HuggingFacePipeline(pipeline=pipe)
            
            elif model_name.startswith("facebook/opt") or model_name.startswith("TinyLlama"):
                # Load causal language model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=512,
                    temperature=0.1,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    do_sample=True
                )
                self.llm = HuggingFacePipeline(pipeline=pipe)
            
            else:
                # Fallback to HuggingFaceHub if model not recognized
                self.llm = HuggingFaceHub(
                    repo_id=model_name,
                    model_kwargs={"temperature": 0.1, "max_length": 512}
                )
        except Exception as e:
            # Fallback to Flan-T5-small if there's an error
            print(f"Error loading model {model_name}: {e}")
            print("Falling back to google/flan-t5-small")
            
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/flan-t5-small",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.2
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
    
    def ingest_documents(self, file_paths: List[str]):
        """
        Process and ingest documents into the vector database.
        Sets up a hybrid retrieval system using both sparse (BM25) and dense (vector) retrieval.
        
        Args:
            file_paths: List of paths to PDF files
            
        Returns:
            Number of document chunks processed
        """
        all_docs = []
        
        # Load and split documents
        for file_path in file_paths:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add source information
            for doc in documents:
                doc.metadata["source_file"] = os.path.basename(file_path)
            
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            all_docs.extend(split_docs)
        
        # Create dense vector store
        self.vectorstore = FAISS.from_documents(all_docs, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Create sparse BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(all_docs)
        self.bm25_retriever.k = 4
        
        # Store the documents for BM25 retrieval
        self.all_docs = all_docs
        
        print(f"Ingested {len(all_docs)} document chunks")
        return len(all_docs)
    
    def _generate_query_expansions(self, question: str) -> List[str]:
        """
        Generate multiple query versions for the first stage of retrieval.
        
        Args:
            question: Original user question
            
        Returns:
            List of rewritten queries
        """
        try:
            # Use the LLM to generate query expansions
            expanded_query = self.llm(self.query_expansion_prompt.format(question=question))
            
            # Parse the result to get multiple queries
            expansions = expanded_query.strip().split("\n")
            queries = [question]  # Always include the original query
            
            for exp in expansions:
                # Remove numbers and other formatting
                clean_exp = exp.strip()
                if clean_exp and not clean_exp.isdigit():
                    # Remove any numbering (like "1.", "2.", etc.)
                    if "." in clean_exp[:2]:
                        clean_exp = clean_exp.split(".", 1)[1].strip()
                    queries.append(clean_exp)
            
            # Ensure we have at least the original query
            if len(queries) == 1:
                # If we couldn't generate expansions, create simple variations
                queries.append(f"financial statement {question}")
                queries.append(f"financial report {question}")
            
            return queries[:3]  # Limit to top 3 queries
        
        except Exception as e:
            print(f"Error in query expansion: {e}")
            # Return just the original query if there's an error
            return [question]
    
    def answer_question(self, question: str, top_k: int = 4) -> Tuple[str, List]:
        """
        Answer a question using hybrid retrieval (BM25 + dense vectors).
        Manually combines results from both retrievers for improved results.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Returns:
            Tuple of (answer, source_documents)
        """
        if not self.vectorstore or not hasattr(self, 'bm25_retriever'):
            return "Please load documents first.", []
        
        # Get documents from vector store (dense retrieval)
        vector_docs = self.vectorstore.similarity_search(question, k=top_k)
        
        # Get documents from BM25 (sparse retrieval)
        bm25_docs = self.bm25_retriever.get_relevant_documents(question)
        
        # Combine results from both retrievers
        all_docs = vector_docs + bm25_docs
        
        # Remove duplicate documents
        unique_docs = []
        seen_contents = set()
        for doc in all_docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)
        
        # Limit to top_k documents
        if len(unique_docs) > top_k:
            unique_docs = unique_docs[:top_k]
        
        # Format context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in unique_docs])
        
        # Create prompt with context and question
        prompt_input = self.qa_prompt.format(context=context, question=question)
        
        # Generate answer using the LLM
        answer = self.llm(prompt_input)
        
        # Clean up the answer
        answer = answer.strip()
        
        return answer, unique_docs
