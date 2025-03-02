import re
from typing import Tuple, List, Set
import nltk
import string
import os

# Ensure NLTK data directory exists
nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Initialize NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")

# Import tokenization tools - with fallback if punkt isn't available
try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except Exception:
    print("NLTK resources not available, using basic tokenization")
    NLTK_AVAILABLE = False
    
    # Define fallback tokenization function
    def word_tokenize(text):
        """Basic fallback tokenizer if NLTK is not available"""
        # Remove punctuation and convert to lowercase
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator).lower()
        # Split on whitespace
        return text.split()

class FinancialGuardrail:
    """
    Implements an input-side guardrail for financial questions.
    """
    
    def __init__(self):
        """Initialize the financial guardrail with relevant terms and patterns."""
        # Financial terms that indicate a relevant financial question
        self.financial_terms = {
            'revenue', 'profit', 'loss', 'earnings', 'eps', 'income', 'statement', 
            'balance', 'sheet', 'cash', 'flow', 'dividend', 'equity', 'asset', 
            'liability', 'debt', 'capital', 'expenditure', 'capex', 'margin', 
            'growth', 'quarter', 'annual', 'fiscal', 'year', 'financial', 
            'stock', 'share', 'shareholder', 'stakeholder', 'investor', 
            'investment', 'market', 'cap', 'value', 'valuation', 'ratio', 
            'p/e', 'price-to-earnings', 'return', 'roi', 'roe', 'roa', 
            'ebitda', 'ebit', 'sales', 'cost', 'expense', 'tax', 'depreciation',
            'amortization', 'liquidity', 'solvency', 'performance', 'forecast',
            'outlook', 'guidance', 'estimate', 'projection', 'target', 'trend',
            'increase', 'decrease', 'gain', 'decline', 'rise', 'fall', 'report',
            'audit', 'accounting', 'budget', 'operation', 'strategy', 'management',
            'executive', 'CEO', 'CFO', 'COO', 'board', 'director'
        }
        
        # Patterns that indicate financial questions
        self.financial_patterns = [
            r'(q[1-4]|quarter|annual|fiscal)',
            r'(20\d{2}|19\d{2})',  # Year patterns
            r'(\$|€|£|¥)\s*\d+',   # Currency amounts
            r'\d+\s*%',            # Percentages
            r'(million|billion|m|b|k)',  # Monetary units
        ]
        
        # Patterns that indicate harmful or off-topic questions
        self.harmful_patterns = [
            r'(hack|steal|fraud|illegal|cheat|manipulate|insider)',
            r'(password|credential|login|account)',
            r'(porn|sex|dating|drug)',
            r'(crypto|bitcoin|ethereum|nft)'
        ]
        
        # Load stopwords if available, otherwise use a basic set
        if NLTK_AVAILABLE:
            try:
                self.stopwords = set(stopwords.words('english'))
            except Exception as e:
                print(f"Error loading stopwords: {e}")
                # Fallback to basic stopwords
                self.stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 
                                 'as', 'what', 'when', 'where', 'how', 'why', 'who', 
                                 'which', 'this', 'that', 'these', 'those', 'then', 
                                 'just', 'so', 'than', 'such', 'both', 'through', 
                                 'about', 'for', 'is', 'of', 'while', 'during', 'to'}
        else:
            # Basic stopwords list
            self.stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 
                             'as', 'what', 'when', 'where', 'how', 'why', 'who', 
                             'which', 'this', 'that', 'these', 'those', 'then', 
                             'just', 'so', 'than', 'such', 'both', 'through', 
                             'about', 'for', 'is', 'of', 'while', 'during', 'to'}
    
    def _contains_financial_term(self, question: str) -> bool:
        """Check if the question contains financial terms."""
        # Tokenize and normalize the question
        tokens = word_tokenize(question.lower())
        tokens = [token for token in tokens if token.isalnum() and token not in self.stopwords]
        
        # Check for financial terms
        for token in tokens:
            if token in self.financial_terms:
                return True
        
        # Check for financial n-grams (terms with multiple words)
        for n in range(2, 4):  # Check for 2 and 3-word terms
            ngrams = self._get_ngrams(tokens, n)
            for ngram in ngrams:
                if ngram in self.financial_terms:
                    return True
        
        return False
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Set[str]:
        """Generate n-grams from a list of tokens."""
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.add(ngram)
        return ngrams
    
    def _matches_financial_pattern(self, question: str) -> bool:
        """Check if the question matches financial patterns."""
        for pattern in self.financial_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return True
        return False
    
    def _is_harmful(self, question: str) -> bool:
        """Check if the question contains harmful patterns."""
        for pattern in self.harmful_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return True
        return False
    
    def validate_input(self, question: str) -> Tuple[bool, str]:
        """
        Validate if the input question is a valid financial question.
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check for empty questions
        if not question or question.strip() == "":
            return False, "Question cannot be empty."
        
        # Check for very short questions
        if len(question.strip()) < 5:
            return False, "Question is too short."
        
        # Check for harmful content
        if self._is_harmful(question):
            return False, "Question contains inappropriate or potentially harmful content."
        
        # Check if the question is about financial topics
        is_financial = self._contains_financial_term(question) or self._matches_financial_pattern(question)
        
        if not is_financial:
            return False, "This doesn't seem to be a financial question."
        
        return True, "Valid financial question."
