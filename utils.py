import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.spatial.distance import cosine
import re

def get_confidence_score(question: str, answer: str, source_docs: List) -> float:
    """
    Calculate a confidence score for the answer based on multiple factors.
    
    Args:
        question: User's question
        answer: Generated answer
        source_docs: Source documents used for generation
        
    Returns:
        Confidence score between 0 and 1
    """
    # Factor 1: Length of the answer (too short might indicate uncertainty)
    length_score = min(1.0, len(answer) / 100.0)
    
    # Factor 2: Presence of uncertainty markers
    uncertainty_patterns = [
        r"I don't know",
        r"I'm not sure",
        r"uncertain",
        r"cannot determine",
        r"unclear",
        r"insufficient information",
        r"unable to find",
        r"not specified"
    ]
    
    uncertainty_score = 1.0
    for pattern in uncertainty_patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            uncertainty_score = 0.3
            break
    
    # Factor 3: Consistency across source documents
    consistency_score = 0.5  # Default mid-point
    
    if len(source_docs) > 1:
        # Check overlap between answer content and source documents
        answer_lower = answer.lower()
        doc_content_overlap = [
            sum(1 for word in doc.page_content.lower().split() if word in answer_lower) / max(1, len(doc.page_content.split()))
            for doc in source_docs
        ]
        
        # Higher score if multiple documents support the answer
        consistency_score = min(1.0, sum(doc_content_overlap) / len(source_docs))
    
    # Factor 4: Presence of specific financial information (dates, numbers, percentages)
    financial_patterns = [
        r'\$\d+', r'\d+%', r'\d{4}',  # Dollar amounts, percentages, years
        r'(million|billion|trillion)',  # Monetary units
        r'(Q[1-4]|quarter)',  # Quarters
        r'(increased|decreased|grew|declined)',  # Financial movements
        r'(revenue|profit|loss|earnings|income|dividend)'  # Financial terms
    ]
    
    financial_match_count = 0
    for pattern in financial_patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            financial_match_count += 1
    
    financial_score = min(1.0, financial_match_count / 4.0)  # Cap at 1.0
    
    # Calculate the weighted average of all factors
    weights = {
        'length': 0.1,
        'uncertainty': 0.3,
        'consistency': 0.3,
        'financial': 0.3
    }
    
    weighted_score = (
        weights['length'] * length_score +
        weights['uncertainty'] * uncertainty_score +
        weights['consistency'] * consistency_score +
        weights['financial'] * financial_score
    )
    
    # Ensure score is between 0 and 1
    return max(0.0, min(1.0, weighted_score))


def extract_metrics_from_text(text: str) -> Dict[str, Any]:
    """
    Extract financial metrics from text.
    
    Args:
        text: Text containing financial information
        
    Returns:
        Dictionary of financial metrics
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
    
    # Extract year-over-year growth
    growth_match = re.search(r'(?:growth|increase|decrease)[^\d]*(\d+(?:\.\d+)?%)', text, re.IGNORECASE)
    if growth_match:
        metrics['growth'] = growth_match.group(1)
        
    return metrics


def format_financial_answer(answer: str) -> str:
    """
    Format an answer to highlight financial information.
    
    Args:
        answer: Raw answer text
        
    Returns:
        Formatted answer with highlighted financial information
    """
    # Highlight dollar amounts
    answer = re.sub(r'(\$\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion|m|b|t))?)', r'**\1**', answer, flags=re.IGNORECASE)
    
    # Highlight percentages
    answer = re.sub(r'(\d+(?:\.\d+)?%)', r'**\1**', answer)
    
    # Highlight financial years
    answer = re.sub(r'(FY\s*\d{4}|\d{4}\s*fiscal year)', r'**\1**', answer, flags=re.IGNORECASE)
    
    # Highlight quarters
    answer = re.sub(r'(Q[1-4]\s*\d{4}|quarter\s*\d\s*\d{4})', r'**\1**', answer, flags=re.IGNORECASE)
    
    return answer


def detect_financial_entities(text: str) -> List[Dict[str, Any]]:
    """
    Detect financial entities in text.
    
    Args:
        text: Input text
        
    Returns:
        List of detected financial entities with their types
    """
    entities = []
    
    # Detect monetary values
    money_pattern = r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|trillion|m|b|t)?'
    for match in re.finditer(money_pattern, text, re.IGNORECASE):
        amount = match.group(1).replace(',', '')
        unit = match.group(2) or ''
        
        # Normalize units
        multiplier = 1
        if unit.lower() in ['million', 'm']:
            multiplier = 1_000_000
        elif unit.lower() in ['billion', 'b']:
            multiplier = 1_000_000_000
        elif unit.lower() in ['trillion', 't']:
            multiplier = 1_000_000_000_000
            
        value = float(amount) * multiplier
        
        entities.append({
            'type': 'money',
            'text': match.group(0),
            'value': value,
            'start': match.start(),
            'end': match.end()
        })
    
    # Detect percentages
    percentage_pattern = r'(\d+(?:\.\d+)?)\s*%'
    for match in re.finditer(percentage_pattern, text):
        entities.append({
            'type': 'percentage',
            'text': match.group(0),
            'value': float(match.group(1)),
            'start': match.start(),
            'end': match.end()
        })
    
    # Detect dates (years, quarters)
    date_patterns = [
        (r'(Q[1-4])\s*(\d{4})', 'quarter'),  # Q1 2022
        (r'(FY)\s*(\d{4})', 'fiscal_year'),  # FY 2022
        (r'(\d{4})', 'year')  # 2022
    ]
    
    for pattern, date_type in date_patterns:
        for match in re.finditer(pattern, text):
            entities.append({
                'type': date_type,
                'text': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
    
    return entities
