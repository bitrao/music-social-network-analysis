from typing import Dict, Any, List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    """Handles sentiment analysis and keyword extraction for text data."""
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.setup_nltk()
        self.setup_sentiment_analyzer()
        
    def setup_nltk(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
            
    def setup_sentiment_analyzer(self):
        """Initialize the VADER sentiment analyzer."""
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using VADER.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, Any]: Sentiment analysis results
        """
        if not text.strip():
            return {'label': 'neutral', 'score': 1.0}
            
        # Get sentiment scores
        scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Determine the label based on compound score
        compound_score = scores['compound']
        if compound_score >= 0.05:
            label = 'positive'
        elif compound_score <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
            
        return {
            'label': label,
            'score': abs(compound_score),  # Use absolute value for score
            'details': {
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'compound': scores['compound']
            }
        }
        
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text (str): Text to extract keywords from
            
        Returns:
            List[str]: List of keywords
        """
        # Tokenize text
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        keywords = [word for word in tokens if word not in stop_words]
        
        return keywords
        
    def calculate_sentiment_metrics(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate sentiment metrics from processed items.
        
        Args:
            items (List[Dict[str, Any]]): List of items with sentiment data
            
        Returns:
            Dict[str, Any]: Sentiment metrics
        """
        total_items = len(items)
        if total_items == 0:
            return {
                'positive_percentage': 0,
                'negative_percentage': 0,
                'neutral_percentage': 0,
                'average_sentiment_score': 0,
                'average_compound_score': 0
            }
            
        sentiment_counts = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        
        total_sentiment_score = 0
        total_compound_score = 0
        
        for item in items:
            sentiment = item['sentiment']['label']
            sentiment_counts[sentiment] += 1
            total_sentiment_score += item['sentiment']['score']
            total_compound_score += item['sentiment']['details']['compound']
            
        return {
            'positive_percentage': (sentiment_counts['positive'] / total_items) * 100,
            'negative_percentage': (sentiment_counts['negative'] / total_items) * 100,
            'neutral_percentage': (sentiment_counts['neutral'] / total_items) * 100,
            'average_sentiment_score': total_sentiment_score / total_items,
            'average_compound_score': total_compound_score / total_items
        } 