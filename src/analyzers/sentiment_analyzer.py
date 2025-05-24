from typing import Dict, Any, List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
import numpy as np
from scipy.special import softmax
import logging
import warnings

# Suppress the specific warning about unused weights
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*")

class SentimentAnalyzer:
    """Handles sentiment analysis and keyword extraction for text data."""
    
    def __init__(self, model_type: str = 'vader'):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_type (str): Type of sentiment analyzer to use ('vader' or 'roberta')
        """
        self.model_type = model_type.lower()
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
        """Initialize the selected sentiment analyzer."""
        if self.model_type == 'vader':
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        elif self.model_type == 'roberta':
            try:
                self.model_path = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.config = AutoConfig.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                
                # Move model to GPU if available
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Using device: {self.device}")
                self.model = self.model.to(self.device)

            except Exception as e:
                logging.error(f"Error loading RoBERTa model: {str(e)}")
                raise
        else:
            raise ValueError("model_type must be either 'vader' or 'roberta'")
            
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for RoBERTa model.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
        
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using the selected model.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, Any]: Sentiment analysis results
        """
        if not isinstance(text, str) or not text.strip():
            return {'label': 'neutral', 'score': 1.0}
            
        try:
            if self.model_type == 'vader':
                return self._analyze_vader(text)
            else:
                return self._analyze_roberta(text)
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {str(e)}")
            return {'label': 'neutral', 'score': 1.0}
            
    def _analyze_vader(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using VADER."""
        scores = self.sentiment_analyzer.polarity_scores(text)
        
        compound_score = scores['compound']
        if compound_score >= 0.05:
            label = 'positive'
        elif compound_score <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
            
        return {
            'label': label,
            'score': abs(compound_score),
            'details': {
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'compound': scores['compound']
            }
        }
        
    def _analyze_roberta(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using RoBERTa."""
        text = self.preprocess_text(text)
        
        # Tokenize and get model output
        encoded_input = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        # Move input tensors to the same device as the model
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        with torch.no_grad():
            output = self.model(**encoded_input)
        
        scores = output[0][0].detach().cpu().numpy()  # Move back to CPU for numpy operations
        scores = softmax(scores)
        
        # Get the highest scoring label
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        label_idx = ranking[0]
        label = self.config.id2label[label_idx].lower()
        score = float(scores[label_idx])
        
        
        return {
            'label': label,
            'score': score,
            'details': {
                'positive': float(scores[2]),  # RoBERTa order: negative, neutral, positive
                'negative': float(scores[0]),
                'neutral': float(scores[1]),
                'compound': score
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