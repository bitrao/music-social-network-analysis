import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import json
from datetime import datetime
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from transformers import pipeline

class DataProcessor:
    """Processes and cleans collected data from various platforms."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        self.setup_nltk()
        self.setup_sentiment_analyzer()
        
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
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
            
    def setup_sentiment_analyzer(self):
        """Initialize the sentiment analysis pipeline."""
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process collected data from all platforms.
        
        Args:
            data (Dict[str, Any]): Raw collected data
            
        Returns:
            Dict[str, Any]: Processed data
        """
        self.logger.info(f"Processing data for song {data['song_id']}")
        
        processed_data = {
            'song_id': data['song_id'],
            'collection_date': data['collection_date'],
            'time_range': data['time_range'],
            'platforms': {}
        }
        
        # Process data from each platform
        for platform, platform_data in data['platforms'].items():
            if platform_data['status'] == 'success':
                processed_data['platforms'][platform] = self._process_platform_data(
                    platform,
                    platform_data['data']
                )
            else:
                processed_data['platforms'][platform] = platform_data
                
        return processed_data
    
    def _process_platform_data(self, platform: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data from a specific platform.
        
        Args:
            platform (str): Platform name
            data (Dict[str, Any]): Platform-specific data
            
        Returns:
            Dict[str, Any]: Processed platform data
        """
        if platform == 'twitter':
            return self._process_twitter_data(data)
        # Add processing for other platforms as they are implemented
        # elif platform == 'instagram':
        #     return self._process_instagram_data(data)
        # etc.
        else:
            return data
            
    def _process_twitter_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Twitter data.
        
        Args:
            data (Dict[str, Any]): Raw Twitter data
            
        Returns:
            Dict[str, Any]: Processed Twitter data
        """
        processed_tweets = []
        
        for tweet in data['tweets']:
            # Clean tweet text
            cleaned_text = self._clean_text(tweet['text'])
            
            # Perform sentiment analysis
            sentiment = self._analyze_sentiment(cleaned_text)
            
            # Extract keywords
            keywords = self._extract_keywords(cleaned_text)
            
            processed_tweet = {
                'id': tweet['id'],
                'text': cleaned_text,
                'created_at': tweet['created_at'],
                'user': tweet['user'],
                'metrics': tweet['metrics'],
                'sentiment': sentiment,
                'keywords': keywords
            }
            
            processed_tweets.append(processed_tweet)
            
        # Calculate aggregated metrics
        engagement_metrics = self._calculate_engagement_metrics(processed_tweets)
        sentiment_metrics = self._calculate_sentiment_metrics(processed_tweets)
        
        return {
            'tweets': processed_tweets,
            'engagement_metrics': engagement_metrics,
            'sentiment_metrics': sentiment_metrics
        }
        
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing URLs, special characters, etc.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, Any]: Sentiment analysis results
        """
        if not text.strip():
            return {'label': 'neutral', 'score': 1.0}
            
        result = self.sentiment_analyzer(text)[0]
        return {
            'label': result['label'].lower(),
            'score': float(result['score'])
        }
        
    def _extract_keywords(self, text: str) -> List[str]:
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
        
    def _calculate_engagement_metrics(self, tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate engagement metrics from processed tweets.
        
        Args:
            tweets (List[Dict[str, Any]]): Processed tweets
            
        Returns:
            Dict[str, Any]: Engagement metrics
        """
        total_tweets = len(tweets)
        total_retweets = sum(t['metrics']['retweet_count'] for t in tweets)
        total_likes = sum(t['metrics']['favorite_count'] for t in tweets)
        
        return {
            'total_tweets': total_tweets,
            'total_retweets': total_retweets,
            'total_likes': total_likes,
            'average_engagement': (total_retweets + total_likes) / total_tweets if total_tweets > 0 else 0
        }
        
    def _calculate_sentiment_metrics(self, tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate sentiment metrics from processed tweets.
        
        Args:
            tweets (List[Dict[str, Any]]): Processed tweets
            
        Returns:
            Dict[str, Any]: Sentiment metrics
        """
        total_tweets = len(tweets)
        if total_tweets == 0:
            return {
                'positive_percentage': 0,
                'negative_percentage': 0,
                'neutral_percentage': 0,
                'average_sentiment_score': 0
            }
            
        sentiment_counts = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        
        total_sentiment_score = 0
        
        for tweet in tweets:
            sentiment = tweet['sentiment']['label']
            sentiment_counts[sentiment] += 1
            total_sentiment_score += tweet['sentiment']['score']
            
        return {
            'positive_percentage': (sentiment_counts['positive'] / total_tweets) * 100,
            'negative_percentage': (sentiment_counts['negative'] / total_tweets) * 100,
            'neutral_percentage': (sentiment_counts['neutral'] / total_tweets) * 100,
            'average_sentiment_score': total_sentiment_score / total_tweets
        }
        
    def save_processed_data(self, data: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """
        Save processed data to disk.
        
        Args:
            data (Dict[str, Any]): Processed data
            output_path (Optional[Path]): Path to save data to
            
        Returns:
            Path: Path to saved file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path("data/processed") / f"{data['song_id']}_processed_{timestamp}.json"
            
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        self.logger.info(f"Processed data saved to {output_path}")
        return output_path 