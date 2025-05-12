import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import networkx as nx
from bertopic import BERTopic
import plotly.graph_objects as go
import plotly.express as px

class SongAnalyzer:
    """Analyzes processed data to extract insights about song performance."""
    
    def __init__(self):
        """Initialize the song analyzer."""
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        self.setup_models()
        
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def setup_models(self):
        """Initialize analysis models."""
        # Topic modeling
        self.topic_model = BERTopic(
            language="english",
            calculate_probabilities=True,
            verbose=True
        )
        
        # TF-IDF vectorizer for keyword extraction
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def analyze_song(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze processed data for a song.
        
        Args:
            data (Dict[str, Any]): Processed data
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        self.logger.info(f"Analyzing data for song {data['song_id']}")
        
        analysis_results = {
            'song_id': data['song_id'],
            'analysis_date': datetime.now().isoformat(),
            'time_range': data['time_range'],
            'platforms': {}
        }
        
        # Analyze data from each platform
        for platform, platform_data in data['platforms'].items():
            if platform_data['status'] == 'success':
                analysis_results['platforms'][platform] = self._analyze_platform_data(
                    platform,
                    platform_data['data']
                )
            else:
                analysis_results['platforms'][platform] = platform_data
                
        # Generate cross-platform insights
        analysis_results['cross_platform_insights'] = self._generate_cross_platform_insights(
            analysis_results['platforms']
        )
        
        return analysis_results
        
    def _analyze_platform_data(self, platform: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data from a specific platform.
        
        Args:
            platform (str): Platform name
            data (Dict[str, Any]): Platform-specific data
            
        Returns:
            Dict[str, Any]: Platform-specific analysis results
        """
        if platform == 'twitter':
            return self._analyze_twitter_data(data)
        # Add analysis for other platforms as they are implemented
        # elif platform == 'instagram':
        #     return self._analyze_instagram_data(data)
        # etc.
        else:
            return data
            
    def _analyze_twitter_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Twitter data.
        
        Args:
            data (Dict[str, Any]): Processed Twitter data
            
        Returns:
            Dict[str, Any]: Twitter analysis results
        """
        # Extract text for analysis
        texts = [tweet['text'] for tweet in data['tweets']]
        
        # Perform topic modeling
        topics, _ = self.topic_model.fit_transform(texts)
        topic_info = self.topic_model.get_topic_info()
        
        # Extract keywords
        tfidf_matrix = self.tfidf.fit_transform(texts)
        feature_names = self.tfidf.get_feature_names_out()
        keywords = self._extract_top_keywords(tfidf_matrix, feature_names)
        
        # Analyze engagement patterns
        engagement_patterns = self._analyze_engagement_patterns(data['tweets'])
        
        # Analyze sentiment trends
        sentiment_trends = self._analyze_sentiment_trends(data['tweets'])
        
        return {
            'topics': {
                'topic_info': topic_info.to_dict(orient='records'),
                'topic_keywords': self.topic_model.get_topics()
            },
            'keywords': keywords,
            'engagement_patterns': engagement_patterns,
            'sentiment_trends': sentiment_trends,
            'metrics': {
                'engagement': data['engagement_metrics'],
                'sentiment': data['sentiment_metrics']
            }
        }
        
    def _extract_top_keywords(self, tfidf_matrix, feature_names, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Extract top keywords from TF-IDF matrix.
        
        Args:
            tfidf_matrix: TF-IDF matrix
            feature_names: Feature names
            top_n (int): Number of top keywords to extract
            
        Returns:
            List[Dict[str, Any]]: Top keywords with their scores
        """
        # Calculate average TF-IDF scores for each term
        avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Get top keywords
        top_indices = np.argsort(avg_scores)[-top_n:][::-1]
        
        return [
            {
                'keyword': feature_names[i],
                'score': float(avg_scores[i])
            }
            for i in top_indices
        ]
        
    def _analyze_engagement_patterns(self, tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze engagement patterns in tweets.
        
        Args:
            tweets (List[Dict[str, Any]]): Processed tweets
            
        Returns:
            Dict[str, Any]: Engagement pattern analysis
        """
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(tweets)
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Group by hour and calculate engagement metrics
        hourly_engagement = df.groupby(df['created_at'].dt.hour).agg({
            'metrics': {
                'retweet_count': 'sum',
                'favorite_count': 'sum'
            }
        }).reset_index()
        
        # Calculate engagement velocity
        engagement_velocity = self._calculate_engagement_velocity(df)
        
        return {
            'hourly_engagement': hourly_engagement.to_dict(orient='records'),
            'engagement_velocity': engagement_velocity
        }
        
    def _calculate_engagement_velocity(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate engagement velocity (rate of change).
        
        Args:
            df (pd.DataFrame): DataFrame with engagement data
            
        Returns:
            Dict[str, float]: Engagement velocity metrics
        """
        # Sort by time
        df = df.sort_values('created_at')
        
        # Calculate time differences
        time_diffs = df['created_at'].diff().dt.total_seconds()
        
        # Calculate engagement differences
        retweet_diffs = df['metrics'].apply(lambda x: x['retweet_count']).diff()
        favorite_diffs = df['metrics'].apply(lambda x: x['favorite_count']).diff()
        
        # Calculate velocities
        retweet_velocity = (retweet_diffs / time_diffs).mean()
        favorite_velocity = (favorite_diffs / time_diffs).mean()
        
        return {
            'retweet_velocity': float(retweet_velocity),
            'favorite_velocity': float(favorite_velocity)
        }
        
    def _analyze_sentiment_trends(self, tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment trends in tweets.
        
        Args:
            tweets (List[Dict[str, Any]]): Processed tweets
            
        Returns:
            Dict[str, Any]: Sentiment trend analysis
        """
        # Convert to DataFrame
        df = pd.DataFrame(tweets)
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Group by hour and calculate sentiment metrics
        hourly_sentiment = df.groupby(df['created_at'].dt.hour).agg({
            'sentiment': {
                'label': lambda x: x.value_counts().to_dict(),
                'score': 'mean'
            }
        }).reset_index()
        
        # Calculate sentiment volatility
        sentiment_volatility = df['sentiment'].apply(lambda x: x['score']).std()
        
        return {
            'hourly_sentiment': hourly_sentiment.to_dict(orient='records'),
            'sentiment_volatility': float(sentiment_volatility)
        }
        
    def _generate_cross_platform_insights(self, platform_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insights across all platforms.
        
        Args:
            platform_analyses (Dict[str, Any]): Analysis results from all platforms
            
        Returns:
            Dict[str, Any]: Cross-platform insights
        """
        # Aggregate engagement metrics
        total_engagement = {
            'tweets': 0,
            'retweets': 0,
            'likes': 0,
            'comments': 0
        }
        
        # Aggregate sentiment metrics
        total_sentiment = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        
        # Collect keywords from all platforms
        all_keywords = []
        
        for platform, analysis in platform_analyses.items():
            if analysis['status'] == 'success':
                # Aggregate engagement
                if 'metrics' in analysis['data']:
                    metrics = analysis['data']['metrics']
                    if 'engagement' in metrics:
                        total_engagement['tweets'] += metrics['engagement'].get('total_tweets', 0)
                        total_engagement['retweets'] += metrics['engagement'].get('total_retweets', 0)
                        total_engagement['likes'] += metrics['engagement'].get('total_likes', 0)
                
                # Aggregate sentiment
                if 'metrics' in analysis['data']:
                    metrics = analysis['data']['metrics']
                    if 'sentiment' in metrics:
                        total_sentiment['positive'] += metrics['sentiment'].get('positive_percentage', 0)
                        total_sentiment['negative'] += metrics['sentiment'].get('negative_percentage', 0)
                        total_sentiment['neutral'] += metrics['sentiment'].get('neutral_percentage', 0)
                
                # Collect keywords
                if 'keywords' in analysis['data']:
                    all_keywords.extend(analysis['data']['keywords'])
        
        # Calculate average sentiment percentages
        platform_count = sum(1 for p in platform_analyses.values() if p['status'] == 'success')
        if platform_count > 0:
            for sentiment in total_sentiment:
                total_sentiment[sentiment] /= platform_count
        
        # Get most common keywords
        keyword_counter = Counter(k['keyword'] for k in all_keywords)
        top_keywords = [
            {'keyword': k, 'count': v}
            for k, v in keyword_counter.most_common(20)
        ]
        
        return {
            'total_engagement': total_engagement,
            'average_sentiment': total_sentiment,
            'top_keywords': top_keywords
        }
        
    def generate_visualizations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visualizations for analysis results.
        
        Args:
            analysis_results (Dict[str, Any]): Analysis results
            
        Returns:
            Dict[str, Any]: Generated visualizations
        """
        visualizations = {}
        
        # Generate engagement trend visualization
        if 'cross_platform_insights' in analysis_results:
            engagement_data = analysis_results['cross_platform_insights']['total_engagement']
            fig = go.Figure(data=[
                go.Bar(
                    x=list(engagement_data.keys()),
                    y=list(engagement_data.values()),
                    name='Engagement'
                )
            ])
            fig.update_layout(
                title='Total Engagement Across Platforms',
                xaxis_title='Metric',
                yaxis_title='Count'
            )
            visualizations['engagement_trends'] = fig.to_json()
        
        # Generate sentiment distribution visualization
        if 'cross_platform_insights' in analysis_results:
            sentiment_data = analysis_results['cross_platform_insights']['average_sentiment']
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(sentiment_data.keys()),
                    values=list(sentiment_data.values()),
                    name='Sentiment Distribution'
                )
            ])
            fig.update_layout(title='Average Sentiment Distribution')
            visualizations['sentiment_distribution'] = fig.to_json()
        
        # Generate keyword cloud visualization
        if 'cross_platform_insights' in analysis_results:
            keyword_data = analysis_results['cross_platform_insights']['top_keywords']
            fig = go.Figure(data=[
                go.Scatter(
                    x=[k['count'] for k in keyword_data],
                    y=[k['keyword'] for k in keyword_data],
                    mode='markers+text',
                    text=[k['keyword'] for k in keyword_data],
                    name='Keywords'
                )
            ])
            fig.update_layout(
                title='Top Keywords',
                xaxis_title='Frequency',
                yaxis_title='Keyword'
            )
            visualizations['keyword_cloud'] = fig.to_json()
        
        return visualizations
        
    def save_analysis_results(self, results: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """
        Save analysis results to disk.
        
        Args:
            results (Dict[str, Any]): Analysis results
            output_path (Optional[Path]): Path to save results to
            
        Returns:
            Path: Path to saved file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path("data/analysis") / f"{results['song_id']}_analysis_{timestamp}.json"
            
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        self.logger.info(f"Analysis results saved to {output_path}")
        return output_path 