import tweepy
from datetime import datetime
from typing import Dict, Any, List
import asyncio
from .base_collector import BaseCollector

class TwitterCollector(BaseCollector):
    """Collector for Twitter data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Twitter collector.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing Twitter API credentials
        """
        super().__init__(config)
        self.setup_api()
        
    def setup_api(self):
        """Set up Twitter API client."""
        auth = tweepy.OAuthHandler(
            self.config['twitter']['api_key'],
            self.config['twitter']['api_secret']
        )
        auth.set_access_token(
            self.config['twitter']['access_token'],
            self.config['twitter']['access_token_secret']
        )
        self.api = tweepy.API(auth, wait_on_rate_limit=True)
        self.rate_limit = self.config['twitter']['rate_limit']
        
    async def collect(self, song_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Collect Twitter data for a specific song.
        
        Args:
            song_id (str): Song identifier
            start_date (datetime): Start date for data collection
            end_date (datetime): End date for data collection
            
        Returns:
            Dict[str, Any]: Collected Twitter data
        """
        self.logger.info(f"Collecting Twitter data for song {song_id}")
        
        # Convert async function to sync for tweepy
        loop = asyncio.get_event_loop()
        
        # Collect tweets
        tweets = await loop.run_in_executor(
            None,
            self._collect_tweets,
            song_id,
            start_date,
            end_date
        )
        
        # Collect engagement metrics
        engagement = await loop.run_in_executor(
            None,
            self._collect_engagement,
            tweets
        )
        
        data = {
            'song_id': song_id,
            'platform': 'twitter',
            'collection_date': datetime.now().isoformat(),
            'time_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'tweets': tweets,
            'engagement': engagement
        }
        
        if await self.validate_data(data):
            return data
        else:
            raise ValueError("Collected data validation failed")
    
    def _collect_tweets(self, song_id: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Collect tweets related to the song."""
        query = f"#{song_id} OR @{song_id}"
        tweets = []
        
        for tweet in tweepy.Cursor(
            self.api.search_tweets,
            q=query,
            lang="en",
            since=start_date.strftime("%Y-%m-%d"),
            until=end_date.strftime("%Y-%m-%d"),
            tweet_mode="extended"
        ).items():
            tweets.append({
                'id': tweet.id_str,
                'text': tweet.full_text,
                'created_at': tweet.created_at.isoformat(),
                'user': {
                    'id': tweet.user.id_str,
                    'name': tweet.user.name,
                    'screen_name': tweet.user.screen_name,
                    'followers_count': tweet.user.followers_count
                },
                'metrics': {
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count
                }
            })
            
        return tweets
    
    def _collect_engagement(self, tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate engagement metrics from collected tweets."""
        total_tweets = len(tweets)
        total_retweets = sum(t['metrics']['retweet_count'] for t in tweets)
        total_likes = sum(t['metrics']['favorite_count'] for t in tweets)
        
        return {
            'total_tweets': total_tweets,
            'total_retweets': total_retweets,
            'total_likes': total_likes,
            'average_engagement': (total_retweets + total_likes) / total_tweets if total_tweets > 0 else 0
        }
    
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate collected Twitter data.
        
        Args:
            data (Dict[str, Any]): Data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        required_fields = ['song_id', 'platform', 'collection_date', 'time_range', 'tweets', 'engagement']
        
        # Check if all required fields are present
        if not all(field in data for field in required_fields):
            self.logger.error("Missing required fields in collected data")
            return False
            
        # Validate time range
        if not isinstance(data['time_range'], dict) or \
           'start' not in data['time_range'] or \
           'end' not in data['time_range']:
            self.logger.error("Invalid time range format")
            return False
            
        # Validate tweets format
        if not isinstance(data['tweets'], list):
            self.logger.error("Tweets must be a list")
            return False
            
        # Validate engagement metrics
        if not isinstance(data['engagement'], dict):
            self.logger.error("Engagement metrics must be a dictionary")
            return False
            
        return True 