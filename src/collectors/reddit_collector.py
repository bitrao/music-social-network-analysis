from datetime import datetime
from typing import Dict, Any, List
import asyncio
import praw
from praw.models import Submission, Comment
from .base_collector import BaseCollector

class RedditCollector(BaseCollector):
    """Collector for Reddit data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Reddit collector.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing Reddit API credentials
        """
        super().__init__(config)
        self.setup_api()
        
    def setup_api(self):
        """Set up Reddit API client."""
        self.reddit = praw.Reddit(
            client_id=self.config['reddit']['client_id'],
            client_secret=self.config['reddit']['client_secret'],
            user_agent=self.config['reddit']['user_agent']
        )
        self.rate_limit = self.config['reddit']['rate_limit']
        
    async def collect(self, song_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Collect Reddit data for a specific song.
        
        Args:
            song_id (str): Song identifier
            start_date (datetime): Start date for data collection
            end_date (datetime): End date for data collection
            
        Returns:
            Dict[str, Any]: Collected Reddit data
        """
        self.logger.info(f"Collecting Reddit data for song {song_id}")
        
        # Convert async function to sync for PRAW
        loop = asyncio.get_event_loop()
        
        # Search for posts
        posts = await loop.run_in_executor(
            None,
            self._search_posts,
            song_id,
            start_date,
            end_date
        )
        
        # Collect post details and comments
        post_details = []
        for post in posts:
            details = await loop.run_in_executor(
                None,
                self._get_post_details,
                post
            )
            comments = await loop.run_in_executor(
                None,
                self._get_post_comments,
                post
            )
            details['comments'] = comments
            post_details.append(details)
        
        # Calculate engagement metrics
        engagement = self._calculate_engagement(post_details)
        
        data = {
            'song_id': song_id,
            'platform': 'reddit',
            'collection_date': datetime.now().isoformat(),
            'time_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'posts': post_details,
            'engagement': engagement
        }
        
        if await self.validate_data(data):
            return data
        else:
            raise ValueError("Collected data validation failed")
    
    def _search_posts(self, song_id: str, start_date: datetime, end_date: datetime) -> List[Submission]:
        """Search for posts related to the song."""
        posts = []
        
        try:
            # Search across multiple subreddits
            subreddits = ['music', 'listentothis', 'hiphopheads', 'popheads', 'indieheads']
            for subreddit_name in subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                search_results = subreddit.search(
                    song_id,
                    sort='relevance',
                    time_filter='all',
                    limit=100
                )
                
                for post in search_results:
                    post_date = datetime.fromtimestamp(post.created_utc)
                    if start_date <= post_date <= end_date:
                        posts.append(post)
                        
        except Exception as e:
            self.logger.error(f"Error searching Reddit posts: {str(e)}")
            
        return posts
    
    def _get_post_details(self, post: Submission) -> Dict[str, Any]:
        """Get detailed information about a post."""
        try:
            return {
                'id': post.id,
                'title': post.title,
                'text': post.selftext,
                'url': post.url,
                'author': post.author.name if post.author else '[deleted]',
                'subreddit': post.subreddit.display_name,
                'created_utc': post.created_utc,
                'score': post.score,
                'upvote_ratio': post.upvote_ratio,
                'num_comments': post.num_comments,
                'is_original_content': post.is_original_content,
                'is_self': post.is_self
            }
        except Exception as e:
            self.logger.error(f"Error getting post details: {str(e)}")
            return {}
    
    def _get_post_comments(self, post: Submission) -> List[Dict[str, Any]]:
        """Get comments for a post."""
        comments = []
        
        try:
            post.comments.replace_more(limit=0)  # Get top-level comments only
            for comment in post.comments.list():
                comments.append({
                    'id': comment.id,
                    'text': comment.body,
                    'author': comment.author.name if comment.author else '[deleted]',
                    'score': comment.score,
                    'created_utc': comment.created_utc,
                    'is_submitter': comment.is_submitter
                })
                
        except Exception as e:
            self.logger.error(f"Error getting post comments: {str(e)}")
            
        return comments
    
    def _calculate_engagement(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate engagement metrics from collected posts."""
        total_score = sum(p['score'] for p in posts)
        total_comments = sum(p['num_comments'] for p in posts)
        total_upvotes = sum(int(p['score'] * p['upvote_ratio']) for p in posts)
        
        return {
            'total_posts': len(posts),
            'total_score': total_score,
            'total_comments': total_comments,
            'total_upvotes': total_upvotes,
            'average_engagement': (total_score + total_comments) / len(posts) if posts else 0
        }
    
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate collected Reddit data.
        
        Args:
            data (Dict[str, Any]): Data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        required_fields = ['song_id', 'platform', 'collection_date', 'time_range', 'posts', 'engagement']
        
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
            
        # Validate posts format
        if not isinstance(data['posts'], list):
            self.logger.error("Posts must be a list")
            return False
            
        # Validate engagement metrics
        if not isinstance(data['engagement'], dict):
            self.logger.error("Engagement metrics must be a dictionary")
            return False
            
        return True 