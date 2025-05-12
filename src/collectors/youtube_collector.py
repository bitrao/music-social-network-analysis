from datetime import datetime
from typing import Dict, Any, List
import asyncio
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from .base_collector import BaseCollector

class YouTubeCollector(BaseCollector):
    """Collector for YouTube data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize YouTube collector.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing YouTube API credentials
        """
        super().__init__(config)
        self.setup_api()
        
    def setup_api(self):
        """Set up YouTube API client."""
        self.youtube = build(
            'youtube',
            'v3',
            developerKey=self.config['youtube']['api_key']
        )
        self.rate_limit = self.config['youtube']['rate_limit']
        
    async def collect(self, song_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Collect YouTube data for a specific song.
        
        Args:
            song_id (str): Song identifier
            start_date (datetime): Start date for data collection
            end_date (datetime): End date for data collection
            
        Returns:
            Dict[str, Any]: Collected YouTube data
        """
        self.logger.info(f"Collecting YouTube data for song {song_id}")
        
        # Convert async function to sync for YouTube API
        loop = asyncio.get_event_loop()
        
        # Search for videos
        videos = await loop.run_in_executor(
            None,
            self._search_videos,
            song_id,
            start_date,
            end_date
        )
        
        # Collect video details and comments
        video_details = []
        for video in videos:
            details = await loop.run_in_executor(
                None,
                self._get_video_details,
                video['id']
            )
            comments = await loop.run_in_executor(
                None,
                self._get_video_comments,
                video['id']
            )
            details['comments'] = comments
            video_details.append(details)
        
        # Calculate engagement metrics
        engagement = self._calculate_engagement(video_details)
        
        data = {
            'song_id': song_id,
            'platform': 'youtube',
            'collection_date': datetime.now().isoformat(),
            'time_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'videos': video_details,
            'engagement': engagement
        }
        
        if await self.validate_data(data):
            return data
        else:
            raise ValueError("Collected data validation failed")
    
    def _search_videos(self, song_id: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Search for videos related to the song."""
        videos = []
        
        try:
            # Search for videos
            search_response = self.youtube.search().list(
                q=song_id,
                part='id,snippet',
                maxResults=50,
                type='video',
                publishedAfter=start_date.isoformat() + 'Z',
                publishedBefore=end_date.isoformat() + 'Z'
            ).execute()
            
            for item in search_response.get('items', []):
                videos.append({
                    'id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'published_at': item['snippet']['publishedAt'],
                    'channel_id': item['snippet']['channelId'],
                    'channel_title': item['snippet']['channelTitle']
                })
                
        except HttpError as e:
            self.logger.error(f"YouTube API error: {str(e)}")
            raise
            
        return videos
    
    def _get_video_details(self, video_id: str) -> Dict[str, Any]:
        """Get detailed information about a video."""
        try:
            response = self.youtube.videos().list(
                part='statistics,contentDetails',
                id=video_id
            ).execute()
            
            if not response['items']:
                return {}
                
            video = response['items'][0]
            return {
                'id': video_id,
                'statistics': {
                    'view_count': int(video['statistics'].get('viewCount', 0)),
                    'like_count': int(video['statistics'].get('likeCount', 0)),
                    'comment_count': int(video['statistics'].get('commentCount', 0)),
                    'favorite_count': int(video['statistics'].get('favoriteCount', 0))
                },
                'duration': video['contentDetails']['duration']
            }
            
        except HttpError as e:
            self.logger.error(f"Error getting video details: {str(e)}")
            return {}
    
    def _get_video_comments(self, video_id: str) -> List[Dict[str, Any]]:
        """Get comments for a video."""
        comments = []
        
        try:
            response = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                textFormat='plainText'
            ).execute()
            
            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'id': item['id'],
                    'text': comment['textDisplay'],
                    'author': comment['authorDisplayName'],
                    'author_channel_id': comment['authorChannelId']['value'],
                    'like_count': comment['likeCount'],
                    'published_at': comment['publishedAt']
                })
                
        except HttpError as e:
            self.logger.error(f"Error getting video comments: {str(e)}")
            
        return comments
    
    def _calculate_engagement(self, videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate engagement metrics from collected videos."""
        total_views = sum(v['statistics']['view_count'] for v in videos)
        total_likes = sum(v['statistics']['like_count'] for v in videos)
        total_comments = sum(v['statistics']['comment_count'] for v in videos)
        
        return {
            'total_videos': len(videos),
            'total_views': total_views,
            'total_likes': total_likes,
            'total_comments': total_comments,
            'average_engagement': (total_likes + total_comments) / len(videos) if videos else 0
        }
    
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate collected YouTube data.
        
        Args:
            data (Dict[str, Any]): Data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        required_fields = ['song_id', 'platform', 'collection_date', 'time_range', 'videos', 'engagement']
        
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
            
        # Validate videos format
        if not isinstance(data['videos'], list):
            self.logger.error("Videos must be a list")
            return False
            
        # Validate engagement metrics
        if not isinstance(data['engagement'], dict):
            self.logger.error("Engagement metrics must be a dictionary")
            return False
            
        return True 