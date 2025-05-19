"""
Social Media Data Collectors for Music Analysis

This module provides a collection of async functions to gather data about songs
from various social media platforms and utilities for social network analysis.
"""

import os
import logging
from datetime import datetime, timedelta, timezone
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import tweepy
import praw
import googleapiclient.discovery
from TikTokApi import TikTokApi
from dateutil.parser import parse
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging with a standardized format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create necessary directories for data storage
Path("data/raw").mkdir(parents=True, exist_ok=True)  # Raw JSON data from APIs
Path("data/processed").mkdir(parents=True, exist_ok=True)  # For future processed data

async def collect_x_data(
    song_id: str,
    api_key: str,
    api_secret: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    days: int = 30
) -> Dict[str, Any]:
    """
    Collect X (formerly Twitter) data related to a specific song.
    
    Args:
        song_id (str): Identifier for the song (can be name, hashtag, etc.)
        api_key (str): X API key for authentication
        api_secret (str): X API secret for authentication
        start_date (datetime, optional): Start date for data collection
        end_date (datetime, optional): End date for data collection
        days (int, optional): Number of days of data to collect if dates not specified. Defaults to 30.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - platform: "x"
            - data: List of post data including engagement metrics
            - error: Error message if collection failed
    
    Note:
        - Uses X API v1.1 for broader search capabilities
        - Limited to 100 posts per search to avoid rate limits
        - Includes reposts to capture viral spread
    """
    try:
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=days)
            
        # Initialize X API client
        auth = tweepy.OAuthHandler(api_key, api_secret)
        api = tweepy.API(auth)
        
        posts = []
        # Search for posts mentioning the song
        for tweet in tweepy.Cursor(api.search_tweets, 
                                 q=song_id,
                                 since=start_date.strftime('%Y-%m-%d'),
                                 until=end_date.strftime('%Y-%m-%d')).items(100):
            # Skip if tweet is outside date range
            if not (start_date <= tweet.created_at <= end_date):
                continue
                
            posts.append({
                'id': tweet.id_str,
                'text': tweet.text,
                'user': tweet.user.screen_name,
                'created_at': tweet.created_at.isoformat(),
                'likes': tweet.favorite_count,
                'reposts': tweet.retweet_count
            })
        
        return {
            'platform': 'x',
            'data': posts,
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"X collection error: {e}")
        return {'platform': 'x', 'error': str(e)}

async def collect_youtube_data(
    song_id: str,
    api_key: str,
) -> Dict[str, Any]:
    """
    Collect YouTube data for a single video related to a specific song.
    
    Args:
        song_id (str): Identifier for the song (title, artist, etc.)
    Returns:
        Dict[str, Any]: Dictionary containing:
            - platform: "youtube"
            - data: Video data including view counts and engagement
            - comments: List of comments and their replies for the video
            - error: Error message if collection failed
    """
    try:
        # Initialize YouTube API client
        youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)
        
        # Search for a single video related to the song
        request = youtube.search().list(
            part="snippet",
            q=song_id,
            type="video",
            maxResults=1
        )
        response = request.execute()
        
        if not response.get('items'):
            return {'platform': 'youtube', 'error': 'No videos found'}
            
        item = response['items'][0]
        video_id = item['id']['videoId']
        
        # Get video statistics
        stats = youtube.videos().list(
            part="statistics",
            id=video_id
        ).execute()['items'][0]['statistics']
        
        video_data = {
            'id': video_id,
            'title': item['snippet']['title'],
            'channel': item['snippet']['channelTitle'],
            'published_at': item['snippet']['publishedAt'],
            'views': int(stats.get('viewCount', 0)),
            'likes': int(stats.get('likeCount', 0)),
            'comments': int(stats.get('commentCount', 0))
        }
        
        # Collect comments and replies for the video
        all_comments = []
        next_page_token = None
        
        while True:
            try:
                # Get top-level comments
                comments_request = youtube.commentThreads().list(
                    part="snippet,replies",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=next_page_token
                )
                comments_response = comments_request.execute()
                
                for comment_item in comments_response.get('items', []):
                    top_comment = comment_item['snippet']['topLevelComment']['snippet']
                    comment_date = parse(top_comment['publishedAt'])
                    
                    # Ensure comment date is timezone-aware
                    if comment_date.tzinfo is None:
                        comment_date = comment_date.replace(tzinfo=timezone.utc)
                    
                    # Process top-level comment
                    comment_data = {
                        'id': comment_item['id'],
                        'video_id': video_id,
                        'parent_id': None,  # Top-level comment has no parent
                        'author': top_comment['authorDisplayName'],
                        'text': top_comment['textDisplay'],
                        'published_at': top_comment['publishedAt'],
                        'likes': top_comment['likeCount'],
                        'is_reply': False
                    }
                    all_comments.append(comment_data)
                    
                    # Process replies if they exist
                    if 'replies' in comment_item:
                        replies = comment_item['replies']['comments']
                        for reply in replies:
                            reply_snippet = reply['snippet']
                            reply_date = parse(reply_snippet['publishedAt'])
                            
                            # Ensure reply date is timezone-aware
                            if reply_date.tzinfo is None:
                                reply_date = reply_date.replace(tzinfo=timezone.utc)
                            
                            reply_data = {
                                'id': reply['id'],
                                'video_id': video_id,
                                'parent_id': comment_item['id'],  # Link to parent comment
                                'author': reply_snippet['authorDisplayName'],
                                'text': reply_snippet['textDisplay'],
                                'published_at': reply_snippet['publishedAt'],
                                'likes': reply_snippet['likeCount'],
                                'is_reply': True
                            }
                            all_comments.append(reply_data)
                
                # Check if there are more pages of comments
                next_page_token = comments_response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            except Exception as e:
                logger.warning(f"Error fetching comments for video {video_id}: {e}")
                break
        
        return {
            'platform': 'youtube',
            'data': video_data,
            'comments': all_comments,
        }
    except Exception as e:
        logger.error(f"YouTube collection error: {e}")
        return {'platform': 'youtube', 'error': str(e)}

async def collect_reddit_data(
    song_id: str,
    client_id: str,
    client_secret: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    days: int = 30
) -> Dict[str, Any]:
    """
    Collect Reddit data related to a specific song.
    
    Args:
        song_id (str): Identifier for the song (title, artist, etc.)
        client_id (str): Reddit API client ID
        client_secret (str): Reddit API client secret
        start_date (datetime, optional): Start date for data collection
        end_date (datetime, optional): End date for data collection
        days (int, optional): Number of days of data to collect if dates not specified. Defaults to 30.
    """
    try:
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=days)
            
        # Initialize Reddit API client
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="music-analysis-bot/1.0"
        )
        
        posts = []
        # Search for posts about the song
        for submission in reddit.subreddit("all").search(song_id, limit=100):
            post_date = datetime.fromtimestamp(submission.created_utc)
            
            # Skip if post is outside date range
            if not (start_date <= post_date <= end_date):
                continue
                
            posts.append({
                'id': submission.id,
                'title': submission.title,
                'author': str(submission.author),
                'created_utc': post_date.isoformat(),
                'score': submission.score,
                'num_comments': submission.num_comments,
                'url': submission.url,
                'subreddit': str(submission.subreddit)
            })
        
        return {
            'platform': 'reddit',
            'data': posts,
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Reddit collection error: {e}")
        return {'platform': 'reddit', 'error': str(e)}

async def collect_tiktok_data(
    song_id: str,
    ms_token: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    days: int = 30
) -> Dict[str, Any]:
    """
    Collect TikTok data related to a specific song.
    
    Args:
        song_id (str): Identifier for the song (sound ID, title, etc.)
        ms_token (str): TikTok MS token for authentication
        start_date (datetime, optional): Start date for data collection
        end_date (datetime, optional): End date for data collection
        days (int, optional): Number of days of data to collect if dates not specified. Defaults to 30.
    """
    try:
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=days)
            
        # Initialize TikTok API client
        api = TikTokApi()
        
        videos = []
        # Search for videos using the song
        async for video in api.search.videos(song_id, count=100):
            video_date = datetime.fromtimestamp(video.create_time)
            
            # Skip if video is outside date range
            if not (start_date <= video_date <= end_date):
                continue
                
            videos.append({
                'id': video.id,
                'desc': video.desc,
                'author': video.author.username,
                'created_at': video_date.isoformat(),
                'likes': video.stats.digg_count,
                'shares': video.stats.share_count,
                'comments': video.stats.comment_count,
                'views': video.stats.play_count,
                'duration': video.video.duration
            })
        
        return {
            'platform': 'tiktok',
            'data': videos,
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }
    except Exception as e:
        logger.error(f"TikTok collection error: {e}")
        return {'platform': 'tiktok', 'error': str(e)}

def save_results(song_id: str, results: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Save collected data for a single song across all platforms.
    
    Args:
        song_id (str): Identifier for the song
        results (List[Dict[str, Any]]): List of results from each platform
    
    Returns:
        Dict[str, str]: Dictionary of saved file paths for each data type
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}
    
    # Create song-specific directories
    song_dir = f"data/songs/{song_id}"
    for dir_name in ['raw', 'processed', 'networks', 'visualizations']:
        Path(f"{song_dir}/{dir_name}").mkdir(parents=True, exist_ok=True)
    
    # Process each platform's data
    for result in results:
        if 'error' in result or 'data' not in result:
            continue
            
        platform = result['platform']
        data = result['data']
        
        if not data:  # Skip if no data
            continue
        
        # Convert main data to DataFrame
        df = pd.DataFrame(data)
        
        # Save raw data
        raw_file = f"{song_dir}/raw/{platform}_data_{timestamp}.csv"
        df.to_csv(raw_file, index=False)
        saved_files[f"{platform}_data"] = raw_file
        
        # Save comments if available
        if 'comments' in result and result['comments']:
            comments_df = pd.DataFrame(result['comments'])
            comments_file = f"{song_dir}/raw/{platform}_comments_{timestamp}.csv"
            comments_df.to_csv(comments_file, index=False)
            saved_files[f"{platform}_comments"] = comments_file
            
            # Merge comments with main data for analysis
            if 'video_id' in comments_df.columns:
                merged_df = pd.merge(
                    df,
                    comments_df.groupby('video_id').agg({
                        'comment_id': 'count',
                        'likes': 'sum',
                        'replies': 'sum'
                    }).reset_index(),
                    left_on='id',
                    right_on='video_id',
                    how='left'
                )
                merged_df = merged_df.drop('video_id', axis=1)
            else:
                merged_df = df
        else:
            merged_df = df
        
        # Save processed data
        processed_file = f"{song_dir}/processed/{platform}_processed_{timestamp}.csv"
        merged_df.to_csv(processed_file, index=False)
        saved_files[f"{platform}_processed"] = processed_file
        
        # Create and save summary statistics
        summary = {
            'platform': platform,
            'total_posts': len(df),
            'total_comments': len(result.get('comments', [])),
            'total_engagement': merged_df['likes'].sum() if 'likes' in merged_df.columns else 0,
            'timestamp': timestamp
        }
        
        summary_file = f"{song_dir}/processed/{platform}_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files[f"{platform}_summary"] = summary_file
    
    # Create combined summary across all platforms
    combined_summary = {
        'song_id': song_id,
        'timestamp': timestamp,
        'platforms': [result['platform'] for result in results if 'error' not in result],
        'total_posts': sum(len(result['data']) for result in results if 'error' not in result),
        'total_comments': sum(len(result.get('comments', [])) for result in results if 'error' not in result)
    }
    
    combined_summary_file = f"{song_dir}/processed/combined_summary_{timestamp}.json"
    with open(combined_summary_file, 'w') as f:
        json.dump(combined_summary, f, indent=2)
    saved_files['combined_summary'] = combined_summary_file
    
    return saved_files
        