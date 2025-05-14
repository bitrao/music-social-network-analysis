"""
Social Media Data Collectors for Music Analysis

This module provides a collection of async functions to gather data about songs
from various social media platforms and utilities for social network analysis.
"""

import os
import logging
from datetime import datetime, timedelta
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
    days: int = 30
) -> Dict[str, Any]:
    """
    Collect X (formerly Twitter) data related to a specific song.
    
    Args:
        song_id (str): Identifier for the song (can be name, hashtag, etc.)
        api_key (str): X API key for authentication
        api_secret (str): X API secret for authentication
        days (int, optional): Number of days of data to collect. Defaults to 30.
    
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
        # Initialize X API client
        auth = tweepy.OAuthHandler(api_key, api_secret)
        api = tweepy.API(auth)
        
        posts = []
        # Search for posts mentioning the song
        for tweet in tweepy.Cursor(api.search_tweets, q=song_id).items(100):
            posts.append({
                'id': tweet.id_str,
                'text': tweet.text,
                'user': tweet.user.screen_name,
                'created_at': tweet.created_at.isoformat(),
                'likes': tweet.favorite_count,
                'reposts': tweet.retweet_count
            })
        
        return {'platform': 'x', 'data': posts}
    except Exception as e:
        logger.error(f"X collection error: {e}")
        return {'platform': 'x', 'error': str(e)}

async def collect_youtube_data(
    song_id: str,
    api_key: str,
    days: int = 30
) -> Dict[str, Any]:
    """
    Collect YouTube data related to a specific song.
    
    Args:
        song_id (str): Identifier for the song (title, artist, etc.)
        api_key (str): YouTube Data API key
        days (int, optional): Number of days of data to collect. Defaults to 30.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - platform: "youtube"
            - data: List of video data including view counts and engagement
            - error: Error message if collection failed
    
    Note:
        - Uses YouTube Data API v3
        - Collects both video metadata and statistics
        - Limited to 50 results per search for efficiency
    """
    try:
        # Initialize YouTube API client
        youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)
        
        # Search for videos related to the song
        request = youtube.search().list(
            part="snippet",
            q=song_id,
            type="video",
            maxResults=50
        )
        response = request.execute()
        
        videos = []
        # Collect detailed statistics for each video
        for item in response.get('items', []):
            video_id = item['id']['videoId']
            stats = youtube.videos().list(
                part="statistics",
                id=video_id
            ).execute()['items'][0]['statistics']
            
            videos.append({
                'id': video_id,
                'title': item['snippet']['title'],
                'channel': item['snippet']['channelTitle'],
                'published_at': item['snippet']['publishedAt'],
                'views': int(stats.get('viewCount', 0)),
                'likes': int(stats.get('likeCount', 0)),
                'comments': int(stats.get('commentCount', 0))
            })
        
        return {'platform': 'youtube', 'data': videos}
    except Exception as e:
        logger.error(f"YouTube collection error: {e}")
        return {'platform': 'youtube', 'error': str(e)}

async def collect_reddit_data(
    song_id: str,
    client_id: str,
    client_secret: str,
    days: int = 30
) -> Dict[str, Any]:
    """
    Collect Reddit data related to a specific song.
    
    Args:
        song_id (str): Identifier for the song (title, artist, etc.)
        client_id (str): Reddit API client ID
        client_secret (str): Reddit API client secret
        days (int, optional): Number of days of data to collect. Defaults to 30.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - platform: "reddit"
            - data: List of Reddit posts and their metadata
            - error: Error message if collection failed
    
    Note:
        - Searches across all subreddits for wider coverage
        - Includes post scores and comment counts
        - Limited to 100 posts per search
    """
    try:
        # Initialize Reddit API client
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="music-analysis-bot/1.0"
        )
        
        posts = []
        # Search for posts about the song
        for submission in reddit.subreddit("all").search(song_id, limit=100):
            posts.append({
                'id': submission.id,
                'title': submission.title,
                'author': str(submission.author),
                'created_utc': datetime.fromtimestamp(submission.created_utc).isoformat(),
                'score': submission.score,
                'num_comments': submission.num_comments,
                'url': submission.url,
                'subreddit': str(submission.subreddit)
            })
        
        return {'platform': 'reddit', 'data': posts}
    except Exception as e:
        logger.error(f"Reddit collection error: {e}")
        return {'platform': 'reddit', 'error': str(e)}

async def collect_tiktok_data(
    song_id: str,
    ms_token: str,
    days: int = 30
) -> Dict[str, Any]:
    """
    Collect TikTok data related to a specific song.
    
    Args:
        song_id (str): Identifier for the song (sound ID, title, etc.)
        ms_token (str): TikTok MS token for authentication
        days (int, optional): Number of days of data to collect. Defaults to 30.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - platform: "tiktok"
            - data: List of TikTok videos and their engagement metrics
            - error: Error message if collection failed
    
    Note:
        - Collects video metadata and engagement metrics
        - Includes share counts for viral tracking
        - Limited to 100 videos per search
    """
    try:
        # Initialize TikTok API client
        api = TikTokApi()
        
        videos = []
        # Search for videos using the song
        async for video in api.search.videos(song_id, count=100):
            videos.append({
                'id': video.id,
                'desc': video.desc,
                'author': video.author.username,
                'created_at': datetime.fromtimestamp(video.create_time).isoformat(),
                'likes': video.stats.digg_count,
                'shares': video.stats.share_count,
                'comments': video.stats.comment_count,
                'views': video.stats.play_count,
                'duration': video.video.duration
            })
        
        return {'platform': 'tiktok', 'data': videos}
    except Exception as e:
        logger.error(f"TikTok collection error: {e}")
        return {'platform': 'tiktok', 'error': str(e)}

def save_results(song_id: str, results: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Save collected data to CSV files for each platform and create network analysis files.
    
    Args:
        song_id (str): Identifier for the song
        results (List[Dict[str, Any]]): List of results from each platform
    
    Returns:
        Dict[str, str]: Dictionary of saved file paths for each data type
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}
    
    # Create directories if they don't exist
    for dir_name in ['raw', 'processed', 'networks', 'visualizations']:
        Path(f"data/{dir_name}").mkdir(parents=True, exist_ok=True)
    
    # Process each platform's data
    for result in results:
        if 'error' in result or 'data' not in result:
            continue
            
        platform = result['platform']
        data = result['data']
        
        if not data:  # Skip if no data
            continue
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save raw data
        csv_file = f"data/processed/{song_id}_{platform}_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        saved_files[f"{platform}_data"] = csv_file
        
        # Create and save network data based on platform
        if platform == 'x':
            # Create user interaction network
            network_file = create_x_network(df, song_id, timestamp)
            saved_files[f"{platform}_network"] = network_file
            
        elif platform == 'youtube':
            # Create channel-video network
            network_file = create_youtube_network(df, song_id, timestamp)
            saved_files[f"{platform}_network"] = network_file
            
        elif platform == 'reddit':
            # Create subreddit-user network
            network_file = create_reddit_network(df, song_id, timestamp)
            saved_files[f"{platform}_network"] = network_file
    
    return saved_files

def create_x_network(df: pd.DataFrame, song_id: str, timestamp: str) -> str:
    """Create network from X data focusing on user interactions."""
    G = nx.Graph()
    
    # Add nodes for each unique user
    users = df['user'].unique()
    G.add_nodes_from(users)
    
    # Create edges based on mentions and reposts
    edges = []
    for _, post in df.iterrows():
        # Extract mentioned users from post text
        mentions = [user.strip('@') for user in post['text'].split() if user.startswith('@')]
        
        # Add edges between post author and mentioned users
        for mention in mentions:
            edges.append((post['user'], mention))
    
    G.add_edges_from(edges)
    
    # Save network data
    network_file = f"data/networks/{song_id}_x_network_{timestamp}.graphml"
    nx.write_graphml(G, network_file)
    
    # Create and save visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=8, font_weight='bold')
    plt.title(f"X Interaction Network for {song_id}")
    plt.savefig(f"data/visualizations/{song_id}_x_network_{timestamp}.png")
    plt.close()
    
    return network_file

def create_youtube_network(df: pd.DataFrame, song_id: str, timestamp: str) -> str:
    """Create network from YouTube data focusing on channel relationships."""
    G = nx.Graph()
    
    # Add nodes for channels and videos
    channels = df['channel'].unique()
    G.add_nodes_from(channels, node_type='channel')
    G.add_nodes_from(df['id'], node_type='video')
    
    # Create edges between channels and their videos
    edges = [(row['channel'], row['id']) for _, row in df.iterrows()]
    G.add_edges_from(edges)
    
    # Save network data
    network_file = f"data/networks/{song_id}_youtube_network_{timestamp}.graphml"
    nx.write_graphml(G, network_file)
    
    # Create and save visualization
    plt.figure(figsize=(12, 8))
    pos = nx.bipartite_layout(G, channels)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen',
            node_size=500, font_size=8, font_weight='bold')
    plt.title(f"YouTube Channel-Video Network for {song_id}")
    plt.savefig(f"data/visualizations/{song_id}_youtube_network_{timestamp}.png")
    plt.close()
    
    return network_file

def create_reddit_network(df: pd.DataFrame, song_id: str, timestamp: str) -> str:
    """Create network from Reddit data focusing on subreddit relationships."""
    G = nx.Graph()
    
    # Add nodes for subreddits and authors
    subreddits = df['subreddit'].unique()
    authors = df['author'].unique()
    G.add_nodes_from(subreddits, node_type='subreddit')
    G.add_nodes_from(authors, node_type='author')
    
    # Create edges between subreddits and authors
    edges = [(row['subreddit'], row['author']) for _, row in df.iterrows()]
    G.add_edges_from(edges)
    
    # Save network data
    network_file = f"data/networks/{song_id}_reddit_network_{timestamp}.graphml"
    nx.write_graphml(G, network_file)
    
    # Create and save visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='salmon',
            node_size=500, font_size=8, font_weight='bold')
    plt.title(f"Reddit Subreddit-Author Network for {song_id}")
    plt.savefig(f"data/visualizations/{song_id}_reddit_network_{timestamp}.png")
    plt.close()
    
    return network_file

def analyze_network(network_file: str) -> Dict[str, Any]:
    """
    Perform social network analysis on a given network.
    
    Args:
        network_file (str): Path to the GraphML network file
        
    Returns:
        Dict[str, Any]: Dictionary containing network metrics
    """
    G = nx.read_graphml(network_file)
    
    metrics = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G),
        'degree_centrality': dict(nx.degree_centrality(G)),
        'betweenness_centrality': dict(nx.betweenness_centrality(G)),
        'connected_components': list(nx.connected_components(G))
    }
    
    return metrics 