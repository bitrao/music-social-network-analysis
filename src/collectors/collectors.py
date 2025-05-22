"""
Youtube Data Collectors

"""
from datetime import timezone
from pathlib import Path
from typing import Dict, Any

import googleapiclient.discovery
from dateutil.parser import parse


# Create necessary directories for data storage
Path("data/raw").mkdir(parents=True, exist_ok=True) 
Path("data/processed").mkdir(parents=True, exist_ok=True)

async def collect_youtube_data(
    query: str,
    api_key: str,
) -> Dict[str, Any]:
    """
    Collect YouTube data for a single video related to a specific song.
    
    Args:
        query (str)
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
            q=query,
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
                    part="snippet",
                    videoId=video_id,
                    maxResults=50,
                    pageToken=next_page_token,
                    order="relevance"
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
                    
                    # Get all replies for this comment thread
                    replies_next_page_token = None
                    while True:
                        try:
                            replies_request = youtube.comments().list(
                                part="snippet",
                                parentId=comment_item['id'],
                                maxResults=100,
                                pageToken=replies_next_page_token
                            )
                            replies_response = replies_request.execute()
                            
                            for reply in replies_response.get('items', []):
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
                            
                            # Check if there are more pages of replies
                            replies_next_page_token = replies_response.get('nextPageToken')
                            if not replies_next_page_token:
                                break
                                
                        except Exception as e:
                            print(f"Error fetching replies for comment {comment_item['id']}: {e}")
                            break
                
                # Check if there are more pages of comments
                next_page_token = comments_response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            except Exception as e:
                print(f"Error fetching comments for video {video_id}: {e}")
                break
        
        return {
            'platform': 'youtube',
            'data': video_data,
            'comments': all_comments,
        }
    except Exception as e:
        print(f"YouTube collection error: {e}")
        return {'platform': 'youtube', 'error': str(e)}


        