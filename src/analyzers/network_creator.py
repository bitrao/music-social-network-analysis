import networkx as nx
from typing import Dict, Any
from pathlib import Path
import numpy as np
from collections import defaultdict
import pandas as pd
from .network_analyzer import NetworkAnalyzer
import nx_cugraph as nxcg

class NetworkCreator:
    """Creates various types of networks from YouTube comment data using GPU acceleration."""
    
    def __init__(self):
        self.comment_network = nxcg.DiGraph()
        self.video_comment_network = nxcg.DiGraph()
        self.semantic_network = nxcg.Graph()
        self.category_network = nxcg.DiGraph()
        self.topic_network = nxcg.DiGraph()
        self.analyzer = NetworkAnalyzer()
        
    def create_comment_reply_network(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Create a network of authors and their reply relationships.
        
        Args:
            data (pd.DataFrame): Processed comment data with columns:
                - id: comment ID
                - text: comment text
                - author: comment author
                - created_at: timestamp
                - likes: comment likes
                - parent_id: ID of parent comment (if reply)
            
        Returns:
            nx.DiGraph: Directed graph of authors and their reply relationships
        """
        self.comment_network.clear()
        
        # Create a mapping of comment IDs to authors
        comment_to_author = {}
        for _, comment in data.iterrows():
            comment_to_author[comment['id']] = comment['author']
            
        # Add all unique authors as nodes
        unique_authors = data['author'].unique()
        for author in unique_authors:
            self.comment_network.add_node(
                author,
                type='author'
            )
            
        # Add edges for replies between authors
        for _, comment in data.iterrows():
            if pd.notna(comment.get('parent_id')):
                parent_author = comment_to_author.get(comment['parent_id'])
                if parent_author is not None:
                    self.comment_network.add_edge(
                        parent_author,
                        comment['author'],
                        relationship='replied_to'
                    )
                    
        return self.comment_network
    
    def create_video_comment_network(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Create a network of videos and their comment authors.
        Authors can be connected to multiple videos if they commented on different videos.
        This creates a bipartite network where authors can bridge between different videos.
        
        Args:
            data (pd.DataFrame): Processed comment data with columns:
                - video_id: ID of the video
                - video_title: title of the video
                - video_metrics: metrics of the video
                - author: comment author
            
        Returns:
            nx.DiGraph: Directed graph of videos and comment authors, where:
                - Each video is connected to its commenters
                - Authors can be connected to multiple videos
                - Videos are not connected to each other
        """
        self.video_comment_network.clear()
        
        # Group data by video
        video_groups = data.groupby('video_id')
        
        # Process each video independently
        for video_id, video_data in video_groups:
            # Add video node
            self.video_comment_network.add_node(
                video_id,
                type='video',
                title=video_data['video_title'].iloc[0],
            )
            
            # Get unique authors who commented on this video
            unique_authors = video_data['author'].unique()
            
            # Add author nodes and connect to video
            # Note: If an author commented on multiple videos, they will have multiple edges
            for author in unique_authors:
                # Add author node if it doesn't exist yet
                if not self.video_comment_network.has_node(author):
                    self.video_comment_network.add_node(
                        author,
                        type='author'
                    )
                # Connect author to this video (edge points TO video)
                self.video_comment_network.add_edge(
                    author,
                    video_id,
                    relationship='commented_on'
                )
            
        return self.video_comment_network
    
    def create_category_network(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Create a hierarchical network of categories -> videos -> comment authors.
        Authors can be connected to multiple videos if they commented on different videos.
        
        Args:
            data (pd.DataFrame): Processed comment data with columns:
                - category: video category
                - video_id: ID of the video
                - video_title: title of the video
                - author: comment author
            
        Returns:
            nx.DiGraph: Directed graph where:
                - Categories are connected to their videos
                - Videos are connected to their commenters
                - Authors can be connected to multiple videos
                - Videos contain category information
        """
        self.category_network.clear()
        
        # Group data by category
        category_groups = data.groupby('category')
        
        # Process each category
        for category, category_data in category_groups:
            # Add category node
            self.category_network.add_node(
                category,
                type='category'
            )
            
            # Group videos within this category
            video_groups = category_data.groupby('video_id')
            
            # Process each video in the category
            for video_id, video_data in video_groups:
                # Add video node with category info
                self.category_network.add_node(
                    video_id,
                    type='video',
                    title=video_data['video_title'].iloc[0],
                    category=category
                )
                
                # Connect video to its category
                self.category_network.add_edge(
                    category,
                    video_id,
                    relationship='contains'
                )
                
                # Get unique authors who commented on this video
                unique_authors = video_data['author'].unique()
                
                # Add author nodes and connect to video
                # Note: If an author commented on multiple videos, they will have multiple edges
                for author in unique_authors:
                    # Add author node if it doesn't exist yet
                    if not self.category_network.has_node(author):
                        self.category_network.add_node(
                            author,
                            type='author'
                        )

                    self.category_network.add_edge(
                        author,
                        video_id,
                        relationship='commented_on'
                    )
            
        return self.category_network
    
    def create_topic_network(self, data: pd.DataFrame) -> nx.DiGraph:
        """
        Create a network connecting topics, comments, and authors.
        Comments are connected to their topics, and comments from the same author are connected.
        
        Args:
            data (pd.DataFrame): Processed comment data with columns:
                - id: comment ID
                - text: comment text
                - author: comment author
                - topic: topic of the comment
                - category: category of the video
            
        Returns:
            nx.DiGraph: Directed graph where:
                - Topics are connected to their comments
                - Comments contain author and category information
                - Comments from the same author are connected (undirected)
        """
        self.topic_network.clear()
        
        # Group comments by author to connect comments from same author
        author_comments = defaultdict(list)
        for _, comment in data.iterrows():
            author_comments[comment['author']].append(comment['id'])
        
        # Add all topics as nodes
        unique_topics = data['topic'].unique()
        for topic in unique_topics:
            self.topic_network.add_node(
                topic,
                type='topic'
            )
        
        # Add comments and connect to topics
        for _, comment in data.iterrows():
            # Add comment node with metadata
            self.topic_network.add_node(
                comment['id'],
                type='comment',
                author=comment['author'],
                category=comment['category'],
                text=comment['text']
            )
            
            # Connect comment to its topic
            self.topic_network.add_edge(
                comment['id'],
                comment['topic'],
                relationship='belongs_to'
            )
        
        # Connect comments from same author (undirected edges)
        for author, comment_ids in author_comments.items():
            # Create edges between all pairs of comments from same author
            for i in range(len(comment_ids)):
                for j in range(i + 1, len(comment_ids)):
                    self.topic_network.add_edge(
                        comment_ids[i],
                        comment_ids[j],
                        relationship='same_author'
                    )
        
        return self.topic_network
    
    def save_network(self, network: nx.Graph, output_path: Path) -> None:
        """
        Save network to disk in GraphML format.
        
        Args:
            network (nx.Graph): Network to save
            output_path (Path): Path to save network to
        """
        nx.write_graphml(network, output_path)
    
