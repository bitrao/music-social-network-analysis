import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
from .network_analyzer import NetworkAnalyzer

class NetworkCreator:
    """Creates various types of networks from YouTube comment data."""
    
    def __init__(self):
        self.comment_network = nx.DiGraph()
        self.video_comment_network = nx.DiGraph()
        self.semantic_network = nx.Graph()
        self.category_network = nx.DiGraph()
        self.analyzer = NetworkAnalyzer()
        
    def create_comment_reply_network(self, data: Dict[str, Any]) -> nx.DiGraph:
        """
        Create a network of comments and their replies.
        
        Args:
            data (Dict[str, Any]): Processed comment data
            
        Returns:
            nx.DiGraph: Directed graph of comments and replies
        """
        self.comment_network.clear()
        
        # Add all comments as nodes
        for comment in data['comments']:
            self.comment_network.add_node(
                comment['id'],
                text=comment['text'],
                author=comment['author'],
                created_at=comment['created_at'],
                metrics=comment['metrics']
            )
            
            # Add edges for replies
            if 'replies' in comment:
                for reply in comment['replies']:
                    self.comment_network.add_edge(
                        comment['id'],
                        reply['id'],
                        relationship='reply'
                    )
                    
        return self.comment_network
    
    def create_video_comment_network(self, data: Dict[str, Any]) -> nx.DiGraph:
        """
        Create a network of videos and their comments.
        
        Args:
            data (Dict[str, Any]): Processed comment data
            
        Returns:
            nx.DiGraph: Directed graph of videos and comments
        """
        self.video_comment_network.clear()
        
        # Add video as central node
        video_id = data.get('video_id', 'unknown')
        self.video_comment_network.add_node(
            video_id,
            type='video',
            title=data.get('video_title', ''),
            metrics=data.get('video_metrics', {})
        )
        
        # Add comments and connect to video
        for comment in data['comments']:
            self.video_comment_network.add_node(
                comment['id'],
                type='comment',
                text=comment['text'],
                author=comment['author']
            )
            self.video_comment_network.add_edge(
                video_id,
                comment['id'],
                relationship='has_comment'
            )
            
        return self.video_comment_network
    
    def create_semantic_network(self, data: Dict[str, Any], similarity_threshold: float = 0.3) -> nx.Graph:
        """
        Create a semantic network based on comment similarity.
        
        Args:
            data (Dict[str, Any]): Processed comment data
            similarity_threshold (float): Minimum similarity score to create an edge
            
        Returns:
            nx.Graph: Undirected graph of semantically related comments
        """
        self.semantic_network.clear()
        
        # Extract comment texts
        comments = [comment['text'] for comment in data['comments']]
        comment_ids = [comment['id'] for comment in data['comments']]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(comments)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Add nodes
        for comment_id, comment in zip(comment_ids, data['comments']):
            self.semantic_network.add_node(
                comment_id,
                text=comment['text'],
                author=comment['author']
            )
        
        # Add edges for similar comments
        for i in range(len(comment_ids)):
            for j in range(i + 1, len(comment_ids)):
                if similarity_matrix[i, j] > similarity_threshold:
                    self.semantic_network.add_edge(
                        comment_ids[i],
                        comment_ids[j],
                        weight=similarity_matrix[i, j]
                    )
                    
        return self.semantic_network
    
    def create_category_network(self, data: Dict[str, Any]) -> nx.DiGraph:
        """
        Create a network of categories -> videos -> comments.
        
        Args:
            data (Dict[str, Any]): Processed comment data
            
        Returns:
            nx.DiGraph: Directed graph of categories, videos, and comments
        """
        self.category_network.clear()
        
        # Add category node
        category = data.get('category', 'unknown')
        self.category_network.add_node(
            category,
            type='category'
        )
        
        # Add video node and connect to category
        video_id = data.get('video_id', 'unknown')
        self.category_network.add_node(
            video_id,
            type='video',
            title=data.get('video_title', '')
        )
        self.category_network.add_edge(
            category,
            video_id,
            relationship='contains'
        )
        
        # Add comments and connect to video
        for comment in data['comments']:
            self.category_network.add_node(
                comment['id'],
                type='comment',
                text=comment['text'],
                author=comment['author']
            )
            self.category_network.add_edge(
                video_id,
                comment['id'],
                relationship='has_comment'
            )
            
        return self.category_network
    
    def save_network(self, network: nx.Graph, output_path: Path) -> None:
        """
        Save network to disk in GraphML format.
        
        Args:
            network (nx.Graph): Network to save
            output_path (Path): Path to save network to
        """
        nx.write_graphml(network, output_path)
    
    def analyze_network(self, network: nx.Graph, is_semantic: bool = False) -> Dict[str, Any]:
        """
        Analyze a network using the NetworkAnalyzer.
        
        Args:
            network (nx.Graph): Network to analyze
            is_semantic (bool): Whether this is a semantic network
            
        Returns:
            Dict[str, Any]: Network metrics
        """
        return self.analyzer.get_network_summary(network, is_semantic)
