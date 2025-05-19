"""
Network Creation Module for Social Media Data Analysis

This module provides functions to create network representations from social media data.
"""

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

def create_x_network(df: pd.DataFrame, target_id: str, timestamp: str = None) -> str:
    """Create network from X (Twitter) data focusing on user interactions."""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    G = nx.Graph()
    
    # Add nodes for each unique user
    users = df['user'].unique()
    G.add_nodes_from(users, node_type='user')
    
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
    network_file = save_network(G, target_id, 'x', timestamp)
    return network_file

def create_youtube_network(df: pd.DataFrame, target_id: str, timestamp: str = None) -> str:
    """Create network from YouTube data focusing on channel relationships."""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    G = nx.Graph()
    
    # Add nodes for channels and videos
    channels = df['channel'].unique()
    G.add_nodes_from(channels, node_type='channel')
    G.add_nodes_from(df['id'], node_type='video')
    
    # Create edges between channels and their videos
    edges = [(row['channel'], row['id']) for _, row in df.iterrows()]
    G.add_edges_from(edges)
    
    # Save network data
    network_file = save_network(G, target_id, 'youtube', timestamp)
    return network_file

def create_reddit_network(df: pd.DataFrame, target_id: str, timestamp: str = None) -> str:
    """Create network from Reddit data focusing on subreddit relationships."""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
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
    network_file = save_network(G, target_id, 'reddit', timestamp)
    return network_file

def create_tiktok_network(df: pd.DataFrame, target_id: str, timestamp: str = None) -> str:
    """Create network from TikTok data focusing on user-sound and user-user interactions."""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    G = nx.Graph()
    
    # Add nodes for users and their videos
    users = df['author'].unique()
    G.add_nodes_from(users, node_type='user')
    G.add_nodes_from(df['id'], node_type='video')
    
    # Create edges between users and their videos
    edges = [(row['author'], row['id']) for _, row in df.iterrows()]
    G.add_edges_from(edges)
    
    # Add edges for duets if available
    if 'duet_from' in df.columns:
        duet_edges = [(row['author'], row['duet_from']) 
                      for _, row in df.iterrows() 
                      if pd.notna(row.get('duet_from'))]
        G.add_edges_from(duet_edges)
    
    # Save network data
    network_file = save_network(G, target_id, 'tiktok', timestamp)
    return network_file

def save_network(G: nx.Graph, target_id: str, platform: str, timestamp: str) -> str:
    """Save network data and create visualization."""
    # Create directories if they don't exist
    Path("data/networks").mkdir(parents=True, exist_ok=True)
    Path("data/visualizations").mkdir(parents=True, exist_ok=True)
    
    # Save network data
    network_file = f"data/networks/{target_id}_{platform}_network_{timestamp}.graphml"
    nx.write_graphml(G, network_file)
    
    # Create and save visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # Color nodes by type if available
    node_types = set(nx.get_node_attributes(G, 'node_type').values())
    if node_types:
        colors = {
            'user': 'lightblue',
            'video': 'lightgreen',
            'channel': 'pink',
            'subreddit': 'orange',
            'author': 'yellow'
        }
        node_colors = [colors.get(G.nodes[node].get('node_type', 'default'), 'gray') 
                      for node in G.nodes()]
    else:
        node_colors = 'lightblue'
    
    nx.draw(G, pos, node_size=20, node_color=node_colors,
            with_labels=False, alpha=0.7)
    plt.title(f"{platform.upper()} Network for {target_id}")
    plt.savefig(f"data/visualizations/{target_id}_{platform}_network_{timestamp}.png")
    plt.close()
    
    return network_file

def export_network(network_file: str, target_id: str, platform: str) -> Dict[str, str]:
    """Export network in multiple formats."""
    G = nx.read_graphml(network_file)
    base_path = f"data/networks/{target_id}_{platform}"
    
    # Export paths
    export_files = {
        'graphml': network_file,
        'gexf': f"{base_path}.gexf",
        'edgelist': f"{base_path}.edgelist"
    }
    
    # Export in different formats
    nx.write_gexf(G, export_files['gexf'])
    nx.write_edgelist(G, export_files['edgelist'])
    
    return export_files

def analyze_network(G: nx.Graph) -> Dict[str, Any]:
    """Calculate basic network metrics."""
    metrics = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_clustering': nx.average_clustering(G),
    }
    
    # Add average path length if network is connected
    if nx.is_connected(G):
        metrics['avg_path_length'] = nx.average_shortest_path_length(G)
    
    # Add node type distribution if available
    node_types = nx.get_node_attributes(G, 'node_type')
    if node_types:
        type_dist = pd.Series(node_types).value_counts().to_dict()
        metrics['node_type_distribution'] = type_dist
    
    return metrics 