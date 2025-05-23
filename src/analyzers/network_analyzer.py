import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
import json
from pathlib import Path
import numpy as np
from collections import defaultdict

class NetworkAnalyzer:
    """Analyzes networks and computes various network metrics."""
    
    def __init__(self):
        pass
        
    def compute_network_metrics(self, network: nx.Graph) -> Dict[str, Any]:
        """
        Compute various network metrics for the given network.
        
        Args:
            network (nx.Graph): Network to analyze
            
        Returns:
            Dict[str, Any]: Dictionary containing various network metrics
        """
        metrics = {}
        
        # Basic network metrics
        metrics['num_nodes'] = network.number_of_nodes()
        metrics['num_edges'] = network.number_of_edges()
        metrics['density'] = nx.density(network)
        
        # Connected components
        if isinstance(network, nx.Graph):
            metrics['num_components'] = nx.number_connected_components(network)
            metrics['largest_component_size'] = len(max(nx.connected_components(network), key=len))
        else:  # For directed graphs
            metrics['num_components'] = nx.number_weakly_connected_components(network)
            metrics['largest_component_size'] = len(max(nx.weakly_connected_components(network), key=len))
        
        # Centrality metrics
        metrics['degree_centrality'] = nx.degree_centrality(network)
        metrics['betweenness_centrality'] = nx.betweenness_centrality(network)
        metrics['closeness_centrality'] = nx.closeness_centrality(network)
        
        # For directed networks
        if isinstance(network, nx.DiGraph):
            metrics['in_degree_centrality'] = nx.in_degree_centrality(network)
            metrics['out_degree_centrality'] = nx.out_degree_centrality(network)
            metrics['pagerank'] = nx.pagerank(network)
        
        # Clustering and community metrics
        if isinstance(network, nx.Graph):
            metrics['average_clustering'] = nx.average_clustering(network)
            metrics['transitivity'] = nx.transitivity(network)
        
        return metrics
    
    def compute_community_metrics(self, network: nx.Graph) -> Dict[str, Any]:
        """
        Compute community detection metrics using Louvain method.
        
        Args:
            network (nx.Graph): Network to analyze
            
        Returns:
            Dict[str, Any]: Dictionary containing community metrics
        """
        try:
            import community as community_louvain
        except ImportError:
            print("Please install python-louvain package for community detection")
            return {}
        
        # Convert to undirected graph if needed
        if isinstance(network, nx.DiGraph):
            network = network.to_undirected()
        
        # Detect communities
        partition = community_louvain.best_partition(network)
        
        # Compute community metrics
        metrics = {
            'num_communities': len(set(partition.values())),
            'community_sizes': defaultdict(int),
            'node_communities': partition
        }
        
        # Count community sizes
        for community_id in partition.values():
            metrics['community_sizes'][community_id] += 1
        
        return metrics
    
    def compute_semantic_metrics(self, network: nx.Graph) -> Dict[str, Any]:
        """
        Compute semantic-specific metrics for the network.
        
        Args:
            network (nx.Graph): Network to analyze
            
        Returns:
            Dict[str, Any]: Dictionary containing semantic metrics
        """
        metrics = {}
        
        # Average edge weight (semantic similarity)
        if network.edges():
            weights = [d['weight'] for _, _, d in network.edges(data=True)]
            metrics['avg_similarity'] = np.mean(weights)
            metrics['max_similarity'] = np.max(weights)
            metrics['min_similarity'] = np.min(weights)
        
        # Most similar comment pairs
        if network.edges():
            edges_with_weights = [(u, v, d['weight']) for u, v, d in network.edges(data=True)]
            edges_with_weights.sort(key=lambda x: x[2], reverse=True)
            metrics['top_similar_pairs'] = edges_with_weights[:5]
        
        return metrics
    
    def compute_user_metrics(self, network: nx.Graph) -> Dict[str, Any]:
        """
        Compute user-specific metrics for the network.
        
        Args:
            network (nx.Graph): Network to analyze
            
        Returns:
            Dict[str, Any]: Dictionary containing user metrics
        """
        metrics = {}
        
        # User activity metrics
        user_comments = defaultdict(int)
        user_replies = defaultdict(int)
        
        for node in network.nodes():
            if 'author' in network.nodes[node]:
                author = network.nodes[node]['author']
                user_comments[author] += 1
                
                # Count replies for directed networks
                if isinstance(network, nx.DiGraph):
                    user_replies[author] += network.out_degree(node)
        
        metrics['user_activity'] = {
            'most_active_users': dict(sorted(user_comments.items(), 
                                           key=lambda x: x[1], 
                                           reverse=True)[:10]),
            'most_replied_users': dict(sorted(user_replies.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:10])
        }
        
        return metrics
    
    def get_network_summary(self, network: nx.Graph, is_semantic: bool = False) -> Dict[str, Any]:
        """
        Get a comprehensive summary of all network metrics.
        
        Args:
            network (nx.Graph): Network to analyze
            is_semantic (bool): Whether this is a semantic network
            
        Returns:
            Dict[str, Any]: Dictionary containing all network metrics
        """
        summary = {
            'basic_metrics': self.compute_network_metrics(network),
            'community_metrics': self.compute_community_metrics(network),
            'user_metrics': self.compute_user_metrics(network)
        }
        
        # Add semantic metrics if it's a semantic network
        if is_semantic:
            summary['semantic_metrics'] = self.compute_semantic_metrics(network)
        
        return summary
    
    def save_metrics(self, metrics: Dict[str, Any], output_path: Path) -> None:
        """
        Save network metrics to a JSON file.
        
        Args:
            metrics (Dict[str, Any]): Network metrics to save
            output_path (Path): Path to save metrics to
        """
        # Convert numpy types to Python native types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        metrics = convert_numpy(metrics)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2) 