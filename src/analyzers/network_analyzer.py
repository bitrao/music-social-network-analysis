import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
import nx_cugraph as nxcg

class NetworkAnalyzer:
    """Analyzes networks and computes various network metrics using GPU acceleration."""
    
    def __init__(self):
        pass
        
    def compute_network_metrics(self, network: nx.Graph, centrality_metrics: bool = False) -> Dict[str, Any]:
        """
        Compute various network metrics for the given network using GPU acceleration where available.
        
        Args:
            network (nx.Graph): Network to analyze
            
        Returns:
            Dict[str, Any]: Dictionary containing various network metrics
        """
        metrics = {}
        
        # Basic network metrics (using NetworkX as nx_cugraph doesn't have these)
        metrics['num_nodes'] = network.number_of_nodes()
        metrics['num_edges'] = network.number_of_edges()
        metrics['density'] = nx.density(network)
        
        # Convert to appropriate graph type for nx_cugraph
        # For connected components, we need to use undirected graph
        if isinstance(network, nx.DiGraph):
            g_undirected = nxcg.Graph()
            g_undirected.add_edges_from(network.edges())
            # Create a NetworkX undirected graph for centrality calculations
            nx_undirected = nx.Graph()
            nx_undirected.add_edges_from(network.edges())
        else:
            g_undirected = nxcg.Graph()
            g_undirected.add_edges_from(network.edges())
            nx_undirected = network
        
        # Connected components (using undirected graph)
        if isinstance(network, nx.Graph):
            metrics['num_components'] = nxcg.number_connected_components(g_undirected)
            metrics['largest_component_size'] = len(max(nxcg.connected_components(g_undirected), key=len))
        else:  # For directed graphs
            metrics['num_components'] = nxcg.number_weakly_connected_components(g_undirected)
            metrics['largest_component_size'] = len(max(nxcg.weakly_connected_components(g_undirected), key=len))
        
        if centrality_metrics:
            try:
                # Use the undirected graph for centrality calculations
                metrics['degree_centrality'] = nx.degree_centrality(nx_undirected)
                metrics['betweenness_centrality'] = nx.betweenness_centrality(nx_undirected)
                metrics['closeness_centrality'] = nx.closeness_centrality(nx_undirected)
                
                # For directed networks, use the original directed graph
                if isinstance(network, nx.DiGraph):
                    metrics['in_degree_centrality'] = nx.in_degree_centrality(network)
                    metrics['out_degree_centrality'] = nx.out_degree_centrality(network)
                    metrics['pagerank'] = nx.pagerank(network)
            except Exception as e:
                print(f"Warning: Some centrality metrics could not be computed: {str(e)}")
            
        # Clustering and community metrics (using NetworkX as nx_cugraph doesn't have these)
        if isinstance(network, nx.Graph):
            try:
                # Use the undirected graph for clustering calculations
                metrics['average_clustering'] = nx.average_clustering(nx_undirected)
                metrics['transitivity'] = nx.transitivity(nx_undirected)
            except Exception as e:
                print(f"Warning: Clustering metrics could not be computed: {str(e)}")
        
        return metrics
    
    def compute_community_metrics(self, network: nx.Graph) -> Dict[str, Any]:
        """
        Compute community detection metrics using NetworkX's Louvain method.
        
        Args:
            network (nx.Graph): Network to analyze
            
        Returns:
            Dict[str, Any]: Dictionary containing community metrics
        """
        try:
            import community  # python-louvain package
        except ImportError:
            print("Please install python-louvain package for community detection")
            return {}
        
        # Convert to undirected graph if needed
        if isinstance(network, nx.DiGraph):
            g = nx.Graph()
            g.add_edges_from(network.edges())
        else:
            g = network
        
        # Detect communities using Louvain method
        partition = community.best_partition(g)
        
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