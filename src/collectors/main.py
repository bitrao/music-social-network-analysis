"""
Main Script for Social Media Data Collection and Network Analysis

This script orchestrates the collection of social media data about songs and performs
network analysis to understand the relationships between users, content, and platforms.

The script will:
1. Collect data from multiple platforms
2. Save data in CSV format
3. Create network visualizations
4. Perform social network analysis

Required Environment Variables:
    X_API_KEY: X API key
    X_API_SECRET: X API secret
    YOUTUBE_API_KEY: YouTube Data API key
    REDDIT_CLIENT_ID: Reddit API client ID
    REDDIT_CLIENT_SECRET: Reddit API client secret
    TIKTOK_MS_TOKEN: TikTok MS token

Usage:
    1. Set up environment variables (use .env file or export directly)
    2. Run the script: python main.py
    3. Enter the song identifier when prompted
    4. Results will be saved in data/raw directory
"""

import os
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

from collectors import (
    collect_x_data,
    collect_youtube_data,
    collect_reddit_data,
    collect_tiktok_data,
    save_results,
    analyze_network
)

async def collect_and_analyze(song_id: str) -> Optional[Dict[str, Any]]:
    """
    Collect data about a song and perform network analysis.
    
    Args:
        song_id (str): Identifier for the song
        
    Returns:
        Optional[Dict[str, Any]]: Analysis results and file paths
    """
    # Load environment variables
    load_dotenv()
    
    # Verify environment variables
    required_vars = [
        'X_API_KEY',
        'X_API_SECRET',
        'YOUTUBE_API_KEY',
        'REDDIT_CLIENT_ID',
        'REDDIT_CLIENT_SECRET',
        'TIKTOK_MS_TOKEN'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file or environment")
        return None
    
    # Create collection tasks
    print(f"\nStarting data collection for '{song_id}'...")
    start_time = datetime.now()
    
    tasks = [
        collect_x_data(
            song_id,
            os.getenv('X_API_KEY'),
            os.getenv('X_API_SECRET')
        ),
        collect_youtube_data(
            song_id,
            os.getenv('YOUTUBE_API_KEY')
        ),
        collect_reddit_data(
            song_id,
            os.getenv('REDDIT_CLIENT_ID'),
            os.getenv('REDDIT_CLIENT_SECRET')
        ),
        collect_tiktok_data(
            song_id,
            os.getenv('TIKTOK_MS_TOKEN')
        )
    ]
    
    # Run collectors
    results = await asyncio.gather(*tasks, return_exceptions=True)
    duration = (datetime.now() - start_time).total_seconds()
    
    # Process results
    successful_platforms = [r['platform'] for r in results if isinstance(r, dict) and 'error' not in r]
    failed_platforms = [r['platform'] for r in results if isinstance(r, dict) and 'error' in r]
    
    print(f"\nCollection completed in {duration:.2f} seconds")
    print(f"Successful platforms: {', '.join(successful_platforms) if successful_platforms else 'None'}")
    if failed_platforms:
        print(f"Failed platforms: {', '.join(failed_platforms)}")
    
    if not successful_platforms:
        print("\nError: All collectors failed. No data to analyze.")
        return None
    
    # Save data and create networks
    print("\nSaving data and creating network visualizations...")
    saved_files = save_results(song_id, results)
    
    # Perform network analysis
    print("\nPerforming network analysis...")
    network_analysis = {}
    for key, file_path in saved_files.items():
        if key.endswith('_network'):
            platform = key.replace('_network', '')
            network_analysis[platform] = analyze_network(file_path)
    
    # Print analysis summary
    print("\nNetwork Analysis Summary:")
    for platform, metrics in network_analysis.items():
        print(f"\n{platform.upper()} Network:")
        print(f"- Nodes: {metrics['num_nodes']}")
        print(f"- Edges: {metrics['num_edges']}")
        print(f"- Network Density: {metrics['density']:.4f}")
        print(f"- Average Clustering: {metrics['avg_clustering']:.4f}")
        print(f"- Number of Connected Components: {len(metrics['connected_components'])}")
    
    return {
        'saved_files': saved_files,
        'network_analysis': network_analysis,
        'collection_time': duration,
        'successful_platforms': successful_platforms
    }

def main():
    """Main entry point for data collection and analysis."""
    # Create necessary directories
    for dir_name in ['raw', 'processed', 'networks', 'visualizations']:
        Path(f"data/{dir_name}").mkdir(parents=True, exist_ok=True)
    
    # Get song ID
    print("\nSocial Media Music Data Collector and Network Analyzer")
    print("--------------------------------------------------")
    song_id = input("\nEnter song identifier (title, artist, or ID): ").strip()
    
    if not song_id:
        print("Error: Song identifier cannot be empty")
        return
    
    # Run collection and analysis
    asyncio.run(collect_and_analyze(song_id))
    
    print("\nAnalysis complete! Check the data/ directory for:")
    print("- CSV files: data/processed/")
    print("- Network files: data/networks/")
    print("- Visualizations: data/visualizations/")

if __name__ == "__main__":
    main() 