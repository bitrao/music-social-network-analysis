import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import yaml
from pathlib import Path
import os
from dotenv import load_dotenv

from .twitter_collector import TwitterCollector
from .youtube_collector import YouTubeCollector
from .reddit_collector import RedditCollector
# Import other collectors as they are implemented
# from .instagram_collector import InstagramCollector
# from .youtube_collector import YouTubeCollector
# etc.

class DataCollectionOrchestrator:
    """Orchestrates data collection from multiple platforms."""
    
    def __init__(self, config_path: str = "config/data_sources.yaml"):
        """
        Initialize the orchestrator.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        self.load_config(config_path)
        self.load_environment()
        self.initialize_collectors()
        
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def load_config(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def load_environment(self):
        """Load environment variables."""
        load_dotenv()
        
    def initialize_collectors(self):
        """Initialize all enabled collectors."""
        self.collectors = {}
        
        # Initialize Twitter collector if enabled
        if self.config['twitter']['enabled']:
            self.collectors['twitter'] = TwitterCollector(self.config)
            
        # Initialize YouTube collector if enabled
        if self.config['youtube']['enabled']:
            self.collectors['youtube'] = YouTubeCollector(self.config)
            
        # Initialize Reddit collector if enabled
        if self.config['reddit']['enabled']:
            self.collectors['reddit'] = RedditCollector(self.config)
            
        # Initialize other collectors as they are implemented
        # if self.config['instagram']['enabled']:
        #     self.collectors['instagram'] = InstagramCollector(self.config)
        # etc.
        
    async def collect_data(self, song_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Collect data for a specific song from all enabled platforms.
        
        Args:
            song_id (str): Song identifier
            days (int): Number of days to collect data for
            
        Returns:
            Dict[str, Any]: Collected data from all platforms
        """
        self.logger.info(f"Starting data collection for song {song_id}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Collect data from all enabled platforms concurrently
        collection_tasks = []
        for platform, collector in self.collectors.items():
            task = asyncio.create_task(
                self._collect_from_platform(collector, song_id, start_date, end_date)
            )
            collection_tasks.append(task)
            
        # Wait for all collection tasks to complete
        results = await asyncio.gather(*collection_tasks, return_exceptions=True)
        
        # Process results
        collected_data = {
            'song_id': song_id,
            'collection_date': datetime.now().isoformat(),
            'time_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'platforms': {}
        }
        
        for platform, result in zip(self.collectors.keys(), results):
            if isinstance(result, Exception):
                self.logger.error(f"Error collecting data from {platform}: {str(result)}")
                collected_data['platforms'][platform] = {
                    'error': str(result),
                    'status': 'failed'
                }
            else:
                collected_data['platforms'][platform] = {
                    'data': result,
                    'status': 'success'
                }
                
        return collected_data
    
    async def _collect_from_platform(
        self,
        collector: Any,
        song_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Collect data from a specific platform.
        
        Args:
            collector: Platform-specific collector instance
            song_id (str): Song identifier
            start_date (datetime): Start date for data collection
            end_date (datetime): End date for data collection
            
        Returns:
            Dict[str, Any]: Collected data from the platform
        """
        try:
            data = await collector.collect(song_id, start_date, end_date)
            return data
        except Exception as e:
            self.logger.error(f"Error in {collector.__class__.__name__}: {str(e)}")
            raise

async def main():
    """Main entry point for data collection."""
    # Example usage
    orchestrator = DataCollectionOrchestrator()
    
    # Collect data for a specific song
    song_id = "example_song_123"
    data = await orchestrator.collect_data(song_id)
    
    # Save collected data
    output_path = Path("data/raw") / f"{song_id}_all_platforms_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
        
    print(f"Data collection completed. Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main()) 