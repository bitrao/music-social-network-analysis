from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import json

class BaseCollector(ABC):
    """Base class for all data collectors."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the collector with configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary for the collector
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @abstractmethod
    async def collect(self, song_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Collect data for a specific song within a date range.
        
        Args:
            song_id (str): Unique identifier for the song
            start_date (datetime): Start date for data collection
            end_date (datetime): End date for data collection
            
        Returns:
            Dict[str, Any]: Collected data
        """
        pass
    
    @abstractmethod
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate the collected data.
        
        Args:
            data (Dict[str, Any]): Data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        pass
    
    def save_data(self, data: Dict[str, Any], song_id: str) -> Path:
        """
        Save collected data to disk.
        
        Args:
            data (Dict[str, Any]): Data to save
            song_id (str): Song identifier
            
        Returns:
            Path: Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{song_id}_{self.__class__.__name__}_{timestamp}.json"
        output_path = Path("data/raw") / filename
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        self.logger.info(f"Data saved to {output_path}")
        return output_path
    
    async def handle_rate_limit(self):
        """Handle rate limiting for API calls."""
        if hasattr(self, 'rate_limit'):
            # Implement rate limiting logic here
            pass
    
    async def retry_on_failure(self, func, *args, max_retries: int = 3, **kwargs):
        """
        Retry a function on failure.
        
        Args:
            func: Function to retry
            *args: Positional arguments for the function
            max_retries (int): Maximum number of retries
            **kwargs: Keyword arguments for the function
            
        Returns:
            Any: Result of the function call
        """
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                await self.handle_rate_limit() 