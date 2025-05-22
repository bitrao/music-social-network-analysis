from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class DataPreprocessor:
    """Preprocesses and cleans YouTube comment data."""
    
    def __init__(self):
        pass
        
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process YouTube comment data.
        
        Args:
            data (Dict[str, Any]): Raw YouTube comment data
            
        Returns:
            Dict[str, Any]: Processed data
        """
        self.logger.info("Processing YouTube comment data")
        
        processed_data = {
            'collection_date': data.get('collection_date', datetime.now().isoformat()),
            'comments': []
        }
        
        for comment in data['comments']:
            # Clean comment text
            cleaned_text = self._clean_text(comment['text'])
            
            processed_comment = {
                'id': comment['id'],
                'text': cleaned_text,
                'original_text': comment['text'],  # Keep original text for reference
                'created_at': comment['created_at'],
                'author': comment['author'],
                'metrics': {
                    'like_count': comment.get('like_count', 0),
                    'reply_count': comment.get('reply_count', 0)
                }
            }
            
            processed_data['comments'].append(processed_comment)
            
        # Calculate engagement metrics
        processed_data['engagement_metrics'] = self._calculate_engagement_metrics(processed_data['comments'])
        
        return processed_data
        
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing URLs, special characters, etc.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        
        # Tokenize - simplified to avoid punkt_tab dependency
        tokens = text.split()
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Join tokens back into string
        preprocessed_text = ' '.join(tokens)
            
        return text
    
        
    def save_processed_data(self, data: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """
        Save processed data to disk.
        
        Args:
            data (Dict[str, Any]): Processed data
            output_path (Optional[Path]): Path to save data to
            
        Returns:
            Path: Path to saved file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path("data/processed") / f"youtube_comments_processed_{timestamp}.json"
            
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        self.logger.info(f"Processed data saved to {output_path}")
        return output_path 