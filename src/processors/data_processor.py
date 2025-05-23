from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import html

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class DataPreprocessor:
    """Preprocesses and cleans text data from a DataFrame."""
    
    def __init__(self):
        pass
        
    def preprocess_data(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Process text data from a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame containing text data
            text_column (str): Name of the column containing text to process
            
        Returns:
            pd.DataFrame: Processed DataFrame with cleaned text
        """
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Clean the text column
        processed_df[f'cleaned_{text_column}'] = processed_df[text_column].apply(self._clean_text)
        
        return processed_df
        
    def _clean_text(self, text: str) -> str:
        """
        Clean text while preserving emojis and emoticons for VADER sentiment analysis.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Join tokens back into string
        preprocessed_text = ' '.join(tokens)
            
        return preprocessed_text
    
    def save_processed_data(self, df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
        """
        Save processed DataFrame to CSV.
        
        Args:
            df (pd.DataFrame): Processed DataFrame
            output_path (Optional[Path]): Path to save data to
            
        Returns:
            Path: Path to saved file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path("../data/processed") / f"processed_data_{timestamp}.csv"
            
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
            
        df.to_csv(output_path, index=False)
        return output_path 