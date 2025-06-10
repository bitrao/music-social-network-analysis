import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from IPython.display import display, HTML
from IPython import get_ipython
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTopicModeller:
    def __init__(self, 
                 n_topics: int = 5,
                 min_topic_size: int = 10,
                 language: str = "english",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 use_cuda: bool = True,
                 reduce_outliers: bool = True):
        """
        Initialize the BERTopic modeler for music content analysis.
        
        Args:
            n_topics (int): Target number of topics to extract (default: 5)
            min_topic_size (int): Minimum size of topics (default: 10)
            language (str): Language of the texts (default: "english")
            embedding_model (str): Name of the sentence-transformer model to use (default: "all-MiniLM-L6-v2")
            use_cuda (bool): Whether to use CUDA for acceleration (default: True)
            reduce_outliers (bool): Whether to use parameters optimized for reducing outliers (default: True)
        """
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        self.language = language
        self.model = None
        self.topic_names = {}  # Store topic names
        self.reduce_outliers = reduce_outliers
        
        # Check CUDA availability
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU device")
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(
            embedding_model,
            device=self.device
        )
        
        # Initialize BERTopic with custom parameters
        self.model = BERTopic(
            language=language,
            min_topic_size=min_topic_size,
            embedding_model=self.embedding_model,
            umap_model=UMAP(
                n_neighbors=30 if reduce_outliers else 15,
                n_components=5,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            ),
            hdbscan_model=HDBSCAN(
                min_cluster_size=min_topic_size,
                min_samples=1 if reduce_outliers else 5,  # More conservative default
                metric='euclidean',
                cluster_selection_method='eom',
                cluster_selection_epsilon=0.1 if reduce_outliers else 0.0,
                prediction_data=True
            ),
            vectorizer_model=CountVectorizer(
                stop_words="english",
                min_df=2,
                ngram_range=(1, 2)
            ),
            verbose=True
        )
        
        # Set number of topics if specified
        if n_topics is not None:
            self.model.nr_topics = n_topics

    def fit(self, texts: List[str]) -> None:
        """
        Fit the BERTopic model to the given texts.
        
        Args:
            texts (List[str]): List of text documents
        """
        # Convert texts to list of strings
        texts = [str(text) for text in texts]
        
        # Fit the model
        self.model.fit(texts)
        
        # Generate default topic names
        self._generate_topic_names()
        
        # Log topic statistics
        topic_info = self.model.get_topic_info()
        n_outliers = len(topic_info[topic_info['Topic'] == -1])
        n_documents = len(texts)
        outlier_percentage = (n_outliers / n_documents) * 100
        
        logger.info(f"Fitted BERTopic model with {len(self.model.get_topics())} topics")
        logger.info(f"Number of outliers: {n_outliers} ({outlier_percentage:.1f}% of documents)")
        
        # Warn if outlier percentage is too high
        if outlier_percentage > 30:
            logger.warning(
                f"High percentage of outliers ({outlier_percentage:.1f}%). "
                "Consider adjusting parameters or using reduce_outliers=True"
            )

    def _generate_topic_names(self) -> None:
        """Generate names for topics based on their top words."""
        topic_info = self.model.get_topic_info()
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id != -1:  # Skip outlier topic
                words = self.model.get_topic(topic_id)
                # Create name from top 3 words
                name = "_".join([word for word, _ in words[:3]])
                self.topic_names[topic_id] = name

    def get_topic_name(self, topic_id: int) -> str:
        """Get the name of a topic."""
        return self.topic_names.get(topic_id, f"Topic_{topic_id}")

    def get_document_topics(self, texts: List[str], filter_outliers: bool = False) -> pd.DataFrame:
        """
        Get the topic distribution for each document.
        
        Args:
            texts (List[str]): List of text documents
            filter_outliers (bool): Whether to filter out outlier documents (default: False)
            
        Returns:
            pd.DataFrame: DataFrame containing original texts and their topic distributions
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Convert texts to list of strings
        texts = [str(text) for text in texts]
        
        # Get topic probabilities
        topics, probs = self.model.transform(texts)
        
        # Create DataFrame with results
        result_df = pd.DataFrame({
            'text': texts,
            'topic_id': topics,
            'topic_name': [self.get_topic_name(topic) for topic in topics],
            'is_outlier': [topic == -1 for topic in topics]
        })
        
        # Add topic probabilities
        for i in range(len(self.model.get_topics())):
            result_df[f'topic_{i}_prob'] = [float(prob[i]) if isinstance(prob, (list, np.ndarray)) else float(prob) for prob in probs]
        
        # Filter outliers if requested
        if filter_outliers:
            result_df = result_df[~result_df['is_outlier']]
            logger.info(f"Filtered out {len(texts) - len(result_df)} outlier documents")
        
        return result_df

    def get_outlier_documents(self, texts: List[str]) -> pd.DataFrame:
        """
        Get documents that are classified as outliers.
        
        Args:
            texts (List[str]): List of text documents
            
        Returns:
            pd.DataFrame: DataFrame containing outlier documents and their details
        """
        result_df = self.get_document_topics(texts)
        outlier_df = result_df[result_df['is_outlier']].copy()
        
        # Add outlier score (1 - max topic probability)
        outlier_df['outlier_score'] = 1 - outlier_df[[col for col in outlier_df.columns if col.startswith('topic_')]].max(axis=1)
        
        return outlier_df.sort_values('outlier_score', ascending=False)

    def visualize_outliers(self, texts: List[str], output_path: Optional[str] = None) -> None:
        """
        Create a visualization of outlier documents and their relationship to topics.
        
        Args:
            texts (List[str]): List of text documents
            output_path (str, optional): Path to save the visualization HTML file
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        try:
            # Get outlier documents
            outlier_df = self.get_outlier_documents(texts)
            
            # Create visualization
            fig = self.model.visualize_distribution(outlier_df['outlier_score'])
            
            # Check if we're in a Jupyter notebook
            try:
                get_ipython()
                # We're in a notebook, display directly
                fig.show()
                logger.info("Displayed outlier visualization in notebook")
            except NameError:
                # Not in a notebook, handle browser display
                if output_path:
                    fig.write_html(output_path)
                    logger.info(f"Outlier visualization saved to {output_path}")
                else:
                    import tempfile
                    import webbrowser
                    import os
                    
                    temp_file = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
                    fig.write_html(temp_file.name)
                    webbrowser.open('file://' + os.path.realpath(temp_file.name))
                    logger.info(f"Opening visualization in browser. File saved at: {temp_file.name}")
            
        except Exception as e:
            logger.error(f"Error creating outlier visualization: {str(e)}")
            raise

    def visualize_topics(self, output_path: Optional[str] = None) -> None:
        """
        Create an interactive visualization of the topics.
        If used in a Jupyter notebook, displays the visualization directly.
        If output_path is provided, saves the visualization to an HTML file.
        
        Args:
            output_path (str, optional): Path to save the visualization HTML file
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        try:
            # Create visualization
            fig = self.model.visualize_topics()
            
            # Check if we're in a Jupyter notebook
            try:
                get_ipython()
                # We're in a notebook, display directly
                fig.show()
                logger.info("Displayed topic visualization in notebook")
            except NameError:
                # Not in a notebook, handle browser display
                if output_path:
                    fig.write_html(output_path)
                    logger.info(f"Topic visualization saved to {output_path}")
                else:
                    import tempfile
                    import webbrowser
                    import os
                    
                    temp_file = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
                    fig.write_html(temp_file.name)
                    webbrowser.open('file://' + os.path.realpath(temp_file.name))
                    logger.info(f"Opening visualization in browser. File saved at: {temp_file.name}")
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            # Fallback to basic visualization
            try:
                fig = self.model.visualize_barchart()
                fig.show()
            except Exception as e2:
                logger.error(f"Error creating fallback visualization: {str(e2)}")
                raise

    def display_topics(self) -> None:
        """
        Display topics in a formatted way in Jupyter notebooks.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Get topic info
        topic_info = self.model.get_topic_info()
        
        # Create HTML table
        html = "<table style='width:100%'>"
        html += "<tr><th>Topic ID</th><th>Topic Name</th><th>Count</th><th>Top Words</th></tr>"
        
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            count = row['Count']
            words = self.model.get_topic(topic_id)
            word_list = ", ".join([f"{word} ({score:.3f})" for word, score in words[:5]])
            topic_name = self.get_topic_name(topic_id)
            html += f"<tr><td>{topic_id}</td><td>{topic_name}</td><td>{count}</td><td>{word_list}</td></tr>"
        
        html += "</table>"
        
        # Display in notebook
        display(HTML(html)) 