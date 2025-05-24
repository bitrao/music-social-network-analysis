import numpy as np
import pandas as pd
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
from typing import List, Dict, Tuple, Optional, Union
import logging
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopicModeller:
    def __init__(self, n_topics: int = 5):
        """
        Initialize the topic modeller for music content analysis.
        
        Args:
            n_topics (int): Number of topics to extract (default: 5)
        """
        self.n_topics = n_topics
        self.dictionary = None
        self.corpus = None
        self.model = None
        self.texts = None
        self.vis_data = None
        self.tokenized_texts = None
        self.topic_names = {}  # Store custom topic names
        
    def preprocess_texts(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess texts for gensim.
        
        Args:
            texts (List[str]): List of text documents
            
        Returns:
            List[List[str]]: List of tokenized documents
        """
        # Simple tokenization (you might want to add more preprocessing)
        tokenized_texts = [text.lower().split() for text in texts]
        return tokenized_texts
        
    def fit(self, texts: List[str]) -> None:
        """
        Fit the topic model to the given texts.
        
        Args:
            texts (List[str]): List of text documents
        """
        self.texts = texts
        
        # Preprocess texts
        self.tokenized_texts = self.preprocess_texts(texts)
        
        # Create dictionary and corpus
        self.dictionary = corpora.Dictionary(self.tokenized_texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.tokenized_texts]
        
        # Train LDA model
        self.model = models.LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.n_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            eta='auto'
        )
        
        # Prepare visualization data
        self.vis_data = pyLDAvis.gensim_models.prepare(
            self.model,
            self.corpus,
            self.dictionary
        )
        
        # Generate default topic names
        self._generate_default_topic_names()
        
        logger.info(f"Fitted LDA model with {self.n_topics} topics")
    
    def _generate_default_topic_names(self) -> None:
        """
        Generate default names for topics based on their top words.
        """
        topics = self.get_topics(n_words=5)
        for topic_id, words in topics.items():
            # Create name from top 3 words
            name = "_".join([word for word, _ in words[:3]])
            self.topic_names[topic_id] = name
    
    def set_topic_name(self, topic_id: int, name: str) -> None:
        """
        Set a custom name for a topic.
        
        Args:
            topic_id (int): ID of the topic
            name (str): Custom name for the topic
        """
        if topic_id not in range(self.n_topics):
            raise ValueError(f"Topic ID must be between 0 and {self.n_topics-1}")
        self.topic_names[topic_id] = name
    
    def get_topic_name(self, topic_id: int) -> str:
        """
        Get the name of a topic.
        
        Args:
            topic_id (int): ID of the topic
            
        Returns:
            str: Name of the topic
        """
        return self.topic_names.get(topic_id, f"Topic_{topic_id}")
    
    def get_topics(self, n_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get the top words for each topic.
        
        Args:
            n_words (int): Number of top words to return per topic
            
        Returns:
            Dict[int, List[Tuple[str, float]]]: Dictionary mapping topic IDs to their top words and scores
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        topics = {}
        for topic_id in range(self.n_topics):
            topic_words = self.model.show_topic(topic_id, n_words)
            topics[topic_id] = topic_words
        
        return topics
    
    def get_document_topics(self, texts: List[str]) -> List[Dict[int, float]]:
        """
        Get the topic distribution for each document.
        
        Args:
            texts (List[str]): List of text documents
            
        Returns:
            List[Dict[int, float]]: List of topic distributions for each document
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Preprocess and convert new texts to corpus
        tokenized_texts = self.preprocess_texts(texts)
        new_corpus = [self.dictionary.doc2bow(text) for text in tokenized_texts]
        
        # Get topic distributions
        results = []
        for doc in new_corpus:
            doc_topics = dict(self.model.get_document_topics(doc))
            results.append(doc_topics)
        
        return results

    def analyze_texts(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze texts and return both the original texts and their topic distributions.
        
        Args:
            texts (List[str]): List of text documents
            
        Returns:
            pd.DataFrame: DataFrame containing original texts, their topic distributions,
                         and the dominant topic name for each document
        """
        # Get topic distributions
        topic_distributions = self.get_document_topics(texts)
        
        # Convert to DataFrame
        result_df = pd.DataFrame({
            'text': texts,
            **{f'topic_{i}': [dist.get(i, 0.0) for dist in topic_distributions] 
               for i in range(self.n_topics)}
        })
        
        # Add dominant topic and its name
        result_df['dominant_topic_id'] = result_df[[f'topic_{i}' for i in range(self.n_topics)]].idxmax(axis=1)
        result_df['dominant_topic_id'] = result_df['dominant_topic_id'].str.extract('(\d+)').astype(int)
        result_df['dominant_topic_name'] = result_df['dominant_topic_id'].apply(self.get_topic_name)
        
        return result_df
    
    def get_coherence_score(self) -> float:
        """
        Calculate the coherence score of the model.
        
        Returns:
            float: Coherence score
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        coherence_model = CoherenceModel(
            model=self.model,
            texts=self.tokenized_texts,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        return coherence_model.get_coherence()
    
    def find_optimal_topics(self, 
                          texts: List[str],
                          start: int = 2,
                          end: int = 20,
                          step: int = 1,
                          coherence_type: str = 'c_v') -> Tuple[int, float, Dict[int, float]]:
        """
        Find the optimal number of topics based on coherence scores.
        
        Args:
            texts (List[str]): List of text documents
            start (int): Starting number of topics
            end (int): Ending number of topics
            step (int): Step size for number of topics
            coherence_type (str): Type of coherence measure ('c_v', 'u_mass', 'c_uci', 'c_npmi')
            
        Returns:
            Tuple[int, float, Dict[int, float]]: Optimal number of topics, best coherence score, and all coherence scores
        """
        # Preprocess texts
        tokenized_texts = self.preprocess_texts(texts)
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        # Calculate coherence scores for different numbers of topics
        coherence_scores = {}
        best_score = -1
        best_n_topics = start
        
        for n_topics in range(start, end + 1, step):
            # Train LDA model
            lda_model = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=n_topics,
                random_state=42,
                passes=10,
                alpha='auto',
                eta='auto'
            )
            
            # Calculate coherence score
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=tokenized_texts,
                dictionary=dictionary,
                coherence=coherence_type
            )
            score = coherence_model.get_coherence()
            coherence_scores[n_topics] = score
            
            # Update best score
            if score > best_score:
                best_score = score
                best_n_topics = n_topics
            
            logger.info(f"Topics: {n_topics}, Coherence Score: {score:.4f}")
        
        return best_n_topics, best_score, coherence_scores
    
    def plot_coherence_scores(self, coherence_scores: Dict[int, float]) -> None:
        """
        Plot coherence scores for different numbers of topics.
        
        Args:
            coherence_scores (Dict[int, float]): Dictionary mapping number of topics to coherence scores
        """
        plt.figure(figsize=(10, 6))
        plt.plot(list(coherence_scores.keys()), list(coherence_scores.values()), 'bo-')
        plt.xlabel('Number of Topics')
        plt.ylabel('Coherence Score')
        plt.title('Coherence Scores for Different Numbers of Topics')
        plt.grid(True)
        plt.show()
    
    def visualize_topics(self, output_path: Optional[str] = None) -> None:
        """
        Create an interactive visualization of the topics using pyLDAvis.
        If used in a Jupyter notebook, displays the visualization directly.
        If output_path is provided, saves the visualization to an HTML file.
        
        Args:
            output_path (str, optional): Path to save the visualization HTML file
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        if self.vis_data is None:
            self.vis_data = pyLDAvis.gensim_models.prepare(
                self.model,
                self.corpus,
                self.dictionary
            )
        
        # Try to display in notebook
        try:
            display(pyLDAvis.display(self.vis_data))
        except:
            # If not in notebook, save to file
            if output_path:
                pyLDAvis.save_html(self.vis_data, output_path)
                logger.info(f"Topic visualization saved to {output_path}")
            else:
                logger.warning("Not in a Jupyter notebook and no output path provided. Visualization not displayed.")
    
    def display_topics_in_notebook(self) -> None:
        """
        Display topics in a formatted way in Jupyter notebooks.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        topics = self.get_topics()
        
        # Create HTML table
        html = "<table style='width:100%'>"
        html += "<tr><th>Topic ID</th><th>Topic Name</th><th>Top Words</th></tr>"
        
        for topic_id, words in topics.items():
            word_list = ", ".join([f"{word} ({score:.3f})" for word, score in words])
            topic_name = self.get_topic_name(topic_id)
            html += f"<tr><td>{topic_id}</td><td>{topic_name}</td><td>{word_list}</td></tr>"
        
        html += "</table>"
        
        # Display in notebook
        display(HTML(html))
    
    def get_topic_documents(self, topic_id: int, n_docs: int = 5) -> List[str]:
        """
        Get the most representative documents for a topic.
        
        Args:
            topic_id (int): ID of the topic
            n_docs (int): Number of documents to return
            
        Returns:
            List[str]: List of most representative documents
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Get topic distribution for each document
        doc_topics = self.get_document_topics(self.texts)
        
        # Sort documents by topic probability
        doc_scores = [(i, dist.get(topic_id, 0.0)) for i, dist in enumerate(doc_topics)]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n_docs documents
        return [self.texts[i] for i, _ in doc_scores[:n_docs]]
