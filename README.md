# YouTube Music Comments Analysis: 
## "Luther" by Kendrick Lamar

This project presents a comprehensive analysis of audience engagement and sentiment through YouTube comments on Kendrick Lamar's "Luther" music video. The analysis focuses on understanding viewer reactions, engagement patterns, and sentiment trends to provide insights into audience reception and potential areas for improvement.

## Introduction
YouTube comments provide valuable insights into how audiences receive and engage with music content. This project analyzes comment data from Kendrick Lamar's "Luther" music video to understand audience sentiment, engagement patterns, and key discussion topics, helping to understand the impact and reception of this specific release.

## Case Study: "Luther" by Kendrick Lamar

### Analysis Focus
- Initial audience reaction
- Lyrical interpretation discussions
- Cultural impact analysis
- Fan engagement patterns
- Sentiment evolution over time

## Methodology

### Data Collection
Our analysis framework focuses on YouTube comments data from the videos around the song "Luther" which are categorized as:
- Official Videos
- Repost Videos
- Reaction Videos
- Analysis Videos 

#### Data Overview
- Comments, replies text and metadata
- User engagement metrics (likes, replies)
- Timestamp data

### Analysis Framework
The project is structured as follows:
```
music-social-network-analysis/
├── src/                    # Source code
│   ├── collectors/         # YouTube data collection modules
│   ├── processors/         # Comment preprocessing
│   ├── analyzers/          # Sentiment, topic and network analysis
│   ├── notebooks/          # Analysis notebooks
│   └── data/              # Processed comment data
├── images/                # Generated visualizations
├── requirements.txt       # Project dependencies
└── .python-version       # Python version specification
```

### Technical Requirements
- Python 3.x (see .python-version for specific version)
- pip (Python package manager)
- Git
- YouTube Data API credentials

## Key Findings

### Topic Modeling

Two distinct topic modeling methods were employed: 
- BERTopic, a modern transformer-based technique, was used to extract 20 granular topics by leveraging BERT embeddings and class-based TF-IDF scores. 
- LDA, a probabilistic model based on word co-occurrence, was used to extract 7 broader thematic clusters for comparison.
  
  ![wordcloud_categories](https://github.com/bitrao/music-social-network-analysis/blob/main/images/wordcloud_categories.png?raw=true)
  
  ![lda_topic_distribution_across_categories](https://github.com/bitrao/music-social-network-analysis/blob/main/images/lda_topic_distribution_across_categories.png?raw=true)
  
  ![bert_topic_distribution_across_categories](https://github.com/bitrao/music-social-network-analysis/blob/main/images/bert_topic_distribution_across_categories.png?raw=true)

#### Topics:

BERTopics:
  ![bertopic_topics](https://github.com/bitrao/music-social-network-analysis/blob/main/images/bertopic_topics.png?raw=true)
LDA:
  ![lda_topics](https://github.com/bitrao/music-social-network-analysis/blob/main/images/lda_topics.png?raw=true)

### Sentiment Analysis

This section explores sentiment analysis using two distinct approaches: VADER (Valence Aware Dictionary and sEntiment Reasoner) and pre-trained RoBERTa.
  ![sentiment_distribution_comparison.png](https://github.com/bitrao/music-social-network-analysis/blob/main/images/sentiment_distribution_comparison.png?raw=true)

Both models generally concurred on the overall trend: "neutral" comments were the most frequent, followed by "positive," and then "negative." RoBERTa appeared to identify a greater number of neutral comments than VADER while VADER identified slightly more positive comments than RoBERTa The counts for negative sentiment were relatively similar between the two models. For the actual results, it can be seen that not only RoBERTa captures the sentiment of emoji much better than VADER but also due to being pre trained with millions of tweets the vocabulary of RoBERTa is more sentimentally precise. 
  ![sentiment_distribution_category](https://github.com/bitrao/music-social-network-analysis/blob/main/images/sentiment_distribution_category.png?raw=true)

Across all analyzed dimensions—individual topics, different video categories, and temporal trends—a recurring pattern emerged: neutral sentiment generally holds a significant, often leading, share, closely followed by positive sentiment. Negative sentiment, while consistently identifiable, typically constitutes the smallest proportion of comments. This broad observation suggests a largely favorable or at least non-negative engagement with the song.

### Social Network Analysis

  ![comments_reply_network](https://github.com/bitrao/music-social-network-analysis/blob/main/images/comments_reply_network.png?raw=true)
  ![video_comment_network](https://github.com/bitrao/music-social-network-analysis/blob/main/images/video_comment_network.png?raw=true)
  ![video_category_network](https://github.com/bitrao/music-social-network-analysis/blob/main/images/video_category_network.png?raw=true)

## Proposed Framework for Analyzing Music Reception
#### Data Acquisition
Gather comments from relevant music videos using APIs (e.g., YouTube Data API). Consider diversifying sources with platforms like TikTok or SoundCloud. Define "relevant videos" carefully, including official, lyric, and fan-made content.
#### Pre-processing
Clean raw text by removing punctuation, normalizing case, and handling emojis. Tokenize text into words, remove stopwords (common, uninformative words), and apply lemmatization (reducing words to their base form) for consistent analysis. Python libraries like NLTK and spaCy are essential here.
#### Sentiment Analysis
Assess the emotional tone of comments. Use pre-trained models like VADER for social media nuances or advanced Transformer-based models (e.g., BERT via Hugging Face) for higher accuracy. Fine-tuning these models on music-specific data can enhance performance.
#### Topic Modeling
Identify key themes and subjects discussed in comments. LDA (Latent Dirichlet Allocation) can uncover general topics, while BERTopic, leveraging BERT embeddings, often yields more coherent and interpretable themes without needing a predefined number of topics.
#### Network Construction
Map conversational interactions to understand influence and discussion flow. Build reply graphs where comments are nodes and replies are edges. Compute centrality metrics (e.g., degree, betweenness, eigenvector) using libraries like NetworkX to identify influential comments or users.
#### Visualization
Present findings clearly through various visuals:
Sentiment Timelines show how audience emotion evolves over time.
Word Clouds highlight frequently used words within specific themes or sentiment categories.
Network Diagrams illustrate reply structures, with nodes colored by sentiment and sized by centrality to reveal key discussion points.
Topic Distribution Charts and Sentiment Charts provide deeper insights into thematic and emotional patterns.



*Assignment 3 of COSC2049 - Social Network and Media Analysis RMIT 2025*

Full report: [drive](https://drive.google.com/file/d/1rKoBghwS8qc7QBacIwgaWPXhqz580P3P/view?usp=drive_link)

