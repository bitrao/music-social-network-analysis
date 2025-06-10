# YouTube Music Comments Analysis Report: "Luther" by Kendrick Lamar

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
<div style="display: flex; justify-content: center;">
  <div style="flex: 1; text-align: center;">
    <img src="https://github.com/bitrao/music-social-network-analysis/blob/main/images/bertopic_topics.png?raw=true" alt="bertopic_topics" style="max-width: 100%;"/>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="https://github.com/bitrao/music-social-network-analysis/blob/main/images/lda_topics.png?raw=true" alt="lda_topics" style="max-width: 100%;"/>
</div>
#### Topics:
<div style="display: flex; justify-content: center;">
  <div style="flex: 1; text-align: center;">
    <img src="https://github.com/bitrao/music-social-network-analysis/blob/main/images/wordcloud_categories.png?raw=true" alt="wordcloud_categories" style="max-width: 100%;"/>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="https://github.com/bitrao/music-social-network-analysis/blob/main/images/lda_topic_distribution_across_categories.png?raw=true" alt="lda_topic_distribution_across_categories" style="max-width: 100%;"/>
  </div>
    <div style="flex: 1; text-align: center;">
    <img src="https://github.com/bitrao/music-social-network-analysis/blob/main/images/bert_topic_distribution_across_categories.png?raw=true" alt="bert_topic_distribution_across_categories" style="max-width: 100%;"/>
  </div>
</div>

### Sentiment Analysis
This section explores sentiment analysis using two distinct approaches: VADER (Valence Aware Dictionary and sEntiment Reasoner) and pre-trained RoBERTa.
   <img src="https://github.com/bitrao/music-social-network-analysis/blob/main/images/sentiment_distribution_comparison.png?raw=true" alt="sentiment_distribution_comparison"/>
Both models generally concurred on the overall trend: "neutral" comments were the most frequent, followed by "positive," and then "negative." RoBERTa appeared to identify a greater number of neutral comments than VADER while VADER identified slightly more positive comments than RoBERTa The counts for negative sentiment were relatively similar between the two models. For the actual results, it can be seen that not only RoBERTa captures the sentiment of emoji much better than VADER but also due to being pre trained with millions of tweets the vocabulary of RoBERTa is more sentimentally precise. 
   <img src="https://github.com/bitrao/music-social-network-analysis/blob/main/images/sentiment_distribution_category.png?raw=true" alt="sentiment_distribution_category"/>
Across all analyzed dimensions—individual topics, different video categories, and temporal trends—a recurring pattern emerged: neutral sentiment generally holds a significant, often leading, share, closely followed by positive sentiment. Negative sentiment, while consistently identifiable, typically constitutes the smallest proportion of comments. This broad observation suggests a largely favorable or at least non-negative engagement with the song.

### Social Network Analysis
<div style="display: flex; justify-content: center;">
  <div style="flex: 1; text-align: center;">
    <img src="https://github.com/bitrao/music-social-network-analysis/blob/main/images/comments_reply_network.png?raw=true" alt="comments_reply_network" style="max-width: 100%;"/>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="https://github.com/bitrao/music-social-network-analysis/blob/main/images/video_comment_network.png?raw=true" alt="video_comment_network" style="max-width: 100%;"/>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="https://github.com/bitrao/music-social-network-analysis/blob/main/images/video_category_network.png?raw=true" alt="video_category_network" style="max-width: 100%;"/>
  </div>
</div>
## Future Work
- Enhanced sentiment analysis
- Real-time comment monitoring
- Automated response suggestions
- Advanced topic modeling
- Multi-language support

**Assignment 3 of COSC2049 - Social Network and Media Analysis RMIT 2025**
Full report: 

