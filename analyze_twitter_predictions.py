"""
Analyzes Twitter prediction results from a CSV file.

Usage:
    python analyze_twitter_predictions.py 
        --input_file predictions.csv 
        --text_column text 
        --prediction_column prediction 
        --top_keywords 15
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
import numpy as np
import textstat

#list of common words to exclude from keyword analysis
STOPWORDS = set([
    'the', 'a', 'an', 'is', 'it', 'to', 'and', 'or', 'of', 'in', 'on', 'for', 'with',
    'at', 'by', 'this', 'that', 'we', 'i', 'you', 'they', 'he', 'she', 'not', 
    'but', 'from', 'so', 'can', 'will', 'was', 'are', 'just', 'like', 'get', 
    'have', 'know', 'think', 'out', 'up', 'down', 'about', 'go', 'us', 'say', 
    'see', 'make', 'would', 'could', 'one', 'use', 'many', 'much', 'time', 'people', 
    'said', 'new', 'year', 'day', 'don', 't', 's', 'm', 're', 'has', 'were', 'his', 'her', 
    'being', 'over', 'amp', 'https', 'tco', 'co', 'retweet', 'rt', 'what', 'when', 'where', 'who',
    'why', 'all', 'any', 'if', 'my', 'your', 'their', 'there', 'here', 'how', 'also', 'than', 'then', 'after',
    'its', 'because', 'while', 'such', 'no', 'yes', 'very', 'more', 'most', 'some', 'these', 'those',
    'been', 'says', 'did', 'do', 'does', 'doing', 'each', 'few', 'further', 'own', 'same', 'too',
    'shows', 'going', 'back', 'still', 'even', 'many', 'much', 'before', 'between', 'under', 'again', 'off',
    'down', 'right', 'left', 'last', 'first', 'better','dont', 'won', 'should', 'now', 'today', 'tomorrow',
    'them', 'him', 'she', 'our', 'us', 'me', 'll', 've', 'd', 'into', 'need', 'wants', 'want', 'only', 'thats'
])

def preprocess(text):
    """
    preprocessing: lowercasing, removing non-alphabetic characters, tokenizing, and removing stopwords.

    :param text: Text string to preprocess
    """
    if not isinstance(text, str):
        return []
    text = re.sub(r'[^a-z\s]', '', text.lower())
    tokens = text.split()
    return [word for word in tokens if word not in STOPWORDS and len(word) >= 3]

def get_metric_counts(text):
    """
    Counts various text metrics, such as number of hashtags, mentions, URLs
    
    :param text: Text to count metrics from
    """

    if not isinstance(text, str) or not text:
        return 0, 0, 0, 0

    hashtags = len(re.findall(r'#\w+', text))
    mentions = len(re.findall(r'@\w+', text))
    urls = len(re.findall(r'http\S+|www\.\S+', text))

    capital_letters = sum(1 for c in text if c.isupper())
    total_letters = sum(1 for c in text if c.isalpha())
    capitalization_ratio = (capital_letters / total_letters) if total_letters > 0 else 0

    return hashtags, mentions, urls, capitalization_ratio

def get_keyword_counts(df_group, text_column='text'):
    """
    Get keyword counts from a DataFrame group.
    
    :param df_group: DataFrame group to analyze
    :param text_column: Name of the column containing text data
    """
    all_words = []
    df_group[text_column].apply(lambda x: all_words.extend(preprocess(x)))
    return Counter(all_words)

def calculate_readability(text):
    """
    Calculate the Flesch-Kincaid Grade Level for the given text.

    :param text: Text string to calculate readability for
    """
    # Texts must have at least a few words for meaningful calculation
    if not isinstance(text, str) or len(text.split()) < 5:
        return 0.0
    try:
        # Returns the US grade level required to understand the text
        return textstat.flesch_kincaid_grade(text)
    except:
        return 0.0

def analyze_predictions(input_file, text_column, prediction_column, top_keywords):
    """
    Analyzes Twitter prediction results from a CSV file.
    
    :param input_file: Path to the CSV file containing predictions
    :param text_column: Name of the column containing tweet texts
    :param prediction_column: Name of the column containing model predictions
    :param top_keywords: Number of top keywords to display per class
    """
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File not found at {input_file}")
        print("Please ensure INPUT_FILE path is correct.")
        return

    df.dropna(subset=[text_column, prediction_column], inplace=True)
    df[prediction_column] = df[prediction_column].astype(int)

    print(f"Loaded {len(df)} records for analysis.")
    
    all_labels = sorted(df[prediction_column].unique().tolist())

    all_labels = sorted(df[prediction_column].unique().tolist())

    df['text_length'] = df[text_column].apply(len)
    
    df['readability_grade'] = df[text_column].apply(calculate_readability)

    df[['hashtag_count', 'mention_count', 'url_count', 'capitalization_ratio']] = df[text_column].apply(get_metric_counts).apply(pd.Series)
    
    avg_metrics_df = df.groupby(prediction_column).agg({
        'text_length': 'mean',
        'readability_grade': 'mean',
        'hashtag_count': 'mean',
        'mention_count': 'mean',
        'url_count': 'mean',
        'capitalization_ratio': 'mean'
    }).reset_index()
    
    avg_metrics_df.columns = [
        'Prediction Class', 
        'Average Text Length (Chars)',
        'Average Readability (Grade Level)',
        'Average Hashtags per Post',
        'Average Mentions per Post',
        'Average URLs per Post',
        'Average Capitalization Ratio'
    ]

    print("Average Text Metrics by Prediction Class")
   
    print(avg_metrics_df.to_markdown(index=False, floatfmt=(".0f", ".0f", ".2f")))
    
    plt.figure(figsize=(9, 5)) 
    plt.bar(avg_metrics_df['Prediction Class'], avg_metrics_df['Average Text Length (Chars)'], 
             color=["#E33636", '#DD8452', '#55A868'][:len(all_labels)])
    plt.title('Average Text Length by Prediction Class') 
    plt.xlabel('Prediction Class')
    plt.xticks(ticks=all_labels, labels="False Misinformed True".split()[:len(all_labels)])
    plt.ylabel('Average Text Length (Characters)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(9, 5))
    plt.bar(avg_metrics_df['Prediction Class'], avg_metrics_df['Average Readability (Grade Level)'], 
             color=["#E33636", '#DD8452', '#55A868'][:len(all_labels)])
    plt.title('Average Readability (Flesch-Kincaid Grade Level)')
    plt.xlabel('Prediction Class')
    plt.xticks(ticks=all_labels, labels="False Misinformed True".split()[:len(all_labels)])
    plt.ylabel('Average Grade Level')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.bar(avg_metrics_df['Prediction Class'], avg_metrics_df['Average Hashtags per Post'], 
             color=["#E33636", '#DD8452', '#55A868'][:len(all_labels)])
    plt.title('Average Hashtags per Post by Prediction Class')
    plt.xlabel('Prediction Class')
    plt.xticks(ticks=all_labels, labels="False Misinformed True".split()[:len(all_labels)])
    plt.ylabel('Average Hashtags per Post')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.bar(avg_metrics_df['Prediction Class'], avg_metrics_df['Average Mentions per Post'], 
             color=["#E33636", '#DD8452', '#55A868'][:len(all_labels)])
    plt.title('Average Mentions per Post by Prediction Class')
    plt.xlabel('Prediction Class')
    plt.xticks(ticks=all_labels, labels="False Misinformed True".split()[:len(all_labels)])
    plt.ylabel('Average Mentions per Post')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.bar(avg_metrics_df['Prediction Class'], avg_metrics_df['Average URLs per Post'],
                color=["#E33636", '#DD8452', '#55A868'][:len(all_labels)])
    plt.title('Average URLs per Post by Prediction Class')
    plt.xlabel('Prediction Class')
    plt.xticks(ticks=all_labels, labels="False Misinformed True".split()[:len(all_labels)])
    plt.ylabel('Average URLs per Post')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.bar(avg_metrics_df['Prediction Class'], avg_metrics_df['Average Capitalization Ratio'],
                color=["#E33636", '#DD8452', '#55A868'][:len(all_labels)])
    plt.title('Average Capitalization Ratio by Prediction Class')
    plt.xlabel('Prediction Class')
    plt.xticks(ticks=all_labels, labels="False Misinformed True".split()[:len(all_labels)])
    plt.ylabel('Average Capitalization Ratio')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    
    #engagement analysis
    ENGAGEMENT_COLUMNS = ['views', 'likes', 'retweets', 'replies'] 
    for col in ENGAGEMENT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    avg_engagement_df = df.groupby(prediction_column).agg({
        'views': 'mean',
        'likes': 'mean',
        'retweets': 'mean',
        'replies': 'mean'
    }).reset_index()

    avg_engagement_df.columns = [
        'Prediction Class', 
        'Average Views',
        'Average Likes',
        'Average Retweets',
        'Average Replies'
    ]

    print("Average Engagement Metrics by Prediction Class")
    print(avg_engagement_df.to_markdown(index=False, floatfmt=".1f"))

    metrics_to_plot = ['Average Views', 'Average Likes', 'Average Retweets', 'Average Replies']
    titles = ['Views', 'Likes', 'Retweets', 'Replies']
    class_labels = "False Misinformed True".split()[:len(all_labels)]
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        ax.bar(
            avg_engagement_df['Prediction Class'], 
            avg_engagement_df[metric], 
            color=colors[i] 
        )
        
        ax.set_title(f'Average {titles[i]} by Prediction Class', fontsize=14)
        ax.set_xlabel('Prediction Class', fontsize=12)
        ax.set_ylabel('Average Count', fontsize=12)
        
        ax.set_xticks(ticks=all_labels)
        ax.set_xticklabels(class_labels, fontsize=10) 
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.suptitle('Engagement Metrics by Prediction Class', fontsize=18, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.9]) 
    plt.show()

    #keyword analysis
    print(f"Top {top_keywords} Keywords by Prediction Class (Min Freq: {2})")

    fig, axes = plt.subplots(len(all_labels), 1, figsize=(10, 5 * len(all_labels)))
    fig.suptitle(f'Top {top_keywords} Keywords by Prediction Class', fontsize=16, y=0.95)

    if not isinstance(axes, np.ndarray): 
        axes_list = [axes]
    else:
        axes_list = axes.flatten() 

    for i, label in enumerate(all_labels):
        group_df = df[df[prediction_column] == label]
        counts = get_keyword_counts(group_df, text_column=text_column)
        
        filtered_counts = Counter({
            word: count for word, count in counts.items() 
            if count >= 2
        })
        
        most_common = filtered_counts.most_common(top_keywords)

        keyword_table = pd.DataFrame(most_common, columns=['Keyword', 'Count'])

        class_name = ""
        if label == 0:
            class_name = "Predicted False"
        elif label == 1:
            class_name = "Predicted Misinformed"
        else:
            class_name = "Predicted True"

        print(f"\n{class_name} Keywords:")
        print(keyword_table.to_markdown(index=False))

        keywords = keyword_table['Keyword'][::-1]
        counts = keyword_table['Count'][::-1]
        
        ax = axes_list[i] 
        ax.barh(keywords, counts, color='darkorange')
        ax.set_title(f'{class_name} (N={len(group_df)} posts)')
        ax.set_xlabel('Frequency Count')
        ax.tick_params(axis='y', labelsize=10)

    plt.subplots_adjust(top=0.85, hspace=0.4)
    plt.show()    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--prediction_column", type=str, default="prediction")
    parser.add_argument("--top_keywords", type=int, default=15)
    args = parser.parse_args()

    analyze_predictions(args.input_file, args.text_column, args.prediction_column, args.top_keywords)