from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from collections import Counter
from cloud_config import llmclient

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

def preprocess_text(text):
    """Cleans text by removing special characters, stopwords, and lemmatizing."""
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.lower().split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return " ".join(cleaned_words)

def extract_topics_from_text(text, max_topics=5, max_top_words=10):
    try:
        cleaned_text = preprocess_text(text)
        if len(cleaned_text.split()) < 10:
            logging.warning("Text too short for meaningful topic extraction")
            return {"raw_topics": [], "interpreted_topics": "Text too short for meaningful analysis"}
        
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_df=0.85,
            min_df=2,
            ngram_range=(1, 2),
            max_features=1000
        )
        
        sentences = nltk.sent_tokenize(text)
        if len(sentences) < 3:
            sentences = [text[i:i+100] for i in range(0, len(text), 100) if len(text[i:i+100].strip()) > 0]
        
        tfidf = vectorizer.fit_transform(sentences)
        
        if tfidf.shape[1] < 2:
            logging.warning("Not enough features extracted for NMF")
            return {"raw_topics": [], "interpreted_topics": "Not enough unique terms for topic modeling"}
        
        n_topics = min(max_topics, min(5, tfidf.shape[1]-1))
        
        nmf = NMF(
            n_components=n_topics,
            random_state=42,
            max_iter=500,
            alpha=0.1,
            l1_ratio=0.5
        )
        
        nmf_result = nmf.fit_transform(tfidf)
        feature_names = vectorizer.get_feature_names_out()
        
        raw_topics = []
        for topic_idx, topic in enumerate(nmf.components_):
            top_features_ind = topic.argsort()[:-max_top_words-1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            
            weights = topic[top_features_ind]
            weights = weights / weights.sum()
            
            weighted_terms = [f"{feature} ({weight:.2f})" for feature, weight in zip(top_features, weights)]
            raw_topics.append(", ".join(weighted_terms))
        
        interpreted_topics = interpret_topics_with_llm(text, raw_topics, llmclient)
        return raw_topics+interpreted_topics
        # return {
        #     "raw_topics": raw_topics,
        #     "interpreted_topics": interpreted_topics
        # }
    
    except Exception as e:
        logging.error(f"Error extracting topics: {e}")
        return {"raw_topics": [], "interpreted_topics": f"Error extracting topics: {str(e)}"}

def interpret_topics_with_llm(text, raw_topics, llmclient):
    try:
        prompt = f"""
        I need you to analyze the following text and the extracted topic keywords to identify 
        the main themes and topics. For each theme, provide a concise label and a brief description.
        
        Text excerpt: {text[:1000]}... (truncated for brevity)
        
        Raw extracted topics:
        {raw_topics}
        
        Please respond with:
        1. A list of 3-5 main themes you identify from the text and keywords
        2. For each theme, provide a concise label and a 1-2 sentence description
        3. List any notable subtopics or related concepts
        
        Format your response as a structured list of themes and descriptions.
        """
        
        response = llmclient.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a topic analysis expert who can identify meaningful themes and topics from text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logging.error(f"Error interpreting topics with LLM: {e}")
        return f"Error interpreting topics: {str(e)}"

def analyze_database_content(text_content, llmclient, user_prompt=None):
    """
    Analyze database content and answer user questions based on topics.
    
    Args:
        text_content (str): The text content from the database
        llmclient: The LLM client
        user_prompt (str, optional): User question to answer
        
    Returns:
        dict: Analysis results with topics and optional answer to user question
    """
    topic_results = extract_topics_from_text(text_content, llmclient)
    
    result = {
        "raw_topics": topic_results["raw_topics"],
        "thematic_analysis": topic_results["interpreted_topics"]
    }
    
    if user_prompt:
        try:
            response = llmclient.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant who answers questions based on data from the database."},
                    {"role": "user", "content": f"Answer the user question based on the following data from the database:\n\nText Content: {text_content[:1000]}... (truncated)\n\nHighlighted Topics: {topic_results['interpreted_topics']}\n\nQuestion: {user_prompt}"}
                ],
                temperature=0.5
            )
            result["answer"] = response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error answering user question: {e}")
            result["answer"] = f"Error answering question: {str(e)}"
    
    return result
