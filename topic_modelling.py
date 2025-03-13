from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if not already available
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """Cleans text by removing special characters, stopwords, and lemmatizing."""
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    # Remove numbers & special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase & tokenize
    words = text.lower().split()
    
    # Remove stopwords & apply lemmatization
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)

def extract_topics_from_text(text, max_topics=3, max_top_words=10):
    try:
        # Preprocess text
        cleaned_text = preprocess_text(text)
        
        # TF-IDF Vectorization with fixed min_df and max_df
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_df=1.0,  # Include all words
            min_df=1,     # Include words appearing at least once
            max_features=1000  # Limit feature set
        )
        tfidf = vectorizer.fit_transform([cleaned_text])

        # Ensure n_topics does not exceed available features
        n_topics = min(max_topics, tfidf.shape[1] or 1)
        nmf = NMF(n_components=n_topics, random_state=42, max_iter=500)
        nmf.fit(tfidf)

        feature_names = vectorizer.get_feature_names_out()
        topics = [
            ", ".join([feature_names[i] for i in topic.argsort()[-max_top_words:][::-1]])
            for topic in nmf.components_
        ]
        return " | ".join(topics)
    
    except Exception as e:
        logging.error(f"Error extracting topics: {e}")
        return "Error extracting topics."
