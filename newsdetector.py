import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import string

class NewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression(max_iter=1000)
        
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        return text
    
    def train(self, X_train, y_train):
        X_train_clean = [self.preprocess_text(text) for text in X_train]
        X_train_vec = self.vectorizer.fit_transform(X_train_clean)
        self.model.fit(X_train_vec, y_train)
        print("Model training completed!")
    
    def predict(self, text):
        clean_text = self.preprocess_text(text)
        text_vec = self.vectorizer.transform([clean_text])
        prediction = self.model.predict(text_vec)[0]
        probability = self.model.predict_proba(text_vec)[0]
        
        return {
            'prediction': 'TRUE' if prediction == 1 else 'FALSE',
            'confidence': max(probability) * 100
        }
    
    def evaluate(self, X_test, y_test):
        X_test_clean = [self.preprocess_text(text) for text in X_test]
        X_test_vec = self.vectorizer.transform(X_test_clean)
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy


if __name__ == "__main__":
    sample_data = {
        'text': [
            "Scientists discover new planet in our solar system with clear evidence",
            "BREAKING: Aliens land in New York City, government confirms",
            "Study shows regular exercise improves mental health outcomes",
            "You won't believe what this celebrity did! Doctors hate this trick!",
            "Research published in Nature reveals breakthrough in cancer treatment",
            "SHOCKING: Moon is actually made of cheese, NASA admits",
            "Climate change report shows rising global temperatures",
            "Click here to get rich quick with this one weird trick",
            "New vaccine shows promising results in clinical trials",
            "Celebrity reveals secret to eternal youth - pharma companies furious!"
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    
    df = pd.DataFrame(sample_data)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    
    detector = NewsDetector()
    detector.train(X_train, y_train)
    detector.evaluate(X_test, y_test)
    
    print("\n" + "="*50)
    print("Testing with new articles:")
    print("="*50)
    
    test_articles = [
        "Breaking research from MIT shows promising results in renewable energy",
        "UNBELIEVABLE! This one trick will make you a millionaire overnight!"
    ]
    
    for article in test_articles:
        result = detector.predict(article)
        print(f"\nArticle: {article[:60]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}%")
