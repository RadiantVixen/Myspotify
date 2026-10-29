import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


def classification_task_pipeline(tracks_data, vocabulary):
    target_vibes = ["love", "war", "happiness", "loneliness", "money"]
    
    # 1. Define high-confidence keyword baskets for pseudo-labeling
    theme_keywords = {
        'love': ['love', 'heart', 'kiss', 'sweet'],
        'war': ['war', 'fight', 'blood', 'dead', 'death'],
        'happiness': ['happi', 'smile', 'joy', 'heaven'],
        'loneliness': ['loneli', 'alon', 'tear', 'dark'],
        'money': ['money', 'gold', 'cash', 'dollar']
    }

    theme_indices = {}
    for theme, words in theme_keywords.items():
        theme_indices[theme] = [vocabulary.index(w) + 1 for w in words if w in vocabulary]

    X_train_dicts = []
    y_train_labels = []
    unlabeled_indices = []

    # 2. Automatically label confident tracks (Weak Supervision)
    for idx, track in enumerate(tracks_data):
        word_counts = track.get("word_counts", {})
        
        scores = {
            theme: sum(word_counts.get(w_idx, 0) for w_idx in w_indices)
            for theme, w_indices in theme_indices.items()
        }
        
        max_score = max(scores.values()) if scores else 0
        
        # If a track has a strong keyword signal, use it as a training label
        if max_score >= 3:
            best_theme = max(scores, key=scores.get)
            X_train_dicts.append(word_counts)
            y_train_labels.append(best_theme)
        else:
            # Otherwise, leave it for the model to predict
            unlabeled_indices.append(idx)

    print(f"Pseudo-labeled training set: {len(y_train_labels)} tracks")
    print(f"Unlabeled set to predict: {len(unlabeled_indices)} tracks")

    # 3. Vectorize sparse word counts into a feature matrix
    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(X_train_dicts)
    y_train = np.array(y_train_labels)

    # 4. Train a Machine Learning Classifier
    print("Training Logistic Regression classifier...")
    classifier = LogisticRegression(max_iter=500, C=1.0, multi_class='multinomial')
    classifier.fit(X_train, y_train)
    print("Model trained successfully!")

    # 5. Predict vibes for the rest of the dataset
    if unlabeled_indices:
        X_unlabeled_dicts = [tracks_data[i].get("word_counts", {}) for i in unlabeled_indices]
        X_unlabeled = vectorizer.transform(X_unlabeled_dicts)
        
        predicted_labels = classifier.predict(X_unlabeled)
        
        # Assign predictions back to the dataset
        for i, pred in zip(unlabeled_indices, predicted_labels):
            tracks_data[i]['dominant_vibe'] = pred

    # Also assign the training labels back
    train_counter = 0
    for i, track in enumerate(tracks_data):
        word_counts = track.get("word_counts", {})
        scores = {theme: sum(word_counts.get(w_idx, 0) for w_idx in w_indices) for theme, w_indices in theme_indices.items()}
        if max(scores.values() if scores else [0]) >= 3:
            track['dominant_vibe'] = y_train_labels[train_counter]
            train_counter += 1

    # Convert final results to DataFrame
    final_df = pd.DataFrame([{
        'track_id': t['track_id'],
        'mxm_track_id': t['mxm_track_id'],
        'dominant_vibe': t.get('dominant_vibe', 'neutral')
    } for t in tracks_data])

    return final_df