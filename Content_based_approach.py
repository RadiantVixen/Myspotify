import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def baseline(tracks_data ,vocabulary):
    theme_keywords = {
        'love': ['love', 'heart', 'kiss', 'sweet'],
        'war': ['war', 'fight', 'blood', 'dead', 'death'],
        'happiness': ['happi', 'smile', 'joy', 'heaven'],
        'loneliness': ['loneli', 'alon', 'tear', 'dark'],
        'money': ['money', 'gold', 'cash', 'dollar']
    }
    
    theme_indices = {}
    for theme, words in theme_keywords.items():
        indices = []
        for w in words:
            if w in vocabulary:
                indices.append(vocabulary.index(w) + 1)
        theme_indices[theme] = indices

    classified_tracks = []
    
    for track in tracks_data:
        word_counts = track.get('word_counts', {})
        
        theme_scores = {}
        for theme, indices in theme_indices.items():
            score = sum(word_counts.get(idx, 0) for idx in indices)
            theme_scores[theme] = score
            
        max_score = max(theme_scores.values())
        if max_score > 0:
            dominant_vibe = max(theme_scores, key=theme_scores.get)
        else:
            dominant_vibe = "neutral"
            
        clean_track = {
            'track_id': track['track_id'],
            'mxm_track_id': track['mxm_track_id'],
            'dominant_vibe': dominant_vibe
        }
        classified_tracks.append(clean_track)
        
    
    return pd.DataFrame(classified_tracks)



def ft_word2vec(tracks_data, vocabulary):
    try:
        from gensim.models import Word2Vec
    except ImportError as exc:
        raise RuntimeError(
            "The word2vec approach requires a gensim/SciPy-compatible installation."
        ) from exc

    sentences = []
    target_vibes = ["love", "war", "happiness", "loneliness", "money"]
    
    for track in tracks_data:
        sentence = []
        for idx, count in track.get("word_counts", {}).items():
            if 0 < idx <= len(vocabulary):
                word = vocabulary[idx - 1]
                sentence.extend([word] * min(count, 5))
        if sentence:
            sentences.append(sentence)

    sentences.append(target_vibes)

    print("Training Word2Vec on track contexts...")
    model1 = Word2Vec(sentences, min_count=1, vector_size=50, window=5, sg=1)
    print("Word2Vec trained successfully!")

    classified_tracks = []
    
    for track in tracks_data:
        vibes_count = {vibe: 0.0 for vibe in target_vibes}
        word_counts = track.get("word_counts", {})
        
        for idx, count in word_counts.items():
            if 0 < idx <= len(vocabulary):
                word = vocabulary[idx - 1]
                if word in model1.wv:
                    for vibe in target_vibes:
                        if vibe in model1.wv:
                            sim = model1.wv.similarity(vibe, word)
                            if sim > 0:
                                vibes_count[vibe] += sim * count
        
        if max(vibes_count.values()) > 0:
            maxvibe = max(vibes_count, key=vibes_count.get)
        else:
            maxvibe = "neutral"
            
        classified_tracks.append({
            "track_id": track['track_id'], 
            "mxm_track_id": track["mxm_track_id"], 
            "dominant_vibe": maxvibe
        })
    
    return pd.DataFrame(classified_tracks)



def classification_task_pipeline(tracks_data, vocabulary):
    target_vibes = ["love", "war", "happiness", "loneliness", "money"]
    
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

    for idx, track in enumerate(tracks_data):
        word_counts = track.get("word_counts", {})
        
        scores = {
            theme: sum(word_counts.get(w_idx, 0) for w_idx in w_indices)
            for theme, w_indices in theme_indices.items()
        }
        
        max_score = max(scores.values()) if scores else 0
        
        if max_score >= 3:
            best_theme = max(scores, key=scores.get)
            X_train_dicts.append(word_counts)
            y_train_labels.append(best_theme)
        else:
            unlabeled_indices.append(idx)

    print(f"Pseudo-labeled training set: {len(y_train_labels)} tracks")
    print(f"Unlabeled set to predict: {len(unlabeled_indices)} tracks")

    if not X_train_dicts or len(set(y_train_labels)) < 2:
        raise ValueError("Classification requires at least two labelled themes")

    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(X_train_dicts)
    y_train = np.array(y_train_labels)

    print("Training Logistic Regression classifier...")
    classifier = LogisticRegression(max_iter=500, C=1.0)
    classifier.fit(X_train, y_train)
    print("Model trained successfully!")

    if unlabeled_indices:
        X_unlabeled_dicts = [tracks_data[i].get("word_counts", {}) for i in unlabeled_indices]
        X_unlabeled = vectorizer.transform(X_unlabeled_dicts)
        
        predicted_labels = classifier.predict(X_unlabeled)
        
        for i, pred in zip(unlabeled_indices, predicted_labels):
            tracks_data[i]['dominant_vibe'] = pred

    train_counter = 0
    for i, track in enumerate(tracks_data):
        word_counts = track.get("word_counts", {})
        scores = {theme: sum(word_counts.get(w_idx, 0) for w_idx in w_indices) for theme, w_indices in theme_indices.items()}
        if max(scores.values() if scores else [0]) >= 3:
            track['dominant_vibe'] = y_train_labels[train_counter]
            train_counter += 1

    final_df = pd.DataFrame([{
        'track_id': t['track_id'],
        'mxm_track_id': t['mxm_track_id'],
        'dominant_vibe': t.get('dominant_vibe', 'neutral')
    } for t in tracks_data])

    return final_df


def Collections(user_id, vocabulary, tracks_data, triplets_df, tracks_df,
                vibe=None, approach="baseline", keyword=None):
    selected_vibe = keyword if keyword is not None else vibe
    if not isinstance(selected_vibe, str) or not selected_vibe.strip():
        raise ValueError("keyword must be a non-empty string")
    if approach not in {"baseline", "word2vec", "classification"}:
        raise ValueError(f"Unknown content-based approach: {approach}")
    required_triplet_columns = {"song_id", "play_count"}
    required_track_columns = {"track_id", "song_id", "artist", "title"}
    if not required_triplet_columns.issubset(triplets_df.columns):
        raise ValueError("triplets_df must contain song_id and play_count")
    if not required_track_columns.issubset(tracks_df.columns):
        raise ValueError("tracks_df must contain track_id, song_id, artist, and title")

    if approach == "baseline":
        classified_tracks = baseline(tracks_data, vocabulary)
    elif approach == "word2vec":
        classified_tracks = ft_word2vec(tracks_data, vocabulary)
    else:
        classified_tracks = classification_task_pipeline(tracks_data, vocabulary)

    classified_tracks = classified_tracks[
        classified_tracks['dominant_vibe'].str.casefold() == selected_vibe.strip().casefold()
    ]
    merged_df = pd.merge(classified_tracks, tracks_df, on='track_id', how='inner')
    merged_df = pd.merge(merged_df, triplets_df, on='song_id', how='inner')
    top_50 = (
        merged_df.sort_values(by='play_count', ascending=False)
        .drop_duplicates(subset=['title'])
        .head(50)
    )
    top_50 = top_50.reset_index(drop=True)
    top_50.index += 1

    return top_50[['artist', 'title', 'play_count']]