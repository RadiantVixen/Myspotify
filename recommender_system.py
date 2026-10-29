import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model    import LogisticRegression
import nltk
from gensim.models import Word2Vec
import pandas as pd
# from nltk.corpus import stopwords
# nltk.download('stopwords')


def Top_250_tracks(triplets_df, tracks_df):
    # 1. Group by song_id and sum up total play counts
    popular_songs = triplets_df.groupby('song_id')['play_count'].sum().reset_index()
    print(len(popular_songs), len (tracks_df))
    
    # 2. Merge with track metadata (artist & title)
    merged_df = pd.merge(popular_songs, tracks_df, on='song_id')
    
    # 3. Sort descendingly by play count and take the top 250[cite: 8]
    top_250 = merged_df.sort_values(by='play_count', ascending=False).head(250)
    
    # 4. Reset index so it starts at 1 (index number)[cite: 8]
    top_250 = top_250.reset_index(drop=True)
    top_250.index += 1
    
    # 5. Return the exact fields requested: artist name, track title, play count[cite: 8]
    return top_250[['artist', 'title', 'play_count']]


def  Top_100_tracks_by_genre(triplets_df, tracks_df, genres_df, genre):

    popular_songs = triplets_df.groupby('song_id')['play_count'].sum().reset_index()
    merged_df = pd.merge(popular_songs, tracks_df, on='song_id')
    genres_df = genres_df[genres_df["majority_genre"] == genre]
    merged_df = pd.merge(merged_df, genres_df, on='track_id')

    top_100 = merged_df.sort_values(by='play_count', ascending=False).head(100)


    top_100 = top_100.reset_index(drop=True)
    top_100.index += 1

    return top_100[['artist', 'title', 'play_count']]




def classify_track_vibes(vocabulary, tracks_data):
    # 1. Map target vibes to their keyword baskets using safe lookups
    theme_keywords = {
        'love': ['love', 'heart', 'kiss', 'sweet'],
        'war': ['war', 'fight', 'blood', 'dead', 'death'],
        'happiness': ['happi', 'smile', 'joy', 'heaven'],
        'loneliness': ['loneli', 'alon', 'tear', 'dark'],
        'money': ['money', 'gold', 'cash', 'dollar']
    }
    
    # 2. Safely find indices for words that actually exist in the vocabulary
    theme_indices = {}
    for theme, words in theme_keywords.items():
        indices = []
        for w in words:
            if w in vocabulary:
                # +1 because word indices in mxm dataset start at 1
                indices.append(vocabulary.index(w) + 1)
        theme_indices[theme] = indices

    classified_tracks = []
    
    for track in tracks_data:
        word_counts = track.get('word_counts', {})
        
        # Calculate total score for each theme
        theme_scores = {}
        for theme, indices in theme_indices.items():
            score = sum(word_counts.get(idx, 0) for idx in indices)
            theme_scores[theme] = score
            
        # Assign dominant vibe, or fall back to neutral only if absolutely no keywords matched
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
        
    # Count neutral tracks safely from the list
    # neutral_count = sum(1 for t in classified_tracks if t['dominant_vibe'] == 'neutral')
    # percentage = (neutral_count / len(classified_tracks)) * 100 if len(classified_tracks) > 0 else 0
    # print("percentage:", percentage)
    
    return pd.DataFrame(classified_tracks)



def vibe_check(tracks_data, vocabulary):
    # 1. Build "sentences" for Word2Vec by reconstructing words from each track's word_counts
    sentences = []
    target_vibes = ["love", "war", "happiness", "loneliness", "money"]
    
    for track in tracks_data:
        sentence = []
        for idx, count in track.get("word_counts", {}).items():
            # word index in file is 1-based, list is 0-based
            if 0 < idx <= len(vocabulary):
                word = vocabulary[idx - 1]
                # Repeat the word based on its count to represent frequency in the song
                sentence.extend([word] * min(count, 5)) # cap at 5 to avoid overpowering
        if sentence:
            sentences.append(sentence)

    # 2. Add target vibes to sentences or ensure they exist
    sentences.append(target_vibes)

    # 3. Train Word2Vec on the song-level context windows
    print("Training Word2Vec on track contexts...")
    model1 = Word2Vec(sentences, min_count=1, vector_size=50, window=5, sg=1)
    print("Word2Vec trained successfully!")

    classified_tracks = []
    
    # 4. Classify each track using the trained model similarities
    for track in tracks_data:
        vibes_count = {vibe: 0.0 for vibe in target_vibes}
        word_counts = track.get("word_counts", {})
        
        for idx, count in word_counts.items():
            if 0 < idx <= len(vocabulary):
                word = vocabulary[idx - 1]
                if word in model1.wv:
                    for vibe in target_vibes:
                        if vibe in model1.wv:
                            # Compute semantic similarity * word frequency
                            sim = model1.wv.similarity(vibe, word)
                            if sim > 0: # Only add positive similarities
                                vibes_count[vibe] += sim * count
        
        # Find dominant vibe, default to neutral if all scores are 0
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


# def classification_algorithm(tracks_data, vocabulary):
#     # X = features, y = target
#     X_train, X_test, y_train, y_test = train_test_split(
#         tracks_data, , test_size=0.2, random_state=42, stratify=y
#     )
#     model = LogisticRegression(
#         max_iter=1000, C=1.0, solver='lbfgs', random_state=42
#     ),
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)



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
    classifier = LogisticRegression(max_iter=500, C=1.0)
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

def Collections(user_id, vocabulary, tracks_data, triplets_df, tracks_df, vibe):

    # classified_tracks = classify_track_vibes(vocabulary, tracks_data)
    # classified_tracks = vibe_check(tracks_data, vocabulary)
    classified_tracks = classification_task_pipeline(tracks_data, vocabulary)
    classified_tracks = classified_tracks[classified_tracks['dominant_vibe'] == vibe]
    # classified_tracks = triplets_df.groupby('song_id')['play_count'].sum().reset_index()
    merged_df = pd.merge(classified_tracks, tracks_df, on='track_id')
    merged_df = pd.merge(merged_df, triplets_df, on='song_id')
    user_tracks = merged_df[merged_df['user_id'] == user_id]
    top_50 = merged.sort_values(by='play_count', ascending=False).drop_duplicates(subset=['title']).head(50)
    top_50 = top_50.reset_index(drop=True)
    top_50.index += 1

    return top_50[['artist', 'title', 'play_count']]


