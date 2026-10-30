import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split



def similar_people_listen_to(triplets_df, tracks_df, target_user_id, k=10):

    print("Filtering dataset to save memory...")
    user_counts = triplets_df['user_id'].value_counts()
    active_users = user_counts[user_counts >= 500].index
    
    song_counts = triplets_df['song_id'].value_counts()
    popular_songs = song_counts[song_counts >= 500].index
    
    triplets_filtered = triplets_df[
        triplets_df['user_id'].isin(active_users) & 
        triplets_df['song_id'].isin(popular_songs)
    ]
    
    print("Building user-item matrix...")
    user_item_matrix = triplets_filtered.pivot_table(
        index='user_id', 
        columns='song_id', 
        values='play_count', 
        fill_value=0
    )
    # print(user_item_matrix.head(1))
    train_data = []
    test_data = {}
    
    for user_id, row in user_item_matrix.iterrows():
        interacted_songs = row[row > 0].index.tolist()
        if len(interacted_songs) < 5:
            continue
            
        train_songs, test_songs = train_test_split(interacted_songs, test_size=0.2, random_state=42)
        test_data[user_id] = set(test_songs)
        
        train_row = row.copy()
        train_row[test_songs] = 0
        train_data.append(train_row)
        
    train_matrix = pd.DataFrame(train_data, index=test_data.keys())
    
    if target_user_id not in train_matrix.index:
        return f"User {target_user_id} not found in training data."

    print("Computing user-user similarity matrix...")
    user_sim = cosine_similarity(train_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=train_matrix.index, columns=train_matrix.index)
    
    precision_scores = []
    
    print("Evaluating recommendations across all test users...")
    for user_id in test_data.keys():
        similar_users = user_sim_df[user_id].sort_values(ascending=False).iloc[1:6].index
        target_user_history = set(train_matrix.loc[user_id][train_matrix.loc[user_id] > 0].index)
        
        recommendation_scores = {}
        for sim_user in similar_users:
            sim_user_row = train_matrix.loc[sim_user]
            sim_score = user_sim_df.loc[user_id, sim_user]
            
            for song_id, play_cnt in sim_user_row[sim_user_row > 0].items():
                if song_id not in target_user_history:
                    recommendation_scores[song_id] = recommendation_scores.get(song_id, 0) + (sim_score * play_cnt)
                    
        sorted_recs = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        rec_song_ids = [song_id for song_id, score in sorted_recs]
        if user_id == target_user_id:
            targeted_songs = rec_song_ids
        
        actual_test_songs = test_data[user_id]
        hits = len(set(rec_song_ids).intersection(actual_test_songs))
        p_at_k = hits / k
        
        precision_scores.append(p_at_k)
        
    average_p_at_k = np.mean(precision_scores) * 100
    print(f"\n--- Evaluation Results ---")
    print(f"Total Users Evaluated: {len(precision_scores)}")
    print(f"Average Precision @ {k} (p@k): {average_p_at_k:.2f}%")
    
    if average_p_at_k >= 10.0:
        print("Success! Your model meets the >= 10% requirement.")
    else:
        print("Model score is below 10%. You may need to tune similarity metrics.")
     
    rec_df = pd.DataFrame({'song_id': targeted_songs})
    rec_df = pd.merge(rec_df, tracks_df[['song_id', 'artist', 'title']], on='song_id', how='left')
    
    rec_df = rec_df.reset_index(drop=True)
    rec_df.index += 1
    return rec_df[['artist', 'title']]




def similar_tracks(triplets_df, tracks_df, target_song_id, k=10):
    print("Filtering dataset to save memory...")
    user_counts = triplets_df['user_id'].value_counts()
    active_users = user_counts[user_counts >= 500].index
    
    song_counts = triplets_df['song_id'].value_counts()
    popular_songs = song_counts[song_counts >= 500].index
    
    triplets_filtered = triplets_df[
        triplets_df['user_id'].isin(active_users) & 
        triplets_df['song_id'].isin(popular_songs)
    ]
    
    print("Building user-item matrix...")
    user_item_matrix = triplets_filtered.pivot_table(
        index='user_id', 
        columns='song_id', 
        values='play_count', 
        fill_value=0
    )
    print("Building track-item matrix...")
    track_item_matrix = triplets_filtered.pivot_table(
        index='song_id',
        columns='user_id', 
        values='play_count', 
        fill_value=0
    )

    print("Computing track similarity matrix (Done once)...")
    track_sim = cosine_similarity(track_item_matrix)
    track_sim_df = pd.DataFrame(track_sim, index=track_item_matrix.index, columns=track_item_matrix.index)

    if target_song_id not in track_sim_df.index:
        return f"Song {target_song_id} not found in matrix."

    similar_song_ids = track_sim_df[target_song_id].sort_values(ascending=False).iloc[1:k+1].index

    precision_scores = []
    print("Evaluating precision across users...")
    
    for user_id, row in user_item_matrix.iterrows():
        user_history = set(row[row > 0].index)
        
        hits = len(set(similar_song_ids).intersection(user_history))
        p_at_k = hits / k
        precision_scores.append(p_at_k)

    avg_precision = np.mean(precision_scores) * 100
    print(f"Average Precision @ {k}: {avg_precision:.2f}%")

    rec_df = pd.DataFrame({'song_id': similar_song_ids})
    rec_df = pd.merge(rec_df, tracks_df[['song_id', 'artist', 'title']], on='song_id', how='left')

    rec_df = rec_df.reset_index(drop=True)
    rec_df.index += 1

    return rec_df[['artist', 'title']]


