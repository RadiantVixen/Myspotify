import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split



def similar_people_listen_to(triplets_df, tracks_df, target_user_id, k=10):
    f = triplets_df[triplets_df['user_id'].isin(triplets_df['user_id'].value_counts()[lambda x: x >= 300].index)]
    f = f[f['song_id'].isin(f['song_id'].value_counts()[lambda x: x >= 300].index)]
    

    mat = f.pivot_table(index="user_id", columns="song_id", values="play_count", fill_value=0)
    train, test, p_scores, target_scores = [], {}, [], {}
    
    for uid, row in mat.iterrows():
        songs = row[row > 0].index.tolist()
        if len(songs) < 2:
            continue
        _, test_songs = train_test_split(songs, test_size=0.2, random_state=42)
        test[uid] = set(test_songs); r = row.copy(); r[list(test_songs)] = 0; train.append(r)
        
    if not train:
        raise ValueError("Training matrix is empty! Try lowering your interaction threshold (e.g., from 500 to 100) so users aren't all filtered out.")
        
    train_mat = pd.DataFrame(train, index=test.keys())
    sim = pd.DataFrame(cosine_similarity(train_mat), index=train_mat.index, columns=train_mat.index)
    
    for uid in train_mat.index:
        history = set(train_mat.loc[uid][train_mat.loc[uid] > 0].index)
        recs = {}
        for suid in sim[uid].sort_values(ascending=False).iloc[1:6].index:
            for song, cnt in train_mat.loc[suid][train_mat.loc[suid] > 0].items():
                if song not in history: recs[song] = recs.get(song, 0) + sim.loc[uid, suid] * cnt

        top = sorted(recs, key=recs.get, reverse=True)[:k]
        if uid == target_user_id:
            top_items = sorted(recs.items(), key=lambda x: x[1], reverse=True)[:k]
            target_scores = pd.DataFrame(top_items, columns=["song_id", "likelihood"])
            
        p_scores.append(len(set(top).intersection(test[uid])) / k)
        
    print(f"Average Precision @ {k}: {np.mean(p_scores)*100:.2f}%")

    res = target_scores.merge(tracks_df[["song_id", "artist", "title"]], on="song_id", how="left").reset_index(drop=True)
    res.index += 1
    return res[["artist", "title", "likelihood"]]







def similar_tracks(triplets_df, tracks_df, target_song_id, target_user_id, k=10):
    if k <= 0:
        raise ValueError("k must be positive")

    f = triplets_df[triplets_df['user_id'].isin(triplets_df['user_id'].value_counts()[lambda x: x >= 300].index)]
    f = f[f['song_id'].isin(f['song_id'].value_counts()[lambda x: x >= 300].index)]
    
    mat = f.pivot_table(index="user_id", columns="song_id", values="play_count", fill_value=0)

    train, test = [], {}
    for uid, row in mat.iterrows():
        songs = row[row > 0].index.tolist()
        if len(songs) < 2:
            continue
        _, test_songs = train_test_split(songs, test_size=0.2, random_state=42)
        test[uid] = set(test_songs)
        r = row.copy()
        r[list(test_songs)] = 0
        train.append(r)
        
    if not train:
        raise ValueError("Training matrix is empty! Try lowering your interaction threshold.")
        
    # Transpose to get item-item matrix (songs as rows, users as columns)
    track_mat = pd.DataFrame(train, index=test.keys()).T
    
    if target_song_id not in track_mat.index:
        raise ValueError(f"Target song ID '{target_song_id}' was filtered out by the threshold. Try lowering it.")

    track_sim = cosine_similarity(track_mat)
    track_sim_df = pd.DataFrame(
        track_sim, 
        index=track_mat.index, 
        columns=track_mat.index
    )

    similar_song_ids = track_sim_df[target_song_id].sort_values(ascending=False).iloc[1:k].index
    p_scores = []
    
    # Get the global similarity vector for the target song
    target_similarities = track_sim_df[target_song_id]
    
    for uid, test_songs in test.items():
        # Get this user's training history vector
        if uid not in track_mat.columns:
            continue
        user_vector = track_mat[uid]
        
        # Customize scores for this user: 
        # Combine item similarity (to target song) with the user's own play counts/history
        # (e.g., dot product or weighted sum of how much the user interacts with similar items)
        user_custom_scores = {}
        for candidate_song, sim_score in target_similarities.items():
            if candidate_song == target_song_id:
                continue
            # Weight the similarity by how much this user plays the candidate song (if at all)
            # Or multiply by user interaction strength
            user_play = user_vector.get(candidate_song, 0)
            
            # Hybrid score: Item similarity + bonus if the user already listens to it/similar items
            user_custom_scores[candidate_song] = sim_score + (0.1 * user_play)

        # Get top k personalized/customized songs for this specific user
        user_top_songs = sorted(user_custom_scores, key=user_custom_scores.get, reverse=True)[:k]
        
        # Calculate precision for this user
        score = len(set(user_top_songs).intersection(test_songs)) / k
        p_scores.append(score)
        
    print(f"Customized Average Precision @ {k}: {np.mean(p_scores)*100:.2f}%")

    # Convert Index to DataFrame before merging
    res = pd.DataFrame({'song_id': similar_song_ids})
    res = res.merge(tracks_df[["song_id", "artist", "title"]], on="song_id", how="left")
    res = res.reset_index(drop=True)
    res.index += 1
    return res[["artist", "title"]]







