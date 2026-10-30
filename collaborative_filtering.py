import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def _validate_inputs(triplets_df, tracks_df):
    if not {"user_id", "song_id", "play_count"}.issubset(triplets_df.columns):
        raise ValueError("triplets_df must contain user_id, song_id, and play_count")
    if not {"song_id", "artist", "title"}.issubset(tracks_df.columns):
        raise ValueError("tracks_df must contain song_id, artist, and title")


def _split_user_history(user_item_matrix, test_size=0.2):
    train_rows = []
    test_data = {}
    for user_id, row in user_item_matrix.iterrows():
        interacted_songs = row[row > 0].index.tolist()
        if len(interacted_songs) < 5:
            continue
        _, test_songs = train_test_split(
            interacted_songs, test_size=test_size, random_state=42
        )
        test_data[user_id] = set(test_songs)
        train_row = row.copy()
        train_row[test_songs] = 0
        train_rows.append(train_row)

    if not train_rows:
        return pd.DataFrame(columns=user_item_matrix.columns), {}
    return pd.DataFrame(train_rows, index=test_data.keys()), test_data


def _result_frame(song_scores, tracks_df, k):
    ranked_scores = sorted(song_scores.items(), key=lambda item: item[1], reverse=True)[:k]
    if not ranked_scores:
        return pd.DataFrame(columns=["artist", "title", "likelihood"])
    rec_df = pd.DataFrame(ranked_scores, columns=["song_id", "likelihood"])
    rec_df = rec_df.merge(
        tracks_df[["song_id", "artist", "title"]], on="song_id", how="left"
    )
    rec_df = rec_df.reset_index(drop=True)
    rec_df.index += 1
    return rec_df[["artist", "title", "likelihood"]]


def similar_people_listen_to(triplets_df, tracks_df, target_user_id, k=10):
    _validate_inputs(triplets_df, tracks_df)
    if k <= 0:
        raise ValueError("k must be positive")

    user_item_matrix = triplets_df.pivot_table(
        index="user_id", columns="song_id", values="play_count", fill_value=0
    )
    train_matrix, test_data = _split_user_history(user_item_matrix)
    if target_user_id not in train_matrix.index:
        raise ValueError(f"User {target_user_id} is not available in the training split")

    user_sim = pd.DataFrame(
        cosine_similarity(train_matrix), index=train_matrix.index, columns=train_matrix.index
    )
    precision_scores = []
    target_scores = {}

    for user_id in train_matrix.index:
        similar_users = user_sim[user_id].sort_values(ascending=False).iloc[1:6].index
        history = set(train_matrix.loc[user_id][train_matrix.loc[user_id] > 0].index)
        recommendation_scores = {}
        for similar_user in similar_users:
            similarity = user_sim.loc[user_id, similar_user]
            for song_id, play_count in train_matrix.loc[similar_user][
                train_matrix.loc[similar_user] > 0
            ].items():
                if song_id not in history:
                    recommendation_scores[song_id] = recommendation_scores.get(song_id, 0) + (
                        similarity * play_count
                    )

        if user_id == target_user_id:
            target_scores = recommendation_scores
        recommendations = sorted(
            recommendation_scores, key=recommendation_scores.get, reverse=True
        )[:k]
        precision_scores.append(
            len(set(recommendations).intersection(test_data[user_id])) / k
        )

    average_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
    print(f"Average Precision @ {k}: {average_precision * 100:.2f}%")
    return _result_frame(target_scores, tracks_df, k)


def similar_tracks(triplets_df, tracks_df, target_song_id, k=10):
    _validate_inputs(triplets_df, tracks_df)
    if k <= 0:
        raise ValueError("k must be positive")

    user_item_matrix = triplets_df.pivot_table(
        index="user_id", columns="song_id", values="play_count", fill_value=0
    )
    train_matrix, test_data = _split_user_history(user_item_matrix)
    if target_song_id not in train_matrix.columns:
        raise ValueError(f"Track {target_song_id} is not available in the training split")

    track_item_matrix = train_matrix.T
    track_sim = pd.DataFrame(
        cosine_similarity(track_item_matrix),
        index=track_item_matrix.index,
        columns=track_item_matrix.index,
    )
    similar_song_ids = track_sim[target_song_id].sort_values(ascending=False).iloc[1:k + 1]
    recommendation_scores = similar_song_ids.to_dict()

    precision_scores = []
    for user_id, held_out_songs in test_data.items():
        train_history = set(train_matrix.loc[user_id][train_matrix.loc[user_id] > 0].index)
        if target_song_id not in train_history:
            continue
        precision_scores.append(
            len(set(similar_song_ids.index).intersection(held_out_songs)) / k
        )

    average_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
    print(f"Average Precision @ {k}: {average_precision * 100:.2f}%")
    return _result_frame(recommendation_scores, tracks_df, k)
