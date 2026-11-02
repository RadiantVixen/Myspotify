import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
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


def _sparse_train_split(triplets_df, target_song_id, max_songs=5000):
    interactions = triplets_df.groupby(["user_id", "song_id"], as_index=False)["play_count"].sum()
    active_users = interactions.groupby("user_id")["song_id"].nunique()
    active_users = active_users[active_users >= 5].index
    interactions = interactions[interactions["user_id"].isin(active_users)]

    popular_songs = interactions.groupby("song_id")["user_id"].nunique()
    popular_songs = popular_songs.nlargest(max_songs).index
    selected_songs = set(popular_songs)
    selected_songs.add(target_song_id)
    interactions = interactions[interactions["song_id"].isin(selected_songs)]

    user_ids = interactions["user_id"].drop_duplicates().tolist()
    song_ids = interactions["song_id"].drop_duplicates().tolist()
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    song_index = {song_id: index for index, song_id in enumerate(song_ids)}
    test_data = {}
    train_rows = []

    for user_id, user_rows in interactions.groupby("user_id"):
        songs = user_rows["song_id"].tolist()
        if len(songs) < 5:
            continue
        _, test_songs = train_test_split(songs, test_size=0.2, random_state=42)
        test_data[user_id] = set(test_songs)
        for row in user_rows.itertuples(index=False):
            if row.song_id not in test_data[user_id]:
                train_rows.append((user_index[user_id], song_index[row.song_id], row.play_count))

    if not train_rows or target_song_id not in song_index:
        return csr_matrix((0, 0)), [], [], {}

    rows, columns, values = zip(*train_rows)
    matrix = csr_matrix((values, (rows, columns)), shape=(len(user_ids), len(song_ids)))
    return matrix, user_ids, song_ids, test_data


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

    train_matrix, user_ids, song_ids, test_data = _sparse_train_split(
        triplets_df, target_song_id
    )
    if train_matrix.shape[0] == 0 or target_song_id not in song_ids:
        raise ValueError(f"Track {target_song_id} is not available in the training split")

    track_item_matrix = train_matrix.T.tocsr()
    target_index = song_ids.index(target_song_id)
    similarities = cosine_similarity(track_item_matrix[target_index], track_item_matrix).ravel()
    ranked_indices = np.argsort(similarities)[::-1]
    ranked_indices = [index for index in ranked_indices if index != target_index][:k]
    recommendation_scores = {
        song_ids[index]: float(similarities[index]) for index in ranked_indices
    }

    precision_scores = []
    for user_id, held_out_songs in test_data.items():
        user_index = user_ids.index(user_id)
        train_history = {
            song_ids[index] for index in train_matrix[user_index].indices
        }
        if target_song_id not in train_history:
            continue
        precision_scores.append(
            len(set(recommendation_scores).intersection(held_out_songs)) / k
        )

    average_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
    print(f"Average Precision @ {k}: {average_precision * 100:.2f}%")
    return _result_frame(recommendation_scores, tracks_df, k)
