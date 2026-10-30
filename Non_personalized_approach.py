import pandas as pd


def _require_columns(dataframe, required_columns, name):
    missing_columns = set(required_columns) - set(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{name} is missing required columns: {missing}")


def Top_250_tracks(triplets_df, tracks_df):
    _require_columns(triplets_df, ["song_id", "play_count"], "triplets_df")
    _require_columns(tracks_df, ["song_id", "artist", "title"], "tracks_df")

    popular_songs = triplets_df.groupby('song_id')['play_count'].sum().reset_index()
    merged_df = pd.merge(popular_songs, tracks_df, on='song_id', how='inner')
    top_250 = merged_df.sort_values(by='play_count', ascending=False).head(250)
    top_250 = top_250.reset_index(drop=True)
    top_250.index += 1

    return top_250[['artist', 'title', 'play_count']]


def Top_100_tracks_by_genre(triplets_df, tracks_df, genres_df, genre):
    if not isinstance(genre, str) or not genre.strip():
        raise ValueError("genre must be a non-empty string")
    _require_columns(triplets_df, ["song_id", "play_count"], "triplets_df")
    _require_columns(tracks_df, ["track_id", "song_id", "artist", "title"], "tracks_df")
    _require_columns(genres_df, ["track_id", "majority_genre"], "genres_df")

    popular_songs = triplets_df.groupby('song_id')['play_count'].sum().reset_index()
    merged_df = pd.merge(popular_songs, tracks_df, on='song_id', how='inner')
    genre_tracks = genres_df[genres_df["majority_genre"].str.casefold() == genre.strip().casefold()]
    merged_df = pd.merge(merged_df, genre_tracks, on='track_id', how='inner')
    top_100 = merged_df.sort_values(by='play_count', ascending=False).head(100)
    top_100 = top_100.reset_index(drop=True)
    top_100.index += 1

    return top_100[['artist', 'title', 'play_count']]

