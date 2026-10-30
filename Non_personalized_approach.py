import pandas as pd


def Top_250_tracks(triplets_df, tracks_df):
    popular_songs = triplets_df.groupby('song_id')['play_count'].sum().reset_index()
    print(len(popular_songs), len (tracks_df))
    
    merged_df = pd.merge(popular_songs, tracks_df, on='song_id')
    
    top_250 = merged_df.sort_values(by='play_count', ascending=False).head(250)
    
    top_250 = top_250.reset_index(drop=True)
    top_250.index += 1
    
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

