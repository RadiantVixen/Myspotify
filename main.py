from parse import parse_mxm_dataset, parse_triplets, parse_tagtraum_genres, parse_unique_tracks
from recommender_system import Top_250_tracks, Top_100_tracks_by_genre, Collections
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)  # Prevents wrapping to a new line



if __name__ == "__main__":

    # Example usage:
    triplets_df = parse_triplets('data/train_triplets.txt')
    # print(f"Total interaction triplets loaded: {len(triplets_df)}")
    # print(triplets_df.head())

    # # Example usage:
    genres_df = parse_tagtraum_genres('data/p02_msd_tagtraum_cd2.cls')
    # print(f"Total genre annotations loaded: {len(genres_df)}")
    # print(genres_df.head(10))


    # Example usage:
    unique_tracks = parse_unique_tracks('data/p02_unique_tracks.txt')
    # print(f"Total genre annotations loaded: {len(unique_tracks)}")
    # print(unique_tracks.head(10))



    # Example usage:
    vocabulary, tracks_data  = parse_mxm_dataset('data/mxm_dataset_train.txt')
    # print(f"Total tracks loaded: {len(tracks_data)}")
    # print(f"First track data: {tracks_data[0]}")



    # top_250 =  Top_250_tracks(triplets_df, unique_tracks)
    # print(top_250.head(1))

    # top_100 = Top_100_tracks_by_genre(triplets_df, unique_tracks, genres_df, "Rock")
    # print(top_100)

    top_50 = Collections("b80344d063b5ccb3212f76538f3d9e43d87dca9e", vocabulary, tracks_data, triplets_df, unique_tracks, "love")
    print(top_50)


