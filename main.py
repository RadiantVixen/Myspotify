import argparse
import pandas as pd
from parse import parse_mxm_dataset, parse_triplets, parse_tagtraum_genres, parse_unique_tracks
from Content_based_approach import Collections
from collaborative_filtering import similar_people_listen_to, similar_tracks
from Non_personalized_approach import Top_250_tracks, Top_100_tracks_by_genre

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Myspotify Music Recommender System")
    parser.add_argument(
        "--approach",
        type=str,
        required=True,
        choices=[
            "top250", "genre", "baseline", "word2vec", "classification",
            "collaborative-user", "collaborative-track"
        ],
        help="The recommendation approach to execute."
    )
    parser.add_argument("--genre", type=str, default="Rock", help="Genre name for genre-based top tracks.")
    parser.add_argument("--keyword", type=str, default="love", help="Keyword/vibe for content-based collections (love, war, happiness, etc.).")
    parser.add_argument("--user_id", type=str, default="001889ee41b5f31c404a1abe4af07b2377fa776b", help="Target user ID for collaborative filtering.")
    parser.add_argument("--track_id", type=str, default=None, help="Target song ID for track-based recommendations.")

    args = parser.parse_args()

    print("Loading datasets...")
    try:
        triplets_df = parse_triplets('data/train_triplets.txt')
        genres_df = parse_tagtraum_genres('data/p02_msd_tagtraum_cd2.cls')
        unique_tracks = parse_unique_tracks('data/p02_unique_tracks.txt')
        vocabulary, tracks_data = parse_mxm_dataset('data/mxm_dataset_train.txt')
    except (OSError, ValueError) as exc:
        parser.error(f"could not load the dataset: {exc}")

    if args.approach == "top250":
        print(f"\n--- Running Top 250 Tracks ---")
        result = Top_250_tracks(triplets_df, unique_tracks)
        print(result)

    elif args.approach == "genre":
        print(f"\n--- Running Top 100 Tracks for Genre: {args.genre} ---")
        result = Top_100_tracks_by_genre(triplets_df, unique_tracks, genres_df, args.genre)
        print(result)

    elif args.approach in ["baseline", "word2vec", "classification"]:
        print(f"\n--- Running Content-Based Approach: {args.approach} (Keyword: {args.keyword}) ---")
        result = Collections(
            user_id=args.user_id,
            vocabulary=vocabulary, 
            tracks_data=tracks_data, 
            triplets_df=triplets_df, 
            tracks_df=unique_tracks,
            keyword=args.keyword,
            approach=args.approach
        )
        print(result)

    elif args.approach == "collaborative-user":
        print(f"\n--- Running Collaborative Filtering for user: {args.user_id} ---")
        result = similar_people_listen_to(triplets_df, unique_tracks, args.user_id, k=10)
        print(result)

    elif args.approach == "collaborative-track":
        if not args.track_id:
            parser.error("--track_id is required for collaborative-track")
        print(f"\n--- Running Collaborative Filtering for track: {args.track_id} ---")
        result = similar_tracks(triplets_df, unique_tracks, args.track_id, k=10)
    else:
        parser.error(f"unsupported approach: {args.approach}")

    print(result)



#SOFRQTD12A81C233C0