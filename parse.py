import pandas as pd
import numpy as np


def parse_triplets(file_path):
    # Define column names based on the project instructions
    col_names = ['user_id', 'song_id', 'play_count']
    
    # Load the tab-delimited file without headers
    triplets_df = pd.read_csv(
        file_path,
        sep='\t',
        header=None,
        names=col_names,
        encoding='utf-8'
    )
    return triplets_df




def parse_tagtraum_genres(file_path):
    genre_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip comment lines or empty lines
            if not line or line.startswith('#'):
                continue
                
            # Split fields by tab
            fields = line.split('\t')
            
            track_id = fields[0]
            majority_genre = fields[1] if len(fields) > 1 else None
            minority_genre = fields[2] if len(fields) > 2 else None
            
            genre_data.append({
                'track_id': track_id,
                'majority_genre': majority_genre,
                'minority_genre': minority_genre
            })
            
    return pd.DataFrame(genre_data)



def parse_unique_tracks(file_path):
    # Load the file using the custom <SEP> separator and python engine
    tracks_df = pd.read_csv(
        file_path,
        sep='<SEP>',
        header=None,
        names=['track_id', 'song_id', 'artist', 'title'],
        engine='python',
        encoding='utf-8',
        on_bad_lines='skip' # Skips any malformed rows safely if they occur
    )
    return tracks_df




def parse_mxm_dataset(file_path):
    vocabulary = []
    tracks_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines or comment lines
            if not line or line.startswith('#'):
                continue
                
            # The line starting with '%' contains the list of words
            if line.startswith('%'):
                # Remove the '%' character and split by comma
                vocabulary = line[1:].split(',')
                continue
                
            # Normal data lines: track_id, mxm_track_id, word_idx:count, ...
            parts = line.split(',')
            track_id = parts[0]
            mxm_track_id = parts[1]
            
            # Parse the sparse word counts into a dictionary {word_index: count}
            word_counts = {}
            for item in parts[2:]:
                if ':' in item:
                    idx_str, count_str = item.split(':')
                    word_counts[int(idx_str)] = int(count_str)
                    
            tracks_data.append({
                'track_id': track_id,
                'mxm_track_id': mxm_track_id,
                'word_counts': word_counts
            })
    return vocabulary, tracks_data

