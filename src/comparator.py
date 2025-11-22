import librosa
import numpy as np
from typing import List, Dict, Any
import scipy.spatial.distance

def extract_features(file_path: str) -> Dict[str, np.ndarray]:
    """
    Extracts summary features for comparison.
    """
    try:
        y, sr = librosa.load(file_path, duration=30) # Analyze first 30s for speed
        
        # MFCCs (Timbre)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Chroma (Harmony/Key)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Spectral Contrast (Production/Texture)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        
        return {
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
            'chroma_mean': chroma_mean,
            'contrast_mean': contrast_mean
        }
    except Exception as e:
        print(f"Error extracting features for {file_path}: {e}")
        return None

def compare_album(file_paths: List[str]) -> Dict[str, Any]:
    """
    Analyzes a list of files for consistency.
    Returns a dictionary with similarity matrix and outlier scores.
    """
    print(f"Comparing {len(file_paths)} tracks for album consistency...")
    
    features_list = []
    valid_files = []
    
    for fp in file_paths:
        feats = extract_features(fp)
        if feats:
            features_list.append(feats)
            valid_files.append(fp)
            
    if len(valid_files) < 2:
        return {'error': "Not enough valid files for comparison"}
        
    # 1. MFCC Similarity (Timbral Consistency)
    # Stack MFCC means into a matrix (N_tracks x N_features)
    mfcc_matrix = np.array([f['mfcc_mean'] for f in features_list])
    
    # Calculate pairwise cosine distance
    # 1 - cosine_distance gives similarity (0 to 1)
    dist_matrix = scipy.spatial.distance.pdist(mfcc_matrix, metric='cosine')
    sim_matrix = 1 - scipy.spatial.distance.squareform(dist_matrix)
    
    # Average similarity per track (how similar is this track to the rest?)
    # We exclude the diagonal (self-similarity is always 1)
    np.fill_diagonal(sim_matrix, np.nan)
    avg_similarity = np.nanmean(sim_matrix, axis=1)
    
    # Identify outliers
    # If a track has very low average similarity compared to the group mean
    group_mean_sim = np.nanmean(sim_matrix)
    outliers = []
    
    for i, sim in enumerate(avg_similarity):
        if sim < group_mean_sim - 0.1: # Threshold for outlier
            outliers.append({
                'file': valid_files[i],
                'similarity_score': float(sim)
            })
            
    return {
        'similarity_matrix': sim_matrix.tolist(), # For visualization if needed
        'group_mean_similarity': float(group_mean_sim),
        'outliers': outliers,
        'track_scores': dict(zip(valid_files, avg_similarity.tolist()))
    }
