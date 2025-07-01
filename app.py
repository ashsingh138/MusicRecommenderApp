import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException
import streamlit.components.v1 as components
import random
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Expert Music Recommender",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Heuristics for Regional Categorization ---
BOLLYWOOD_ARTISTS = ['Arijit Singh', 'Shreya Ghoshal', 'Lata Mangeshkar', 'Kishore Kumar', 'Pritam', 'A.R. Rahman', 'Sonu Nigam', 'Neha Kakkar']
SOUTH_INDIAN_ARTISTS = ['S. P. Balasubrahmanyam', 'K. S. Chithra', 'Sid Sriram', 'Anirudh Ravichander', 'Ilaiyaraaja', 'K. J. Yesudas']

def categorize_region(artists_str):
    if not isinstance(artists_str, str): return 'Hollywood/Other'
    if any(artist in artists_str for artist in BOLLYWOOD_ARTISTS): return 'Bollywood'
    if any(artist in artists_str for artist in SOUTH_INDIAN_ARTISTS): return 'South Indian'
    return 'Hollywood/Other'

# --- Spotify API Authentication ---
@st.cache_resource
def setup_spotify_client():
    try:
        return spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=st.secrets["SPOTIPY_CLIENT_ID"], client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"]))
    except (KeyError, SpotifyException):
        return None

sp_client = setup_spotify_client()

# --- Caching Data Loading and Processing ---
@st.cache_data
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(subset=['track_name', 'artists', 'track_genre'], inplace=True)
    df.drop_duplicates(subset=['track_name', 'artists'], keep='first', inplace=True)
    df.rename(columns={'track_genre': 'genres'}, inplace=True)
    df['popularity_normalized'] = MinMaxScaler().fit_transform(df[['popularity']])
    df['region'] = df['artists'].apply(categorize_region)
    return df.reset_index(drop=True)

@st.cache_data
def create_model(_df):
    features = ['danceability', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(_df[features])
    kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
    _df['cluster'] = kmeans.fit_predict(X_scaled)
    return X_scaled, scaler, _df, features

# --- Advanced Model & Recommendation Logic ---
@st.cache_data(show_spinner=False)
def get_track_info(_sp_client, track_name, artist_name):
    if not _sp_client: return {'id': None, 'artist_url': None}
    try:
        primary_artist = artist_name.split(';')[0].split(',')[0]
        query = f"track:{track_name} artist:{primary_artist}"
        results = _sp_client.search(q=query, type='track', limit=1)
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            artist_url = track['artists'][0]['external_urls'].get('spotify') if track['artists'] else None
            return {'id': track.get('id'), 'artist_url': artist_url}
    except Exception: pass
    return {'id': None, 'artist_url': None}

def hybrid_recommendation(song_name, df, X_scaled, scaler, features, content_weight=0.5, top_n=100):
    """Generates a hybrid score based on content similarity and collaborative signals."""
    song_row = df[df['track_name'].str.lower() == song_name.lower()].iloc[0:1]
    if song_row.empty: return None, None

    # 1. Content-Based Score (Cosine Similarity)
    features_song = scaler.transform(song_row[features])
    content_sims = cosine_similarity(features_song, X_scaled).flatten()
    
    # 2. Collaborative-Based Score (Artist & Genre Similarity)
    song_artist = song_row['artists'].iloc[0]
    song_genre = song_row['genres'].iloc[0]
    
    # Give a score boost to songs by the same artist or in the same genre
    collab_scores = np.zeros(len(df))
    collab_scores[df['artists'] == song_artist] = 0.2  # Artist boost
    collab_scores[df['genres'] == song_genre] = 0.1    # Genre boost
    collab_scores = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min() + 1e-9) # Normalize

    # 3. Combine Scores
    hybrid_score = (content_weight * content_sims) + ((1 - content_weight) * collab_scores)
    df['similarity'] = content_sims
    df['hybrid_score'] = hybrid_score
    
    recs = df.sort_values(by='hybrid_score', ascending=False)[1:top_n+1]
    return recs, song_row.iloc[0]

@st.cache_data
def calculate_inertia(X_scaled, max_k=30):
    """Calculates KMeans inertia for a range of k values."""
    inertia = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    return k_range, inertia

# --- UI Components ---
def get_mood_tags(valence, energy):
    if energy > 0.75 and valence > 0.6: return "🥳 Upbeat "
    if energy > 0.6 and valence > 0.5: return "😊 Happy "
    if valence < 0.35 and energy < 0.5: return "😢 Sad "
    if energy < 0.45: return "😌 Chill "
    return ""

def display_song_features(song_details, features):
    feature_values = song_details[features[:-1]].values
    fig = go.Figure(data=go.Scatterpolar(r=feature_values, theta=features[:-1], fill='toself', marker_color='rgba(29, 185, 84, 0.7)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, title=dict(text="Audio Feature Profile", x=0.5), height=300, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)

def embed_spotify_player(track_id):
    components.iframe(f"https://open.spotify.com/embed/track/{track_id}", height=80)

def display_recommendations(recs_df, title, start_key=0):
    st.subheader(title)
    if recs_df.empty:
        st.warning("No recommendations found. Try adjusting your filters.")
        return

    for i, row in recs_df.iterrows():
        st.markdown(f"**{row['track_name']}**")
        mood_tags = get_mood_tags(row['valence'], row['energy'])
        
        col_info, col_button = st.columns([4, 1])
        with col_info:
            track_info = row['track_info']
            if track_info and track_info['artist_url']:
                st.markdown(f"by [{row['artists']}]({track_info['artist_url']})")
            else:
                st.markdown(f"by *{row['artists']}*")
            st.caption(f"Mood: {mood_tags if mood_tags else 'Neutral'}")

        with col_button:
            if st.button("More Like This", key=f"more_{i}_{start_key}"):
                st.session_state.song_input = row['track_name']
                st.rerun()
        
        st.progress(row['similarity'], text=f"Audio Similarity: {row['similarity']:.0%}")
        
        if track_info and track_info['id']:
            embed_spotify_player(track_info['id'])
        else:
            st.info("Could not find this track on Spotify to embed.")
        st.divider()

# --- Main Application UI ---
st.title("🤖 Expert Music Recommender")
st.markdown("A hybrid recommender using content-based and collaborative filtering signals.")

if 'song_input' not in st.session_state: st.session_state.song_input = ""

if not sp_client:
    st.error("Spotify API credentials not set up correctly. Please check `secrets.toml`.")
else:
    try:
        df_original = load_and_prepare_data('dataset.csv')
        X_scaled, scaler, df_model, features = create_model(df_original)

        with st.sidebar:
            st.header("⚙️ Settings & Controls")
            
            with st.expander("Model Details & Evaluation"):
                st.markdown("### KMeans Clustering Evaluation")
                st.markdown("The 'Elbow Method' helps find the optimal number of clusters (`k`). The 'elbow' of the curve suggests a good value for `k`.")
                k_range, inertia = calculate_inertia(X_scaled)
                elbow_fig = go.Figure(data=go.Scatter(x=list(k_range), y=inertia, mode='lines+markers'))
                elbow_fig.update_layout(title='Elbow Method for Optimal k', xaxis_title='Number of Clusters (k)', yaxis_title='Inertia')
                st.plotly_chart(elbow_fig, use_container_width=True)

            st.header("🔍 Filters")
            region_filter = st.radio("Region:", ["All", "Bollywood", "Hollywood/Other", "South Indian"])
            
            filtered_df = df_model[df_model['region'] == region_filter] if region_filter != "All" else df_model
            song_options = sorted(filtered_df['track_name'].unique())
            
            try: current_song_index = song_options.index(st.session_state.song_input)
            except ValueError: current_song_index = None

            def update_song(): st.session_state.song_input = st.session_state.song_selector
            st.selectbox("Choose a song:", options=song_options, index=current_song_index, key="song_selector", on_change=update_song)
            
            if st.button("Surprise Me!"):
                if not filtered_df.empty:
                    st.session_state.song_input = random.choice(song_options)
                    st.rerun()

            st.header("🧠 Recommendation Engine")
            content_weight = st.slider("Content vs. Collab Filtering", 0.0, 1.0, 0.7, 0.05, help="1.0 = Purely audio similarity. 0.0 = Purely artist/genre similarity.")
            
            st.header("🎧 Audio Feature Filters")
            dance_range = st.slider("Danceability", 0.0, 1.0, (0.0, 1.0), 0.05)
            energy_range = st.slider("Energy", 0.0, 1.0, (0.0, 1.0), 0.05)
            
        if st.session_state.song_input:
            recs_pool, selected_song = hybrid_recommendation(st.session_state.song_input, df_model, X_scaled, scaler, features, content_weight)
            
            if recs_pool is not None:
                def apply_filters(df): return df[(df['danceability'].between(*dance_range)) & (df['energy'].between(*energy_range))]

                primary_recs = apply_filters(recs_pool[recs_pool['region'] == region_filter] if region_filter != "All" else recs_pool).head(10)
                other_recs = apply_filters(recs_pool[recs_pool['region'] != region_filter]).head(5) if region_filter != "All" else pd.DataFrame()

                for df_slice in [primary_recs, other_recs]:
                    if not df_slice.empty:
                        df_slice['track_info'] = df_slice.apply(lambda row: get_track_info(sp_client, row['track_name'], row['artists']), axis=1)

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader(f"Analysis of '{st.session_state.song_input}'")
                    st.metric("Artist(s)", selected_song['artists'])
                    if selected_song.get('genres'): st.markdown(f"**Genre:** {selected_song['genres']}")
                    display_song_features(selected_song, features)

                with col2:
                    display_recommendations(primary_recs, "Your Recommendations")
                    if not other_recs.empty:
                        st.header("", divider="rainbow")
                        display_recommendations(other_recs, "From Other Regions", start_key=100)
            else:
                st.error(f"Could not find '{st.session_state.song_input}' in the dataset.")
        else:
            st.info("Choose a song or click 'Surprise Me!' to get started.")

    except FileNotFoundError:
        st.error("Error: `dataset.csv` not found. Please ensure it is in the same directory.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

