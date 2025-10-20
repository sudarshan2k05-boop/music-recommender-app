# app.py - AI Music Recommender Application
import streamlit as st
import pandas as pd
import numpy as np
# --- 1. Configuration and Data Simulation ---
# Using Streamlit's cache to load data only once for efficiency
@st.cache_data
def load_music_data():
    """Loads a simulated dataset of songs and their attributes."""
    data = {
        'Song': ['Starlight', 'Adventure of a Lifetime', 'Bohemian Rhapsody', 
                 'Smooth Criminal', 'Thunderstruck', 'Back in Black', 
                 'Shape of You', 'Uptown Funk', 'Imagine', 'Happier Than Ever',
                 'Levitating', 'Watermelon Sugar'],
        'Artist': ['Muse', 'Coldplay', 'Queen', 
                   'Michael Jackson', 'AC/DC', 'AC/DC', 
                   'Ed Sheeran', 'Mark Ronson', 'John Lennon', 'Billie Eilish',
                   'Dua Lipa', 'Harry Styles'],
        'Genre': ['Rock', 'Pop', 'Rock', 
                  'Pop', 'Hard Rock', 'Hard Rock', 
                  'Pop', 'Pop', 'Ballad', 'Pop',
                  'Pop', 'Pop'],
        'Popularity_Score': [8.5, 9.1, 9.8, 
                             9.5, 9.0, 9.2, 
                             8.9, 8.8, 9.3, 8.7, 
                             9.6, 9.4]
    }
    return pd.DataFrame(data)

# Load the data once
df = load_music_data()

# --- 2. Application Layout and Title ---
st.set_page_config(page_title="Music Recommender", layout="centered")
st.title('🎶 AI-Powered Music Recommender Prototype')
st.markdown("""
    This app simulates a music recommendation engine based on user preference and song popularity.
    Select a genre and the number of songs you want to discover!
""")

# --- 3. Recommendation Logic (Simulated AI Model) ---
def get_recommendations(preferred_genre, top_n=3):
    """
    Simulates a recommendation model:
    1. Filters songs by preferred genre.
    2. Ranks them by 'Popularity_Score'.
    """
    
    # Handle the "surprise me" option
    if preferred_genre == "All Genres (Surprise Me!)":
        filtered_songs = df
    else:
        # Filter songs based on the user's preferred genre
        filtered_songs = df[df['Genre'] == preferred_genre]

    if filtered_songs.empty:
        st.warning(f"No songs found in the '{preferred_genre}' category.")
        return pd.DataFrame({'Song': ['N/A'], 'Artist': ['N/A'], 'Genre': ['N/A']})

    # Sort by Popularity_Score and return the top N
    recommendations = filtered_songs.sort_values(by='Popularity_Score', ascending=False).head(top_n)
    
    return recommendations[['Song', 'Artist', 'Genre']]

# --- 4. User Input Widgets in Sidebar ---
with st.sidebar:
    st.header("Customize Your Search")
    
    # Get unique genres for the selectbox
    genres = ['All Genres (Surprise Me!)'] + sorted(df['Genre'].unique().tolist())

    # Widget for user to select a genre
    selected_genre = st.selectbox(
        '1. Select your preferred genre:',
        genres
    )

    # Widget for user to select how many songs they want
    num_recommendations = st.slider(
        '2. How many top songs do you want to see?',
        min_value=1, max_value=5, value=3
    )
    
# --- 5. Generate and Display Recommendations ---
st.header('Recommendation Results')

if st.button(f'▶️ Get {num_recommendations} Recommendations'):
    with st.spinner(f'Searching through the catalog for top {num_recommendations} {selected_genre} songs...'):
        
        # Get the recommendations using the function
        recommendations_df = get_recommendations(selected_genre, num_recommendations)
        
        if 'No songs found' not in recommendations_df['Song'].iloc[0]:
            st.subheader(f'🎶 Top Picks in {selected_genre}:')
            
            # Apply styling to the table
            st.dataframe(
                recommendations_df.style.set_properties(**{'background-color': '#f0f2f6', 'color': 'black'}), 
                use_container_width=True,
                hide_index=True
            )
            st.balloons()
            st.success('Recommendations generated successfully! Enjoy your new playlist.')

st.markdown("---")
st.caption("Disclaimer: This is a proof-of-concept application. A real AI music recommender would use complex collaborative filtering models trained on millions of user interactions.")
