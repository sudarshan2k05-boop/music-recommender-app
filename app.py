# app.py - AI Music Recommender Application
import streamlit as st
import pandas as pd
import numpy as np
# --- 1. Configuration and Data Simulation ---
# Using Streamlit's cache to load data only once for efficiency
@st.cache_data
def load_music_data():
    """Loads a simulated dataset with expanded Tamil and English Movie Songs 
    covering classic to modern eras, with corresponding YouTube playback links."""
    data = {
        'Song': [
            # English Movie Songs (Classic, 90s, Modern)
            'Singin\' in the Rain (1952)', 'The Sound of Silence (1967)', 
            'A Whole New World (Aladdin 1992)', 'My Heart Will Go On (Titanic 1997)',
            'Happy (Despicable Me 2013)', 'Shallow (A Star Is Born 2018)', 
            'Bohemian Rhapsody (Film 2018)', 'No Time To Die (Bond 2021)',
            
            # Tamil Movie Songs (Classic, 90s, Modern)
            'Chinna Machan Sella Machan (Pudhu Nellu Pudhu Naadhu 1991)', 
            'Chinna Chinna Aasai (Roja 1992)', 'Kadhal Rojave (Roja 1992)',
            'Vaseegara (Minnale 2001)', 'Vaathi Coming (Master 2021)', 
            'Naatu Naatu (RRR Tamil Dub 2022)', 'Nee Partha Vizhigal (Pudhiya Mugam 1993)',
            'Enna Solla Pogirai (Kandukondain Kandukondain 2000)'
        ],
        'Artist': [
            'Gene Kelly', 'Simon & Garfunkel', 
            'Brad Kane & Lea Salonga', 'Celine Dion', 
            'Pharrell Williams', 'Lady Gaga & Bradley Cooper', 
            'Queen', 'Billie Eilish',
            
            'S. P. Balasubrahmanyam, S. Janaki', 
            'Minmini', 'S. P. Balasubrahmanyam',
            'Hariharan', 'Anirudh Ravichander', 
            'Rahul Sipligunj & Kaala Bhairava', 'Unni Krishnan',
            'Shankar Mahadevan'
        ],
        # Categorized by Language/Era
        'Language_Genre': [
            'English/Classic', 'English/Classic', 
            'English/90s', 'English/90s', 
            'English/2010s', 'English/2010s', 
            'English/2010s', 'English/Modern',
            
            'Tamil/90s', 'Tamil/90s', 'Tamil/90s', 
            'Tamil/2000s', 'Tamil/Modern', 
            'Tamil/Modern', 'Tamil/90s',
            'Tamil/2000s'
        ],
        'Popularity_Score': [
            8.5, 8.8, 
            9.0, 9.9, 
            9.4, 9.6, 
            9.8, 9.1,
            
            8.7, 9.2, 9.5, 
            9.1, 9.7, 
            9.6, 8.9, 
            9.0
        ],
        # YouTube links for playback
        'Listen': [
            'https://www.youtube.com/watch?v=D1ZYhVpgXbQ', # Singin' in the Rain
            'https://www.youtube.com/watch?v=4fK1c1950eM', # The Sound of Silence
            'https://www.youtube.com/watch?v=hG9E9X94T3Q', # A Whole New World
            'https://www.youtube.com/watch?v=FHG2FD4Ww7o', # My Heart Will Go On
            'https://www.youtube.com/watch?v=ZbZSe6N_BXs', # Happy
            'https://www.youtube.com/watch?v=bo_efYhYU2A', # Shallow
            'https://www.youtube.com/watch?v=fJ9rUzIMcZQ', # Bohemian Rhapsody
            'https://www.youtube.com/watch?v=GBwLw6gX7c8', # No Time To Die
            
            'https://www.youtube.com/watch?v=2r1p09Xb8oM', # Chinna Machan Sella Machan
            'https://www.youtube.com/watch?v=vVjV-7P_F3o', # Chinna Chinna Aasai
            'https://www.youtube.com/watch?v=sS9d-8_tLwQ', # Kadhal Rojave
            'https://www.youtube.com/watch?v=H6Uo4748NqQ', # Vaseegara
            'https://www.youtube.com/watch?v=d_kS_j_P5Lw', # Vaathi Coming
            'https://www.youtube.com/watch?v=3R-9tIuXh0I', # Naatu Naatu
            'https://www.youtube.com/watch?v=0kH8s4zPj6o', # Nee Partha Vizhigal
            'https://www.youtube.com/watch?v=S012sW3D51M'  # Enna Solla Pogirai
        ]
    }
    return pd.DataFrame(data)

# Load the data once
df = load_music_data()

# --- 2. Application Layout and Title ---
st.set_page_config(page_title="Movie Song Recommender", layout="centered")
st.title('🎬 Tamil & English Movie Song Recommender')
st.markdown("""
    This app recommends popular Tamil and English movie songs covering classic to modern eras.
    Select an option from the sidebar:
    1. **Combined List:** Choose the 'Top Tamil & English Mix' option.
    2. **Separate List:** Choose a specific genre like 'Tamil/90s' or 'English/Classic'.
    
    **Click the 'Listen' column link to play the song separately on YouTube.**
""")

# --- 3. Recommendation Logic (Simulated AI Model) ---
def get_recommendations(preferred_genre, top_n=3):
    """
    Simulates a recommendation model:
    1. Filters songs by preferred Language_Genre.
    2. Ranks them by 'Popularity_Score'.
    """
    
    # Handle the combined mix option
    if preferred_genre == "Top Tamil & English Mix (Combined)":
        filtered_songs = df
    else:
        # Filter songs based on the user's preferred genre
        filtered_songs = df[df['Language_Genre'] == preferred_genre]

    if filtered_songs.empty:
        st.warning(f"No songs found in the '{preferred_genre}' category.")
        # Include 'Listen' column in the empty DataFrame
        return pd.DataFrame({'Song': ['N/A'], 'Artist': ['N/A'], 'Language_Genre': ['N/A'], 'Listen': ['']})

    # Sort by Popularity_Score and return the top N
    recommendations = filtered_songs.sort_values(by='Popularity_Score', ascending=False).head(top_n)
    
    # Return all necessary columns, including 'Listen'
    return recommendations[['Song', 'Artist', 'Language_Genre', 'Listen']]

# --- 4. User Input Widgets in Sidebar ---
with st.sidebar:
    st.header("Customize Your Search")
    
    # Get unique genres/languages for the selectbox. Updated label for combined mix.
    # Note: Genres now include the era (e.g., 'Tamil/90s', 'English/Classic')
    genres = ['Top Tamil & English Mix (Combined)'] + sorted(df['Language_Genre'].unique().tolist())

    # Widget for user to select a genre/language
    selected_genre = st.selectbox(
        '1. Select your preferred Language/Era:',
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
    # Determine the heading text
    heading_text = "Top Combined Mix" if selected_genre == "Top Tamil & English Mix (Combined)" else selected_genre

    with st.spinner(f'Searching through the catalog for top {num_recommendations} songs...'):
        
        # Get the recommendations using the function
        recommendations_df = get_recommendations(selected_genre, num_recommendations)
        
        if 'No songs found' not in recommendations_df['Song'].iloc[0]:
            st.subheader(f'🎶 Top Picks in {heading_text}:')
            
            # Apply styling and display the dataframe with clickable links
            st.dataframe(
                recommendations_df.style.set_properties(**{'background-color': '#f0f2f6', 'color': 'black'}), 
                use_container_width=True,
                hide_index=True,
                column_config={"Listen": st.column_config.LinkColumn("Listen", display_text="▶️ Play")}
            )
            
            st.markdown(f"""
                <div style="font-size: small; margin-top: 10px; color: gray;">
                    Tip: Click the '▶️ Play' link in the table to open the music in a new tab.
                </div>
            """, unsafe_allow_html=True)
            
            st.balloons()
            st.success('Recommendations generated successfully! Enjoy your new playlist.')

st.markdown("---")
st.caption("Disclaimer: This is a proof-of-concept application.")
