# app.py - AI Music Recommender Application

import streamlit as st
import pandas as pd
import numpy as np

# --- 1. Configuration and Data Simulation ---
# Using Streamlit's cache to load data only once for efficiency
@st.cache_data
def load_music_data():
    """
    Loads a simulated dataset with expanded Tamil and English Movie Songs 
    covering classic to the latest hits (2025), with corresponding YouTube playback links.
    """
    data = {
        'Song': [
            # English Movie Songs (Classic)
            'Singin\' in the Rain (1952)', 'Moon River (Breakfast at Tiffany\'s 1961)',
            'The Sound of Silence (The Graduate 1967)', 
            # English Movie Songs (90s-2010s)
            'A Whole New World (Aladdin 1992)', 'My Heart Will Go On (Titanic 1997)',
            'Happy (Despicable Me 2013)', 'Shallow (A Star Is Born 2018)', 
            
            # **LATEST ENGLISH SONGS (2023-2025) - AUGMENTED**
            'Speed Drive (Barbie 2023)', 'Flowers (Miley Cyrus - Movie Playlist)', 
            'Woke Up in Love (2022/2023 Pop Hit)', 'Bones (Imagine Dragons - Movie Playlist)',
            'I\'m Good (Blue) (2023 Pop Hit)',
            # New 2025 English Songs (Simulated hits from anticipated movies)
            'Way of Water Theme (Avatar 3 2025)', 
            'Impossible Mission (MI 8 2025)',
            'Someday My Prince Will Come (Snow White 2025)',
            'The Eternal City (Gladiator 2 2025)',

            # Tamil Movie Songs (Classic/90s)
            'Senthoora Poove (16 Vayathinile 1977)', 'Chinna Chinna Aasai (Roja 1992)',
            'Kadhal Rojave (Roja 1992)', 'Nee Partha Vizhigal (Pudhiya Mugam 1993)',
            
            # Tamil Movie Songs (2000s-2020)
            'Vaseegara (Minnale 2001)', 'Enna Solla Pogirai (Kandukondain Kandukondain 2000)',
            'Vaathi Coming (Master 2021)', 'Naatu Naatu (RRR Tamil Dub 2022)', 

            # **LATEST TAMIL SONGS (2023-2025) - AUGMENTED**
            'Manasilaayo (Vettaiyan 2025)', 'Oorum Blood (Dude 2025)',
            'Naa Ready (Leo 2023)', 'Arabic Kuthu (Beast 2022)',
            'Katchi Sera (Think Indie 2024)', 'Dippam Dappam (KRKK 2022)',
            'Tum Tum (Enemy 2021)',
            # New 2025 Tamil Songs (Simulated hits from anticipated movies)
            'Vaa Thalaivaa (Thalaivar 171 2025)', 
            'Aayiram Kili (Indian 2 2025)',
            'Kadhal Mazhai (Vidaamuyarchi 2025)',
            'Vetri Nadai (Captain Miller 2 2025)' # Example sequel song
        ],
        'Artist': [
            'Gene Kelly', 'Audrey Hepburn', 'Simon & Garfunkel', 
            'Brad Kane & Lea Salonga', 'Celine Dion', 
            'Pharrell Williams', 'Lady Gaga & Bradley Cooper', 
            
            # English Artists
            'Charli XCX', 'Miley Cyrus', 
            'Kygo, Gryffin, Calum Scott', 'Imagine Dragons',
            'David Guetta & Bebe Rexha',
            # New 2025 English Artists
            'Simon Franglen', 'Lorne Balfe', 
            'Rachel Zegler', 'Harry Gregson-Williams',

            # Tamil Artists
            'S. Janaki', 'Minmini', 
            'S. P. Balasubrahmanyam', 'Unni Krishnan',
            'Hariharan', 'Shankar Mahadevan',
            'Anirudh Ravichander', 'Rahul Sipligunj & Kaala Bhairava', 

            # New Tamil Artists
            'Anirudh Ravichander & Co.', 'Sai Abhyankkar & Paal Dabba',
            'Anirudh Ravichander & Vijay', 'Anirudh & Jonita Gandhi',
            'Sai Abhyankkar', 'Anirudh Ravichander & Anthony Daasan',
            'S. Thaman & Srivardhini',
            # New 2025 Tamil Artists
            'Anirudh Ravichander', 'A.R. Rahman & Shreya Ghoshal', 
            'Yuvan Shankar Raja', 'G.V. Prakash Kumar' 
        ],
        # Categorized by Language/Era
        'Language_Genre': [
            'English/Classic', 'English/Classic', 'English/Classic', 
            'English/90s', 'English/90s', 
            'English/2010s', 'English/2010s', 
            
            'English/Modern-2025', 'English/Modern-2025',
            'English/Modern-2025', 'English/Modern-2025',
            'English/Modern-2025',
            # New 2025 English Genres
            'English/Modern-2025', 'English/Modern-2025',
            'English/Modern-2025', 'English/Modern-2025',

            'Tamil/Classic', 'Tamil/90s', 
            'Tamil/90s', 'Tamil/90s',
            
            'Tamil/2000s', 'Tamil/2000s',
            'Tamil/2020s', 'Tamil/2020s', 

            'Tamil/Modern-2025', 'Tamil/Modern-2025',
            'Tamil/Modern-2025', 'Tamil/Modern-2025',
            'Tamil/Modern-2025', 'Tamil/Modern-2025',
            'Tamil/Modern-2025',
            # New 2025 Tamil Genres
            'Tamil/Modern-2025', 'Tamil/Modern-2025', 
            'Tamil/Modern-2025', 'Tamil/Modern-2025'
        ],
        'Popularity_Score': [
            8.5, 8.7, 8.8,
            9.0, 9.9, 
            9.4, 9.6, 
            
            9.7, 9.8,
            9.5, 9.4,
            9.6,
            # New 2025 English Popularity
            9.6, 9.7, 9.5, 9.4,

            8.4, 9.2, 
            9.5, 8.9, 
            
            9.1, 9.0,
            9.7, 9.6, 

            9.9, 9.6, 
            9.7, 9.5,
            9.4, 9.3,
            9.2,
            # New 2025 Tamil Popularity
            9.8, 9.5, 9.6, 9.3
        ],
        # YouTube links for playback
        'Listen': [
            'https://www.youtube.com/watch?v=D1ZYhVpgXbQ', # Singin' in the Rain
            'https://www.youtube.com/watch?v=Q7o6w8JbT8U', # Moon River
            'https://www.youtube.com/watch?v=4fK1c1950eM', # The Sound of Silence
            
            'https://www.youtube.com/watch?v=hG9E9X94T3Q', # A Whole New World
            'https://www.youtube.com/watch?v=FHG2FD4Ww7o', # My Heart Will Go On
            'https://www.youtube.com/watch?v=ZbZSe6N_BXs', # Happy
            'https://www.youtube.com/watch?v=bo_efYhYU2A', # Shallow
            
            'https://www.youtube.com/watch?v=BwE97H_xOTo', # Speed Drive (Barbie)
            'https://www.youtube.com/watch?v=G7KNmw9l7e4', # Flowers
            'https://www.youtube.com/watch?v=1F3b1v8i2Kk', # Woke Up in Love
            'https://www.youtube.com/watch?v=D6Fv6X0e_6Y', # Bones
            'https://www.youtube.com/watch?v=Vl3rP86xRk0', # I'm Good (Blue)
            # New 2025 English Links
            'https://www.youtube.com/watch?v=new-en-A3', 
            'https://www.youtube.com/watch?v=new-en-MI8',
            'https://www.youtube.com/watch?v=new-en-SW',
            'https://www.youtube.com/watch?v=new-en-G2',
            
            'https://www.youtube.com/watch?v=W-Lz01mGq4Q', # Senthoora Poove
            'https://www.youtube.com/watch?v=vVjV-7P_F3o', # Chinna Chinna Aasai
            'https://www.youtube.com/watch?v=sS9d-8_tLwQ', # Kadhal Rojave
            'https://www.youtube.com/watch?v=0kH8s4zPj6o', # Nee Partha Vizhigal
            
            'https://www.youtube.com/watch?v=H6Uo4748NqQ', # Vaseegara
            'https://www.youtube.com/watch?v=S012sW3D51M',  # Enna Solla Pogirai
            'https://www.youtube.com/watch?v=d_kS_j_P5Lw', # Vaathi Coming
            'https://www.youtube.com/watch?v=3R-9tIuXh0I', # Naatu Naatu
            
            'https://www.youtube.com/watch?v=5WsUIeNAtbM', # Manasilaayo (Vettaiyan 2025)
            'https://www.youtube.com/watch?v=4Bsc2uI_LsM', # Oorum Blood (Dude 2025)
            'https://www.youtube.com/watch?v=42zC2G-jXwE', # Naa Ready (Leo 2023)
            'https://www.youtube.com/watch?v=e_n0G0B1J_M', # Arabic Kuthu (Beast 2022)
            'https://www.youtube.com/watch?v=cM35vXkX0-g', # Katchi Sera (Indie 2024)
            'https://www.youtube.com/watch?v=gT5-W-2l_sY', # Dippam Dappam (KRKK 2022)
            'https://www.youtube.com/watch?v=b0wX3Y-hW0E',  # Tum Tum (Enemy 2021)
            # New 2025 Tamil Links
            'https://www.youtube.com/watch?v=new-tamil-T171',
            'https://www.youtube.com/watch?v=new-tamil-I2',
            'https://www.youtube.com/watch?v=new-tamil-VM',
            'https://www.youtube.com/watch?v=new-tamil-CM2'
        ]
    }
    return pd.DataFrame(data)

# Load the data once
df = load_music_data()

# --- 2. Application Layout and Title ---
st.set_page_config(page_title="Movie Song Recommender", layout="centered")
st.title('🎬 Tamil & English Movie Song Recommender (2025 Updated)')
st.markdown("""
    This app recommends popular Tamil and English movie songs covering classic eras right up to the newly released hits of 2025.
    Select an option from the sidebar:
    1. **Combined List:** Choose the 'Top Tamil & English Mix' option.
    2. **Separate List:** Choose a specific genre like **'Tamil/Modern-2025'** or **'English/Modern-2025'** to see the latest hits.
    
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
        st.warning(f"No songs found in the '{preferred_genre}' category. Try the combined list.")
        # Return an empty-like DataFrame to maintain column structure
        return pd.DataFrame({'Song': ['N/A'], 'Artist': ['N/A'], 'Language_Genre': ['N/A'], 'Listen': ['']})

    # Sort by Popularity_Score and return the top N
    # We now take the minimum of the requested top_n and the actual number of songs available
    max_available = len(filtered_songs)
    n = min(top_n, max_available)

    recommendations = filtered_songs.sort_values(by='Popularity_Score', ascending=False).head(n)
    
    # Return all necessary columns, including 'Listen'
    return recommendations[['Song', 'Artist', 'Language_Genre', 'Listen']]

# --- 4. User Input Widgets in Sidebar ---
with st.sidebar:
    st.header("Customize Your Search")
    
    # Get unique genres/languages for the selectbox. Updated label for combined mix.
    # Note: The new modern categories will now appear in the list.
    genres = ['Top Tamil & English Mix (Combined)'] + sorted(df['Language_Genre'].unique().tolist())

    # Widget for user to select a genre/language
    selected_genre = st.selectbox(
        '1. Select your preferred Language/Era:',
        genres
    )

    # Widget for user to select how many songs they want
    num_recommendations = st.slider(
        '2. How many top songs do you want to see?',
        min_value=1, 
        max_value=1000000, # MAX VALUE UPDATED to 1000000
        value=10      # Default value set to 10
    )
    
# --- 5. Generate and Display Recommendations ---
st.header('Recommendation Results')

if st.button(f'▶️ Get {num_recommendations} Recommendations'):
    # Determine the heading text
    heading_text = "Top Combined Mix" if selected_genre == "Top Tamil & English Mix (Combined)" else selected_genre

    with st.spinner(f'Searching through the catalog for top {num_recommendations} songs...'):
        
        # Get the recommendations using the function
        recommendations_df = get_recommendations(selected_genre, num_recommendations)
        
        if 'N/A' not in recommendations_df['Song'].iloc[0]:
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
            st.success(f'Found {len(recommendations_df)} recommendations successfully! Enjoy your updated playlist.')
        else:
             st.error("Could not find any recommendations for this selection.")


st.markdown("---")
st.caption(f"Total songs in catalog: {len(df)}")
st.caption("Disclaimer: This is a proof-of-concept application.")
