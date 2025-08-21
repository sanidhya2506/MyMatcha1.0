import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import re
import nltk
# Removed Dialogflow imports and configuration

# Ensure NLTK 'punkt' tokenizer data is available
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

# --- Data Loading and Preprocessing ---
@st.cache_data # Cache the data loading and preprocessing for efficiency
def load_and_preprocess_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1-bHukazIbC7jwghhwUuQO50rtEqGA27JgLfl5jK03jk/export?format=csv"
    df = pd.read_csv(sheet_url)

    # Fill missing values (addressing FutureWarning by not using inplace=True directly on series)
    df['Episodes'] = df['Episodes'].fillna(12)
    df['Duration'] = df['Duration'].fillna("45 min")
    df['Rating'] = df['Rating'].fillna("5.7/10 (MDL)") # Using a placeholder string
    df['Release Date'] = df['Release Date'].fillna("2021-01-01")

    # Drop duplicates and reset index
    df.drop_duplicates(subset='Title', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Replace hyphens in Themes and create Tags column
    df['Themes'] = df['Themes'].apply(
        lambda x: x.replace('-', ' ') if isinstance(x, str) else x
    )
    df['Tags'] = df.apply(
        lambda row: f"{row['Summary']} {row['Genres']} {row['Themes']}".lower(),
        axis=1
    )

    # Stemming function
    ps = PorterStemmer()
    def stem(text):
      y = []
      for i in text.split():
        y.append(ps.stem(i))
      return " ".join(y)

    # Apply stemming to Tags
    newdf = df[['Title', 'Country', 'Rating', 'Tags', 'Poster_URL']].copy()
    newdf['Tags'] = newdf['Tags'].apply(stem)

    # Vectorization and Similarity
    cv = CountVectorizer(max_features=len(newdf), stop_words='english')
    vectors = cv.fit_transform(newdf['Tags']).toarray()
    similarity = cosine_similarity(vectors)

    return newdf, similarity, df # Return original df for rating display

# Load data and model components
newdf, similarity, original_df = load_and_preprocess_data()


# --- Recommendation Logic ---
def recommend(drama_title):
    """
    Returns a list of recommended drama titles based on the input drama.
    Returns empty list if the drama is not found or an error occurs.
    """
    try:
        # Find the index of the input drama (case-insensitive)
        drama_index_series = newdf[newdf['Title'].str.lower() == drama_title.lower()].index
        if drama_index_series.empty:
            return [] # Drama title not found

        drama_index = drama_index_series[0]
        distances = similarity[drama_index]

        # Get top 5 similar dramas (excluding itself)
        drama_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommendation_titles = []
        for i in drama_list:
            recommendation_titles.append(newdf.iloc[i[0]].Title)
        return recommendation_titles
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return []

# --- Rating Extraction Function ---
def extract_numeric_rating(rating):
    if isinstance(rating, str):
        rating = rating.strip().replace('\xa0', ' ')
        match = re.search(r'(\d+(\.\d+)?)', rating)
        if match:
            return float(match.group(1))
    return None

# Apply numeric rating extraction to the original dataframe
original_df['Numeric_Rating'] = original_df['Rating'].apply(extract_numeric_rating)


# --- Top Dramas by Country Logic ---
def get_top_dramas_by_country(dataframe, country_name, top_n=5):
    filtered = dataframe[dataframe['Country'].str.contains(country_name, case=False, na=False)]
    top_dramas = filtered.drop_duplicates(subset='Title') \
                         .dropna(subset=['Numeric_Rating']) \
                         .sort_values(by='Numeric_Rating', ascending=False) \
                         .head(top_n) \
                         .reset_index(drop=True)

    return top_dramas

# --- Streamlit UI ---
st.set_page_config(layout="wide") # Use wide layout

# Custom CSS for light theme with background image and improved layout
st.markdown("""
<style>
.main {
    background: url("https://i.pinimg.com/736x/30/03/53/3003532468569d0294364a26df8b752f.jpg") no-repeat center center fixed;
    background-size: cover;
    backdrop-filter: blur(8px); /* Apply blur effect */
    -webkit-backdrop-filter: blur(8px); /* Safari support */
    color: #333333; /* Dark text for readability */
    padding: 20px;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Modern font */
}
h1, h2, h3 {
    color: #0f4c75; /* Dark blue headings */
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1); /* Subtle text shadow for readability */
    margin-bottom: 15px;
}
.stTextInput>div>div>input {
    border: 2px solid #0f4c75; /* Darker blue border */
    border-radius: 8px; /* More rounded corners */
    padding: 12px; /* Increased padding */
    background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent white background for input */
    color: #333333; /* Dark text in input */
    font-size: 1rem;
}
.stTextInput>label {
    color: #333333; /* Label color */
    font-weight: bold;
    font-size: 1.1rem;
    margin-bottom: 5px;
}
.stButton>button {
    background-color: #0f4c75; /* Dark blue button */
    color: white;
    border-radius: 8px; /* More rounded corners */
    padding: 12px 28px; /* Increased padding */
    font-size: 1.1rem;
    margin-top: 15px; /* Add space above the button */
    transition: background-color 0.3s ease; /* Smooth transition */
    border: none; /* Remove default border */
}
.stButton>button:hover {
    background-color: #1b6b9a; /* Lighter blue on hover */
    color: #e0e0e0; /* Slightly lighter text on hover */
}
.recommendation-item {
    background-color: rgba(255, 255, 255, 0.95); /* More opaque white background for recommendation items */
    color: #333333; /* Dark text */
    border-left: 6px solid #0f4c75; /* Thicker dark blue left border */
    padding: 15px; /* Increased padding */
    margin-bottom: 10px; /* More space between items */
    border-radius: 8px; /* Rounded corners */
    box-shadow: 3px 3px 8px rgba(0,0,0,0.25); /* More prominent shadow */
    transition: transform 0.2s ease; /* Smooth hover effect */
}
.recommendation-item:hover {
    transform: translateY(-5px); /* Lift effect on hover */
}

/* Style for suggestions dropdown */
.suggestions-box {
    border: 1px solid #bbb; /* Lighter border */
    border-radius: 4px;
    max-height: 150px;
    overflow-y: auto;
    background-color: rgba(255, 255, 255, 0.98); /* Almost opaque white background */
    position: absolute;
    z-index: 1000;
    width: calc(100% - 20px);
    margin-top: 5px;
    color: #333333;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.15);
}
.suggestion-item {
    padding: 10px 12px; /* Increased padding */
    cursor: pointer;
    color: #333333;
    border-bottom: 1px solid #eee; /* Separator line */
    transition: background-color 0.2s ease;
}
.suggestion-item:last-child {
    border-bottom: none; /* No border on the last item */
}
.suggestion-item:hover {
    background-color: #e0e0e0; /* Light gray on hover */
    color: #000000; /* Black text on hover */
}
a {
    color: #0f4c75; /* Dark blue links */
    text-decoration: none; /* No underline by default */
}
a:hover {
    color: #1b6b9a; /* Lighter blue on hover */
    text-decoration: underline; /* Underline on hover */
}

/* Container for cards/sections for better structure */
.stContainer {
    background-color: rgba(255, 255, 255, 0.9); /* Semi-transparent white background for containers */
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

/* Style for columns to add spacing */
.st-eq {
    padding: 0 10px; /* Add horizontal padding to columns */
}


</style>
""", unsafe_allow_html=True)


st.title('MyMatcha 🍵 - BL Drama Recommendations') # Updated title

st.markdown("<p style='font-size: 1.2rem; color: #444; margin-bottom: 30px;'>Find your next favorite BL drama based on titles you love!</p>", unsafe_allow_html=True)


# Recommendation Section
st.header('🔍 Get Your Recommendations')

# Use a unique key for the text input
drama_title = st.text_input('Enter a drama title:', '', key='drama_input')

# Function to filter and display suggestions
def show_suggestions(input_text, all_titles):
    if input_text:
        # Filter titles that contain the input text (case-insensitive)
        suggestions = [
            title for title in all_titles if input_text.lower() in title.lower()
        ]
        # Display suggestions, limit to top 10
        if suggestions:
            # Create a container for suggestions
            with st.container():
                st.markdown("<div class='suggestions-box'>", unsafe_allow_html=True)
                for i, suggestion in enumerate(suggestions[:10]): # Limit to 10 suggestions
                    # Use st.write with HTML and a custom data attribute to handle click
                    # We can't directly trigger Streamlit events from simple HTML,
                    # so we'll guide the user to click the button after selecting.
                    st.markdown(f"<div class='suggestion-item' data-suggestion='{suggestion}'>{suggestion}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Use an empty markdown to clear suggestions if input is empty
        st.markdown("")


# Get all drama titles for suggestions
all_drama_titles = newdf['Title'].tolist()

# Display suggestions based on current input (will be handled by the JS and input change)
# We need a placeholder to display suggestions dynamically
# suggestions_placeholder = st.empty()

# Re-run the show_suggestions function whenever the input changes
# This will re-render the suggestions_placeholder content
# show_suggestions(drama_title, all_drama_titles) # Removed this as JS handles rendering


if st.button('Find Similar Dramas'):
    if drama_title:
        recommended_titles = recommend(drama_title)
        if recommended_titles:
            st.subheader(f'Here are 5 recommendations similar to "{drama_title}":')
            # Use a container for recommendations
            with st.container():
                for i, title in enumerate(recommended_titles):
                    st.markdown(f"<div class='recommendation-item'>{i+1}. {title}</div>", unsafe_allow_html=True)
        else:
            st.warning(f"Sorry, I couldn't find recommendations for '{drama_title}'. Please try a different title.")
    else:
        st.warning("Please enter a drama title to get recommendations.")

# Javascript to handle clicks on suggestions (updated to interact with Streamlit input)
st.markdown("""
<script>
const textInput = document.querySelector('[data-testid="stTextInput"] input');
const suggestionsContainer = document.querySelector('.suggestions-box');

if (textInput) {
    textInput.addEventListener('input', function() {
        // Since we are rerunning the script on every input change
        // the suggestions will be re-rendered by the Python code
        // We just need to make sure the suggestions box appears below the input
        // This positioning is handled by the CSS 'position: absolute' and margin-top
    });
}


// Function to handle clicks on suggestions
function handleSuggestionClick(event) {
    const suggestionItem = event.target.closest('.suggestion-item');
    if (suggestionItem) {
        const suggestionValue = suggestionItem.getAttribute('data-suggestion');
        const inputElement = document.querySelector('[data-testid="stTextInput"] input');
        if (inputElement) {
            inputElement.value = suggestionValue;
            // Trigger a change event so Streamlit knows the value updated
            inputElement.dispatchEvent(new Event('change', { bubbles: true }));
            // Optional: Hide suggestions after clicking
            // if (suggestionsContainer) {
            //     suggestionsContainer.style.display = 'none';
            // }
            // Note: Auto-triggering the button click is complex with Streamlit's
            // current event model via simple JS. Users will need to click the button.
        }
    }
}

// Add click listener to the document and delegate to suggestion items
document.addEventListener('click', handleSuggestionClick);

// Re-run the suggestion display based on the current input value when the app loads or reruns
// This is needed because Streamlit reruns the entire script
const currentInput = textInput ? textInput.value : '';
if (currentInput) {
    // Simulate an input event to trigger the suggestion display logic in Python
     if (textInput) {
         textInput.dispatchEvent(new Event('input', { bubbles: true }));
     }
}

</script>
""", unsafe_allow_html=True)

# Call show_suggestions on initial load and reruns
show_suggestions(drama_title, all_drama_titles)


st.markdown("---") # Separator

# Top Dramas by Country Section
st.header('🏆 Top Dramas by Country')

countries_to_display = ['South Korea', 'Thailand', 'Japan'] # Limit to only these three countries

# Use columns for better alignment (three columns)
cols = st.columns(3)

# Display top dramas for each country in their respective columns
for i, country in enumerate(countries_to_display):
    with cols[i]: # Place each country's list in a separate column
        st.subheader(f'{country}') # Country name as subheader
        top_dramas = get_top_dramas_by_country(original_df, country, top_n=5) # Display top 5
        if not top_dramas.empty:
            for j, row in top_dramas.iterrows():
                st.write(f"**{j+1}.** {row['Title']}  ⭐ {row['Numeric_Rating']}", unsafe_allow_html=True) # Bold number
        else:
            st.write(f"⚠️ No valid rated dramas found for {country}")


st.markdown("---") # Separator

# Featured Dramas Section
st.header("✨ You Should Watch Next:")

featured_dramas = [
    {"title": "2gether The Series", "image_url": "https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcS-y4Vp35wbn8yIBeakPrXEGuij2djmctVlxHpQMA4n28Vv1sHN"},
    {"title": "Love in the Air", "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQXsgce6r_jq2Mks-ZCshGpQUHFe9eWyspjvlfOOb13IzCN5nOA"},
    {"title": "Jun & Jun", "image_url": "https://i.mydramalist.com/d0O3r5_4f.jpg"}
]

cols = st.columns(len(featured_dramas)) # Create columns for featured dramas

for i, drama_info in enumerate(featured_dramas):
    with cols[i]:
        st.subheader(drama_info["title"])
        st.image(drama_info["image_url"], caption=drama_info["title"], use_container_width=True)


st.markdown("---") # Separator

# Removed Chatbot Section

# Footer Section
st.markdown("""
<div style="text-align: center; padding: 20px; color: #555555; font-size: 0.9rem;">
    <p>Disclaimer: This recommendation model is based on data scraped from various sources and is for informational purposes only. Ratings and data accuracy may vary.</p>
    <p>Data source: <a href="https://docs.google.com/spreadsheets/d/1-bHukazIbC7jwghhwUuQO50rtEqGA27JgLfl5jK03jk/export?format=csv" style="color: #0f4c75;">Google Sheets</a></p>
    <p>&copy; 2025 MyMatcha</p>
</div>
""", unsafe_allow_html=True)
