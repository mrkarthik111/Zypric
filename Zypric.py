import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline

# Load Pre-trained Models
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Load Product Data from CSV
@st.cache_data
def load_data():
    df = pd.read_csv('products.csv')
    df['Reviews'] = df['Reviews'].apply(lambda x: x.split(';'))  # Convert reviews from string to list
    return df

product_data = load_data()

# Function to analyze sentiment
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

# Function to detect emotion
def detect_emotion(text):
    result = emotion_pipeline(text)[0]
    return result['label'], result['score']

# Function to generate a word cloud
def generate_wordcloud(data):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(data))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Define a dictionary to map emotions to emojis
emotion_to_emoji = {
    'happy': '😄',
    'joy': '😄',
    'sad': '😢',
    'sadness': '😢',
    'angry': '😠',
    'surprised': '😲',
    'neutral': '😐',
    'fear': '😱',
    'disgust': '🥴'
}

# Function to get emoji based on detected emotion
def get_emoji(emotion):
    return emotion_to_emoji.get(emotion, '🤔')  # Default to thinking face if not found

# Apply custom styling
st.markdown("""
    <style>
        body {
            background-color: #ffffff;
            font-family: 'Verdana', sans-serif;
        }

        /* Title */
        .main-title {
            text-align: center;
            color: #3498db; /* Title color */
            font-size: 30px;
            font-weight: bold;
            margin-bottom: 20px;
        }

        /* Search bar */
        .search-container {
            text-align: center;
            margin-bottom: 20px;
        }

        /* Warning Message */
        .warning {
            text-align: center;
            color: #ff0000; /* Text color */
            font-size: 30px;
            font-weight: bold;
            margin-bottom: 20px;
        }

        /* Product card */
        .product-card {
            background: #ffffff; /* Card background color */
            padding: 20px;
            color: #000000;
            margin: 15px 0;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: 0.3s;
        }

        /* Price */
        .price {
            font-size: 20px; /* Adjusted font size */
            font-weight: bold;
            color: #000000; /* Price color */
        }

        /* Custom Button */
        .custom-button {
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: 0.3s;
        }

        .custom-button:hover {
            background-color: #2980b9;
        }

        /* Emoji styling */
        .emoji {
            font-size: 30px; /* Set the emoji size */
        }
    </style>
""", unsafe_allow_html=True)

# Logo Image
st.image("logo.png", width=700)  # Add your logo image here

# Search Bar
st.markdown('<div class="search-container">', unsafe_allow_html=True)
product_name = st.text_input("", placeholder="🔍 Search products...", key="search")
st.markdown('</div>', unsafe_allow_html=True)

# Button to trigger search (acting like 'Enter')
if st.button("Search", key="search_button") or product_name:
    # Use na=False to avoid NaN issues
    results = product_data[product_data['Product'].str.contains(product_name, case=False, na=False)]

    if not results.empty:
        for _, product in results.iterrows():
            st.markdown(f"""
                <div class="product-card">
                    <h2>{product["Product"]}</h2>
                    <h3 class="price">Amazon: ₹{product['Amazon_Price']} | Flipkart: ₹{product['eBay_Price']} | Snapdeal: ₹{product['Walmart_Price']}</h3>
                    <p><b>Reviews:</b></p>
                    <ul>
                        {"".join([f"<li>{review}</li>" for review in product['Reviews']])}
                    </ul>
                </div>
            """, unsafe_allow_html=True)

            # Analyze reviews
            sentiments = [analyze_sentiment(review) for review in product["Reviews"]]
            emotions = [detect_emotion(review) for review in product["Reviews"]]

            sentiment_labels, sentiment_scores = zip(*sentiments)
            emotion_labels, emotion_scores = zip(*emotions)

            # Display Sentiment Analysis
            st.write("### Sentiment Analysis")
            sentiment_df = pd.DataFrame({"Review": product["Reviews"], "Sentiment": sentiment_labels, "Score": sentiment_scores})
            st.dataframe(sentiment_df)

            # Display Emotion Analysis with emojis
            st.write("### Emotion Analysis")
            emotion_df = pd.DataFrame({
                "Review": product["Reviews"],
                "Emotion": [f"{emotion} {get_emoji(emotion)}" for emotion in emotion_labels],
                "Score": emotion_scores
            })
            st.dataframe(emotion_df)

            # Generate Word Cloud Button
            if st.button("Generate Word Cloud", key=product["Product"]):
                generate_wordcloud(product["Reviews"])
    else:
        st.markdown('<h3 class="warning">NO PRODUCTS FOUND</h3>', unsafe_allow_html=True)