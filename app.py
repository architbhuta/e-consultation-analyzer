import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
from streamlit_option_menu import option_menu

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Comment Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling (No changes here) ---
css_string = """
/* Card styling for the dashboard */
.dashboard-card {
    padding: 25px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
    height: 100%; /* Make cards of equal height */
}
.dashboard-card:hover {
    transform: translateY(-5px);
    border: 1px solid rgba(147, 112, 219, 0.5);
}
.dashboard-card h3 { font-size: 1.5rem; color: #FFFFFF; }
.dashboard-card p { color: #B0B0B0; }
"""
st.markdown(f"<style>{css_string}</style>", unsafe_allow_html=True)


# --- Model Loading (No changes here) ---
@st.cache_resource
def load_models():
    roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    vader_analyzer = SentimentIntensityAnalyzer()
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device="cpu")
    return roberta_tokenizer, roberta_model, vader_analyzer, summarizer

roberta_tokenizer, roberta_model, vader_analyzer, summarizer = load_models()


# --- All Analysis Functions (No changes here) ---
def analyze_roberta_sentiment(text):
    processed_text = " ".join([t if not t.startswith('@') else '@user' for t in text.split()])
    encoded_text = roberta_tokenizer(processed_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = roberta_model(**encoded_text)
    scores = F.softmax(output.logits, dim=1)[0].tolist()
    return {'Negative': scores[0], 'Neutral': scores[1], 'Positive': scores[2]}

def generate_summary(text, min_len, max_len):
    if len(text.split()) > 40:
        summary_list = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return summary_list[0]['summary_text']
    return "The provided text is too short to generate a meaningful summary."

def extract_key_themes(text, num_themes=5):
    text = re.sub(r'http\S+|@\S+|[^A-Za-z\s]', '', text.lower())
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english", max_features=num_themes)
    vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

def create_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color=None, mode="RGBA", colormap='viridis').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def scrape_comments(url):
    comments = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until='networkidle', timeout=20000)
            elements = page.query_selector_all("p")
            for element in elements:
                text = element.inner_text()
                if text and len(text.split()) > 8:
                    comments.append(text)
            browser.close()
            if not comments: st.warning("Could not find significant text using the generic 'p' tag selector.")
            return comments
    except PlaywrightTimeoutError: st.error(f"Timeout Error: The page at {url} took too long to load.")
    except Exception as e: st.error(f"An error occurred during scraping: {e}")
    return []


# --- UI Rendering Functions ---

# --- UPDATED: render_dashboard function is now interactive ---
def render_dashboard():
    st.title("AI-Powered E-Consultation Suite")
    st.markdown("##### Leverage AI to streamline stakeholder feedback review. Select a tool from the sidebar or a card below to begin.")
    st.markdown("---")

    # Use st.columns to ensure cards are aligned and have equal height
    col1, col2 = st.columns(2)
    
    with col1:
        # We use a container to group the card and button
        with st.container():
            st.markdown(
                """
                <div class="dashboard-card">
                    <h3>🧠 Sentiment Analysis</h3>
                    <p>Analyze comments from text or a URL to gauge public opinion. Get detailed breakdowns, key themes, and more.</p>
                </div>
                """, unsafe_allow_html=True)
            # This button is now linked to the session state
            if st.button("Launch Sentiment Analysis", use_container_width=True, key="nav_senti"):
                st.session_state.page = "Sentiment Analysis"
                st.rerun() # Rerun the script to navigate to the new page

    with col2:
        with st.container():
            st.markdown(
                """
                <div class="dashboard-card">
                    <h3>📝 Smart Summaries</h3>
                    <p>Paste long documents or articles to generate concise, AI-powered summaries. Perfect for quick insights.</p>
                </div>
                """, unsafe_allow_html=True)
            if st.button("Launch Smart Summarizer", use_container_width=True, key="nav_summary"):
                st.session_state.page = "Smart Summaries"
                st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("Coming Soon: Trend Visualization and Dynamic Word Clouds", icon="✨")


# --- The other page rendering functions remain the same ---
def render_analysis_page():
    # This function's content is unchanged
    st.header("🧠 Sentiment Analysis")
    st.markdown("Provide text to analyze sentiment, extract key themes, and generate insights.")
    input_type = st.radio("Choose input method:", ("Direct Text Input", "Scrape from URL"), horizontal=True)
    comments_to_analyze = []
    if input_type == "Direct Text Input":
        user_input = st.text_area("Paste your comments here (one per line):", height=200, placeholder="This is a fantastic initiative.\nThe proposal has some major flaws.")
        if user_input:
            comments_to_analyze = [c.strip() for c in user_input.split('\n') if c.strip()]
    else:
        url = st.text_input("Enter the URL to scrape comments from:")
        if st.button("Scrape Comments"):
            with st.spinner(f"Scraping {url}..."):
                scraped_comments = scrape_comments(url)
                if scraped_comments:
                    st.session_state.scraped_text = "\n".join(scraped_comments)
                    st.success(f"Successfully scraped {len(scraped_comments)} paragraphs.")
        user_input = st.text_area("Scraped/Editable Text:", value=st.session_state.get('scraped_text', ''), height=200)
        if user_input:
            comments_to_analyze = [c.strip() for c in user_input.split('\n') if c.strip()]

    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if not comments_to_analyze:
            st.warning("Please provide text to analyze.")
        else:
            with st.spinner("AI is thinking..."):
                all_roberta_scores = [analyze_roberta_sentiment(c) for c in comments_to_analyze]
                avg_scores = pd.DataFrame(all_roberta_scores).mean()
                st.session_state.analysis_results = { 'overall_sentiment': avg_scores.idxmax(), 'confidence': avg_scores.max(), 'avg_scores': avg_scores, 'full_text': " ".join(comments_to_analyze), 'num_comments': len(comments_to_analyze) }
    
    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        sentiment_color = "green" if results['overall_sentiment'] == 'Positive' else "red" if results['overall_sentiment'] == 'Negative' else "orange"
        st.markdown(f"### Overall Sentiment: <span style='color:{sentiment_color};'>{results['overall_sentiment']}</span> (Confidence: {results['confidence']:.1%})", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Detailed Breakdown**")
            fig = px.bar(x=results['avg_scores'].index, y=results['avg_scores'].values, labels={'x': 'Sentiment', 'y': 'Score'}, color=results['avg_scores'].index, color_discrete_map={'Positive': '#2ca02c', 'Negative': '#d62728', 'Neutral': '#7f7f7f'}, text=[f"{v:.1%}" for v in results['avg_scores'].values])
            fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
            st.write("**🔑 Key Themes**")
            key_themes = extract_key_themes(results['full_text'])
            for theme in key_themes: st.markdown(f"- `{theme.title()}`")
        with col2:
            st.write("**☁️ Word Cloud**")
            wordcloud_fig = create_word_cloud(results['full_text'])
            st.pyplot(wordcloud_fig)

def render_summary_page():
    # This function's content is unchanged
    st.header("📝 Smart Summarizer")
    st.markdown("Paste a long article, report, or any block of text to generate a concise summary.")
    source_text = st.text_area("Paste your text here:", height=300, placeholder="Paste a long article here...")
    st.markdown("### Configure Summary Length")
    col1, col2 = st.columns(2)
    with col1: min_len = st.slider("Minimum Summary Length (words)", 20, 200, 30)
    with col2: max_len = st.slider("Maximum Summary Length (words)", 50, 500, 150)

    if st.button("Generate Summary", type="primary", use_container_width=True):
        if len(source_text.split()) > 40:
            with st.spinner("Generating summary..."):
                summary = generate_summary(source_text, min_len, max_len)
                st.session_state.summary_output = { 'summary': summary, 'original_wc': len(source_text.split()), 'summary_wc': len(summary.split()) }
        else:
            st.warning("Please enter text with more than 40 words.")
    
    if 'summary_output' in st.session_state:
        results = st.session_state.summary_output
        st.markdown("---")
        st.subheader("📄 Generated Summary")
        st.success(results['summary'])
        reduction = 100 - (results['summary_wc'] / results['original_wc'] * 100)
        st.info(f"**Original Word Count:** {results['original_wc']} | **Summary Word Count:** {results['summary_wc']} | **Reduction:** {reduction:.1f}%")

# --- UPDATED: Main App Router now uses session_state ---

# 1. Initialize page state
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# 2. Define the pages and their corresponding indices
pages = ["Dashboard", "Sentiment Analysis", "Smart Summaries"]
icons = ['house-door-fill', 'bar-chart-line-fill', 'file-text-fill']
# Get the index of the current page from session_state
try:
    current_page_index = pages.index(st.session_state.page)
except ValueError:
    current_page_index = 0 # Default to Dashboard if page is invalid

# 3. Render the sidebar and update state on click
with st.sidebar:
    selected_page = option_menu(
        "Main Menu", pages,
        icons=icons,
        menu_icon="cast", default_index=current_page_index,
        styles={ "container": {"background-color": "#1a1a1a"}, "icon": {"color": "#9370DB", "font-size": "20px"}, "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#4B0082"}, "nav-link-selected": {"background-color": "#4B0082"}, }
    )
    # If the selection in the sidebar changes, update the session_state and rerun
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()

# 4. Render the page based on the state
if st.session_state.page == "Dashboard":
    render_dashboard()
elif st.session_state.page == "Sentiment Analysis":
    render_analysis_page()
elif st.session_state.page == "Smart Summaries":
    render_summary_page()