# 🤖 AI-Powered E-Consultation Suite

An end-to-end web application designed to analyze, summarize, and derive actionable insights from large volumes of public feedback for government e-consultations. This project was developed as a solution for the Smart India Hackathon (SIH).

---

### ✨ Live Demo & Preview

**[➡️ Click here for the Live Interactive Demo]([https://e-consultation-analyzer-8yxqu8ddokfyjm9mqebwxq.streamlit.app/])** *(Deploy on Streamlit Community Cloud and replace this link)*

![App Demo GIF]([SCREENRECORDING-OF-YOUR-APP.GIF])
*(**Important:** Record a short GIF of you using the app and place it here. This is the most effective part of the README.)*

---

### 🎯 The Problem

Government ministries and corporate bodies receive thousands of public comments during e-consultations. Manually reading, filtering, and analyzing this feedback is a slow, expensive, and often biased process. This bottleneck can lead to a disconnect between policymakers and citizen concerns, eroding public trust. Our solution automates this entire workflow, making public consultation faster, smarter, and more inclusive.

---

### ⭐ Key Features

* **🧠 State-of-the-Art Sentiment Analysis:** Utilizes a RoBERTa-based Transformer model to understand the context and nuance of comments, providing accurate sentiment scores (Positive, Negative, Neutral).
* **📝 Abstractive Text Summarization:** Condenses long, detailed submissions into concise summaries using a DistilBART model, allowing for rapid review.
* **🕷️ Automated Web Scraping:** Ingests public comments directly from dynamic, modern websites using Playwright.
* **🚫 Intelligent Spam Filtering:** A rule-based pre-processing step to clean data and remove irrelevant or malicious comments before they reach the AI models.
* **📊 Interactive Dashboard:** A user-friendly interface built with Streamlit that presents all insights in real-time through dynamic charts and tables powered by Plotly.
* **🔑 Thematic Analysis:** Automatically extracts key themes and recurring phrases to quickly identify what topics are most important to the public.

---

### 🔧 Tech Stack

* **Backend & App Framework:** Python, Streamlit
* **AI & NLP:** Hugging Face Transformers, PyTorch, Scikit-learn
* **Data Handling:** Pandas
* **Web Scraping:** Playwright
* **Data Visualization:** Plotly

---

###  flowchart Architecture

This flowchart illustrates the end-to-end data pipeline, from raw input to the final analysis dashboard.


`![Flowchart Architecture](flowchart_1.png)`

---

### 🚀 Setup and Local Installation

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [YOUR-GITHUB-REPO-LINK-HERE]
    cd [YOUR-REPO-NAME]
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Make sure you have a `requirements.txt` file in your repository.)*

4.  **Install Playwright's browser binaries:**
    ```bash
    playwright install
    ```

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser.

---

### 🤝 Acknowledgements
This project leverages the incredible work done by the teams behind Streamlit, Hugging Face, and the various open-source models used.
* **Sentiment Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
* **Summarization Model:** `sshleifer/distilbart-cnn-12-6`
