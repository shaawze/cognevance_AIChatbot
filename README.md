
# AI Chatbot Using NLP

## Project Overview
This project implements a simple AI chatbot using Natural Language Processing (NLP) and machine learning. It classifies user messages into predefined intents and generates suitable responses through a Streamlit web interface.

## Technologies Used
- Python
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit
- Joblib

## Features
- Greeting and goodbye responses
- Intent recognition
- Responses to help requests
- Chatbot name and availability information
- Interactive web interface

## Project Files
- `app.py` — Streamlit chatbot application
- `chatbot_model.pkl` — Trained machine-learning model
- `responses.json` — Predefined chatbot responses
- `requirements.txt` — Required Python packages

## Setup and Installation
Install the required packages:

```bash
pip install -r requirements.txt
```

Make sure `app.py`, `chatbot_model.pkl`, and `responses.json` are in the same directory.

Run the application:

```bash
streamlit run app.py
```

## How It Works
1. User enters a message.
2. TF-IDF converts the text into numerical features.
3. Logistic Regression predicts the message intent.
4. The chatbot selects a response associated with the predicted intent.
5. Streamlit displays the conversation.

## Limitations
The chatbot uses a small, predefined training dataset. Its responses and intent recognition may be inaccurate for unfamiliar or ambiguous messages.

## Future Improvements
- Expand the training dataset
- Improve intent classification and fallback handling
- Add more conversational intents
- Evaluate performance on a larger test dataset

## Author
Developed as an AI/ML internship project.

