# Fake News Detection API

A machine learning-powered API that detects whether a piece of news is real or fake using text preprocessing and classification models.

## Features

- **FastAPI-based REST API** - Fast and modern Python web framework
- **Real-time Prediction** - Classify news as Real or Fake instantly
- **Web UI** - Simple and intuitive user interface for testing
- **Text Preprocessing** - Automatic text cleaning (lowercasing, URL removal, number removal, punctuation removal)
- **Pre-trained Models** - Uses TF-IDF vectorizer and trained machine learning model

## Project Structure

```
fake_news_api/
├── app.py                 # Main FastAPI application
├── preprocess.py          # Text preprocessing module
├── requirements.txt       # Python dependencies
├── models/                # Pre-trained models directory
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
├── static/                # Static files (CSS, images, etc.)
│   └── style.css
└── templates/             # HTML templates
    └── index.html
```

## Installation

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Server

Start the development server:
```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`

### API Endpoints

#### GET `/`
Returns the web UI for interactive testing.

#### POST `/predict`
Predicts whether the provided news text is real or fake.

**Request Body:**
```json
{
  "text": "Your news text here"
}
```

**Response:**
```json
{
  "prediction": "Real News"
}
```
or
```json
{
  "prediction": "Fake News"
}
```

## Models

The project uses pre-trained models stored in the `models/` directory:
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer for text feature extraction
- `fake_news_model.pkl` - Machine learning classification model

## Implementation Details

- **Text Preprocessing**: Converts text to lowercase, removes URLs, numbers, and punctuation
- **Feature Extraction**: Uses TF-IDF vectorization
- **Classification**: Uses a pre-trained machine learning model

## Technologies Used

- **Framework**: FastAPI
- **Server**: Uvicorn
- **ML Libraries**: joblib, scikit-learn
- **Templating**: Jinja2
- **Frontend**: HTML, CSS

## License

This project is part of Gen AI coursework at LPU.
