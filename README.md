# Wentworth Academy NLP Chatbot

An end-to-end Natural Language Processing (NLP) chatbot built for the fictional **Wentworth Academy**. The system classifies student queries across school-related topics using custom feature engineering, TF-IDF vectorization, and a Multilayer Perceptron (MLP) neural network, delivering controlled answers through a hybrid rule-based response layer.

---

## Overview

Wentworth Academy Chatbot serves as a conversational assistant for daily school inquiries. Students can ask about class schedules, faculty assignments, campus facilities, examination periods, student clubs, and academic tutoring.

This project demonstrates the complete machine learning lifecycle:
- Dataset structuring and automated ingestion
- Lexical and syntactic NLP feature engineering
- Text vectorization with TF-IDF and numerical standardization
- Traditional baseline modeling vs. neural network classification
- Real-time inference with prediction confidence scoring
- Deterministic, contextual response generation
- Out-of-scope fallback handling for unknown queries
- Self-contained web application served via Python's standard library

---

## Features

- **Multi-Feature NLP Pipeline**: Combines unigram TF-IDF representations with custom engineered statistical and syntactic features.
- **Neural Network Intent Classification**: Multi-class classification powered by a 3-layer MLP architecture.
- **Hybrid Response Architecture**: Decouples intent classification from response generation, using query keyword extraction to resolve specific entities (e.g., faculty names, building locations).
- **Out-of-Scope Handling**: Dedicated `unknown` class prevents hallucinated answers for out-of-domain queries.
- **Confidence Scoring**: Real-time confidence percentage returned alongside every classification.
- **Lightweight Web Interface**: Modern, responsive academy portal interface with an interactive topic discovery guide and verified clickable example prompts.
- **Minimal Dependencies**: Requires only `pandas` and `scikit-learn`, using Python's built-in HTTP server for local hosting.

---

## Supported Intents

The system supports 10 distinct intents defined in the dataset:

| Intent | Description | Verified Example Query |
|---|---|---|
| `greeting` | Welcomes the student to the academy | "Hi there" |
| `school_location` | Directions to campus buildings and facilities | "Where can I find the library?" |
| `class_schedule` | Timetable and schedule information | "Can I check my class schedule" |
| `teacher_information` | Subject-specific teacher and faculty assignments | "Who teaches chemistry?" |
| `exam_schedule` | Final and midterm examination dates | "When are the final exams?" |
| `club_information` | Student extracurricular activities and clubs | "What clubs can I join?" |
| `school_rules` | Campus policies and guidelines | "Are students allowed to use phones during lessons" |
| `academic_help` | Subject-level tutoring and assistance | "I need help with algebra" |
| `goodbye` | Polite farewell greetings | "Goodbye" |
| `unknown` | Fallback for off-topic or unsupported questions | "How do I bake a chocolate cake?" |

---

## Project Architecture

The application follows a sequential data and inference pipeline:

```
[User Message]
      |
      v
[Data / Feature Processing]
      |
      v
[TF-IDF + Numeric Features]
      |
      v
[MLP Neural Network]
      |
      v
[Predicted Intent + Confidence]
      |
      v
[Contextual Response Layer]
      |
      v
[Web UI - app.py]
```

Training data is loaded and structured through `src/data_loader.py` before being processed by the feature engineering and transformation modules.

---

## NLP Approach

The feature extraction pipeline combines statistical text representations with domain-specific numerical features:

1. **TF-IDF Vectorization**: `TfidfVectorizer(ngram_range=(1, 1))` extracts unigram term frequency-inverse document frequency features from the raw message text.
2. **Engineered Text Features**:
   - `word_count`: Total word count in the query.
   - `char_count`: Total character length of the query.
   - `avg_word_length`: Mean character length per word.
   - `question_mark`: Binary indicator for the presence of `?`.
   - `exclamation_mark`: Binary indicator for the presence of `!`.
   - `exam_signal`: Binary indicator detecting assessment keywords (`exam`, `test`, `tested`, `testing`, `quiz`, `assessment`, `midterm`, `final`).
3. **Feature Scaling**: `StandardScaler` standardizes numerical features to zero mean and unit variance.
4. **ColumnTransformer**: Combines text TF-IDF vectors and scaled numerical features into a single feature matrix.

---

## Model

### 1. Traditional Baseline (`src/training.py`)
- **Algorithm**: `LogisticRegression(C=10, max_iter=1000)`
- **Role**: Serves as a traditional machine learning baseline to benchmark classification performance before moving to neural architectures.

### 2. Final Neural Network (`src/training_mlp.py`)
- **Algorithm**: `sklearn.neural_network.MLPClassifier`
- **Architecture**:
  - Hidden Layers: 3 layers with `(128, 64, 32)` neurons
  - Activation: Rectified Linear Unit (`activation="relu"`)
  - Optimizer: Adam (`solver="adam"`)
  - Training Iterations: `max_iter=500`
  - Random State: `random_state=42`

---

## Performance

Model performance evaluated on the stratified held-out test split (30% test size, 68 test samples):

- **Accuracy**: 94%
- **Macro F1-Score**: 95%
- **Weighted F1-Score**: 94%

> Note: These metrics reflect performance on the curated held-out test split under controlled experimental conditions and do not represent generalized open-domain certainty.

---

## Response Layer

The application uses a **hybrid architecture** that separates machine learning classification from response generation:

1. **Machine Learning Model**: Predicts the query's intent (e.g., `teacher_information`) and calculates prediction confidence.
2. **Rule-Based Response Layer (`src/response.py`)**:
   - Inspects the original message text alongside the predicted intent.
   - Resolves specific entities:
     - **Teachers**: Identifies subjects (Chemistry: Ms. Carter, Mathematics: Mr. Brooks, Biology: Mr. Wilson, Physics: Mr. Anderson, English: Ms. Parker, History: Ms. Bennett).
     - **Locations**: Identifies campus facilities (Library, Science Building, Laboratories, Cafeteria, Student Center, Gymnasium, Main Office).
     - **Clubs & Policies**: Matches specific student organizations (Debate, Science, Music, Art, Sports) and student conduct rules.
   - Returns deterministic, fictional school answers or a helpful fallback message for out-of-scope queries.

This separation ensures that fictional facts remain factual and consistent without relying on unconstrained text generation.

---

## Project Structure

```
app.py
README.md
requirements.txt
test_chatbot.py
test_data_loader.py
test_features.py
test_mlp.py
test_prediction_layer.py

data/
    intents.csv

src/
    data_loader.py
    features.py
    prediction.py
    preprocessing.py
    response.py
    training.py
    training_mlp.py
```

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/patelsmit515/wentworth_academy_chatbot.git
   cd wentworth_academy_chatbot
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Create environment
   python -m venv .venv

   # Activate on Windows:
   .venv\Scripts\activate

   # Activate on macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

Start the local web server by executing:

```bash
python app.py
```

When launched:
1. The dataset is loaded and features are extracted.
2. The MLP neural network trains in memory.
3. The local server starts at:
   ```
   http://127.0.0.1:5000
   ```
4. Your default web browser opens automatically with the interactive chat interface.

---

## Testing

Individual pipeline modules can be verified using the standalone test scripts:

```bash
# Verify dataset loading and intent distribution
python test_data_loader.py

# Verify text feature extraction
python test_features.py

# Evaluate MLP classification metrics and confusion matrix
python test_mlp.py

# Test prediction layer and confidence scoring
python test_prediction_layer.py

# Run end-to-end integration test
python test_chatbot.py
```

---

## Limitations

- **Fictional Domain**: Wentworth Academy and its faculty, facilities, and schedules are entirely fictional.
- **Closed Knowledge Base**: The chatbot is constrained to the 10 defined intent categories and does not answer general-knowledge questions.
- **Single-Turn Context**: Each message is processed independently without multi-turn dialogue memory.
- **Fallback Behavior**: Questions outside the supported topics will trigger the `unknown` intent response.

---

## Future Improvements

- Expanding the dataset with more conversational variations and student queries.
- Implementing multi-turn dialogue state tracking for contextual follow-up questions.
- Containerizing the application with Docker for cloud deployment.
