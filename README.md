# Sentiment Analysis on Amazon Product Reviews

## Objective
Build a machine learning model that can classify Amazon product reviews as **Positive**, **Neutral**, or **Negative** based on their content.

## Dataset
- Source: Datafiniti Amazon Consumer Reviews Dataset
- File used: `1429_1.csv` (from `archive/` folder)
- Features used:
  - `reviews.text`: actual review content
  - `reviews.rating`: review score (1, 3, or 5)

## Technologies Used
- Python
- Pandas, NumPy
- NLTK for text preprocessing
- Scikit-learn (TF-IDF, Logistic Regression, Metrics)
- TensorFlow + Keras (for LSTM model)
- Matplotlib + Seaborn for visualization

##  Steps Performed
1. Traditional Machine Learning (Logistic Regression)
    - Loaded and cleaned raw review data
    - Removed punctuation and stopwords
    - Converted review text into numerical vectors using TF-IDF
    - Trained a Logistic Regression classifier
    - Evaluated using accuracy, precision, recall, F1-score
    - Visualized with a confusion matrix

2. Deep Learning (LSTM Model)
   - Cleaned text and tokenized using Keras Tokenizer
   - Converted text into padded sequences
   - Labeled reviews as positive, neutral, or negative based on rating
   - Built an LSTM model: `Embedding → LSTM → Dropout → Dense (Softmax)`
   - Compiled with Adam optimizer and categorical crossentropy
   - Evaluated using classification report and confusion matrix


## Output
- The logistic regression model performed well on vectorized TF-IDF features.
- The LSTM model was able to learn contextual patterns in text data.
- Both models were evaluated and compared.


## Sample Results
1. Logistic Regression:
     - F1-score: ~0.85
     - Confusion matrix: good class separation

2. LSTM Model:
     - Improved performance for longer reviews
     - Classification report generated using test set

## Built by
Chelsi Patel
