# 📝 Sentiment Analysis on Amazon Product Reviews

## 📌 Objective
Build a machine learning model that can classify Amazon product reviews as **Positive**, **Neutral**, or **Negative** based on their content.

## 📊 Dataset
- Source: Datafiniti Amazon Consumer Reviews Dataset
- File used: `1429_1.csv` (from `archive/` folder)
- Features used:
  - `reviews.text`: actual review content
  - `reviews.rating`: review score (1, 3, or 5)

## 🧠 Technologies Used
- Python
- Pandas, NumPy
- NLTK for text preprocessing
- Scikit-learn (TF-IDF, Logistic Regression, Metrics)
- Matplotlib + Seaborn for visualization

## 🛠 Steps Performed
1. Loaded and cleaned raw review data
2. Removed special characters and stopwords
3. Converted review text into numerical form using TF-IDF
4. Trained a Logistic Regression classifier
5. Evaluated the model using accuracy, precision, recall, F1
6. Visualized results using a confusion matrix

## ✅ Output
The model predicts sentiment of unseen product reviews with good accuracy.

## 🧪 Sample Results
Classification report and confusion matrix showing sentiment performance.

## 🙋‍♀️ Built by
Chelsi Patell (AI/ML Intern Project)
