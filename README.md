# Detecting Fake News

This project uses a machine learning model to detect fake news. The model is trained using the PassiveAggressiveClassifier from scikit-learn, and the text data is vectorized using TfidfVectorizer.

## How It Works

1. **Data Preparation**: The dataset containing news articles is loaded, and any missing values in the text data are handled.
2. **TF-IDF Vectorization**: The text data is converted into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF) vectorization.
3. **Model Training**: A PassiveAggressiveClassifier is trained on the vectorized text data.
4. **Prediction and Evaluation**: The trained model is used to predict labels on the test set, and the accuracy and confusion matrix are calculated to evaluate the model's performance.

## Output

After running the script, you should see the following output:

    Accuracy: 99.64%
    [[4529 22]
    [ 10 4293]]


This indicates that the model achieved an accuracy of 99.64%, and the confusion matrix shows the number of true positives, false positives, true negatives, and false negatives.
