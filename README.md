# IMDB Review Sentiment Analysis

## Project Overview

This project focuses on sentiment analysis of IMDB movie reviews using deep learning techniques. The model is developed with TensorFlow and Keras and deployed using Streamlit. The project involves key stages including data preprocessing, model building, training, evaluation, and deployment.

## Project Structure

- **Data Preprocessing**: 
  - Encoded sentiment labels (`positive` as `1` and `negative` as `0`).
  - Performed text analysis by removing punctuation and stopwords, and applying lemmatization.
  - Tokenized the text data and padded sequences to ensure uniform input length.

- **Model Building**:
  - Constructed a Sequential model with:
    - `Embedding` layer to convert words to dense vectors.
    - `GlobalAveragePooling1D` layer for dimensionality reduction.
    - `Dense` output layer with softmax activation for binary classification.
  - Compiled the model using the Adam optimizer and categorical cross-entropy loss.

- **Training**:
  - Split the dataset into training and testing sets.
  - Trained the model for 10 epochs, tracking both training and validation metrics.

- **Evaluation**:
  - Visualized the model’s loss and accuracy over epochs using Matplotlib.

- **Deployment**:
  - Saved the trained model and tokenizer for future use.
  - Developed a Streamlit app allowing users to input reviews and receive sentiment predictions.

## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/imdb-review-sentiment-analysis
   cd imdb-review-sentiment-analysis
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

4. **Input Text**: Enter any movie review in the text box provided and hit 'Predict' to see the sentiment result.

## Model Performance

The model was trained on a subset of the IMDB dataset and achieved competitive accuracy on the test set. The text preprocessing steps, combined with a well-structured neural network, contributed to the model’s effectiveness in classifying movie reviews.

## Future Work

- **Model Enhancement**: Experiment with more complex architectures, such as recurrent neural networks (RNNs) or transformer-based models, for improved accuracy.
- **Dataset Expansion**: Utilize the full IMDB dataset for training to enhance model performance and generalization.
- **Additional Features**: Explore adding sentiment intensity scoring for a more nuanced analysis of reviews.

## Conclusion

This project highlights the use of deep learning in text classification tasks, specifically in the domain of sentiment analysis. The README provides a comprehensive guide to replicating and extending this work.
