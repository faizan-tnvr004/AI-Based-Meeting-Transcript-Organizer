# Meeting Transcript Analyzer and Summarizer

## Project Overview
This project is a natural language processing system built entirely from scratch to analyze and summarize meeting transcripts. It features a custom feed-forward neural network that categorizes individual sentences into Action Items, Decisions, and Discussion Points. It also includes an unsupervised extractive summarizer to generate concise meeting overviews.

## Technical Constraints and Compliance
This system strictly adheres to the requirement of manual implementation. No external APIs, pre-trained foundation models, or large language models are utilized. The neural network architecture is defined, compiled, and trained locally using raw data, ensuring complete ownership of the machine learning pipeline.

## Features
* Automated Text Cleaning: Removes noise, timestamps, and filler words using Regular Expressions and NLTK.
* TF-IDF Vectorization: Converts text into a 5000-feature numerical matrix based on term significance.
* Custom Classifier: A multi-layer perceptron built with Keras, featuring 64-neuron and 32-neuron hidden layers with ReLU activation and Dropout regularization.
* Graph-Based Summarization: Implements the TextRank algorithm via NetworkX to extract the top five most central sentences from a transcript.
* Performance Evaluation: Generates classification reports and confusion matrices to mathematically validate model accuracy.

## Dataset Requirements
The model is designed to be trained and tested on the following types of datasets:
* AMI Meeting Corpus: Provides sentence-level dialogue acts used for mapping explicit action items and discussion points.
* MeetingBank: Provides transcripts aligned with official city council minutes, ideal for isolating definitive decisions.

## Prerequisites and Installation
To run this project, you must have Python installed along with the following primary libraries. You can install them via your terminal or command prompt.

```bash
pip install pandas numpy scikit-learn tensorflow nltk networkx matplotlib seaborn
```

You will also need to download the required NLTK data dictionaries for stopwords and lemmatization. The primary script handles this automatically upon execution.

## Project Structure
The project is organized into three sequential technical phases.

### Phase 0: Data Preprocessing
Loads the raw transcript data, applies text normalization, encodes the categorical target labels into integers, and transforms the text into a sparse TF-IDF matrix. The data is then split into an 80/20 training and testing configuration.

### Phase 1: Model Architecture
Initializes the Keras Sequential model. It defines the input layer, the two dense hidden layers, the dropout mechanism to prevent overfitting, and the final 3-neuron softmax output layer. The model is compiled using the Adam optimizer and the sparse categorical cross-entropy loss function.

### Phase 2: Training and Evaluation
Executes the training loop using the preprocessed data matrix. After generating the learned weights, the script runs predictions on the reserved test set to calculate Precision, Recall, and the F1-Score. It subsequently runs the unsupervised NetworkX summarizer on a sample transcript to output the final meeting summary.

## Usage Instructions
1. Clone this repository or download the source Jupyter Notebook file to your local environment or Google Colab.
2. Place your raw meeting dataset CSV file into the root directory and update the file path in the Pandas loading function.
3. Execute the code cells sequentially, starting from Phase 0 to ensure proper data flow.
4. Review the generated metrics, the visual confusion matrix, and the extracted summary output at the conclusion of the script.
