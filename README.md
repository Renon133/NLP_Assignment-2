# NLP_Assignment-2

This repository contains the code and resources for Assignment 2 in the NLP course. The focus of this assignment is on tokenization, preprocessing, model testing, and analyzing text data in the context of Gujarati language processing.

## Repository Structure

The structure of the repository is as follows:

### Directories
- **Tokenizer/**: Contains files and scripts related to tokenization logic and experiments.
- **gujarati_tokenizer/**: A specific tokenizer implementation for the Gujarati language.

### Notebooks
- **fertility_score.ipynb**: A notebook that computes the fertility score for different tokenization methods or models, used as a measure of language model efficiency.
- **load_dataset.ipynb**: Contains code to load and explore datasets, including initial data visualization and statistics.
- **preprocess_txt.ipynb**: A preprocessing notebook that includes steps to clean and prepare raw text data for model training.
- **test_final_model.ipynb**: A notebook dedicated to testing the final model, including evaluation metrics and qualitative analysis.
- **test_tokenizer.ipynb**: Tests the performance and correctness of various tokenizers.
- **testing.ipynb**: A general testing notebook used for experimentation and trying out different code snippets.
- **training_final.ipynb**: The main notebook for training the final NLP model, including hyperparameter tuning and performance tracking.

### Scripts
- **tokenizer.py**: A Python script containing the tokenizer implementation and supporting functions.

### Other Files
- **README.md**: This file, containing information about the project and its contents.

## Assignment Objectives

1. **Tokenization**: Develop and test tokenization strategies, particularly focusing on the Gujarati language.
2. **Preprocessing**: Clean and standardize text data to improve model performance.
3. **Model Training and Testing**: Implement and evaluate models using preprocessed data, measuring their performance.
4. **Fertility Score Analysis**: Analyze the fertility score of the trained models to assess their efficiency in handling language-specific data.

## Key Steps and Usage

1. **Loading the Dataset**:
   - Run `load_dataset.ipynb` to load and inspect the dataset.
2. **Text Preprocessing**:
   - Use `preprocess_txt.ipynb` for data cleaning and formatting.
3. **Tokenization**:
   - Implement custom tokenizers by running `tokenizer.py` or experimenting with `Tokenizer/` and `gujarati_tokenizer/`.
4. **Model Training**:
   - Train models by executing `training_final.ipynb` and adjusting hyperparameters as needed. Contains perplexity of the model at every 0.1 epoch
5. **Evaluation**:
   - Evaluate the trained models using `test_final_model.ipynb` and analyze the results in `fertility_score.ipynb`.

## License

This project is intended for educational purposes and follows standard academic integrity guidelines.
-----
