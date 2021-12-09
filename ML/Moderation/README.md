# Moderation

The goal is to use machine learning models to filter out hateful and grossly obscene speech within the platform. 

## File Structure

- data
  - Stores training data
  - Both raw and processed files are saved
- images
  - Stores all images used throughout repo
- models
  - Stores byte files of trained model object and feature vectors
- notebooks
  - Stores all notebooks including EDA and model training stage.
- pipeline
  - Stores files where models are combined into a voting classifier for better predictions