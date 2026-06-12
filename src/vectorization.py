from sklearn.feature_extraction.text import TfidVectorizer

vectorizer = TfidVectorizer(
  lowercase = True, 
  ngram_range = (1, 2), 
  min_df = 1,
  max_df = 0.95
)
