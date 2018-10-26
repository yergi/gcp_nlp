import pandas as pd

original_data = pd.read_csv('happydb/cleaned_hm.csv')

# Export training data
test_data = original_data[['cleaned_hm', 'ground_truth_category']].dropna()
test_data.to_csv('happydb/test_data.csv', index=False, header=False)

# Export only text
text_only = original_data.cleaned_hm
text_only.to_csv('happydb/text_only.csv', index=False, header=False)
