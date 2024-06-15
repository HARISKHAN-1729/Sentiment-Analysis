import pandas as pd

def load_and_prepare_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        if 'Review Text' in data.columns:
            data_cleaned = data.dropna(subset=['Review Text']).copy()
            print("Missing 'Review Text' rows dropped.")
        else:
            print("Error: 'Review Text' column not found.")
            return None
        if 'Recommended IND' in data.columns:
            data_cleaned.loc[:, 'not_recommended'] = 1 - data_cleaned.loc[:, 'Recommended IND']
            print("'not_recommended' column created.")
        else:
            print("Error: 'Recommended IND' column not found.")
            return None
        column_rename_mapping = {
            'Clothing ID': 'clothing_id',
            'Age': 'age',
            'Title': 'title',
            'Review Text': 'review_text',
            'Rating': 'rating',
            'Recommended IND': 'recommended',
            'Positive Feedback Count': 'positive_feedback_count',
            'Division Name': 'division_name',
            'Department Name': 'department_name',
            'Class Name': 'class_name'
        }
        data_cleaned.rename(columns=column_rename_mapping, inplace=True)
        print("Columns renamed.")
        return data_cleaned
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
