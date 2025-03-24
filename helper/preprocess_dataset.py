from sklearn.model_selection import train_test_split

def split_dataset(dataset, test_size=0.2, random_state=42):

	train, test = train_test_split(dataset, test_size=test_size, random_state=random_state)

	return train, test

def convert_number(value):
    try:
        return int(value) if value.isdigit() else float(value) if value.replace('.', '', 1).isdigit() else value
    except ValueError:
        return value