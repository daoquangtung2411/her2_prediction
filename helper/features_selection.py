import numpy as np

def remove_noise_columns(df, threshold=0.95):

	"""

	- Since the binary fingerprints being used, some of the columns will have mostly zero values, 
	which can consider as noise

	- This function will help remove columns contain more zeros values than threshold.

	- Input:
		df: DataFrame of features
		threshold: remove columns with more zeros values than threshold

	"""

	features_col = df.filter(like='desc_').columns
	mean_zero = (df[features_col] == 0).mean()
	column_to_drop = mean_zero[mean_zero > threshold].index

	return df.drop(columns = column_to_drop)

def remove_correlation(df, threshold=0.7):
	"""

	- Binary features might return more than one columns that has high correlation, so remove correlation might help
	model to learn better
	- The original threshold was set at 0.7

	- Input:
		df: DataFrame of features
		threshold: the correlation coefficient set to remove multicolinear columns

	"""

	features_col = df.filter(like='desc_').columns
	corr_matrix = df[features_col].corr().abs()
	upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
	columns_to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

	return df.drop(columns=columns_to_drop)