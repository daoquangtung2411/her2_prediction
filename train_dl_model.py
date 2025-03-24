import numpy as np
import pandas as pd
import argparse
import random
from helper.preprocess_dataset import split_dataset, convert_number
from helper.fp_gen import smiles_to_ecfp, smiles_to_maccs
from helper.features_selection import remove_noise_columns, remove_correlation
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tabulate import tabulate
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import pickle

"""

Define argument for train model

- train: train dataset, contain target variables and features
- test: test dataset, contain target variables and features
- target variables: define target columns (default=pIC50)
- smiles columns: define smiles columns (default=SMILES)

"""

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)  # For TensorFlow

parser = argparse.ArgumentParser(description='Train machine learning model to predict compound characteristics')
parser.add_argument("--dataset", type=str, help="Path to unsplit dataset (csv format)")
parser.add_argument("--test_size", type=float, default=0.2, help="Size of test set (float from 0 to 1)")
parser.add_argument("--random_state", type=int, default=42, help="Set this to get reproducible results, default 42")
parser.add_argument("--train", type=str, default='dataset/train_set.csv', help="Path to train dataset (csv format)")
parser.add_argument("--test", type=str, default='dataset/test_set.csv', help="Path to test dataset (csv format)")
parser.add_argument("--target", type=str, default='pIC50')
parser.add_argument("--smiles", type=str, default='SMILES')
parser.add_argument("--fingerprint", type=str, default='ecfp')
parser.add_argument("--ecfp_radius", type=int, default=2)
parser.add_argument("--ecfp_bits", type=int, default=2048)
parser.add_argument("--zero_threshold", type=float, default=0.95)
parser.add_argument("--correlation_threshold", type=float, default=0.70)
parser.add_argument("--dense", type=int, default=2)
parser.add_argument("--units", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--activation", type=str, default='relu')
parser.add_argument("--save_model", type=str, default='ann_model')


args = parser.parse_args()

print('* Loading dataset...')

# Define dataset
if args.dataset is None:
	train = pd.read_csv(args.train)
	test = pd.read_csv(args.test)
else:
	train, test = split_dataset(pd.read_csv(args.dataset), test_size=args.test_size, random_state=args.random_state)

print(f'  | Train size: {len(train)}')
print(f'  | Test size: {len(test)}')
print('\n')

# Define target variables

# y_train = pd.DataFrame(train[args.target])
# y_test = pd.DataFrame(test[args.target])

y_train = train[args.target]
y_test = test[args.target]

# Generate fingerprint

print('* Generating fingerprint...')
print('\n')

# Define generate fingerprints function

X_train_ecfp = train[args.smiles].apply(smiles_to_ecfp, radius=args.ecfp_radius, n_bits=args.ecfp_bits)
X_train_ecfp_df = pd.DataFrame(X_train_ecfp.tolist(), columns=[f'desc_ecfp_{i}' for i in range(len(X_train_ecfp.iloc[0]))])
X_test_ecfp = test[args.smiles].apply(smiles_to_ecfp, radius=args.ecfp_radius, n_bits=args.ecfp_bits)
X_test_ecfp_df = pd.DataFrame(X_test_ecfp.tolist(), columns=[f'desc_ecfp_{i}' for i in range(len(X_test_ecfp.iloc[0]))])

X_train_maccs = train[args.smiles].apply(smiles_to_maccs)
X_train_maccs_df = pd.DataFrame(X_train_maccs.tolist(), columns=[f'desc_maccs_{i}' for i in range(len(X_train_maccs.iloc[0]))])
X_test_maccs = test[args.smiles].apply(smiles_to_maccs)
X_test_maccs_df = pd.DataFrame(X_test_maccs.tolist(), columns=[f'desc_maccs_{i}' for i in range(len(X_test_maccs.iloc[0]))])

X_train_emm_df = pd.concat([X_train_ecfp_df.astype(int), X_train_maccs_df], axis=1)
X_test_emm_df = pd.concat([X_test_ecfp_df.astype(int), X_test_maccs_df], axis=1)

# Apply features selection

print('* Apply features selection...')

X_train_emm_df_fil = remove_noise_columns(X_train_emm_df, threshold=args.zero_threshold)

X_train_emm_df_fil = remove_correlation(X_train_emm_df_fil, threshold=args.correlation_threshold)

features_size = X_train_emm_df_fil.shape[1]

print('\n')
print(f'  | Feature size after selection: {features_size}')
print('\n')
# Train model

print(f'Start train Multilayer Perception model...\n')
print(f'Train Multilayer Perceptron model with following parameters:')

model_params = ["dense", "units", "activation", "learning_rate", "batch_size"]

# Print model parameters in a table
table_data = [(key, getattr(args, key)) for key in model_params]
print(tabulate(table_data, headers=["Parameter", "Value"], tablefmt="grid"))

# Define model and add input layer

model = Sequential([
	Input(shape=(features_size,)),
	Dense(units=1, activation=args.activation)
	])

# Add hidden layer
for _ in range(args.dense):
	model.add(Dense(units=args.dense, activation=args.activation))

# Add output layer

model.add(Dense(units=1, activation='linear'))

# Add optimizer

optimizer = Adam(learning_rate=args.learning_rate)

# Compile model

model.compile(optimizer=optimizer, loss='mean_squared_error')

history = model.fit(X_train_emm_df_fil, y_train, validation_split=0.2, epochs=100, batch_size=args.batch_size, verbose=0)
print('\n')
print(f'Train model Multilayer Perceptron successfully. Calculating performance...')
print('\n')
X_test_emm_df_fil = X_test_emm_df[X_train_emm_df_fil.filter(like='desc_').columns]

y_train_hat = model.predict(X_train_emm_df_fil, verbose=0)
y_test_hat = model.predict(X_test_emm_df_fil, verbose=0)

r2_train = r2_score(y_train, y_train_hat)
r2_test = r2_score(y_test, y_test_hat)

mse_train = mean_squared_error(y_train, y_train_hat)
mse_test = mean_squared_error(y_test, y_test_hat)

rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

mae_train = mean_absolute_error(y_train, y_train_hat)
mae_test = mean_absolute_error(y_test, y_test_hat)

print(f'Model Multilayer Perceptron metrics:')

headers = ["Metrics", "Train", "Test"]

data = [
	["R2", round(r2_train,2), round(r2_test,2)],
	["MSE", round(mse_train,2), round(mse_test,2)],
	["RMSE", round(rmse_train,2), round(rmse_test,2)],
	["MAE", round(mae_train,2), round(mae_test,2)]
]

print(tabulate(data, headers=headers, tablefmt="grid"))

print('\n')


print(f'Saving model to file name {args.save_model}.pkl')

with open(f'{args.save_model}.pkl', 'wb') as model_file:

	pickle.dump(model, model_file)

print(f'Save model successfully to file {args.save_model}.pkl')




