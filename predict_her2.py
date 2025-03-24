import numpy as np
import pandas as pd
import argparse
import pickle
from xgboost import XGBRegressor
from helper.fp_gen import smiles_to_ecfp, smiles_to_maccs
from tabulate import tabulate
import datetime
import os

"""

Define argument for predict HER2 pIC50

- train: train dataset, contain target variables and features
- test: test dataset, contain target variables and features
- target variables: define target columns (default=pIC50)
- smiles columns: define smiles columns (default=SMILES)

"""
parser = argparse.ArgumentParser(description='Train machine learning model to predict compound characteristics')
parser.add_argument("--smiles", type=str, help="SMILES to predict")
parser.add_argument("--file", type=str, help="List of SMILES to predict")
parser.add_argument("--smiles_col", type=str, default='SMILES')
parser.add_argument("--save_dir", type=str, default='predict_result')


args = parser.parse_args()

xgb_model = pickle.load(open('pretrained_model/xgb_model.pkl', 'rb'))

features_col = np.load('helper/features_col.npy', allow_pickle=True)

if args.smiles:
	pred_ecfp = smiles_to_ecfp(args.smiles, radius=2, n_bits=2048)
	pred_ecfp_df = pd.DataFrame([pred_ecfp], columns=[f'desc_ecfp_{i}' for i in range(len(pred_ecfp))])
	pred_maccs = smiles_to_maccs(args.smiles)
	pred_maccs_df = pd.DataFrame([pred_maccs], columns=[f'desc_maccs_{i}' for i in range(len(pred_maccs))])
	pred_emm_df = pd.concat([pred_ecfp_df.astype(int), pred_maccs_df], axis=1)

elif args.file:
	pred = pd.read_csv(args.file)
	pred_ecfp = pred[args.smiles_col].apply(smiles_to_ecfp, radius=2, n_bits=2048)
	pred_ecfp_df = pd.DataFrame(pred_ecfp.tolist(), columns=[f'desc_ecfp_{i}' for i in range(len(pred_ecfp.iloc[0]))])
	pred_maccs = pred[args.smiles_col].apply(smiles_to_maccs)
	pred_maccs_df = pd.DataFrame(pred_maccs.tolist(), columns=[f'desc_maccs_{i}' for i in range(len(pred_maccs.iloc[0]))])
	pred_emm_df = pd.concat([pred_ecfp_df.astype(int), pred_maccs_df], axis=1)

else:
	raise ValueError('Provide SMILES or SMILES file to predict')

X_predict = pred_emm_df[features_col]
pred_val = xgb_model.predict(X_predict)

if args.smiles:
	df = pd.DataFrame({"SMILES": [args.smiles], "pred_IC50": pred_val})
else:
	df = pd.DataFrame({"SMILES": pred["SMILES"], "pred_IC50": pred_val})

# Print as a Table
print(tabulate(df, headers="keys", tablefmt="grid"))

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

save_dir = args.save_dir if args.save_dir else "predict_result"
os.makedirs(save_dir, exist_ok=True)

# Save to CSV
save_path = f'{args.save_dir}/{timestamp}_prediction.csv'
df.to_csv(save_path, index=False)

print(f"Saved prediction to file {save_path}")
