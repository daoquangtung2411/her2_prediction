
# Harnessing Machine Learning and Advanced Molecular Simulations for Discovering Novel HER2 Inhibitors

This repository contains a machine learning and deep learning framework designed to discover novel HER2 inhibitors by leveraging molecular simulations and machine learning techniques. It can be used to reproduce our results and predict novel HER2 inhibitors from SMILES strings.

## Requirements

To run the code, you will need to install the necessary libraries from the `requirements.txt` file. This can be done with the following command:

```bash
pip install -r requirements.txt
```

Alternatively, if you're only interested in a minimal installation, the following libraries are required:

- `scikit-learn`
- `tabulate`
- `numpy`
- `pandas`
- `xgboost`
- `pickle`

## Reproducing Results

To reproduce our best results (XGBoost model), simply run the following command:

```bash
python train_ml_model.py
```

### Customizing Training with Different Datasets or Target Variables

If you want to train the model with a different dataset or target variable, you can replace the training and test set paths, adjust the machine learning model, or modify the configuration files in the `ml_config` directory.

To do this, use the following command structure:

```bash
python train_ml_model.py --dataset DATASET_PATH --test_size TRAIN_TEST_SPLIT_SIZE --random_state RANDOM_STATE_TO_REPRODUCE --train TRAIN_PATH --test TEST_PATH --target TARGET_COLUMNS --smiles SMILES_COLUMNS --ecfp_radius RADIUS_FOR_ECFP_FINGERPRINT --ecfp_bits NUMBER_OF_ECFP_BITS --zero_threshold PERCENTAGE_OF_REMOVE_NOISE_WITH_MOSTLY_ZERO --correlation_threshold THRESHOLD_FOR_REMOVE_COLLINEAR_COL --model ML_MODELS_[rf_RANDOMFOREST_xgb_XGBOOST_lr_LINEARREGRESSION] --save_model MODEL_NAME_TO_SAVE
```

Where:
- `DATASET_PATH`: Path to the complete dataset (optional).
- `TRAIN_PATH`: Path to the training dataset.
- `TEST_PATH`: Path to the testing dataset.
- `TARGET_COLUMNS`: The target variable columns in the dataset (default is `pIC50`).
- `SMILES_COLUMNS`: The column containing the SMILES strings.
- `RADIUS_FOR_ECFP_FINGERPRINT`: Radius for ECFP fingerprint (default is `2`).
- `NUMBER_OF_ECFP_BITS`: Number of ECFP bits (default is `2048`).
- `PERCENTAGE_OF_REMOVE_NOISE_WITH_MOSTLY_ZERO`: Threshold for removing noisy features (default is `0.95`).
- `THRESHOLD_FOR_REMOVE_COLLINEAR_COL`: Correlation threshold for removing collinear columns (default is `0.70`).
- `ML_MODELS_[rf_RANDOMFOREST_xgb_XGBOOST_lr_LINEARREGRESSION]`: Choose the model to train (`rf` for Random Forest, `xgb` for XGBoost, `lr` for Linear Regression).
- `MODEL_NAME_TO_SAVE`: The name to save the trained model.

### Training a Multilayer Perceptron (MLP/ANN) Model

If you wish to train a Multilayer Perceptron (MLP) or an Artificial Neural Network (ANN), use the following command:

```bash
python train_dl_model.py --dataset DATASET_PATH --test_size TRAIN_TEST_SPLIT_SIZE --random_state RANDOM_STATE_TO_REPRODUCE --train TRAIN_PATH --test TEST_PATH --target TARGET_COLUMNS --smiles SMILES_COLUMNS --ecfp_radius RADIUS_FOR_ECFP_FINGERPRINT --ecfp_bits NUMBER_OF_ECFP_BITS --zero_threshold PERCENTAGE_OF_REMOVE_NOISE_WITH_MOSTLY_ZERO --correlation_threshold THRESHOLD_FOR_REMOVE_COLLINEAR_COL --dense NUMBER_OF_HIDDEN_LAYER --units NUMBER_OF_HIDDEN_LAYER_NODE --learning_rate LEARNING_RATE --batch_size BATCH_SIZE_TO_SPLIT_WHEN_TRAIN_MODEL --activation ACTIVATION_FUNCTION --save_model NAME_OF_MODEL_TO_SAVE
```

Where:
- `NUMBER_OF_HIDDEN_LAYER`: The number of hidden layers in the neural network.
- `NUMBER_OF_HIDDEN_LAYER_NODE`: The number of nodes per hidden layer.
- `LEARNING_RATE`: Learning rate for training the model.
- `BATCH_SIZE_TO_SPLIT_WHEN_TRAIN_MODEL`: Batch size to use when splitting data for training.
- `ACTIVATION_FUNCTION`: Activation function for the network (e.g., `relu`, `sigmoid`).

## Running Prediction

### Predicting from Single SMILES

To make predictions on a single SMILES string, run the following command:

```bash
python predict_her2.py --smiles SMILES
```

Where `SMILES` is the SMILES string you want to predict on (e.g., `CC[C@H]1COC(=N1)NC2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)OCC5=NC=CS5)Cl`).

### Predicting from a File with Multiple SMILES

If you want to predict multiple compounds from a CSV file containing SMILES, use the following command:

```bash
python predict_her2.py --file FILE_TO_PREDICT_PATH --smiles_col SMILES_COLUMN
```

Where:
- `FILE_TO_PREDICT_PATH`: Path to the file containing SMILES.
- `SMILES_COLUMN`: The name of the column in the file that contains SMILES strings.

### Output

The predictions will be printed as a table with columns `SMILES` and `pred_IC50`. The results will be saved in a CSV file with the timestamp as a prefix to predict_result folder.

### Virtual screening result

You can download our virtual screening result via link: https://drive.google.com/file/d/1eD3iGVt5n3JCfu3afmXO32jLp1d2BguO/view?usp=share_link

---

## License

This code is licensed under the BSD-3-Clause - see the [LICENSE](LICENSE) file for details.
