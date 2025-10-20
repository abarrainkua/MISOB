## Practical implementation of MISOB 

import numpy as np
import pandas as pd
from carla.models.catalog import  MLModelCatalog
from carla.data.catalog import OnlineCatalog, CsvCatalog, DataCatalog
from carla.recourse_methods import GrowingSpheres, ActionableRecourse, CCHVAE, Wachter, Face
from carla import RecourseMethod
from carla.models.negative_instances import predict_negative_instances
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.utils import resample
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import sys
from sklearn.utils import resample
import torch.nn as nn
import torch
import itertools
import argparse


def train_social_burden(dataset='adult', file_path=None, sens_attr=['race'], lr=0.001, epochs=10, batch_size=256, base_model="ann",
                        hidden_sizes=[128, 128], activation_name="relu", verbose=True, pretrain_epochs=2, save_training_metrics=True,
                        n_inst_eval_train_metrics=100, recourse_method="GS", recourse_hyperparam={}, 
                        results_file=None, random_state=42):
    
    """
    Train a base classifier under the MISOB framework.

    This function trains a base classifier on a given dataset, with the possibility of tracking performance 
    and fairness metrics (sensitive information needs to be known for this).
    Parameters:
    ----------
    dataset : str, default='adult'
        The name of the dataset to use.
    
    file_path : str or None
        Path to load the dataset. 

    sens_attr : list of str, default=['race']
        List of sensitive attribute(s) to evaluate fairness metrics across.
    
    lr : float, default=0.001
        Learning rate for the optimizer.
    
    epochs : int, default=10
        Number of training epochs for the whole training loop.

    batch_size : int, default=256
        Batch size used during training.

    base_model : str, default='ann'
        Type of base model to train. Supported options: 'ann' (artificial neural network), 'linear' (linear regression).
    
    hidden_sizes : list of int, default=[128, 128]
        Sizes of hidden layers for the ANN model. Ignored if base_model is not 'ann'.

    activation_name : str, default='relu'
        Activation function to use in the ANN model (e.g., 'relu', 'tanh').

    verbose : bool, default=True
        Whether to print training progress and evaluation summaries.
    
    pretrain_epochs : int, default=5
        Number of warm-up epochs before applying the recourse method.

    save_training_metrics : bool, default=True
        Whether to log and save performance and fairness metrics throughout training.

    n_inst_eval_train_metrics : int, default=100
        Number of instances used to evaluate training metrics (sampling from the training set).

    recourse_method : str, default='GS'
        Recourse method under consideration in MISOB (e.g., 'GS', 'MISOB', 'POSTPRO').

    recourse_hyperparam : dict, default={}
        Hyperparameters specific to the chosen recourse method.

    results_file : str or None
        Path to save the results and metrics. If None, results are not saved to disk.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns:
    -------
    
    learner : Pytorch model
        The resulting classifier.
    
    metrics_df : pd.Dataframe
        Dataframe containing training metrics.
    
    ml_model : MLModelCatalog from CARLA library
        The resulting classifier.

    """
    
    
     # ---- Load training data ----
    
    if dataset == "adult":
        
        continuous = ["age", "fnlwgt", "education-num", "capital-gain", "hours-per-week", "capital-loss"]
        categorical = ["marital-status", "native-country", "occupation", "race", "relationship", "sex", "workclass"]
        immutable = ["age", "sex", "race"]
        y_var = "income"
        
        # The sensitive attribute mapping
        mapping = {
            "race": "race_White",
            "sex": "sex_Male",
            "age": "age_bin"
        }
        
    elif dataset == "givemesomecredit":
        continuous = ["RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio",
                      "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate", 
                      "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"]
        categorical = []
        immutable = ["age"]
        y_var = 'no_dlq'
        
        # The sensitive attribute mapping
        mapping = {
            "age": "age_bin"
        }
    
    
    elif dataset == "credit":
        categorical = ["status", "credit_history", "purpose", "savings", "employment", "sex", "other_debtors", "property", "age_bin", "installment_plans", "housing", "skill_level", "telephone", "foreign_worker"]
        continuous = ["month", "credit_amount", "investment_as_income_percentage", "residence_since", "number_of_credits", "people_liable_for"]
        immutable = ["age_bin", "sex"]
        y_var = "credit"
        
        # The sensitive attribute mapping
        mapping = {
            "sex": "sex_1",
            "age_bin": "age_bin_1"
        }
        
        
    # Create the corresponding s_var list
    s_var = [mapping[attr] for attr in sens_attr]

    dataset_train = CsvCatalog(file_path=file_path,
                        continuous=continuous,
                        categorical=categorical,
                        immutables=immutable,
                        target=y_var)
    
    
    # ----- Store the sensitive information (to obtain training metrics) ------

    data_no_preprocess = pd.read_csv(file_path) # necessary to do this for the attribute age (to have orginal without processing)
    s_vals_list = []
    for attr, var in zip(sens_attr, s_var):
        if attr == "age":
            # Apply binarization for age directly from raw data
            s_col = (data_no_preprocess["age"] > 30).astype(int).to_numpy()
        else:
            # Use the processed column from the dataset
            s_col = dataset_train.df[var].to_numpy()
        
        s_vals_list.append(s_col)

    # Stack the columns horizontally to get a 2D matrix
    s_vals = np.column_stack(s_vals_list)
    
    # Augment s_vals with all the possible intersectional groups
    df_sens = pd.DataFrame(s_vals, columns=s_var)
    augmented_df = df_sens.copy()

    # Add intersectional combinations
    for r in range(2, len(s_var) + 1):
        for combo in itertools.combinations(s_var, r):
            combo_name = "_&_".join(combo)
            group_ids, _ = pd.factorize(list(zip(*(df_sens[col] for col in combo))))
            augmented_df[combo_name] = group_ids
            
    # Convert again to matrix
    s_vals = augmented_df.to_numpy()
    
    # Get all the sensitive groupings
    s_column_names = augmented_df.columns
    
    
    # ----- Specifications of the base model and the recourse method ----

    # Map from string to PyTorch activation class
    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU
    }
    
    if activation_name.lower() not in activation_map:
        raise ValueError(f"Unsupported activation: {activation_name}. Supported: {list(activation_map.keys())}")
    
    activation = activation_map[activation_name.lower()]
    
    # Map from string to recourse method
    recourse_map = {
        "GS": GrowingSpheres,
        "AR": ActionableRecourse,
        "CCHVAE": CCHVAE,
        "WT": Wachter,
        "FACE": Face
    }
    
    if recourse_method.upper() not in recourse_map:
        raise ValueError(f"Unsupported recourse method: {recourse_method}. Supported: {list(recourse_map.keys())}")
    
    recourse_model_obj = recourse_map[recourse_method.upper()]
    
    # Generate MLModelCatalog object from CARLA
    ml_model = MLModelCatalog(
            dataset_train,
            model_type=base_model,
            load_online=False,
            backend="pytorch"
            )


    if base_model == "ann":

        training_params = {"lr": lr, "epochs": 1, "batch_size": batch_size,
                        "hidden_size": hidden_sizes}

        # Initialize MLModelCatalog object from CARLA
        ml_model.train(
            learning_rate=training_params["lr"],
            epochs=training_params["epochs"],
            batch_size=training_params["batch_size"],
            hidden_size=training_params["hidden_size"],
            force_train=True
        )

    elif base_model == "linear":

        training_params = {"lr": lr, "epochs": 1, "batch_size": batch_size}

        # Initialize MLModelCatalog object from CARLA
        ml_model.train(
            learning_rate=training_params["lr"],
            epochs=training_params["epochs"],
            batch_size=training_params["batch_size"],
            force_train=True
        )

    
    if recourse_method == "CCHVAE":
        recourse_hyperparam = {
        "data_name": f"{dataset}_{sens_attr}",
        "n_search_samples": 100,
        "p_norm": 1,
        "step": 0.1,
        "max_iter": 1000,
        "clamp": True,
        "binary_cat_features": True,
        "vae_params": {
            "layers": [
                # (len(continuous) + len(categorical) - len(immutable)), 
                len(ml_model.feature_input_order) - len(immutable),
                64, 
                32, 
                8
            ],
            "train": True,
            "lambda_reg": 1e-6,
            "epochs": 5,
            "lr": 1e-3,
            "batch_size": 128,
        },
    }
    
    
    # The loss function of the model
    loss_learner = nn.CrossEntropyLoss(reduction="none")
    
    # Get the underlying model
    learner = ml_model._model.to(device)
    
    # The optimizer of the learner
    optimizer_learner = torch.optim.SGD(learner.parameters(), lr=lr)
    
    # List to store the performance metrics
    metrics_log = []
    
    
    # ----- Start the training process ---
    
    for epoch in range(epochs):
        
        # Define the batches of the epoch
        dataset_len = len(dataset_train.df)
        batch_permu = np.random.permutation(dataset_len)
        batch_indices = np.array_split(batch_permu, np.ceil(dataset_len / batch_size))
        
        for current_batch in batch_indices:
            
            instance_weights = torch.ones(len(current_batch), requires_grad=True).to(device)
            
            # Create dataframe with current batch
            train_batch_df = dataset_train.df.iloc[current_batch]
            
            X_batch = train_batch_df.drop(columns=[y_var]).to_numpy()
            y_batch = train_batch_df[y_var].to_numpy()
            s_batch = s_vals[current_batch]
            
            X_batch = torch.from_numpy(X_batch).to(torch.float32).to(device)
            y_batch = torch.from_numpy(preprocessing.LabelEncoder().fit_transform(y_batch)).to(device)
            s_batch = torch.from_numpy(s_batch).to(torch.int32).to(device)
            
            
            # If we are past the warm-up perform the re-weighing process
            if epoch >= pretrain_epochs: 
                
                # Initialize the recourse method
                recourse_m = recourse_model_obj(ml_model, recourse_hyperparam)

                #Get predictions for test instances for current model
                y_pred_scores = ml_model.predict(train_batch_df) # here to device?

                # Binarize predictions
                y_pred_bin = (y_pred_scores > .5).astype(int).reshape(1,-1)[0] 
                
                # Get instances that will be subject to recourse
                factuals = train_batch_df[y_pred_bin == 0] 
                
                if factuals.size > 0:
                    counterfactuals = recourse_m.get_counterfactuals(factuals)
                    # See if there is any nan element
                    nan_indices = counterfactuals[counterfactuals.isna().any(axis=1)].index

                    # Eliminate those instances with nan entry
                    counterfactuals = counterfactuals.drop(index=nan_indices)
                    factuals = factuals.drop(index=nan_indices)
                else:
                    counterfactuals = factuals.copy()  # Empty dataframe
                
                # Create dataframe with new representations, after recourse    
                train_batch_new = train_batch_df.copy()
                factual_indices = factuals.index  # index of factuals
                cf_columns = counterfactuals.columns
                train_batch_new.loc[factual_indices, cf_columns] = counterfactuals.values  # replace by counterfactuals
                
                # Convert dataframes into numpy array
                old_test_array = train_batch_df.to_numpy()
                new_test_array = train_batch_new.to_numpy()
                
                # Compute recourse costs for each instance in the batch
                recourse_costs = np.linalg.norm(new_test_array - old_test_array, axis=1)
                 
                # Start building the full recourse DataFrame from the sensitive information
                recourse_df = pd.DataFrame(
                    s_batch.cpu().detach().numpy(),  # sensitive attribute values
                    columns=s_column_names  # both one-dimensional and intersectional names
                )

                # Add outcome and cost information
                recourse_df["y_true"] = train_batch_df[y_var].to_numpy()
                recourse_df["cost"] = recourse_costs
                recourse_df["burden"] = np.where(recourse_df["y_true"] == 0, 0, recourse_costs)

                # Update instance weight based on burden
                social_burden_tensor = torch.tensor(recourse_df["burden"].to_numpy(), dtype=torch.float32)
                total_burden = social_burden_tensor.sum()
                epsilon = 1e-8  # small value to prevent division by zero
                scaling = len(current_batch) * 0.2 / (total_burden + epsilon)
                instance_weights = 1 + scaling * social_burden_tensor
                
                instance_weights = instance_weights.to(device)
                
            
            # Get the underlying model
            learner = ml_model._model
            
            # Get learner loss value
            loss_value_learner = loss_learner(learner(X_batch), y_batch).to(device)
            weighted_loss_learner = loss_value_learner * instance_weights
            weighted_loss_learner = torch.mean(weighted_loss_learner)

            # Gradient step    
            optimizer_learner.zero_grad()
            weighted_loss_learner.backward()
            optimizer_learner.step()

            # Update ML model
            ml_model._model = learner
        
        # If save_training_metrics = True, Get performance stats for the model at this point of the training
        if save_training_metrics:
        
            with torch.no_grad():
                # Full training data
                train_df = dataset_train.df.copy()
                # Get a subsample of n_inst_eval_train_metrics instances
                subset_train_df = train_df.sample(n=n_inst_eval_train_metrics, random_state=random_state+99*epoch)
                # Save which instances have been selected
                subset_indices = subset_train_df.index
                X_train_full = subset_train_df.drop(columns=[y_var]).to_numpy()
                y_train_full = subset_train_df[y_var].to_numpy()
                s_train_full = s_vals[subset_indices]

                # Get model predictions
                X_train_tensor = torch.from_numpy(X_train_full).to(torch.float32).to(device)
                y_pred_scores = ml_model._model(X_train_tensor).cpu().detach().numpy()[:,1]
                y_pred_bin = (y_pred_scores > 0.5).astype(int)

            # Accuracy
            y_encoded = preprocessing.LabelEncoder().fit_transform(y_train_full)
            train_accuracy = np.mean(y_pred_bin == y_encoded)

            y_true = y_encoded 
            s_groups = s_train_full

            # Initialize dictionaries to store metrics
            acc_by_group = {}
            tpr_by_group = {}
            fpr_by_group = {}
            ar_by_group = {}

            for i, col_name in enumerate(s_column_names):
                s_col = s_groups[:, i]  # extract the i-th sensitive attribute column

                for group_val in np.unique(s_col):
                    idx = s_col == group_val
                    y_true_group = y_true[idx]
                    y_pred_group = y_pred_bin[idx]

                    # Build key using actual values for each attribute
                    if "_&_" in col_name:
                        attrs = col_name.split("_&_")
                        example_idx = np.where(idx)[0][0]  # get one matching row index
                        full_label_parts = []

                        for attr in attrs:
                            attr_index = s_column_names.get_loc(attr)
                            attr_val = s_groups[example_idx, attr_index]
                            full_label_parts.append(f"{attr}_{attr_val}")
                        key_prefix = "_&_".join(full_label_parts)
                    else:
                        key_prefix = f"{col_name}_{group_val}"

                    # Compute and store metrics using the full label
                    acc_by_group[f"acc_{key_prefix}"] = np.mean(y_pred_group == y_true_group)

                    positives = y_true_group == 1
                    tpr = np.sum((y_pred_group == 1) & positives) / (np.sum(positives) + 1e-8)
                    tpr_by_group[f"tpr_{key_prefix}"] = tpr

                    negatives = y_true_group == 0
                    fpr = np.sum((y_pred_group == 1) & negatives) / (np.sum(negatives) + 1e-8)
                    fpr_by_group[f"fpr_{key_prefix}"] = fpr

                    ar = np.mean(y_pred_group == 1)
                    ar_by_group[f"ar_{key_prefix}"] = ar
                    
            # Recompute counterfactuals to evaluate burden
            factuals = subset_train_df[y_pred_bin == 0]
            recourse_m = recourse_model_obj(ml_model, recourse_hyperparam)
            
            if factuals.size > 0:
                counterfactuals = recourse_m.get_counterfactuals(factuals)
                # See if there is any nan element
                nan_indices = counterfactuals[counterfactuals.isna().any(axis=1)].index

                # Eliminate those instances with nan entry
                counterfactuals = counterfactuals.drop(index=nan_indices)
                factuals = factuals.drop(index=nan_indices)
            else:
                counterfactuals = factuals.copy()  # Empty dataframe

            train_new_df = subset_train_df.copy()
            factual_indices = factuals.index
            train_new_df.loc[factual_indices, counterfactuals.columns] = counterfactuals.values

            old_array = subset_train_df.to_numpy()
            new_array = train_new_df.to_numpy()
            recourse_costs = np.linalg.norm(new_array - old_array, axis=1)
            
            # Start building the full recourse DataFrame from the sensitive information
            recourse_info = pd.DataFrame(
                s_train_full,  # sensitive attribute values
                columns=s_column_names  # both one-dimensional and intersectional names
            )

            # Add outcome and cost information
            recourse_info["y_true"] = y_train_full
            recourse_info["cost"] = recourse_costs
            recourse_info["burden"] = np.where(np.array(y_train_full) == 0, 0, recourse_costs)
            
            cost_by_group_all = {}
            burden_by_group_all = {}
            cost_gap_all = {}
            burden_gap_all = {}

            for col in s_column_names:
                # Compute group-wise means for cost and burden
                cost_by_group = recourse_info.groupby(col)["cost"].mean().to_dict()
                burden_by_group = recourse_info.groupby(col)["burden"].mean().to_dict()

                # If the column is intersectional, decode the values
                if "_&_" in col:
                    # Parse original attribute names
                    attrs = col.split("_&_")

                    # Decode each unique group into readable attribute-value pairs
                    for group_id, cost in cost_by_group.items():
                        group_mask = recourse_info[col] == group_id
                        example_row = recourse_info.loc[group_mask].iloc[0]  # any representative row

                        # Create a name like race_White_0.0&_sex_Male_1.0
                        full_label_parts = []
                        for attr in attrs:
                            val = example_row[attr]
                            full_label_parts.append(f"{attr}_{val}")
                        full_label = "_&_".join(full_label_parts)

                        cost_by_group_all[f"cost_{full_label}_group_{group_id}"] = cost
                        burden_by_group_all[f"burden_{full_label}_group_{group_id}"] = burden_by_group[group_id]
                else:
                    # Single-attribute case
                    for group_id, cost in cost_by_group.items():
                        full_label = f"{col}_{group_id}"
                        cost_by_group_all[f"cost_{full_label}_group_{group_id}"] = cost
                        burden_by_group_all[f"burden_{full_label}_group_{group_id}"] = burden_by_group[group_id]

                # Compute gap (max - min)
                cost_vals = list(cost_by_group.values())
                burden_vals = list(burden_by_group.values())
                cost_gap_all[f"cost_gap_{col}"] = np.max(cost_vals) - np.min(cost_vals) if cost_vals else np.nan
                burden_gap_all[f"burden_gap_{col}"] = np.max(burden_vals) - np.min(burden_vals) if burden_vals else np.nan


            # Build dictionary for this epoch
            epoch_metrics = {
                "epoch": epoch,
                "accuracy": train_accuracy
            }
            epoch_metrics.update(cost_gap_all)
            epoch_metrics.update(burden_gap_all)
            epoch_metrics.update(acc_by_group)
            epoch_metrics.update(tpr_by_group)
            epoch_metrics.update(fpr_by_group)
            epoch_metrics.update(ar_by_group)
            epoch_metrics.update(burden_by_group_all)
            epoch_metrics.update(cost_by_group_all)
            
            # Append to log
            metrics_log.append(epoch_metrics)
            
        if verbose:
                print(f"epoch={epoch} loss={weighted_loss_learner}")
        
    metrics_df = pd.DataFrame(metrics_log)
    metrics_df.to_csv(results_file, index=False)

    return learner, metrics_df, ml_model
    

def test_recourse(dataset=None, file_path=None, sens_attr=['race'], ml_model=None,
                        recourse_method="GS", recourse_hyperparam={}, 
                        results_file=None, random_state=42):
    
    
    """
    Evaluate the performance of a trained classifier under a specified recourse method.

    This function tests the final classifier by applying a recourse strategy to individuals in the dataset.
    It evaluates different metrics with respect to the sensitive attribute and stores the results
    if a file path is provided.

    Parameters:
    ----------
    dataset : str or None
        Dataset name.
    
    file_path : str or None
        Path to a CSV containing test data.
    
    sens_attr : list of str, default=['race']
        List of sensitive attribute(s) to evaluate fairness metrics across.
    
    ml_model : MLModelCatalog object from CARLA or None
        Trained machine learning model used for prediction and evaluation.
    
    recourse_method : str, default='GS'
        The name of the recourse method to apply (e.g., 'GS', 'WT', 'CCHVAE').
    
    recourse_hyperparam : dict, default={}
        Dictionary containing hyperparameters specific to the chosen recourse method.
    
    results_file : str or None
        If provided, the evaluation results will be saved to this file path.
    
    random_state : int, default=42
        Random seed for reproducibility.

    Returns:
    -------
    results_df
        A Dataframe containing evaluation metrics.
    """
    
    
    # ---- Load test data ----
    
    if dataset == "adult":
        
        continuous = ["age", "fnlwgt", "education-num", "capital-gain", "hours-per-week", "capital-loss"]
        categorical = ["marital-status", "native-country", "occupation", "race", "relationship", "sex", "workclass"]
        immutable = ["age", "sex", "race"]
        y_var = "income"
        
        # The sensitive attribute mapping
        mapping = {
            "race": "race_White",
            "sex": "sex_Male",
            "age": "age_bin"
        }
        
        
    elif dataset == "givemesomecredit":
        continuous = ["RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio",
                      "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate", 
                      "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"]
        categorical = []
        immutable = ["age"]
        y_var = 'no_dlq'
        
        # The sensitive attribute mapping
        mapping = {
            "age": "age_bin"
        }
        
    elif dataset == "credit":
        categorical = ["status", "credit_history", "purpose", "savings", "employment", "sex", "other_debtors", "property", "age_bin", "installment_plans", "housing", "skill_level", "telephone", "foreign_worker"]
        continuous = ["month", "credit_amount", "investment_as_income_percentage", "residence_since", "number_of_credits", "people_liable_for"]
        immutable = ["age_bin", "sex"]
        y_var = "credit"
        
        # The sensitive attribute mapping
        mapping = {
            "sex": "sex_1",
            "age_bin": "age_bin_1"
        }
        
    # Create the corresponding s_var list
    s_var = [mapping[attr] for attr in sens_attr]
    

    # Load test data to appropriate CARLA class
    dataset_test = CsvCatalog(file_path=file_path,
                        continuous=continuous,
                        categorical=categorical,
                        immutables=immutable,
                        target=y_var)
    
    
    # ----- Store the sensitive information ------
    
    # Create the s_vals matrix with the values of the sensitive attribute for all the characterizations
    data_no_preprocess = pd.read_csv(file_path) # necessary to do this for the attribute age (to have orginal without CARLA-based processing)
    s_vals_list = []
    for attr, var in zip(sens_attr, s_var):
        if attr == "age":
            # Apply binarization for age directly from raw data
            s_col = (data_no_preprocess["age"] > 30).astype(int).to_numpy()
        else:
            # Use the processed column from the dataset
            s_col = dataset_test.df[var].to_numpy()
        
        s_vals_list.append(s_col)

    # Stack the columns horizontally to get a 2D matrix
    s_vals = np.column_stack(s_vals_list)
    
    # Augment s_vals with all the possible intersectional groups
    df_sens = pd.DataFrame(s_vals, columns=s_var)
    augmented_df = df_sens.copy()

    # Add intersectional combinations
    for r in range(2, len(s_var) + 1):
        for combo in itertools.combinations(s_var, r):
            combo_name = "_&_".join(combo)
            group_ids, _ = pd.factorize(list(zip(*(df_sens[col] for col in combo))))
            augmented_df[combo_name] = group_ids
            
    # Convert again to matrix
    s_vals = augmented_df.to_numpy()
    
    # Get all the sensitive groupings
    s_column_names = augmented_df.columns
            
    # Map from string to recourse method
    recourse_map = {
        "GS": GrowingSpheres,
        "CCHVAE": CCHVAE,
        "WT": Wachter
    }
    
    if recourse_method.upper() not in recourse_map:
        raise ValueError(f"Unsupported recourse method: {recourse_method}. Supported: {list(recourse_map.keys())}")
    
    recourse_model_obj = recourse_map[recourse_method.upper()]
    
    if recourse_method == "CCHVAE":
        recourse_hyperparam = {
        "data_name": f"{dataset}_{sens_attr}",
        "n_search_samples": 100,
        "p_norm": 1,
        "step": 0.1,
        "max_iter": 1000,
        "clamp": True,
        "binary_cat_features": True,
        "vae_params": {
            "layers": [
                len(ml_model.feature_input_order) - len(immutable),
                64, 
                32, 
                8
            ],
            "train": True,
            "lambda_reg": 1e-6,
            "epochs": 5,
            "lr": 1e-3,
            "batch_size": 128,
        },
    }
    
    
    # --- Classify test instances and get performance metrics ---
    
    with torch.no_grad():
        
            # Full training data
            test_df = dataset_test.df.copy()
            X_test_full = test_df.drop(columns=[y_var]).to_numpy()
            y_test_full = test_df[y_var].to_numpy()
            s_test_full = s_vals
            
            # Get model predictions
            X_test_tensor = torch.from_numpy(X_test_full).to(torch.float32).to(device)
            y_pred_scores = ml_model._model(X_test_tensor).cpu().detach().numpy()[:,1]
            y_pred_bin = (y_pred_scores > 0.5).astype(int)

    # Accuracy
    y_encoded = preprocessing.LabelEncoder().fit_transform(y_test_full)
    test_accuracy = np.mean(y_pred_bin == y_encoded)

    y_true = y_encoded 
    s_groups = s_test_full

    # Initialize dictionaries to store metrics
    acc_by_group = {}
    tpr_by_group = {}
    fpr_by_group = {}
    ar_by_group = {}

    # Loop through each sensitive attribute column
    for i, col_name in enumerate(s_column_names):
        s_col = s_groups[:, i]  # extract the i-th sensitive attribute column
        
        for group_val in np.unique(s_col):
            idx = s_col == group_val
            y_true_group = y_true[idx]
            y_pred_group = y_pred_bin[idx]

            # Build key using actual values for each attribute
            if "_&_" in col_name:
                attrs = col_name.split("_&_")
                example_idx = np.where(idx)[0][0]  # get one matching row index
                full_label_parts = []

                for attr in attrs:
                    attr_index = s_column_names.get_loc(attr)
                    attr_val = s_groups[example_idx, attr_index]
                    full_label_parts.append(f"{attr}_{attr_val}")
                key_prefix = "_&_".join(full_label_parts)
            else:
                key_prefix = f"{col_name}_{group_val}"

            # Compute and store metrics using the full label
            acc_by_group[f"acc_{key_prefix}"] = np.mean(y_pred_group == y_true_group)

            positives = y_true_group == 1
            tpr = np.sum((y_pred_group == 1) & positives) / (np.sum(positives) + 1e-8)
            tpr_by_group[f"tpr_{key_prefix}"] = tpr

            negatives = y_true_group == 0
            fpr = np.sum((y_pred_group == 1) & negatives) / (np.sum(negatives) + 1e-8)
            fpr_by_group[f"fpr_{key_prefix}"] = fpr

            ar = np.mean(y_pred_group == 1)
            ar_by_group[f"ar_{key_prefix}"] = ar
    
    
    # Recompute counterfactuals to evaluate burden
    factuals = test_df[y_pred_bin == 0]
    recourse_m = recourse_model_obj(ml_model, recourse_hyperparam)
    
    if factuals.size > 0:
        counterfactuals = recourse_m.get_counterfactuals(factuals)
        # See if there is any nan element
        nan_indices = counterfactuals[counterfactuals.isna().any(axis=1)].index
        # Eliminate those instances with nan entry
        counterfactuals = counterfactuals.drop(index=nan_indices)
        factuals = factuals.drop(index=nan_indices)
    else:
        counterfactuals = factuals.copy()  # Empty dataframe

    test_new_df = test_df.copy()
    factual_indices = factuals.index
    test_new_df.loc[factual_indices, counterfactuals.columns] = counterfactuals.values

    old_array = test_df.to_numpy()
    new_array = test_new_df.to_numpy()
    recourse_costs = np.linalg.norm(new_array - old_array, axis=1)
    
    # Start building the full recourse DataFrame from the sensitive information
    recourse_info = pd.DataFrame(
        s_test_full,  # sensitive attribute values
        columns=s_column_names  # both one-dimensional and intersectional names
    )

    # Add outcome and cost information
    recourse_info["y_true"] = y_test_full
    recourse_info["cost"] = recourse_costs
    recourse_info["burden"] = np.where(np.array(y_test_full) == 0, 0, recourse_costs)
    
    cost_by_group_all = {}
    burden_by_group_all = {}
    cost_gap_all = {}
    burden_gap_all = {}

    for col in s_column_names:
        # Compute group-wise means for cost and burden
        cost_by_group = recourse_info.groupby(col)["cost"].mean().to_dict()
        burden_by_group = recourse_info.groupby(col)["burden"].mean().to_dict()

        # If the column is intersectional, decode the values
        for group_id, cost in cost_by_group.items():
            group_mask = recourse_info[col] == group_id
            example_row = recourse_info.loc[group_mask].iloc[0]  # any representative row

            if "_&_" in col:
                attrs = col.split("_&_")
                full_label_parts = []
                for attr in attrs:
                    val = example_row[attr]
                    full_label_parts.append(f"{attr}_{val}")
                key_prefix = "_&_".join(full_label_parts)
            else:
                key_prefix = f"{col}_{group_id}"

            cost_by_group_all[f"cost_{key_prefix}"] = cost
            burden_by_group_all[f"burden_{key_prefix}"] = burden_by_group[group_id]


    ## Initialize result dictionary
    results_dict = {
        "overall_accuracy": {"all": test_accuracy},
    }
    
    # Helper function to extract attr name + group value
    def parse_group_key(key, prefix):
        """
        Extracts the full attribute string after the metric prefix.
        For example:
        key = "acc_race_White_0.0&_sex_Male_1.0", prefix = "acc_"
        returns: "race_White_0.0&_sex_Male_1.0"
        """
        if not key.startswith(prefix):
            raise ValueError(f"Key '{key}' does not start with expected prefix '{prefix}'")
        return key[len(prefix):]  # just strip the prefix and return the rest


    # Generic updater
    def update_nested_dict(metric_name, group_dict, prefix):
        for k, v in group_dict.items():
            group_label = parse_group_key(k, prefix)  # e.g., "race_White_0.0&_sex_Male_1.0"
            results_dict.setdefault(metric_name, {})[group_label] = v
            
    # Group-wise accuracy, TPR, AR
    update_nested_dict("group_accuracy", acc_by_group, prefix="acc_")
    update_nested_dict("group_tpr", tpr_by_group, prefix="tpr_")
    update_nested_dict("group_ar", ar_by_group, prefix="ar_")

    # Cost and burden
    update_nested_dict("group_cost", cost_by_group_all, prefix="cost_")
    update_nested_dict("group_burden", burden_by_group_all, prefix="burden_")

    # Optional: Flatten results_dict for DataFrame conversion
    results_df = pd.json_normalize(results_dict, sep="/").T
    results_df.columns = ["value"]
    results_df.index.name = "metric/group"
    results_df = results_df.sort_index()
    
    results_df.to_csv(results_file, index=True)
    
    return results_df 

def fix_give_me_credit(df: pd.DataFrame):
    
    # Pre-processing from https://github.com/unitn-sml/pear-personalized-algorithmic-recourse/tree/master

    # https://www.kaggle.com/code/simonpfish/comp-stats-group-data-project-final
    df.dropna(inplace=True)
    df.loc[df['DebtRatio'] > 1, 'DebtRatio'] = 1
    df.loc[df['MonthlyIncome'] > 17000, 'MonthlyIncome'] = 17000
    df.loc[df['RevolvingUtilizationOfUnsecuredLines'] > 1, 'RevolvingUtilizationOfUnsecuredLines'] = 1
    dfn98 = df.copy()
    dfn98.loc[dfn98['NumberOfTime30-59DaysPastDueNotWorse'] > 90, 'NumberOfTime30-59DaysPastDueNotWorse'] = 18
    dfn98.loc[dfn98['NumberOfTime60-89DaysPastDueNotWorse'] > 90, 'NumberOfTime60-89DaysPastDueNotWorse'] = 18
    dfn98.loc[dfn98['NumberOfTimes90DaysLate'] > 90, 'NumberOfTimes90DaysLate'] = 18
    return dfn98

def parse_args():
    parser = argparse.ArgumentParser()

    # fmt: off

    parser.add_argument("--random-state", type=int, default=1234)
    parser.add_argument("--dataset-name", type=str, default="adult", help="The name of the dataset")
    parser.add_argument("--sens-attr", default=["race", "sex"], nargs='?', help="The sensitive attribute(s)")
    parser.add_argument("--test-size", type=float, default=0.3, help="The proportion of points for test.")
    parser.add_argument("--pre-epoch", type=int, default=3, help="The number of pre-train epochs (warm-up).")
    parser.add_argument("--total-epoch", type=int, default=6, help="The number of total epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training the classifier")
    parser.add_argument("--n-eval-train-metrics", type=int, default=100, help="The number of instances in which to evaluate the training metrics.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for training the classifier")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function for the NN")
    parser.add_argument("--train-recourse-method", type=str, default="WT", help="The method for recourse at training.")
    parser.add_argument("--test-recourse-method", type=str, default=None, help="The method for recourse at deployment.")
    parser.add_argument("--hidden-sizes", default=[128, 128], nargs='?', help="The size(s) of the hidden layer(s).")
    parser.add_argument("--fair-strategy", type=str, default="burden", help="The fairness strategy.")
    parser.add_argument("--base-model", type=str, default="ann", help="The base model.")
    
    # fmt: on

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
   
    # --- Load dataset ---
    
    if args.dataset_name == "adult":
        
        df_orig = pd.read_csv(f"{args.dataset_name}.csv")
        
    elif args.dataset_name == "givemesomecredit":
        
        df_orig = pd.read_csv(f"{args.dataset_name}.csv", index_col=0)
        df_orig = fix_give_me_credit(df_orig)
        
        # Balance dataset re. to class label
        
        # Separate the classes
        df_majority = df_orig[df_orig['SeriousDlqin2yrs'] == df_orig['SeriousDlqin2yrs'].value_counts().idxmax()]
        df_minority = df_orig[df_orig['SeriousDlqin2yrs'] != df_orig['SeriousDlqin2yrs'].value_counts().idxmax()]

        # Downsample majority class
        df_majority_downsampled = resample(df_majority,
                                        replace=False,  # sample without replacement
                                        n_samples=len(df_minority),  # match minority count
                                        random_state=42)  # for reproducibility

        # Combine back into a single DataFrame
        df_balanced = pd.concat([df_minority, df_majority_downsampled])

        # Shuffle the rows
        df_orig = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Get new class label (desired)
        df_orig['no_dlq'] = 1 - df_orig['SeriousDlqin2yrs']
        df_orig = df_orig.drop(['SeriousDlqin2yrs'], axis=1)
        
    elif args.dataset_name == "credit":
        
        df_orig = pd.read_csv("german_categorical-binsensitive.csv")
        
        # Change class labels from [1,2] to [1,0]
        df_orig['credit'] = (df_orig['credit'] == 1).astype(int)
        
        # Copy the already binarized age column
        df_orig['age_bin'] = (df_orig['age']).copy()
        df_orig = df_orig.drop(['sex-age', 'age'], axis=1)
        
    df_orig = df_orig.dropna()
    

    # --- Training recourse method ---

    if args.train_recourse_method == 'GS':
        recourse_hyperparam_train = {}
    elif args.train_recourse_method == "CCHVAE":
        recourse_hyperparam_train = {} # defined inside the train function due to model requirements
    elif args.train_recourse_method == "WT":
        recourse_hyperparam_train = {"loss_type": "MSE", "y_target": [1.0], "binary_cat_features": True}

    
    # --- Test recourse method ---
    
    if args.test_recourse_method is None:
        args.test_recourse_method = args.train_recourse_method

    if args.test_recourse_method == 'GS':
        recourse_hyperparam_test = {}
    elif args.test_recourse_method == "CCHVAE":
        recourse_hyperparam_test = {} # defined inside the train function due to model requirements
    elif args.test_recourse_method == "WT":
        recourse_hyperparam_test = {"loss_type": "MSE", "y_target": [1.0], "binary_cat_features": True}


    # --- Prepare data ---

    # Split train and test
    df_train, df_test = train_test_split(df_orig, test_size=args.test_size, random_state=args.random_state)
        
    # Save train into dataframe 
    df_train.to_csv(f"{args.dataset_name}_train.csv", index=False)
    df_test.to_csv(f"{args.dataset_name}_test.csv", index=False)
    
    
    # --- Files to store training and testing metrics ---

    if args.base_model == "ann":
        
        train_file_path = f"{args.dataset_name}_train.csv"
        train_results_file_path = f"results_num/{args.dataset_name}_{args.train_recourse_method}_{args.fair_strategy}_pretrain{args.pre_epoch}_total{args.total_epoch}_sens{'_'.join(args.sens_attr)}_rs{args.random_state}_training_metrics_log.csv"

        test_file_path = f"{args.dataset_name}_test.csv"
        test_results_file_path = f"results_num/{args.dataset_name}_train{args.train_recourse_method}_test{args.test_recourse_method}_{args.fair_strategy}_pretrain{args.pre_epoch}_total{args.total_epoch}_sens{'_'.join(args.sens_attr)}_rs{args.random_state}_test_metrics_log.csv"


    elif args.base_model == "linear":
        
        train_file_path = f"{args.dataset_name}_train.csv"
        train_results_file_path = f"results_num/lin_{args.dataset_name}_{args.train_recourse_method}_{args.fair_strategy}_pretrain{args.pre_epoch}_total{args.total_epoch}_sens{'_'.join(args.sens_attr)}_rs{args.random_state}_training_metrics_log.csv"

        test_file_path = f"{args.dataset_name}_test.csv"
        test_results_file_path = f"results_num/lin_{args.dataset_name}_train{args.train_recourse_method}_test{args.test_recourse_method}_{args.fair_strategy}_pretrain{args.pre_epoch}_total{args.total_epoch}_sens{'_'.join(args.sens_attr)}_rs{args.random_state}_test_metrics_log.csv"

        
    # --- Begin training ---

    print("Training...")

    if args.fair_strategy == "burden":
        my_trained_model, train_metrics, ml_model = train_social_burden(dataset=args.dataset_name, file_path=train_file_path, sens_attr=args.sens_attr,
                                                                        base_model=args.base_model,
                                            lr=args.learning_rate, epochs=args.total_epoch, batch_size=args.batch_size, n_inst_eval_train_metrics=args.n_eval_train_metrics,
                                            hidden_sizes=args.hidden_sizes, activation_name=args.activation, verbose=True, pretrain_epochs=args.pre_epoch, 
                                            recourse_method=args.train_recourse_method, recourse_hyperparam=recourse_hyperparam_train, 
                                            results_file=train_results_file_path, random_state=args.random_state)

    print("Training ended!")

    # --- Test the resulting model ---

    print("Testing...")
    
    if args.fair_strategy == "burden":

        test_metrics = test_recourse(dataset=args.dataset_name, file_path=test_file_path, sens_attr=args.sens_attr, ml_model=ml_model,
                                recourse_method=args.test_recourse_method, recourse_hyperparam=recourse_hyperparam_test, 
                                results_file=test_results_file_path, random_state=args.random_state)

    print("Testing ended!")

print(test_metrics)
