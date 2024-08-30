import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sentence_transformers import SentenceTransformer, losses
from torch.optim import Adam
from torch.utils.data import DataLoader


def evaluate_model(
    all_data,
    model_name="bert-base-nli-mean-tokens",
    k=5,
    epoch=5,
    optimizer_params=None,
):
    if optimizer_params is None:
        optimizer_params = {"lr": 5e-5, "eps": 1e-8, "weight_decay": 0}

    # Define the optimizer class
    optimizer_class = Adam

    # Define the number of folds
    kf = KFold(n_splits=k)

    # Convert NaN values to 'missing' and others to string
    all_data["student_answer"] = all_data["student_answer"].apply(
        lambda x: "missing" if pd.isna(x) else str(x)
    )
    all_data["model_answer"] = all_data["model_answer"].apply(
        lambda x: "missing" if pd.isna(x) else str(x)
    )

    # Prepare your data
    questions = all_data["question"].unique().tolist()

    mae_percentages = []
    correlations = []
    mae_percentages_train = []
    correlations_train = []
    train_data_list = []
    test_data_list = []

    for train_index, test_index in kf.split(questions):
        # Split data
        train_questions = [questions[i] for i in train_index]
        test_questions = [questions[i] for i in test_index]

        # Get the corresponding data for these questions
        train_data = all_data[all_data["question"].isin(train_questions)]
        train_data_list.append(train_data)
        test_data = all_data[all_data["question"].isin(test_questions)]
        test_data_list.append(test_data)

        # Train model with Adam optimizer
        final_model = SentenceTransformer(model_name)
        train_dataset = data_loader_local(train_data)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(model=final_model)
        optimizer = optimizer_class(final_model.parameters(), **optimizer_params)
        final_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epoch,
            warmup_steps=330,
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
        )

        # Test model - TEST - The actual one to keep later
        students_embed = final_model.encode(
            test_data["student_answer"].to_list(), convert_to_tensor=True
        )
        model_answer_embed = final_model.encode(
            test_data["model_answer"].to_list(), convert_to_tensor=True
        )

        y_pred, y = norm_eval_sbert(
            students_embed, model_answer_embed, test_data, "fair"
        )

        # Calculate MAE percentage
        mae_percent = 1 - mean_absolute_error(
            [i / 5 for i in y], [i / 5 for i in y_pred]
        )
        mae_percentages.append(mae_percent)

        # Calculate correlation coefficient
        correlation, _ = spearmanr(y, y_pred)
        correlations.append(correlation)

        # Test model - TRAIN
        students_embed_train = final_model.encode(
            train_data["student_answer"].to_list(), convert_to_tensor=True
        )
        model_answer_embed_train = final_model.encode(
            train_data["model_answer"].to_list(), convert_to_tensor=True
        )

        y_pred_train, y_train = norm_eval_sbert(
            students_embed_train, model_answer_embed_train, train_data, "fair"
        )

        # Calculate MAE percentage
        mae_percent_train = 1 - mean_absolute_error(
            [i / 5 for i in y_train], [i / 5 for i in y_pred_train]
        )
        mae_percentages_train.append(mae_percent_train)

        # Calculate correlation coefficient
        correlation_train, _ = spearmanr(y_train, y_pred_train)
        correlations_train.append(correlation_train)

    # Calculate average MAE percentage and correlation for test set
    average_mae_percent = np.mean(mae_percentages)
    average_correlation = np.mean(correlations)

    # Calculate average MAE percentage and correlation for train set
    average_mae_percent_train = np.mean(mae_percentages_train)
    average_correlation_train = np.mean(correlations_train)

    return (
        average_mae_percent,
        average_correlation,
        mae_percentages,
        correlations,
        average_mae_percent_train,
        average_correlation_train,
        mae_percentages_train,
        correlations_train,
        train_data_list,
        test_data_list,
    )
