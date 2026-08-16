import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
import yaml
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from sklearn.metrics import f1_score,mean_absolute_error,accuracy_score,mean_squared_error,roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

def load_config():
    with open("backend/config.yaml","r") as f:
        return yaml.safe_load(f)


def load_data(config):

    mode=config["mode"]

    if mode=="sample":
        ticket_path=config["data"]["sample"]["ticket"]
        delay_path=config["data"]["sample"]["delay"]

    elif mode=="full":
        ticket_path=config["data"]["full"]["ticket"]
        delay_path=config["data"]["full"]["delay"]

    else:
        raise ValueError("Invalid mode in config.yaml. Must be 'sample' or 'full'.")
    
    ticket_df=pd.read_csv(ticket_path)
    delay_df=pd.read_csv(delay_path)
    print(f"Ticket   : {ticket_df.shape}")
    print(f"Delay    : {delay_df.shape}")

    return ticket_df, delay_df



def get_ticket_column_types(df, target):
    # ordered columns — label encode
    label_cols = [
        "Holiday or Peak Season",   # Yes/No has no real order but binary fine
        "journey_month",            # already numeric
        "journey_dayofweek",        # already numeric
        "days_before_journey"       # already numeric
    ]
    onehot_cols = [
        "Class of Travel",
        "Quota",
        "Source Station",
        "Destination Station",
        "Train Type",
        "Special Considerations",
        "Seat Availability"
    ]

    
    
    numeric_cols = [
        col for col in df.columns
        if col not in label_cols
        and col not in onehot_cols
        and col != target
    ]

    return label_cols, onehot_cols, numeric_cols


def get_delay_column_types(df, target):
    # ordered
    label_cols = [
        "Season",
        "journey_month",
        "journey_dayofweek"
    ]

    # no-order categorical
    onehot_cols = [
        "Source",
        "Destination",
        "Run_frequency"
    ]

    # already numeric
    numeric_cols = [
        col for col in df.columns
        if col not in label_cols
        and col not in onehot_cols
        and col != target
    ]

    return label_cols, onehot_cols, numeric_cols

def build_preprocessor(label_cols, onehot_cols, numeric_cols):
    
    label_pipeline=Pipeline(steps=[('le', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])

    one_hotpipeline=Pipeline(steps=[('ohe', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor=ColumnTransformer(transformer=[("label encoder",label_pipeline,label_cols),("onehot encoder",one_hotpipeline,onehot_cols),("numeric","passthrough",numeric_cols)], remainder="passthrough")

    return preprocessor 


def train_classifier(ticket_df):

    config=load_config()

    target=config["features"]["ticket"]["target"]

    X=ticket_df.drop(columns=[target])

    y=ticket_df[target]

    label_cols,one_hot_cols,numeric_cols=get_ticket_column_types(ticket_df, target)

    preprocessor=build_preprocessor(label_cols, one_hot_cols, numeric_cols)

    pipeline=Pipeline(steps=[("pre processing",preprocessor),("trainer",XGBClassifier(random_state=42))])

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    pipeline.fit(X_train,y_train)

    y_pred=pipeline.predict(X_test)

    acc=accuracy_score(y_test,y_pred)

    f1=f1_score(y_test,y_pred)

    auc=roc_auc_score(y_test,y_pred)

    print(f"Ticket Classifier - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
    
    return pipeline


def train_regressor(delay_df):

    config=load_config()

    target=config["features"]["delay"]["target"]

    X=delay_df.drop(columns=[target])

    y=delay_df[target]

    label_cols,one_hot_cols,numeric_cols=get_delay_column_types(delay_df, target)

    preprocessor=build_preprocessor(label_cols, one_hot_cols, numeric_cols)

    pipeline=Pipeline(steps=[("pre processing",preprocessor),("trainer",XGBRegressor(random_state=42))])

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    pipeline.fit(X_train,y_train)

    y_pred=pipeline.predict(X_test)

    mae=mean_absolute_error(y_test,y_pred)

    mse=mean_squared_error(y_test,y_pred)

    rmse=np.sqrt(mse)

    print(f"Delay Regressor - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

    return pipeline

def save_models(classifier, regressor, config):
    os.makedirs("backend/ml", exist_ok=True)

    joblib.dump(classifier, config["models"]["classifier"]["path"])
    joblib.dump(regressor,  config["models"]["regressor"]["path"])

    
    print(f"Classifier → {config['models']['classifier']['path']}")
    print(f"Regressor  → {config['models']['regressor']['path']}")


if __name__ == "__main__":

    config=load_config()

    ticket_df,delay_df=load_data(config) 

    classifier=train_classifier(ticket_df)
    regressor=train_regressor(delay_df)

    save_models(classifier, regressor, config)

    print("Models trained and saved successfully."
          )   

