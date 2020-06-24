import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def input_data(DATA_PATH='../data/bank-full.csv'):
    df = pd.read_csv(DATA_PATH, sep=';')
    return df


def train_test_data(df):
    X = df.drop('y', axis=1)
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def create_pipeline(df):
    '''
    Creates a pipeline

    Args:
        df: dataframe

    Returns:
        pipe: pipe object
    '''
    num_feat = df.drop('y', axis=1).select_dtypes(include=np.number).columns
    cat_feat = df.drop('y', axis=1).select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('one_hot', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_feat),
            ('cat', categorical_transformer, cat_feat)
        ])

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier',  LogisticRegression(class_weight='balanced',
                                           random_state=0))
    ])

    return pipe


def main():
    '''
    Trains a model and prints the confusion matrix and classification report
    '''
    df = input_data()
    pipe = create_pipeline(df)
    model = pipe.fit(X_train, y_train)
    target_names = y_test.unique().astype(str)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred))

if __name__ == '__main__':
    main()
