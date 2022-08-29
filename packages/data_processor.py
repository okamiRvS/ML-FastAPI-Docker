from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd 

def prepare_data(path_to_data):
    """
        @params:
            - path_to_data: the path to the data

        @return:
            - dictionary with following keys: 
                - input: the actual input
                - label: the label associated to that input
    """
    # Read data from path
    df = pd.read_csv(path_to_data)
    df = df.drop(columns=["Id"])

    # Encode labels
    le = LabelEncoder()
    df["Species"]= le.fit_transform(df["Species"])
    X = df.drop(columns=["Species"]).values
    y = df["Species"].values

    return {'input':X, 
            'label':y}

def create_train_test_data(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return {'x_train': X_train, 'x_test': X_test,
            'y_train': y_train, 'y_test': y_test}