# Packages
import joblib

# Get customized functions from library
import packages.data_processor as dp
import packages.model_trainer as mt


# Path to data
path_to_data = './data/Iris.csv'

# Prepare the data
prepared_data = dp.prepare_data(path_to_data)

# Create train - test split
train_test_data = dp.create_train_test_data(prepared_data['input'], 
                                         prepared_data['label'], 
                                         0.33, 2021)

# Run training
model = mt.run_model_training(train_test_data['x_train'], train_test_data['x_test'], 
                           train_test_data['y_train'], train_test_data['y_test'])

# Save the trained model
joblib.dump(model, './models/test_model.pkl')
