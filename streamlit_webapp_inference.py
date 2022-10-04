import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

# Preprocess data file 
data = pd.read_csv("space_trip.csv", index_col = 0)
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace = True)
data['Time'] = np.arange(len(data.index))

# Create Model
# Parameters file
target = '# of Trips'
features = ['Total Passengers', 'Fuel']
test_size = 0.2

# # Model Configs
model_config = {
  'LSTM_layer_units': None, 

  'dense_layers_units':[128, 64,32,16,4 ], # Hidden Layer units, List length needs to be same as activation below

  'dense_layers_activation': ['relu', 'relu', 'relu', 'relu', 'relu'], # Hidden Layer activations, List length needs to be same as units above

  'dropout_layers_size': [0.2, 0.3],

  'optimizer':keras.optimizers.Adam(0.001),
  
  'loss':'mean_squared_error',

  'metrics':['mse'],

  'epochs': 100,

  'batch_size' : 16,

  'validation_split': 0.2,

  'verbose': 0,

  'shuffle': False #For Validation split, do NOT shuffle time series data 
}

# Separate Data
data_int = data[[target]+features].copy().dropna()
X = data_int.loc[:,features].values
y = data_int.loc[:,target].values

# Reshape Data (Only needed for LSTM)
if model_config['LSTM_layer_units'] != None:
    scaler =  MinMaxScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(X.shape[0], 1, X.shape[1])

# Split Data
xtrain_lstm, xtest_lstm, ytrain_lstm, ytest_lstm=train_test_split(X, y, test_size=test_size, shuffle = False)

# Normalize Data
if model_config['LSTM_layer_units'] == None:
    scaler =  MinMaxScaler()
    xtrain_lstm = scaler.fit_transform(xtrain_lstm)
    xtest_lstm = scaler.transform(xtest_lstm)

def LSTM_model(xtrain, ytrain, model_config):
    """Returns ANN Model, with performance metrics and plot of test period performance

      Args:
        data: (pandas dataframe) time series data
        target: (str) target column for model
        features: (list[str]) features to be used in model 
        normalize: (bool) if true, training and testing features will be normalized
        test_size: (float) value between 0 and 1 for size of test data split
        return_metrics: (bool) if true, will print model training and testing set MSE and R2
        plot_result: (bool) if true, will plot model testing period performance
        model_config: (dict) contains various ann model parameters

      Returns:
        Print of model performance metrics if True
        Plot Object if True
    """
  # Build ANN Model
    model = keras.Sequential()

  # Add LSTM layer
    if model_config['LSTM_layer_units'] != None:
        model.add(tf.keras.layers.LSTM(model_config['LSTM_layer_units'], 
                                  input_shape=( xtrain.shape[1], xtrain.shape[2])))
  
    for i in range(len(model_config['dense_layers_units'])):
        model.add(tf.keras.layers.Dense(model_config['dense_layers_units'][i], activation = model_config['dense_layers_activation'][i]))
        while i < len(model_config['dropout_layers_size']):
            model.add(tf.keras.layers.Dropout(model_config['dropout_layers_size'][i]))
            break
  
    model.add(tf.keras.layers.Dense(1)) # Add final layer for output layer

  # Compile the model
    model.compile(loss=model_config['loss'], optimizer=model_config['optimizer'], metrics=model_config['metrics'])

    history = model.fit(
    xtrain, ytrain,
    epochs=model_config['epochs'],
    batch_size=model_config['batch_size'],
    validation_split=model_config['validation_split'],
    verbose=model_config['verbose'],
    shuffle=False
)
    return model    

lstm_mod = LSTM_model(xtrain_lstm, ytrain_lstm, model_config)


####### BEGIN STREAMLIT APP #######

st.header("Inference App")
col7, col8 = st.columns(2)

with col7:
    passengers = st.number_input('Total Passengers')

with col8:
    fuel = st.number_input('Fuel')
    
if st.button('Make Inference'):
    feature_values = scaler.transform(np.array([ passengers, fuel]).reshape(1, -1))
    
    st.metric("Model Inference (Number of Estimated Trips): ", np.round(lstm_mod(feature_values).numpy(),1))