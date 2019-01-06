# PredictFutureSales


1. Unpack the .zip file with data

2. Main file is train_predict.py
The program requires to set few settings:

data_path = 'Data/' 

file_name = 'dataset.npz' -> name of .npz file with training, validation and test sets

model     = 'xgb'         -> type of machine learning algorithms: 'xgb' stands for XGB Model, 'dnn' means Neural Network

ver       = 0             -> version number

make_data = True          -> if True program runs feature_engineering.py and creates a new dataset, if False .npz file with file_name is loaded

make_baseline = True      -> if True program makes a .csv file with baseline answers
