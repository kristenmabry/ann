# Artificial Neural Network

- Trains an ANN for ten letters using 8 moment values
- Contains training data of 100 rows, evaluation set 1 with 20 rows, evaluation set 2 with 100 rows
- Contains variables for learning rate, bias, momentum, # of hidden nodes, and train by epoch or pattern, activation function, 

Files
- project3.py: loads and normalizes data, trains ANN for different cases, classifies evaluation data, creates confusion matrices, saves results
- ann.py: contains the code for training network and classifying data
- read_data.py: reads the data from the txt file and converts to pandas dataframe
- results.py: calculates error and saves to txt file, creates confusion matrix and saves to png file
- rms.py: calculates RMS values for data, reads RMS values from txt file
- traindat.txt: training data
- eval1dat.txt: evaluation set 1
- eval2dat.txt: evaluation set 2
- rms_vals.txt: RMS values for training data
- wji.txt, wkj.txt: starting weights

Read Mabry_Kristen_Report.pdf for more info
