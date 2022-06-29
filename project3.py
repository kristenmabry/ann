from read_data import *
from rms import getRMSVals
from ann import *
from results import *

# normalize training data
training = readData('traindat.txt', ' |\t')
rmsVals = getRMSVals()
for moment in rmsVals:
    training[moment] = training[moment] / rmsVals[moment]

# normalize eval data 1
eval1 = readData('eval1dat.txt', '    |   |  | ')
for moment in rmsVals:
    eval1[moment] = eval1[moment] / rmsVals[moment]

# normalize eval data 2
eval2 = readData('eval2dat.txt', '    |   |  | |\t')
for moment in rmsVals:
    eval2[moment] = eval2[moment] / rmsVals[moment]


act_lambda = 2
def activation_function(x):
    return 1 / (1 + np.exp(-1*act_lambda*x))
def der_activation_function(x):
    outer = np.outer(activation_function(x), (1 - activation_function(x)))
    return np.matrix(act_lambda * np.diag(outer)).T

expected_results = np.identity(10)
save = False

# 1) train by epoch
weights1, weights2, jws, perc_err = train_network(training, expected_results, activation_function, der_activation_function)
results0 = annClassify(training, weights1, activation_function, weights2)
getResults(training, results0, 'training', 'results_epoch.txt', 'training_epoch.png', save)
results1 = annClassify(eval1, weights1, activation_function, weights2)
getResults(eval1, results1, 'eval1', 'results_epoch.txt', 'eval1_epoch.png', save)
results2 = annClassify(eval2, weights1, activation_function, weights2)
getResults(eval2, results2, 'eval2', 'results_epoch.txt', 'eval2_epoch.png', save)
plotPoints(jws, 'J(w)', 'J(w) vs. Epoch', 'jw_plot_epoch.png', save)
plotPoints(perc_err, 'Percent Error (%)', 'Percent Error vs. Epoch', 'pe_plot_epoch.png', save)

# 2) train by pattern
weights1, weights2, jws, perc_err = train_network(training, expected_results, activation_function, der_activation_function, False)
results0 = annClassify(training, weights1, activation_function, weights2)
getResults(training, results0, 'training', 'results_pattern.txt', 'training_pattern.png', save)
results1 = annClassify(eval1, weights1, activation_function, weights2)
getResults(eval1, results1, 'eval1', 'results_pattern.txt', 'eval1_pattern.png', save)
results2 = annClassify(eval2, weights1, activation_function, weights2)
getResults(eval2, results2, 'eval2', 'results_pattern.txt', 'eval2_pattern.png', save)
plotPoints(jws, 'J(w)', 'J(w) vs. Epoch', 'jw_plot_pattern.png', save)
plotPoints(perc_err, 'Percent Error (%)', 'Percent Error vs. Epoch', 'pe_plot_pattern.png', save)

# 3) change activation function lambda
for i in [.5, 1, 2.5, 3]:
    act_lambda = i
    weights1, weights2, jws, perc_err = train_network(training, expected_results, activation_function, der_activation_function)
    results0 = annClassify(training, weights1, activation_function, weights2)
    getResults(training, results0, 'training', f'results_lambda_{i}.txt', f'training_lambda_{i}.png', save)
    results1 = annClassify(eval1, weights1, activation_function, weights2)
    getResults(eval1, results1, 'eval1', f'results_lambda_{i}.txt', f'eval1_lambda_{i}.png', save)
    results2 = annClassify(eval2, weights1, activation_function, weights2)
    getResults(eval2, results2, 'eval2', f'results_lambda_{i}.txt', f'eval2_lambda_{i}.png', save)
    plotPoints(jws, 'J(w)', 'J(w) vs. Epoch', f'jw_plot_lambda_{i}.png', save)
    plotPoints(perc_err, 'Percent Error (%)', 'Percent Error vs. Epoch', f'pe_plot_lambda_{i}.png', save)
act_lambda = 2

# 4) change learning rate
for i in [.01, .05, .5, 1]:
    weights1, weights2, jws, perc_err = train_network(training, expected_results, activation_function, der_activation_function, True, i)
    results0 = annClassify(training, weights1, activation_function, weights2)
    getResults(training, results0, 'training', f'results_lr_{i}.txt', f'training_lr_{i}.png', save)
    results1 = annClassify(eval1, weights1, activation_function, weights2)
    getResults(eval1, results1, 'eval1', f'results_lr_{i}.txt', f'eval1_lr_{i}.png', save)
    results2 = annClassify(eval2, weights1, activation_function, weights2)
    getResults(eval2, results2, 'eval2', f'results_lr_{i}.txt', f'eval2_lr_{i}.png', save)
    plotPoints(jws, 'J(w)', 'J(w) vs. Epoch', f'jw_plot_lr_{i}.png', save)
    plotPoints(perc_err, 'Percent Error (%)', 'Percent Error vs. Epoch', f'pe_plot_lr_{i}.png', save)

# 5) change number of hidden nodes
for i in [2, 3, 5, 6]:
    weights1, weights2, jws, perc_err = train_network(training, expected_results, activation_function, der_activation_function, True, .1, i)
    results0 = annClassify(training, weights1, activation_function, weights2)
    getResults(training, results0, 'training', f'results_hn_{i}.txt', f'training_hn_{i}.png', save)
    results1 = annClassify(eval1, weights1, activation_function, weights2)
    getResults(eval1, results1, 'eval1', f'results_hn_{i}.txt', f'eval1_hn_{i}.png', save)
    results2 = annClassify(eval2, weights1, activation_function, weights2)
    getResults(eval2, results2, 'eval2', f'results_hn_{i}.txt', f'eval2_hn_{i}.png', save)
    plotPoints(jws, 'J(w)', 'J(w) vs. Epoch', f'jw_plot_hn_{i}.png', save)
    plotPoints(perc_err, 'Percent Error (%)', 'Percent Error vs. Epoch', f'pe_plot_hn_{i}.png', save)

# 6.1) momentum
weights1, weights2, jws, perc_err = train_network(training, expected_results, activation_function, der_activation_function, True, .1, 4, False, True)
results0 = annClassify(training, weights1, activation_function, weights2)
getResults(training, results0, 'training', 'results_momentum.txt', 'training_momentum.png', save)
results1 = annClassify(eval1, weights1, activation_function, weights2)
getResults(eval1, results1, 'eval1', 'results_momentum.txt', 'eval1_momentum.png', save)
results2 = annClassify(eval2, weights1, activation_function, weights2)
getResults(eval2, results2, 'eval2', 'results_momentum.txt', 'eval2_momentum.png', save)
plotPoints(jws, 'J(w)', 'J(w) vs. Epoch', 'jw_plot_momentum.png', save)
plotPoints(perc_err, 'Percent Error (%)', 'Percent Error vs. Epoch', 'pe_plot_momentum.png', save)

# 6.2) bias
training['bias'] = 1
eval1['bias'] = 1
eval2['bias'] = 1
weights1, weights2, jws, perc_err = train_network(training, expected_results, activation_function, der_activation_function, True, .1, 4, True)
results0 = annClassify(training, weights1, activation_function, weights2)
getResults(training, results0, 'training', 'results_bias.txt', 'training_bias.png', save)
results1 = annClassify(eval1, weights1, activation_function, weights2)
getResults(eval1, results1, 'eval1', 'results_bias.txt', 'eval1_bias.png', save)
results2 = annClassify(eval2, weights1, activation_function, weights2)
getResults(eval2, results2, 'eval2', 'results_bias.txt', 'eval2_bias.png', save)
plotPoints(jws, 'J(w)', 'J(w) vs. Epoch', 'jw_plot_bias.png', save)
plotPoints(perc_err, 'Percent Error (%)', 'Percent Error vs. Epoch', 'pe_plot_bias.png', save)