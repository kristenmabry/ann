def rms(data):
    sum = 0
    for i in data:
        sum += i ** 2
    return (sum / len(data)) ** (1/2)

def getRMSVals():
    with open('rms_vals.txt') as f:
        lines = f.readlines()

    rmsVals = {}

    for line in lines:
        words = line.split(' ')
        rmsVals[words[0]] = float(words[1])

    return rmsVals