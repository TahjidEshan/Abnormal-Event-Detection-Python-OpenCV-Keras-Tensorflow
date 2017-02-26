import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    data = pd.read_csv('test.csv')
    N = 5
    ind = np.arange(N)  # the x locations for the groups
    #fig=plt.figure()
    width = 0.35       # the width of the bars
    values = [data.loc[0, data.columns[0]], data.loc[0, data.columns[1]], data.loc[
        0, data.columns[2]], data.loc[0, data.columns[3]], data.loc[0, data.columns[4]]]
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, values, width, color='r')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparison among accuracy of classifiers')
    ax.set_xticks(ind)
    ax.set_xticklabels((data.columns[0], data.columns[1], data.columns[
                       2], data.columns[3], data.columns[4]))

    fig.savefig('AccuracyTest.png')
    plt.show()

if __name__ == '__main__':
    main()
