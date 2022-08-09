import numpy as np

"""Dealing with imbalanced datasets when there's only two labels is easy.
Generalizing to a multi-class situation seems a bit more challenging."""
def get_max_label_imbalance(y, raw_count=False):
    """
    Return the difference between the frequencies of your most and
    least frequent classes divided by the length of your dataset.

    Set raw_count to True to obtain frequencies rather than ratios.
    """

    y = np.asarray(y)
    frequency = np.sort(np.unique(y, return_counts=True)[1])[::-1]
    print(frequency)

    if len(frequency) < 2:
        m = ('The labelset passed to get_label_imbalance_array has less than '
             ' 2 unique classes.')
        raise RuntimeError(m)


    if raw_count:
        return frequency[0] - frequency[-1]
    return (frequency[0] - frequency[-1]) / len(y)

def get_label_imbalance_array(y, raw_count=False):
    """
    Return a ndarray quantifying the relative differences between classes.

    For a dataset with n types of labels, this function returns a nxn array.
    The mxnth value is the difference between the frequencies of the mth most
    common label and the nth most common label, scaled by the length of the
    dataset.

    Set raw_count to True to just get frequencies

    For example:
        Given a label list that looks like [1,2,1,2,1,1,3,1] the returned array
        looks like

        [[ 0.     0.375  0.5  ]
         [-0.375  0.     0.125]
         [-0.5   -0.125  0.   ]]
    """
    y = np.asarray(y)
    frequency = np.sort(np.unique(y, return_counts=True)[1])[::-1]

    if len(frequency) < 2:
        m = ('The labelset passed to get_label_imbalance_array has less than '
             ' 2 unique classes.')
        raise RuntimeError(m)

    ret = np.zeros((len(frequency), len(frequency)))
    print(frequency)
    for i in range(0, len(frequency) - 1):
        for j in range(i + 1, len(frequency)):
            ret[i, j] = frequency[i] - frequency[j]
            ret[j, i] = -ret[i, j]
    if raw_count:
        return ret
    return ret / len(y)

def is_normalized(X):
    """
    Return a list specifying whether the associated column has been
    scaled and shaped to the standard normal distribution.
    """    
    X = np.asarray(X)
    ave = np.average(X,axis=0)
    std = np.std(X,axis=0)
    ret = []
    for i in ave:
        if ave[i] = 0 and std[i] = 1:
            ret.append("yes")
        else:
            ret.append("no")
    return ret

def normal_likelihood(X):
    """Return a list specifying the likelihood that the associated column is normal distributed"""
    X = np.asarray(X)
    