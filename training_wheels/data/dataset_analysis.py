import numpy as np
import statistics as st

"""Dealing with imbalanced datasets when there's only two labels is easy.
Generalizing to a multi-class situation seems a bit more challenging."""
def get_label_imbalance_max(y, raw_count=False):
    """
    Return the difference between the frequencies of your most and
    least frequent classes normalized by the length of your dataset.

    Set raw_count to True to obtain frequencies rather than ratios.
    """

    y = np.asarray(y)
    frequency = np.sort(np.unique(y, return_counts=True)[1])[::-1]

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

    for i in range(0, len(frequency) - 1):
        for j in range(i + 1, len(frequency)):
            ret[i, j] = frequency[i] - frequency[j]
            ret[j, i] = -ret[i, j]
    if raw_count:
        return ret
    return ret / len(y)

def get_scale_array(X):
    """
    Return an array with tuples with the ratio of the mean and median of the associated features.

    Median and mean are calculated for each feature and divided by eachother to get a tuple of ratios.
    These ratios are then added to an array with position corresponding to the feaure positions.

    For example:
        Given an array of [[100, 2, 6 ],
                           [ 50, 6, 2 ],
                           [ 70, 5, 30]]

        the returned array looks like:
        [[(1.0 ,  1.0), (14.0, 18.25), (11.67, 6.08)],
         [(0.07, 0.05), ( 1.0,   1.0), ( 0.83, 0.33)],
         [(0.09, 0.16), ( 1.2,   3.0), (  1.0, 1.0 )]]
    """    
    X = np.asarray(X)
    if len(np.shape(X)) != 2:
        if len(np.shape(X)) > 2:
            m_more = ('The dataset passed to get_scale_array is greater the 2 dimensions.')
            raise RuntimeError(m_more)

        else:
            m_less = ('The dataset passed to get_scale_array is less than 2 dimensions.')
            raise RuntimeError(m_less)

    elif np.shape(X)[1] < 2:
        m_scale = ('The dataset passed to get_scale_array has fewer than 2 predictors. No scaling is needed.')
        raise RuntimeError(m_scale)
    
    X_transformed = X.T
    ret = []
    for i in range(np.shape(X_transformed)[0]):
        temp = []
        for j in range(np.shape(X_transformed)[0]):
            temp.append((round(st.median(X_transformed[i])/st.median(X_transformed[j]),2),round(st.mean(X_transformed[i])/st.mean(X_transformed[j]),2)))
        ret.append(temp)
    return ret

def get_deviation(X):
    """
    Return a list with the standard deviation for each column.
    """
    X = np.asarray(X)
    ret = []
    for i in range(np.shape(X)[1]):
        ret.append(round(np.std(X[:,i]),2))
    return ret