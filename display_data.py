import math
import numpy as np
import matplotlib.pyplot as P
import matplotlib.ticker as MT
import matplotlib.cm as CM


def display_data(X, width=None):

    [m, n] = X.shape
    if not width:
        width = round(math.sqrt(n))

    height = int(n / width)

    display_rows = (math.floor(math.sqrt(m)))
    display_cols = (math.ceil(m / display_rows))

    # Between images padding
    pad = 1

    display_array = - np.ones([pad + display_rows * (height + pad),
                              pad + display_cols * (width + pad)])

    curr_ex = 0
    for j in range(0, display_rows):
        for i in range(0, display_cols):
            if curr_ex >= m:
                break

            # Get the max value of the patch
            max_val = np.max(abs(X[curr_ex, :]))
            row = pad + j * (height + pad)
            col = pad + i * (width + pad)
            display_array[row:(row+height), col:(col+width)] = \
                np.true_divide(np.reshape(X[curr_ex, :], [height, width], order='F'), max_val)
            curr_ex += 1

        if curr_ex >= m:
            break

    scaledimage(display_array)
    P.show()

def scaledimage(W, pixwidth=1, ax=None, grayscale=True):
    """
    Do intensity plot, similar to MATLAB imagesc()
    W = intensity matrix to visualize
    pixwidth = size of each W element
    ax = matplotlib Axes to draw on
    grayscale = use grayscale color map
    Rely on caller to .show()
    """
    # N = rows, M = column
    (N, M) = W.shape
    # Need to create a new Axes?
    if not ax:
        ax = P.figure().gca()
    # extents = Left Right Bottom Top
    exts = (0, pixwidth * M, 0, pixwidth * N)
    if grayscale:
        ax.imshow(W,
                  interpolation='nearest',
                  cmap=CM.gray,
                  extent=exts)
    else:
        ax.imshow(W,
                  interpolation='nearest',
                  extent=exts)

    ax.xaxis.set_major_locator(MT.NullLocator())
    ax.yaxis.set_major_locator(MT.NullLocator())
    return ax