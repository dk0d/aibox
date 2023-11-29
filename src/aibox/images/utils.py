import numpy as np


def cv_read_image(path, **kwargs):
    """helps read an image using opencv

    returns the image in RGB format

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    import cv2 as cv

    img = cv.imread(path, **kwargs)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def get_dominant_colors(img, n_colors=5):
    """helps get the dominant colors in an image

    Returns the image with only those dominant colors applied

    Args:
        img (_type_): _description_
        n_colors (int, optional): _description_. Defaults to 5.
        return_palette (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    from sklearn.cluster import KMeans

    height, width, channels = img.shape
    img = img.reshape((height * width, channels))
    clt = KMeans(n_clusters=n_colors, n_init="auto", init="random")
    labels = clt.fit_predict(img)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape((height, width, channels))
    labels = labels.reshape((height, width))
    color_palette = np.uint8(clt.cluster_centers_)
    return quant, color_palette, labels
