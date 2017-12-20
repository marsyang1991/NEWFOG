"""
1. several samples comprise a frame
2. several frames comprise a sequence
3. classification based on a sequence
"""

def make_squence(sampled_data):
    """
    Transform the sampled data into a sequential format.
    :param sampled_data: first column is timestamp, the last column is annotation and the followings are the data
    :return: numpy.ndarray, the annotation is based on the 
    """

