"""
Pooling layers module for Python
============================

spp is a pooling layers module for keras, and requires keras version 2.0 or greater.

One type of pooling layers are currently available:
    SpatialPyramidPooling: 
        Apply the pooling procedure on the entire image, given an image batch. 
        This is especially useful if the image input can have varying dimensions, 
        but needs to be fed to a fully connected layer.
"""