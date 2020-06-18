.. module:: bioimg
.. automodule:: bioimg
   :noindex:

API reference
=============

Core functions
--------------
.. module:: bioimg.base
.. currentmodule:: bioimg
Here we have some core functions for I/O, image processing, etc

.. autosummary::
   :toctree: .

   base.read_image
   base.load_imgstack
   base.load_image_series
   base.write_image
   base.write_imgstack
   base.plot_channels
   base.combine_channels
   base.show_bbox
   base.threshold_img

Working with image segmentations
--------------------------------
.. module:: bioimg.segment
.. currentmodule:: bioimg
Module for working with segmented (labelled) images.

.. autosummary::
   :toctree: .

   segment.ImgX
   segment.compute_region_props
   segment.read_bbox
   segment.IncrementalClassifier

