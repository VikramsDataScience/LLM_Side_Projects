#  Q&A model based on research papers from arXiv in PDF (WIP)


## On Optical Character Recognition (OCR)


### Identified bugs
- From time to time the image detection and extraction code I've written doesn't always extract images cleanly. for instance, in Document ID: 0704.0300 the image detection has correctly identified images in pages 1&2, but has extracted them in portions. Instead of cleanly extracting the image, it's broken up one image into several portions. As a POC, this is only a small problem, since the majority of the other images have been correctly extracted. But, in a production setting this could become a significant problem, and would need to be addressed.