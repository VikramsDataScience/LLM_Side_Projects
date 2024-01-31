#  Optical Character Recognition (OCR) model based on research papers from arXiv in PDF (WIP)


# Possible use cases
Whilst, this model has been developed using the text and images extracted from research papers sourced from arXiv (that was more of a personal desire and preference), the use case for such an  OCR model could make up the Document Retrieval capability of a Multi Modal system. As language models mature and start to incorporate many models into a comprehensive system (the front of which is the chatbot that users experience), document retrieval will make up part of that system. ChatGPT is already beginning to role out such a feature in the paid tiers, whereby, users can upload PDFs to the chatbot, and it can read the PDF to extract meaningful insights about the user based on the documents that they upload.

As such, this OCR model, with the sufficient maturity beyond POC (since this model is super basic), would be required would make up that aspect of the Multi Modal chatbot (or Language Model). It could conceivably be used to extract the text and images from uploaded PDFs, perform OCR on the images, and perform whichever required NLP tasks on the text. The results of the OCR would then be fed into the multi modal chatbot, which can then use its own generative capabilities to provide responses and suggestions to users.

## On Optical Character Recognition (OCR)


### Identified bugs
- From time to time the image detection and extraction code I've written doesn't always extract images cleanly. for instance, in Document ID: 0704.0300 the image detection has correctly identified images in pages 1&2, but has extracted them in portions. Instead of cleanly extracting the image, it's broken up one image into several portions. As a POC, this is only a small problem, since the majority of the other images have been correctly extracted. But, in a production setting this could become a significant problem, and would need to be addressed.