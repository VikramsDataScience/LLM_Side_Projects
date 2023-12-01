# Shopping Cart Recommender Model (WIP)
This project is a recommender model based on ecommerce orders data (including whether certain products were reordered).

# EDA
## On Missing Values
The `ProfileReport` seemingly exposes about 10,565 missing values from the dataset. However, the missing values appear to be visits to particular product pages, but where no orders were made: <br>
![Alt text](image.png)<br>
We can see that there are missing rows in the columns `'order_id'`, `'add_to_cart_order'`, and `'reordered'` - all of these columns appear to relate to orders placed. Being as this is a recommender, I felt it would be prudent to keep these rows, since they contain important information on how users are navigating the site.

## On Feature Extraction
Word2Vec and GloVe are both feature extraction techniques used in natural language processing (NLP) to convert words or phrases into numerical vectors. While both methods are used to capture the meaning and context of words, there are some key differences between them:

1. Training Objective:
Word2Vec is trained to predict a word's context, while GloVe is trained to predict a word's co-occurrence probability. In other words, Word2Vec aims to reconstruct the input sentence by predicting the missing word based on its context, while GloVe aims to predict the likelihood of a word appearing in a given context.
2. Vector Space:
Word2Vec represents words in a high-dimensional vector space, where words with similar meanings are closer together. GloVe, on the other hand, represents words in a lower-dimensional vector space, where words with similar co-occurrence probabilities are closer together.
3. Vector Dimensionality:
Word2Vec typically uses a fixed vector dimensionality, while GloVe uses a dynamic vector dimensionality that adjusts based on the number of words in the vocabulary.
4. Training Time:
GloVe is generally faster to train than Word2Vec, especially for large datasets.
5. Performance:
Both methods have been shown to be effective in various NLP tasks, but Word2Vec tends to perform better in tasks that require explicit context, such as language modeling and text classification. GloVe, on the other hand, tends to perform better in tasks that require implicit context, such as word similarity and clustering.
6. Interpretability:
GloVe is more interpretable than Word2Vec, as it provides a direct measure of the word's co-occurrence probability, which can be useful for understanding the relationships between words in a sentence.

The other option are <b>N-Grams</b>:<br>
N-Grams are a type of feature extraction technique that involves breaking down text data into sequences of n-grams, which are contiguous sequences of n items. For example, if we have a shopping cart with the items "apple", "banana", and "orange", we can represent this as a sequence of 3-grams: "apple banana", "banana orange", and "apple orange".
Using N-Grams for the shopping cart recommender system has some advantages over GloVe. For example:
1. N-Grams can capture the sequential dependencies between items in the shopping cart, which can be useful for making recommendations. For example, if a customer has previously purchased "apple" and "banana" together, the system can recommend "orange" as the next item in the sequence.
2. N-Grams can be more efficient to compute than GloVe, especially for large shopping carts. This is because N-Grams can be computed in O(n) time, where n is the number of items in the shopping cart, while GloVe requires O(n^2) time.
However, N-Grams also have some limitations. For example:
3. N-Grams may not capture the full context and meaning of the items in the shopping cart, especially if the items are highly correlated. For example, if a customer has previously purchased "apple" and "banana" together, the system may not be able to capture the fact that "orange" is also a fruit and may recommend it as the next item in the sequence.
4. N-Grams may not be able to capture the relationships between items that are not in the same sequence. For example, if a customer has previously purchased "apple" and "orange" together, but not in the same sequence, the system may not be able to capture this relationship using N-Grams.

Overall, the choice between N-Grams and GloVe will depend on the specific requirements of the shopping cart recommender system and the characteristics of the data. Both techniques have their strengths and weaknesses, and the best approach will depend on the specific use case.
