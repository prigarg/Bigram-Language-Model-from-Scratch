# N-gram in Natural Language Processing
### Bigram-Language-Model
In this python program a Bigram Language Model is trained for the training corpus with no-smoothing and add-one smoothing. For testing purpose, bigram counts, bigram probabilities for the test sentence along with the probability of test sentence under the trained model is printed to a text file. A detailed working explanation of code is documented in the program.
### Training Corpus
There are `10059` sentences , `17139` of unique words and `218619` words in the corpus. 

### How to run the ngrams.py file
Enter `0` for no smoothing and `1` for smoothing.

Type the following command to take input and output text file:

no-smooting::
> python -u ngrams.py 0 train_corpus.txt > results_no_smoothing.txt

add-one smooting::
> python -u ngrams.py 1 train_corpus.txt > resutls_add_one_smoothing.txt

The structure of the command is ::
> python -u <python-file-name.py> <smoothing(0 or 1)> <input-txt-data.txt> > <output-txt-file.txt>

#### Note: There is .ipynb file is also there which can be directly opened on Jupyter Notebook, make sure training corpus is in the same folder. It also contains the detailed explaination of the program.


