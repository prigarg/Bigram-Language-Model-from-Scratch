# N-gram in Natural Language Processing
### Bigram-Language-Model
In this python program a Bigram Language Model is build from scratch and trained for the training corpus with no-smoothing and add-one smoothing. A detailed working explanation of code is documented in the program.
### Training Corpus
There are `10059` sentences , `17139` of unique words and `218619` words in the corpus. 

### Test Sentences
We check our model for two sentences::
1) `thus , because no man can follow another into these halls`
2) `upon this the captain started , and eagerly desired to know more`

Which are entered as list in the main program.

### Results
To test model's performace for the the above two sentences bigram counts and bigram probabilities along with the probability of test sentence under the trained model is printed to the text files `results_no_smoothing` (Results without smoothing) and `resutls_add_one_smoothing` (Results with add one smoothing).

### How to run the ngrams.py file
Enter `0` for no smoothing and `1` for smoothing.

Type the following command to take input and output text file:

no-smooting::
> python -u ngrams.py 0 train_corpus.txt > results_no_smoothing.txt

add-one smooting::
> python -u ngrams.py 1 train_corpus.txt > resutls_add_one_smoothing.txt

The structure of the command is ::
> python -u <python-file-name.py> <smoothing(0 or 1)> <input-txt-data.txt> > <output-txt-file.txt>

##### Note: There is `bigram_model.ipynb` file also which can be directly opened on Jupyter Notebook, make sure training corpus is in the same folder. It also contains the detailed explaination of the program.


