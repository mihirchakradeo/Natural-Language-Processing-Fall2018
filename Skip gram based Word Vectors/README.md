# Generating Word Embeddings using the Skip Gram formulation

- Python version: 2.7

1. Batch Generation
	
	- Created temp array of size window_size (data_index to data_index + window_size). Collected skip_window number of words from the left and right part of the center word in the temp array and put them inside the labels array. For tagging these collected outer words with the center word, just put the center word in the batch array for each entry in the labels array. Incremented current count (which keeps track of how many words have been considered) and global data_index by 1 for each word which was put inside batch array

	- Shifted the window by one position when all the words inside the window (except center word) were grabbed


2. Cross Entropy Loss
	
	- Took the product of the matrices which hold the embeddings for center word and the embeddings for the outer (context) words. The log and exp term cancel each other out, so the final step to get matrix A is just getting the diagonal elements from the product matrix. Extracted the diagonal elements from the above multiplication to get the matrix A using tf.diag_part()

	- For calculating the matrix B, first, took the exponential of the result of matrix mul obtained in first step. Then, the matrix B is simply obtained by taking row wise sum of the exponentiated matrix, by using tf.reduce_sum() over axis=1, which signifies row wise sum. Finally, took the log of the result of summation operation to obtain matrix B

	- Returned difference in matrix B and matrix A

3. NCE Loss

	- First, performed preprocessing on the given data:
		
		- Calculated the value of K from the expression in PDF by getting the len(sample)
		
		- Obtained the matrices/vectors for predicting words by gathering tensors from the weight and biases matrices using tf.gather(weights, labels) and tf.gather(biases,labels). For getting the unigram_prob of predicting words, first had to conver the unigram_prob list to a numpy array of float32 dtype. Then, performed tf.gather() on the updated unigram_prob np array to get the unigram probabilities of the predicting words
		
		- Similarly, obtained the matrices/vectors for negative samples by gathering tensors from weight and biases matrices, only this time using the sample as the second argument to tf.gather()

		- Other preprocessing also involved reshaping tensors to match the shape for performing matrix multiplications and additions

	- Calculating the first term from the expression: Log Pr(D=1,wo|wc)

		- Split the calculation into multiple steps like:
			i. Calculated the value of s(wo,wc): ucTuo + bo
			ii. Subtracted log KPr(wo) from s
			iii. Took the sigmoid of the result from step ii, then took log of it and stored into var called first_term
		
		Had to add a small value of 1e-10 to the log expression to handle case of NaN

	- Calculating the second term from the expression: Summation (Log(1-Pr(D=1,wx|wc)))

		- Split the calculation into multiple steps like:
			i. Calculated the value of s(wx,wc): ucTux + bx
			ii. Subtracted log KPr(wx) from s
			iii. Took the sigmoid of the result from step ii, then took log of 1-sigmoid, summed over all negative samples, and then stored in a variable called final_second_term

		Had to add a small value of 1e-10 to the log expression to handle case of NaN

	- Finally added first_term and final_second_term and returned the negative of that result


4. Word analogy task
	
	- Defined a function called cosine_similarity which returns the cosine similarity between two inputs A and B

	- For each line in the dev txt file performed the following steps:

		i. Performed preprocessing like splitting over "||", splitting over "," to get examples and choices
		ii. Created two arrays: 
			diff_examples: which holds the difference in the embeddings of example pairs
			diff_choices: which hold the difference in the embeddings of the choice pairs
		iii. Calculated the cosine similarity between the diff_examples and diff_choices arrays
		iv. Averaged the cosine similarity result over the rows (average similarity of each choice pair with the corresponding example pair)
		v. Took the argmax and argmin from the avg vector
		vi. Constructed the output string

	- Wrote the output string to a file called temp_dev_ce.txt or temp_dev_nce.txt depending on the model used

5. Finding top 20 similar words
	
	- For finding the top 20 similar words, I added a code snippet in the word_analogy.py file which performs the following steps:
		i. For each word in the dictionary, fetch the embedding of the current word and the embedding of the 3 words (american, first, would), and find the cosine similarity between the embeddings. Store the cosine similarities in a list.
		ii. Sort the list containing the cosine similarities and return the top 20


###############################################################
EXPERIMENT DETAILS
###############################################################

The details about the different hyperparameters used can be found in the report.
Best model hyperparameters:
1. Cross Entropy 
	- Configuration:
		batch_size = 128
		embedding_size = 128
		skip_window = 2
		num_skips = 4
		max_num_steps = 500001
		learning_rate = 0.0001

  	- Accuracy:
  		Accuracy of Least Illustrative Guesses:            35.6%
		Accuracy of Most Illustrative Guesses:             32.4%
		Overall Accuracy:                                  34.0%

2. NCE
	- Configuration:
		batch_size = 128
		embedding_size = 128
		skip_window = 4
		num_skips = 8
		valid_size = 16
	  	valid_window = 100
	  	num_sampled = 64
	  	learning_rate = 0.001
	  	max_num_steps = 200001

  	- Accuracy:
		Accuracy of Least Illustrative Guesses:            36.1%
		Accuracy of Most Illustrative Guesses:             33.6%
		Overall Accuracy:                                  34.8%
