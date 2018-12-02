import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """    
    uo = true_w
    vc = inputs

    # doing the matrix multiplication uoTvc
    uoTvc = tf.tensordot(vc, tf.transpose(uo), 1)

    # we want the elements on the diagonal of the resultant matrix
    A = tf.diag_part(uoTvc)

    exp_B = tf.exp(uoTvc)
    sum_exp = tf.reduce_sum(exp_B, axis=1)
    B = tf.log(sum_exp)

    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    import numpy as np
    k = len(sample)
    
    # positive data
    uc = inputs
    uo = tf.reshape(tf.gather(weights, labels), [-1, tf.shape(weights)[1]])
    bo = tf.reshape(tf.gather(biases, labels), [-1,1])
    unigram_prob_1 = np.float32(unigram_prob) # unigram_prob is a list, convert to np array for gather
    uni_prob = tf.gather(unigram_prob_1, labels)

    # we want the elements on the diagonal of the matrix product
    ucTuo_diag = tf.reshape(tf.diag_part(tf.tensordot(uc,tf.transpose(uo),1)), [-1,1])
    s1 = tf.add(ucTuo_diag, bo)
    log_term1 = tf.log(tf.scalar_mul(k,uni_prob)+1e-10) # adding small val to handle nan
    first_term = tf.log(tf.sigmoid(s1-log_term1)+1e-10)

    # negative data
    uc = inputs    
    ux = tf.gather(weights, sample)
    bx = tf.gather(biases, sample)
    bx = tf.reshape(bx, [-1,1])
    neg_uni_prob = tf.reshape(tf.gather(unigram_prob_1, sample), [-1,1])

    s2 = tf.add(tf.transpose(tf.tensordot(uc,tf.transpose(ux),1)), bx)
    log_term2 = tf.log(tf.scalar_mul(k,neg_uni_prob)+1e-10)
    second_term = tf.sigmoid(tf.transpose(s2-log_term2))
    final_second_term = tf.reduce_sum(tf.log(1.0-second_term+1e-10), 1)

    result = tf.negative(tf.add(first_term, final_second_term))
    return result