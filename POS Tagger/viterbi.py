import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    # SHAPES 
    # N = 5, L = 3
    # emission_scores = (5,3), trans_scores = (3,3)
    # start_scores = (3,), end_scores = (3,)

    # Creating the transition DP matrix
    T = [[0 for _ in range(N)] for _ in range(L)]
    backpointers = [[0 for _ in range(N)] for _ in range(L)]

    # Filling the first column
    for row in range(L):
        T[row][0] = emission_scores[0][row] + start_scores[row] # emission_scores matrix is (N X L)
        
    # Filling the rest of the transition matrix
    for col in range(1, N):
        for row in range(L):
            prev_list = []
            for prev_label in range(L):
                prev_list.append(trans_scores[prev_label, row] + T[prev_label][col-1])
            T[row][col] = max(prev_list) + emission_scores[col][row] 
            backpointers[row][col] = np.argmax(prev_list)

    # Filling the last column
    for row in range(L):
        T[row][N-1] += end_scores[row]

    # print for debug
    # print "T"
    # for i in T:
    #     print i
    
    # print 
    # print

    # print "B"
    # for i in backpointers:
    #     print i

    # Finding max score in last column of T matrix
    T = np.array(T)
    score = np.asscalar(np.max(T[:,N-1]))
    location = np.asscalar(np.argmax(T[:,N-1]))

    # Getting best sequence from right to left using backpointers
    y = [location]
    for col in range(N-1, 0, -1):
        y.insert(0, backpointers[location][col])
        location = backpointers[location][col]

    '''
    y = []
    for i in xrange(N):
        # stupid sequence
        y.append(i % L)
    # score set to 0
    return (0.0, y)
    '''
    return (score, y)