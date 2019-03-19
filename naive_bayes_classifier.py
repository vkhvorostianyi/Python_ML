class Naive_Bayes(object):
    """
    Parameters:
    -----------
    alpha: int
        The smoothing coeficient.
    """
    def __init__(self, alpha):
        self.alpha = alpha
        
        self.train_set_x = None
        self.train_set_y = None
        
        self.all_words_list = []
        self.ham_words_list = []
        self.spam_words_list = []
    
    def fit(self, train_set_x, train_set_y):
        
        # Generate all_words_list, ham_words_list, spam_words_list using function 'categories_words'; 
        # Calculate probability of each word in both categories
        ### START CODE HERE ### 
        all_words_list, ham_words_list, spam_words_list = categories_words(train_set_x, train_set_y)
        
        self.train_set_x = train_set_x
        self.train_set_y = train_set_y
        
        self.all_words_list = all_words_list
        self.ham_words_list = ham_words_list
        self.spam_words_list = spam_words_list
        
        unique, counts = np.unique(ham_words_list, return_counts=True)
        ham_count = dict( zip(unique, counts) )
        unique, counts = np.unique(spam_words_list, return_counts=True)
        spam_count = dict( zip(unique, counts) )
        
        unique, counts = np.unique(train_set_y, return_counts=True)
        y_count = dict(zip(unique, counts))
        
        self.prob_ham = y_count['ham'] / len(train_set_y)
        self.prob_spam = y_count['spam'] / len(train_set_y)
        N = len(self.all_words_list)
        k1 = len(self.ham_words_list)
        k2 = len(self.spam_words_list)
        self.probs_ham = {word: (ham_count.get(word, 0) + self.alpha) / (N + self.alpha * k1) for word in all_words_list}
        self.probs_spam = {word: (spam_count.get(word, 0) + self.alpha) / (N + self.alpha * k2) for word in all_words_list}
        
        ### END CODE HERE ### 
        
    def predict(self, test_set_x):
        
        # Return list of predicted labels for test set; type(prediction) -> list, len(prediction) = len(test_set_y)
        ### START CODE HERE ###
        N = len(self.all_words_list)
        k1 = len(self.ham_words_list)
        k2 = len(self.spam_words_list)
        prediction = [
            'ham' if np.log(self.prob_ham) + sum(map(np.log, map(lambda x: self.probs_ham.get(x, self.alpha / (N + self.alpha * k1)), sentence) )) >
            np.log(self.prob_spam) + sum(map(np.log, map(lambda x: self.probs_spam.get(x, self.alpha / (N + self.alpha * k2)), sentence) )) 
            else 'spam'
          for sentence in test_set_x
        ]
            
        ### END CODE HERE ### 
        return prediction
