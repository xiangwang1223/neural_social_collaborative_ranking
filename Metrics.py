import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate_auc(all_prediction_score, testRatings, elimateRatings):
    valid_users = []
    n_valid_users = len(testRatings)
    n_all_users, n_items = all_prediction_score.shape

    auc = 0.
    for user in testRatings.keys():
        per_elimate_ratings = elimateRatings[user]

        # print per_elimate_ratings
        y_true = np.zeros(shape=(1,n_items))
        y_score = np.asarray(all_prediction_score[user])

        # print (y_true.shape, type(y_true))
        # print (y_score.shape, type(y_score))

        test_gnd = testRatings[user]
        y_true[0,test_gnd] = 1

        y_true = np.squeeze(np.delete(y_true, per_elimate_ratings, 1))
        y_score = np.squeeze(np.delete(y_score, per_elimate_ratings, 1))

        # print (y_true.shape, type(y_true))
        # print (y_score.shape, type(y_score))
        try:
            auc += roc_auc_score(y_true, y_score)
        except Exception:
            # print (y_true.shape, type(y_true), max(y_true), min(y_true))
            # print (y_score.shape, type(y_score), max(y_score), min(y_score))
            continue
    auc = auc/n_valid_users

    return auc

def evaluate_pre_rec_auc(all_prediction_score, testRatings, Ks=[5,10,15,20,25]):
    valid_users = []
    n_valid_users = len(testRatings)
    n_all_users, n_items = all_prediction_score.shape

    Y_gnd = np.zeros(shape=(n_valid_users, n_items))

    count = 0
    for user in testRatings.keys():
        user_gnd = testRatings[user]
        for gnd in user_gnd:
            Y_gnd[count, gnd] = 1
        valid_users.append(user)
        count += 1

    Y_predicted = np.squeeze(np.asarray(all_prediction_score[valid_users]))
    precision_vec, selection_vec, recall_vec = PrecisionSelection(Y_gnd=Y_gnd, Y_predicted=Y_predicted, Ks=Ks)
    return precision_vec, selection_vec, recall_vec


# The Implementation of Precision@K and Selection@K
def PrecisionSelection(Y_gnd, Y_predicted, Ks=range(1, 11)):
    Y_gnd = np.asmatrix(Y_gnd)
    Y_predicted = np.asmatrix(Y_predicted)

    instance_num, label_num = Y_gnd.shape

    # Fetch the sorted predicted Y;
    sorted_Y, sorted_index = sortMatrix(Y_predicted)

    precision_vec = []
    selection_vec = []
    recall_vec = []

    for k in Ks:
        t_score = 0.
        s_score = 0.
        r_score = 0.

        for instance in range(0, instance_num):
            instance_flag = 0.
            r_instance_score = 0.
            for k_index in range(0, k):
                # if sorted_Y[instance, k_index] <= 0:
                #     break

                check_index = int(sorted_index[instance, k_index])

                if Y_gnd[instance, check_index] == 1:
                    t_score += 1.
                    r_instance_score += 1.
                    instance_flag = 1.

            s_score  += instance_flag
            r_score += r_instance_score/np.count_nonzero(Y_gnd[instance, ])

        precision_vec.append(float(t_score)/((k)*instance_num))
        selection_vec.append(float(s_score)/instance_num)
        recall_vec.append(float(r_score)/instance_num)

    return precision_vec, selection_vec, recall_vec

# The Implementation of sorting each rows of matrix;
# Output: the sorted matrix and its correspondence of index;
def sortMatrix(X, axis = 1, descend = True):
    X = np.asmatrix(X)

    instance_num, label_num = X.shape

    sorted_matrix = np.asmatrix(np.zeros([instance_num, label_num]))
    sorted_index = np.asmatrix(np.zeros([instance_num, label_num]))
    # Fetch the sorted (descending or ascending) index of the given matrix;
    if descend:
        # Fetch the sorted matrix;
        for instance in range(0, instance_num):
            sorted_index[instance, :] = np.argsort(-X[instance, :], axis=axis)
            sorted_matrix[instance, :] = - np.sort(-X[instance, :])
    else:
        # Fetch the sorted (ascending) matrix;
        for instance in range(0, instance_num):
            sorted_index[instance, :] = np.argsort(X[instance, :], axis=axis)
            sorted_matrix[instance, :] = np.sort(X[instance, :])

    return sorted_matrix, sorted_index