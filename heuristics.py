import numpy as np
import sklearn

def hueristic_value_ascore(feature_num, dataset, targets):
    roads_E = np.zeros(feature_num*feature_num*4, dtype="float64").reshape(4, feature_num, feature_num)

#     arr = np.corrcoef(dataset)
#     R = abs(arr)

    ## F-score :
    classes = np.unique(targets)
    class_num = len(classes)
    total_mean_a = dataset.mean(0)
    nominator = 0
    denominator = 0

#     nominator = np.zeros(feature_num, dtype="int64")
#     denominator = np.zeros(feature_num, dtype="int64")

    sample_num_of_this_tag = np.zeros(class_num, dtype="int64")
    for i in range(0, class_num):
        tags = np.zeros((len(targets)), dtype="int64")
        bool_arr = np.equal(targets, classes[i])
        tags[bool_arr] = 1
        sample_num_of_this_tag[i] = np.sum(tags)
        dataset_only_class = dataset[bool_arr, :]
        class_mean_a = dataset_only_class.mean(0)
        class_mean_a = np.round(class_mean_a, decimals=4)


        nominator = nominator + np.power(np.subtract(class_mean_a, total_mean_a), 2)
        denominator = denominator + sum(np.power(np.subtract(dataset_only_class, np.matlib.repmat(total_mean_a, dataset_only_class.shape[0],1)), 2)) / (sample_num_of_this_tag[i]-1)

    Acc_score = np.divide(nominator, denominator)


    roads_E[0, :, :] = (0.5/feature_num) * sum(Acc_score)
    roads_E[1, :, :] = np.matlib.repmat(Acc_score, feature_num, 1)

    roads_E[2, :, :] = (0.5/feature_num) * sum(Acc_score)
    roads_E[3, :, :] = np.matlib.repmat(Acc_score, feature_num, 1)

    return roads_E

