import random
from sklearn import svm
import timeit
from sklearn.datasets import load_breast_cancer
from plotaco import *
import warnings
import feature_selection_ga as fga
from numpy import matlib
import sys


def baco(best_individual, x_data, y_data, alpha = 1.0, beta = 0.0, t_percent=40, iter_num=10):


    (my_bool, msg_err) = check_baco_args(t_percent, iter_num)
    if(not my_bool):
        print("problem with arguments for mbaco()!!!")
        print(msg_err)
        exit() #############



    train_percentage = 100 - int(t_percent)

    time_temp = 0
    start = timeit.default_timer()
    (best_fitnesses_each_iter, average_fitnesses_each_iter, num_of_features_selected_by_best_ant_each_iter, best_fit_so_far, best_ant_road) = run_feature_selection(best_individual,generations = iter_num, alpha = alpha, beta = beta, T0 = 0.1, T1= 0.2, Min_T = 0.1, Max_T = 6, q = 0.95, Q = 0.3, ant_num = 50, feature_num = len(x_data[1]), dataset=x_data, targets=y_data, train_percentage=train_percentage)
    end = timeit.default_timer()
    time_temp = time_temp + (end - start)


    acc_before_run = get_single_fit(x_data, y_data, train_percentage)

    total_feature_num = len(x_data[1])
    sample_num = len(x_data[:,1])

    best_selected_features_num = np.sum(best_ant_road)
    return (best_ant_road, acc_before_run, best_fit_so_far, total_feature_num, best_selected_features_num, best_fitnesses_each_iter, average_fitnesses_each_iter ,num_of_features_selected_by_best_ant_each_iter, time_temp, sample_num)


def check_baco_args(t_percent, iter_num):
    msg_err = ""
    try:
        int(t_percent)
    except Exception as e:
        msg_err = "t_percent should be integer!"
        return (False, msg_err)

    try:
        int(iter_num)
    except Exception as e:
        msg_err = "iter_num should be integer!"
        return (False, msg_err)

    if(iter_num > 100):
        msg_err = "iter_num should be less than 100!"
        return (False, msg_err)

    if(iter_num < 5):
        msg_err = "iter_num should be more than 5!"
        return (False, msg_err)


    return (True, msg_err)


def run_feature_selection(best_individual, generations, alpha, beta , T0, T1, Min_T, Max_T, q, Q, ant_num, feature_num, dataset, targets, train_percentage):

    best_fitnesses_each_iter = []
    average_fitnesses_each_iter = []
    num_of_features_selected_by_best_ant_each_iter = []
    road_map = np.random.randint(2, size=ant_num*feature_num).reshape((ant_num, feature_num))
    road_maps = np.zeros(ant_num*feature_num*generations, dtype="int64").reshape(generations, ant_num, feature_num)
    best_roads_list = []

    best_fit_so_far = 0
    best_road_so_far = np.zeros(feature_num, dtype="int64")

    np.set_printoptions(suppress=True, threshold=1000)

    pheremones_1 = T0 * np.asarray(best_individual) + T1
    pheremones_1 = np.matlib.repmat(pheremones_1,feature_num,1).reshape(feature_num,feature_num)
    opp_best_individual = np.subtract(np.ones(feature_num),best_individual)
    pheremones_2 = T0 * opp_best_individual + T1
    pheremones_2 = np.matlib.repmat(pheremones_2,feature_num,1).reshape(feature_num,feature_num)
    pheremone = np.vstack((pheremones_2,pheremones_1))
    pheremone = np.vstack((pheremone,pheremone))
    pheremone = pheremone.reshape(4,feature_num,feature_num)
    pheremones= np.zeros(feature_num*feature_num*4, dtype="float64").reshape(4, feature_num, feature_num) + pheremone


    for i in range(0, generations):

        visibility = ascore(feature_num, dataset, targets)


        (road_map, pointer) = baco_road_selection(pheremones, visibility, alpha, beta, ant_num, feature_num)

        (iter_best_fit, best_road_so_far, best_fit_so_far, iter_best_road, fitnesses, iter_average_fit, ants_num_of_features_selected) = do_calculations(road_map, dataset, targets, best_fit_so_far, best_road_so_far, train_percentage)

        pheremones= trial_update(fitnesses, pheremones, Min_T, Max_T, Q, q, iter_best_road, feature_num)

        road_maps[i] = road_map
        best_fitnesses_each_iter.append(iter_best_fit)
        average_fitnesses_each_iter.append(iter_average_fit)
        num_of_features_selected_by_best_ant_each_iter.append(sum(best_road_so_far))
        best_roads_list.append(best_road_so_far)


    ccc = 0
    maxx = max(best_fitnesses_each_iter)
    for each in best_fitnesses_each_iter:
        if(each == maxx):
            my_indx = ccc
        ccc = ccc + 1
    return (best_fitnesses_each_iter, average_fitnesses_each_iter, num_of_features_selected_by_best_ant_each_iter, best_fit_so_far, best_roads_list[my_indx])


def ascore(feature_num, dataset, targets):
    visibility = np.zeros(feature_num*feature_num*4, dtype="float64").reshape(4, feature_num, feature_num)

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


    visibility[0, :, :] = (0.5/feature_num) * sum(Acc_score)
    visibility[1, :, :] = np.matlib.repmat(Acc_score, feature_num, 1)

    visibility[2, :, :] = (0.5/feature_num) * sum(Acc_score)
    visibility[3, :, :] = np.matlib.repmat(Acc_score, feature_num, 1)

    return visibility


def get_accuracy_for_this_solution(train_dataset, train_targets, test_dataset, test_targets):
    lin_clf = svm.LinearSVC()
    lin_clf.fit(train_dataset,train_targets)
    predicted_targets = lin_clf.predict(test_dataset)
    l = len(test_targets)
    num_of_correct = 0
    for i in range(l):
        if(test_targets[i] == predicted_targets[i]):
            num_of_correct = num_of_correct + 1
    return num_of_correct/l


def separate_datasets(dataset, targets, train_percentage):

    # in case you wanted the data to be random every single time you wanted get fitnesses
    leng = len(dataset[:, 0])
    s = int(leng*(train_percentage/100))

    samples_list = random.sample(range(0, leng), s)

    mask = np.zeros((leng), dtype=bool)
    mask[samples_list] = True

    train_dataset = dataset[mask, :]
    test_dataset = dataset[~mask, :]

    train_targets = targets[mask]
    test_targets = targets[~mask]


    return (train_dataset, test_dataset, train_targets, test_targets)


def get_fitnesses(road_map, dataset, targets, train_percentage):
    total_feature_num = len(road_map[1])
    total_sample_num = len(dataset[:,0])
    num_of_features_selected = list()
    fitnesses = list()

    count = 0
    for ant_solution in road_map:
        count = count + 1
        if np.sum(ant_solution) == 0:
            fitnesses.append(0)
        else:
            new_dataset = np.zeros(total_sample_num, dtype="float64").reshape(total_sample_num, 1)

            for i in range(0, len(ant_solution)):
                if(ant_solution[i] == 1):
                    new_dataset = np.append(new_dataset, dataset[:, i].reshape(total_sample_num, 1), axis=1)

            new_dataset = np.delete(new_dataset, 0, axis=1) # removing first column

            num_of_features_selected.append(new_dataset.shape[1])

            (train_dataset, test_dataset, train_targets, test_targets) = separate_datasets(new_dataset, targets, train_percentage)

            fitnesses.append(get_accuracy_for_this_solution(train_dataset, train_targets, test_dataset, test_targets))

    return num_of_features_selected, fitnesses


def get_single_fit(dataset, targets, train_percentage):

    (train_dataset, test_dataset, train_targets, test_targets) = separate_datasets(dataset, targets, train_percentage)

    return get_accuracy_for_this_solution(train_dataset, train_targets, test_dataset, test_targets)


def pick_next_location(probs, feature_num):
    sum = 0
    zero_or_one = 1
    r = np.random.random_sample()
    for x in range(len(probs)):
        sum = sum + probs[x]
        if(r < sum):
            index = x
            # because it is now (feature_num + feature_num) long, we should correct it :
            if(index >= feature_num):
                index = index - feature_num
                zero_or_one = 1
            else:
                zero_or_one = 0
            return (index, zero_or_one)


def baco_road_selection(pheremones, visibility, alpha, beta, ant_num, feature_num):
    road_map = np.zeros(ant_num*feature_num, dtype="int64").reshape(ant_num, feature_num)
    pointer = np.zeros(ant_num*feature_num, dtype="int64").reshape(ant_num, feature_num)

    for k in range(0, ant_num):
        indx = np.multiply(np.power(pheremones, alpha), np.power(visibility, beta))
        for j in range(0, feature_num):

            # for the first feature :
            if(j == 0):
                cur_feature = np.random.randint(0, feature_num, 1)[0]
                pointer[k,j] = cur_feature
                # this is just for selection of 0 or 1 for the first feature (if it's more interesting the likelihood is higher)
                temp = np.sum(pheremones[0, :, cur_feature] + pheremones[2, :, cur_feature]) / np.sum(pheremones[0, :, cur_feature] + pheremones[1, :, cur_feature] + pheremones[2, :, cur_feature] + pheremones[3, :, cur_feature])
                rand = np.random.random_sample()

                if (rand < temp):
                    road_map[k, cur_feature] = 0
                else:
                    road_map[k, cur_feature] = 1

            else:
                if(road_map[k, pointer[k,j-1]] == 1):
                    nominator = np.hstack((indx[2, pointer[k,j-1], :], indx[3, pointer[k,j-1], :]))
                    denominator = sum(nominator) ##################################### should be right!!!!!
                    probability = np.divide(nominator, denominator) # total=total/sum(total) # should be editted.it is not
                    (selected_feature_indx, zero_or_one) = pick_next_location(probability, feature_num)
                    pointer[k,j] = selected_feature_indx


                    if(zero_or_one == 0):
                        road_map[k, pointer[k,j]] = 0
                    else:
                        road_map[k, pointer[k,j]] = 1

                else: # == 0
                    nominator = np.hstack((indx[0, pointer[k,j-1], :], indx[1, pointer[k,j-1], :]))
                    denominator = sum(nominator)
                    probability = np.divide(nominator, denominator)
                    (selected_feature_indx, zero_or_one) = pick_next_location(probability, feature_num)
                    pointer[k,j] = selected_feature_indx


                    if(zero_or_one == 0):
                        road_map[k, pointer[k,j]] = 0
                    else:
                        road_map[k, pointer[k,j]] = 1

            # update indx (so by doing this, the probability of selection for this feature, is gonna be zero!)
            indx[:, :, pointer[k, j]] = 0
    return (road_map, pointer)


def do_calculations(road_map, dataset, targets, best_fit_so_far, best_road_so_far, train_percentage):

    ants_num_of_features_selected, fitnesses = get_fitnesses(road_map, dataset, targets, train_percentage)

    iter_average_fit = np.mean(fitnesses, axis=0)
    iter_best_fit = max(fitnesses)
    iter_best_ant = fitnesses.index(iter_best_fit)
    iter_best_road = road_map[iter_best_ant, :]

    if(iter_best_fit > best_fit_so_far):
        best_fit_so_far = iter_best_fit
        best_road_so_far = iter_best_road
    return (iter_best_fit, best_road_so_far, best_fit_so_far, iter_best_road, fitnesses, iter_average_fit, ants_num_of_features_selected)


def trial_update(fitnesses, pheremones, Min_T, Max_T, Q, q, iter_best_road, feature_num):

    pheremones= pheremones* q #pheromone evaporation
    # class_err = 1 - fitnesses # not this because fitnesses is a list and doesn't work this way
    class_err = np.array([1-i for i in fitnesses])
    min_err = min(class_err)
    min_err_indx = np.where(class_err == min_err)[0][0]
    max_fit = max(fitnesses)
    # max_fit_indx = np.where(fitnesses == max_fit)[0][0]


    change_pheremones = np.zeros(feature_num*feature_num*4, dtype="float64").reshape(4, feature_num, feature_num)

    # here we assign one to best road edges in change_pheremones.
    for i in range(0, len(iter_best_road)):
        if(iter_best_road[i] == 0):
            change_pheremones[0, :, i] = 1
            change_pheremones[2, :, i] = 1
        else:
            change_pheremones[1, :, i] = 1
            change_pheremones[3, :, i] = 1


    # if(class_err[min_err_indx] == 0):
    #     change_pheremones = (Q/(class_err[min_err_indx] + 0.001)) * change_pheremones
    # else:
    #     change_pheremones = (Q/(class_err[min_err_indx])) * change_pheremones

    if(max_fit == 0):
        change_pheremones = (1/(max_fit+0.001)) * change_pheremones
    else:
        change_pheremones = (1/(max_fit)) * change_pheremones

    pheremones= pheremones+ change_pheremones
    # now we make sure all of them are in interval :
    for each in np.nditer(pheremones, op_flags=['readwrite']):
        if(each > Max_T):
            each[...] = Max_T
        else:
            if(each < Min_T):
                each[...] = Min_T


    return pheremones


alpha = float(input("Enter alpha : "))
beta = float(input("Enter beta : "))
iris = load_breast_cancer()
x_train = iris.data
y_train = iris.target
model = svm.LinearSVC()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fsga = fga.FeatureSelectionGA(model,x_train,y_train)
    (pop,best,fitness_value) = fsga.generate(20)
    solution = baco(best, x_train, y_train,alpha=alpha,beta=beta, t_percent=40, iter_num=10)
    (best_ant_road, acc_before_run, best_fit_so_far, total_feature_num, best_selected_features_num,
     best_fitnesses_each_iter, average_fitnesses_each_iter, num_of_features_selected_by_best_ant_each_iter, time_temp,
     sample_num) = solution
    if fitness_value[0] < best_fitnesses_each_iter[-1]:
        print("Binary Representation of Feature selection : ",best_ant_road)
        print("Total number of features in Dataset : ", total_feature_num)
        print("Number of features selected : ",num_of_features_selected_by_best_ant_each_iter[-1])
        print("Accuracy before MBACO : ",acc_before_run)
        print("Accuracy with best fit : ",best_fit_so_far)
        print("Accuracy of best ant in last iteration : ",best_fitnesses_each_iter[-1])
        draw_baco(solution)
    else:
        print("GA gave the best individual which is",best," (",fitness_value,")")
