import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import os
import numpy as np



def show_res_for_this_run(best_fitnesses_each_iter, average_fitnesses_each_iter, num_of_features_selected_by_best_ant_each_iter, feature_num):

    iterations = np.arange(1,len(best_fitnesses_each_iter)+1, dtype="int64")

    # Spacing between each line
    intervals = 1
    loc = plticker.MultipleLocator(base=intervals)

    # ax.xaxis.set_major_locator(loc)

    ##################################
    fig, ax1 = plt.subplots(figsize=(10,8))

    plt.subplot(221)


    xx1 = np.array(iterations)
    yy1 = np.array(best_fitnesses_each_iter)

    plt.plot(xx1, yy1, 'bo', xx1, yy1, 'k')

    plt.xlabel('iteration num')
    plt.ylabel('accuracy (fitness)')
    plt.title('Visualization of Accuracy over each Iteration')

    ax1 = fig.gca()
    ax1.xaxis.set_major_locator(loc)
    plt.grid(True)



    ##################################
    plt.subplot(222)

    xx2 = np.array(iterations)
    yy2 = np.array(average_fitnesses_each_iter)

    plt.plot(xx2, yy2, 'bo', xx2, yy2, 'k')

    plt.xlabel('iteration num')
    plt.ylabel('average accuracy')
    plt.title('Visualization of Average of Accuracy over each Iteration')

    ax2 = fig.gca()
    ax2.xaxis.set_major_locator(loc)

    ##################################
    # plt.subplot(223)
    #
    # N = len(num_of_features_selected_by_best_ant_each_iter)
    #
    # ind = np.arange(N)  # the x locations for the groups
    # width = 0.25       # the width of the bars

    # ax3 = fig.gca()
    # rects = ax3.bar(ind, num_of_features_selected_by_best_ant_each_iter, width, color='c')
    #
    #
    # ax3.set_ylabel('num of selected features (by best ant)')
    # # ax3.set_title('selected features over each iteration')
    # ax3.set_xticks(ind + width / 2)
    # ax3.set_xticklabels(np.arange(1, N+1))
    # ax3.set_ylim([0, feature_num])

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            # ax3.text(rect.get_x() + rect.get_width()/2., 1.05*height,
            #         '%d' % int(height),
            #         ha='center', va='bottom')

    # autolabel(rects)

    plt.show()

def draw_baco(solution):

    if(len(solution) != 10):
        print("+++ can't draw the solution due to problem with it! +++")
        return

    (best_ant_road, acc_before_run, best_fit_so_far, total_feature_num, best_selected_features_num, best_fitnesses_each_iter, average_fitnesses_each_iter ,num_of_features_selected_by_best_ant_each_iter, time_temp, sample_num) = solution

    show_res_for_this_run(best_fitnesses_each_iter, average_fitnesses_each_iter, num_of_features_selected_by_best_ant_each_iter, total_feature_num)