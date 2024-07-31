import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable


def apply_KMeans(Z_train, Z_test, n_clusters = 4, random_state = 57):
    kmeans = KMeans(n_clusters = 4,n_init = 'auto', random_state = random_state)
    
    kmeans.fit(Z_train)
    
    train_labels = kmeans.predict(Z_train)
    test_labels = kmeans.predict(Z_test)
    
    return train_labels, test_labels, kmeans

def generate_param_labels(param_combinations, labels, n_iterations = 5, n_param_value_combos = 100):
    '''
    generate_param_labels : Assign a class labels to each parameter combination. Each combination has 5 iterations: assign the most common class label to the overall combination.

    inputs:
    
        param_combinations: list of combinations of parameters (order matters)
        labels : dictionary, with keys matching param_combinations, where associated values are numpy arrays containing class labels for each observation.
                 Numpy array shape = (n_param_value_combinations , n_iterations) 
        n_iterations : # of iterations per combination of parameter values
        n_param_value_combinations : # of combinations of parameter values for a given combination of parameters
        
    outputs:

        param_labels : dictionary containing class label for each value combination for each parameter combination
    
    '''
    labels = labels.copy()
    
    param_labels = {}
    margins = {}
    
    # Iterate over each combination of perturbed parameters e.g. (c_a,eta1), (c_a,eta2), ...
    for param_combination in param_combinations:
        
        # Initialize array of labels for each pair of parameter values
        param_labels[param_combination] = np.zeros(n_param_value_combos, dtype = int)
        
        # margins[param_combination] = np.zeros(n_param_value_combos)
        # Iterate over each pair of parameter values for the current pair of perturbed parameters
        for param_value_combo in range(n_param_value_combos):
            # Count how many times each label is assigned to samples with this pair of parameter values
            label_counts = np.bincount(labels[param_combination][param_value_combo, :])
            
            # Take the most common label as the overall label for this pair of parameter values
            param_labels[param_combination][param_value_combo] = np.argmax(label_counts)
            
            # margins[param_combination][param_value_combo] = label_counts[np.argmax(label_counts)]/np.sum(label_counts)
            
    return param_labels, margins

        
    

def generate_confusion_matrix(param_labels, kmeans_predict_labels, n_clusters):
    '''
    generate_confusion_matrix : Create normalized confusion matrix from parameter class labels

    inputs:
    
        param_labels : dictionary containing class label for each value combination for each parameter combination
                        {key = (param1_name,param2_name) : value = numpy.array(n_param_value_combinations)}
        kmeans_predict_labels : dictionary, with keys matching param_combinations, where associated values are numpy arrays containing class labels for each observation.
                 Numpy array shape = (n_param_value_combinations , n_iterations) 
        n_clusters : # of clusters used by k-means algorithm
    outputs:

        
    
    '''
    #create confusion matrix
    cm = confusion_matrix(param_labels, kmeans_predict_labels, labels=np.arange(n_clusters))
    #normalize
    cm_sum = cm.sum(axis=1)
    cm = cm / cm_sum[:,np.newaxis]
    cm[np.isnan(cm)] = 0
    
    return cm


def plot_confusion_matrix_and_OOS(confusion_matrix, y_true, y_pred, n_clusters, xlabel, ylabel, title):
    acc = accuracy_score(y_true,y_pred)
    font = {'size'   : 20}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    #plot confusion matrix
    im = plt.matshow(confusion_matrix,cmap = plt.cm.Blues,vmin=0,vmax=1,fignum=0)
    if title is not None:
        plt.title(title + " confusion matrix,\n "+str(round(acc*100,1))+"% OOS Accuracy",fontsize=18,pad=-3)
    else:
        plt.title("Confusion Matrix, ("+str(round(acc*100,1))+"% OOS Accuracy) \n ")
    
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    ax.xaxis.tick_bottom()
    #ax.set_xticklabels(np.arange(num_clusters))
    #ax.set_yticklabels(np.arange(num_clusters))
    
    

    for i in np.arange(n_clusters):
        for j in np.arange(n_clusters):
            if confusion_matrix[j,i] < .8:
                plt.text(j,i, str(round(confusion_matrix[i,j]*100,0))+"%", va='center', ha='center',color="black")
            else:
                plt.text(j,i, str(round(confusion_matrix[i,j]*100,0))+"%", va='center', ha='center',color="white")
    
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad = 0.5)
    
    plt.colorbar(im, cax=cax)
    
    return fig
            
                
    
    
    
    
    