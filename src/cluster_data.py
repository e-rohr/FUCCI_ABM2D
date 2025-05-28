import numpy as np
import yaml
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.feature_extraction import split_data




def generate_param_labels(param_combinations, labels, n_iterations=5, n_param_value_combos=100):
    """
    Assigns the most common cluster label across multiple iterations to each parameter value combo.

    Parameters
    ----------
    param_combinations : list of str
        List of parameter combination names.
    labels : dict
        Dictionary of shape (combo_name → [param_value_combo × iteration]) with cluster labels.
    n_iterations : int
        Number of clustering iterations.
    n_param_value_combos : int
        Number of parameter value combinations per pair.

    Returns
    -------
    param_labels : dict
        Dictionary of shape (combo_name → [param_value_combo]) with dominant cluster label.
    margins : dict
        Empty dictionary (can store confidence margins if enabled).
    """

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

def clustering_pipeline(X, descriptor_name, title, scaler, descriptor_size, num_clusters=4):
    """
    Full pipeline to preprocess data, perform PCA and k-means clustering, and evaluate
    cluster assignments using a confusion matrix based on true parameter combinations.

    Parameters
    ----------
    X : np.ndarray
        Full dataset to cluster.
    descriptor_name : str
        Name used for saving outputs.
    title : str
        Title for plots.
    scaler : function
        Function to scale the data.
    descriptor_size : int
        Number of features per sample after flattening.
    num_clusters : int
        Number of clusters for k-means.

    Returns
    -------
    results : dict
        Dictionary with keys:
            'X_reduced' : PCA-reduced representation of X.
            'X_labels' : Cluster labels assigned to each sample.
            'explained_variance_ratio' : PCA variance explained.
    """
    
    # Initialize steps in the pipeline
    pca = PCA(n_components = 3, random_state = 0)
    kmeans = KMeans(n_clusters = num_clusters,n_init = 'auto', random_state = 0)
    pipeline = Pipeline(steps=[("pca", pca), ("kmeans", kmeans)])
    
    X_train, X_test = split_data(X, itr_cutoff = 5)
    X_train, X_test, Xt = scaler(X_train, X_test, X)
    
    X_train = X_train.reshape((num_samples // 2, descriptor_size))
    X_test = X_test.reshape((num_samples // 2, descriptor_size))
    Xt = Xt.reshape((num_samples, descriptor_size))
    
    X_train_labels = pipeline.fit_predict(X_train)
    X_predicted_test_labels = pipeline.predict(X_test)
    X_labels = pipeline.predict(Xt)
    X_reduced = pca.transform(Xt)
    
    centers = kmeans.cluster_centers_

    with open("../src/parameters.yaml") as p:
            params = yaml.safe_load(p)
            dataset2_info = params["dataset2"]
    param_combinations = dataset2_info['parameter_combinations']

    # Reformatting the training labels into a dictionary for later use 
    labels = {}
    for i, combo in enumerate(param_combinations):
        labels[combo] = X_train_labels[i*605:(i+1)*605].reshape((121,5))
    
   
        
    # Assign most common label among 5 training iterations to each parameter region 
    param_labels, _ = generate_param_labels(param_combinations = param_combinations,
                                            labels = labels,
                                            n_iterations = 5,
                                            n_param_value_combos = 121)

    
    ## Create a vector of c_a values corresponding to all training data
    c_a_vec = []
    c_a_range = np.logspace(np.log2(0.2), np.log2(0.8), 11, base = 2)
    for i, combo in enumerate(param_combinations):
        if 'c_a' in combo.split(','):
            c_a_vec.append(np.repeat(c_a_range,11))
        else:
            c_a_vec.append(np.full(shape = 121,
                               fill_value = 0.4))
    c_a_vec = np.array(c_a_vec).flatten()
    
    
    ## Computing the mean value of c_a over the flattened array of labels
    common_label = []
    for combo in param_labels.keys():
        common_label.append(param_labels[combo])
    common_label = np.array(common_label, dtype = int).flatten()

    c_a_means = np.zeros(num_clusters)
    for c in np.arange(num_clusters):
        c_a_means[c] = np.mean(c_a_vec[common_label == c])

    
    common_label_copy = np.zeros(common_label.shape)
    X_train_labels_copy = np.zeros(X_train_labels.shape)
    X_predicted_test_labels_copy = np.zeros(X_predicted_test_labels.shape)
    X_labels_copy = np.zeros(X_labels.shape)
    centers_copy = np.copy(centers)

    # Applying the new ordering to each array of labels 
    for iold, inew in enumerate(np.argsort(c_a_means)):
        common_label_copy[common_label == inew] = iold
        X_train_labels_copy[X_train_labels == inew] = iold
        X_predicted_test_labels_copy[X_predicted_test_labels == inew] = iold
        X_labels_copy[X_labels == inew] = iold
        centers_copy[iold] = centers[inew]
    np.save(f"../results/dataset2/{descriptor_name}_centers", centers_copy, allow_pickle = True)
    
    # Create confusion matrix with labels based on parameter values as "ground truth" and k-means labels as predicted labels
    cm = generate_confusion_matrix(param_labels = np.repeat(common_label_copy, 5),
                                   kmeans_predict_labels = X_predicted_test_labels_copy,
                                   n_clusters = num_clusters)
    
    # Plot the confusion matrix and display out-of-sample accuracy for each cluster
    cm_fig = plot_confusion_matrix_and_OOS(confusion_matrix = cm,
                                           y_true = np.repeat(common_label_copy, 5),
                                           y_pred = X_predicted_test_labels_copy,
                                           n_clusters = num_clusters,
                                           xlabel = "Group predicted by k-means",
                                           ylabel = "Group based on true parameter values",
                                           title = title);
    cm_fig.savefig(f"../figures/dataset2/{descriptor_name}_confusion_matrix.png")
    
    
    return {'X_reduced': X_reduced, 'X_labels': X_labels_copy, 'explained_variance_ratio': pca.explained_variance_ratio_}
    

def generate_confusion_matrix(param_labels, kmeans_predict_labels, n_clusters):
    """
    Computes and normalizes the confusion matrix between parameter-inferred and predicted labels.

    Parameters
    ----------
    param_labels : np.ndarray
        Ground-truth labels inferred from parameter combinations.
    kmeans_predict_labels : np.ndarray
        Predicted labels from k-means.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    cm : np.ndarray
        Normalized confusion matrix.
    """
    ...
    #create confusion matrix
    cm = confusion_matrix(param_labels, kmeans_predict_labels, labels=np.arange(n_clusters))
    #normalize
    cm_sum = cm.sum(axis=1)
    cm = cm / cm_sum[:,np.newaxis]
    cm[np.isnan(cm)] = 0
    
    return cm

def plot_confusion_matrix_and_OOS(confusion_matrix, y_true, y_pred, n_clusters, xlabel, ylabel, title):
    """
    Visualizes the normalized confusion matrix and annotates it with OOS accuracy.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Normalized confusion matrix.
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    n_clusters : int
        Number of clusters.
    xlabel : str
        Label for x-axis.
    ylabel : str
        Label for y-axis.
    title : str
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure with annotated confusion matrix.
    """
    ...
    
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


