"""
A script which exposes plotting functionality used in the experiments. 
The following script included the following functionality;
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from IPython.core.display import display, HTML

def plot_feature_imp(results, N, figsize=(17,10)):
    for model_key, model_value in results.items():
        for feature_set, feature_set_value in model_value.items():       
            title = "'{}' on '{}' feature set - TOP {} Features".format(model_key, feature_set, N)
            sorted_imp = feature_set_value["importance"].sort_values("importance", ascending=False) 
            ax = sorted_imp.head(N).plot.barh(rot=0, title=title, figsize=figsize)
            plt.show()
            title = "'{}' on '{}' feature set - BOTTOM {} Features".format(model_key, feature_set, N)
            ax = sorted_imp.tail(N).plot.barh(rot=0, title=title, figsize=figsize)
            plt.show()
            display(HTML("</hr>"))

def plot_result_matrices(results, figsize, columns=2):

    # loop and extract confusion matrices 
    confusion_matrices = []
    for model_key, model_value in results.items():
        for feature_set, feature_set_value in model_value.items():
            confusion_matrix = feature_set_value["metrics"]["confusion"]
            tn, fp, fn, tp = confusion_matrix.ravel()
            tpr = tp / (tp+ fn)
            tnr = tn / (tn + fp)
            plot_title = "'{}' on '{}' feature set with \nTPR: {} \nTNR: {}".format(model_key,feature_set, round(tpr,4), round(tnr,4))
            confusion_matrices.append((plot_title, confusion_matrix))
        
    # display plots 
    plot_confusion_matrix(matrices=confusion_matrices, figsize=figsize, columns=columns)

def plot_confusion_matrix(matrices, 
                          figsize=(17,15), 
                          columns=2, 
                          font_scale=1.2, 
                          hspace=0.4, 
                          wspace=0.4):
    
    total = len(matrices)
    if columns>total:
        columns = total
    
    rows = int(total/columns) 

    current_plot = 1
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=hspace, wspace=wspace)  
    sns.set(font_scale=font_scale)
    for m in matrices:
        title = ""
        matrix = m 
        if isinstance(m, tuple):
            title = m[0]
            matrix = m[1]
            
        ax = fig.add_subplot(rows, columns, current_plot)
        sns.heatmap(matrix, annot=True, cmap="Blues", fmt="g")
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        current_plot+=1

def elliptic_time_indexed_results(results):

    # loop and extract time indexed f1 score for each feature set 
    defaults_results_time_results = {}
    for model_key, model_value in results.items():
        for feature_set, feature_set_value in model_value.items(): 
            tmp_time_metrics = feature_set_value["time_metrics"]
            if feature_set not in defaults_results_time_results:
                defaults_results_time_results[feature_set] = {}
                defaults_results_time_results[feature_set]["scores"] = []
                defaults_results_time_results[feature_set]["timestep"] = tmp_time_metrics["timestep"]
                defaults_results_time_results[feature_set]["total_pos_label"] = tmp_time_metrics["total_pos_label"]
            defaults_results_time_results[feature_set]["scores"].append((model_key, tmp_time_metrics["score"]))

    # plot results over test time span 
    for feat_key, time_results in defaults_results_time_results.items():
        plot_title = "Illicit F1 results over test time span using '{}' feature set".format(feat_key)
        plot_time_indexed_results(time_steps=time_results["timestep"],
                                indexed_total_samples=time_results["total_pos_label"],
                                indexed_scores=time_results["scores"],
                                metric_title="Illicit F1",
                                plot_title=plot_title)

COLORS = ["red", "green", "purple", "orange", "pink", "blue", "black"]
MARKERS = ["o", "p", "D", "*", "X", "+", "s"]
def plot_time_indexed_results(time_steps, 
                              indexed_total_samples,
                              indexed_scores,
                              metric_title,
                              plot_title,
                              figsize=(16,6),
                              font_scale=1.2):
    
    # maximum models to plot must no exceed 7, as the plot becomes to cluttered  
    total_instances = len(indexed_scores)
    if total_instances > len(MARKERS):
        raise ValueError("Cannot plot more than 7 different types of scores")

    # set properties/style for plot 
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "lightblue"
    plt.rcParams["axes.linewidth"]  = 1.25
    fig = plt.figure(figsize=figsize)

    # add bar plot which shows total samples
    ax1 = fig.add_subplot(111)
    ax1 = sns.barplot(x=time_steps, y=indexed_total_samples, color="lightblue")
    ax1.grid(False)    
    ax1.set_xlabel("Time Index")
    ax2 = ax1.twinx()
    ax1.grid(False)
    ax1.set_ylabel("Num samples")
    ax1.yaxis.tick_right()   
    ax1.yaxis.set_label_position("right")
    ax1.tick_params(axis=u'both', which=u'both',length=0)

    # add line plot to show score for different models 
    ax2.set_ylabel(metric_title)
    ax2.yaxis.set_label_position("left")
    ax2.yaxis.tick_left()
    ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1])
    ax2.set_ylim(0.0, 1)

    # add scores to score plot 
    i = 0
    legend_items = []
    legend_items.append(mlines.Line2D([], 
                                      [],
                                      color="lightblue",
                                      marker="s",
                                      linestyle="None",
                                      markersize=8,
                                      label="# Total Samples"))
    for s in indexed_scores: 
        tmp_color = COLORS[i]
        tmp_marker = MARKERS[i]
        legend_item = mlines.Line2D([], 
                                    [], 
                                    color=tmp_color, 
                                    marker=tmp_marker, 
                                    linestyle="None",
                                    markersize=8, 
                                    label=s[0])      
        legend_items.append(legend_item)
        ax2.plot(s[1],markersize=8, marker=tmp_marker, color=tmp_color)
        i+=1
        
    ax2.grid(True, axis="y", color="lightblue")    
    ax2.tick_params(axis=u'both', which=u'both',length=0)
    ax2.set_title(plot_title)

    # show legend and plot 
    plt.legend(handles=legend_items, 
               loc="upper center", 
               bbox_to_anchor=(0.5, -0.1),
               fancybox=True, 
               shadow=True, 
               ncol=total_instances + 1)
    plt.show()

def plot_metric_dist(results, metric, figsize, font_scale=1.2):
    for model_key, model_value in results.items():
        for feature_set, feature_set_value in model_value.items():       
            fig = plt.figure(figsize=figsize)
            title = "'{}' on '{}' feature set - distribution plot for '{}' metric".format(model_key, feature_set, metric)
            sns.set(font_scale=font_scale)
            ax = sns.distplot(feature_set_value["metrics_iterations"][metric], rug=True, hist=False).set_title(title)
            plt.show()
