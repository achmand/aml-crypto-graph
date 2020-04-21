"""
A script which exposes plotting functionality used in the experiments. 
The following script included the following functionality;
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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

COLORS = ["red", "green", "purple", "orange", "pink", "blue", "black"]
MARKERS = ["o", "p", "D", "*", "X", "+", "s"]
def plot_time_indexed_results(time_steps, 
                              indexed_total_samples,
                              indexed_scores,
                              metric_title,
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

    # show legend and plot 
    plt.legend(handles=legend_items, 
               loc="upper center", 
               bbox_to_anchor=(0.5, -0.1),
               fancybox=True, 
               shadow=True, 
               ncol=total_instances + 1)
    plt.show()
    