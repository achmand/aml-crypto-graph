"""
A script which exposes plotting functionality used in the experiments. 
The following script included the following functionality;
"""

# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(matrices, 
                          titles=None, 
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
       