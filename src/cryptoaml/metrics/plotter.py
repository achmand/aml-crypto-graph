import seaborn as sns

def confusion_plt(data, title):
    ax = sns.heatmap(data, 
                     annot=True, 
                     cmap="Blues",  
                     fmt="g")
    ax.set_title(title)
    return ax


