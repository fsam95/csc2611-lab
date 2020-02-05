import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation(df, x_column, y_column, title):
    sns.set()
    ax = sns.regplot(x=x_column, y=y_column, data=df, ci=None)
    ax.figure.savefig('{}.eps'.format(title), format='eps', dpi=1000)
    plt.show()

def plot_time_course(time_points, change_values, change_x, change_y, word):
    sns.set()
    fig, axes = plt.subplots(figsize=(15, 5))
    axes.set_xticklabels(['1900-1910', '1910-1920', '1920-1930', '1930-1940', '1940-1950', '1950-1960', '1960-1970', '1970-1980', '1980-1990', '1990-2000'])
    axes.set_xlabel('Decade')
    axes.set_ylabel('Semantic change over decade (cosine similarity)')
    fig.suptitle(word)

    axes.scatter(time_points, change_values) 
    axes.scatter(change_x, change_y, c='r', label='Change point')
    axes.plot(time_points, change_values, c='b')
    axes.legend()
    fig.savefig('{}.eps'.format(word), format='eps', dpi=1000)
    plt.show()