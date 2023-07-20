import matplotlib.pyplot as plt
from sklearn import metrics

cmap = plt.get_cmap('tab10')


def plot_alg(ax, points, true_labels, pred_labels):
    c = [cmap(l) for l in pred_labels]
    s = [cmap(l) for l in true_labels]
    ax.scatter(*points, c=c, edgecolors=s)
    rand_score = metrics.adjusted_rand_score(true_labels, pred_labels)
    mi = metrics.normalized_mutual_info_score(true_labels, pred_labels)
    fw = metrics.fowlkes_mallows_score(true_labels, pred_labels)
    ax.set_xlabel(f'rand_score={rand_score:.2f}, MI={mi:.2f}, fw={fw:.2f}')
    return rand_score, mi, fw


def plot_experiment(algorithms, labels, fp, reverse=[]):
    n_alg = len(algorithms) + 1
    logs = {}
    fig, axes = plt.subplots(1, n_alg, figsize=(4 * n_alg, 3))
    true_labels = labels['true_labels'].tolist()
    for j, alg in enumerate(algorithms):
        alg_labels = labels[f'{alg}_labels']
        if alg in reverse:
            alg_labels = 1 - alg_labels
        rand_score, mi, fw = plot_alg(axes[j], labels['pca_reduced_states'].T.tolist(), true_labels, alg_labels)
        logs[alg] = [rand_score, mi, fw]

    for j, alg in enumerate(algorithms):
        axes[j].set_title(alg)
        axes[j].yaxis.set_ticks([])
        axes[j].xaxis.set_ticks([])

    plot_alg(axes[-1], labels['pca_reduced_states'].T.tolist(), true_labels, true_labels)
    axes[-1].set_title('True')
    axes[-1].yaxis.set_ticks([])
    axes[-1].xaxis.set_ticks([])
    plt.savefig(fp)
    plt.close()
    return logs
