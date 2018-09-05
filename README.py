import numpy as np
import pandas as pd
from scipy import sparse
from Predictome import Predictome
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import LogisticRegression
from matplotlib.ticker import NullFormatter
from Classifiers.src.classifiers import clf_log, clf_nn, clf_gb

"""
NOTES ON TEST MATRIX

WE CREATE A TEST MATRIX LIKE SO:

n_feats = 10000
n_rows = 1000
test_matrix = np.zeros((n_rows,n_feats))
for x in range(n_rows):
  n_u = np.random.uniform(0.05,0.5, n_feats)
  test_matrix[x] = np.random.binomial(2,n_u,n_feats)

test_matrix = np.hstack((test_matrix,np.random.binomial(1,0.5,(1000,1))))
sparse_test_matrix = sparse.csr_matrix(test_matrix)

feature_def = pd.DataFrame({'CHR':np.random.choice(np.arange(22), 10000), 'POS':np.random.choice(np.arange(1000000), 10000), 'REF':np.random.choice(np.asarray(['A','T']), 10000), 'ALT':np.random.choice(np.asarray(['C','G']), 10000)})
sample_def = pd.DataFrame({'samples' : np.random.choice(np.arange(10000,100000), 1000, replace=False)})3

"""


""" cd into the predictome directory """
pme = Predictome()  # intialize class
pme.build_from_VCF(matrix_type='variant', vcf_folders_by_labels=['test_input/test_case', 'test_input/test_control'])  # pass location of test_data
pme.variant.make_matrix()  # build variant matrix
pme.variant.reliability()  # compute relability

"""now that we have our matrix lets compute some odds ratios"""

"""dataframe to hold info"""
snp_stats_or = pd.DataFrame(index=np.arange(pme.variant.gt_matrix.shape[1] - 1))
snp_stats_pval = pd.DataFrame(index=np.arange(pme.variant.gt_matrix.shape[1] - 1))

matrix_copy = sparse.csr_matrix.copy(pme.variant.gt_matrix)  # copy the matrix since we bootstrap over it
pme.variant.fisher_coords = True  # set this property to True so we bypass a precomputation to speed up other fisher calculations

"""bootstrap phase takes ~ 5 minutes"""
for x in range(200):
    bsidx = np.random.choice(np.arange(pme.variant.gt_matrix.shape[0]), size=pme.variant.gt_matrix.shape[0])
    pme.variant.gt_matrix = pme.variant.gt_matrix[bsidx]  # we dont actually care about row info
    pme.variant.filter_matrix(by='fisherp')
    snp_stats_or[x] = pme.variant.odds_ratio
    snp_stats_pval[x] = pme.variant.pval_arry
    pme.variant.gt_matrix = matrix_copy
    print(x / 200)

"""Compute the bootstrap CI"""
snp_stats_or['CL_min'] = snp_stats_or.quantile(0.025, axis=1)
snp_stats_or['CL_max'] = snp_stats_or.iloc[:, :-1].quantile(0.975, axis=1)
snp_stats_or['means'] = snp_stats_or.iloc[:, :-2].mean(axis=1)
snp_stats_pval['CL_min'] = snp_stats_pval.quantile(0.025, axis=1)
snp_stats_pval['CL_max'] = snp_stats_pval.iloc[:, :-1].quantile(0.975, axis=1)
snp_stats_pval['means'] = snp_stats_pval.iloc[:, :-2].mean(axis=1)

odds_mean = snp_stats_or['means']
pval_mean = snp_stats_pval['means']
pval_mean_vals = pval_mean.values

cleaned_means = odds_mean[~odds_mean.isin([np.nan, np.inf, -np.inf])].dropna()
real_index = cleaned_means.index

neg_idx = cleaned_means[cleaned_means < 0].sort_values().index
pos_idx = cleaned_means[cleaned_means > 0].sort_values(ascending=False).index

neg_inf = odds_mean[odds_mean.isin([-np.inf]).dropna()]
neg_inf_index = neg_inf.index

pos_inf = odds_mean[odds_mean.isin([np.inf]).dropna()]
pos_inf_index = pos_inf.index


"""
****************************************
 INDEPENDENT PLOTTING OF CONFIDENCE INTERVALS
****************************************
"""

fig, ax = plt.subplots(figsize=(20, 11))
scatter = ax.scatter(snp_stats_or['CL_min'], snp_stats_or['CL_max'], c=snp_stats_or['means'], cmap='RdBu_r', alpha=0.75)
ax.axhline(y=0.0, color='k', linestyle='--')
ax.axvline(x=0.0, color='k', linestyle='--')
plt.title("95 percent confidence intervals on the odds ratios after 200 bootstraps")
plt.xlabel("min confidence interval")
plt.ylabel("max confidence interval")
cbar = plt.colorbar(scatter)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Odds ratio mean', rotation=270)
fig.tight_layout()
plt.show()

"""
****************************************
 SELECT FEATURES
****************************************
"""

#  NUMERIC TIGHT CONFINDENCE
CL_min_max_real = snp_stats_or.loc[real_index, ['CL_min', 'CL_max']]
signed = CL_min_max_real.apply(np.sign)
signed_idx = signed[signed['CL_min'] == signed['CL_max']].index

#  INFINITY SIGNIFICANT  < 0.05
neg_sig_vals_idx = pval_mean.loc[neg_inf_index][pval_mean.loc[neg_inf_index] <= 0.05].index
pos_sig_vals_idx = pval_mean.loc[pos_inf_index][pval_mean.loc[pos_inf_index] <= 0.05].index

# INDICES
col_indx = np.hstack((neg_sig_vals_idx, pos_sig_vals_idx, signed_idx, -1))


"""
****************************************
ANALYZING THE DATA WITH LOGISTIC REGRESSION
****************************************
"""

pme.variant.gt_matrix = matrix_copy
pme.variant.gt_matrix = pme.variant.gt_matrix[:, col_indx]

log_test = clf_log(matrix_class=pme.variant, feature_df=pme.variant.columns_df, grid_search_random=1, original_cols=pme.variant.original_cols[col_indx[:-1]], bst=True, num_folds=500)
model = [LogisticRegression(penalty='l2', C=5, class_weight='balanced')]  # manual input the model alternative to this is a log_search function log_model_search()
log_test.models = model
log_test.train()

"""POST PROCESSING ANALYSIS"""
log_test.prediction_tracker['CL_min'] = log_test.prediction_tracker.astype(float).quantile(0.025, axis=1)
log_test.prediction_tracker['CL_max'] = log_test.prediction_tracker.iloc[:, :-1].astype(float).quantile(0.975, axis=1)
log_test.prediction_tracker['means'] = log_test.prediction_tracker.iloc[:, :-2].astype(float).mean(axis=1)
log_test.prediction_tracker['label'] = pme.variant.gt_matrix[:, -1].A.T[0]

new_mean_grps = np.linspace(log_test.prediction_tracker['means'].min(), log_test.prediction_tracker['means'].max(), 11)
for i, x in enumerate(new_mean_grps):
    if i + 1 >= len(new_mean_grps):
        break
    mean_grp_idx = log_test.prediction_tracker[(log_test.prediction_tracker['means'] >= x) & (log_test.prediction_tracker['means'] <= new_mean_grps[i + 1])].index
    log_test.prediction_tracker.loc[mean_grp_idx, 'MEAN_GRP'] = x

"""
****************************************
            PLOT RESULTS
****************************************
"""

def plot_dual_histogram(df, condition, x, c, switch=False):
    """
    A function for plotting the data such that we can visualize the density of the scatter plot using two histograms

    Parameters:
    ======================================

    df -> This is a panda DataFrame object that should come from the classifier class i.e. prediction_tracker variable

    condition -> We need to have a condition for the fucntion. This must be manually added to the prediction_tracker df

    x -> If there are multiple conditions, x is a single condition emitted from a loop

    c -> pass a color from matplotlib e.g. use colormap function

    swtich -> If you want to plot the values that are not part of the condition set to True


    Notes:
    ======================================


    Examples:
    ======================================

    >>>>  plt.figure(1, figsize=(20, 11))
    >>>>  plt.clf()
    >>>>  r_cohort = log_test.prediction_tracker['label'].dropna().unique()
    >>>>  colors = cm.RdBu(np.linspace(0, 1, len(r_cohort)))
    >>>>  c = 0
    >>>>  r_cohort.sort()
    >>>>  for x in r_cohort:
    >>>>    plot_dual_histogram(log_test.prediction_tracker, 'label', x, 'grey', switch=True)
    >>>>    plot_dual_histogram(log_test.prediction_tracker, 'label', x, colors[c], switch=False)
    >>>>    c += 1
    >>>>    plt.show()
    >>>>    plt.clf()
    """
    nullfmt = NullFormatter()         # no labels
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    if switch is False:
        X = df[df[condition] == x]['CL_min']
        Y = df[df[condition] == x]['CL_max']
    else:
        X = df[df[condition] != x]['CL_min']
        Y = df[df[condition] != x]['CL_max']
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    # the scatter plot:
    axScatter.scatter(X, Y, c=c, alpha=0.5)
    axScatter.axhline(y=0.0, color='k', linestyle='--')
    axScatter.axvline(x=0.0, color='k', linestyle='--')
    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = np.max([np.max(np.fabs(X)), np.max(np.fabs(Y))])
    lim = (int(xymax / binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(X, bins=bins, color=c, alpha=0.5, edgecolor='k')
    axHisty.hist(Y, bins=bins, orientation='horizontal', color=c, alpha=0.5, edgecolor='k')
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())


plt.clf()
r_cohort = log_test.prediction_tracker['label'].dropna().unique()
colors = cm.RdBu(np.linspace(0, 1, len(r_cohort)))
c = 0
r_cohort.sort()
for x in r_cohort:
    plot_dual_histogram(log_test.prediction_tracker, 'label', x, 'grey', switch=True)
    plot_dual_histogram(log_test.prediction_tracker, 'label', x, colors[c], switch=False)
    c += 1
    plt.show()
    plt.clf()


"""
********************************************************************************
      ANALYZING THE DATA WITH NEURAL NET YOU NEED TENSORFLOW INSTALLED
********************************************************************************
"""

nn_clf = clf_nn(matrix_class=pme.variant, feature_df=pme.variant.columns_df, grid_search_random=1, original_cols=pme.variant.original_cols[col_indx[:-1]], bst=True, num_folds=500)
parameters = {'layers': 3, 'neurons': [5, 10, 15], 'penalty': 'dropout', 'keep': 0.9, 'lin:reg': False}
model = [parameters]
nn_clf.models = model
nn_clf.train()

"""POST PROCESSING ANALYSIS"""
nn_clf.prediction_tracker['CL_min'] = nn_clf.prediction_tracker.astype(float).quantile(0.025, axis=1)
nn_clf.prediction_tracker['CL_max'] = nn_clf.prediction_tracker.iloc[:, :-1].astype(float).quantile(0.975, axis=1)
nn_clf.prediction_tracker['means'] = nn_clf.prediction_tracker.iloc[:, :-2].astype(float).mean(axis=1)
nn_clf.prediction_tracker['label'] = pme.variant.gt_matrix[:, -1].A.T[0]

new_mean_grps = np.linspace(nn_clf.prediction_tracker['means'].min(), nn_clf.prediction_tracker['means'].max(), 11)
for i, x in enumerate(new_mean_grps):
    if i + 1 >= len(new_mean_grps):
        break
    mean_grp_idx = nn_clf.prediction_tracker[(nn_clf.prediction_tracker['means'] >= x) & (nn_clf.prediction_tracker['means'] <= new_mean_grps[i + 1])].index
    nn_clf.prediction_tracker.loc[mean_grp_idx, 'MEAN_GRP'] = x

plt.clf()
r_cohort = nn_clf.prediction_tracker['label'].dropna().unique()
colors = cm.RdBu(np.linspace(0, 1, len(r_cohort)))
c = 0
r_cohort.sort()
for x in r_cohort:
    plot_dual_histogram(nn_clf.prediction_tracker, 'label', x, 'grey', switch=True)
    plot_dual_histogram(nn_clf.prediction_tracker, 'label', x, colors[c], switch=False)
    c += 1
    plt.show()
    plt.clf()


"""
********************************************************************************
 ANALYZING THE DATA WITH GRADIENT BOOSTED DECISION TREES YOU NEED XGB INSTALLED
********************************************************************************
"""

gb_clf = clf_gb(matrix_class=pme.variant, feature_df=None, grid_search_random=1, original_cols=pme.variant.original_cols[col_indx[:-1]], bst=True, num_folds=500)
gb_clf.num_boost_round = 100
param = {'max_depth': 2, 'eta': 1.3, 'silent': 1, 'objective': 'binary:logitraw'}
model = [param]
gb_clf.models = model
gb_clf.train()

"""POST PROCESSING ANALYSIS"""
gb_clf.prediction_tracker['CL_min'] = gb_clf.prediction_tracker.astype(float).quantile(0.025, axis=1)
gb_clf.prediction_tracker['CL_max'] = gb_clf.prediction_tracker.iloc[:, :-1].astype(float).quantile(0.975, axis=1)
gb_clf.prediction_tracker['means'] = gb_clf.prediction_tracker.iloc[:, :-2].astype(float).mean(axis=1)
gb_clf.prediction_tracker['label'] = pme.variant.gt_matrix[:, -1].A.T[0]

new_mean_grps = np.linspace(gb_clf.prediction_tracker['means'].min(), gb_clf.prediction_tracker['means'].max(), 11)
for i, x in enumerate(new_mean_grps):
    if i + 1 >= len(new_mean_grps):
        break
    mean_grp_idx = gb_clf.prediction_tracker[(gb_clf.prediction_tracker['means'] >= x) & (gb_clf.prediction_tracker['means'] <= new_mean_grps[i + 1])].index
    gb_clf.prediction_tracker.loc[mean_grp_idx, 'MEAN_GRP'] = x


plt.clf()
r_cohort = gb_clf.prediction_tracker['label'].dropna().unique()
colors = cm.RdBu(np.linspace(0, 1, len(r_cohort)))
c = 0
r_cohort.sort()
for x in r_cohort:
    plot_dual_histogram(gb_clf.prediction_tracker, 'label', x, 'grey', switch=True)
    plot_dual_histogram(gb_clf.prediction_tracker, 'label', x, colors[c], switch=False)
    c += 1
    plt.show()
    plt.clf()


"""
**************************************
    COMPARE ALL THREE CLASSIFIERS
**************************************
"""


def sigmoid_transform(item):
    return (np.exp((-1 * item)) + 1)**-1


plt.clf()
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True)


min_ci_x_0 = []
min_ci_y_0 = []
min_ci_x_1 = []
min_ci_y_1 = []
for group, rows in nn_clf.prediction_tracker.groupby(['label', 'MEAN_GRP']):  # group by braak
    if group[0] == 0.0:
        min_ci_y_0.append(sigmoid_transform(rows.iloc[:, :500].astype(float)).unstack().quantile(0.025))
        min_ci_x_0.append(group[1])
    else:
        min_ci_y_1.append(sigmoid_transform(rows.iloc[:, :500].astype(float)).unstack().quantile(0.025))
        min_ci_x_1.append(group[1])

max_ci_x_0 = []
max_ci_y_0 = []
max_ci_x_1 = []
max_ci_y_1 = []
for group, rows in nn_clf.prediction_tracker.groupby(['label', 'MEAN_GRP']):  # group by braak
    if group[0] == 0.0:
        max_ci_y_0.append(sigmoid_transform(rows.iloc[:, :500].astype(float)).unstack().quantile(0.975))
        max_ci_x_0.append(group[1])
    else:
        max_ci_y_1.append(sigmoid_transform(rows.iloc[:, :500].astype(float)).unstack().quantile(0.975))
        max_ci_x_1.append(group[1])

x_0 = []
y_0 = []
x_1 = []
y_1 = []
sizes_0 = []
sizes_1 = []
for group, rows in nn_clf.prediction_tracker.groupby(['label', 'MEAN_GRP']):  # group by braak
    if group[0] == 0.0:
        y_0.append(sigmoid_transform(rows.iloc[:, :500].astype(float)).unstack().mean())
        x_0.append(group[1])
        sizes_0.append(len(rows))
    else:
        y_1.append(sigmoid_transform(rows.iloc[:, :500].astype(float)).unstack().mean())
        x_1.append(group[1])
        sizes_1.append(len(rows))

ax[0].scatter(x_0,y_0, color='darkred', edgecolor='k', s=sizes_0, alpha=0.5)
ax[0].plot(min_ci_x_0, min_ci_y_0, color='r', linestyle='--')
ax[0].plot(max_ci_x_0, max_ci_y_0, color='r', linestyle='--')
ax[0].fill_between(x_0, min_ci_y_0 ,max_ci_y_0, color='darkred', alpha=0.25)
ax[0].set_ylabel("Pr(1|x) Neural Nets")
ax[0].set_xlabel("mean score after 500 bootstraps")
ax[0].set_title("Confidence intervals of the probability correlated to mean groups")
ax[0].scatter(x_1,y_1, color='steelblue', edgecolor='k', s=sizes_1, alpha=0.5)
ax[0].plot(min_ci_x_1, min_ci_y_1, color='b', linestyle='--')
ax[0].plot(max_ci_x_1, max_ci_y_1, color='b', linestyle='--')
ax[0].fill_between(x_1, min_ci_y_1 ,max_ci_y_1, color='steelblue', alpha=0.25)

legend_sizes_0 = np.sort(sizes_0)[::len(sizes_0) // 4][-3:]
legend_sizes_1 = np.sort(sizes_1)[::len(sizes_1) // 4][-3:]
indices_0 = [np.where(sizes_0 == v)[0][0] for v in legend_sizes_0]
indices_1 = [np.where(sizes_1 == v)[0][0] for v in legend_sizes_1]
for i in indices_1:
    ax[0].scatter(x_1[i], y_1[i], color='steelblue', label='{:.2f}'.format(sizes_1[i]), edgecolor='k', s=sizes_1[i], alpha=0.25)

for i in indices_0:
    ax[0].scatter(x_0[i], y_0[i], color='darkred', label='{:.2f}'.format(sizes_0[i]), edgecolor='k', s=sizes_0[i], alpha=0.25)

ax[0].legend(loc=0)
ax[0].legend(loc=0)


min_ci_x_0 = []
min_ci_y_0 = []
min_ci_x_1 = []
min_ci_y_1 = []
for group, rows in log_test.prediction_tracker.groupby(['label', 'MEAN_GRP']):  # group by braak
    if group[0] == 0.0:
        min_ci_y_0.append(sigmoid_transform(rows.iloc[:, :500].astype(float)).unstack().quantile(0.025))
        min_ci_x_0.append(group[1])
    else:
        min_ci_y_1.append(sigmoid_transform(rows.iloc[:, :500].astype(float)).unstack().quantile(0.025))
        min_ci_x_1.append(group[1])

max_ci_x_0 = []
max_ci_y_0 = []
max_ci_x_1 = []
max_ci_y_1 = []
for group, rows in log_test.prediction_tracker.groupby(['label', 'MEAN_GRP']):  # group by braak
    if group[0] == 0.0:
        max_ci_y_0.append(sigmoid_transform(rows.iloc[:, :500].astype(float)).unstack().quantile(0.975))
        max_ci_x_0.append(group[1])
    else:
        max_ci_y_1.append(sigmoid_transform(rows.iloc[:, :500].astype(float)).unstack().quantile(0.975))
        max_ci_x_1.append(group[1])

x_0 = []
y_0 = []
x_1 = []
y_1 = []
sizes_0 = []
sizes_1 = []
for group, rows in log_test.prediction_tracker.groupby(['label', 'MEAN_GRP']):  # group by braak
    if group[0] == 0.0:
        y_0.append(sigmoid_transform(rows.iloc[:, :500].astype(float)).unstack().mean())
        x_0.append(group[1])
        sizes_0.append(len(rows))
    else:
        y_1.append(sigmoid_transform(rows.iloc[:, :500].astype(float)).unstack().mean())
        x_1.append(group[1])
        sizes_1.append(len(rows))

ax[1].scatter(x_0,y_0, color='darkred', edgecolor='k', s=sizes_0, alpha=0.5)
ax[1].plot(min_ci_x_0, min_ci_y_0, color='r', linestyle='--')
ax[1].plot(max_ci_x_0, max_ci_y_0, color='r', linestyle='--')
ax[1].fill_between(x_0, min_ci_y_0 ,max_ci_y_0, color='darkred', alpha=0.25)
ax[1].set_ylabel("Pr(1|x) logistic regression")
ax[1].set_xlabel("mean score after 500 bootstraps")
ax[1].set_title("Confidence intervals of the probability correlated to mean groups")
ax[1].scatter(x_1,y_1, color='steelblue', edgecolor='k', s=sizes_1, alpha=0.5)
ax[1].plot(min_ci_x_1, min_ci_y_1, color='b', linestyle='--')
ax[1].plot(max_ci_x_1, max_ci_y_1, color='b', linestyle='--')
ax[1].fill_between(x_1, min_ci_y_1 ,max_ci_y_1, color='steelblue', alpha=0.25)

legend_sizes_0 = np.sort(sizes_0)[::len(sizes_0) // 4][-3:]
legend_sizes_1 = np.sort(sizes_1)[::len(sizes_1) // 4][-3:]
indices_0 = [np.where(sizes_0 == v)[0][0] for v in legend_sizes_0]
indices_1 = [np.where(sizes_1 == v)[0][0] for v in legend_sizes_1]
for i in indices_1:
    ax[1].scatter(x_1[i], y_1[i], color='steelblue', label='{:.2f}'.format(sizes_1[i]), edgecolor='k', s=sizes_1[i], alpha=0.25)

for i in indices_0:
    ax[1].scatter(x_0[i], y_0[i], color='darkred', label='{:.2f}'.format(sizes_0[i]), edgecolor='k', s=sizes_0[i], alpha=0.25)

ax[1].legend(loc=0)
ax[1].legend(loc=0)
min_ci_x_0 = []
min_ci_y_0 = []
min_ci_x_1 = []
min_ci_y_1 = []

for group, rows in gb_clf.prediction_tracker.groupby(['label', 'MEAN_GRP']):  # group by braak
    if group[0] == 0.0:
        min_ci_y_0.append(sigmoid_transform(rows.iloc[:, :500].astype(float)).unstack().quantile(0.025))
        min_ci_x_0.append(group[1])
    else:
        min_ci_y_1.append(sigmoid_transform(rows.iloc[:, :500].astype(float)).unstack().quantile(0.025))
        min_ci_x_1.append(group[1])

max_ci_x_0 = []
max_ci_y_0 = []
max_ci_x_1 = []
max_ci_y_1 = []
for group, rows in gb_clf.prediction_tracker.groupby(['label', 'MEAN_GRP']):  # group by braak
    if group[0] == 0.0:
        max_ci_y_0.append(sigmoid_transform(rows.iloc[:, :500].astype(float)).unstack().quantile(0.975))
        max_ci_x_0.append(group[1])
    else:
        max_ci_y_1.append(sigmoid_transform(rows.iloc[:, :500].astype(float)).unstack().quantile(0.975))
        max_ci_x_1.append(group[1])

x_0 = []
y_0 = []
x_1 = []
y_1 = []
sizes_0 = []
sizes_1 = []
for group, rows in gb_clf.prediction_tracker.groupby(['label', 'MEAN_GRP']):  # group by braak
    if group[0] == 0.0:
        y_0.append(sigmoid_transform(rows.iloc[:, :500].astype(float)).unstack().mean())
        x_0.append(group[1])
        sizes_0.append(len(rows))
    else:
        y_1.append(sigmoid_transform(rows.iloc[:, :500].astype(float)).unstack().mean())
        x_1.append(group[1])
        sizes_1.append(len(rows))

ax[2].scatter(x_0,y_0, color='darkred', edgecolor='k', s=sizes_0, alpha=0.5)
ax[2].plot(min_ci_x_0, min_ci_y_0, color='r', linestyle='--')
ax[2].plot(max_ci_x_0, max_ci_y_0, color='r', linestyle='--')
ax[2].fill_between(x_0, min_ci_y_0 ,max_ci_y_0, color='darkred', alpha=0.25)
ax[2].set_ylabel("Pr(1|x) GBD")
ax[2].set_xlabel("mean score after 500 bootstraps")
ax[2].set_title("Confidence intervals of the probability correlated to mean groups")
ax[2].scatter(x_1,y_1, color='steelblue', edgecolor='k', s=sizes_1, alpha=0.5)
ax[2].plot(min_ci_x_1, min_ci_y_1, color='b', linestyle='--')
ax[2].plot(max_ci_x_1, max_ci_y_1, color='b', linestyle='--')
ax[2].fill_between(x_1, min_ci_y_1 ,max_ci_y_1, color='steelblue', alpha=0.25)

legend_sizes_0 = np.sort(sizes_0)[::len(sizes_0) // 4][-3:]
legend_sizes_1 = np.sort(sizes_1)[::len(sizes_1) // 4][-3:]
indices_0 = [np.where(sizes_0 == v)[0][0] for v in legend_sizes_0]
indices_1 = [np.where(sizes_1 == v)[0][0] for v in legend_sizes_1]
for i in indices_1:
    ax[2].scatter(x_1[i], y_1[i], color='steelblue', label='{:.2f}'.format(sizes_1[i]), edgecolor='k', s=sizes_1[i], alpha=0.25)

for i in indices_0:
    ax[2].scatter(x_0[i], y_0[i], color='darkred', label='{:.2f}'.format(sizes_0[i]), edgecolor='k', s=sizes_0[i], alpha=0.25)

ax[2].legend(loc=0)
ax[2].legend(loc=0)
plt.show()