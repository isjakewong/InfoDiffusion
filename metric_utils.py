import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import logging
import os.path as osp
from sklearn.ensemble import RandomForestRegressor
import os
import sys
from sklearn.linear_model import Lasso, LassoCV
import seaborn as sns
import matplotlib.pyplot as plt


class DCIMetric():
    """ Impementation of the metric in:
        A FRAMEWORK FOR THE QUANTITATIVE EVALUATION OF DISENTANGLED
        REPRESENTATIONS
        The code is from:
        https://github.com/fjxmlzn/InfoGAN-CR
        which is adapted from:
        https://github.com/cianeastwood/qedr
    """
    def __init__(self, regressor="RandomForestIBGAN", *args, **kwargs):
        super(DCIMetric, self).__init__(*args, **kwargs)

        self._regressor = regressor
        if regressor == "Lasso":
            self.regressor_class = Lasso
            self.alpha = 0.02
            # constant alpha for all models and targets
            self.params = {"alpha": self.alpha}
            # weights
            self.importances_attr = "coef_"
        elif regressor == "LassoCV":
            self.regressor_class = LassoCV
            # constant alpha for all models and targets
            self.params = {}
            # weights
            self.importances_attr = "coef_"
        elif regressor == "RandomForest":
            self.regressor_class = RandomForestRegressor
            # Create the parameter grid based on the results of random search
            max_depths = [4, 5, 2, 5, 5]
            # Create the parameter grid based on the results of random search
            self.params = [{"max_depth": max_depth, "oob_score": True}
                           for max_depth in max_depths]
            self.importances_attr = "feature_importances_"
        elif regressor == "RandomForestIBGAN":
            # The parameters that IBGAN paper uses
            self.regressor_class = RandomForestRegressor
            # Create the parameter grid based on the results of random search
            max_depths = [4, 2, 4, 2, 2]
            # Create the parameter grid based on the results of random search
            self.params = [{"max_depth": max_depth, "oob_score": True}
                           for max_depth in max_depths]
            self.importances_attr = "feature_importances_"
        elif regressor == "RandomForestCV":
            self.regressor_class = GridSearchCV
            # Create the parameter grid based on the results of random search
            param_grid = {"max_depth": [i for i in range(2, 16)]}
            self.params = {
                "estimator": RandomForestRegressor(),
                "param_grid": param_grid,
                "cv": 3,
                "n_jobs": -1,
                "verbose": 0
            }
            self.importances_attr = "feature_importances_"
        elif "RandomForestEnum" in regressor:
            self.regressor_class = RandomForestRegressor
            # Create the parameter grid based on the results of random search
            self.params = {
                "max_depth": int(regressor[len("RandomForestEnum"):]),
                "oob_score": True
            }
            self.importances_attr = "feature_importances_"
        else:
            raise NotImplementedError()

        self.TINY = 1e-12

    def normalize(self, X):
        mean = np.mean(X, 0) # training set
        stddev = np.std(X, 0) # training set
        #print('mean', mean)
        #print('std', stddev)
        return (X - mean) / stddev

    def norm_entropy(self, p):
        '''p: probabilities '''
        n = p.shape[0]
        return - p.dot(np.log(p + self.TINY) / np.log(n + self.TINY))

    def entropic_scores(self, r):
        '''r: relative importances '''
        r = np.abs(r) + self.TINY
        ps = r / np.sum(r, axis=0) # 'probabilities'
        hs = [1 - self.norm_entropy(p) for p in ps.T]
        return hs

    def evaluate(self, codes, latents):
        codes = self.normalize(codes)
        latents = self.normalize(latents)
        R = []

        for j in range(latents.shape[-1]):
            if isinstance(self.params, dict):
              regressor = self.regressor_class(**self.params)
            elif isinstance(self.params, list):
              regressor = self.regressor_class(**self.params[j])
            regressor.fit(codes, latents[:, j])

            # extract relative importance of each code variable in
            # predicting the latent z_j
            if self._regressor == "RandomForestCV":
                best_rf = regressor.best_estimator_
                r = getattr(best_rf, self.importances_attr)[:, None]
            else:
                r = getattr(regressor, self.importances_attr)[:, None]

            R.append(np.abs(r))

        R = np.hstack(R) #columnwise, predictions of each z

        # disentanglement
        disent_scores = self.entropic_scores(R.T)
        # relative importance of each code variable
        c_rel_importance = np.sum(R, 1) / np.sum(R)
        disent_w_avg = np.sum(np.array(disent_scores) * c_rel_importance)

        # completeness
        complete_scores = self.entropic_scores(R)
        complete_avg = np.mean(complete_scores)

        return {
            "DCI_{}_disent_metric_detail".format(self._regressor): \
                disent_scores,
            "DCI_{}_disent_metric".format(self._regressor): disent_w_avg,
            "DCI_{}_complete_metric_detail".format(self._regressor): \
                complete_scores,
            "DCI_{}_complete_metric".format(self._regressor): complete_avg,
            "DCI_{}_metric_detail".format(self._regressor): R
            }

# function that takes a lists of latent indices, thresholds, and signs for classification
class LatentClass(object):

    def __init__(self, targ_ind, lat_ind, is_pos, thresh, __max, __min):
        super(LatentClass, self).__init__()
        self.targ_ind = targ_ind
        self.lat_ind = lat_ind
        self.is_pos = is_pos
        self.thresh = thresh
        self._max = __max
        self._min = __min
        self.it = list(zip(self.targ_ind, self.lat_ind, self.is_pos, self.thresh))

    def __call__(self, z, y_dim):
        # expect z to be [batch, z_dim]
        out = torch.ones((z.shape[0], y_dim))
        for t_i, l_i, is_pos, t in self.it:
            ma, mi = self._max[l_i], self._min[l_i]
            thr = t * (ma - mi) + mi
            res = (z[:, l_i] >= thr if is_pos else z[:, l_i] < thr).type(torch.int)
            out[:, t_i] = res
        return out

class TADMetric():
    """ Impementation of the metric in:
        NashAE: Disentangling Representations Through Adversarial Covariance Minimization
        The code is from:
        https://github.com/ericyeats/nashae-beamsynthesis
    """
    def __init__(self, y_dim, all_attrs):
        self.y_dim = y_dim
        self.all_attrs = all_attrs

    def calculate_auroc(self, targ, targ_ind, lat_ind, z, _ma, _mi, stepsize=0.01):
        thr = torch.arange(0.0, 1.0001, step=stepsize)
        total = targ.shape[0]
        pos_total = targ.sum(dim=0)[targ_ind].item()
        neg_total = total - pos_total
        p_fpr_tpr = torch.zeros((thr.shape[0], 2))
        n_fpr_tpr = torch.zeros((thr.shape[0], 2))
        for i, t in enumerate(thr):
            local_lc = LatentClass([targ_ind], [lat_ind], [True], [t], _ma, _mi)
            pred = local_lc(z.clone(), self.y_dim).to(targ.device)
            p_tp = torch.logical_and(pred == targ, pred).sum(dim=0)[targ_ind].item()
            p_fp = torch.logical_and(pred != targ, pred).sum(dim=0)[targ_ind].item()
            p_fpr_tpr[i][0] = p_fp / neg_total
            p_fpr_tpr[i][1] = p_tp / pos_total
            local_lc = LatentClass([targ_ind], [lat_ind], [False], [t], _ma, _mi)
            pred = local_lc(z.clone(), self.y_dim).to(targ.device)
            n_tp = torch.logical_and(pred == targ, pred).sum(dim=0)[targ_ind].item()
            n_fp = torch.logical_and(pred != targ, pred).sum(dim=0)[targ_ind].item()
            n_fpr_tpr[i][0] = n_fp / neg_total
            n_fpr_tpr[i][1] = n_tp / pos_total
        p_fpr_tpr = p_fpr_tpr.sort(dim=0)[0]
        n_fpr_tpr = n_fpr_tpr.sort(dim=0)[0]
        p_dists = p_fpr_tpr[1:, 0] - p_fpr_tpr[:-1, 0]
        p_area = (p_fpr_tpr[1:, 1] * p_dists).sum().item()
        n_dists = n_fpr_tpr[1:, 0] - n_fpr_tpr[:-1, 0]
        n_area = (n_fpr_tpr[1:, 1] * n_dists).sum().item()
        return p_area, n_area

    def aurocs(self, _z, targ, targ_ind, _ma, _mi):
        # perform a grid search of lat_ind to find the best classification metric
        aurocs = torch.ones(_z.shape[1]) * 0.5  # initialize as random guess
        for lat_ind in range(_z.shape[1]):
            if _ma[lat_ind] - _mi[lat_ind] > 0.2:
                p_auroc, n_auroc = self.calculate_auroc(targ, targ_ind, lat_ind, _z.clone(), _ma, _mi)
                m_auroc = max(p_auroc, n_auroc)
                aurocs[lat_ind] = m_auroc
                # print("{}\t{:1.3f}".format(lat_ind, m_auroc))
        return aurocs

    def aurocs_search(self, a, y):
        aurocs_all = torch.ones((y.shape[1], a.shape[1])) * 0.5
        base_rates_all = y.sum(dim=0)
        base_rates_all = base_rates_all / y.shape[0]
        _ma = a.max(dim=0)[0]
        _mi = a.min(dim=0)[0]
        print("Calculate for attribute:")
        for i in range(y.shape[1]):
            print(i)
            aurocs_all[i] = self.aurocs(a, y, i, _ma, _mi)
        return aurocs_all.cpu(), base_rates_all.cpu()

    def evaluate(self, a, y):
        auroc_result, base_rates_raw = self.aurocs_search(torch.FloatTensor(a), torch.IntTensor(y))
        base_rates = base_rates_raw.where(base_rates_raw <= 0.5, 1. - base_rates_raw)
        targ = torch.IntTensor(y)
        dim_y = y.shape[1]

        thresh = 0.75
        ent_red_thresh = 0.2
        max_aur, argmax_aur = torch.max(auroc_result.clone(), dim=1)
        norm_diffs = torch.zeros(dim_y)
        aurs_diffs = torch.zeros(dim_y)
        for ind, tag, max_a, argmax_a, aurs in zip(range(dim_y), self.all_attrs, max_aur.clone(), argmax_aur.clone(),
                                                   auroc_result.clone()):
            norm_aurs = (aurs.clone() - 0.5) / (aurs.clone()[argmax_a] - 0.5)
            aurs_next = aurs.clone()
            aurs_next[argmax_a] = 0.0
            aurs_diff = max_a - aurs_next.max()
            aurs_diffs[ind] = aurs_diff
            norm_aurs[argmax_a] = 0.0
            norm_diff = 1. - norm_aurs.max()
            norm_diffs[ind] = norm_diff

        # calculate mutual information shared between attributes
        # determine which share a lot of information with each other
        with torch.no_grad():
            not_targ = 1 - targ
            j_prob = lambda x, y: torch.logical_and(x, y).sum() / x.numel()
            mi = lambda jp, px, py: 0. if jp == 0. or px == 0. or py == 0. else jp * torch.log(jp / (px * py))

            # Compute the Mutual Information (MI) between the labels
            mi_mat = torch.zeros((dim_y, dim_y))
            for i in range(dim_y):
                # get the marginal of i
                i_mp = targ[:, i].sum() / targ.shape[0]
                for j in range(dim_y):
                    j_mp = targ[:, j].sum() / targ.shape[0]
                    # get the joint probabilities of FF, FT, TF, TT
                    # FF
                    jp = j_prob(not_targ[:, i], not_targ[:, j])
                    pi = 1. - i_mp
                    pj = 1. - j_mp
                    mi_mat[i][j] += mi(jp, pi, pj)
                    # FT
                    jp = j_prob(not_targ[:, i], targ[:, j])
                    pi = 1. - i_mp
                    pj = j_mp
                    mi_mat[i][j] += mi(jp, pi, pj)
                    # TF
                    jp = j_prob(targ[:, i], not_targ[:, j])
                    pi = i_mp
                    pj = 1. - j_mp
                    mi_mat[i][j] += mi(jp, pi, pj)
                    # TT
                    jp = j_prob(targ[:, i], targ[:, j])
                    pi = i_mp
                    pj = j_mp
                    mi_mat[i][j] += mi(jp, pi, pj)

            # fig, ax = plt.subplots(1, 2)
            # im = ax[0].imshow(mi_mat)
            # fig.colorbar(im, ax=ax[0], shrink=0.6)
            # mi_mat_ent_norm = mi_mat / mi_mat.diag().unsqueeze(1)
            # im = ax[1].imshow(mi_mat_ent_norm)
            # fig.colorbar(im, ax=ax[1], shrink=0.6)

            # plt.figure(figsize=(10, 7))
            # mi_comp = (mi_mat.sum(dim=1) - mi_mat.diag()) / mi_mat.diag()
            # plt.bar(range(len(all_attrs)), mi_comp, tick_label=all_attrs)
            # plt.xticks(rotation=90)
            # plt.title("Total Mutual Information")

            # plt.figure(figsize=(10, 7))
            mi_maxes, mi_inds = (mi_mat * (1 - torch.eye(dim_y))).max(dim=1)
            ent_red_prop = 1. - (mi_mat.diag() - mi_maxes) / mi_mat.diag()
            # plt.bar(range(len(all_attrs)), ent_red_prop, tick_label=all_attrs)
            # plt.xticks(rotation=90)
            # plt.title("Proportion of Entropy Reduced by Another Trait")
            # plt.grid(axis='y')
            # print(mi_mat.diag())

        # calculate Average Norm AUROC Diff when best detector score is at a certain threshold
        filt = (max_aur >= thresh).logical_and(ent_red_prop <= ent_red_thresh)
        aurs_diffs_filt = aurs_diffs[filt]
        return aurs_diffs_filt.sum().item(), auroc_result.cpu().numpy(), len(aurs_diffs_filt)



all_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
             'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
             'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
             'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
             'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
             'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
             'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
             'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
             ]

# data_dict = np.load("celeba_10d_0_1mmd_disentangle.npz", allow_pickle=True)
# a = data_dict["all_a"][:5000,:]
# y = data_dict["all_attr"][:5000,:]
# evaluater = TADMetric(y.shape[1], all_attrs)
# tad_score, auroc_result, num_attr = evaluater.evaluate(a, y)
#
# print("TAD SCORE: ", tad_score, "Attributes Captured: ", num_attr)
# sns.heatmap(auroc_result.transpose(), xticklabels = all_attrs)
# plt.show()


