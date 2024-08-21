import numpy as np
from tqdm import tqdm
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from data_retrieval import get_wnet


class ProgressRandomForestRegressor(RandomForestRegressor):
    def fit(self, X, y):
        self.n_estimators = getattr(self, 'n_estimators', 50)
        self.estimators_ = []
        self._validate_y_class_weight(y)
        
        # Initialize the base estimator
        self.base_estimator_ = DecisionTreeRegressor(random_state=self.random_state)
        
        # Progress bar
        with tqdm(total=self.n_estimators, desc="Fitting Random Forest") as pbar:
            for i in range(self.n_estimators):
                tree = clone(self.base_estimator_)
                if self.bootstrap:
                    n_samples = X.shape[0]
                    indices = np.random.choice(n_samples, n_samples, replace=True)
                    X_sample = X[indices]
                    y_sample = y[indices]
                    tree.fit(X_sample, y_sample, sample_weight=None, check_input=False)
                else:
                    tree.fit(X, y, sample_weight=None, check_input=False)
                self.estimators_.append(tree)
                pbar.update(1)  # Update progress bar for each fitted tree

        # Call the parent class's fit method to ensure all necessary attributes are set
        super().fit(X, y)

        return self

#Those are the sites for constructing prediction
#sites = ['asi', 'cor', 'ena', 'lei', 'manus', 'mao', 'lim', 'nsa', 'pgh', 'sgp_pbl']

def construct_prediction(sites: list, split_const = 0.8):
    """
    Constructs a set of predictions for the set of sites
    Input:
    sites: a list of sites for which we are constructing the prediction
    Output:
    y_W_all: get the predictions of W_net
    y_pred_all: predicted sigma_W values
    y_true_all: ground truth sigma_W
    """
    
    #Levels of the troposphere
    high_troposphere = [54, 55, 56, 57, 58, 59, 60]
    mid_troposphere = [61, 62, 63, 64, 65, 66]
    low_troposphere = [67, 68, 69, 70, 71]

    #Names of the models
    paths = ['high', 'mid', 'low']
    level_sets = [high_troposphere, mid_troposphere, low_troposphere]
    i = 0

    y_W_all = []
    y_pred_all = []
    y_true_all = []
    for levels in level_sets:
        X_T = []
        y_T = []
        y_W = []
        for site in sites:
            df_X = get_site_df(site)
            df_X = df_X[df_X['lev'].isin(levels)]
            df_X = df_X.drop(['lev', 'time'], axis = 1)

            #Splitting
            split_const = 0.8
            target_var = 'W_obs'
            X, y = df_X.drop([target_var], axis = 1), np.array(df_X[target_var])
            y_wnet = get_wnet(site, levels)

            # ==== Comment or uncomment (This biases the graph, as it shows better performance on training data)
            #HOWEVER, the data for training between Wnet and RF is the same, so the WE CAN compare
            X, y = X[:int(split_const*len(df_X))], y[:int(split_const*len(df_X))]
            y_wnet = y_wnet[:int(split_const*len(df_X))]
            #============
            X_T.append(X)
            y_T.append(y)
            y_W.append(y_wnet)

        X_T = pd.concat(X_T, axis=0, ignore_index=True)
        y_T = np.concatenate(y_T, axis=0)
        y_W = pd.concat(y_W, axis=0, ignore_index=True)
        y_W = np.array(y_W.wnet_pred)

        rf = joblib.load(f'./RF/RF_{paths[i]}_troposphere.joblib')
        y_pred = rf.predict(X_T)

        #Putting it all together
        y_W_all.append(y_W)
        y_pred_all.append(y_pred)
        y_true_all.append(y_T)
        i+=1

    y_W_all = np.concatenate(np.array(y_W_all), axis=0)
    y_pred_all = np.concatenate(np.array(y_pred_all), axis=0 )
    y_true_all = np.concatenate(np.array(y_true_all), axis=0 )
    
    return y_W_all, y_pred_all, y_true_all

def visualize_prediction(y_W_all, y_pred_all, y_true_all):
    """
    This function exists ot visualize how close are predicted distributions are to the ground truth
    
    Input:
    y_W_all: a list of predictions made by Wnet
    y_pred_all: a list of predictions made by a current model
    y_true_all: a list of ground truth predictions
    
    """
    bins = np.linspace(-3, 1.5, 300)  # pdf bounds
    dx = (bins[1:] - bins[:-1])
    bx = (bins[1:] + bins[:-1]) / 2

    plt.figure(figsize=(10, 5))

    pdf_obs = make_pdf(y_true_all, bins, dx)
    pdf_pred = make_pdf(y_pred_all, bins, dx)
    pdf_wnet = make_pdf(y_W_all, bins, dx)

    plt.plot(bx, pdf_obs, color = 'red', linestyle='-', linewidth=2, label=r'$\sigma_W$ Observations')
    plt.plot(bx, pdf_pred, color = 'darkgreen', linestyle='-', linewidth=2, label=r'$\sigma_W$ Random Forest')
    plt.plot(bx, pdf_wnet, color = 'black', linestyle='-', linewidth=2,  label=r'$\sigma_W$ Wnet')

    plt.title(f'COR (PBL) training', fontsize=22)
    plt.xlabel(r'$\log_{10}(\sigma_W \ m \ s^{-1})$', fontsize=18)
    plt.ylabel(r'$\frac{dP(\sigma_W)}{d \log(\sigma_W)}$', fontsize=24)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.show()