import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from copy import deepcopy
import os, warnings


# check if the type of (all elements of) inputs are inputs_types.
# if not, raise TypeError.
def chk_types(inputs, inputs_types):
    """
    inputs       : data of which type is to be verified.
    inputs_types : (type) 
    """
    if not (isinstance(inputs_types, list) 
            or isinstance(inputs_types, np.ndarray)):
        inputs_types = [inputs_types]
        
    chk_result = [isinstance(inputs, inputs_type) 
                  for inputs_type in inputs_types]
    if not any(chk_result):
        raise TypeError(f"{type(inputs)} is not supported.")


# a class for data storage, 
# including N-dimensional histogram bins and conditional probability
class Feature:
    def __init__(self, name, data):
        """
        name : (str) name of the feature
        data : feature data to be stored.
        """
        chk_types(name, str)
        chk_types(data, np.ndarray)
        self.data = data
        self.name = name
        self.nbins = None
        self.condprob = None
        self.bindedges = None
        self.bincenters = None

    def set_nbin(self, nbin):
        """
        nbin : (int) number of bins
        """
        chk_types(nbin, int)
        self.nbins = nbin

    def set_binedges(self, binedges):
        """
        binedges : (list or numpy.ndarray) histogram bin edges
        """
        chk_types(binedges, [list, np.ndarray])
        self.binedges = binedges
        self.bincenters = (binedges[:-1] + binedges[1:])/2
        
    def get_bincenter(self, idx):
        return self.bincenters[idx]


# perform Gibbs Sampling based on given data
class GibbsSampling:
    # n_data : number of data to generate
    # input_data: input dataframe either numpy or dataframe
    # list of bins numbers to be divided
    # filename to save

    def __init__(self, 
                 n_data, 
                 input_data, 
                 bins=None):
        """
        n_data      : (int) number of data to be generated
        input_data  : (pandas.DataFrame, numpy.ndarray or filename)
                      If input_data is a file, it should be .pkl format pandas.DataFrame
        bins    : (list) default=[10] * input_data features
                      Number of bins for each features.
        """
        # input_data is supposed to be stored as pandas.DataFrame
        # For the case of input_data is a string, trying to read the file.
        def set_input_data(input_data):
            if isinstance(input_data, str):            
                try:
                    if input_data.endswith(".csv"):
                        _input_data = pd.read_csv(input_data)
                    elif input_data.endswith(".pkl"):
                        _input_data = pd.read_pickle(input_data)
                    elif input_data.endswith(".xlsx"):
                        _input_data = pd.read_excel(input_data)
                    else: # default: pickle
                        _input_data = pd.read_pickle(input_data)
                except FileNotFoundError:
                    raise FileNotFoundError(f"{filename} is not found")
            elif isinstance(input_data, pd.DataFrame):
                _input_data = input_data
            else: # numpy.ndarray
                try:
                    columns = [f"X{i}" for i in range(input_data.shape[1])]
                except IndexError: # 1D array
                    columns = ["X0"]
                _input_data = pd.DataFrame(data=input_data, columns=columns)

            self.ndim = _input_data.shape[1]
            self.input_data = _input_data
            
        # create bins for data discretization
        def set_bins(bins):
            if not bins:
                _bins = [10] * self.ndim
            elif isinstance(bins, int):
                _bins = [bins] * self.ndim
            elif isinstance(bins, list) and len(bins) == self.ndim:
                _bins = bins
            elif isinstance(bins, list) and len(bins) != self.ndim:
                raise ValueError(f"dimension of bins={len(bins)} \
                does not match with the data dimension {self.ndim}.")
            else:
                raise TypeError()

            self.bins = _bins
        
        # input parameters
        self.n_data = n_data        
        set_input_data(input_data)       
        set_bins(bins)
        
        # features as dictionary type
        self.Xs = {}  
        
        # probability
        self.binedges_nd = None
        self.p = None
        self.p_gen = None
        
        # generated data
        self.X_gen = None  # for each bins
        self.X_gen_center = None  # bin center for each bins
        self.X_gen_real = None  # real coordinate
        
        # run sampling
        self.sampling()

        
    # generate features with given names
    def __get_features(self):
        for col, nbin in zip(self.input_data.columns, self.bins):
            self.Xs[col] = Feature(col , self.input_data[col].values)
            self.Xs[col].set_nbin(nbin)

    # calculate probability of each bins
    def __get_prob(self, data, isgibbs=True):
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        if not isgibbs: # input data
            hist_nd, binedges_nd = np.histogramdd(data, bins=self.bins)
            self.binedges_nd = binedges_nd
            [X.set_binedges(binedges) for X, binedges in zip(self.Xs.values(), binedges_nd)]
            p = hist_nd / self.input_data.shape[0]  # probability of each bin
            self.p = p
        else: # Gibbs sampled data, containing index.
            hist_nd, binedges_nd = np.histogramdd(data, bins=self.binedges_nd)
            p = hist_nd / self.input_data.shape[0]  # probability of each bin
            return p
    
    # calculate conditional probability for each bins
    def __get_condprob(self, p):
        def condprob(p, axis):
            p_sa = p.swapaxes(0, axis)
            p_sum = p_sa.sum(axis=0)  # integration @axis

            warnings.filterwarnings("ignore")  # hide zero divide warnings
            p_conds = [p_sa[i] / p_sum for i in range(p.shape[axis])]  # conditional probability
            warnings.filterwarnings("default") # restore zero divide warnings

            p_conds = np.nan_to_num(p_conds, 0).swapaxes(0, axis)  # nan removal and restore axes

            return p_conds

        for i, X in enumerate(self.Xs.values()):
            X.condprob = condprob(p, i)


    
    # Gibbs Sampling
    def __get_sampling(self, n_data=None, kfold=5):
        for k in range(kfold):
            # initialize memory storage for sampling
            # create zeros array for data storage
            if not n_data:
                n_data = self.n_data
                
            X_gen_ = np.zeros((int(n_data/kfold), len(self.Xs)), dtype=int)
            X_gen_center_ = np.zeros((int(n_data/kfold), len(self.Xs)))
        
            # set initilize starting point, where the probability is max
            if k == 0:
                p_gibbs = self.p
            else:
                p_gen = self.__get_prob(X_gen_center, isgibbs=True)
                self.p_gen = p_gen
                p_gibbs = self.p - p_gen
            
            X_gen_[0] = np.unravel_index(np.argmax(p_gibbs, axis=None), p_gibbs.shape)
            
            for i in range(1, int(n_data/kfold)):
                for j, X in enumerate(self.Xs.values()):
                    # 1. conditional probability
                    block_0 = f"X.condprob["
                    block_2 = [f"X_gen_[i-1, {f}]" for f in range(len(self.Xs))]
                    block_2[j] = ":"
                    block_1 = [b.replace("i-1", "i") for b in block_2[:j]]
                    block_1.extend(block_2[j:])
                    block_m = ", ".join(block_1)
                    condprob = eval(block_0 + block_m + "]")

                    # 2. generate data
                    try:
                      condprob_ = condprob
                    except:
                      print(f"condprob.sum()={condprob.sum()}")
                      condprob = np.nan_to_num(condprob, 0) # again, nan removal
                      condprob_ = condprob/condprob.sum()   # again
                    X_gen_[i, j] = np.random.choice(np.arange(X.nbins), p=condprob_) ##
                    
            # convert to real values (bincenter)
            for i in range(self.ndim):
                X = self.Xs[list(self.Xs.keys())[i]]
                X_gen_center_.T[i] = X.get_bincenter(X_gen_.T[i])                        
            
            if k == 0:
                X_gen = deepcopy(X_gen_)
                X_gen_center = deepcopy(X_gen_center_)
            else:
                X_gen = np.concatenate([X_gen, X_gen_], axis=0)
                X_gen_center = np.concatenate([X_gen_center, X_gen_center_], axis=0)

        self.X_gen = X_gen
        self.X_gen_center = X_gen_center
    
    # convert bin indices to coordinates by uniform random
    def __get_coord(self):
        self.X_gen_real = np.zeros(self.X_gen.shape)
        
        def coord(X_id, idx):
            X = list(self.Xs.values())[X_id]
            idx_coord = self.X_gen[:, X_id][idx]
            low, high = X.binedges[idx_coord], X.binedges[idx_coord + 1]

            return np.random.uniform(low, high)
    
        for i in range(self.ndim):
            for j in range(len(self.X_gen[:, i])):
                self.X_gen_real.T[i, j] = coord(i, j)


    # Save sampling data as dataframe
    # Raise error if run prior to sampling()
    # .pkl or .csv according to the filename
    def to_df(self, filename=None):
        """
        Parameters
        ----------
        filename    : (str) default=None. 
                        If any, generated data is saved as .pkl format with the filename
        """
        if not (self.X_gen_real).all:
            raise RuntimeError("No sampling data yet. `sampling()` is the prerequisite.")

        else:
            columns = list(self.input_data.columns)
            X_gen_out = pd.DataFrame(data=self.X_gen_real, columns=columns)
            
            if filename.endswith(".pkl"): # save as pickle file
                X_gen_out.to_pickle(filename)
            elif filename.endswith(".csv"): # save as csv file
                X_gen_out.to_csv(filename, index=False)
            else:
                filename = filename + ".pkl"
                X_gen_out.to_pickle(filename)                
                print(f"# (default) exporting as pickle")
            
            print(f"# Sampling data export complete: {filename}")
            
    # Gibbs sampling based on input data
    def sampling(self, n_data=None, kfold=5, filename=None):
        """
        Parameters
        ----------
        n_data      : (int) number of data to be generated
        kfold       : (int) number of fold in Gibbs Sampling.
                            conditional probability is recalculated at every fold data generation
        filename    : (str) default=None. 
                        If any, generated data is saved as .pkl format with the filename
        
        Returns
        ----------
        (numpy.array) Gibbs sampled data
        """
        self.__get_features()
        self.__get_prob(self.input_data, isgibbs=False)
        self.__get_condprob(self.p)
        self.__get_sampling(n_data=n_data, kfold=kfold)
        self.__get_coord()
        
        if not n_data:
          n_data = self.n_data
          
        if filename:
            self.to_df(filename)

        return self.X_gen_real.T

    # visulaization function
    # user could choose the feature they want to plot
    # default features for 3-d scatter are 0,1,2
    def plot(self, features=None, kind="hist2d", org=False, 
             cmap="Blues", alpha=0.1, grid=True, figsize=None, 
             filename=None, dpi=72):
        """
        Parameters
        ----------
        features : (int, str, list or np.ndarray) 
                    if int, indices of features to be visualized
                    if str, names of features to be visualized
        figsize  : (tuple) default = 3*(number of features -1 )
                    matplotlib figure size.
        kind     : ("scatter" or "hist2d") default="hist2d"
                    kind of visualization.
        org      : (Boolean) default=False
                    if True, plot original data distribution.
                    if False, plot Gibbs sampled data distribution.
        cmap     : (str) default="Blues"
                    colormap of 2D histogram. ignored if kind == "scatter".
        alpha    : (float) default=0.1
                    opacity of scatter plot. ignored if kind == "hist2d".
        grid     : (Boolean) default=True
                    grid on 2D histogram sor scatter plots.
        filename : (str) default=None
                    if not None, output file is generated.
        dpi      : (int) default=72
                    resolution of the output file.
                    ignored if filename == None
        
        Return
        ----------
        (matplotlib.figure)
        """
        columns = list(self.Xs.keys())
        
        # feature index validity check
        if features == None: # take all features
            features = columns             
        elif isinstance(features, int) or isinstance(features, str): # 1-dimensional
            features = [columns[features]]
        elif isinstance(features, list) or isinstance(features, np.ndarray): # >= 2-dimensional
            assert all([isinstance(f, int) or isinstance(f, str) for f in features])
        
        assert all([f in columns for f in features if isinstance(f, str)])
        assert all([f < len(columns) for f in features if isinstance(f, int)])
            
        features = [f if isinstance(f, str) else columns[f] for f in features]

        titles = ["input", "Gibbs sampling"]
        
        # Data 
        if org:
            data = self.input_data.to_numpy()
        else:
            data = self.X_gen_real
        
        
        # Case 1. 1-Dimensional
        if len(features) == 1 or self.ndim == 1:
            feature_name = features[0]
            
            if not figsize:
                figsize = (6, 3)
            fig, axs = plt.subplots(ncols=2, figsize=figsize)
            axs[0].hist(self.input_data[feature_name], edgecolor="w", bins=self.bins[0])
            axs[1].hist(self.X_gen_real[:, columns.index(feature_name)], edgecolor="w", bins=self.bins[0])
            axs[0].set_ylabel("counts")
            
            for title, ax in zip(titles, axs):
                ax.set_xlabel(feature_name)
                ax.set_title(title, pad=8)
            
            fig.tight_layout()
        
        # Case 2. >= 2-Dimensional
        else:
            nfm1 = len(features) - 1
            if not figsize:
                figsize = (3*nfm1, 3*nfm1)
            fig, axes = plt.subplots(ncols=nfm1, nrows=nfm1, 
                                     figsize=figsize)
            
            for i in range(nfm1):
                for j in range(nfm1):
                    if nfm1 == 1:
                        ax = axes
                    else:
                        ax = axes[i, j]
                    
                    idx_ip1 = columns.index(features[i+1])
                    idx_j = columns.index(features[j])
                    
                    if i >= j:
                        if kind == "scatter":
                            ax.scatter(data[:, idx_j], data[:, idx_ip1], 
                                       s=1, alpha=alpha)
                        if kind == "hist2d":
                            bins = [self.binedges_nd[idx_j], self.binedges_nd[idx_ip1]]
                            ax.hist2d(data[:, idx_j], data[:, idx_ip1], 
                                       cmap=cmap, bins=bins)
                        ax.set_xlabel(f"{features[j]}")
                        ax.set_ylabel(f"{features[i+1]}")
                        ax.grid(grid)
                    else:
                        ax.axis(False)

            fig.tight_layout()
            
            
        if filename:
            fig.savefig(filename)
          
        plt.close()
            
        return fig
        
