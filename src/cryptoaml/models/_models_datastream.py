
# Author: Dylan Vassallo <dylan.vassallo.18@um.edu.mt>

###### importing dependencies #############################################
import numpy as np
import xgboost as xgb
from collections import Counter

from skmultiflow.core.base import BaseSKMObject, ClassifierMixin
from skmultiflow.drift_detection import ADWIN
from skmultiflow.utils import get_dimensions


###### AdaptiveXGBoost ###################################################

# not my implementation code taken from 
# https://github.com/jacobmontiel/AdaptiveXGBoostClassifier
class AdaptiveXGBoostClassifier(BaseSKMObject, ClassifierMixin):
    _PUSH_STRATEGY = 'push'
    _REPLACE_STRATEGY = 'replace'
    _UPDATE_STRATEGIES = [_PUSH_STRATEGY, _REPLACE_STRATEGY]

    def __init__(self,
                 n_estimators=30,
                 learning_rate=0.3,
                 max_depth=6,
                 max_window_size=1000,
                 min_window_size=None,
                 detect_drift=False,
                 update_strategy='replace'):
        """
        Adaptive XGBoost classifier.

        Parameters
        ----------
        n_estimators: int (default=5)
            The number of estimators in the ensemble.

        learning_rate:
            Learning rate, a.k.a eta.

        max_depth: int (default = 6)
            Max tree depth.

        max_window_size: int (default=1000)
            Max window size.

        min_window_size: int (default=None)
            Min window size. If this parameters is not set, then a fixed size
            window of size ``max_window_size`` will be used.

        detect_drift: bool (default=False)
            If set will use a drift detector (ADWIN).

        update_strategy: str (default='replace')
            | The update strategy to use:
            | 'push' - the ensemble resembles a queue
            | 'replace' - oldest ensemble members are replaced by newer ones

        Notes
        -----
        The Adaptive XGBoost [1]_ (AXGB) classifier is an adaptation of the
        XGBoost algorithm for evolving data streams. AXGB creates new members
        of the ensemble from mini-batches of data as new data becomes
        available.  The maximum ensemble  size is fixed, but learning does not
        stop once this size is reached, the ensemble is updated on new data to
        ensure consistency with the current data distribution.

        References
        ----------
        .. [1] Montiel, Jacob, Mitchell, Rory, Frank, Eibe, Pfahringer,
           Bernhard, Abdessalem, Talel, and Bifet, Albert. “AdaptiveXGBoost for
           Evolving Data Streams”. In:IJCNN’20. International Joint Conference
           on Neural Networks. 2020. Forthcoming.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self._first_run = True
        self._ensemble = None
        self.detect_drift = detect_drift
        self._drift_detector = None
        self._X_buffer = np.array([])
        self._y_buffer = np.array([])
        self._samples_seen = 0
        self._model_idx = 0
        if update_strategy not in self._UPDATE_STRATEGIES:
            raise AttributeError("Invalid update_strategy: {}\n"
                                 "Valid options: {}".format(update_strategy,
                                                            self._UPDATE_STRATEGIES))
        self.update_strategy = update_strategy
        self._configure()

    def _configure(self):
        if self.update_strategy == self._PUSH_STRATEGY:
            self._ensemble = []
        elif self.update_strategy == self._REPLACE_STRATEGY:
            self._ensemble = [None] * self.n_estimators
        self._reset_window_size()
        self._init_margin = 0.0
        self._boosting_params = {"silent": True,
                                 "objective": "binary:logistic",
                                 "eta": self.learning_rate,
                                 "max_depth": self.max_depth}
        if self.detect_drift:
            self._drift_detector = ADWIN()

    def reset(self):
        """
        Reset the estimator.
        """
        self._first_run = True
        self._configure()

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """
        Partially (incrementally) fit the model.

        Parameters
        ----------
        X: numpy.ndarray
            An array of shape (n_samples, n_features) with the data upon which
            the algorithm will create its model.

        y: Array-like
            An array of shape (, n_samples) containing the classification
            targets for all samples in X. Only binary data is supported.

        classes: Not used.

        sample_weight: Not used.

        Returns
        -------
        AdaptiveXGBoostClassifier
            self
        """
        for i in range(X.shape[0]):
            self._partial_fit(np.array([X[i, :]]), np.array([y[i]]))
        return self

    def _partial_fit(self, X, y):
        if self._first_run:
            self._X_buffer = np.array([]).reshape(0, get_dimensions(X)[1])
            self._y_buffer = np.array([])
            self._first_run = False
        self._X_buffer = np.concatenate((self._X_buffer, X))
        self._y_buffer = np.concatenate((self._y_buffer, y))
        while self._X_buffer.shape[0] >= self.window_size:
            self._train_on_mini_batch(X=self._X_buffer[0:self.window_size, :],
                                      y=self._y_buffer[0:self.window_size])
            delete_idx = [i for i in range(self.window_size)]
            self._X_buffer = np.delete(self._X_buffer, delete_idx, axis=0)
            self._y_buffer = np.delete(self._y_buffer, delete_idx, axis=0)

            # Check window size and adjust it if necessary
            self._adjust_window_size()

        # Support for concept drift
        if self.detect_drift:
            correctly_classifies = self.predict(X) == y
            # Check for warning
            self._drift_detector.add_element(int(not correctly_classifies))
            # Check if there was a change
            if self._drift_detector.detected_change():
                # Reset window size
                self._reset_window_size()
                if self.update_strategy == self._REPLACE_STRATEGY:
                    self._model_idx = 0

    def _adjust_window_size(self):
        if self._dynamic_window_size < self.max_window_size:
            self._dynamic_window_size *= 2
            if self._dynamic_window_size > self.max_window_size:
                self.window_size = self.max_window_size
            else:
                self.window_size = self._dynamic_window_size

    def _reset_window_size(self):
        if self.min_window_size:
            self._dynamic_window_size = self.min_window_size
        else:
            self._dynamic_window_size = self.max_window_size
        self.window_size = self._dynamic_window_size

    def _train_on_mini_batch(self, X, y):
        if self.update_strategy == self._REPLACE_STRATEGY:
            booster = self._train_booster(X, y, self._model_idx)
            # Update ensemble
            self._ensemble[self._model_idx] = booster
            self._samples_seen += X.shape[0]
            self._update_model_idx()
        else:   # self.update_strategy == self._PUSH_STRATEGY
            booster = self._train_booster(X, y, len(self._ensemble))
            # Update ensemble
            if len(self._ensemble) == self.n_estimators:
                self._ensemble.pop(0)
            self._ensemble.append(booster)
            self._samples_seen += X.shape[0]

    def _train_booster(self, X: np.ndarray, y: np.ndarray, last_model_idx: int):
        d_mini_batch_train = xgb.DMatrix(X, y.astype(int))
        # Get margins from trees in the ensemble
        margins = np.asarray([self._init_margin] * d_mini_batch_train.num_row())
        for j in range(last_model_idx):
            margins = np.add(margins,
                             self._ensemble[j].predict(d_mini_batch_train, output_margin=True))
        d_mini_batch_train.set_base_margin(margin=margins)
        booster = xgb.train(params=self._boosting_params,
                            dtrain=d_mini_batch_train,
                            num_boost_round=1,
                            verbose_eval=False)
        return booster

    def _update_model_idx(self):
        self._model_idx += 1
        if self._model_idx == self.n_estimators:
            self._model_idx = 0

    def predict(self, X):
        """
        Predict the class label for sample X

        Parameters
        ----------
        X: numpy.ndarray
            An array of shape (n_samples, n_features) with the samples to
            predict the class label for.

        Returns
        -------
        numpy.ndarray
            A 1D array of shape (, n_samples), containing the
            predicted class labels for all instances in X.

        """
        if self._ensemble:
            if self.update_strategy == self._REPLACE_STRATEGY:
                trees_in_ensemble = sum(i is not None for i in self._ensemble)
            else:   # self.update_strategy == self._PUSH_STRATEGY
                trees_in_ensemble = len(self._ensemble)
            if trees_in_ensemble > 0:
                d_test = xgb.DMatrix(X)
                for i in range(trees_in_ensemble - 1):
                    margins = self._ensemble[i].predict(d_test, output_margin=True)
                    d_test.set_base_margin(margin=margins)
                predicted = self._ensemble[trees_in_ensemble - 1].predict(d_test)
                return np.array(predicted > 0.5).astype(int)
        # Ensemble is empty, return default values (0)
        return np.zeros(get_dimensions(X)[0])

    def predict_proba(self, X):
        """
        Not implemented for this method.
        """
        raise NotImplementedError("predict_proba is not implemented for this method.")

class AdaptiveStackedBoostClassifier():
    def __init__(self,
                 min_window_size=None, 
                 max_window_size=2000,
                 n_base_models=5,
                 n_rounds_eval_base_model=3,
                 meta_learner_train_ratio=0.4):
        
        self._first_run = True
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        
        # validate 'n_base_models' 
        if n_base_models <= 1:
            raise ValueError("'n_base_models' must be > 1")
        self._n_base_models = n_base_models
        # validate 'n_rounds_eval_base_model' 
        if n_rounds_eval_base_model > n_base_models or n_rounds_eval_base_model <= 0:
            raise ValueError("'n_rounds_eval_base_model' must be > 0 and <= to 'n_base_models'")
        self._n_rounds_eval_base_model = n_rounds_eval_base_model
        self._meta_learner = xgb.XGBClassifier()
        self.meta_learner_train_ratio = meta_learner_train_ratio
        self._X_buffer = np.array([])
        self._y_buffer = np.array([])

        # 3*N matrix 
        # 1st row - base-level model
        # 2nd row - evaluation rounds 
        self._base_models = [[None for x in range(n_base_models)] for y in range(3)]
        
        self._reset_window_size()
        
    def _adjust_window_size(self):
        if self._dynamic_window_size < self.max_window_size:
            self._dynamic_window_size *= 2
            if self._dynamic_window_size > self.max_window_size:
                self.window_size = self.max_window_size
            else:
                self._window_size = self._dynamic_window_size

    def _reset_window_size(self):
        if self.min_window_size:
            self._dynamic_window_size = self.min_window_size
        else:
            self._dynamic_window_size = self.max_window_size
        self._window_size = self._dynamic_window_size

        
    def partial_fit(self, X, y):
        if self._first_run:
            self._X_buffer = np.array([]).reshape(0, X.shape[1])
            self._y_buffer = np.array([])
            self._first_run = False
                           
        self._X_buffer = np.concatenate((self._X_buffer, X))
        self._y_buffer = np.concatenate((self._y_buffer, y))
        while self._X_buffer.shape[0] >= self._window_size:
            self._train_on_mini_batch(X=self._X_buffer[0:self._window_size, :],
                                      y=self._y_buffer[0:self._window_size])
            delete_idx = [i for i in range(self._window_size)]
            self._X_buffer = np.delete(self._X_buffer, delete_idx, axis=0)
            self._y_buffer = np.delete(self._y_buffer, delete_idx, axis=0)
    
    def _train_new_base_model(self, X_base, y_base, X_meta, y_meta):
        
        # new base-level model  
        new_base_model = xgb.XGBClassifier()
        # first train the base model on the base-level training set 
        new_base_model.fit(X_base, y_base)
        # then extract the predicted probabilities to be added as meta-level features
        y_predicted = new_base_model.predict_proba(X_meta)   
        # once the meta-features for this specific base-model are extracted,
        # we incrementally fit this base-model to the rest of the data,
        # this is done so this base-model is trained on a full batch 
        new_base_model.fit(X_meta, y_meta, xgb_model=new_base_model.get_booster())
        return new_base_model, y_predicted
    
    def _construct_meta_features(self, meta_features):
        
        # get size of of meta-features
        meta_features_shape = meta_features.shape[1]  
        # get expected number of features,
        # binary probabilities from the total number of base-level models
        meta_features_expected = self._n_base_models * 2
        
        # since the base-level models list is not full, 
        # we need to fill the features until the list is full, 
        # so we set the remaining expected meta-features as 0
        if meta_features_shape < meta_features_expected:
            diff = meta_features_expected - meta_features_shape
            empty_features = np.zeros((meta_features.shape[0], diff))
            meta_features = np.hstack((meta_features, empty_features)) 
        return meta_features 
        
    def _get_weakest_base_learner(self):
        
        # loop rounds
        worst_model_idx = None 
        worst_performance = 1
        for idx in range(len(self._base_models[0])):
            current_round = self._base_models[1][idx]
            if current_round < self._n_rounds_eval_base_model:
                continue 
            
            current_performance = self._base_models[2][idx].sum()
            if current_performance < worst_performance:
                worst_performance = current_performance 
                worst_model_idx = idx

        return worst_model_idx
    
    def _train_on_mini_batch(self, X, y):
        
        # ----------------------------------------------------------------------------
        # STEP 1: split mini batch to base-level and meta-level training set
        # ----------------------------------------------------------------------------
        base_idx = int(self._window_size * (1.0 - self.meta_learner_train_ratio))
        X_base = X[0: base_idx, :]
        y_base = y[0: base_idx] 

        # this part will be used to train the meta-level model,
        # and to continue training the base-level models on the rest of this batch
        X_meta = X[base_idx:self._window_size, :]  
        y_meta = y[base_idx:self._window_size]
        
        # ----------------------------------------------------------------------------
        # STEP 2: train previous base-models 
        # ----------------------------------------------------------------------------
        meta_features = []
        base_models_len = self._n_base_models - self._base_models[0].count(None)
        if base_models_len > 0: # check if we have any base-level models         
            base_model_performances = self._meta_learner.feature_importances_
            for b_idx in range(base_models_len): # loop and train and extract meta-level features 
                    
                # continuation of training (incremental) on base-level model,
                # using the base-level training set 
                base_model = self._base_models[0][b_idx]
                base_model.fit(X_base, y_base, xgb_model=base_model.get_booster())
                y_predicted = base_model.predict_proba(X_meta) # extract meta-level features 
                                
                # extract meta-features 
                meta_features = y_predicted if b_idx == 0 else np.hstack((meta_features, y_predicted))                    
                
                # once the meta-features for this specific base-model are extracted,
                # we incrementally fit this base-model to the rest of the data,
                # this is done so this base-model is trained on a full batch 
                base_model.fit(X_meta, y_meta, xgb_model=base_model.get_booster())
                                
                # update base-level model list 
                self._base_models[0][b_idx] = base_model
                current_round = self._base_models[1][b_idx]
                last_performance = base_model_performances[b_idx * 2] + base_model_performances[(b_idx*2)+1] 
                self._base_models[2][b_idx][current_round%self._n_rounds_eval_base_model] = last_performance
                self._base_models[1][b_idx] = current_round + 1
                
        # ----------------------------------------------------------------------------
        # STEP 3: with each new batch, we create/train a new base model 
        # ----------------------------------------------------------------------------
        new_base_model, new_base_model_meta_features = self._train_new_base_model(X_base, y_base, X_meta, y_meta)

        insert_idx = base_models_len
        if base_models_len == 0:
            meta_features = new_base_model_meta_features
        elif base_models_len > 0 and base_models_len < self._n_base_models: 
            meta_features = np.hstack((meta_features, new_base_model_meta_features))     
        else: 
            insert_idx = self._get_weakest_base_learner()           
            meta_features[:, insert_idx * 2] = new_base_model_meta_features[:,0]
            meta_features[:, (insert_idx * 2) + 1] = new_base_model_meta_features[:,1]
            
        self._base_models[0][insert_idx] = new_base_model 
        self._base_models[1][insert_idx] = 0 
        self._base_models[2][insert_idx] = np.zeros(self._n_rounds_eval_base_model) 

        # STEP 4: train the meta-level model 
        meta_features = self._construct_meta_features(meta_features)
        
        if base_models_len == 0:
            self._meta_learner.fit(meta_features, y_meta)
        else:
            self._meta_learner.fit(meta_features, y_meta, xgb_model=self._meta_learner.get_booster())

    def predict(self, X):
      
        # only one model in ensemble use its predictions 
        base_models_len = self._n_base_models - self._base_models[0].count(None)
        if base_models_len < self._n_base_models:
            predictions = []
            for i in range(base_models_len):
                tmp_predictions = self._base_models[0][i].predict(X)
                predictions.append(tmp_predictions)
            output = [int(Counter(col).most_common(1)[0][0]) for col in zip(*predictions)] 
            return output
        
        # predict via meta learner 
        meta_features = []           
        for b_idx in range(base_models_len):
            y_predicted = self._base_models[0][b_idx].predict_proba(X) 
            meta_features = y_predicted if b_idx == 0 else np.hstack((meta_features, y_predicted))                    
        meta_features = self._construct_meta_features(meta_features)
        return self._meta_learner.predict(meta_features)