# author Qin Liu
# date 2017/07/07
# note that content largely reproduces Max Berggren's article: Using ANNs on small data – Deep Learning vs. Xgboost. 
# the lightgbm part is my own work
# This is mostly for my own studying purposes. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from xgboost.sklearn import XGBClassifier
from skopt import gp_minimize
import lightgbm as lgb
import time

seed = 123456
np.random.seed(seed)
plt.style.use('ggplot')


    
from IPython.display import display

# function for dataframe processing
# input: data: string; target_variable: string
# output: X, y, X_train, X_test, y_train, y_test
def data_process(data,target_variable):
    df = (
        pd.read_csv(data)

        # Rename columns to lowercase and underscores
        .pipe(lambda d: d.rename(columns={
            k: v for k, v in zip(
                d.columns, 
                [c.lower().replace(' ', '_') for c in d.columns]
            )
        }))
        # Switch categorical classes to integers
        .assign(**{target_variable: lambda r: r[target_variable].astype('category').cat.codes})
        .pipe(lambda d: pd.get_dummies(d))
    )

    y = df[target_variable].values
    X = (
        # Drop target variable
        df.drop(target_variable, axis=1)
        # Min-max-scaling (only needed for the DL model)
        .pipe(lambda d: (d-d.min())/d.max()).fillna(0)
        .as_matrix()
    )
    
    # training set and testing set randomly split into 2:1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=seed
    )
    return X,y,X_train, X_test, y_train, y_test 

# artificial neural network for multiclass target
# input: X,y,X_train,X_test,y_train, y_test -- data
#        lr: learning rate; patience: early stopping rounds, load: load model, string
# output: y_pred_class, y_pred_prob and accuracy printed
def ann(X,y,X_train, X_test, y_train, y_test,lr,patience,load):
    
    # model building
    m = Sequential()
    m.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
    m.add(Dropout(0.5))
    m.add(Dense(128, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(128, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(len(np.unique(y)), activation='softmax'))
 
    # compile the model
    m.compile(
         optimizer=optimizers.Adam(lr=lr),
         loss='categorical_crossentropy',
         metrics=['accuracy']
    )
    start_time = time.clock()
    
    if load == "":
        # train the model
        m.fit(
             X_train, 
             # Target class one-hot-encoded
             pd.get_dummies(pd.DataFrame(y_train), columns=[0]).as_matrix(),  
             epochs=200, # Iterations to be run if not stopped by EarlyStopping
             callbacks=[
                 EarlyStopping(monitor='val_loss', patience=patience),
                 ModelCheckpoint('best.model', monitor='val_loss',
                                 save_best_only=True,verbose=1)],
             verbose=2,
             validation_split=0.1,
             batch_size=256, 
         )
        # Load the best model
        m.load_weights("best.model")
   
        print("neural network :",time.clock() - start_time, "seconds")
        print("##############################################################################")
    else:     m.load_weights(load)     
    
    # predict class
    y_pred_class=m.predict_classes(X_test)
    # predict probability of each class
    y_pred_prob=m.predict_proba(X_test)
    # calculate accuracy and display results
    print ('Three layer deep neural net')
    print ('Accuracy: {0:.3f}'.format(accuracy_score(y_test, y_pred_class)))
    display(pd.crosstab(
         pd.Series(y_test, name='Actual'),
         pd.Series(y_pred_class, name='Predicted'),
         margins=True
     ))
    return y_pred_class,y_pred_prob        

# artificial neural network for binary target
# input: X,y,X_train,X_test,y_train, y_test -- data
#        lr: learning rate; patience: early stopping rounds, load: load model, string
# output: y_pred_class, y_pred_prob and accuracy printed
 
def ann_binary(X,y,X_train, X_test, y_train, y_test,lr,patience,load):
    
    m = Sequential()
    m.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
    m.add(Dropout(0.5))
    m.add(Dense(128, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(128, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(len(np.unique(y)), activation='softmax'))
 
    m.compile(
         optimizer=optimizers.Adam(lr=lr),
         loss='binary_crossentropy',
         metrics=['accuracy']
    )
    start_time = time.clock()
    
    if load == "":
        m.fit(
             X_train, # Feature matrix
             # Target class one-hot-encoded
             pd.get_dummies(pd.DataFrame(y_train), columns=[0]).as_matrix(),  
             epochs=200, # Iterations to be run if not stopped by EarlyStopping
             callbacks=[
                 EarlyStopping(monitor='val_loss', patience=patience),
                 ModelCheckpoint('best.model', monitor='val_loss',
                                 save_best_only=True,verbose=1)],
             verbose=2,
             validation_split=0.1,
             batch_size=256, 
         )
    
        print("neural network :",time.clock() - start_time, "seconds")
        print("##############################################################################")
        m.load_weights("best.model")
    else: m.load_weights(load)
    
    y_pred_class=m.predict_classes(X_test)
    y_pred_prob=m.predict_proba(X_test)
    print('\n')
    print ('Accuracy: {0:.3f}'.format(accuracy_score(y_test, y_pred_class)))
    print ('AUC_ROC: {0:.3f}'.format(roc_auc_score(y_test, y_pred_prob[:,1])))
    display(pd.crosstab(
         pd.Series(y_test, name='Actual'),
         pd.Series(y_pred_class, name='Predicted'),
         margins=True
     ))
    return y_pred_class,y_pred_prob

#==================================================================================================
################## XGB ##############################################################################

#===================================================================================================    
#note that in XGBCLassifier, no need to switch between multi and binary, as it will 
#do so automatically
#https://stackoverflow.com/questions/35384977/xgbclassifier-num-class-is-invalid
# evals_result = {}
# self.classes_ = list(np.unique(y))
# self.n_classes_ = len(self.classes_)
#
# if self.n_classes_ > 2:
# # Switch to using a multiclass objective in the underlying XGB instance
# xgb_options["objective"] = "multi:softprob"
# xgb_options['num_class'] = self.n_classes_
# https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py
#===================================================================================================    



# xgbclassifier for both binary and multiclass
# input: data, num_class, load_params: load best params saved
# output: y_pred_prob,y_pred_class,params

def xgb(X_train, X_test, y_train, y_test, num_class=2,load_params={}):

    if load_params=={}:
        params_fixed = {
             'objective': 'binary:logistic',
             'silent': 1,
             'seed': seed,
         }
        # parameter search space
        space = {
             'max_depth': [1, 5],
             'learning_rate': (10**-4, 10**-1),
             'n_estimators': [10, 200],
             'min_child_weight': [1, 20],
             'subsample': (0.05, 0.95),
             'colsample_bytree': (0.3, 1)
         }
        # instantiate
        reg = XGBClassifier(**params_fixed)
        
        # the function to be minimized
        def objective_f(params):
             """ Wrap a cross validated inverted `accuracy` as objective func """
             reg.set_params(**{k: p for k, p in zip(space.keys(), params)})
             return 1-np.mean(cross_val_score(
                 reg, X_train, y_train, cv=5, n_jobs=-1,
                 scoring='accuracy')
             )
            
        start_time = time.clock()
        # use gp_minimize to find minimum
        res_gp = gp_minimize(objective_f, space.values(), n_calls=50, random_state=seed,verbose=True)
        # extract the best parameters
        best_hyper_params = {k: v for k, v in zip(space.keys(), res_gp.x)}
        
        # update
        params = best_hyper_params.copy()
        params.update(params_fixed)
    else: params=load_params
    
    one_iter=time.clock()  
    # pass in the best parameters
    clf = XGBClassifier(**params)
    # fit
    clf.fit(X_train, y_train)
    end_time=time.clock()
    # predict probabilities of each class
    y_pred_prob =clf.predict_proba(X_test)
    # predict class
    y_pred_class = clf.predict(X_test)
    
    # display results
    print("##############################################################################")
    print("one iter time:",end_time-one_iter, 'seconds')
    if load_params=={}:
        print("xgboost ",end_time-start_time, "seconds")
    print("##############################################################################")
    if num_class==2:
        print ('AUC_ROC: {0:.3f}'.format(roc_auc_score(y_test, y_pred_prob[:,1])))
    
    print ('\n')
    print ('Accuracy: {0:.3f}'.format(accuracy_score(y_test, y_pred_class)))
    display(pd.crosstab(
         pd.Series(y_test, name='Actual'),
         pd.Series(y_pred_class, name='Predicted'),
         margins=True
    ))
    return y_pred_prob,y_pred_class,params

################################################################################    

#Lightgbm

#################################################################################
def f_lgb(X_train, X_test, y_train, y_test,num_class,load_param={}):
    num=len(X_train)*0.1
    num=int(num)
    num2=int(num+1)
    lgb_train=lgb.Dataset(X_train[num2:,:],y_train[num2:])
    lgb_eval=lgb.Dataset(X_train[:num,:],y_train[:num])
    
    if load_param =={}:   
        lgb_params_fixed = {
                'task':'train',
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'num_class': num_class,
                'metric': {'multi_logloss'},
                'metric_freq': 1,
                'is_training_metric':'true',
                'verbose': 0,
                }    
        # paramater search space
        lgb_space ={
                'max_bin':[150,350], 
                'learning_rate': (0.001,0.03), 
                'num_leaves':[8,128], 
                'num_trees':[50,300]
                }
        obj=str(lgb_params_fixed['metric']); obj=obj[2:-2]+'-mean'
        # the function to minimize
        def objective_f2(lgb_params):
            params=lgb_params_fixed.copy()
            lgb_params={k: p for k, p in zip(lgb_space.keys(), lgb_params)}
            params.update(lgb_params)
            bst=lgb.cv(params,lgb_train,num_boost_round=1000,nfold=5,early_stopping_rounds=50)
            n=len(bst[obj])-1
            return bst[obj][int(n)]
            
        start_time=time.clock() 
        res_gp = gp_minimize(objective_f2, lgb_space.values(), n_calls=50, random_state=seed,verbose=True)
        best_hyper_params = {k: v for k, v in zip(lgb_space.keys(), res_gp.x)}
        # update parameter
        params = best_hyper_params.copy()
        params.update(lgb_params_fixed)
    else: params=load_param
    obj=str(params['metric']); obj=obj[2:-2]+'-mean'
    
    bst=lgb.cv(params,lgb_train,num_boost_round=5000,nfold=5,early_stopping_rounds=50)
    one_iter=time.clock()
    gbm = lgb.train(params,train_set=lgb_train,valid_sets=lgb_eval,num_boost_round=len(bst[obj]))
    print(time.clock()-one_iter, 'seconds')
    if load_param=={}:
        print("total time",time.clock()-start_time, "seconds")
    print("##############################################################################")

    gbm.save_model('model.txt')
    
    y_pred_prob = gbm.predict(X_test)
    y_pred_class=y_pred_prob.argmax(axis=1)
    print ('Accuracy: {0:.4f}'.format(accuracy_score(y_test, y_pred_class)))
    display(pd.crosstab(
         pd.Series(y_test, name='Actual'),
         pd.Series(y_pred_class, name='Predicted'),
         margins=True
     ))
    return y_pred_prob, y_pred_class,params



def f_lgb_binary(X_train, X_test, y_train, y_test, load_param):
    num=len(X_train)*0.1
    num=int(num)
    num2=int(num+1)
    lgb_train=lgb.Dataset(X_train[num2:,:],y_train[num2:])
    lgb_eval=lgb.Dataset(X_train[:num,:],y_train[:num])
    
    if load_param=={}:
        lgb_params_fixed = {
                'task':'train',
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': {'auc'},
                'metric_freq': 1,
                'is_training_metric':'true',
                'verbose': 0,
                }
    
        obj=str(lgb_params_fixed['metric']); obj=obj[2:-2]+'-mean'
            
        lgb_space ={
                'max_bin':[150,350],
                'learning_rate': (0.001,0.03),
                'num_leaves':[8,128], 
                'num_trees':[50,200]
                }
               
        def objective_f2(lgb_params):
            params=lgb_params_fixed.copy()
            lgb_params={k: p for k, p in zip(lgb_space.keys(), lgb_params)}
            params.update(lgb_params)
            bst=lgb.cv(params,lgb_train,num_boost_round=1000,nfold=5,early_stopping_rounds=50)
            n=len(bst[obj])-1
            return 1-bst[obj][int(n)]
            
        start_time=time.clock() 
        res_gp = gp_minimize(objective_f2, lgb_space.values(), n_calls=50, random_state=seed,verbose=True)
        best_hyper_params = {k: v for k, v in zip(lgb_space.keys(), res_gp.x)}
        
        params = best_hyper_params.copy()
        params.update(lgb_params_fixed)
    else: params=load_param
    obj=str(params['metric']); obj=obj[2:-2]+'-mean'

    
    bst=lgb.cv(params,lgb_train,num_boost_round=5000,nfold=5,early_stopping_rounds=50)
    one_iter=time.clock()
    gbm = lgb.train(params,train_set=lgb_train,valid_sets=lgb_eval,num_boost_round=len(bst[obj]))
    print(time.clock()-one_iter, 'seconds')
    if load_param=={}:
        print(time.clock()-start_time, "seconds")
    print("##############################################################################")

    gbm.save_model('model.txt')
    
    y_pred_prob = gbm.predict(X_test)
    y_pred_class=[0]*(len(y_test))

    for i in range(len(y_pred_prob)):
        if y_pred_prob[i]>=0.5:
            y_pred_class[i]=1
        else:
            y_pred_class[i]=0
    
    
    print ('Accuracy: {0:.3f}'.format(accuracy_score(y_test, y_pred_class)))
    print ('AUC_ROC: {0:.3f}'.format(roc_auc_score(y_test, y_pred_prob)))
    display(pd.crosstab(
         pd.Series(y_test, name='Actual'),
         pd.Series(y_pred_class, name='Predicted'),
         margins=True
     ))
    return y_pred_prob, y_pred_class,params

#y_pred_prob, y_pred_class,params=f_lgb_binary(X_train, X_test, y_train, y_test)
#y_pred_prob, y_pred_class,params=f_lgb(X_train, X_test, y_train, y_test,num_class)

  
def ann_binary_8layers(X,y,X_train, X_test, y_train, y_test,lr,patience,load):
    
#    m = Sequential()
#    m.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
#    m.add(Dropout(0.3))
#    m.add(Dense(256, activation='relu'))
#    m.add(Dropout(0.2))
#    m.add(Dense(512, activation='relu'))
#    m.add(Dropout(0.3))
#    m.add(Dense(128, activation='relu'))
#    m.add(Dropout(0.3))
#    m.add(Dense(256, activation='relu'))
#    m.add(Dropout(0.3))
#    m.add(Dense(256, activation='relu'))
#    m.add(Dropout(0.3))
#    m.add(Dense(256, activation='relu'))
#    m.add(Dropout(0.3))
#    m.add(Dense(len(np.unique(y)), activation='softmax'))   
    m = Sequential()
    m.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
    m.add(Dropout(0.5))
    m.add(Dense(64, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(32, activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(1, activation='sigmoid'))    
 
    m.compile(
         optimizer=optimizers.Adam(lr=lr),
         loss='binary_crossentropy',
         metrics=['accuracy']
    )
    start_time = time.clock()
    
    if load == "":
        m.fit(
             X_train, # Feature matrix
             # Target class one-hot-encoded
             pd.get_dummies(pd.DataFrame(y_train), columns=[0]).as_matrix(),  
             epochs=200, # Iterations to be run if not stopped by EarlyStopping
             callbacks=[
                 EarlyStopping(monitor='val_loss', patience=patience),
                 ModelCheckpoint('best.model', monitor='val_loss',
                                 save_best_only=True,verbose=1)],
             verbose=2,
             validation_split=0.1,
             batch_size=256, 
         )
    
        print("neural network :",time.clock() - start_time, "seconds")
        print("##############################################################################")
        m.load_weights("best.model")
    else: m.load_weights(load)
    
    y_pred_class=m.predict_classes(X_test)
    y_pred_prob=m.predict_proba(X_test)
    print('\n')
    print ('Accuracy: {0:.3f}'.format(accuracy_score(y_test, y_pred_class)))
    print ('AUC_ROC: {0:.3f}'.format(roc_auc_score(y_test, y_pred_prob[:,1])))
    display(pd.crosstab(
         pd.Series(y_test, name='Actual'),
         pd.Series(y_pred_class, name='Predicted'),
         margins=True
     ))
    return y_pred_class,y_pred_prob

def f(x):
    return np.sin(4*x[0])*(1-np.tanh(x[0]**2))
res=gp_minimize(
        f,
        [(-2.0,2.0)],
        n_calls=30,
        n_random_starts=5,
        random_state=123)
print("x*=%.4f,f(x*)=%.4f" % (res.x[0],res.fun))
