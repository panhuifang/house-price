# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 09:02:16 2017

@author: hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
import stats
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation, metrics
train_df=pd.read_csv('...train.csv')
test_df=pd.read_csv('...test.csv')

df=pd.concat([train_df.loc[:,'MSSubClass':'SaleCondition'],test_df.loc[:,'MSSubClass':'SaleCondition']])
'''
df.reset_index(inplace=True)
df.drop('index',axis=1,inplace=True)
df=df.reindex_axis(train_df.columns,axis=1)
'''
#数据转换
p=train_df.loc[:,'SalePrice']
train_df.drop('SalePrice',axis=1,inplace=True)
for col in train_df.columns:
    if train_df[col].dtype!=np.object:
        if train_df[col].dropna().skew()>0.75:
            train_df[col]=np.log(train_df[col]+1)
        else:
            pass
    else:
        pass
for col in test_df.columns:
    if test_df[col].dtype!=np.object:
        if test_df[col].dropna().skew()>0.75:
            test_df[col]=np.log(test_df[col]+1)
        else:
            pass
    else:
        pass            
        

#数据初探
train_df['SalePrice'].describe()


#sns.distplot(pd.DataFrame(train_df['SalePrice']))
#查看类别个数情况
def cat_num(df,columns):
    print(df[columns].value_counts())
def cat_null(df,columns,value):
    df.loc[df[columns].isnull(),columns]=value
           
#MZ
cat_num(test_df,'MSZoning')
cat_num(train_df,'MSSubClass')
test_df['MSZoning'].groupby(test_df['MSSubClass']).agg('count')
pd.crosstab(test_df['MSZoning'],test_df['MSSubClass'])
test_df.loc[test_df['MSZoning'].isnull(),'MSZoning']
print(test_df[test_df['MSZoning'].isnull() == True])
test_df.loc[(test_df['MSZoning'].isnull())&(test_df['MSSubClass']==20),'MSZoning']='RL'
test_df.loc[(test_df['MSZoning'].isnull())&(test_df['MSSubClass']==30),'MSZoning']='RM'
test_df.loc[(test_df['MSZoning'].isnull())&(test_df['MSSubClass']==70),'MSZoning']='RM'


#Utilities
cat_num(test_df,'Utilities')
cat_num(train_df,'Utilities')
test_df.drop(['Utilities'],axis=1,inplace=True)
train_df.drop(['Utilities'],axis=1,inplace=True)

#Exterior
cat_num(test_df,'Exterior1st')
cat_num(test_df,'Exterior2nd')
pd.crosstab(test_df['Exterior1st'],test_df['Exterior2nd'])

print(test_df[test_df['Exterior1st'].isnull()==True])
test_df['Exterior1st'][test_df['Exterior1st'].isnull()]='VinylSd'
test_df['Exterior2nd'][test_df['Exterior2nd'].isnull()]='VinylSd'

# MasVnrType & MasVnrArea
print(test_df[['MasVnrType','MasVnrArea']][test_df['MasVnrType'].isnull()==True])
print(train_df[['MasVnrType','MasVnrArea']][train_df['MasVnrType'].isnull()==True])
cat_num(test_df, 'MasVnrType')
cat_num(train_df, 'MasVnrType')
test_df['MasVnrType'][test_df['MasVnrType'].isnull()]='None'
train_df['MasVnrType'][train_df['MasVnrType'].isnull()]='None'
test_df['MasVnrArea'][test_df['MasVnrArea'].isnull()]=0
train_df['MasVnrArea'][train_df['MasVnrArea'].isnull()]=0

#Bsmt
columns=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','BsmtFullBath','BsmtHalfBath']
cat_columns=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']
print(test_df[columns][test_df['BsmtFinType2'].isnull()==True])
print(train_df[columns][train_df['BsmtFinType2'].isnull()==True])
cat_num(test_df,'BsmtQual')
cat_num(test_df,'BsmtCond')
cat_num(test_df,'BsmtExposure')
cat_num(test_df,'BsmtFinType1')
cat_num(test_df,'BsmtFinType2')
cat_num(train_df,'BsmtQual')
cat_num(train_df,'BsmtCond')
cat_num(train_df,'BsmtExposure')
cat_num(train_df,'BsmtFinType1')
cat_num(train_df,'BsmtFinType2')
cat_null(test_df,'BsmtFinSF1',0)
cat_null(test_df,'BsmtFinSF2',0)
cat_null(test_df,'BsmtUnfSF',0)
cat_null(test_df,'BsmtFullBath',0)
cat_null(test_df,'BsmtHalfBath',0)
for col in cat_columns:
    cat_null(train_df,col,'None')

pd.crosstab(test_df['BsmtQual'],test_df['BsmtCond'])
test_df.loc[(test_df['BsmtQual'].isnull())&(test_df['BsmtCond']=='TA'),'BsmtQual']='TA'
test_df.loc[(test_df['BsmtQual'].isnull())&(test_df['BsmtCond']=='Fa'),'BsmtQual']='TA'
for col in cat_columns:
    cat_null(test_df,col,'None')
    
test_df[test_df.columns[test_df.isnull().any()].tolist()].isnull().sum()
train_df[train_df.columns[train_df.isnull().any()].tolist()].isnull().sum()

#df['BsmtFinType2'].value_counts()

#TotalBsmtSF
TB=pd.concat([train_df.TotalBsmtSF,train_df.SalePrice],axis=1)
TB.plot.scatter(x='TotalBsmtSF',y='SalePrice',ylim=(0,800000),xlim=(0,7000))
test_df.loc[test_df['TotalBsmtSF'].isnull(),'TotalBsmtSF']=0

#KitchenQual
test_df['KitchenQual'].value_counts()
pd.crosstab(train_df['KitchenQual'],train_df['KitchenAbvGr'])
test_df.loc[test_df['KitchenQual'].isnull(),'KitchenQual']='TA'

test_df.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
train_df.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

#lotarea
test_df['SqrtLotArea'] = np.sqrt(test_df['LotArea'])
train_df['SqrtLotArea'] = np.sqrt(train_df['LotArea'])
test_df['LotFrontage'].corr(test_df['LotArea'])#0.64
train_df['LotFrontage'].corr(train_df['LotArea'])#0.42

test_df['LotFrontage'].corr(test_df['SqrtLotArea'])#0.7
train_df['LotFrontage'].corr(train_df['SqrtLotArea'])#0.6
test_df['LotFrontage'][test_df['LotFrontage'].isnull()]=test_df['SqrtLotArea'][test_df['LotFrontage'].isnull()]
train_df['LotFrontage'][train_df['LotFrontage'].isnull()]=train_df['SqrtLotArea'][train_df['LotFrontage'].isnull()]

#Functional
test_df['Functional'].value_counts()
test_df['Functional'][test_df['Functional'].isnull()]='Typ'

#FireplaceQu
train_df['GarageFinish'].value_counts()
test_df['GarageFinish'].value_counts()

pd.crosstab(test_df['FireplaceQu'],test_df['Fireplaces'])
test_df['Fireplaces'][test_df['FireplaceQu'].isnull()==True].describe()
train_df['Fireplaces'][train_df['FireplaceQu'].isnull()==True].describe()
test_df['FireplaceQu'][test_df['FireplaceQu'].isnull()]='None'
train_df['FireplaceQu'][train_df['FireplaceQu'].isnull()]='None'


#Garage
col=['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']
print(test_df[col][test_df['GarageType'].isnull()==True])

for columns in col:
    if test_df[columns].dtype==np.object:
        test_df[columns][test_df[columns].isnull()==True]='None'
    else:
        test_df[columns][test_df[columns].isnull()==True]=0
        
for columns in col:
    if train_df[columns].dtype==np.object:
        train_df[columns][train_df[columns].isnull()==True]='None'
    else:
        train_df[columns][train_df[columns].isnull()==True]=0
        

#SaleType
test_df['SaleType'].value_counts()
test_df['SaleType'][test_df['SaleType'].isnull()==True]='WD'

#Electrical
train_df['Electrical'].value_counts()
train_df['Electrical'][train_df['Electrical'].isnull()==True]='SBrkr'

for col in test_df.columns:
    if test_df[col].dtype!=train_df[col].dtype:
        print(col,test_df[col].dtype,train_df[col].dtype)

cols=['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']
for col in cols:
    tm=test_df[col].astype(pd.np.int64)
    tm=pd.DataFrame({col:tm})
    test_df.drop(col,axis=1,inplace=True)
    test_df=pd.concat([test_df,tm],axis=1)
for col in cols:
    tm=train_df[col].astype(pd.np.int64)
    tm=pd.DataFrame({col:tm})
    train_df.drop(col,axis=1,inplace=True)
    train_df=pd.concat([train_df,tm],axis=1)
test_df = test_df.replace({"MSSubClass": {20: "A", 30: "B", 40: "C", 45: "D", 50: "E",
                                                60: "F", 70: "G", 75: "H", 80: "I", 85: "J",
                                                90: "K", 120: "L", 150: "M", 160: "N", 180: "O", 190: "P"}})
train_df = train_df.replace({"MSSubClass": {20: "A", 30: "B", 40: "C", 45: "D", 50: "E",
                                                60: "F", 70: "G", 75: "H", 80: "I", 85: "J",
                                                90: "K", 120: "L", 150: "M", 160: "N", 180: "O", 190: "P"}})

test_df=test_df.replace({'ExterQual':{'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po': 1,'None':0}})
train_df=train_df.replace({'ExterQual':{'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po': 1,'None':0}})
test_df=test_df.replace({'ExterCond':{'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po': 1,'None':0}})
train_df=train_df.replace({'ExterCond':{'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po': 1,'None':0}})
test_df=test_df.replace({'GarageQual':{'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po': 1,'None':0}})
train_df=train_df.replace({'GarageQual':{'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po': 1,'None':0}})
test_df=test_df.replace({'GarageCond':{'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po': 1,'None':0}})
train_df=train_df.replace({'GarageCond':{'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po': 1,'None':0}})   
test_df=test_df.replace({'GarageFinish':{'Fin':3,'RFn':2,'Unf':1, 'None':0}})
train_df=train_df.replace({'GarageFinish':{'Fin':3,'RFn':2,'Unf':1,'None':0}})   
#heatingqc
test_df=test_df.replace({'HeatingQC':{'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po': 1,'None':0}})
train_df=train_df.replace({'HeatingQC':{'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po': 1,'None':0}})
test_df=test_df.replace({'FireplaceQu':{'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po': 1,'None':0}})
train_df=train_df.replace({'FireplaceQu':{'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po': 1,'None':0}})
test_df=test_df.replace({'KitchenQual':{'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po': 1,'None':0}})
train_df=train_df.replace({'KitchenQual':{'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po': 1,'None':0}})
test_df=test_df.replace({'BsmtQual':{'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po': 1,'None':0}})
train_df=train_df.replace({'BsmtQual':{'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po': 1,'None':0}})
test_df=test_df.replace({'BsmtCond':{'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po': 1,'None':0}})
train_df=train_df.replace({'BsmtCond':{'Ex':5,'Gd':4,'TA':3,'Fa':2, 'Po': 1,'None':0}})
test_df=test_df.replace({'BsmtExposure':{'Gd':5,'Av':4,'Mn':3,'No':2, 'NA': 1,'None':0}})
train_df=train_df.replace({'BsmtExposure':{'Gd':5,'Av':4,'Mn':3,'No':2, 'NA': 1,'None':0}})
test_df=test_df.replace({'BsmtFinType2':{'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3, 'LwQ': 2,'Unf':1,'None':0}})
train_df=train_df.replace({'BsmtFinType2':{'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3, 'LwQ': 2,'Unf':1,'None':0}})
test_df=test_df.replace({'BsmtFinType1':{'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3, 'LwQ': 2,'Unf':1,'None':0}})
train_df=train_df.replace({'BsmtFinType1':{'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3, 'LwQ': 2,'Unf':1,'None':0}})

#定量变量        
sns.distplot(test_df['SalePrice'], fit='norm')
sns.distplot(test_df['GrLivArea'], fit='norm')
sns.distplot(test_df['LotArea'], fit='norm')
sns.distplot(test_df['MasVnrArea'], fit='norm')######删？
sns.distplot(test_df['2ndFlrSF'], fit='norm')##shan
sns.distplot(test_df['WoodDeckSF'], fit='norm')##shan
sns.distplot(test_df['OpenPorchSF'], fit='norm')##shan
sns.distplot(test_df['EnclosedPorch'], fit='norm')##shan
sns.distplot(test_df['3SsnPorch'], fit='norm')##删
sns.distplot(test_df['ScreenPorch'], fit='norm')##删
sns.distplot(test_df['PoolArea'], fit='norm')##删
plt.scatter(train_df['Heating'],train_df['SalePrice'])
sns.boxplot(x=train_df['Heating'],y=train_df['SalePrice'])
train_df['Heating'].value_counts()
sns.distplot(test_df['MiscVal'], fit='norm')##删
sns.distplot(test_df['BsmtFinSF1'], fit='norm')##shan
sns.distplot(test_df['BsmtFinSF2'], fit='norm')##删
sns.distplot(test_df['BsmtUnfSF'], fit='norm')
sns.distplot(train_df['TotalBsmtSF'], fit='norm')
sns.distplot(int(test_df['GarageArea']), fit='norm')
#TotalBsmtSF
'''
for n in train_df['TotalBsmtSF'].values:
    if n>0:
        train_df.loc[train_df['TotalBsmtSF']==n,'Bsmt_has']=1
    else:
        train_df.loc[train_df['TotalBsmtSF']==n,'Bsmt_has']=0

train_df['TotalBsmtSF']=np.log(train_df['TotalBsmtSF'])
for n in test_df['TotalBsmtSF'].values:
    if n>0:
        test_df.loc[test_df['TotalBsmtSF']==n,'Bsmt_has']=1
    else:
        test_df.loc[test_df['TotalBsmtSF']==n,'Bsmt_has']=0
'''
#
var='OverallQual'
f,ax=plt.subplots(figsize=(16,8))
data=pd.concat([train_df['KitchenQual'],train_df['SalePrice']],axis=1)
fig=sns.boxplot(x='KitchenQual',y='SalePrice',data=data)
plt.xticks(rotation=90)#rotation刻度旋转角度

#train_df['SalePrice'].skew()#1.88偏度右偏
#train_df['SalePrice'].kurt()#6.54峰度尖顶峰
#train_df['logprice']=np.log(train_df['SalePrice'])
#data=pd.concat([train_df['GrLivArea'],train_df['logprice']],axis=1)
#data.plot.scatter(x='GrLivArea',y='logprice')
train_df[train_df.columns[train_df.isnull().any()].tolist()].isnull().sum()#Alley PoolQC Fence MiscFeature
test_df[test_df.columns[test_df.isnull().any()].tolist()].isnull().sum()#Alley PoolQC Fence MiscFeature

#sns.distplot(pd.DataFrame(train_df['logprice']))
#train_df['logprice'].skew()#0.12偏度右偏
#train_df['logprice'].kurt()#0.8
#虚拟变量和连续值转换
test_df.drop(['Id','Street','LandSlope','Condition2','RoofMatl','Heating','3SsnPorch','ScreenPorch','PoolArea','MiscVal'],axis=1,inplace=True)
train_df.drop(['Id','Street','LandSlope','Condition2','RoofMatl','Heating','3SsnPorch','ScreenPorch','PoolArea','MiscVal'],axis=1,inplace=True)
test_df.drop(['Id'],axis=1,inplace=True)
train_df.drop(['Id'],axis=1,inplace=True)

n=0
for col in train_df.columns:
    if train_df[col].dtype==np.object:
        print(col,cat_num(train_df,col))
        n+=1
m=0
for col in test_df.columns:
    if test_df[col].dtype==np.object:
        print(col,cat_num(test_df,col))
        m+=1
#定性变量中可能特征不一样，导致dummy后的变量不统一
df=pd.concat([train_df,test_df],axis=1)
df.reset_index(inplace=True)
 #ont_hot编码    
dm=pd.DataFrame()
pm=pd.DataFrame()
for col in train_df.columns:
    if train_df[col].dtype==np.object:
        dm=pd.get_dummies(train_df[col]).rename(columns=lambda x:col+'_'+str(x))
        train_df=pd.concat([train_df,dm],axis=1)
        train_df.drop(col,axis=1,inplace=True)
        pm=pd.concat([pm,dm],axis=1)
   
dm_test=pd.DataFrame()
pm_test=pd.DataFrame()
for col in test_df.columns:
    if test_df[col].dtype==np.object:
        dm_test=pd.get_dummies(test_df[col]).rename(columns=lambda x:col+'_'+str(x))
        test_df=pd.concat([test_df,dm_test],axis=1)
        test_df.drop(col,axis=1,inplace=True)
        pm_test=pd.concat([pm_test,dm_test],axis=1)
p=train_df.loc[:,'SalePrice']
for col in train_df.columns:
    if col in test_df.columns:
        pass
    else:
        train_df.drop(col,axis=1,inplace=True)
        
for col in test_df.columns:
    if col in train_df.columns:
        pass
    else:
        test_df.drop(col,axis=1,inplace=True)
train_df=pd.concat([train_df,p],axis=1)


#corr


'''
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea','GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_df[cols], size = 2.5)
plt.show()
'''
rf0 = RandomForestRegressor(oob_score=True, random_state=10)
rf0.fit(train_df.ix[:,:-1],train_df.ix[:,-1])
print(rf0.oob_score_)
y_predprob = np.array(rf0.oob_prediction_,dtype='int64')#0.704082364851

#网络搜索参数
#n_estimaters
param_test1={'n_estimators':np.arange(10,600,20)}
gsearch1=GridSearchCV(estimator=RandomForestRegressor(min_samples_split=10,min_samples_leaf=20,
                                                      max_depth=8,max_features='sqrt',
                                                      random_state=10),
param_grid=param_test1,cv=10)
gsearch1.fit(train_df.ix[:,:-1],train_df.ix[:,-1])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_#650
         
#最大深度和内部节点划分最小样本数
param_test2={'max_depth':np.arange(2,10,2),'min_samples_split':np.arange(2,10,1)}
gsearch2=GridSearchCV(estimator=RandomForestRegressor(n_estimators=650,max_features='sqrt',
                                                      min_samples_leaf=10,random_state=10),
param_grid=param_test2,cv=10)
gsearch2.fit(train_df.ix[:,:-1],train_df.ix[:,-1])
gsearch2.grid_scores_,gsearch2.best_params_,gsearch2.best_score_#8,20
rf1=RandomForestRegressor(n_estimators=650,max_depth=8,min_samples_split=20,min_samples_leaf=20,
                         random_state=10,max_features='sqrt',oob_score=True)
rf1.fit(train_df.ix[:,:-1],train_df.ix[:,-1])
print(rf1.oob_score_)#0.776826810446
#叶子节点最小样本数和内部节点划分最小样本数
param_test3={'min_samples_leaf':np.arange(5,50,5),'min_samples_split':np.arange(5,50,5)}
gsearch3=GridSearchCV(estimator=RandomForestRegressor(n_estimators=650,max_features='sqrt',
                                                      max_depth=8,random_state=10),
param_grid=param_test3,cv=10)
gsearch3.fit(train_df.ix[:,:-1],train_df.ix[:,-1])
gsearch3.grid_scores_,gsearch3.best_params_,gsearch3.best_score_#5,5

rf2=RandomForestRegressor(n_estimators=650,max_depth=8,min_samples_split=5,min_samples_leaf=5,
                         random_state=10,max_features='sqrt',oob_score=True)
rf2.fit(train_df.ix[:,:-1],train_df.ix[:,-1])
print(rf2.oob_score_)#0.823693054231

param_test4={'max_features':np.arange(5,190,5)}
gsearch4=GridSearchCV(estimator=RandomForestRegressor(n_estimators=650,
                                                      max_depth=8,random_state=10,min_samples_leaf=5,
                                                      min_samples_split=5),
param_grid=param_test4,cv=10)
gsearch4.fit(train_df.ix[:,:-1],train_df.ix[:,-1])
gsearch4.grid_scores_,gsearch4.best_params_,gsearch4.best_score_

rf3=RandomForestRegressor(n_estimators=650,max_depth=8,min_samples_split=2,min_samples_leaf=5,
                         random_state=10,max_features=70,oob_score=True)
rf3.fit(train_df.ix[:,:-1],train_df.ix[:,-1])
print(rf3.oob_score_)#70
np.sqrt(190)
print(rf3.feature_importances_)
imp=rf3.feature_importances_
'''
imp=pd.DataFrame({'feature':train_df.columns[:-1],'imp':imp})
imp=imp.sort_values(['imp'],ascending=[0])
select_feature=imp['feature'][:30]
train_df_feature=train_df.loc[:,select_feature]
'''
rf_select_model=rf3.fit(train_df.ix[:,:-1],train_df.ix[:,-1])
subm = pd.read_csv("C:/Users/hp/Desktop/在家学习/房价预测/sample_submission.csv")
subm.iloc[:,1] = np.array(rf_select_model.predict(np.array(test_df)))
cat_num(subm, 'SalePrice')
#subm['SalePrice'] = np.expm1(subm[['SalePrice']])
subm.to_csv('C:/Users/hp/Desktop/在家学习/房价预测/RandomForest-submission1.csv', index=None)
#0.15166
#明天开始降维处理 模型融合
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor  
from sklearn.linear_model import LogisticRegression  
#Stacking  
train_y = np.log1p(train_df.pop('SalePrice')).as_matrix()  # 训练标签  
df = pd.concat([train_df, test_df])   
train_X = train_df.values  
test_X = test_df.values  
  
clfs = [RandomForestRegressor(n_estimators=500,max_features=.3),  
            XGBRegressor(max_depth=6,n_estimators=500),  
            Ridge(15)]  
    #训练过程  
dataset_stack_train = np.zeros((train_X.shape[0],len(clfs)))  
dataset_stack_test = np.zeros((test_X.shape[0],len(clfs)))  
for j,clf in enumerate(clfs):  #遍历索引+元素
    clf.fit(train_X,train_y)  
    y_submission = clf.predict(test_X)  
    y_train = clf.predict(train_X)  
    dataset_stack_train[:,j] = y_train  
    dataset_stack_test[:,j] = y_submission  
print("开始Stacking....")  
clf = RandomForestRegressor(n_estimators=1000,max_depth=8)  
clf.fit(dataset_stack_train,train_y)  
y_submission = clf.predict(dataset_stack_test)  
predictions = np.expm1(y_submission)  
result = pd.DataFrame({"Id": test_df.index, "SalePrice": predictions})  
result.to_csv('C:/Users/hp/Desktop/在家学习/房价预测/stack_result.csv', index=False)  
#0.13082


