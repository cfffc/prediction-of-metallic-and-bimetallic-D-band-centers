#!/usr/bin/env python
# coding: utf-8

# In[161]:


import numpy as np
import pandas as pd


# ### 数据处理

# In[162]:


with open('./dcenter_data.csv',encoding='UTF-8') as f:
    df=pd.read_csv(f,index_col=0)


# In[163]:


df


# 查看数据

# In[164]:


df.iloc[1,1]


# In[165]:


df.loc['Fe':'Co','Fe':'Co']


# In[166]:


df.shape


# 依次打印数据

# In[167]:


for i in df.index:
    for j in df.columns:
        print('host '+i+' guest '+j+' val: '+str(df.loc[i,j]))


# 处理数据，将相对值改为绝对值

# In[168]:


for i in df.index:
    for j in df.columns:
        if(i!=j):
            df.loc[i,j]+=df.loc[i,i]
df


# In[169]:


with open('./metal_data.csv',encoding='UTF-8') as f:
    feature=pd.read_csv(f,index_col=0)
feature


# 查看数据

# In[170]:


feature.head()


# In[171]:


feature.iloc[1:3,4:5]


# In[172]:


feature.loc['Co']


# In[173]:


feature.shape


# 制作输入输出特征向量

# In[174]:


x=list()
y=list()
for i in df.index:
    for j in df.columns:
        vec_i=feature.loc[i].to_numpy()
        vec_j=feature.loc[j].to_numpy()
        x_val=np.concatenate((vec_i,vec_j))
        y_val=df.loc[i,j]
        x.append(x_val)
        y.append(y_val)
        if(i=='Fe' and j=='Co'):
            print('host:'+i+' guest  :'+j+' input:'+str(x_val)+' output:'+str(y_val))


# In[175]:


X=np.array(x)
y=np.array(y)


# In[176]:


X.shape,y.shape


# ### 进行机器学习处理

# 1.采用线性回归算法

# In[177]:


#打乱数据
from sklearn.utils import shuffle
X_r,y_r=shuffle(X,y)
X_train,y_train=X_r[:-30,:],y_r[:-30]
X_test,y_test=X_r[-30:,:],y_r[-30:]


# In[178]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)


# In[179]:


y_pred_train_lr=lr.predict(X_train)
y_pred_test_lr=lr.predict(X_test)


# In[180]:


import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.scatter(y_train,y_pred_train_lr,alpha=0.5,color='blue',label='training')
plt.scatter(y_test,y_pred_test_lr,alpha=0.5,color='red',label='test')
plt.legend()
plt.xlabel('DFT')
plt.ylabel('prediction')


# In[181]:


from sklearn.metrics import mean_squared_error
rmse_tr_lr=mean_squared_error(y_train,y_pred_train_lr,squared=False)
rmse_te_lr=mean_squared_error(y_test,y_pred_test_lr,squared=False)
print('RMSE(training)%.3f'%rmse_tr_lr)
print('RMSE(test)%.3f'%rmse_te_lr)


# In[182]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
crossvalidation=KFold(n_splits=10,shuffle=True)
r2_scores_lr=cross_val_score(lr,X,y,scoring='r2',cv=crossvalidation)
rmse_scores_lr=cross_val_score(lr,X,y,scoring='neg_root_mean_squared_error'
                               ,cv=crossvalidation)
print('cv result:')
print('Flod: %i,mean R2:%.3f'%(len(r2_scores_lr),r2_scores_lr.mean()))
print('Flod: %i,mean RMSE:%.3f'%(len(rmse_scores_lr),rmse_scores_lr.mean()))


# 2.采用梯度上升回归的方法

# In[183]:


from sklearn.ensemble import GradientBoostingRegressor


# In[184]:


gbr=GradientBoostingRegressor()
gbr.fit(X_train,y_train)
y_pred_train_gbr=gbr.predict(X_train)
y_pred_test_gbr=gbr.predict(X_test)


# In[185]:


plt.figure(figsize=(6,6))
plt.scatter(y_train,y_pred_train_gbr,alpha=0.5,color='blue',label='training')
plt.scatter(y_test,y_pred_test_gbr,alpha=0.5,color='red',label='test')
plt.legend()
plt.xlabel('DFT')
plt.ylabel('prediction')


# In[186]:


rmse_tr_gbr=mean_squared_error(y_train,y_pred_train_gbr,squared=False)
rmse_te_gbr=mean_squared_error(y_test,y_pred_test_gbr,squared=False)
print('RMSE(training)%.3f'%rmse_tr_gbr)
print('RMSE(test)%.3f'%rmse_te_gbr)


# In[187]:


crossvalidation=KFold(n_splits=10,shuffle=True)
r2_scores_gbr=cross_val_score(gbr,X,y,scoring='r2',cv=crossvalidation)
rmse_scores_gbr=cross_val_score(gbr,X,y,scoring='neg_root_mean_squared_error'
                               ,cv=crossvalidation)
print('cv result:')
print('Flod: %i,mean R2:%.3f'%(len(r2_scores_gbr),r2_scores_gbr.mean()))
print('Flod: %i,mean RMSE:%.3f'%(len(rmse_scores_gbr),rmse_scores_gbr.mean()))


# 3.采用SVM算法

# In[188]:


from sklearn.svm import SVR
svm = SVR()
svm.fit(X_train, y_train)
y_pred_train_svm=svm.predict(X_train)
y_pred_test_svm=svm.predict(X_test)


# In[189]:


plt.figure(figsize=(6,6))
plt.scatter(y_train,y_pred_train_svm,alpha=0.5,color='blue',label='training')
plt.scatter(y_test,y_pred_test_svm,alpha=0.5,color='red',label='test')
plt.legend()
plt.xlabel('DFT')
plt.ylabel('prediction')


# In[190]:


rmse_tr_svm=mean_squared_error(y_train,y_pred_train_svm,squared=False)
rmse_te_svm=mean_squared_error(y_test,y_pred_test_svm,squared=False)
print('RMSE(training)%.3f'%rmse_tr_svm)
print('RMSE(test)%.3f'%rmse_te_svm)


# In[191]:


crossvalidation=KFold(n_splits=10,shuffle=True)
r2_scores_svm=cross_val_score(svm,X,y,scoring='r2',cv=crossvalidation)
rmse_scores_svm=cross_val_score(svm,X,y,scoring='neg_root_mean_squared_error'
                               ,cv=crossvalidation)
print('cv result:')
print('Flod: %i,mean R2:%.3f'%(len(r2_scores_svm),r2_scores_svm.mean()))
print('Flod: %i,mean RMSE:%.3f'%(len(rmse_scores_svm),rmse_scores_svm.mean()))


# 利用其他方法进行数据分析

# In[192]:


df=pd.DataFrame(x)
df['y']=y
df


# In[193]:


df.plot.box()


# In[194]:


idx=[0,'y']
df[idx].plot()


# In[195]:


df.plot.scatter(x=0,y='y')


# In[196]:


df.describe()


# In[197]:


df.corr()['y']


# In[198]:


import seaborn as sns
plt.figure(figsize=(15,1))
sns.heatmap(df.corr()['y'].to_frame().T,cmap='RdYlGn',annot=True)
plt.show()


# In[199]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),cmap='RdYlGn',annot=True)


# In[ ]:




