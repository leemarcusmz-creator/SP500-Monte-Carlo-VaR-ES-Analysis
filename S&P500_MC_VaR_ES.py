#Question 1
import pandas as pd 
import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.lines

sp500 = pd.read_csv('sp500yahoo.csv')
sp500_40days = sp500[0::40].copy()

sp500_40days['lagClose'] = sp500_40days.Close.shift(1)
sp500_40days = sp500_40days[1:] 

sp500_40days['logdiff']=np.log(sp500_40days['Adj Close'])-np.log(sp500_40days['lagClose'])
sp500_40days = sp500_40days[1:]
retVec = sp500_40days['logdiff'].values

# estimate mean and std
retMean = np.mean(retVec)
retStd = np.std(retVec)

T = len(retVec)

#Question 1a
# horizon length and VaR level
rstar = np.percentile(retVec, 2.5)
q1a_log_var = -100*rstar

print('--- Problem 1.a ---')
print('40 day VAR(p=0.025): ', q1a_log_var)

#Question 1b
sp500['lagClose'] = sp500.Close.shift(1)
sp500 = sp500[1:] 

sp500['logdiff']=np.log(sp500['Adj Close'])-np.log(sp500['lagClose'])
retVec_2 = sp500['logdiff'].values

# estimate mean and std
retMean_2 = np.mean(retVec_2)
retStd_2 = np.std(retVec_2)
retStd_2_1 = retStd_2*np.sqrt(40)
T_2 = len(retVec_2)

rstar_delta_norm = stats.norm.ppf(.025,loc=40*retMean_2,scale=retStd_2_1)
q1b_VAR_delta_norm = 100-100.*np.exp(rstar_delta_norm)

print('--- Problem 1.b ---')
print('40 day delta normal VaR(p=0.025): ', q1b_VAR_delta_norm)

#Question 1c
h = 40
p = 0.025
nboot = 10000

port40dayb = np.zeros(nboot)
for i in range(nboot):
    # bootstrap vectors of length h
    retb = np.random.choice(retVec_2,size=h,replace=True)
    # build h day compounded price
    # many ways to do this (sum of logs) port20dayb[i] = 100.*np.exp(np.sum(retb))
    port40dayb[i] = 100.*np.prod(np.exp(retb))  

q1c_VaR = 100. - np.percentile(port40dayb,100.*p)

print('--- Problem 1.c ---')
print('VaR through bootstrap (10,000 iterations): ', q1c_VaR)

#Question 1d
port40dayb_2 = np.zeros(nboot)
for i in range(nboot):
    # bootstrap vectors of length h
    start = np.random.randint(low=0,high=(T_2-h+1),size=1)
    retb_d = retVec_2[start[0]:(start[0]+h)]
    # build 10 day compounded price
    # many ways to do this (sum of logs)
    port40dayb_2[i] = np.prod(np.exp(retb_d))*100.

q1d_VaR = 100. - np.percentile(port40dayb_2,100.*p)

print('--- Problem 1.d ---')
print('VaR through bootstrap (10,000 iterations): ', q1d_VaR)

#Question 2a
p_q2 = 0.01 
nMC = 10000
retmean_MC_q2a = 101
retSTD_MC_q2a = 1

retMC_q2a = np.random.normal(loc=retmean_MC_q2a, scale=retSTD_MC_q2a, size=nMC)
RStar_q2a = np.percentile(retMC_q2a,100.*p_q2)
RTilde_q2a = np.mean(retMC_q2a[retMC_q2a<=RStar_q2a])
esMCNorm_q2a = -100*RTilde_q2a

price_q2a = (1+retMC_q2a)*100.
V = 101 - ((price_q2a - 101.)**2)
V_0 = 100
profitloss = V - V_0
VaRMC_q2a = -np.percentile(profitloss, 1)

print('--- Problem 2.a ---')
print('VaR: ', VaRMC_q2a)
print('ES: ', esMCNorm_q2a)

#Question 2b
rstar_delta_norm_q2b = stats.norm.ppf(p_q2,loc=retmean_MC_q2a,scale=retSTD_MC_q2a)
V = 101 - ((rstar_delta_norm_q2b - 101.)**2)
V_0 = 100
profitloss = V - V_0
q2b_VAR_delta_norm = 100-100.*np.exp(profitloss)

print('--- Problem 2.b ---')
print('VaR: ', q2b_VAR_delta_norm)

print('--- Problem 2.c ---')
print('If price really follows tha normal distribution, in order to calculate the maximum loss expected (worst case scenario), a delta normal method would be more accurate as it assumes all asset returns are normally distributed. As the portfolio return is linear combination of normal variables, it is also normally distributed. ')

print('--- Problem 3.a ---')
print('Judging from the wealth at long horizon, W(T)/W(1) will be is an approximate log normal, where it would be equal to log(100)/r1 + sum(rt,1:T)/r1.')
#Question 3b
print('--- Problem 3.b ---')
print('Tt is expected that you will underperform more than 50% of the time. If returns are lognormal over time, the lognormal is postuvely skewed to the right, where mean is greater than median, which is greater than mode. In this aspect, median return which is the return of 50% of the time is less than the mean return. ')