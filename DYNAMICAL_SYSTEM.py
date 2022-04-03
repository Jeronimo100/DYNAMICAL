#%%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math as mt
import pandas
from mpmath import *
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

sigma = 0.01

#ASSIGN THE Constants AND RANDOMIZE THEM numpy.random.normal(loc=0.0, scale=1.0, size=None)
#mufib  pfibp  ninfty  cinfty  df  xfn  pfibm  kfnm  pbl  xcf  cfinfty  hcf  kcn  kcfr  xcfr  Ncrit  cfrinfty  snr  munr  knf  mun  Pinfty  kpm  sm  mum  kmp  O2
#Vinfty  Knn  knw  Finfty  knp  Ocrit  kpg0  kpg  betap
mufib = 0.1 + np.random.normal(loc=0.0, scale=sigma, size=None)
#mufib[0:401] = 1 + np.rgaussian(0, sigma)
pfibp = 0.4 + np.random.normal(loc=0.0, scale=sigma, size=None) # should be pfibx, numpy.random.normal(loc=0.0, scale=1.0, size=None)
ninfty = 0.6 + np.random.normal(loc=0.0, scale=sigma, size=None) # numpy.random.normal(loc=0.0, scale=1.0, size=None)
cinfty = 0.5 + np.random.normal(loc=0.0, scale=sigma, size=None) # numpy.random.normal(loc=0.0, scale=1.0, size=None)
pfibm = 0.6 + np.random.normal(loc=0.0, scale=sigma, size=None)
pfiba = 0.3 + np.random.normal(loc=0.0, scale=sigma, size=None) # WHERE IS THAT USED?
df = 0.3  + np.random.normal(loc=0.0, scale=sigma, size=None)  # numpy.random.normal(loc=0.0, scale=1.0, size=None)
xfn = 0.6 + np.random.normal(loc=0.0, scale=sigma, size=None)
kfnm = 0.5 + np.random.normal(loc=0.0, scale=sigma, size=None)
pbl = 0.08 + np.random.normal(loc=0.0, scale=sigma, size=None)
kfnp = 0.4 + np.random.normal(loc=0.0, scale=sigma, size=None) # should be kfnx, numpy.random.normal(loc=0.0, scale=1.0, size=None)
kfna = 0.3 + np.random.normal(loc=0.0, scale=sigma, size=None)
xfn = 0.6  + np.random.normal(loc=0.0, scale=sigma, size=None) # numpy.random.normal(loc=0.0, scale=1.0, size=None)
cfinfty = 0.8 + np.random.normal(loc=0.0, scale=sigma, size=None)
pbl = 0.08 + np.random.normal(loc=0.0, scale=sigma, size=None)  # numpy.random.normal(loc=0.0, scale=1.0, size=None) # FIX IT WHEN COL * FIBa >0.01
xcf = 4.2 + np.random.normal(loc=0.0, scale=sigma, size=None)  # WHERE IS THE hcf VALUE
hfc = 1.0 # + np.random.normal(loc=0.0, scale=sigma, size=None)  # numpy.random.normal(loc=0.0, scale=1.0, size=None)
kcn = 2 + np.random.normal(loc=0.0, scale=sigma, size=None)  # numpy.random.normal(loc=0.0, scale=1.0, size=None)
kcfr = 500 + np.random.normal(loc=0.0, scale=sigma, size=None) # numpy.random.normal(loc=0.0, scale=1.0, size=None)
xcfr = 5 + np.random.normal(loc=0.0, scale=sigma, size=None) # numpy.random.normal(loc=0.0, scale=1.0, size=None)
Ncrit = 0.01 + np.random.normal(loc=0.0, scale=sigma, size=None)
cfrinfty = 1.8 + np.random.normal(loc=0.0, scale=sigma, size=None)
snr = 2 + np.random.normal(loc=0.0, scale=sigma, size=None)
munr = 2.88 + np.random.normal(loc=0.0, scale=sigma, size=None)
knf = 0.1 + np.random.normal(loc=0.0, scale=sigma, size=None)
mun = 1.2 + np.random.normal(loc=0.0, scale=sigma, size=None)
Pinfty = 20 + np.random.normal(loc=0.0, scale=sigma, size=None)
kpm = 0.6 + np.random.normal(loc=0.0, scale=sigma, size=None)
sm = 0.12 + np.random.normal(loc=0.0, scale=sigma, size=None)
mum = 0.048 + np.random.normal(loc=0.0, scale=sigma, size=None)
kmp = 0.108 + np.random.normal(loc=0.0, scale=sigma, size=None) # WHAT IS THAT?
kpn = 0.2 + np.random.normal(loc=0.0, scale=sigma, size=None)
O2 = 30 + np.random.normal(loc=0.0, scale=sigma, size=None) # WHAT'S THE CORRECT VALUE HERE?
Ocrit = 25 #+ np.random.normal(loc=0.0, scale=sigma, size=None)
kcf = 1 + np.random.normal(loc=0.0, scale=sigma, size=None)  # WHAT IS THAT? numpy.random.normal(loc=0.0, scale=1.0, size=None)
xcf = 4.2 + np.random.normal(loc=0.0, scale=sigma, size=None)  # WHERE IS THAT USED? numpy.random.normal(loc=0.0, scale=1.0, size=None)
# COMPLETE THE REST ...
Knn = 3 + np.random.normal(loc=0.0, scale=sigma, size=None)
knw = 2 + np.random.normal(loc=0.0, scale=sigma, size=None)
Finfty = 6 + np.random.normal(loc=0.0, scale=sigma, size=None)   # fi(*P+*N^3+*, FIB, )
knp = 0.5 + np.random.normal(loc=0.0, scale=sigma, size=None)
kpg0 = 0.55 + np.random.normal(loc=0.0, scale=sigma, size=None)
kpg = 1 + np.random.normal(loc=0.0, scale=sigma, size=None) # WHAT IS THAT?
betap = 0.3 + np.random.normal(loc=0.0, scale=sigma, size=None) #WS
WS0 = 0.6 #+ np.random.normal(loc=0.0, scale=sigma, size=None)
wsg = 0.6 + np.random.normal(loc=0.0, scale=sigma, size=None)

# our own functions
def get_u(t):
    if t < 5:
        return 0
    else:
        return 1

def fh(x, V):
    #if V+x != 0:
        return x / (V + x)
    #else:
    #    return x / (V*1.01 + x)

def sh(x):
    return x / (1 + np.exp(-50*x) ) # mt.exp() was giving 'OverflowError: math range error'.

def fi(x, V, Vinfty):
    return x / (1+ mt.pow(V/Vinfty, 2)) #(V/Vinfty)^2)

def fc(x,cinfty,hinfty):
    return 1/(1+mt.pow(x/cinfty, hinfty)) #(x/cinfty)^hinfty)

def Ri(WS, FIBp,FIBm,FIBa,COL,N,P):
    #print('knp*P=', knp*P)
    #print('knp*P+Knn*mt.pow(N,3)+knw*WS=', knp*P+Knn*mt.pow(N,3)+knw*WS)
    a = fi(knp*P+Knn*mt.pow(N,3)+knw*WS, FIBa, Finfty)
    #print(a)
    return a

def kpg(O2):
    if O2 >= Ocrit:
        return kpg0
    else:
        return kpg0+betap*(1-O2/Ocrit)

def g(V):
    return 1 - 1.1 / (1 + 0.1 * mt.exp(-0.3*(V-Ocrit)))


# function that returns dz/dt
def model(z,t, input_dict):
    FIBp  = z[0]
    FIBm  = z[1]
    FIBa  = z[2]
    COL   = z[3]
    N     = z[4]
    P     = z[5]

    """#ASSIGN HERE THE DICT'S CONTENT
    mufib = input_dict[' ']
    fi =
    fc =
    fh =
    #Ri = fi(knp*P+Knn*N^3+knw*WS, FIB, Finfty, t) """
    #mufib  pfibp  ninfty  cinfty  df  xfn  pfibm  kfnm  pbl  xcf  cfinfty  hcf  kcn  kcfr  xcfr  Ncrit  cfrinfty  snr  munr  knf  mun  Pinfty  kpm  sm  mum  kmp  O2
    WS = max( (1-COL) * WS0 / (1 - wsg * g(O2)), 0)
    #print('WS=', WS)
    #print('P=', P)
    #print('knp*P+Knn*mt.pow(N,3)+knw*WS=', knp*P+Knn*mt.pow(N,3)+knw*WS)
    #print('FIBa=',FIBa)
    #print('Finfty=', Finfty)
    """print('ninfty=',ninfty)
    print('N=',N)
    print('FIBp=',FIBp)
    print('N/ninfty = ', N/ninfty^2)
    print('fi mocked calculation = ', FIBp / (1+ (N/ninfty)^2))
    print(fi(FIBp,N,ninfty))"""
    dFIBpdt = (-1)* mufib * FIBp + pfibp * fi(FIBp,N,ninfty) * fc(COL,cinfty,4) - df * fi(FIBp, N, ninfty) - kfnp * fh(N, xfn) * FIBp #(-x + get_u(t))/2.0
    dFIBmdt = (-1)* mufib * FIBm + pfibm * fi(FIBm,N,ninfty) * fc(COL,cinfty,4) - df * fi(FIBm, N, ninfty) + df * fi(FIBp, N, ninfty) - kfnm * fh(N, xfn) * FIBm #(-y + x)/5.0
    dFIBadt = (-1)* mufib * FIBa + pfiba * fi(FIBa,N,ninfty) * fc(COL,cinfty,4) + df * fi(FIBm, N, ninfty) - kfna * fh(N, xfn) * FIBa + pbl
    dCOLdt =  kcf * fh(FIBa,xcf) * fc(COL,cfinfty,hfc) - kcn * N * COL - kcfr * fh(FIBa, xcfr) * sh(Ncrit - N) * (1 - fc(COL, cfrinfty, 12))
    dNdt = snr* Ri(WS,FIBp,FIBm,FIBa,COL,N,P) /(munr+Ri(WS,FIBp,FIBm,FIBa,COL,N,P)) - knf * FIBa * N - mun * N # WHICH FIB SHOULD BE PUT IN THE Ri() OF THIS?
    #print('g(O2)=', g(O2))
    dPdt = kpg(O2) * P * (1 - P/Pinfty) - kpm * sm * P / (mum + kmp * P) - kpn * P * fi(N, FIBa, Finfty) * (1 - g(O2))

    dzdt = [dFIBpdt, dFIBmdt, dFIBadt, dCOLdt, dNdt, dPdt]
    return dzdt

# number of time points
n = 401

# time points
t = np.linspace(0,21,n) #(0,21,n)

# step input
u = np.zeros(n)
# change to 2.0 at time = 5.0
u[51:] = 2.0

# store solution
FIBp = np.empty_like(t)
FIBm = np.empty_like(t)
FIBa = np.empty_like(t)
COL  = np.empty_like(t)
N    = np.empty_like(t)
P    = np.empty_like(t)

# initial condition
z0 = [10,0,0,0,0,1.05]

# record initial conditions
FIBp[0] = z0[0]
FIBm[0] = z0[1]
FIBa[0] = z0[2]
COL[0]  = z0[3]
N[0]    = z0[4]
P[0]    = z0[5]

""" FIBp_local = input_dict['Fibp']
    FIBm_local = input_dict['FIBm']
    FIBa_local = input_dict['FIBa']
    COL_local  = input_dict['COL']
    N_local    = input_dict['N']
    P_local    = input_dict['P']
    FIBp[0] = z0[0]
    FIBm[0] = z0[1]
    FIBa[0] = z0[2]
    COL[0] = z0[3]
    N[0] = z0[4]
    P[0] = z0[5]

#INITIALIZE VARs
FIBp = 1
COL = 1
N = 1 """

dict_1 = []
# solve ODE
for i in range(1,n):
    # span for next time step
    tspan = [t[i-1],t[i]]

    """ # CALCULATE THE VALUES OF THE EXTERNAL FUNCTIONS
    dict_1[] =
    dict_1[] =
    dict_1[] =
    dict_1[] =
    dict_1[] =
    dict_1[] =    """

    # solve for next step
    z = odeint(model,z0,tspan, args=(dict_1, )) # odeint(model,z0,tspan)
    # WANT TO PRINT STH HERE?

    # store solution for plotting
    FIBp[i] = z[1][0]
    FIBm[i] = z[1][1]
    FIBa[i] = z[1][2]
    COL[i]  = z[1][3]
    N[i]    = z[1][4]
    P[i]    = z[1][5]
    # next initial condition
    z0 = z[1]
    # print(z)

# plot results
#plt.plot(t,u,'g:',label='u(t)')
#plt.plot(t,FIBp,'b-',label='FIBp(t)')
#plt.plot(t,FIBm,'m|',label='FIBm(t)')
plt.plot(t,FIBa,'c-',label='FIBa(t)')
plt.plot(t,COL,'g-',label='COL(t)')
plt.plot(t,N,'y-',label='N(t)')
#plt.plot(t,P,'k--',label='P(t)')
plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()

# opening the csv file in 'w+' mode
"""file_FIBp = open('FIBp.csv', 'w+', newline='')
# writing the data into the file
with file_FIBp:
    write = csv.writer(file_FIBp)
    write.writerows(FIBp)   # FIBp.writeto(file_FIBp) """

# import pandas
pandas.DataFrame(FIBp).to_csv("FIBp_csv", header=None)
#pandas.DataFrame(FIBp).to_csv("FIBp.csv",header=None) # , index=None)
pandas.DataFrame(FIBm).to_csv("FIBm_csv", header=None)
pandas.DataFrame(FIBa).to_csv("FIBa_csv", header=None)
pandas.DataFrame(COL).to_csv("COL_csv", header=None)
pandas.DataFrame(N).to_csv("N_csv", header=None)
pandas.DataFrame(P).to_csv("P_csv", header=None)

aggr = np.zeros((n,6))
aggr[:,0] = FIBp
aggr[:,1] = FIBm
aggr[:,2] = FIBa
aggr[:,3] = COL
aggr[:,4] = N
aggr[:,5] = P
pandas.DataFrame(aggr).to_csv("all_data_multi_d.csv", header=None)

dataset = pandas.read_csv('all_data_multi_d.csv', header=0, index_col=0)
count = dataset.shape[0]
values = dataset.values
# integer encode direction
#encoder = LabelEncoder()
#values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = pandas.DataFrame(scaled) #(values) #series_to_supervised(scaled, 1, 1)
pandas.DataFrame(reframed).to_csv("all_data_multi_d_reframed.csv", header=None)

# PLOT ALL IN NON REFRAMED FORM HERE
##plt.plot(t,u,'g:',label='u(t)')
plt.plot(t,FIBp,'b-',label='FIBp(t)')
print(FIBp.shape)
plt.plot(t,FIBm,'m|',label='FIBm(t)')
plt.plot(t,FIBa,'c-',label='FIBa(t)')
plt.plot(t,COL,'g-',label='COL(t)')
plt.plot(t,N,'y-',label='N(t)')
plt.plot(t,P,'k--',label='P(t)')

plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()

# PLOT THE REFRAMED HERE
FIBp = scaled[:,0]
print(FIBp.shape)
FIBm = scaled[:,1]
FIBa = scaled[:,2]
COL  = scaled[:,3]
N    = scaled[:,4]
P    = scaled[:,5]

t = t[:400] # print(t[:400].shape)
plt.plot(t,FIBp,'b-',label='FIBp(t)')
plt.plot(t,FIBm,'m|',label='FIBm(t)')
plt.plot(t,FIBa,'c-',label='FIBa(t)')
plt.plot(t,COL,'g-',label='COL(t)')
plt.plot(t,N,'y-',label='N(t)')
plt.plot(t,P,'k--',label='P(t)')

plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()


#import pandas
#import matplotlib.pyplot as plt
#dataset = pandas.read_csv('airline-passengers.csv', usecols=[1], engine='python')
#plt.plot(dataset)
#plt.show()
#print(type(dataset))
#with open('listfile.txt', 'w') as filehandle:
#    dataset.to_string(filehandle)
#    """for listitem in dataset:
#       filehandle.write('%s\n' % listitem) """
# np.savetxt(r'listfile.txt', dataset, fmt='%d')
