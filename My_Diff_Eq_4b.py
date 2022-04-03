import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import odeint
import math as mt
import pandas
from mpmath import *
import csv

import tensorflow as tf

sigma = 0.01
#sigma = tf.constant([sigma], dtype="float64")

# ARGS to be passed to ODEINT solver: mufib, xcf, pfibp, df, xfn, pfiba, pbl

#ASSIGN THE Constants AND RANDOMIZE THEM numpy.random.normal(loc=0.0, scale=1.0, size=None)
#mufib  pfibp  ninfty  cinfty  df  xfn  pfibm  kfnm  pbl  xcf  cfinfty  hcf  kcn  kcfr  xcfr  Ncrit  cfrinfty  snr  munr  knf  mun  Pinfty  kpm  sm  mum  kmp  O2
#Vinfty  Knn  knw  Finfty  knp  Ocrit  kpg0  kpg  betap

#mufib = 0.1 + np.random.normal(loc=0.0, scale=sigma, size=None)
        #tf.constant([0.1], dtype="float64") + np.random.normal(loc=0.0, scale=sigma, size=None)
#mufib = tf.constant([mufib], dtype="float64")

##mufib[0:401] = 1 + np.rgaussian(0, sigma)
#pfibp = 0.4 + np.random.normal(loc=0.0, scale=sigma, size=None) # should be pfibx, numpy.random.normal(loc=0.0, scale=1.0, size=None)
#pfibp = tf.constant([pfibp], dtype="float64")

ninfty = 0.6 + np.random.normal(loc=0.0, scale=sigma, size=None) # numpy.random.normal(loc=0.0, scale=1.0, size=None)
ninfty = tf.constant([ninfty], dtype="float64")

cinfty = 0.5 + np.random.normal(loc=0.0, scale=sigma, size=None) # numpy.random.normal(loc=0.0, scale=1.0, size=None)
cinfty = tf.constant([cinfty], dtype="float64")

pfibm = 0.6 + np.random.normal(loc=0.0, scale=sigma, size=None)
pfibm = tf.constant([pfibm], dtype="float64")

#pfiba = 0.3 + np.random.normal(loc=0.0, scale=sigma, size=None) # WHERE IS THAT USED?
#pfiba = tf.constant([pfiba], dtype="float64")

#df = 0.3  + np.random.normal(loc=0.0, scale=sigma, size=None)  # numpy.random.normal(loc=0.0, scale=1.0, size=None)
#df = tf.constant([df], dtype="float64")

#xfn = 0.6 + np.random.normal(loc=0.0, scale=sigma, size=None)
#xfn = tf.constant([xfn], dtype="float64")

kfnm = 0.5 + np.random.normal(loc=0.0, scale=sigma, size=None)
kfnm = tf.constant([kfnm], dtype="float64")

#pbl = 0.08 + np.random.normal(loc=0.0, scale=sigma, size=None)
#pbl = tf.constant([pbl], dtype="float64")

kfnp = 0.4 + np.random.normal(loc=0.0, scale=sigma, size=None) # should be kfnx, numpy.random.normal(loc=0.0, scale=1.0, size=None)
kfnp = tf.constant([kfnp], dtype="float64")

kfna = 0.3 + np.random.normal(loc=0.0, scale=sigma, size=None)
kfna = tf.constant([kfna], dtype="float64")

xfn = 0.6  + np.random.normal(loc=0.0, scale=sigma, size=None) # numpy.random.normal(loc=0.0, scale=1.0, size=None)
xfn = tf.constant([xfn], dtype="float64")

cfinfty = 0.8 + np.random.normal(loc=0.0, scale=sigma, size=None)
cfinfty = tf.constant([cfinfty], dtype="float64")

pbl = 0.08 + np.random.normal(loc=0.0, scale=sigma, size=None)  # numpy.random.normal(loc=0.0, scale=1.0, size=None) # FIX IT WHEN COL * FIBa >0.01
pbl = tf.constant([pbl], dtype="float64")

#xcf = 4.2 + np.random.normal(loc=0.0, scale=sigma, size=None)  # WHERE IS THE hcf VALUE
#xcf = tf.constant([xcf], dtype="float64")

hfc = 1.0 # + np.random.normal(loc=0.0, scale=sigma, size=None)  # numpy.random.normal(loc=0.0, scale=1.0, size=None)
hfc = tf.constant([hfc], dtype="float64")

kcn = 2 + np.random.normal(loc=0.0, scale=sigma, size=None)  # numpy.random.normal(loc=0.0, scale=1.0, size=None)
kcn = tf.constant([kcn], dtype="float64")

kcfr = 500 + np.random.normal(loc=0.0, scale=sigma, size=None) # numpy.random.normal(loc=0.0, scale=1.0, size=None)
kcfr = tf.constant([kcfr], dtype="float64")

xcfr = 5 + np.random.normal(loc=0.0, scale=sigma, size=None) # numpy.random.normal(loc=0.0, scale=1.0, size=None)
xcfr = tf.constant([xcfr], dtype="float64")

Ncrit = 0.01 + np.random.normal(loc=0.0, scale=sigma, size=None)
Ncrit = tf.constant([Ncrit], dtype="float64")

cfrinfty = 1.8 + np.random.normal(loc=0.0, scale=sigma, size=None)
cfrinfty = tf.constant([cfrinfty], dtype="float64")

snr = 2 + np.random.normal(loc=0.0, scale=sigma, size=None)
snr = tf.constant([snr], dtype="float64")

munr = 2.88 + np.random.normal(loc=0.0, scale=sigma, size=None)
munr = tf.constant([munr], dtype="float64")

knf = 0.1 + np.random.normal(loc=0.0, scale=sigma, size=None)
knf = tf.constant([knf], dtype="float64")

mun = 1.2 + np.random.normal(loc=0.0, scale=sigma, size=None)
mun = tf.constant([mun], dtype="float64")

Pinfty = 20 + np.random.normal(loc=0.0, scale=sigma, size=None)
Pinfty = tf.constant([Pinfty], dtype="float64")

kpm = 0.6 + np.random.normal(loc=0.0, scale=sigma, size=None)
kpm = tf.constant([kpm], dtype="float64")

sm = 0.12 + np.random.normal(loc=0.0, scale=sigma, size=None)
sm = tf.constant([sm], dtype="float64")

mum = 0.048 + np.random.normal(loc=0.0, scale=sigma, size=None)
mum = tf.constant([mum], dtype="float64")

kmp = 0.108 + np.random.normal(loc=0.0, scale=sigma, size=None) # WHAT IS THAT?
kmp = tf.constant([kmp], dtype="float64")

kpn = 0.2 + np.random.normal(loc=0.0, scale=sigma, size=None)
kpn = tf.constant([kpn], dtype="float64")

O2 = 26 + np.random.normal(loc=0.0, scale=sigma, size=None) # WHAT'S THE CORRECT VALUE HERE?
O2 = tf.constant([O2], dtype="float64")

Ocrit = 25 #+ np.random.normal(loc=0.0, scale=sigma, size=None)
Ocrit = tf.constant([Ocrit ], dtype="float64")

kcf = 1 + np.random.normal(loc=0.0, scale=sigma, size=None)  # WHAT IS THAT? numpy.random.normal(loc=0.0, scale=1.0, size=None)
kcf = tf.constant([kcf], dtype="float64")

xcf = 4.2 + np.random.normal(loc=0.0, scale=sigma, size=None)  # WHERE IS THAT USED? numpy.random.normal(loc=0.0, scale=1.0, size=None)
xcf = tf.constant([xcf], dtype="float64")

# COMPLETE THE REST ...
Knn = 3 + np.random.normal(loc=0.0, scale=sigma, size=None)
Knn = tf.constant([Knn], dtype="float64")

knw = 2 + np.random.normal(loc=0.0, scale=sigma, size=None)
knw = tf.constant([knw], dtype="float64")

Finfty = 6 + np.random.normal(loc=0.0, scale=sigma, size=None)   # fi(*P+*N^3+*, FIB, )
Finfty = tf.constant([Finfty], dtype="float64")

knp = 0.5 + np.random.normal(loc=0.0, scale=sigma, size=None)
knp = tf.constant([knp], dtype="float64")

kpg0 = 0.55 + np.random.normal(loc=0.0, scale=sigma, size=None)
kpg0 = tf.constant([kpg0], dtype="float64")

kpg = 1 + np.random.normal(loc=0.0, scale=sigma, size=None) # WHAT IS THAT?
kpg = tf.constant([kpg], dtype="float64")

betap = 0.3 + np.random.normal(loc=0.0, scale=sigma, size=None) #WS
betap = tf.constant([betap], dtype="float64")

WS0 = 0.49  #+ np.random.normal(loc=0.0, scale=sigma, size=None)
WS0 = tf.constant([WS0], dtype="float64")

wsg = 0.6 + np.random.normal(loc=0.0, scale=sigma, size=None)
wsg = tf.constant([wsg], dtype="float64")


# our own function 1
def get_u(t):
    if t < 5:
        return tf.constant([0.], dtype="float64") #0
    else:
        return tf.constant([1.0], dtype="float64") #1


# our own function 2: NO TF CONVERSION
def fh(x, V):
    if V+x != 0:
        return x / (V + x) #tf.constant([x / (V+x) * 1.0], dtype="float64")  # x / (V + x)
    else:
        return x / (V*1.01 + x) # tf.constant([x / (V*1.01 + x) * 1.0], dtype="float64") # x / (V*1.01 + x) 


# our own function 3: PROB. NO TF CONVERSION
def sh(x):
    return x / (1 + np.exp(-50*x) ) # tf.constant([ x / (1 + np.exp(-50*x) ) * 1.0], dtype="float64")
           # mt.exp() was giving 'OverflowError: math range error'.


# our own function 4: PROB. NO TF CONVERSION
def fi(x, V, Vinfty):
    return x / (1+ mt.pow(V/Vinfty, 2)) # tf.constant([ x / (1+ mt.pow(V/Vinfty, 2)) * 1.0], dtype="float64") # x / (1+ mt.pow(V/Vinfty, 2)) #(V/Vinfty)^2)


# our own function 5
def fc(x,cinfty,hinfty):
    return tf.constant([ 1/(1+mt.pow(x/cinfty, hinfty)) * 1.0], dtype="float64") #1/(1+mt.pow(x/cinfty, hinfty)) #(x/cinfty)^hinfty)


# our own function 6: NO TF CONVERSION
def Ri(WS, FIBp,FIBm,FIBa,COL,N,P):
    #print('knp*P=', knp*P)
    #print('knp*P+Knn*mt.pow(N,3)+knw*WS=', knp*P+Knn*mt.pow(N,3)+knw*WS)
    a = fi(knp*P+Knn*mt.pow(N,3)+knw*WS, FIBa, Finfty)
    #print(a)
    return a # tf.constant([ a * 1.0], dtype="float64")  # a


# our own function 7: NO TF CONVERSION
def kpg(O2):
    if O2 >= Ocrit:
        return kpg0 * 1.0 #  tf.constant([ kpg0 * 1.0], dtype="float64") # kpg0
    else:
        return kpg0+betap*(1-O2/Ocrit) # tf.constant([ kpg0+betap*(1-O2/Ocrit) * 1.0], dtype="float64") # kpg0+betap*(1-O2/Ocrit)


# our own function 8
def g(V):
    return tf.constant([ 1 - 1.1 / (1 + 0.1 * mt.exp(-0.3*(V-Ocrit))) * 1.0], dtype="float64") 
                  # 1 - 1.1 / (1 + 0.1 * mt.exp(-0.3*(V-Ocrit)))


# our own function 9
def WS(COL): 
    zero1 = tf.constant([0.], dtype="float64")
    
    return max( (1-COL) * WS0 / (1 - wsg * g(O2)), zero1) 
    # tf.constant([ max( (1-COL) * WS0 / (1 - wsg * g(O2)), 0) * 1.0], dtype="float64") # max( (1-COL) * WS0 / (1 - wsg * g(O2)), 0)

# EXAMPLE: tf.constant([ 2 * 1.0], dtype="float64")

# initial condition
# z = [10,0,0,0,0,1.05]


true_params = [0.1, 4.2, 0.4, 0.3, 0.6, 0.3, 0.08]
    

def parametric_ode_system(ts, u, args):
    #print(u.shape)
    #print(ts.shape)
    #print('\n\n')
    #return 0
    #print(u)
    #return u

    mufib, xcf, pfibp, df, xfn, pfiba, pbl = \
        args[0], args[1], args[2], args[3], args[4], args[5], args[6]
    #x, y = u[0], u[1]
    #dx_dt = a1*tf.math.cos(x*y) + b1*y + c1*tf.math.exp(-d1*t)
    #dy_dt = a2*x + b2*y + c2*tf.math.exp(-d2*t)

    FIBp, FIBm, FIBa, COL, N, P  = u[0], u[1], u[2], u[3], u[4], u[5]
    
 
    dFIBpdt = (-1)* mufib * FIBp + pfibp * fi(FIBp,N,ninfty) * fc(COL,cinfty,4) - df * fi(FIBp, N, ninfty) - kfnp * fh(N, xfn) * FIBp #(-x + get_u(t))/2.0
    
    dFIBmdt = (-1)* mufib * FIBm + pfibm * fi(FIBm,N,ninfty) * fc(COL,cinfty,4) - df * fi(FIBm, N, ninfty) + df * fi(FIBp, N, ninfty) - kfnm * fh(N, xfn) * FIBm #(-y + x)/5.0
    
    dFIBadt = (-1)* mufib * FIBa + pfiba * fi(FIBa,N,ninfty) * fc(COL,cinfty,4) + df * fi(FIBm, N, ninfty) - kfna * fh(N, xfn) * FIBa + pbl
    
    dCOLdt =  kcf * fh(FIBa,xcf) * fc(COL,cfinfty,hfc) - kcn * N * COL - kcfr * fh(FIBa, xcfr) * sh(Ncrit - N) * (1 - fc(COL, cfrinfty, 12))
    
    dNdt = snr* Ri(WS(COL),FIBp,FIBm,FIBa,COL,N,P) /(munr+Ri(WS(COL),FIBp,FIBm,FIBa,COL,N,P)) - knf * FIBa * N - mun * N # WHICH FIB SHOULD BE PUT IN THE Ri() OF THIS?
    #print('g(O2)=', g(O2))

    dPdt = kpg(O2) * P * (1 - P/Pinfty) - kpm * sm * P / (mum + kmp * P) - kpn * P * fi(N, FIBa, Finfty) * (1 - g(O2))  
    
    return tf.stack([dFIBpdt, dFIBmdt, dFIBadt, dCOLdt, dNdt, dPdt])
    
        
