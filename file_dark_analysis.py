# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 18:08:48 2020

@author: Teo
"""

#leggo i file generati dallo script read_file_hk.py e costruisco grafici


import os
import glob

import pandas as pd
import csv
import io
import pickle

import math
import array
import numpy as np
import numpy.linalg
from matplotlib import pylab as pt
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from PIL import Image
from mpl_toolkits import mplot3d
from IPython.display import display, Image

#%%
os.chdir('C:\\Alessandra\\STC_CAL\\DARK_IN_FLIGHT\\02-ICO_01\\04_-_STC_Performance_Test\\stc')
file=glob.glob('.\\dati*')



f420=np.zeros((64,896,1))
f550=np.zeros((64,896,1))
f750=np.zeros((64,896,1))
f920=np.zeros((64,896,1))
WX=np.zeros((64,128,1))
f420m=np.zeros((64,896,1))
f550m=np.zeros((64,896,1))
f750m=np.zeros((64,896,1))
f920m=np.zeros((64,896,1))
WXm=np.zeros((64,128,1))
uWX=np.zeros((64,128,1))
u420=np.zeros((64,896,1))
u550=np.zeros((64,896,1))
u750=np.zeros((64,896,1))
u920=np.zeros((64,896,1))

calWX=np.zeros((64,128,1))
cal420=np.zeros((64,896,1))
cal550=np.zeros((64,896,1))
cal750=np.zeros((64,896,1))
cal920=np.zeros((64,896,1))
meanWX=np.zeros(1)

# cal420b=np.zeros((64,896,1))
# cal550b=np.zeros((64,896,1))
# cal750b=np.zeros((64,896,1))
# cal920b=np.zeros((64,896,1))

# cal420_2=np.zeros((64,896,1))
# cal550_2=np.zeros((64,896,1))
# cal750_2=np.zeros((64,896,1))
# cal920_2=np.zeros((64,896,1))

cal420_2b=np.zeros((64,896,1))
cal550_2b=np.zeros((64,896,1))
cal750_2b=np.zeros((64,896,1))
cal920_2b=np.zeros((64,896,1))


IT=[]
ITf=np.zeros(1)
ITw=np.zeros(1)
TFPA1=np.zeros(1)
TFPA2=np.zeros(1)
RT=[]
for i in range(0,20): #len(file)
    filen=file[i]
    print(filen)
    with open(filen,'rb') as fi:
        data =pickle.load(fi)
       # print(data['namef'][41:50])
        #print(len(data['namef']))
   # breakpoint()
    # for ji in range(1,len(data['namef'])):
    #     if data['namef'][ji] == 'filterx':
    #         for j in range(1,len(data['img_WX'][0,0,:])):
    #             wx = data['img_WX'][:,:,j]
    #             WX=np.dstack((WX,wx))
    #     else:
    #         for jj in range(1,len(data['img_cfilt'][0,0,:])):
    #             print('currente name=',data['namef'][jj])
    #             if data['namef'][jj] == 'f750':
    #                 matf = data['img_cfilt'][:,:,jj]
    #                 f750=np.dstack((f750,matf))
    #               #  print('nome=',data['namef'])
    #             if data['namef'][jj] == 'f420':
    #                 matf = data['img_cfilt'][:,:,jj]
    #                 f420=np.dstack((f420,matf))
    #              #   print('nome=',data['namef'])
    #             if data['namef'][jj] == 'f550':
    #                 matf = data['img_cfilt'][:,:,jj]
    #                 f550=np.dstack((f550,matf))
    #                # print('nome=',data['namef'])
    #             if data['namef'][jj] == 'f920':
    #                 matf = data['img_cfilt'][:,:,jj]
    #                 f920=np.dstack((f920,matf))
    #               #  print('nome=',data['namef'])

    wx = data['img_WX'][:,:,1:10]
    WX=np.dstack((WX,wx))
    matf750 = data['img_cfilt'][:,:,1:10]
    f750=np.dstack((f750,matf750))
    matf420 = data['img_cfilt'][:,:,11:20]
    f420=np.dstack((f420,matf420))
    matf550 = data['img_cfilt'][:,:,21:30]
    f550=np.dstack((f550,matf550))
    matf920 = data['img_cfilt'][:,:,31:40]
    f920=np.dstack((f920,matf920))
 #   breakpoint()

    media750=np.average(f750,axis=2)
    f750m=np.dstack((f750m,media750))
    media420=np.average(f420,axis=2)
    f420m=np.dstack((f420m,media420))
    media550=np.average(f550,axis=2)
    f550m=np.dstack((f550m,media550))
    media920=np.average(f920,axis=2)
    f920m=np.dstack((f920m,media920))
    mediawx=np.average(WX,axis=2)
    WXm=np.dstack((WXm,mediawx))

#   img a cui viene sottratta l'ultima immagine di ogni serie di IT
    ultimaWX=data['img_WX'][:,:,10]
    uWX=np.dstack((uWX,ultimaWX))
    meanWX=np.hstack((meanWX,np.average(uWX))) 
    ultima750=data['img_cfilt'][:,:,10]
    u750=np.dstack((u750,ultima750))
    ultima420=data['img_cfilt'][:,:,20]
    u420=np.dstack((u420,ultima420))
    ultima550=data['img_cfilt'][:,:,30]
    u550=np.dstack((u550,ultima550))
    ultima920=data['img_cfilt'][:,:,40]
    u920=np.dstack((u920,ultima920))

    zval=len(wx[0,0,:])
    calWX=wx-np.repeat(ultimaWX.reshape(64,128,1),zval,axis=2)       
    meanWX=np.average(calWX)
     
    cal750=matf750-np.repeat(ultima750.reshape(64,896,1),zval,axis=2)-meanWX 
    cal420=matf420-np.repeat(ultima420.reshape(64,896,1),zval,axis=2)-meanWX  
    cal550=matf550-np.repeat(ultima550.reshape(64,896,1),zval,axis=2)-meanWX
    cal920=matf920-np.repeat(ultima920.reshape(64,896,1),zval,axis=2)-meanWX

    # cal750_2=cal750-meanWX
    # cal420_2=cal420-meanWX
    # cal550_2=cal550-meanWX RISCRIVERE SENZA SATURARE LA RAM!!!!!!!!!!!!!!!!!!!
    # cal920_2=cal920-meanWX    
    
    # cal750b=np.dstack((cal750b,cal750))
    # cal420b=np.dstack((cal420b,cal420))
    # cal550b=np.dstack((cal550b,cal550))
    # cal920b=np.dstack((cal920b,cal920))

    cal750_2b=np.dstack((cal750_2b,cal750))
    cal420_2b=np.dstack((cal420_2b,cal420))
    cal550_2b=np.dstack((cal550_2b,cal550))
    cal920_2b=np.dstack((cal920_2b,cal920))

    del ultimaWX,ultima420,ultima550,ultima750,ultima920,media750,media420,media550,media920,mediawx,cal750,cal550,cal420,cal920

    #breakpoint()

    IT.append(data['IT'][10])
    f_IT=np.array(data['IT'][0:len(data['img_cfilt'][0,0,1:10])],dtype=float)
    ITf=np.hstack((ITf,f_IT))
    print('LUNGHEZZA ITF=',len(ITf))
    print('lunghezza f420=',len(f420[0,0,:]))
  #  breakpoint()
    w_IT=np.array(data['IT'][0:len(data['img_WX'][0,0,:])],dtype=float)
    ITw=np.hstack((ITw,w_IT))   
    RT.append(data['RT'])
    
#breakpoint()



# sott2=np.subtract(cal750,meanWX)
# cal750b=np.dstack((cal750b,sott2))
# sott2=np.subtract(cal420,meanWX)
# cal420b=np.dstack((cal420b,sott2))
# sott2=np.subtract(cal550,meanWX)
# cal550b=np.dstack((cal550b,sott2))
# sott2=np.subtract(cal920,meanWX)
# cal920b=np.dstack((cal920b,sott2))
#%% immagini medie non calibrate    
path_out2='C:\\Alessandra\\STC_CAL\\DARK_IN_FLIGHT\\02-ICO_01\\04_-_STC_Performance_Test\\stc\\grafici_NON_calibr\\' 
for t in range(0,len(IT)):
    if IT[t] == 0:
        IT[t] = 4.0e-5

#ind=1
#for ind in range(0,len(ITf)):
for ind1 in range(1,len(IT)):
  #  breakpoint()
    plt.subplot(411)
    plt.title('F920 IT='+str(IT[ind1-1])+' ms')
    plt.imshow(f920m[:,:,ind1], cmap=plt.cm.BuPu_r)
    plt.subplot(412)
    plt.title('F550')
    plt.imshow(f550m[:,:,ind1], cmap=plt.cm.BuPu_r)
    plt.subplot(413)
    plt.title('F420')
    plt.imshow(f420m[:,:,ind1], cmap=plt.cm.BuPu_r)
    plt.subplot(414)
    plt.title('F750')
    plt.imshow(f750m[:,:,ind1], cmap=plt.cm.BuPu_r)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax,orientation='vertical')    
   # plt.show()
    plt.savefig(path_out2+'NON_calib_IT'+str(IT[ind1-1])+'ms_'+str(ind1)+'.png')
    plt.clf()
  #  breakpoint()
    print(ind1)
   # breakpoint()
#plt.subplot(111)
#%%immagini a cui è tolta l'ultima img di ciascuna serie di IT e la media della WX (anch'essa calibrata prendendp l'ultima img
#di ciascuna serie e poi fatta la media spaziale)
#ind=172
path_out='C:\\Alessandra\\STC_CAL\\DARK_IN_FLIGHT\\02-ICO_01\\04_-_STC_Performance_Test\\stc\\grafici_sottratto_WX\\'

for t in range(0,len(ITf)):
    if ITf[t] == 0:
        ITf[t] = 4.0e-5


# f, (ax1, ax2) = plt.subplots(2, 1,sharex=True)

# # Jet
# p = ax1.pcolormesh(lon,lat,data, vmin=0, vmax=0.9, cmap='jet')
# f.colorbar(p,ax=ax1)
# ax1.set_title('jet')

# # Discrete
# p = ax2.pcolormesh(lon,lat,data, cmap=cmap, norm=norm)
# f.colorbar(p,ax=ax2,ticks=bounds)
# ax2.set_title('defined')

# cdict = {
#   'red'  :  ( (0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
#   'green':  ( (0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
#   'blue' :  ( (0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))}
        
#cm = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
#ax, _ = mpl.colorbar.make_axes(plt.gca(), shrink=0.5)
#cbar = mpl.colorbar.ColorbarBase(ax, cmap=cm ,norm=mpl.colors.Normalize(vmin=-min_val, vmax=max_val))
#cbar.set_clim(-2.0, 2.0)    
    
for ind in range(0,len(ITf)):
    f,(ax1,ax2,ax3,ax4)=plt.subplots(4,1,figsize=(5,10))
    p=ax1.pcolormesh(cal920_2b[:,:,ind], cmap=plt.cm.BuPu_r)
    ax1.set_title('F920 IT='+str(ITf[ind]*1000.0)+' ms')
    aa=cal920_2b[:,:,ind]
    mini=np.min((aa))
    maxi=np.max((aa))  
    #print(mini,maxi)
    min_val=np.mean(aa)-3*np.std(aa)     
    max_val=np.mean(aa)+3*np.std(aa)
    #breakpoint()
 #   cbar = mpl.colorbar.ColorbarBase(ax1, cmap=cm ,norm=mpl.colors.Normalize(vmin=-min_val, vmax=max_val),orientation='horizontal')         
    f.colorbar(p,ax=ax1,orientation='horizontal',cmap=plt.cm.BuPu_r) #,boundaries=(min_val,max_val)

    #p=plt.title('F550')
    p=ax2.pcolormesh(cal550_2b[:,:,ind], cmap=plt.cm.BuPu_r)
    ax2.set_title('F550')
    aa=cal550_2b[:,:,ind]
    mini=np.nanmin((aa))
    maxi=np.nanmax((aa)) 
   # print(mini,maxi)        
    f.colorbar(p,ax=ax2,orientation='horizontal')#,boundaries=(mini,maxi),cmap=plt.cm.BuPu_r)

#    p=plt.title('F420')
    p=ax3.pcolormesh(cal420_2b[:,:,ind], cmap=plt.cm.BuPu_r)
    ax3.set_title('F420')
    aa=cal420_2b[:,:,ind]
    mini=np.nanmin((aa))
    maxi=np.nanmax((aa))  
    #print(mini,maxi)       
    f.colorbar(p,ax=ax3,orientation='horizontal') #,boundaries=(mini,maxi),cmap=plt.cm.BuPu_r)

#    p=plt.title('F750')
    p=ax4.pcolormesh(cal750_2b[:,:,ind], cmap=plt.cm.BuPu_r)
    ax4.set_title('F750')
    aa=cal750_2b[:,:,ind]
    mini=np.nanmin((aa))
    maxi=np.nanmax((aa))        
    f.colorbar(p,ax=ax4,orientation='horizontal') #,boundaries=(mini,maxi),cmap=plt.cm.BuPu_r)

    plt.savefig(path_out+'calib_IT'+str(ITf[ind]*1000.0)+'ms_'+str(ind)+'.png')
    plt.clf()
   
    print(ind)
   # breakpoint()
#plt.subplot(111)    

#%% immagini non calibrate
    
path_out='C:\\Alessandra\\STC_CAL\\DARK_IN_FLIGHT\\02-ICO_01\\04_-_STC_Performance_Test\\stc\\grafici_non_calib\\'

for t in range(0,len(ITf)):
    if ITf[t] == 0:
        ITf[t] = 4.0e-5


# f, (ax1, ax2) = plt.subplots(2, 1,sharex=True)

# # Jet
# p = ax1.pcolormesh(lon,lat,data, vmin=0, vmax=0.9, cmap='jet')
# f.colorbar(p,ax=ax1)
# ax1.set_title('jet')

# # Discrete
# p = ax2.pcolormesh(lon,lat,data, cmap=cmap, norm=norm)
# f.colorbar(p,ax=ax2,ticks=bounds)
# ax2.set_title('defined')

# cdict = {
#   'red'  :  ( (0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
#   'green':  ( (0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
#   'blue' :  ( (0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))}
        
#cm = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
#ax, _ = mpl.colorbar.make_axes(plt.gca(), shrink=0.5)
#cbar = mpl.colorbar.ColorbarBase(ax, cmap=cm ,norm=mpl.colors.Normalize(vmin=-min_val, vmax=max_val))
#cbar.set_clim(-2.0, 2.0)    
    
for ind in range(0,len(ITf)):
    f,(ax1,ax2,ax3,ax4)=plt.subplots(4,1,figsize=(5,10))
    p=ax1.pcolormesh(f920[:,:,ind], cmap=plt.cm.BuPu_r)
    ax1.set_title('F920 IT='+str(ITf[ind]*1000.0)+' ms')
    aa=f920[:,:,ind]
    mini=np.min((aa))
    maxi=np.max((aa))  
    #print(mini,maxi)
    min_val=np.mean(aa)-3*np.std(aa)     
    max_val=np.mean(aa)+3*np.std(aa)
    #breakpoint()
 #   cbar = mpl.colorbar.ColorbarBase(ax1, cmap=cm ,norm=mpl.colors.Normalize(vmin=-min_val, vmax=max_val),orientation='horizontal')         
    f.colorbar(p,ax=ax1,orientation='horizontal',cmap=plt.cm.BuPu_r) #,boundaries=(min_val,max_val)

    #p=plt.title('F550')
    p=ax2.pcolormesh(f550[:,:,ind], cmap=plt.cm.BuPu_r)
    ax2.set_title('F550')
    aa=f550[:,:,ind]
    mini=np.nanmin((aa))
    maxi=np.nanmax((aa)) 
   # print(mini,maxi)        
    f.colorbar(p,ax=ax2,orientation='horizontal')#,boundaries=(mini,maxi),cmap=plt.cm.BuPu_r)

#    p=plt.title('F420')
    p=ax3.pcolormesh(f420[:,:,ind], cmap=plt.cm.BuPu_r)
    ax3.set_title('F420')
    aa=f420[:,:,ind]
    mini=np.nanmin((aa))
    maxi=np.nanmax((aa))  
    #print(mini,maxi)       
    f.colorbar(p,ax=ax3,orientation='horizontal') #,boundaries=(mini,maxi),cmap=plt.cm.BuPu_r)

#    p=plt.title('F750')
    p=ax4.pcolormesh(f750[:,:,ind], cmap=plt.cm.BuPu_r)
    ax4.set_title('F750')
    aa=f750[:,:,ind]
    mini=np.nanmin((aa))
    maxi=np.nanmax((aa))        
    f.colorbar(p,ax=ax4,orientation='horizontal') #,boundaries=(mini,maxi),cmap=plt.cm.BuPu_r)

    plt.savefig(path_out+'NON_calib_IT'+str(ITf[ind]*1000.0)+'ms_'+str(ind)+'.png')
    plt.clf()
   
    print(ind)
   # breakpoint()
#plt.subplot(111)    
    





    
#%%





ind=10
f,(ax1,ax2,ax3,ax4)=plt.subplots(4,1,figsize=(5,10))
p=plt.title('F920 IT='+str(ITf[ind]*1000.0)+' ms')
p=ax1.pcolormesh(cal920_2b[:,:,ind], cmap=plt.cm.BuPu_r)
aa=cal920_2b[:,:,ind]
mini=np.min((aa))
maxi=np.max((aa))        
f.colorbar(p,ax=ax1,orientation='horizontal',boundaries=(mini,maxi),cmap=plt.cm.BuPu_r)
plt.savefig(path_out+'calib_IT'+str(ITf[ind]*1000.0)+'ms_'+str(ind)+'.png')


#%%immagini COME LE VUOLE EMANUELE a cui è tolta l'ultima img di ciascuna serie di IT e la media della WX (anch'essa calibrata prendendp l'ultima img
#di ciascuna serie e poi fatta la media spaziale)
#ind=172
    
path_out='C:\\Alessandra\\STC_CAL\\DARK_IN_FLIGHT\\02-ICO_01\\04_-_STC_Performance_Test\\stc\\grafici_sottratto_WX\\'

for t in range(0,len(ITf)):
    if ITf[t] == 0:
        ITf[t] = 4.0e-5

    
for ind in range(0,len(ITf)):
    plt.subplot(411)
    plt.title('F920 IT='+str(ITf[ind])+' ms')
    plt.imshow(cal920_2b[:,:,ind], cmap=plt.cm.BuPu_r)
    plt.subplot(412)
    plt.title('F550')
    plt.imshow(cal550_2b[:,:,ind], cmap=plt.cm.BuPu_r)
    plt.subplot(413)
    plt.title('F420')
    plt.imshow(cal420_2b[:,:,ind], cmap=plt.cm.BuPu_r)
    plt.subplot(414)
    plt.title('F750')
    plt.imshow(cal750_2b[:,:,ind], cmap=plt.cm.BuPu_r)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax,orientation='vertical')    
   # plt.show()
    plt.savefig(path_out+'calib_IT'+str(ITf[ind])+'ms_'+str(ind)+'.png')
    plt.clf()
   
    print(ind)
   # breakpoint()
#plt.subplot(111)   

    
    
#%%
fig=plt.figure()
ax=fig.add_subplot(131)
cax =ax.imshow(cal920_2b[:,:,10])
cbar =fig.colorbar(cax,orientation='horizontal')
cbar.ax.minorticks_off()
ax1 =fig.add_subplots(132)
cax1 =ax1.imshow(cal550_2b[:,:,10])
cbar1 =fig.colorbar(cax1,orientation='horizontal')
cbar1.ax1.minorticks_off()

fig,ax,ax1 =plt.subplots(1)
#cax =ax.imshow(cal750_2b)
cax =ax.imshow(cal920_2b[:,:,10])
cbar =fig.colorbar(cax,orientation='horizontal')
cbar.ax.minorticks_off()

#cax =ax.imshow(cal750_2b)
cax1 =ax1.imshow(cal550_2b[:,:,10])
cbar1 =fig.colorbar(cax1,orientation='horizontal')
cbar1.ax1.minorticks_off()
fig,ax =plt.subplots(3)
#cax =ax.imshow(cal750_2b)
cax =ax.imshow(cal420_2b[:,:,10])
cbar =fig.colorbar(cax,orientation='horizontal')
cbar.ax.minorticks_off()
fig,ax =plt.subplots(4)
#cax =ax.imshow(cal750_2b)
cax =ax.imshow(cal750_2b[:,:,10])
cbar =fig.colorbar(cax,orientation='horizontal')
cbar.ax.minorticks_off()



        
    #breakpoint()
y420_1=f420[32,498,1:190]
y550_1=f550[32,498,1:190]
y750_1=f750[32,498,1:190]
y920_1=f920[32,498,1:190]
y420_2=f420[32,498,191:384]
y550_2=f550[32,498,191:384]
y750_2=f750[32,498,191:384]
y920_2=f920[32,498,191:384]

plt.figure()
ITfa=ITf[1:190]
ITfb=ITf[191:384]
plt.plot(ITfa,y420_1,'-',label='F420')
plt.plot(ITfa,y550_1,'-',label='F550')
plt.plot(ITfa,y750_1,'-',label='F750')
plt.plot(ITfa,y920_1,'-',label='F920')
plt.title('bassi RT')
plt.grid()
plt.xlabel('IT [S]')
plt.ylabel('central pixel F420')
plt.legend()
plt.show()        

plt.figure()
plt.plot(ITfb,y420_2,'-.',label='F420')
plt.plot(ITfb,y550_2,'-.',label='F550')
plt.plot(ITfb,y750_2,'-.',label='F750')
plt.plot(ITfb,y920_2,'-.',label='F920')
plt.title('alti RT')
plt.grid()
plt.xlabel('IT [S]')
plt.ylabel('central pixel F420')
plt.legend()
plt.show()        

#grafico dark in funzione del IT px centrale
plt.figure()
y420m=f420m[32,498,1:len(f420[0,0,:])-1]
y550m=f550m[32,498,1:len(f420[0,0,:])-1]
y750m=f750m[32,498,1:len(f420[0,0,:])-1]
y920m=f920m[32,498,1:len(f420[0,0,:])-1]
ITv=np.array(IT,dtype=float)*1000.0
ITv= np.around(ITv,decimals=2)
plt.plot(ITv,y420m,'-^',label='F420')
plt.plot(ITv,y550m,'-+',label='F550')
plt.plot(ITv,y750m,'-*',label='F750')
plt.plot(ITv,y920m,'-d',label='F920')
plt.grid()
plt.xlabel('IT [ms]')
plt.ylabel('central pixel F420')
ax=plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(5000))
plt.legend()
plt.show()      

#%%

#grafico dark in funzione del numero di immagine px centrale
plt.figure()
xval=np.arange(1,len(ITf)-1)
y420=f420[32,498,1:len(f420[0,0,:])-1]
y550=f550[32,498,1:len(f420[0,0,:])-1]
y750=f750[32,498,1:len(f420[0,0,:])-1]
y920=f920[32,498,1:len(f420[0,0,:])-1]
ITv=np.array(IT,dtype=float)*1000.0
ITv= np.around(ITv,decimals=2)
plt.plot(xval,y420,'-^',label='F420')
plt.plot(xval,y550,'-+',label='F550')
plt.plot(xval,y750,'-*',label='F750')
plt.plot(xval,y920,'-d',label='F920')
plt.grid()
plt.title('central px')
plt.xlabel('acquisitions')
plt.ylabel('[DN]')
ax=plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
plt.legend()
plt.show()      

#grafico dark in funzione del numero di immagine px 0,0 (coordinate immagine)
plt.figure()
xval=np.arange(1,len(ITf)-1)
y420=f420[63,0,1:len(f420[0,0,:])-1]
y550=f550[63,0,1:len(f420[0,0,:])-1]
y750=f750[63,0,1:len(f420[0,0,:])-1]
y920=f920[63,0,1:len(f420[0,0,:])-1]
ITv=np.array(IT,dtype=float)*1000.0
ITv= np.around(ITv,decimals=2)
plt.plot(xval,y420,'-^',label='F420')
plt.plot(xval,y550,'-+',label='F550')
plt.plot(xval,y750,'-*',label='F750')
plt.plot(xval,y920,'-d',label='F920')
plt.grid()
plt.title('0,0 px')
plt.xlabel('acquisitions')
plt.ylabel('[DN]')
ax=plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
plt.legend()
plt.show()   

#grafico dark in funzione del numero di immagine px 0,895 (coordinate immagine)
plt.figure()
xval=np.arange(1,len(ITf)-1)
y420=f420[63,895,1:len(f420[0,0,:])-1]
y550=f550[63,895,1:len(f420[0,0,:])-1]
y750=f750[63,895,1:len(f420[0,0,:])-1]
y920=f920[63,895,1:len(f420[0,0,:])-1]
ITv=np.array(IT,dtype=float)*1000.0
ITv= np.around(ITv,decimals=2)
plt.plot(xval,y420,'-^',label='F420')
plt.plot(xval,y550,'-+',label='F550')
plt.plot(xval,y750,'-*',label='F750')
plt.plot(xval,y920,'-d',label='F920')
plt.grid()
plt.title('0,895 px')
plt.xlabel('acquisitions')
plt.ylabel('[DN]')
ax=plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
plt.legend()
plt.show()   

#grafico dark in funzione del numero di immagine px 63,895 (coordinate immagine)
plt.figure()
xval=np.arange(1,len(ITf)-1)
y420=f420[0,895,1:len(f420[0,0,:])-1]
y550=f550[0,895,1:len(f420[0,0,:])-1]
y750=f750[0,895,1:len(f420[0,0,:])-1]
y920=f920[0,895,1:len(f420[0,0,:])-1]
ITv=np.array(IT,dtype=float)*1000.0
ITv= np.around(ITv,decimals=2)
plt.plot(xval,y420,'-^',label='F420')
plt.plot(xval,y550,'-+',label='F550')
plt.plot(xval,y750,'-*',label='F750')
plt.plot(xval,y920,'-d',label='F920')
plt.grid()
plt.title('63,895 px')
plt.xlabel('acquisitions')
plt.ylabel('[DN]')
ax=plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
plt.legend()
plt.show()   

#grafico dark in funzione del numero di immagine px 63,0 (coordinate immagine)
plt.figure()
xval=np.arange(1,len(ITf)-1)
y420=f420[0,0,1:len(f420[0,0,:])-1]
y550=f550[0,0,1:len(f420[0,0,:])-1]
y750=f750[0,0,1:len(f420[0,0,:])-1]
y920=f920[0,0,1:len(f420[0,0,:])-1]
ITv=np.array(IT,dtype=float)*1000.0
ITv= np.around(ITv,decimals=2)
plt.plot(xval,y420,'-^',label='F420')
plt.plot(xval,y550,'-+',label='F550')
plt.plot(xval,y750,'-*',label='F750')
plt.plot(xval,y920,'-d',label='F920')
plt.grid()
plt.title('63,0 px')
plt.xlabel('acquisitions')
plt.ylabel('[DN]')
ax=plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
plt.legend()
plt.show() 

#%%
plt.figure()
y420m_1=f420m[32,498,1:19]
y550m_1=f550m[32,498,1:19]
y750m_1=f750m[32,498,1:19]
y920m_1=f920m[32,498,1:19]
y420m_2=f420m[32,498,21:40]
y550m_2=f550m[32,498,21:40]
y750m_2=f750m[32,498,21:40]
y920m_2=f920m[32,498,21:40]
RT=np.array(RT,dtype=float)
RT= np.around(RT,decimals=2)
RTa=RT[1:19]
RTb=RT[21:40]
plt.plot(RTa,y420m_1,'-^',label='F420')
plt.plot(RTa,y550m_1,'-+',label='F550')
plt.plot(RTa,y750m_1,'-*',label='F750')
plt.plot(RTa,y920m_1,'-d',label='F920')
plt.plot(RTb,y420m_2,'-^',label='F420')
plt.plot(RTb,y550m_2,'-+',label='F550')
plt.plot(RTb,y750m_2,'-*',label='F750')
plt.plot(RTb,y920m_2,'-d',label='F920')
plt.grid()
plt.xlabel('RT [S]')
plt.ylabel('central pixel F420')
ax=plt.axes()
#ax.xaxis.set_major_locator(ticker.MultipleLocator(5000))
plt.legend()
plt.show()      










































