# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 17:34:29 2020

@author: slemera
"""
#prova2
#prova fisso
import os
import glob
import sys

import pandas as pd
import csv
import io
import pickle

import math
import array
import numpy as np
import numpy.linalg
from matplotlib import pylab as pt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits import mplot3d
from IPython.display import display, Image

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astropy.io.votable import parse
from astropy.io import fits
from astropy.table import Table, Column

import spiceypy

import xml.etree.ElementTree as ET
import xml.etree.ElementInclude as elt
#from xml.dom import minidom
#from xml import objectify
import xml4h as xml
import iso8601
#import pdb; pdb.set_trace()
#import datatable as dt

#%%

META_KERNEL = 'C:\\Users\\slemera\\Alessandra\\modello_radiometrico_Mercurio\\Bepi_spice\\spice180316\\bc_preops_v041_MTC_mar18.mk' # 'solo_ANC_soc-flown-mk.tm'
KERNEL_PATH = 'C:\\Users\\slemera\\Alessandra\\modello_radiometrico_Mercurio\\Bepi_spice\\spice180316\\'  # '../kernels/solar-orbiter/kernels/mk'
SPICE_LOADED = True

def chdir(path):
    CWD = os.getcwd()
    os.chdir(path)
    try:
        yield
    except:
        print('Exception caught: ',sys.exc_info()[0])
    finally:
        os.chdir(CWD)


def spice_init():
    
    META_KERNEL = 'C:\\Users\\slemera\\Alessandra\\modello_radiometrico_Mercurio\\Bepi_spice\\spice180316\\bc_preops_v041_MTC_mar18.mk' # 'solo_ANC_soc-flown-mk.tm'
    spiceypy.furnsh(META_KERNEL)
    # global SPICE_LOADED
    # if not SPICE_LOADED:
    #     with chdir(KERNEL_PATH):
    #         spiceypy.furnsh(META_KERNEL)
    #     SPICE_LOADED = True
        
def spice_clear():
    global SPICE_LOADED
    spiceypy.kclear()
    SPICE_LOADED = False


#%%
CBD = 64
#tele_name ='science_0001'
#filtro = 'filterx'  #'panl' #'panh'
spice_init()
os.chdir('C:\\Users\\slemera\\Alessandra\\STC_CAL\\ICO1\\stc\\')
direct= 'C:\\Users\\slemera\\Alessandra\\STC_CAL\\ICO1\\stc\\' 
direct='C:\\Users\\slemera\\Alessandra\\STC_CAL\\01-dNECP\\02_-_STC_Mitigate_Reset_Test\\stc\\'
# direct='C:\\Users\\slemera\\Alessandra\\STC_CAL\\01-dNECP\\01_-_STC_All_FPA_Test\\stc\\'
# direct='C:\\Users\\slemera\\Alessandra\\STC_CAL\\01-dNECP\\03_pixel_caldo_test\\stc\\'
     #   'C:\Users\slemera\Alessandra\STC_CAL\01-dNECP\03_-_STC_Hot_Pixel_Test\stc'
#direct='C:\\Users\\slemera\\Alessandra\\STC_CAL\\01-dNECP\\06_-_Orbit_Test\\stc\\'


directories=glob.glob(direct+'science_*')
#directories=glob.glob('C:\\Users\\slemera\\Alessandra\\STC_CAL\\ICO1\\stc\\science*')
#breakpoint()
#dire = os.getcwd()+'\\'+tele_name+'\\'


dati_tot=np.zeros((1,27))
nomi_tot=np.zeros((1,9))
#dati_tot=np.transpose(dati_tot)
for g in range(0,len(directories)-1):
    
    tele_name=[]
    str_split=directories[g].split('\\')
    telecomando=str_split[8] #7
    dire = directories[g]
    #tele_name.append(telecomando)
    print('direct=',directories[g])
    
    win ='xxx_0'
    filtro0,out0,last0,gen0,start_obs0 = dati_utili(dire,win)
  #  breakpoint()
    win ='xxx_1'
    filtro1,out1,last1,gen1,start_obs1  = dati_utili(dire,win)
    #breakpoint()
    if len(out1[0,:]) ==0:
        out1=np.zeros(out0.shape)
        filtro1= np.zeros(filtro0.shape)
   # brew2akpoint()
    win ='xxx_2'
    filtro2,out2,last2,gen2,start_obs2  = dati_utili(dire,win)
    if len(out2[0,:]) ==0:
        out2=np.zeros(out0.shape)
        filtro2= np.zeros(filtro0.shape)
    win ='xxx_3'
    filtro3,out3,last3,gen3,start_obs3  = dati_utili(dire,win)
    if len(out3[0,:]) ==0:
        out3=np.zeros(out0.shape)
        filtro3= np.zeros(filtro0.shape)
    win ='xxx_4'
    filtro4,out4,last4,gen4,start_obs4  = dati_utili(dire,win)
    if len(out4[0,:]) ==0:
        out4=np.zeros(out0.shape)
        filtro4= np.zeros(filtro0.shape)
    win ='xxx_5'
    filtro5,out5,last5,gen5,start_obs5  = dati_utili(dire,win)
    if len(out5[0,:]) ==0:
        out5=np.zeros(out0.shape)
        filtro5= np.zeros(filtro0.shape)

   # breakpoint()
    
    dati_fx =np.array(len(out0[:,0]))
    for gg in range(0,len(out0[0,:])):
        tele_name.append(telecomando)
    tname=np.transpose(np.array(tele_name))
    print('telecomando=',telecomando)
   # breakpoint()      
    
    # tgen=np.transpose(gen0)
    # tout0=np.transpose(out0)
    # tout1=np.transpose(out1)
    # tout2=np.transpose(out2)
    # tout3=np.transpose(out3)
    # tout4=np.transpose(out4)
    # tout5=np.transpose(out5)
    #breakpoint()
    dati = np.vstack((gen0,out0,out1,out2,out3,out4,out5))
    nomi = np.vstack((tname,last0,start_obs0,filtro0,filtro1,filtro2,filtro3,filtro4,filtro5))
      
  #  breakpoint()
    dati_tras = np.transpose(dati)
    dati_tot = np.vstack((dati_tot,dati_tras))
    nomi_tras=np.transpose(nomi)
    nomi_tot = np.vstack((nomi_tot,nomi_tras))
  #  breakpoint() 
#dati_t= np.transpose(dati_tot)

for u in range(2,len(dati_tot[:,0])):
    dati_tot[u,3] = float(dati_tot[u,0])-float(dati_tot[u-1,0])
    
colonne2 =  ['start_obs_et_[s]','IT_[s]','RT_mean_[s]','WT_[s]','TFPA1_[K]','TFPA2_[K]','TCh1_[K]','TCh2_[K]','TPE_[K]','mean_W1_[DN]','mean64_W1_[DN]','DSNU_W1','mean_W2_[DN]','mean64_W2_[DN]','DSNU_W2','mean_W3_[DN]','mean64_W3_[DN]','DSNU_W3','mean_W4_[DN]','mean64_W4_[DN]','DSNU_W4','mean_W5_[DN]','mean64_W5_[DN]','DSNU_W5','mean_W6_[DN]','mean64_W6_[DN]','DSNU_W6']    
colonne1 =  ['TC','last_image','start_obs','name_W1','name_W2','name_W3','name_W4','name_W5','name_W6']
#breakpoint()
#dati_tot[0,11] =math.nan
#dati_tot[1,11] =math.nan
#df =pd.DataFrame(nomi_tot,dati_tot,columns = ['telecomando','last_image','start_obs','name_W1','name_W2','name_W3','name_W4','name_W5','name_W6','start_obs_et_[s]','IT_[s]','RT_mean_[s]','WT_[s]','TFPA1_[K]','TFPA2_[K]','TCh1_[K]','TCh2_[K]','TPE_[K]','mean_W1_[DN]','mean64_W1_[DN]','DSNU_W1','mean_W2_[DN]','mean64_W2_[DN]','DSNU_W2','mean_W3_[DN]','mean64_W3_[DN]','DSNU_W3','mean_W4_[DN]','mean64_W4_[DN]','DSNU_W4','mean_W5_[DN]','mean64_W5_[DN]','DSNU_W5','mean_W6_[DN]','mean64_W6_[DN]','DSNU_W6'])

df1 =pd.DataFrame(nomi_tot,columns = colonne1)
df2 =pd.DataFrame(dati_tot,columns = colonne2)
df = df1.join(df2)
breakpoint()
os.chdir(direct)
df.to_excel('data_'+'STC_ICO1.xls')
#f='imgs_'+tele_name+'_'+filtro+'.txt'
#f.write(np.array2string(img,separator=','))


#%%
def dati_utili(directory,window):
    
    filexml=glob.glob(directory+'\\sim_img_*'+window+'_*.xml')
    filedat=glob.glob(directory+'\\sim_img_*'+window+'*.dat')
    
    #breakpoint()
    
    start_obs = []
    start_obs_et = []
    IT = []
    media64 = []
    media= []
    DSNU = []
    TFPA1 = []
    TFPA2 = []
    TCh1 = []
    TCh2 = []
    TPE = []
    last_img = []
    name_filter = []
    RT1 = np.zeros(len(filexml))
    
    
    for k in range(0,len(filexml)):
          #print('iterazione=',k)
    
        tree = xml.parse(filexml[k])
        root = tree.root
      #  breakpoint()
        stringhe = filexml[k].split('_')
        #breakpoint()
        nome_filtro = stringhe[13]   #13  #11   #8
        name_filter.append(nome_filtro)
        ti_name=stringhe[15].split('.') #15  #13 #10
        t_name=ti_name[0]
       # breakpoint()

        last = 'FALSE'
        if k == len(filexml)-1:
            last ='TRUE'
        last_img.append(last)        
        # for param in root.find('Obervation_Area'):
        #     nome_sequenza=param.attributes['name']
        #     timei = param.child('Time_Coordinates').child('Start_date_time').text
        #     breakpoint()            
        # for param in root.find('Product_Observational'):
        #     nome_area = param.child('Observation_Area')
        #     #nome_sequenza=param.attributes['name']
        #     timei = param.child('Time_Coordinates').child('Start_date_time').text
        #     breakpoint()        
        for i in root.find('Observation_Area'):
            chld=i.children()
            start_data = chld[0].child('start_date_time').text
            data1 =str.split(start_data,'Z')
            start_obs.append(data1[0])
            start_obs_et.append(spiceypy.utc2et(data1[0]))
            if k >= 1:
               # breakpoint()
                RT1[k] =float(start_obs_et[k])-float(start_obs_et[k-1])
                if RT1[k] <0:
                    if name_filter[k] != name_filter[k-1]:
                        RT1[k] = 0
                    #breakpoint()
            else:
                RT1[k] = 0.0
               # breakpoint()
            if len(filexml) < 2:
                RT1[k] = 0.0
                
            for u in range(0,len(chld)):
                #print('u=', u)
                #print(chld[u])
                if str(chld[u]) == '<xml4h.nodes.Element: "Mission_Area">':
                    chld2 = chld[u].children()
                    # print(chld2)                
                    # print(chld[u])
                    # breakpoint()
                    break
            
            #chld2 = chld[u].children()
            #simbioSTC = chld2[14].children()
            for uu in range(0,len(chld2)):
                #print('uu=', uu)
                if str(chld2[uu]) == '<xml4h.nodes.Element: "simbio:STC">':
                    simbioSTC = chld2[uu].children()
                    # print(chld2[uu])               
                    # print(chld2[uu])
                    break
    
            #breakpoint()
            tempo = simbioSTC[0].text
            if float(tempo) == 0:
                tempo = 400.0e-9
            IT.append(tempo)
            #print(simbioSTC[0].name, '=',simbioSTC[0].text)
            #breakpoint()
            wn = simbioSTC[3].children()
            start_row =wn[1].children()[0].children()[0].text
            start_col = wn[1].children()[0].children()[1].text
            start_col =float(start_col)*CBD
            stop_row =wn[1].children()[1].children()[0].text
            stop_col = wn[1].children()[1].children()[1].text
            stop_col = (float(stop_col)+1)*CBD-1
            dim_wind=np.zeros(4)
            dim_wind = [np.array(start_col,dtype=int),np.array(stop_col,dtype=int)+1,np.array(start_row,dtype=int),np.array(stop_row,dtype=int)]
            # breakpoint()
            dim_wind_x = (dim_wind[1]-dim_wind[0]) #diff colonne
            dim_wind_y = (dim_wind[3]-dim_wind[2]+1) #diff righe
            #dim_img=np.multiply(dim_wind[0],dim_wind[1])
           # breakpoint()
            if k == 0:
                img =np.zeros((dim_wind_y,dim_wind_x))
            #breakpoint()
            img_file=glob.glob(dire+'\\*'+window+'*'+t_name+'.dat')
            #breakpoint()
            #prova = open(img_file[0],'rb')
            #imag = prova.read(dim_img[0])
        #    breakpoint()
            mat_filter=np.fromfile(img_file[0], dtype ='int16')
           # breakpoint()
            mat_filter=mat_filter.reshape([dim_wind_y,dim_wind_x])
            #img = np.dstack((img,mat_filter))
            media.append(np.mean(mat_filter)) #media in DN del filtro intero
            numcol=len(mat_filter[0,:])
            media64.append(np.mean(mat_filter[:,numcol-65:numcol-1])) #media delle ultime 64 colonne del filtro
            DSNU.append(np.std(mat_filter))
           
    
            for uu in range (0,len(chld2)-1):
                if chld2[uu] == '<xml4h.nodes.Element: "simbio:STC">':
                    simbioSTC = chld2[uu+1].children()
                    print(chld2[uu])
                    print(chld2[uu+1])
                    break
                
            TFPA1.append(chld2[15].children()[0].children()[1].text)
            TFPA2.append(chld2[15].children()[1].children()[1].text)
            TPE.append(chld2[15].children()[2].children()[1].text)
            TCh1.append(chld2[15].children()[3].children()[1].text)
            TCh2.append(chld2[15].children()[4].children()[1].text)


    name_filter= np.array(name_filter)
    IT =np.array(IT,dtype=float)
    RT =np.array(RT1[0:len(RT1)],dtype=float) #REPETITION TIME MEDIO
    #breakpoint()
    WT = RT1   #WAITING TIME
    TFPA1 =np.array(TFPA1,dtype=float)
    TFPA2 =np.array(TFPA2,dtype=float)
    TCh1 =np.array(TCh1,dtype=float)
    TCh2 =np.array(TCh2,dtype=float)
    TPE =np.array(TPE,dtype=float)
    start_obs_et=np.array(start_obs_et,dtype=float)
    start_obs=np.array(start_obs)
    media=np.array(media,dtype=float)
    media64=np.array(media64,dtype=float)
    DSNU=np.array(DSNU,dtype=float)            
    
    last_img=np.array(last_img)
 #   breakpoint()
    RT_mean=np.zeros(RT.shape)
    # breakpoint()
    for h in range(0,len(RT)):
        RT_mean[h]=np.nanmean(RT[1:len(RT)])
    if len(filexml) == 1:
        RT_mean = np.zeros(1)
        #breakpoint()
        
    # print('RT_MEAN=',RT_mean)
    # print('RT=',RT)
    # print('RT1=',RT1)
    # if len(RT) < 3:
    #     if len(RT) < 2:
    #         RT_mean = RT
    #     else:
    #         RT_mean = RT[1]
    #print(RT)
   # breakpoint()
    out_window=np.vstack((media,media64,DSNU))
    out_gen=np.vstack((start_obs_et,IT,RT_mean,WT,TFPA1,TFPA2,TCh1,TCh2,TPE))
    
    # if window == 'xxx_0':        
    #     breakpoint()
    #return IT,RT,TFPA1,TFPA2,TCh1,TCh2,TPE,start_obs,start_obs_et,media,media64,DSNU,name_filter#,mat_filter   
    return name_filter,out_window,last_img,out_gen,start_obs
    





            
#%%            

#def save_dati()


       

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    