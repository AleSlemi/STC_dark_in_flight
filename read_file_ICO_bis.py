# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:19:39 2020

@author: slemera
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 17:34:29 2020

@author: slemera
"""

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
directories=glob.glob('C:\\Users\\slemera\\Alessandra\\STC_CAL\\ICO1\\stc\\science*')
#dire = os.getcwd()+'\\'+tele_name+'\\'


dati_tot=np.zeros((1,19))
#dati_tot=np.transpose(dati_tot)

for g in range(42,len(directories)):
    
    tele_name=[]
    str_split=directories[g].split('\\')
    telecomando=str_split[7]
    dire = directories[g]
    #tele_name.append(telecomando)
    
    filtro ='filterx'
    ITfx,RTfx,TFPA1,TFPA2,TCh1,TCh2,TPE,start_obs_fx,start_obs_et_fx,media_fx,media64_fx,DSNU_fx,filter_fx  = dati_utili(dire,filtro)
  #  breakpoint()
    filtro ='panl'
    ITpl,RTpl,TFPA1,TFPA2,TCh1,TCh2,TPE,start_obs_pl,start_obs_et_pl,media_pl,media64_pl,DSNU_pl,filter_pl  = dati_utili(dire,filtro)
   # breakpoint()
    filtro ='panh'
    ITph,RTph,TFPA1,TFPA2,TCh1,TCh2,TPE,start_obs_ph,start_obs_et_ph,media_ph,media64_ph,DSNU_ph,filter_ph  = dati_utili(dire,filtro)

    dati_fx =np.array(len(ITfx))
    for gg in range(0,len(ITfx)):
        tele_name.append(telecomando)
    
    dati = np.vstack((tele_name,start_obs_fx,start_obs_et_fx,ITfx,RTfx,TFPA1,TFPA2,TCh1,TCh2,TPE,media_fx,media64_fx,DSNU_fx,media_pl,media64_pl,DSNU_pl,media_ph,media64_ph,DSNU_ph))
    #breakpoint()
    dati_tras = np.transpose(dati)
    dati_tot = np.vstack((dati_tot,dati_tras))
    print('telecomando=',telecomando)
    

#dati_t= np.transpose(dati_tot)
breakpoint()
df =pd.DataFrame(dati_tot,columns = ['telecomando','start_obs','start_obs_et_[s]','IT_[s]','RT_[s]','TFPA1_[K]','TFPA2_[K]','TCh1_[K]','TCh2_[K]','TPE_[K]','media_FX_[DN]','media64_FX_[DN]','DSNU_FX','media_PANL_[DN]','media64_PANL_[DN]','DSNU_PANL','media_PANH_[DN]','media64_PANH_[DN]','DSNU_PANH'])

df.to_excel('data_'+'.xls','w+b')
#f='imgs_'+tele_name+'_'+filtro+'.txt'
#f.write(np.array2string(img,separator=','))


#%%
def dati_utili(directory,nome_filtro):
    
    filexml=glob.glob(directory+'\\sim_img_*'+nome_filtro+'_*.xml')
    filedat=glob.glob(directory+'\\sim_img_*'+nome_filtro+'*.dat')
    
    breakpoint()
    
    start_obs= []
    start_obs_et= []
    IT = []
    media64 =[]
    media=[]
    DSNU = []
    TFPA1 = []
    TFPA2 = []
    TCh1 = []
    TCh2 = []
    TPE = []
    
    name_filter = []
    RT = np.zeros(len(filexml))
    
    
    for k in range(0,len(filexml)-1):
          #print('iterazione=',k)
    
        tree = xml.parse(filexml[k])
        root = tree.root
       # breakpoint()
        stringhe = filexml[k].split('_')
        nome_filtro = stringhe[8]
        name_filter.append(nome_filtro)
        ti_name=stringhe[10].split('.')
        t_name=ti_name[0]
        #breakpoint()
        
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
                RT[k] =float(start_obs_et[k])-float(start_obs_et[k-1])
                if RT[k] <0:
                    if name_filter[k] != name_filter[k-1]:
                        RT[k] = 0
                    #breakpoint()
                
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
            
            dim_wind_x = (dim_wind[1]-dim_wind[0]) #diff colonne
            dim_wind_y = (dim_wind[3]-dim_wind[2]+1) #diff righe
            dim_img=np.multiply(dim_wind[0],dim_wind[1])
           # breakpoint()
            if k == 0:
                img =np.zeros((dim_wind_y,dim_wind_x))
            #breakpoint()
            img_file=glob.glob(dire+'\\*'+nome_filtro+'*'+t_name+'.dat')
            #breakpoint()
            #prova = open(img_file[0],'rb')
            #imag = prova.read(dim_img[0])
            mat_filter=np.fromfile(img_file[0], dtype ='int16')
            
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
    RT =np.array(RT[0:len(RT)-1],dtype=float)
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
   # breakpoint()
    return IT,RT,TFPA1,TFPA2,TCh1,TCh2,TPE,start_obs,start_obs_et,media,media64,DSNU,name_filter#,mat_filter   






            
#%%            

#def save_dati()


       

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    