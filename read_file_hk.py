# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:34:57 2020

@author: Teo
"""


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
from xml.dom import minidom
#from xml import objectify
#import pdb; pdb.set_trace()
#import datatable as dt

#%%
#carco il metakernel di bepicolombo
spiceypy.furnsh('H:\\modello_radiometrico_Mercurio\\Bepi_spice\\spice190729\\kernels\\bc_preops_v041_MTC_mar17.mk')
a = np.arange(43,84)
cbd=64
breakpoint()

for j in range(0,len(a)):
  #  breakpoint()
    nome_telecomando='science_00'+str(a[j])
    print('a=',a[j],'cartella=',nome_telecomando)
    print(nome_telecomando)
    #breakpoint()
    #time='20190607T090300014480Z'  #time nel nome del file
    #time1='2019-06-07T09:03:00.014480' #time da aggiustare per trovare l'et corrispondente
   # et = spiceypy.utc2et(time1)
    os.chdir('C:\\Alessandra\\STC_CAL\\DARK_IN_FLIGHT\\02-ICO_01\\04_-_STC_Performance_Test\\stc')
    file=glob.glob('C:\\Alessandra\\STC_CAL\DARK_IN_FLIGHT\\02-ICO_01\\04_-_STC_Performance_Test\\stc\\'+nome_telecomando+'\\*hk*.xml')
    
    # breakpoint()
    start_time=[]
    start_time_et=np.zeros(1)
    stop_time_et=np.zeros(1)
    rt=np.zeros(1)
    stop_time=[]
    ITime=[]
    filt=[]
    TFPA1=[]
    TFPA2=[]   
    col_s=[]
    row_s=[]
    col_e=[]
    row_e=[]
    mat_WX=np.zeros((64,128))
    mat_cfilt=np.zeros((64,896))
   # window=np.array()
         
    for aaa in range(0,len(file)):  
        mytree=ET.parse(file[aaa])
        myroot=mytree.getroot()        
        stringhe = file[aaa].split('_')
        nome_filtro = stringhe[15]
        ti_name=stringhe[17].split('.')
        t_name=ti_name[0]
        
        
        filt.append(nome_filtro)
        if nome_filtro != 'panh' and nome_filtro != 'panl':        
            ns = {'xsi':'http://www.w3.org/2001/XMLSchema-instance',
                 'geom':'http://pds.nasa.gov/pds4/geom/v1', 
                 'img':'http://pds.nasa.gov/pds4/img/v1', 
                 'psa':'http://psa.esa.int/psa/v1',
                 'simbio':'http://psa.esa.int/psa/bc/simbio/v1',
                 'pds4':'http://pds.nasa.gov/pds4/pds/v1'}           
            for elem in mytree.iter():    
                    #breakpoint()
                for kk in elem.findall('img:Imaging_Instrument_Parameters', ns):
                    for hh in kk.findall('img:Filter',ns):
        #                img_seq = hh.findall('img:sequence',ns)
                        for jj in hh.findall('img:sequence',ns):
                            name_filter = jj.find('img:filter_id',ns)                    
                            #print('nome_filtro_interno =',name_filter.text)
                    #if name_filter.text != 'P700':
                    for yy in kk.findall('img:Device_Temperature',ns):
                        name_TFPA = yy.find('img:name',ns)
                        TFPA = yy.find('img:temperature_calibrated',ns)
                       # print(name_TFPA.text,'=',TFPA.text)
                        if name_TFPA.text == 'TemperatureFPA1':
                            TFPA1.append(TFPA.text)
                        if name_TFPA.text == 'TemperatureFPA2':
                            TFPA2.append(TFPA.text)
                        #breakpoint()    
                if elem.tag == 'Time_Coordinates':  # Questo trova un elemento specifico e ritorna il valore di un suo figlio
                    #print(elem.tag)
                    for ii in elem.getchildren():
                      #  print(ii.tag,'=',ii.text)
                        if ii.tag == 'start_date_time':
                            start_time.append(ii.text)
                            et =ii.text.split('Z')
                            time_eti = spiceypy.utc2et(et[0])                           
                            start_time_et=np.hstack((start_time_et,time_eti))
                          #  breakpoint()
                        if ii.tag == 'stop_date_time':
                            stop_time.append(ii.text)
                            et =ii.text.split('Z')
                            time_etf = spiceypy.utc2et(et[0])
                            stop_time_et=np.hstack((stop_time_et,time_etf))
                if elem.tag == 'Mission_Area':    
                    for jj in elem.findall('simbio:STC',ns):
                        IT = jj.find('simbio:Integration_Time',ns)
                       # print('IT=',IT.text)
                        ITime.append(IT.text)                        
                        for hh in jj.findall('simbio:Window',ns):
                            #wind = hh.find('simbio:Pixel',ns)
                            for oo in hh.findall('simbio:Pixels',ns):
                               # breakpoint()
                                # for zz in hh.getchildren():
                                #     print(zz.tag,'=',zz.text)
                                #     breakpoint()
                                for tt in oo.findall('simbio:start_pixel',ns):
                                    start_col = tt.find('simbio:column',ns)  
                                    #colonnas = np.array(colonna, dtype=np.float64)
                                    col_s.append(start_col.text)
                                    start_row = tt.find('simbio:row',ns)
                                    row_s.append(start_row.text)
                                  # print('col_start=',start_col.text)
                                   # print('raw_start=',start_row.text)                                   
                                    #breakpoint()
                                for uu in oo.findall('simbio:stop_pixel',ns):
                                    stop_col = uu.find('simbio:column',ns) 
                                    #colonnae = (stop_col.text+1)*cbd-1
                                    col_e.append(stop_col.text) 
                                    stop_row = uu.find('simbio:row',ns)
                                    row_e.append(stop_row.text)
                                   # print('col_stop=',stop_col.text)
                                   # print('raw_stop=',stop_row.text)  
        
            if filt[aaa] == 'filterx':
                Reptime= start_time_et[aaa]-start_time_et[aaa-1]
                rt=np.hstack((rt,Reptime))
            RT=np.mean(rt)
               # breakpoint()
            #window = np.hstack(np.transpose(np.array(row_s)),np.transpose(np.array(row_e)),np.transpose(np.array(col_s)),np.transpose(np.array(col_e))) 
            dim_wind=np.zeros(4)
            dim_wind = [np.array(start_col.text,dtype=int),np.array(stop_col.text,dtype=int),np.array(start_row.text,dtype=int),np.array(stop_row.text,dtype=int)]
            dim_wind[0]=dim_wind[0]*cbd
            dim_wind[1]=(dim_wind[1]+1)*cbd-1
            dim_wind_x = (dim_wind[1]-dim_wind[0]+1) #diff colonne
            dim_wind_y = (dim_wind[3]-dim_wind[2]+1) #diff righe
            dim_img=np.multiply(dim_wind[0],dim_wind[1])
            
            img_file=glob.glob('C:\\Alessandra\\STC_CAL\DARK_IN_FLIGHT\\02-ICO_01\\04_-_STC_Performance_Test\\stc\\'+nome_telecomando+'\\*'+nome_filtro+'*'+t_name+'*.dat')
            #prova = open(img_file[0],'rb')
            #imag = prova.read(dim_img[0])
            mat_filter =np.fromfile(img_file[0], dtype ='int16')
            mat_filter=mat_filter.reshape([dim_wind_y,dim_wind_x])
            
           # breakpoint()
            if nome_filtro == 'filterx':
                mat_WX=np.dstack((mat_WX,mat_filter)) 
            else:
                if nome_filtro != 'panh' and nome_filtro !='panl':
                    mat_cfilt=np.dstack((mat_cfilt,mat_filter))  
    
 #   RT=np.mean(ITime[])
    np.save('prova_filtro',mat_cfilt)
    np.save('prova_wx',mat_WX)       
   # breakpoint()
    
        #fig = plt.figure
        # plt.figure(figsize = (dim_wind_x[0],dim_wind_y[0]))
        # color_map = plt.imshow(imag)
        # color_map.set_cmap("Blues")
        # plt.colorbar()
        # breakpoint()        
        #display(Image(filename=img_file))
        #plt.imshow(mpimg.imread(img_file))
        # breakpoint()
        # I = np.fromfile(img_file[0],dtype='int16')
        # I = I.reshape([dim_wind_x[0],dim_wind_y[0]])
        # plt.imshow(I)
        # breakpoint()
        
        
        
    window = np.zeros((len(col_e),4))
    #window = np.vstack((np.array(row_s),np.array(row_e),np.array(col_s),np.array(col_e))) 
    window[:,0]=np.array(col_s,dtype=float)
    window[:,1]=np.array(col_e,dtype=float)
    window[:,2]=np.array(row_s,dtype=float)
    window[:,3]=np.array(row_e,dtype=float)
    #window = np.transpose(window)
    #window = np.array(window)
    window[:,0]=window[:,0]*cbd
    window[:,1]=(window[:,1]+1)*cbd-1
    #col_e1=(np.array(col_e)+1)*cbd-1
    #col_s1=np.array(col_s)*cbd
    # dim_wind_x = (col_e-col_s+1)
    # dim_wind_y = (np.array(row_e)-np.array(row_s)+1)
    # prova = open(img_file[0],'rb')
    # prova.read(dim_wind_x*dim_wind_y)
    dict = {'img_cfilt':mat_cfilt,'img_WX':mat_WX,'window':window,'namef':filt,'T1':TFPA1,'T2':TFPA1,'IT':ITime,'RT':RT,'acq':start_time}
    #path='C:\\Alessandra\\STC_CAL\\DARK_IN_FLIGHT\\02-ICO_01\\04_-_STC_Performance_Test\\stc\\'
    nome_pkl='dati_science_00'+str(a[j])
    f=open(nome_pkl,'wb')
    pickle.dump(dict,f)
    f.close()
   # breakpoint()                                   #  breakpoint()
spiceypy.spiceypy.kclear()                        #print(ITime)
                        #breakpoint()
                       
                    
#suggerimenti di Ciprian
# if elem == 'Product_Observational':
#     print(elem.tag)
#     breakpoint()
#     for kk in elem.getchildren():
#         print(kk.tag,'=',kk.text)
# Questo trova un elemento specifico e ritorna il suo valore 
#{http://psa.esa.int/psa/bc/simbio/v1} è l'insieme di appartenenza del valore (in questo caso simbio)
#Integration_time è l'element da cercare (tag è il tome e text il valore)    
# if elem.tag == '{http://psa.esa.int/psa/bc/simbio/v1}Integration_Time':
#     print(elem.tag)
#     print(elem.text)
#breakpoint()
    





                #breakpoint()
#print('stampa res =')
# for child in myroot[0].iter('rank'):   
#     print(child.attrib)
#breakpoint()
#for x in myroot[2]:
#    print(x.tag,x.attrib)
# lista=[]
# for i in (0,len(file)-1):
#     string = file[i]
#     t = string.rfind(time)
#     if t > 0:
#         lista.append(string)
# breakpoint()

