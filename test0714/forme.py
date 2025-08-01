# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 12:53:12 2025

@author: nv1r09
"""

#TEST Beam Quality for 
from LightPipes import *
import cv2 as cv
import numpy as np
from numpy import genfromtxt
import os  # 添加这行
from matplotlib import pyplot as plt

# from progress.bar import Bar
import time


import pandas as pd

    
def ModeOverlap(Modes,Fout):
    
    Overlap=[]
    #note can we save this mornalization?
    P2=np.abs(np.sum(Fout.field*np.conj(Fout.field)))
    
    for i in range(np.shape(Modes)[0]):
        P1=np.abs(np.sum(Modes[i].field*np.conj(Modes[i].field)))
        Overlap.append(np.power(np.abs(np.sum(Modes[i].field*np.conj(Fout.field))),2)/P1/P2)
    
    return np.asarray(Overlap)
#    return np.power(np.abs(Cn(Modes,Fout)),2)
  

def Cn(Modes,Fout):
    
    Overlap=[]
    
#    P2=np.abs(np.sum(Fout.field*np.conj(Fout.field)))
    P2=np.sum(np.power(np.abs(Fout.field),2))
    for i in range(np.shape(Modes)[0]):
#        P1=np.abs(np.sum(Modes[i].field*np.conj(Modes[i].field)))
        P1=np.sum(np.power(np.abs(Modes[i].field),2))
        Overlap.append(np.sum(np.conj(Modes[i].field)*Fout.field)/P1/P2)
    
    return np.asarray(Overlap)

def test_function(BW,mode_data,wavelength,F0F,grid_size,grid_dimension):
    Modes=[]
    F_blank=Begin(BW,wavelength,np.shape(mode_data)[1])
    for i in range(np.shape(mode_data)[0]):
        F_blank.field=np.squeeze(mode_data[i,:,:])
        Modes.append(Interpol(grid_size,grid_dimension, 0, 0, 0, 1, F_blank))
    return np.abs(1-np.sum(ModeOverlap(Modes,F0F)))


def RotateComplexField(field,angle_rotation):
    out=rotate(np.real(field),angle_rotation, reshape=False)+rotate(np.imag(field),angle_rotation, reshape=False)*1j

    return out

def RotateF(F,angle_rotation):
    F.field=RotateComplexField(F.field,angle_rotation)
    return F


def PrintCompPowerLP(array_names,array_overlap):
   for i in range(len(array_names)):
       print(array_names[i]+' :  {:.2f}'.format(array_overlap[i]))



def PrintCompPowerLP1(array_names,array_overlap):
#   for i in range(len(array_names)):
   i=0
   while i< len(array_names):
       temp=0
       if len(array_names[i])==4:
           print(array_names[i]+' :  {:.2f}'.format(array_overlap[i]))
       else:
           for j in range(len(array_names)):
               if array_names[i][2:-1]==array_names[j][2:-1]:
                   temp+=array_overlap[j]
        
           print(array_names[i][:-1]+' :  {:.2f}'.format(temp))
           i=i+1
       i+=1  

from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator

def ResizeMode(mode_data,n_new,m_new):
    n_max=np.shape(mode_data)[0]
    m_max=np.shape(mode_data)[1]
   
    # 使用 RegularGridInterpolator 替代 interp2d
    x=np.arange(0,n_max,1)
    y=np.arange(0,m_max,1)
    
    # 创建新的网格
    xnew=np.arange(0,n_max,n_max/n_new)
    ynew=np.arange(0,m_max,m_max/m_new)
    
    # 使用 RegularGridInterpolator
    from scipy.interpolate import RegularGridInterpolator
    
    f_real = RegularGridInterpolator((x, y), np.real(mode_data), method='linear')
    f_imag = RegularGridInterpolator((x, y), np.imag(mode_data), method='linear')
    
    # 创建新网格点
    xv, yv = np.meshgrid(xnew, ynew, indexing='ij')
    points = np.column_stack([xv.ravel(), yv.ravel()])
    
    # 插值
    real_interp = f_real(points).reshape(len(xnew), len(ynew))
    imag_interp = f_imag(points).reshape(len(xnew), len(ynew))
    
    new_data = real_interp + 1j * imag_interp
    
    return new_data


#########
plt.close('all')

#================================================================
####### loading and preparing numerical beam at NF (fibre output)
#================================================================


filename = os.path.join(os.path.dirname(__file__), 'Modes_50um.xlsx')
if not('X' in globals()) or (X.size ==0):
    print('Loading reference modes...')    
    xls = pd.ExcelFile(filename)
    xls.parse('info')
    
    
    xls.sheet_names
    
    tempstring=xls.parse('info').to_string().split('\n0')
    modes_name= tempstring[0].split()
    modes_M2=list(map(float,tempstring[1].split()))
    
    
    X=np.array(xls.parse('X',header=None))
    #df = pd.read_excel(filename, header=None, sheet_name="Y")
    Y=np.array(xls.parse('Y',header=None))
    #Y=np.array(df)
    dim=np.shape(X)
    
    Etemp=[]
#    print('\n')
    # bar=Bar('\n Loading Modes data: ', max=len(modes_name))
    # bar.check_tty = False
    for sheet_name in modes_name:
        # bar.next()
        modetemp=np.array(xls.parse(sheet_name,header=None))
        Etemp.append(modetemp[:dim[0],:]+1j*modetemp[(dim[0]+1):,:])
        
    mode_data=np.asarray(Etemp)
    
#================================================================


###### MODELLING FOR PHASE PLATE

#################################
################################
#code for spirals - short
LP71=np.squeeze(mode_data[31,:,:]) #0 is LP01, 6 is LP31, 22 is LP41, 31 is LP71
modeLP=LP71
plt.close('all')

wavelength=1.075*um
N=2000
size=1*mm
# original NF
F=Begin(size,wavelength,128)
F.field=ResizeMode(modeLP,128,128)
resize=5*size
F=Interpol(resize,N, 0, 0, 0, 1, F)

plt.close('all')

Fin=F

plt.figure()
plt.suptitle('Top:Intensity, Bottom: Phase')
plt.subplot(2,1,1)
plt.imshow(Intensity(1,Fin),cmap='jet')
plt.subplot(2,1,2)
plt.imshow(Phase(F),cmap='Blues')

z1=10*cm
F=Forvard(z1,F) #Propagate back to the near field
F=Lens(z1,0,0,F)
F=Forvard(z1,F) #Propagate back to the near field
plt.figure()
plt.imshow(Intensity(1,F),cmap='jet')
FPhase=PhaseSpiral(F,m=-6) # m=6 or -6
F=Lens(z1,0,0,FPhase)
F=Forvard(z1,F) #Propagate back to the near field
plt.suptitle('LP31 mode after propagation through a SPP; Top:Intensity, Bottom: Phase')
plt.figure()
plt.subplot(2,1,1)
plt.imshow(Intensity(1,F),cmap='jet')
plt.subplot(2,1,2)
plt.imshow(Phase(F),cmap='Blues')


F=Forvard(2*cm,F) #Propagate back to the near field   2cm for LP71, 5cm for LP31
plt.subplot(2,1,1)
plt.imshow(Intensity(1,F),cmap='jet')
plt.subplot(2,1,2)
plt.imshow(Phase(F),cmap='Blues')

# save the intensity as CSV
final_intensity = Intensity(1,F)
intensity_output_path = os.path.join(os.path.dirname(__file__), "final_intensity_LP71_m6_0cm.csv")
np.savetxt(intensity_output_path, final_intensity, delimiter=",")

# save the phase as CSV
final_phase = Phase(F)
phase_output_path = os.path.join(os.path.dirname(__file__), "final_phase_LP71_m6_0cm.csv")
np.savetxt(phase_output_path, final_phase, delimiter=",")

plt.figure(1000)
plt.pcolormesh(final_intensity, shading='auto', cmap='jet')  # view(2) equivalent for top-down view
plt.colorbar()
plt.title('Imported Field Intensity from CSV')
plt.xlabel('X pixels')
plt.ylabel('Y pixels')
plt.show()





