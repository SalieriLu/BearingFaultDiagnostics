# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 21:53:10 2021

@author: luhao
"""
#load data resample and creat sub-signals


import numpy as np
import os,sys
import xlrd
import scipy.io as scio
from scipy import signal

from scipy.interpolate import interp1d
xlrd.xlsx.ensure_elementtree_imported(False, None)
xlrd.xlsx.Element_has_iter = True
#define the dir 

work_path=r'C:\Users\luhao\Dropbox\IWSHM_code\bearingforconference'
dB_list=[-8,-10,-12]



sf= 12800


#adding_Noise
def adding_Noise(signal,target_SNR):
    
    # Calculate signal power and convert to dB 
    sig_avg_watts = np.mean(signal)
    sig_avg_db = 10 * np.log10(abs(sig_avg_watts))
    # Calculate noise according then convert to watts
    noise_avg_db = sig_avg_db - target_SNR
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(signal))
    signal2 = signal + noise_volts
    return signal2






def res(array, npts):
    interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled

def Signal_processing_step1(new_sf,noise_db,sub_len,stride):
    sub_number=int((10*new_sf-sub_len)//stride);
    os.chdir(file_path)
    foldername = ['ball_defect','combine_defect','inner_defect','none_defect','outer_defect']
    for label_name in foldername:
        files= os.listdir(label_name+'/test')
        print(files)
        if label_name=='none_defect':
            index='0';
        if label_name=='inner_defect':
            index='1';
        if label_name=='outer_defect':
            index='2';
        if label_name == 'combine_defect':
            index = '3';
        if label_name == 'ball_defect':
            index = '4';

        data_r=np.zeros(sub_len+2);
        for i in range(np.size(files)):
            x1 = xlrd.open_workbook(label_name + '/test/' + files[i])
            data1 = x1.sheet_by_name('sheet1')
            d = data1.col_values(1)[1:]
            
            if new_sf != 25600:
                d= res(d,10*new_sf)
        
            if noise_db==None:
                d2=d
                folder_mark='Raw'
            else:
                d2=adding_Noise(d,noise_db)
                folder_mark=str(noise_db)
                
            speed=data1.col_values(4)[1]
            datac = d2[0:sub_len]
            datac=np.concatenate((datac,[float(speed),int(index)]),0)
            for i2 in range(1,sub_number+1):
                a=d2[int(i2 * stride):(sub_len + int(i2 * stride))]
                a=np.concatenate((a,[float(speed),int(index)]),0)
                datac = np.column_stack((datac,a))
            datac=datac.T
            data_r = np.row_stack((data_r, datac))
            print('file '+files[i]+' finished')
            # scio.savemat(pp+files[i], {'data': datac.T})
            if not os.path.exists(save_file_path+data_type+'/'+folder_mark):
                os.makedirs(save_file_path+data_type+'/'+folder_mark)
        os.chdir(save_file_path+data_type+'/'+folder_mark)
        data_r=np.delete(data_r,0,axis=0)
        scio.savemat(label_name+str(new_sf)+'.mat',{'data'+label_name:data_r})
        print('folder '+label_name+' finished')
        os.chdir(file_path)
    return 0



# #Training
for i in dB_list:
        
    
    file_path = work_path+'\dataset1\\'#1 for train 2 for test
    
    save_file_path=r'C:\Users\luhao\Dropbox\Research_folder\Paper writting\Preparation of Journal paper\Feature-Weightting Paper\CNN-CWL\Signal_processing_folder\Generated_data'
    #parameters
    sf=12800
    data_type='/train/'
    
    #Generate training data
    Signal_processing_step1(sf,i,1*sf,0.5*sf)
    
    
    ###Test
    
    file_path = work_path+'\dataset2\\'#1 for train 2 for test
    
    save_file_path=r'C:\Users\luhao\Dropbox\Research_folder\Paper writting\Preparation of Journal paper\Feature-Weightting Paper\CNN-CWL\Signal_processing_folder\Generated_data'
        
    #Generate test data
    data_type='/test/'

    Signal_processing_step1(sf,i,1*sf,0.5*sf)
    
    




#second part for signal process


import scipy.fft
import scipy.signal

#define the dir 
work_path=r'C:\Users\luhao\Dropbox\Bearing diagnosis\\'
file_path = work_path+'\data\dataset\\'
code_path = work_path+'\code\\'
save_file_path=file_path+'\Bearing diagnosis\\'


def frequency_cot(data,sf,speed,order_range):
  data_input=data;
  data_sf=sf;
  fre_range = np.linspace(0, data_sf, num=np.shape(data_input)[0], endpoint=True)

  inter_f = interp1d(fre_range/speed, data_input, kind='cubic')
  output_data=inter_f(order_range)
  return output_data

def data_analysis(data,sf):
  #take the data,sampling frequency and sampling time as imput and returns envelope - fft results.
  time=10;
  signal_len=np.size(data,0)
  h_fft_result=np.abs(np.fft.fft(np.abs(scipy.signal.hilbert(data))))
  return h_fft_result


saveing_size=1600

def signalanalysis(file_path,save_path,foldername,new_sf):
    ##function define
    sub_len = 1 * new_sf  
    ##
    os.chdir(file_path)
    
    ##data merge
    data0 = np.zeros(sub_len)
    samples = sub_len
    os.chdir(file_path)
       
    zz = scio.loadmat(file_path + '\\' + foldername[0] +str(new_sf))#+ '.mat')
    data = zz['data' + foldername[0]]
    signal_n=len(data)
    speed0 = np.zeros(signal_n)
    label0 = np.zeros(signal_n)
    or_range=np.linspace(0,saveing_size//100,num=saveing_size,endpoint=True) #Signal was converted into order domain and only those parts are keeped
    
    for f_name in foldername:
        os.chdir(file_path)
       
        zz = scio.loadmat(file_path + '\\' + f_name +str(new_sf))#+ '.mat')
        data = zz['data' + f_name]
        signal_data = data[0:signal_n, 0:sub_len]
        data0 = np.row_stack((data0, signal_data))
        speed0 = np.row_stack((speed0, data[:signal_n, -2]))
        label0 = np.row_stack((label0, data[:signal_n, -1]))
    data0 = np.delete(data0, 0, axis=0)
    speed0 = np.delete(speed0, 0, axis=0)
    label0 = np.delete(label0, 0, axis=0)
    speed0 = speed0.reshape(-1, 1)
    label0 = label0.reshape(-1, 1)
    print('finish data merge ')
    ####
    ###data process
    processed_data = np.zeros([0, saveing_size + 2])
    for i in range(len(data0)):
        subsignal = data0[i, :]
        envelope_spectreum=data_analysis(subsignal,new_sf)
        new_data=frequency_cot(envelope_spectreum,new_sf,speed0[i],or_range)
        subspeed=speed0[i]*np.ones(sub_len)
        new_data=np.concatenate((new_data,speed0[i],label0[i]),axis=0).reshape(1,saveing_size+2)
        processed_data = np.concatenate((processed_data, new_data), axis=0)

        # scio.savemat(pp+files[i], {'data': datac.T})
    if not os.path.exists(save_path):
        os.makedirs(save_path)      
    os.chdir(save_path)
    scio.savemat('result'+str(new_sf)+'.mat', {f_name: processed_data})
    print('file  ' + save_path + str(new_sf)+'  finished')
    return()

new=12800

for db_setting in dB_list:
    
    
    file_path1 = r'C:\Users\luhao\Dropbox\Research_folder\Paper writting\Preparation of Journal paper\Feature-Weightting Paper\CNN-CWL\Signal_processing_folder\Generated_data\train/'+str(db_setting)
    file_path2 = r'C:\Users\luhao\Dropbox\Research_folder\Paper writting\Preparation of Journal paper\Feature-Weightting Paper\CNN-CWL\Signal_processing_folder\Generated_data\test/'+str(db_setting)
    save_path1 = r'C:\Users\luhao\Dropbox\Research_folder\Paper writting\Preparation of Journal paper\Feature-Weightting Paper\CNN-CWL\Signal_processing_folder\Ready_data\train/'+str(db_setting)
    save_path2 = r'C:\Users\luhao\Dropbox\Research_folder\Paper writting\Preparation of Journal paper\Feature-Weightting Paper\CNN-CWL\Signal_processing_folder\Ready_data\test/'+str(db_setting)
    if not os.path.exists(save_path1):
        os.makedirs(save_path1)
    if not os.path.exists(save_path2):
        os.makedirs(save_path2)
    
    foldername = ['ball_defect','combine_defect','inner_defect','none_defect','outer_defect']
    signalanalysis(file_path1, save_path1, foldername, new)
    signalanalysis(file_path2, save_path2, foldername, new)
