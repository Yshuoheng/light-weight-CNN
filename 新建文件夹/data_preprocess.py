from scipy.io import loadmat
import pandas as pd
import numpy as np
from scipy import signal



def band_filter_process(input):
    
    b_HPF_1,a_HPF_1 = signal.butter(4, 0.048, btype='highpass') #4阶，6Hz ， 6/(sampling_rate/2) 和iirfilter效果一样
    b_LPF_1,a_LPF_1 = signal.butter(4, 0.64, btype='lowpass') #4阶，80Hz ，18/(sampling_rate/2) 和iirfilter效果一样
    
    
    
    data_HF_1 = signal.filtfilt(b_HPF_1, a_HPF_1, input) #零相位滤波，滤两次，反向多滤了一次
    band1 = signal.filtfilt(b_LPF_1, a_LPF_1, data_HF_1) #6-16
    
    
    
    return band1

def data_batch_FFT_one_tester_complex_UI(test_trainer,time_length):
    

    all_trailers = ['S1', 'S2', 'S3', 'S4', 'S5',
                    'S6', 'S7', 'S8', 'S9', 'S10',
                    'S11','S12','S13','S14','S15',
                    'S16','S17','S18','S19','S20',
                    'S21','S22','S23','S24','S25',
                    'S26','S27','S28','S29','S30',
                    'S31','S32','S33','S34','S35']

    FFT_PARAMS = {
    'resolution': 0.2930,
    'start_frequency': 3.0,
    'end_frequency': 60.0,
    'sampling_rate': 250
    }

    NFFT = round(FFT_PARAMS['sampling_rate']/FFT_PARAMS['resolution'])
    fft_index_start = int(round(FFT_PARAMS['start_frequency']/FFT_PARAMS['resolution']))
    fft_index_end = int(round(FFT_PARAMS['end_frequency']/FFT_PARAMS['resolution']))+1


    simpling_rate = 250
    window_length = int(250*time_length) #窗口长度1s
    data_length = 1250 #5s
    fft_scale = int(data_length/window_length) #5s则scale=5

    all_datas_1 = []
    
    all_labels = []
    
    test_datas_1 = []
    test_datas_2 = []
    test_datas_3 = []
    test_labels = []
    
    test_trainer_position = all_trailers.index(test_trainer)
    
    all_trailers.pop(test_trainer_position) #删除test_trainer
    trailer = all_trailers
    
    
    for i in range(len(trailer)):
        # print(trailer[i])
        rawdata = loadmat('../database/' + trailer[i] + '.mat') #1号被试
        rawdata = rawdata['data']
        
        for j in range(6): #6个block
        
            block_index = j
            
            for k in range(40): #40个trail
                
                target_index = k
                O1_data = rawdata[60,160:1410,target_index,block_index]
                Oz_data = rawdata[61,160:1410,target_index,block_index]
                O2_data = rawdata[62,160:1410,target_index,block_index]
                PO3_data = rawdata[54,160:1410,target_index,block_index]
                POz_data = rawdata[55,160:1410,target_index,block_index]
                PO4_data = rawdata[56,160:1410,target_index,block_index]
                PO5_data = rawdata[53,160:1410,target_index,block_index]
                Pz_data = rawdata[47,160:1410,target_index,block_index]
                PO6_data = rawdata[57,160:1410,target_index,block_index]
                
                yd4_1 = band_filter_process(O1_data)
                yd4_2 = band_filter_process(Oz_data)
                yd4_3 = band_filter_process(O2_data)
                yd4_4 = band_filter_process(PO3_data)
                yd4_5 = band_filter_process(POz_data)
                yd4_6 = band_filter_process(PO4_data)
                yd4_7 = band_filter_process(PO5_data)
                yd4_8 = band_filter_process(Pz_data)
                yd4_9 = band_filter_process(PO6_data)
                
                for m in range(fft_scale):

                    yd4_1_fft = np.fft.fft(yd4_1[m*window_length:(m+1)*window_length],NFFT)/window_length
                    yd4_2_fft = np.fft.fft(yd4_2[m*window_length:(m+1)*window_length],NFFT)/window_length
                    yd4_3_fft = np.fft.fft(yd4_3[m*window_length:(m+1)*window_length],NFFT)/window_length
                    yd4_4_fft = np.fft.fft(yd4_4[m*window_length:(m+1)*window_length],NFFT)/window_length
                    yd4_5_fft = np.fft.fft(yd4_5[m*window_length:(m+1)*window_length],NFFT)/window_length
                    yd4_6_fft = np.fft.fft(yd4_6[m*window_length:(m+1)*window_length],NFFT)/window_length
                    yd4_7_fft = np.fft.fft(yd4_7[m*window_length:(m+1)*window_length],NFFT)/window_length
                    yd4_8_fft = np.fft.fft(yd4_8[m*window_length:(m+1)*window_length],NFFT)/window_length
                    yd4_9_fft = np.fft.fft(yd4_9[m*window_length:(m+1)*window_length],NFFT)/window_length
                    
                    all_datas_1.append(np.array([np.concatenate((np.real(yd4_1_fft)[fft_index_start:fft_index_end,],np.imag(yd4_1_fft)[fft_index_start:fft_index_end,]),axis=0),
                                                 np.concatenate((np.real(yd4_2_fft)[fft_index_start:fft_index_end,],np.imag(yd4_2_fft)[fft_index_start:fft_index_end,]),axis=0),
                                                 np.concatenate((np.real(yd4_3_fft)[fft_index_start:fft_index_end,],np.imag(yd4_3_fft)[fft_index_start:fft_index_end,]),axis=0),
                                                 np.concatenate((np.real(yd4_4_fft)[fft_index_start:fft_index_end,],np.imag(yd4_4_fft)[fft_index_start:fft_index_end,]),axis=0),
                                                 np.concatenate((np.real(yd4_5_fft)[fft_index_start:fft_index_end,],np.imag(yd4_5_fft)[fft_index_start:fft_index_end,]),axis=0),
                                                 np.concatenate((np.real(yd4_6_fft)[fft_index_start:fft_index_end,],np.imag(yd4_6_fft)[fft_index_start:fft_index_end,]),axis=0),
                                                 np.concatenate((np.real(yd4_7_fft)[fft_index_start:fft_index_end,],np.imag(yd4_7_fft)[fft_index_start:fft_index_end,]),axis=0),
                                                 np.concatenate((np.real(yd4_8_fft)[fft_index_start:fft_index_end,],np.imag(yd4_8_fft)[fft_index_start:fft_index_end,]),axis=0),
                                                 np.concatenate((np.real(yd4_9_fft)[fft_index_start:fft_index_end,],np.imag(yd4_9_fft)[fft_index_start:fft_index_end,]),axis=0),]))
                                              
                     
                    all_labels.append(target_index)
    
    all_datas = np.expand_dims(np.array(all_datas_1),axis=-1)
    
    all_labels = np.array(all_labels)
    
    
    rawdata = loadmat('../database/' + test_trainer + '.mat')
    rawdata = rawdata['data']
    
    for j in range(6): 
    
        block_index = j
        
        for k in range(40):
            
            target_index = k
           
            O1_data = rawdata[60,160:1410,target_index,block_index] 
            Oz_data = rawdata[61,160:1410,target_index,block_index]
            O2_data = rawdata[62,160:1410,target_index,block_index]
            PO3_data = rawdata[54,160:1410,target_index,block_index]
            POz_data = rawdata[55,160:1410,target_index,block_index]
            PO4_data = rawdata[56,160:1410,target_index,block_index]
            PO5_data = rawdata[53,160:1410,target_index,block_index]
            Pz_data = rawdata[47,160:1410,target_index,block_index]
            PO6_data = rawdata[57,160:1410,target_index,block_index]
            
            yd4_1 = band_filter_process(O1_data)
            yd4_2 = band_filter_process(Oz_data)
            yd4_3 = band_filter_process(O2_data)
            yd4_4 = band_filter_process(PO3_data)
            yd4_5 = band_filter_process(POz_data)
            yd4_6 = band_filter_process(PO4_data)
            yd4_7 = band_filter_process(PO5_data)
            yd4_8 = band_filter_process(Pz_data)
            yd4_9 = band_filter_process(PO6_data)
            
            for m in range(fft_scale):
                
                yd4_1_fft = np.fft.fft(yd4_1[m*window_length:(m+1)*window_length],NFFT)/window_length
                yd4_2_fft = np.fft.fft(yd4_2[m*window_length:(m+1)*window_length],NFFT)/window_length
                yd4_3_fft = np.fft.fft(yd4_3[m*window_length:(m+1)*window_length],NFFT)/window_length
                yd4_4_fft = np.fft.fft(yd4_4[m*window_length:(m+1)*window_length],NFFT)/window_length
                yd4_5_fft = np.fft.fft(yd4_5[m*window_length:(m+1)*window_length],NFFT)/window_length
                yd4_6_fft = np.fft.fft(yd4_6[m*window_length:(m+1)*window_length],NFFT)/window_length
                yd4_7_fft = np.fft.fft(yd4_7[m*window_length:(m+1)*window_length],NFFT)/window_length
                yd4_8_fft = np.fft.fft(yd4_8[m*window_length:(m+1)*window_length],NFFT)/window_length
                yd4_9_fft = np.fft.fft(yd4_9[m*window_length:(m+1)*window_length],NFFT)/window_length
                
                
                
                test_datas_1.append(np.array([np.concatenate((np.real(yd4_1_fft)[fft_index_start:fft_index_end,],np.imag(yd4_1_fft)[fft_index_start:fft_index_end,]),axis=0),
                                                 np.concatenate((np.real(yd4_2_fft)[fft_index_start:fft_index_end,],np.imag(yd4_2_fft)[fft_index_start:fft_index_end,]),axis=0),
                                                 np.concatenate((np.real(yd4_3_fft)[fft_index_start:fft_index_end,],np.imag(yd4_3_fft)[fft_index_start:fft_index_end,]),axis=0),
                                                 np.concatenate((np.real(yd4_4_fft)[fft_index_start:fft_index_end,],np.imag(yd4_4_fft)[fft_index_start:fft_index_end,]),axis=0),
                                                 np.concatenate((np.real(yd4_5_fft)[fft_index_start:fft_index_end,],np.imag(yd4_5_fft)[fft_index_start:fft_index_end,]),axis=0),
                                                 np.concatenate((np.real(yd4_6_fft)[fft_index_start:fft_index_end,],np.imag(yd4_6_fft)[fft_index_start:fft_index_end,]),axis=0),
                                                 np.concatenate((np.real(yd4_7_fft)[fft_index_start:fft_index_end,],np.imag(yd4_7_fft)[fft_index_start:fft_index_end,]),axis=0),
                                                 np.concatenate((np.real(yd4_8_fft)[fft_index_start:fft_index_end,],np.imag(yd4_8_fft)[fft_index_start:fft_index_end,]),axis=0),
                                                 np.concatenate((np.real(yd4_9_fft)[fft_index_start:fft_index_end,],np.imag(yd4_9_fft)[fft_index_start:fft_index_end,]),axis=0),]))
                                              
                
                
                test_labels.append(target_index)
    
    test_datas = np.expand_dims(np.array(test_datas_1),axis=-1)
    test_datas_2 = np.expand_dims(np.array(test_datas_2),axis=-1)
    test_datas_3 = np.expand_dims(np.array(test_datas_3),axis=-1)
    test_labels = np.array(test_labels)

    return all_datas,all_labels,test_datas,test_labels

'''
Main
'''

if __name__ == '__main__':
    
    # data_batch_FFT(True)
    data_batch_FFT_one_tester('S1')


















