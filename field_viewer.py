# -*- coding: utf-8 -*-
"""
Script that coordinates audio and video
and makes a movie out of it -
this allows for a combined visualisation of the
bats, mic and the recording rms - or other such parameters

Created on Mon Jul 17 22:57:09 2017

@author: tbeleyur
"""
import scipy.signal as signal
import numpy as np
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 100000
import pandas as pd
import peakutils
import cv2
import glob


# TO DO :
# 1) convert all of these functions into a class based function collection



def rms_calculator(wav_file,block_size,**kwargs):
    '''
    splits a wav file into multiple chunks of block_size
    and calculates the rms value of this wav file

    Input    :
    wav_file : string. address of the WAV file
    block_size: integer. number of samples over which the rms will be
            calculated. DefAULT VALUE 320 SAMPLES

    **kwargs:
        synchro_channel:
            which has frame time information. Default = True

    Outputs:
    rms_chunked : 1x num_chunks np.array. num_chunks is
                rounded up after calculating wav_file.size / block_size

    '''
    fs,recording = load_wavfile(wav_file)

    print(wav_file)

    try:
        block_size = int(block_size)
    except:
        ValueError('Problem converting block_size into integer')

    # if there's no synchro_channel - then assume that the video starts at the
    # 0th sample

    if not('synchro_channel' in kwargs):
        start_segment = 0
    else:
        start_segment = np.min( kwargs['synchro_channel']  )

    print('the start index is ',start_segment)
    end_segment = start_segment+ block_size -1

    rms_chunked_list = []

    # as of 18/7/2017 the script calculates rms in blocks only to
    # the right of the first frame. Data from the left of this
    # sample is *ignored*

    #  round up to avoid losing out data at the very end

    num_chunks = int(np.ceil( (recording.size-start_segment) /float(block_size)))

    i = 0
    for k in range(num_chunks):
        i +=1

        try:
            rec_segment = recording[start_segment:end_segment ]

        except:
            # in the end of the recording, when the block size
            # exceeds the length of the array
            rec_segment = recording[start_segment:]

        else :
            ValueError('Indexing problem with extracting last segment of recording..')

        rms_chunk = np.std(rec_segment)
        rms_chunked_list.append(rms_chunk)

        start_segment += block_size
        end_segment += block_size


    print('num of chunks: ',i)

    try:
        rms_chunked = np.asanyarray(rms_chunked_list)
    except:
        Exception('error in converting rms_chunked_list into array')

    chunked_rms_data = {'chunked_rmsdata':rms_chunked,'block_size':block_size}

    return(chunked_rms_data)



def load_wavfile(wav_file):
    try:
        fs,recording = wav.read(wav_file)

        if np.max(np.abs(recording)) >1:
            raise ValueError('The wav file is not between -1 and +1 , please check')
    except:
        Exception('Problem reading wav file')



    return(fs,recording)


def extract_frametimes(synchron_channel,fs=192000,vid_sig_Hz = 25, lowpass_freq = 10.0*10**3):
    '''
    when a square wave + high frequency signal is
    fed in as a single channel recording - then it extracts
    the sample index of the maximum value on the rising edge.

    IMPORTANT : The function expects a 25/30 Hz signal

    Inputs:
    synchron_channel: 1xN samples np.array with square wave. The frames
            are recorded when the values hit their peak +ve value.
    fs

    '''

    # LP the recording, find a zero crossing with a +ve slope
    # and then find the position of highest values in its vicinity

    if not (vid_sig_Hz in [25,30]):
        ValueError('this video signal frequency is not supported by the current function')

    # lowpass filter the signal
    if lowpass_freq <= [25,30]:
        ValueError('lowpass frequency too low - please check value once more')

    b,a = signal.butter(8, lowpass_freq/float(fs), 'low' )

    # use filtfilt to get zero phase delay
    synchron_lp = signal.filtfilt(b,a,synchron_channel)

    # now extract all points where the transition from -ve to +ve occurs :

    trans_indxs = get_frame_times(synchron_lp)

    return(trans_indxs)



def get_frame_times(synchron_channel):
    '''
    extracts the sample points at which the
    frame was recorded. In the TeAx FLIR Tau2 cores
    , frames are recorded when voltage increases above a
    particular +ve voltage.

    Inputs:
    synchron_channel: 1 x N np.array channel with
                    square waves of 25 or 30 Hz.
                    IMPORTANT: currently no other video frame rate is
                    supported

    Outputs:
    frame_inds: 1 x num_frames np.array. Has the indices at which the square wave
                rises to its maximum value - this is the sample at which
                frame recording was triggered in the camera


    '''

    # remember TO CHANGE THIS WHEN IT'S IN CLASS MODE ;
    FS = 192000

    threshold = np.max(synchron_channel) * 0.75

    min_pk_2_pk = (1/30.0)*FS

    diff_synchron = np.diff(synchron_channel)

    # this is the approximate peaks in the channel got from the
    # derivative
    aprx_pks = peakutils.indexes(diff_synchron,thres=threshold, min_dist = min_pk_2_pk)


    # now we get the precise maximum within +/- 100 sampls of this approximate position

    precise_pks = []
    for each_aprx_pk in aprx_pks:

        # when the aprx_pk is in the 'middle' of the array
        left_edge = each_aprx_pk - 100
        right_edge = each_aprx_pk + 100
        search_segment = synchron_channel[left_edge:right_edge]

        if left_edge <0:
            # when the aprx_pk is to the extreme left of the array
            left_edge = 0
            right_edge = each_aprx_pk + 100
            search_segment = synchron_channel[left_edge:right_edge]

        if right_edge > (synchron_channel.size -1) :
            # when the aprx_pk is to the extreme right of the array
            left_edge = each_aprx_pk - 100
            right_edge =  (synchron_channel.size -1)
            search_segment = synchron_channel[left_edge:]

        # add on the local search_segment index maxima to get
        # the location of the +ve peak in the whole recording

        if np.max(search_segment) >= threshold:
            search_seg_argmax = np.argmax(search_segment)
            synchron_channel_indx = left_edge + (search_seg_argmax - 1 )
            precise_pks.append(  synchron_channel_indx )
        else :
            pass

    frame_inds = np.asanyarray(precise_pks)


    return(frame_inds)



def play_AV(videoin_address,videoout_address,mics_rms,mics_pos,rms_vals_per_frame,DLTdv5=True,**kwargs):
    '''
    function which plays the vide and plots audio rms

    Inputs:
        videoin_address: string. path to video file

        videoout_address:string. path to output video file with overlay

        mics_rms : N_samples x num_channels np.array of recorded data from the
                inputs mics whose rms values will be plotted

        mics_pos: 1 x (num_mics x 2) np.array with the x,y pixel coordinates

        rms_vals_per_frame: integer. Number of rms values that are to be plotted
                        for one frame.

        DLTDv5 : Boolean. Default= True. Whether the xy coordinates were digitised
                from DLTdv5
        **kwargs:
        bat_positions: num_video_frames x (num_batsx2) np.array, with the xy pixel
                        coordinates over time
    '''



    # now let's load the video, and update the audio accordingly :
    try:
        cap =  cv2.VideoCapture(videoin_address)
        num_frames = int ( cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height,width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_shape = (height,width)
    except:
        raise ValueError('Unable to load video - please check file or address')

    try:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        vid_out = cv2.VideoWriter(videoout_address,fourcc,10,(width,height))
    except:
        raise Exception('Unable to open videowriter object, check videoout_address')



    # check if num mics and num audio channels match:

    num_mics, num_audio_channels = check_channels_to_mics(mics_pos,mics_rms)

    num_total_blocks = mics_rms.shape[0]

    # get all the mic positions:
    mics_x = mics_pos[0::2]
    mics_y = mics_pos[1::2]

    if DLTdv5:
        mics_y = conv_DLTdv5_to_opencv2(mics_y,frame_shape)


    disp_frame = 0
    audio_blocknum = 0
    dispd_blocks = 0

    while (disp_frame < num_frames) & (audio_blocknum < num_total_blocks):

        dispd_blocks = 0
        #cap.set(1,disp_frame); # Where frame_no is the frame you want
        ret, frame = cap.read() # Read the frame

        while dispd_blocks < rms_vals_per_frame :


            rms_radii =  np.apply_along_axis(conv_rms_to_radius,0,mics_rms,audio_blocknum)

            for each_mic in range(num_mics):
                cv2.circle(frame, (mics_x[each_mic] , mics_y[each_mic] ), rms_radii[each_mic], (16,105,255), -1 )


            cv2.putText(frame,str(disp_frame/25.0),(500,200),cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

            # TO BE ADDED HERE: plotting of bat positions - when given in kwargs
            if np.remainder(audio_blocknum,23.0)==0 :
                cv2.putText(frame,str(audio_blocknum),(500,100),cv2.FONT_HERSHEY_SIMPLEX, 1, 255)


            cv2.imshow('field_viewer', frame) # show frame on window

            # TO BE ADDED HERE: save the displayed frame into a VideoWriter object

            # write the frame with all the overlayed information
            vid_out.write(frame)

            ch = 0xFF & cv2.waitKey(1)

            if ch == ord('q') :
                disp_frame = num_frames+1


            audio_blocknum +=1
            dispd_blocks +=1




        disp_frame += 1

    cap.release()
    vid_out.release()
    cv2.destroyAllWindows()



    return()


def conv_DLTdv5_to_opencv2(y_coods,frame_shape):
    '''
    converts the y image coordinates
    from DLTdv5 into opencv2 compatible coordinates
    Inputs:
    y_coods : N x 1 np.array. with y coordinates
    frame_shape : 1 x 2 np.array with integer values of pixel height and width of
                frame
    '''

    frame_height = frame_shape[0]

    y_compatible = frame_height - y_coods

    return(y_compatible)

def conv_rms_to_radius(rms_array,index):
    '''
    converts a np.array with rms values into a viewable
    circle with rms proportional radius in pixels
    '''
    magnif_factor = 100
    rms_value = rms_array[index]

    if rms_value < 0:
        raise ValueError('rms value cannot be less than 0 - please check how rms was calculated')
    else:
        radius = int(  np.around( magnif_factor*rms_value) ) + 3

        return( radius )



def read_csv_files(csv_file):
    '''
    reads the csv file
    '''
    csv_data = pd.read_csv(csv_file)
    print('CSV file  ' + csv_file +' read succesfully')

    return(csv_data)


def check_for_micpos():
    '''
    checks if the micpos.csv file is proper or not

    '''

    return()

def check_for_batpos():
    '''
    '''

    return()


def check_integer_blocks_per_frame():
    '''
    '''

    return()

def check_channels_to_mics(mic_pos,mic_audio):
    '''
    checks if the number of mic audio channels and the
    number of xy positions match

    mic_pos: np.array. 1 x (2 x number of mics). with xy positions
    mic_audio: np.array. (N_blocks x num_mics),

    '''
    try:
        num_mics = int(mic_pos.shape[1]/2.0)
    except:
        num_mics = int(mic_pos.shape[0]/2.0)

    num_channels = mic_audio.shape[1]

    if num_mics == num_channels:

        return(num_mics,num_channels)
    else:
        ValueError('number of mic coordinates and mic channels do not match')





def adjust_origin():
    '''
    shifts the coordinates so that
    the graph type origins ( w 0,0 at bottom left)
    is compatible with the matrix type origin (w 0,0 at top left)
    '''

    return()











if __name__ == '__main__':

    folder = 'C:\\Users\\tbeleyur\\Documents\\common\\Python_common\\field_viewer\\test_data\\play_av_test\\'
    video = 'K3_allbats_P09_5000_21_21_30.avi'
    synchro_channel = 'synchro.wav'
    micspos_csv =   'micpos_0_11.csv'
    mic_wavs = glob.glob(folder+'Mic*.wav')

    fs, synchron = wav.read(folder+synchro_channel)
    f_times = extract_frametimes(synchron)
    rms_chunked = [ rms_calculator( each_file,320,synchro_channel = f_times)['chunked_rmsdata'] for each_file in mic_wavs ]
    mics_rms = np.column_stack(rms_chunked)
    micpos = np.asanyarray(read_csv_files(folder+micspos_csv)).flatten()
    micpos = micpos.astype('int16')

    output_video = folder+'TEST_OUT_23.avi'

    play_AV(folder+video,output_video,mics_rms,micpos,24)

