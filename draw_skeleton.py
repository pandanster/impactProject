'''
Created on Apr 6, 2018

@author: Amin
'''
import sys
sys.path.remove('/home/gmuadmin/catkin_ws/devel/lib/python2.7/dist-packages')
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import numpy as np
import cv2
import random
from random import shuffle
import matplotlib.pyplot as plt
from collections import defaultdict


joint_line_dic = \
{ 20:[1, 2, 4, 8], 0 :[1, 12, 16], 2:[3], 8:[9], 10:[9,24], 11:[10,23], \
4:[5], 6:[5,7,22], 7:[21], 17:[16, 18], 18:[19], 13:[12, 14], 14:[15]}


def connect_joint(image_frame='', joint_list=''):
    for k in joint_line_dic.keys():
        neighbor_joints = joint_line_dic[k]
        for nj in neighbor_joints:
            cv2.line(image_frame, joint_list[int(k)], joint_list[int(nj)], (150, 150, 150), 5)
    return image_frame

'''
given a rgb file and sk file, draw skeleton on rgb file
given that actual gesture samples are segmented by 'start' and 'end' text
interleaved in lines of skeleton data file
'''

def draw_on_rgb(vid_file_loc='', sk_file_loc='', sample_no=0, vid_out=''):
    vid = cv2.VideoCapture(vid_file_loc)
    sk_file = open(sk_file_loc,'r')
    file_lines = [line for line in sk_file.readlines()]
    indices=[i for i,j in enumerate(file_lines) if "start" in j or "end" in j]
    temp_lines=[]
    for x in range(0,len(indices),2):
       temp_lines+=file_lines[indices[x]+1:indices[x+1]]
    file_lines=temp_lines
    s, e = 0, len(file_lines)
  
    temp_file_lines = file_lines.copy()
    dist_list = []
    file_lines = []

    dist_id = 0
    while dist_id<len(temp_file_lines)-1:
      curr_frame = temp_file_lines[dist_id]
      next_frame = temp_file_lines[dist_id+1]
      
      if curr_frame in ['start\n', 'end\n']:
        curr_frame = temp_file_lines[dist_id-1]
      if next_frame in ['start\n', 'end\n']:
        dist_id += 1
        continue
      
      curr_frame_spmid = (curr_frame.strip().split(',')[1].strip().split(' ')[:3])
      next_frame_spmid = (next_frame.strip().split(',')[1].strip().split(' ')[:3])
      spmid_dist = sum([(float(t[0])-float(t[1]))**2 for t in zip(next_frame_spmid, curr_frame_spmid)])
      
      if spmid_dist>0.2:
        file_lines.append(curr_frame)
        dist_id += 2
      else:
        file_lines.append(curr_frame)
        dist_id += 1
      
    print ('avi frame count', vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print ('sk file line no. ', len(file_lines))
    
    f_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)//2) - 200       # for tagging texts
    
    base_joint = 1
    considered_joints = [1, 8, 9, 10, 11, 23, 24, 4, 5, 6, 7, 21, 22]
    
    vwidth = vid.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    vheight = vid.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    print (vwidth, vheight)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_writer = cv2.VideoWriter(vid_out, fourcc, 10.0, (int(vwidth*.5),int(vheight*.5)), True)
    
    read_cnt = 0
    while read_cnt<s:
      ret, frame = vid.read()
      read_cnt += 1
    #sample_no = int(vid_file_loc.split('\\')[-1].split('.')[0].split('_')[-2])
    #sample_no = int(vid_file_loc.split('/')[-1].split('.')[0].split('_')[-2])
    sample_no=0	
    for i, line in enumerate(file_lines):
        joint_dat = line.strip().split(',')
        if ('start' in joint_dat or 'end' in joint_dat):
            continue
        ret, frame = vid.read()
        
        resized_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        joint_pos_list = []
        for jn, joint_details in enumerate(joint_dat):
            joint_details = joint_details.strip().split(' ')
            joint_x = int(joint_details[-4])
            joint_y = int(joint_details[-3])
            joint_pos_list.append((joint_x, joint_y))
            if jn==6:
              cv2.circle(frame, (joint_x, joint_y), 10, (255,0,255), -1)
            elif jn==10:
              cv2.circle(frame, (joint_x, joint_y), 10, (67,25,55), -1)
            else:
              cv2.circle(frame, (joint_x, joint_y), 10, (0,0,255), -1)
        frame=np.zeros(frame.shape)
        skeletoned_frame = connect_joint(frame, joint_pos_list)
        resized_sk_frame = cv2.resize(skeletoned_frame, (0,0), fx=0.5, fy=0.5) 
        cv2.putText(resized_sk_frame, 'sign:'+os.path.basename(vid_file_loc).split('_')[0]+' || sample '+str(sample_no+1)+' || frame_no_'+str(i),(f_width//4,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 23, 22), 2)
#         cv2.imshow('frame', resized_frame)
        cv2.imshow('sk_frame',resized_sk_frame)
        vid_writer.write(resized_sk_frame)
        k = cv2.waitKey(-1)
    vid_writer.release()


'''
given a skeleton file input, it moves specific start stop flags for 
aligning segmented gestures 

'''
def move_start_end_flags(file_loc=''):
  
  start_pos= -1
  gest_list = []
  line_list = [] 
  with open(file_loc,'r') as f:
    for ln, line in enumerate(f):
      if line.strip()=='start':
        start_pos=ln
      elif line.strip()=='end':
        gest_list.append((start_pos, ln))
      
      line_list.append(line.strip())
  
  print (gest_list)
  gest_list = [(t[0]-25, t[1]) for t in gest_list]
  print (gest_list)
  print (len(gest_list))
  new_file = file_loc.split('.')[0]+'_aligned.txt'
  f = open(new_file, 'w')
  seg_id = 0
  
  for i, ln in enumerate(line_list):
    
    if gest_list[seg_id][0]==i:
      f.write('start\n')
    if gest_list[seg_id][1]==i:
      f.write('end\n')
      seg_id += 1
    if ln =='start' or ln=='end':
      continue
    f.write(ln)
    f.write('\n')
    if seg_id >= len(gest_list):
      break
  f.close()
  
  return True

'''
segment vid and skeleton files based on 'start' and 'end' text
interleaved in skeleton lines
'''  
def segment_from_whole_v3(vid_file_loc='', sk_file_loc='', save_loc='', \
        already_files=0, saved_file_name=''):
    
    vid = cv2.VideoCapture(vid_file_loc)
    sk_file = open(sk_file_loc,'r')
    file_lines = [line for line in sk_file]
    
    
    temp_file_lines = file_lines.copy()
    file_lines = []
    '''
    for i, ln in enumerate(temp_file_lines):
      if i >=176 and i<=395:
        if i%2==0:
          file_lines.append(ln)
      else:
        file_lines.append(ln)
    '''
    # this portion eliminates the two skeleton in one rgb frame problem
    dist_id = 0
    while dist_id<len(temp_file_lines)-1:
      curr_frame = temp_file_lines[dist_id]
      next_frame = temp_file_lines[dist_id+1]
      
      if curr_frame in ['start\n', 'end\n']:
        file_lines.append(curr_frame)
        curr_frame = temp_file_lines[dist_id-1]
      if next_frame in ['start\n', 'end\n']:
        dist_id += 1
        continue
      
      curr_frame_spmid = (curr_frame.strip().split(',')[1].strip().split(' ')[:3])
      next_frame_spmid = (next_frame.strip().split(',')[1].strip().split(' ')[:3])
      spmid_dist = sum([(float(t[0])-float(t[1]))**2 for t in zip(next_frame_spmid, curr_frame_spmid)])
      
      if spmid_dist>0.2:
        file_lines.append(curr_frame)
        dist_id += 2
      else:
        file_lines.append(curr_frame)
        dist_id += 1
      
    print ('avi frame count', vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print ('sk file line no. ', len(file_lines))
    sample_count = already_files
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    f_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    f_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    
    sk_writer = ''
    vid_writer = ''
    actual_frame_cnt =0
    temp_sk_frame =[]
    temp_rgb_frame=[]
    sample_flag=False
    for i, line in enumerate(file_lines):
        joint_dat = line.strip().split(',')         # string works because segment mark are 'start' 'end'
        if 'start' in joint_dat:
            # start new skeleton and rgb
            sample_flag = True
            temp_rgb_frame.clear()
            temp_sk_frame.clear()
        elif 'end' in joint_dat:
            # save segmented video and sk
            sample_initial = saved_file_name+str(sample_count+1)
            sample_flag = False
            vid_file_loc = os.path.join(save_loc,sample_initial+'_rgb.avi')
            sk_file_loc = os.path.join(save_loc,sample_initial+'_bodyData.txt')
            
#             if os.path.exists(vid_file_loc):    # this check is now in calling func
#               print (saved_file_name, 'already segmented, returning...')
#               return True
            
            vid_writer = cv2.VideoWriter(vid_file_loc,fourcc, 20.0, (f_width,f_height))
            sk_writer = open(sk_file_loc,'w')
            for fr, ln in list(zip(temp_rgb_frame, temp_sk_frame)):
                sk_writer.write(ln)
                vid_writer.write(fr)
            sk_writer.close()
            vid_writer.release()
            sample_count += 1
            print ('sample written ', sample_count, 'named', sample_initial)
    
        elif sample_flag:
            # keep saving
            ret, frame = vid.read()
            temp_rgb_frame.append(frame)
            temp_sk_frame.append(line)
        else:
            ret, frame = vid.read()
            # just discard
            
        '''
        if '\n' not in joint_dat:
            ret, frame = vid.read()
        
        if '\n' in joint_dat:
            sample_count += 1
            
            
            if sample_count and sample_count%2==0:
                      
                total_frame = len(temp_sk_frame)
                print ('total frame', total_frame)
                extra_frame = total_frame-50
                front_extra = max(0, (extra_frame//3)*2)
                back_extra = max(0, extra_frame-front_extra)
                print ('sample: ', (sample_count+1)//2, 'front - back extra', front_extra, back_extra)
                for fr, ln in list(zip(temp_rgb_frame, temp_sk_frame))[front_extra:total_frame-back_extra]:
                    sk_writer.write(ln)
                    vid_writer.write(fr)
                temp_sk_frame.clear()
                temp_rgb_frame.clear()
            
            if sample_count and sample_count%2:         # starting new file
                
                vid_file_loc = os.path.join(save_loc,saved_file_name+str(already_files+(sample_count+1)//2)+'_rgb.avi')
                sk_file_loc = os.path.join(save_loc,saved_file_name+str(already_files+(sample_count+1)//2)+'_bodyData.txt')
                vid_writer = cv2.VideoWriter(vid_file_loc,fourcc, 20.0, (f_width,f_height))
                sk_writer = open(sk_file_loc,'w')
                actual_frame_cnt = 0
            
            
        elif sample_count and sample_count%2:  #skips whatever at begining
            temp_sk_frame.append(line)
            temp_rgb_frame.append(frame)
#             sk_writer.write(line)
#             vid_writer.write(frame)
       '''


'''
given whole vid file location and sk file location it segments based on 
new lines in the sk files. each new line means separate sample 
'''
def segment_from_whole_v2(vid_file_loc='', sk_file_loc='', save_loc='', \
        already_files=0, saved_file_name=''):
    
    vid = cv2.VideoCapture(vid_file_loc)
    sk_file = open(sk_file_loc,'r')
    file_lines = [line for line in sk_file]
    print ('avi frame count', vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print ('sk file line no. ', len(file_lines))
    
    
    sample_count = 0
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    f_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    f_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    
    sk_writer = ''
    vid_writer = ''
    actual_frame_cnt =0
    temp_sk_frame =[]
    temp_rgb_frame=[]
    for i, line in enumerate(file_lines):
        joint_dat = line.split(',')
        
        if '\n' not in joint_dat:
            ret, frame = vid.read()
        
        if '\n' in joint_dat:
            sample_count += 1
            
            
            if sample_count and sample_count%2==0:
                      
                total_frame = len(temp_sk_frame)
                print ('total frame', total_frame)
                extra_frame = total_frame-50
                front_extra = max(0, (extra_frame//3)*2)
                back_extra = max(0, extra_frame-front_extra)
                print ('sample: ', (sample_count+1)//2, 'front - back extra', front_extra, back_extra)
                for fr, ln in list(zip(temp_rgb_frame, temp_sk_frame))[front_extra:total_frame-back_extra]:
                    sk_writer.write(ln)
                    vid_writer.write(fr)
                temp_sk_frame.clear()
                temp_rgb_frame.clear()
            
            if sample_count and sample_count%2:         # starting new file
                
                vid_file_loc = os.path.join(save_loc,saved_file_name+str(already_files+(sample_count+1)//2)+'_rgb.avi')
                sk_file_loc = os.path.join(save_loc,saved_file_name+str(already_files+(sample_count+1)//2)+'_bodyData.txt')
                vid_writer = cv2.VideoWriter(vid_file_loc,fourcc, 20.0, (f_width,f_height))
                sk_writer = open(sk_file_loc,'w')
                actual_frame_cnt = 0
            
            
        elif sample_count and sample_count%2:  #skips whatever at begining
            temp_sk_frame.append(line)
            temp_rgb_frame.append(frame)
#             sk_writer.write(line)
#             vid_writer.write(frame)
        
    

            

def segment_from_whole(vid_file_loc='', sk_file_loc='', start_pos_file='', \
            save_mode=False, cut_file_loc='', start_pos =0, idx=0):
    '''
    This function segments samples from the whole video based on some
    start positions. Saves segmented videos and skleton files whien save_mode
    '''
    
    vid = cv2.VideoCapture(vid_file_loc)
    sk_file = open(sk_file_loc,'r')
    file_lines = [line for line in sk_file]
    print ('avi frame count', vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print ('sk file line no. ', len(file_lines))
    f_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    f_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    text_pos_x = f_width//2-200
    
    # start pos for thrmon_professor
#     start_pos = [240, 430, 640, 835, 1035, 1220, 1425, 1625, 1825, 2015, \
#         2015, 2190, 2370, 2555, 2730, 2945, 3125, 3320, 3495, 3660, 3845]
    # calculating start pos from file
#     cut_locs = open(cut_file_loc, 'r')
#     start_pos = cut_locs.readline().strip().split(' ')
#     start_pos = [int(p) for p in start_pos]
    start_pos = [start_pos]     # only in case of int alamin segmentation
    
    #write video when savemode on
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    
    vid_writer =''
    sk_writer = ''
    for sample_id, sp in enumerate(start_pos):
        if save_mode==True:
            name_pos = 0
            if idx<9:
                name_pos=-9
            else:
                name_pos=-10
            saved_file_name = vid_file_loc[:name_pos]+str(idx+1)+'_'
            print ('saved_file_name', saved_file_name+'bodyData40.txt')
            vid_writer = cv2.VideoWriter(saved_file_name+'rgb40.avi',fourcc, 20.0, (f_width,f_height))
            sk_writer = open(saved_file_name+'bodyData40.txt','w')
      
            
        vid.set(cv2.CAP_PROP_POS_FRAMES, sp)
        skeleton_data = file_lines[sp:sp+40]
        for i, line in enumerate(skeleton_data):
            joint_dat = line.split(',')
            ret, frame = vid.read()
            
            if save_mode==True:
                sk_writer.write(line)
                vid_writer.write(frame)
            
            print (joint_dat)
            joint_pos_list = []
            for joint_details in (joint_dat):
                joint_details = joint_details.split(' ')
                joint_x = int(joint_details[-4])
                joint_y = int(joint_details[-3])
                joint_pos_list.append((joint_x, joint_y))
                cv2.circle(frame, (joint_x, joint_y), 5, (0,255,0), -1)
            skeletoned_frame = connect_joint(frame, joint_pos_list)
            resized_frame = cv2.resize(skeletoned_frame, (0,0), fx=0.5, fy=0.5) 
            cv2.putText(resized_frame, str(sample_id+1)+'frame_no_'+str(i),(text_pos_x//2,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 23, 22), 2)
            cv2.imshow('frame',resized_frame)
            cv2.waitKey(50)
        if save_mode==True:
            sk_writer.close()
            vid_writer.release()


def data_stats(file_dir=''):
    '''
    this method provide how many samples from each gestures
    '''
    file_names_all = os.listdir(file_dir)
    file_names = sorted([f for f in file_names_all if '.txt' in f])
    avi_file_names = sorted([f for f in file_names_all if '.avi' in f])
    '''
    # avi vs txt check
    for tfile in file_names:
      sptfile = tfile.split('_')
      rgb_file = sptfile[0]+'_'+sptfile[1]+'_'+sptfile[2]+'_rgb.avi'
      if rgb_file not in avi_file_names:
        print ('mismatch', tfile)
      
    
#     sys.exit()
    '''
    print ('total files', len(file_names))    
    class_dic ={}
    
    for f in file_names:
        cls = f.split('_')[:2]
        cls = cls[0]
        try:
            class_dic[cls] = class_dic[cls]+1
        except:
            class_dic[cls] = 1
    for k in sorted(class_dic.keys()):
        print (k, class_dic[k])
    print (len(class_dic.keys()))
    
def process_data(file_dir='', T=20):
    '''
    this method scaled down each sequence into T time stamps
    '''  
    out_file_name = 'processed_dict_T20_shuffled_newer'
    file_names = os.listdir(file_dir)
    avi_files = [f for f in file_names if '.avi' in f]
    file_names = [f for f in file_names if '.txt' in f]
  
    print ('total files', len(file_names))
    shuffle((file_names))
        
    data_dict = {}
    base_joint_id = 1 # spine mid is origin of body coordinate
    spine_base_id = 0
    head_id = 3
    considered_joints = [8, 9, 10, 11, 23, 24, 4, 5, 6, 7, 21, 22]
    num_joints = len(considered_joints)
    sampling_f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
    sampled_seq = np.zeros([T, num_joints, 3])   # for T sampling only
    for f in file_names:
        print ('processing ', f)
        full_file_path = file_dir+f
        file_r = open(full_file_path,'r')
        flines = [ln for ln in file_r]
        frame_count = len(flines)
        if frame_count==0:
            continue
        sampled_index = sampling_f(T, frame_count)
        print (frame_count)
        print (sampled_index)
        sampled_lines = [flines[l] for l in sampled_index]
        
        sample_data = []
        for l in sampled_lines:
            joint_data = l.split(',')
#             print (joint_data)
            base_joint = np.array([joint_data[base_joint_id].strip().split(' ')[:3]], dtype=np.float32)
            picked_joints = np.array([joint_data[c].strip().split(' ')[:3] for c in considered_joints], dtype=np.float32)
            
#             print (base_joint.shape, picked_joints.shape)
            origin_shifted = np.subtract(picked_joints, base_joint)
            spine_base_data = np.array([joint_data[spine_base_id].strip().split(' ')[:3]], dtype=np.float32)
            head_data = np.array([joint_data[head_id].strip().split(' ')[:3]], dtype=np.float32)
            head2spbase_dist = np.linalg.norm(spine_base_data-head_data)
            #sample_data.append(origin_shifted/head2spbase_dist) # in case of scaling
            sample_data.append(origin_shifted)
        data_dict[f] = np.array(sample_data)
        
        print ('processed ',f,data_dict[f].shape)
    np.save('../data/'+out_file_name, data_dict)
            
            
        
  
def segment_motion(file_loc='', c_joints = [5,6, 9,10], origin_joint_id = 0, T=15):
  print ('segmentation for file', file_loc)
  with open(file_loc, 'r') as f:
    flines = [l for l in f]
    joints_dat = [f.strip().split(',') for f in flines]
    
    rt_wrist_dat = [j_dat[10] for j_dat in joints_dat]
    lt_wrist_dat = [j_dat[6] for j_dat in joints_dat]
    rt_wrist_dat = [r.strip().split(' ')[:3] for r in rt_wrist_dat]
    lt_wrist_dat = [l.strip().split(' ')[:3] for l in lt_wrist_dat]
    rt_wrist_dat = [np.array(list(map(lambda x:float(x) , c))) for c in rt_wrist_dat]
    lt_wrist_dat = [np.array(list(map(lambda x:float(x) , c))) for c in lt_wrist_dat]
    rt_wrist_dist = [np.linalg.norm(rt_wrist_dat[-1]-p)**2 for p in rt_wrist_dat]
    lt_wrist_dist = [np.linalg.norm(lt_wrist_dat[-1]-p)**2 for p in lt_wrist_dat]
    
    
  fig = plt.figure()
  plt.subplots_adjust(hspace = 0.5, wspace = 0)
  ax = fig.add_subplot(3, 1, 1)
  ax.set_title('right wrist motion')
  ax.plot(rt_wrist_dist)
  ax = fig.add_subplot(3, 1, 2)
  ax.set_title('left wrist motion')
  ax.plot(lt_wrist_dist)

  comb_motion = [j[0]+j[1] for j in zip(lt_wrist_dist, rt_wrist_dist)]
  avg_motion = sum(comb_motion)/len(comb_motion)
  seg_cands = []
  motion_mask = comb_motion >= avg_motion/2.
  tflag = False
  ss, se = -1, len(comb_motion)-1
  for i, m in enumerate(motion_mask):
    if tflag==False and m==True:
      tflag=True
      ss = i
    elif tflag==True and m==False:
      tflag = False
      se = i-1
      seg_cands.append((ss, se))
      ss, se = i, len(comb_motion)-1
  if se==len(comb_motion)-1:
    seg_cands.append((ss, se))
  seg_cand_lens = [sum(comb_motion[t[0]:t[1]]) for t in seg_cands]
  seg_start, seg_end = seg_cands[seg_cand_lens.index(max(seg_cand_lens))]
  seg_start = max(0, seg_start-1)
  seg_end = min(len(comb_motion)-1, seg_end+1)
  
  
#   if seg_end-seg_start+1>=30:
#     print (seg_cands)
#     print ('seg and full', seg_start, seg_end, 0, len(comb_motion)-1)
#     print ('seg len', seg_end-seg_start+1)
#     print (seg_start, seg_end)
#     print (motion_mask)
    
  ax = fig.add_subplot(3, 1, 3)
  ax.scatter([seg_start, seg_end], [comb_motion[seg_start], comb_motion[seg_end]])
  ax.plot(comb_motion)
  ax.plot([avg_motion]*len(comb_motion))
  ax.plot([avg_motion/2]*len(comb_motion))
  ax.set_title('combined wrist motion')
  plt.show()
  
  print ('seg st end', seg_start, seg_end)
  # storing needed joint data

  joi_joints_dat = [[jframe[cj] for cj in c_joints] for jframe in joints_dat[seg_start:seg_end]]
  joi_joints_dat = [[jstr.strip().split(' ')[:3] for jstr in jframe] for jframe in joi_joints_dat]
  joi_joints_dat = [[list(map(lambda x:float(x), xyz)) for xyz in jj] for jj in joi_joints_dat]
  
  origin_joint_dat = joints_dat[seg_start:seg_end]
  origin_joint_dat = [tj[origin_joint_id].strip().split(' ')[:3] for tj in origin_joint_dat]
  origin_joint_dat = [list(map(lambda x:float(x), xyz)) for xyz in origin_joint_dat]
  
  print (np.array(joi_joints_dat).shape, np.array(origin_joint_dat).shape)
  
  # killer list comprehension, t is like (considered joint data, base joint data)
  joi_origin_tr = [[[a-b for a, b in zip(cj, t[1])] \
        for cj in t[0]] for t in zip(joi_joints_dat, origin_joint_dat)]
  sampling_f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
  sampled_index = sampling_f(T, len(joi_origin_tr))
  joi_origin_tr = np.array(joi_origin_tr)
  
  print (joi_origin_tr.shape)
  print (joi_origin_tr[sampled_index].shape)
  
  print (sampled_index)
  return seg_start, seg_end, joi_origin_tr[sampled_index]
  
        
def segment_in_folder_files(file_loc='', output_loc=''):
  fnames = os.listdir(file_loc)
  fnames = [f for f in fnames if f.split('.')[-1]=='txt']
  
  class_set = set()
  subject_name = file_loc.split('\\')[-1].split('_')[0]
  print (subject_name)
  for fid, fn in enumerate(fnames):
    sk_file = os.path.join(file_loc, fn)
    vid_file = os.path.join(file_loc, fn.split('_')[0]+'_rgb.avi')
    class_name = fn.split('_')[0]
    print (os.path.exists(sk_file), os.path.exists(vid_file))
    if os.path.exists(sk_file) and os.path.exists(vid_file):
      print (fid, class_name, subject_name)
      existing_files = os.listdir(output_loc)
      existing_files = [e for e in existing_files if set([class_name, subject_name]).issubset(e.split('_'))]
      len_existing = len(existing_files)//2
      print ('existing files', len_existing)
      class_set.add(class_name)
      if len_existing >= 21:
        print ('all files segmented of ',class_name, subject_name)
        continue
      
      segment_from_whole_v3(vid_file, sk_file, output_loc, \
            already_files=len_existing, saved_file_name='{}_{}_'.format(class_name, subject_name))
  print (list(class_set))

'''
Given a subject and a class name plots distance for four joints
to see if distance stats is less for some subjects
'''
def distance_plot(subj='', gest_class=''):
  file_loc = 'C:\\DATA\\segmented_data'
  for i in range(1, 25):
    sk_file = '{}\\{}\\{}_{}_{}_bodyData.txt'.format(file_loc, subj, gest_class, subj, i)


'''
xtract hand shape based on wrist joint
'''
def xract_hand_shape(vid_file_loc='', sk_file_loc='', w_sizse=(50,50)):
  print ('here')
  vid = cv2.VideoCapture(vid_file_loc)
  sk_file = open(sk_file_loc,'r')
  file_lines = [line for line in sk_file]
  s, e = 0, len(file_lines)

  temp_file_lines = file_lines.copy()
  dist_list = []
  file_lines = []
  dist_id = 0
  while dist_id<len(temp_file_lines)-1:
    curr_frame = temp_file_lines[dist_id]
    next_frame = temp_file_lines[dist_id+1]
    
    if curr_frame in ['start\n', 'end\n']:
      curr_frame = temp_file_lines[dist_id-1]
    if next_frame in ['start\n', 'end\n']:
      dist_id += 1
      continue
    
    curr_frame_spmid = (curr_frame.strip().split(',')[1].strip().split(' ')[:3])
    next_frame_spmid = (next_frame.strip().split(',')[1].strip().split(' ')[:3])
    spmid_dist = sum([(float(t[0])-float(t[1]))**2 for t in zip(next_frame_spmid, curr_frame_spmid)])
    
    if spmid_dist>0.2:
      file_lines.append(curr_frame)
      dist_id += 2
    else:
      file_lines.append(curr_frame)
      dist_id += 1
    
  print ('avi frame count', vid.get(cv2.CAP_PROP_FRAME_COUNT))
  print ('sk file line no. ', len(file_lines))
  
  f_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)//2) - 200       # for tagging texts
  
  base_joint = 1
  considered_joints = [1, 8, 9, 10, 11, 23, 24, 4, 5, 6, 7, 21, 22]
  
  vwidth = vid.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
  vheight = vid.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
  print (vwidth, vheight)
  
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     vid_writer = cv2.VideoWriter(vid_out, fourcc, 10.0, (int(vwidth*.5),int(vheight*.5)), True)
  
  read_cnt = 0
  while read_cnt<s:
    ret, frame = vid.read()
    read_cnt += 1
  sample_no = int(vid_file_loc.split('\\')[-1].split('.')[0].split('_')[-2])
#     sample_no = 0

  lhands, rhands = [], []
  sk_frames = []
  for i, line in enumerate(file_lines):
    joint_dat = line.strip().split(',')
    if ('start' in joint_dat or 'end' in joint_dat):
        continue
    ret, frame = vid.read()
    tempframe = frame.copy()
    print (joint_dat)
    print (frame.shape)
    resized_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    joint_pos_list = []
    frame = np.ones(frame.shape, np.uint8)*200
    for jn, joint_details in enumerate(joint_dat):
        joint_details = joint_details.strip().split(' ')
        print (joint_details)
        joint_x = int(joint_details[-4])
        joint_y = int(joint_details[-3])
        joint_pos_list.append((joint_x, joint_y))
        if jn==6:
          cv2.circle(frame, (joint_x, joint_y), 20, (0,10,15), -1)
        elif jn==10:
          cv2.circle(frame, (joint_x, joint_y), 20, (0,10,15), -1)
        else:
          cv2.circle(frame, (joint_x, joint_y), 20, (0,10,15), -1)
    
    cw, ch = w_sizse
    joint_details = joint_dat[7].strip().split(' ')
    joint_x = int(joint_details[-4])
    joint_y = int(joint_details[-3])
    sx, sy = joint_x - ch//2, joint_y - cw//2
    print (sx, sy, sx+ch, sy+cw)
    left_hand = tempframe[sy:sy+cw,sx:sx+ch,:]
    
    joint_details = joint_dat[11].strip().split(' ')
    joint_x = int(joint_details[-4])
    joint_y = int(joint_details[-3])
    sx, sy = joint_x - ch//2, joint_y - cw//2
    print (sx, sy, sx+ch, sy+cw)
    right_hand = tempframe[sy:sy+cw,sx:sx+ch,:]
    
    hands = np.concatenate((left_hand, right_hand), axis=1)
    print (hands.shape)
    
    skeletoned_frame = connect_joint(frame, joint_pos_list)
    resized_sk_frame = cv2.resize(skeletoned_frame, (0,0), fx=0.5, fy=0.5)
    
    lhands.append(left_hand)
    rhands.append(right_hand)
    sk_frames.append(cv2.resize(skeletoned_frame, (0,0), fx=0.3, fy=0.4))
    
    hands = np.concatenate((lhands[-1], rhands[-1]), axis=0)
    print (hands.shape)
    cv2.imshow('hands',hands)
    k = cv2.waitKey(-1)
    cv2.imshow('sked', sk_frames[-1])
    k = cv2.waitKey(-1)
  
  
       
#       cv2.putText(resized_sk_frame, 'sign:'+os.path.basename(vid_file_loc).split('_')[0]+' || sample '+str(sample_no+1)+' || frame_no_'+str(i),(f_width//4,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 23, 22), 2)
  
      
def main():    
  
  sk_file_loc ='newWordLs/Kinect/wakeup_bodyData.txt'
  rgb_file_loc = 'newWordLs/Kinect/wakeup_rgb.avi'
  draw_on_rgb(rgb_file_loc, sk_file_loc)

if __name__ == '__main__':
    main()
