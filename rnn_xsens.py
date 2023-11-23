"""
same as rnn_ntf_pos.py but with
weighted joints
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict
import networkx as nx
import scipy.linalg as sclinalg

import os, sys, time, subprocess
import numpy as np
import math

from common import utils
from common import bvh_tools as bvh
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, slerp
from common.pose_renderer import PoseRenderer

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Mocap Data
"""

# mocap settings
mocap_data_path = "D:/Data/mocap/muriel_athens_xsens_awinda_april_2023/MurielDancing-005.bvh"
mocap_valid_frame_range = [ 50, 6400 ]
joint_loss_weights = [ 1.0 ] * 28
mocap_fps = 50

# load mocap data
bvh_tools = bvh.BVH_Tools()
mocap_tools = mocap.Mocap_Tools()

bvh_data = bvh_tools.load(mocap_data_path)
mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])

# get mocap data
pose_sequence = mocap_data["motion"]["rot_local"].astype(np.float32)

total_sequence_length = pose_sequence.shape[0]
joint_count = pose_sequence.shape[1]
joint_dim = pose_sequence.shape[2]
pose_dim = joint_count * joint_dim

offsets = mocap_data["skeleton"]["offsets"].astype(np.float32)
parents = mocap_data["skeleton"]["parents"]
children = mocap_data["skeleton"]["children"]

# create edge list
def get_edge_list(children):
    edge_list = []

    for parent_joint_index in range(len(children)):
        for child_joint_index in children[parent_joint_index]:
            edge_list.append([parent_joint_index, child_joint_index])
    
    return edge_list

edge_list = get_edge_list(children)

"""
Dataset
"""

seq_input_length = 64
seq_output_length = 10 # this is only used for non-teacher forcing scenarios
X = []
y = []

for pI in range(total_sequence_length - seq_input_length - seq_output_length - 1):
    X_sample = pose_sequence[pI:pI+seq_input_length]
    X.append(X_sample.reshape((seq_input_length, pose_dim)))
    
    Y_sample = pose_sequence[pI+seq_input_length:pI+seq_input_length+seq_output_length]
    y.append(Y_sample.reshape((seq_output_length, pose_dim)))

X = np.array(X)
y = np.array(y)

X = torch.from_numpy(X)
y = torch.from_numpy(y)

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx, ...], self.y[idx, ...]

full_dataset = SequenceDataset(X, y)


test_percentage = 0.1
test_size = int(test_percentage * len(full_dataset))
train_size = len(full_dataset) - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


"""
Reccurent Model
"""

class Reccurent(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_count):
        super(Reccurent, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_count = layer_count
        self.output_dim = output_dim
            
        rnn_layers = []
        
        rnn_layers.append(("rnn", nn.LSTM(self.input_dim, self.hidden_dim, self.layer_count, batch_first=True)))
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        dense_layers = []
        dense_layers.append(("dense", nn.Linear(self.hidden_dim, self.output_dim)))
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
    
    def forward(self, x):
        x, (_, _) = self.rnn_layers(x)
        
        x = x[:, -1, :] # only last time step 
        x = self.dense_layers(x)
        
        return x

rnn = Reccurent(pose_dim, 512, pose_dim, 2).to(device)
print(rnn)

# test Reccurent model

batch_x, _ = next(iter(train_loader))
batch_x = batch_x.to(device)

print(batch_x.shape)

test_y2 = rnn(batch_x)

print(test_y2.shape)

"""
# load model weights if previously stored
PATH = "results_lstm_ntf_pos_weighted_joints/weights/rnn_weights_epoch_400"
rnn.load_state_dict(torch.load(PATH))
"""

"""
Training
"""


learning_rate = 1e-4
norm_loss_scale = 0.1
pos_loss_scale = 0.1
quat_loss_scale = 0.9
teacher_forcing_prob = 0.0
model_save_interval = 50
save_weights = True
epochs = 400

optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1) # reduce the learning every 20 epochs by a factor of 10

joint_loss_weights = torch.tensor(joint_loss_weights, dtype=torch.float32)
joint_loss_weights = joint_loss_weights.reshape(1, 1, -1).to(device)

def norm_loss(yhat):
    _yhat = yhat.view(-1, 4)
    _norm = torch.norm(_yhat, dim=1)
    _diff = (_norm - 1.0) ** 2
    _loss = torch.mean(_diff)
    return _loss

def forward_kinematics(rotations, root_positions):
    """
    Perform forward kinematics using the given trajectory and local rotations.
    Arguments (where N = batch size, L = sequence length, J = number of joints):
     -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
     -- root_positions: (N, L, 3) tensor describing the root joint positions.
    """

    assert len(rotations.shape) == 4
    assert rotations.shape[-1] == 4
    
    toffsets = torch.tensor(offsets).to(device)
    
    positions_world = []
    rotations_world = []

    expanded_offsets = toffsets.expand(rotations.shape[0], rotations.shape[1], offsets.shape[0], offsets.shape[1])

    # Parallelize along the batch and time dimensions
    for jI in range(offsets.shape[0]):
        if parents[jI] == -1:
            positions_world.append(root_positions)
            rotations_world.append(rotations[:, :, 0])
        else:
            positions_world.append(qrot(rotations_world[parents[jI]], expanded_offsets[:, :, jI]) \
                                   + positions_world[parents[jI]])
            if len(children[jI]) > 0:
                rotations_world.append(qmul(rotations_world[parents[jI]], rotations[:, :, jI]))
            else:
                # This joint is a terminal node -> it would be useless to compute the transformation
                rotations_world.append(None)

    return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

def pos_loss(y, yhat):
    
    #print("pos_loss")
    #print("y s ", y.shape)
    #print("yhat s ", yhat.shape)
    
    # y and yhat shapes: batch_size, seq_length, pose_dim

    # normalize tensors
    _yhat = yhat.view(-1, 4)

    _yhat_norm = nn.functional.normalize(_yhat, p=2, dim=1)
    _y_rot = y.view((y.shape[0], y.shape[1], -1, 4))
    _yhat_rot = _yhat.view((y.shape[0], y.shape[1], -1, 4))
    
    #print("_y_rot s ", _y_rot.shape)
    #print("_yhat_rot s ", _yhat_rot.shape)

    zero_trajectory = torch.zeros((y.shape[0], y.shape[1], 3), dtype=torch.float32, requires_grad=True).to(device)

    _y_pos = forward_kinematics(_y_rot, zero_trajectory)
    _yhat_pos = forward_kinematics(_yhat_rot, zero_trajectory)
    
    #print("_y_pos s ", _y_pos.shape)
    #print("_yhat_pos s ", _yhat_pos.shape)

    _pos_diff = torch.norm((_y_pos - _yhat_pos), dim=3)
    
    #print("_pos_diff s ", _pos_diff.shape)
    
    _pos_diff_weighted = _pos_diff * joint_loss_weights
    
    _loss = torch.mean(_pos_diff_weighted)

    return _loss

def quat_loss(y, yhat):
    
    #print("quat_loss")
    #print("y s ", y.shape)
    #print("yhat s ", yhat.shape)
    
    # y and yhat shapes: batch_size, seq_length, pose_dim
    
    # normalize quaternion
    
    _y = y.view((-1, 4))
    _yhat = yhat.view((-1, 4))
    _yhat_norm = nn.functional.normalize(_yhat, p=2, dim=1)
    
    #print("_y s ", _y.shape)
    #print("_yhat_norm s ", _yhat_norm.shape)
    
    # inverse of quaternion: https://www.mathworks.com/help/aeroblks/quaternioninverse.html
    _yhat_inv = _yhat_norm * torch.tensor([[1.0, -1.0, -1.0, -1.0]], dtype=torch.float32).to(device)

    # calculate difference quaternion
    _diff = qmul(_yhat_inv, _y)
    # length of complex part
    _len = torch.norm(_diff[:, 1:], dim=1)
    # atan2
    _atan = torch.atan2(_len, _diff[:, 0])
    # abs
    _abs = torch.abs(_atan)
    
    _abs = _abs.reshape(-1, 1, joint_count)
    
    #print("_abs s ", _abs.shape)
    
    _abs_weighted = _abs * joint_loss_weights
    
    _loss = torch.mean(_abs_weighted)   
    return _loss

# autoencoder loss function
def loss(y, yhat):
    _norm_loss = norm_loss(yhat)
    _pos_loss = pos_loss(y, yhat)
    _quat_loss = quat_loss(y, yhat)
    
    _total_loss = 0.0
    _total_loss += _norm_loss * norm_loss_scale
    _total_loss += _pos_loss * pos_loss_scale
    _total_loss += _quat_loss * quat_loss_scale
    
    return _total_loss, _norm_loss, _pos_loss, _quat_loss

def train_step(pose_sequences, target_poses, teacher_forcing):

    #print("ar_train_step")    
    #print("teacher_forcing ", teacher_forcing)
    #print("pose_sequences s ", pose_sequences.shape)
    #print("target_poses s ", target_poses.shape)

    #_input_poses = pose_sequences.detach().clone()    
    _input_poses = pose_sequences  
    output_poses_length = target_poses.shape[1]
    
    #print("output_poses_length ", output_poses_length)
    
    _loss_total = torch.zeros((1)).to(device)
    _norm_loss_total = torch.zeros((1)).to(device)
    _pos_loss_total = torch.zeros((1)).to(device)
    _quat_loss_total = torch.zeros((1)).to(device)
    
    for o_i in range(1, output_poses_length):
        
        #print("_input_poses s ", _input_poses.shape)
        
        _pred_poses = rnn(_input_poses)
        _pred_poses = torch.unsqueeze(_pred_poses, axis=1)
        
        #print("_pred_poses s ", _pred_poses.shape)
        
        _target_poses = target_poses[:,o_i,:].detach().clone()
        _target_poses = torch.unsqueeze(_target_poses, axis=1)

        #print("_target_poses s ", _target_poses.shape)

        _loss, _norm_loss, _pos_loss, _quat_loss = loss(_target_poses, _pred_poses) 
        
        
        #print("_ar_loss s ", _ar_loss.shape)
        #print("_ar_loss_total s ", _ar_loss_total.shape)

        #print("_ae_pos_loss ", _ae_pos_loss)
    
        # Backpropagation
        optimizer.zero_grad()
        _loss.backward()
        optimizer.step()
        
        _loss_total += _loss
        _norm_loss_total += _norm_loss_total
        _pos_loss_total += _pos_loss
        _quat_loss_total += _quat_loss
        
        # shift input pose seqeunce one pose to the right
        # remove pose from beginning input pose sequence
        # detach necessary to avoid error concerning running backprob a second time
        _input_poses = _input_poses[:, 1:, :].detach().clone()
        _target_poses = _target_poses.detach().clone()
        _pred_poses = _pred_poses.detach().clone()
        
        # add predicted or target pose to end of input pose sequence
        if teacher_forcing == True:
            _input_poses = torch.concat((_input_poses, _target_poses), axis=1)
        else:
            #_pred_poses = torch.reshape(_pred_poses, (_pred_poses.shape[0], 1, joint_count, joint_dim))
            _input_poses = torch.cat((_input_poses, _pred_poses), axis=1)
            
        #print("_input_poses s ", _input_poses.shape)

        
        #print("_input_poses 2 s ", _input_poses.shape)
        
    _loss_total /= o_i
    _norm_loss_total /= o_i
    _pos_loss_total /= o_i
    _quat_loss_total /= o_i
    
    #print("_ar_loss_total mean s ", _ar_loss_total.shape)
    
    #return _ar_loss, _ar_norm_loss, _ar_quat_loss
    
    return _loss_total, _norm_loss_total, _pos_loss_total, _quat_loss_total

def test_step(pose_sequences, target_poses, teacher_forcing):
    
    rnn.eval()

    #print("ar_train_step")    
    #print("teacher_forcing ", teacher_forcing)
    #print("pose_sequences s ", pose_sequences.shape)
    #print("target_poses s ", target_poses.shape)

    #_input_poses = pose_sequences.detach().clone()    
    _input_poses = pose_sequences  
    output_poses_length = target_poses.shape[1]
    
    #print("output_poses_length ", output_poses_length)
    
    _loss_total = torch.zeros((1)).to(device)
    _norm_loss_total = torch.zeros((1)).to(device)
    _pos_loss_total = torch.zeros((1)).to(device)
    _quat_loss_total = torch.zeros((1)).to(device)
    
    for o_i in range(1, output_poses_length):
        
        with torch.no_grad():
            #print("_input_poses s ", _input_poses.shape)
        
            _pred_poses = rnn(_input_poses)
            _pred_poses = torch.unsqueeze(_pred_poses, axis=1)
        
            #print("_pred_poses s ", _pred_poses.shape)
        
            _target_poses = target_poses[:,o_i,:].detach().clone()
            _target_poses = torch.unsqueeze(_target_poses, axis=1)

            #print("_target_poses s ", _target_poses.shape)

            _loss, _norm_loss, _pos_loss, _quat_loss = loss(_target_poses, _pred_poses) 
        
        
            #print("_ar_loss s ", _ar_loss.shape)
            #print("_ar_loss_total s ", _ar_loss_total.shape)

            #print("_ae_pos_loss ", _ae_pos_loss)
    
            _loss_total += _loss
            _norm_loss_total += _norm_loss_total
            _pos_loss_total += _pos_loss
            _quat_loss_total += _quat_loss
        
        # shift input pose seqeunce one pose to the right
        # remove pose from beginning input pose sequence
        # detach necessary to avoid error concerning running backprob a second time
        _input_poses = _input_poses[:, 1:, :].detach().clone()
        _target_poses = _target_poses.detach().clone()
        _pred_poses = _pred_poses.detach().clone()
        
        # add predicted or target pose to end of input pose sequence
        if teacher_forcing == True:
            _input_poses = torch.concat((_input_poses, _target_poses), axis=1)
        else:
            #_pred_poses = torch.reshape(_pred_poses, (_pred_poses.shape[0], 1, joint_count, joint_dim))
            _input_poses = torch.cat((_input_poses, _pred_poses), axis=1)
            
        #print("_input_poses s ", _input_poses.shape)

        
        #print("_input_poses 2 s ", _input_poses.shape)
        
    _loss_total /= o_i
    _norm_loss_total /= o_i
    _pos_loss_total /= o_i
    _quat_loss_total /= o_i
    
    #print("_ar_loss_total mean s ", _ar_loss_total.shape)
    
    #return _ar_loss, _ar_norm_loss, _ar_quat_loss
    
    rnn.train()
    
    return _loss_total, _norm_loss_total, _pos_loss_total, _quat_loss_total


def train(train_dataloader, test_dataloader, epochs):
    
    loss_history = {}
    loss_history["train"] = []
    loss_history["test"] = []
    loss_history["norm"] = []
    loss_history["pos"] = []
    loss_history["quat"] = []

    for epoch in range(epochs):
        start = time.time()
        
        _train_loss_per_epoch = []
        _norm_loss_per_epoch = []
        _pos_loss_per_epoch = []
        _quat_loss_per_epoch = []

        for train_batch in train_dataloader:
            input_pose_sequences = train_batch[0].to(device)
            target_poses = train_batch[1].to(device)
            
            use_teacher_forcing = np.random.uniform() < teacher_forcing_prob
            
            _loss, _norm_loss, _pos_loss, _quat_loss = train_step(input_pose_sequences, target_poses, use_teacher_forcing)
            
            _loss = _loss.detach().cpu().numpy()
            _norm_loss = _norm_loss.detach().cpu().numpy()
            _pos_loss = _pos_loss.detach().cpu().numpy()
            _quat_loss = _quat_loss.detach().cpu().numpy()
            
            _train_loss_per_epoch.append(_loss)
            _norm_loss_per_epoch.append(_norm_loss)
            _pos_loss_per_epoch.append(_pos_loss)
            _quat_loss_per_epoch.append(_quat_loss)

        _train_loss_per_epoch = np.mean(np.array(_train_loss_per_epoch))
        _norm_loss_per_epoch = np.mean(np.array(_norm_loss_per_epoch))
        _pos_loss_per_epoch = np.mean(np.array(_pos_loss_per_epoch))
        _quat_loss_per_epoch = np.mean(np.array(_quat_loss_per_epoch))

        _test_loss_per_epoch = []
        
        for test_batch in test_dataloader:
            input_pose_sequences = train_batch[0].to(device)
            target_poses = train_batch[1].to(device)
            
            use_teacher_forcing = np.random.uniform() < teacher_forcing_prob
            
            _loss, _, _, _ = test_step(input_pose_sequences, target_poses, use_teacher_forcing)
            #_loss, _, _ = test_step(input_pose_sequences, target_poses)
            
            _loss = _loss.detach().cpu().numpy()
            
            _test_loss_per_epoch.append(_loss)
        
        _test_loss_per_epoch = np.mean(np.array(_test_loss_per_epoch))
        
        if epoch % model_save_interval == 0 and save_weights == True:
            torch.save(rnn.state_dict(), "results/weights/rnn_weights_epoch_{}".format(epoch))
        
        loss_history["train"].append(_train_loss_per_epoch)
        loss_history["test"].append(_test_loss_per_epoch)
        loss_history["norm"].append(_norm_loss_per_epoch)
        loss_history["pos"].append(_pos_loss_per_epoch)
        loss_history["quat"].append(_quat_loss_per_epoch)
        
        scheduler.step()
        
        print ('epoch {} : train: {:01.4f} test: {:01.4f} norm {:01.4f} pos {:01.4f} quat {:01.4f} time {:01.2f}'.format(epoch + 1, _train_loss_per_epoch, _test_loss_per_epoch, _norm_loss_per_epoch, _pos_loss_per_epoch, _quat_loss_per_epoch, time.time()-start))
    
    return loss_history

# fit model
loss_history = train(train_loader, test_loader, epochs)

# save history
utils.save_loss_as_csv(loss_history, "results/histories/rnn_history_{}.csv".format(epochs))
utils.save_loss_as_image(loss_history, "results/histories/rnn_history_{}.png".format(epochs))

# save model weights
torch.save(rnn.state_dict(), "results/weights/rnn_weights_epoch_{}".format(epochs))

# inference and rendering 
poseRenderer = PoseRenderer(edge_list)

# visualization settings
view_ele = 90.0
view_azi = -90.0
view_line_width = 4.0
view_size = 8.0


# create ref pose sequence
def create_ref_sequence_anim(start_pose_index, pose_count, file_name):
    
    print("start_pose_index ", start_pose_index)
    print("pose_count ", pose_count)
    
    start_pose_index = max(start_pose_index, seq_input_length)
    pose_count = min(pose_count, total_sequence_length - start_pose_index)
    
    print("total_sequence_length ", total_sequence_length)
    print("start_pose_index ", start_pose_index)
    print("pose_count ", pose_count)
    
    sequence_excerpt = pose_sequence[start_pose_index:start_pose_index + pose_count, :]
    sequence_excerpt = np.reshape(sequence_excerpt, (pose_count, joint_count, joint_dim))
    
    print("sequence_excerpt s ", sequence_excerpt.shape)

    sequence_excerpt = torch.tensor(np.expand_dims(sequence_excerpt, axis=0)).to(device)
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32)).to(device)
    
    skel_sequence = forward_kinematics(sequence_excerpt, zero_trajectory)
    
    print("skel_sequence s ", skel_sequence.shape)

    skel_sequence = np.squeeze(skel_sequence.cpu().numpy())
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0)


def create_pred_sequence_anim(start_pose_index, pose_count, file_name):
    rnn.eval()
    
    start_pose_index = max(start_pose_index, seq_input_length)
    pose_count = min(pose_count, total_sequence_length - start_pose_index)
    
    start_seq = pose_sequence[start_pose_index - seq_input_length:start_pose_index, :]
    start_seq = torch.from_numpy(start_seq).to(device)
    start_seq = torch.reshape(start_seq, (seq_input_length, pose_dim))
    
    next_seq = start_seq
    
    pred_poses = []
    
    for i in range(pose_count):
        
        with torch.no_grad():
            pred_pose = rnn(torch.unsqueeze(next_seq, axis=0))

        # normalize pred pose
        pred_pose = torch.squeeze(pred_pose)
        pred_pose = pred_pose.reshape((-1, 4))
        pred_pose = nn.functional.normalize(pred_pose, p=2, dim=1)
        pred_pose = pred_pose.reshape((1, pose_dim))

        pred_poses.append(pred_pose)
    
        #print("next_seq s ", next_seq.shape)
        #print("pred_pose s ", pred_pose.shape)

        next_seq = torch.cat([next_seq[1:,:], pred_pose], axis=0)

    pred_poses = torch.cat(pred_poses, dim=0)
    pred_poses = pred_poses.reshape((1, pose_count, joint_count, joint_dim))


    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)
    
    skel_poses = forward_kinematics(pred_poses, zero_trajectory)
    
    skel_poses = skel_poses.detach().cpu().numpy()
    skel_poses = np.squeeze(skel_poses)
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_poses)
    pose_images = poseRenderer.create_pose_images(skel_poses, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)

    pose_images[0].save(file_name, save_all=True, append_images=pose_images[1:], optimize=False, duration=33.0, loop=0) 

    rnn.train()

def create_pred_sequence(start_pose_index, pose_count):
    rnn.eval()
    
    start_pose_index = max(start_pose_index, seq_input_length)
    pose_count = min(pose_count, total_sequence_length - start_pose_index)
    
    start_seq = pose_sequence[start_pose_index - seq_input_length:start_pose_index, :]
    start_seq = torch.from_numpy(start_seq).to(device)
    start_seq = torch.reshape(start_seq, (seq_input_length, pose_dim))
    
    next_seq = start_seq
    
    pred_poses = []
    
    for i in range(pose_count):
        
        with torch.no_grad():
            pred_pose = rnn(torch.unsqueeze(next_seq, axis=0))

        # normalize pred pose
        pred_pose = torch.squeeze(pred_pose)
        pred_pose = pred_pose.reshape((-1, 4))
        pred_pose = nn.functional.normalize(pred_pose, p=2, dim=1)
        pred_pose = pred_pose.reshape((1, pose_dim))

        pred_poses.append(pred_pose)
    
        #print("next_seq s ", next_seq.shape)
        #print("pred_pose s ", pred_pose.shape)

        next_seq = torch.cat([next_seq[1:,:], pred_pose], axis=0)

    pred_poses = torch.cat(pred_poses, dim=0)
    pred_poses = pred_poses.reshape((pose_count, joint_count, joint_dim))

    rnn.train()
    
    return pred_poses.detach().cpu().numpy()

seq_start_pose_index = 1500
seq_pose_count = 1000

# export as gif animations

create_ref_sequence_anim(seq_start_pose_index, seq_pose_count, "results/anims/ref_range_{}_{}.gif".format(seq_start_pose_index, seq_pose_count))
create_pred_sequence_anim(seq_start_pose_index, seq_pose_count, "results/anims/pred_ep_{}_range_{}_{}.gif".format(epochs, seq_start_pose_index, seq_pose_count))

# export as bvh files

pred_poses = create_pred_sequence(seq_start_pose_index, seq_pose_count)

pred_poses.shape

pred_dataset = {}
pred_dataset["frame_rate"] = mocap_data["frame_rate"]
pred_dataset["rot_sequence"] = mocap_data["rot_sequence"]
pred_dataset["skeleton"] = mocap_data["skeleton"]
pred_dataset["motion"] = {}
pred_dataset["motion"]["pos_local"] = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), seq_pose_count, axis=0)
pred_dataset["motion"]["rot_local"] = pred_poses
pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler(pred_dataset["motion"]["rot_local"], pred_dataset["rot_sequence"])

pred_bvh = mocap_tools.mocap_to_bvh(pred_dataset)

bvh_tools.write(pred_bvh, "results/anims/pred_{}_{}_epoch_{}.bvh".format(seq_start_pose_index, seq_pose_count, epochs))
