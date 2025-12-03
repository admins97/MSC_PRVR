import json
import torch
import torch.utils.data as data
import numpy as np
import re
import h5py
import os
import pickle
import math
import torch.nn.functional as F

def load_jsonl(filename):
    import json
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    return vid_id


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()


def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list

def token_merge(x, target_len, merge_size=None, merged_frame_idx=None):
    """
    x: Tensor of shape [N, D]
    target_len: int, number of rows after reduction
    Keeps the merged results in the same relative pair positions.
    """
    N, D = x.shape
    assert N % 2 == 0, "Input must have even number of rows"
    num_pairs = N // 2
    merge_count = N - target_len

    if merge_size is None:
        merge_size = torch.ones(N, 1)

    x_pairs = x.view(num_pairs, 2, D)  # [num_pairs, 2, D]
    size_pairs = merge_size.view(num_pairs, 2, 1)
    
    a, b = x_pairs[:, 0], x_pairs[:, 1]
    sim = (a * b).sum(dim=1)  # [num_pairs] 64

    sorted_indices = torch.argsort(sim, descending=True) # 64
    merge_indices = set(sorted_indices[:merge_count].tolist()) # 

    result = []
    new_merge_size = []
    for i in range(num_pairs):
        if i in merge_indices:
            w1, w2 = size_pairs[i, 0], size_pairs[i, 1]
            merged = (x_pairs[i, 0] * w1 + x_pairs[i, 1] * w2) / (w1 + w2)

            anchor_label = merged_frame_idx[torch.where(merged_frame_idx == i*2)[0]]
            merged_frame_idx[torch.where(merged_frame_idx == (i*2+1))[0]] = anchor_label if len(anchor_label) == 1 else anchor_label[0]

            result.append(merged.unsqueeze(0))
            new_merge_size.append(w1 + w2)
        else:
            result.append(x_pairs[i][0].unsqueeze(0))
            result.append(x_pairs[i][1].unsqueeze(0))
            new_merge_size.append(size_pairs[i, 0])
            new_merge_size.append(size_pairs[i, 1])
    merged_frame_idx = merged_frame_idx.tolist()
    class_list = list(set(merged_frame_idx))
    class_map = {}
    adjusted_label = 0
    num_classes = len(class_list)
    for cls_ in class_list:
        if cls_ in merged_frame_idx:
            class_map[cls_] = adjusted_label
            adjusted_label += 1
    adjusted_merged_frame_idx = torch.tensor([class_map[label] for label in merged_frame_idx])
    
    merged = torch.cat(result, dim=0)  # [target_len, D]
    merged_size = torch.cat(new_merge_size, dim=0)
    
    return merged, merged_size, adjusted_merged_frame_idx #.to(torch.float16)

def average_to_fixed_length(visual_input, map_size, clip=None):
    visual_input = torch.from_numpy(visual_input)
    num_sample_clips = map_size
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips

    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))

    new_visual_input = []

    for i in range(num_sample_clips):

        s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)#.numpy()
    
    
    if clip is not None:
        merge_size = None
        merged_frame_idx = torch.arange(128) # N = 128    
        for target in [80, 52, 32]:
            new_visual_input, merge_size, merged_frame_idx = token_merge(new_visual_input, target, merge_size, merged_frame_idx)
            # import pdb
            # pdb.set_trace()
        new_visual_input = new_visual_input.numpy()
        merge_size = merge_size.numpy()
        merged_frame_idx = merged_frame_idx.detach().numpy()
        return new_visual_input, merge_size, merged_frame_idx
    else:
        return new_visual_input.numpy()


def uniform_feature_sampling(features, max_len):
    num_clips = features.shape[0]
    if max_len is None or num_clips <= max_len:
        return features
    idxs = np.arange(0, max_len + 1, 1.0) / max_len * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_features = []
    for i in range(max_len):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_features.append(np.mean(features[s_idx:e_idx], axis=0))
        else:
            new_features.append(features[s_idx])
    new_features = np.asarray(new_features)
    return new_features


def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)


def collate_train(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    clip_video_features, frame_video_features, captions, idxs, cap_ids, video_ids, st, ed, merge_size, merged_frame_idx = zip(*data)

    # videos
    clip_videos = torch.cat(clip_video_features, dim=0).float()
    merge_size = torch.cat(merge_size, dim=0)
    merged_frame_idx = torch.cat(merged_frame_idx, dim=0)
    
    st = torch.tensor([item for sub in st for item in sub])
    ed = torch.tensor([item for sub in ed for item in sub])

    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    # captions
    feat_dim = captions[0][0].shape[-1]

    merge_captions = []
    all_lengths = []
    labels = []

    for index, caps in enumerate(captions):
        labels.extend(index for i in range(len(caps)))
        all_lengths.extend(len(cap) for cap in caps)
        merge_captions.extend(cap for cap in caps)

    target = torch.zeros(len(all_lengths), max(all_lengths), feat_dim)
    words_mask = torch.zeros(len(all_lengths), max(all_lengths))

    for index, cap in enumerate(merge_captions):
        end = all_lengths[index]
        target[index, :end, :] = cap[:end, :]
        words_mask[index, :end] = 1.0

    return dict(clip_video_features=clip_videos,
                frame_video_features=frame_videos,
                videos_mask=videos_mask,
                text_feat=target,
                text_mask=words_mask,
                text_labels=labels,
                st_indices=st,
                ed_indices=ed,
                merge_size=merge_size,
                merged_frame_label = merged_frame_idx
                )


def collate_frame_val(data):
    clip_video_features, frame_video_features, idxs, video_ids, merge_size = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    # videos
    clip_videos = torch.cat(clip_video_features, dim=0).float()
    merge_size = torch.cat(merge_size, dim=0)

    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    return clip_videos, frame_videos, videos_mask, idxs, video_ids, merge_size


def collate_text_val(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, idxs, cap_ids, st, ed = zip(*data)

    st = torch.tensor(st, dtype=torch.int32)
    ed = torch.tensor(ed, dtype=torch.int32)

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths), captions[0].shape[-1])
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None

    return target, words_mask, idxs, cap_ids, st, ed


class Dataset4PRVR(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, visual_feat, text_feat_path, cfg, video2frames=None, is_clip=False,
                 path_query_json=None):
        # Captions
        self.cfg = cfg
        self.captions = {}
        self.cap_ids = []
        self.video_ids = []
        self.vid_caps = {}
        self.video2frames = video2frames
        self.is_clip = is_clip
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                if video_id not in self.video_ids:
                    self.video_ids.append(video_id)

                if video_id in self.vid_caps:
                    self.vid_caps[video_id].append(cap_id)

                else:
                    self.vid_caps[video_id] = []
                    self.vid_caps[video_id].append(cap_id)

        if is_clip:
            self.visual_feat = h5py.File(visual_feat, 'r')
        else:
            self.visual_feat = visual_feat
        self.cap2queryinfo = None
        if path_query_json is not None:
            query_json = load_jsonl(path_query_json)
            self.cap2queryinfo = {}

            if cfg['collection'] == 'qvhighlight':
                for i in range(len(query_json)):
                    one_data = query_json[i]
                    cap = one_data['query'].rstrip()
                    # self.cap2queryinfo[cap] = [one_data['duration'], one_data['ts'][0], one_data['ts'][1]]
            else:
                for i in range(len(query_json)):
                    one_data = query_json[i]
                    cap = one_data['desc'].rstrip()
                    self.cap2queryinfo[cap] = [one_data['duration'], one_data['ts'][0], one_data['ts'][1]]

            self.use_st_ed = True

        self.text_feat_path = text_feat_path

        self.map_size = cfg['map_size']
        self.max_ctx_len = cfg['max_ctx_l']
        self.max_desc_len = cfg['max_desc_l']
        self.collection = cfg['collection']

        self.open_file = False
        self.length = len(self.vid_caps)

    def __getitem__(self, index):

        if self.open_file:
            self.open_file = True
        else:
            self.text_feat = h5py.File(self.text_feat_path, 'r')
            self.open_file = True

        video_id = self.video_ids[index]
        cap_ids = self.vid_caps[video_id]

        # video
        if self.is_clip:
            if self.collection == 'activitynet':
                video_id = video_id[2:]
            frame_vecs = self.visual_feat[video_id][...]

        else:
            frame_list = self.video2frames[video_id]
            frame_vecs = []
            for frame_id in frame_list:
                frame_vecs.append(self.visual_feat.read_one(frame_id))
            frame_vecs = np.array(frame_vecs)

        clip_video_feature, merge_size, merged_frame_idx = average_to_fixed_length(frame_vecs, self.map_size, clip=True)
        # import pdb
        # pdb.set_trace()
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)
        merge_size = torch.from_numpy(merge_size).unsqueeze(0)
        merged_frame_idx = torch.from_numpy(merged_frame_idx).unsqueeze(0)

        frame_video_feature = average_to_fixed_length(frame_vecs, self.map_size, clip=None)
        # frame_video_feature = uniform_feature_sampling(frame_vecs, self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        sts = []
        eds = []

        if self.cap2queryinfo is not None:
            for i in range(len(cap_ids)):
                if self.cfg['collection'] == 'qvhighlight':
                    st, ed = 0., 0.
                else:
                    du, st, ed = self.cap2queryinfo[self.captions[cap_ids[i]]]
                    if len(frame_video_feature) < self.max_ctx_len:
                        st = int(math.floor(st * 3))
                        ed = int(math.ceil(ed * 3))

                    else:
                        st = int(math.floor(st / du * self.max_ctx_len))
                        ed = int(math.ceil(ed / du * self.max_ctx_len))

                    if st >= self.max_ctx_len:
                        st = 0
                        ed = self.max_ctx_len - 1

                sts.append(st)
                eds.append(ed)
        else: 
            st, ed, du = 0., 0., 0.
            sts.append(st)
            eds.append(ed)
            

        # text
        cap_tensors = []
        for cap_id in cap_ids:
            cap_feat = self.text_feat[cap_id][...]
            cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
            cap_tensors.append(cap_tensor)

        return clip_video_feature, frame_video_feature, cap_tensors, index, cap_ids, video_id, sts, eds, merge_size, merged_frame_idx

    def __len__(self):
        return self.length


class VisDataSet4PRVR(data.Dataset):

    def __init__(self, visual_feat, video2frames, cfg, video_ids=None, is_clip=False):
        if is_clip:
            self.visual_feat = h5py.File(visual_feat, 'r')
        else:
            self.visual_feat = visual_feat

        self.video2frames = video2frames
        self.is_clip = is_clip

        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = video2frames.keys()

        self.length = len(self.video_ids)
        self.map_size = cfg['map_size']
        self.max_ctx_len = cfg['max_ctx_l']
        self.collection = cfg['collection']

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        if self.is_clip:
            if self.collection == 'activitynet':
                video_id = video_id[2:]
            frame_vecs = self.visual_feat[video_id][...]

        else:
            frame_list = self.video2frames[video_id]
            frame_vecs = []
            for frame_id in frame_list:
                frame_vecs.append(self.visual_feat.read_one(frame_id))
            frame_vecs = np.array(frame_vecs)

        clip_video_feature, merge_size, merged_frame_idx = average_to_fixed_length(frame_vecs, self.map_size, clip=True)
        clip_video_feature = l2_normalize_np_array(clip_video_feature)
        clip_video_feature = torch.from_numpy(clip_video_feature).unsqueeze(0)
        merge_size = torch.from_numpy(merge_size).unsqueeze(0)

        frame_video_feature = average_to_fixed_length(frame_vecs, self.map_size, clip=None)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        return clip_video_feature, frame_video_feature, index, video_id, merge_size

    def __len__(self):
        return self.length


class TxtDataSet4PRVR(data.Dataset):
    """
    Load captions
    """

    def __init__(self, cap_file, text_feat_path, cfg, path_query_json=None):
        # Captions
        self.cfg = cfg
        self.captions = {}
        self.cap_ids = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
        self.text_feat_path = text_feat_path
        self.max_desc_len = cfg['max_desc_l']
        self.open_file = False
        self.length = len(self.cap_ids)
        self.max_ctx_len = cfg['max_ctx_l']
        self.cap2queryinfo = None
        if path_query_json is not None:
            query_json = load_jsonl(path_query_json)
            self.cap2queryinfo = {}

            if cfg['collection'] == 'qvhighlight':
                for i in range(len(query_json)):
                    one_data = query_json[i]
                    cap = one_data['query'].rstrip()
                    # self.cap2queryinfo[cap] = [one_data['duration'], one_data['ts'][0], one_data['ts'][1]]
            else:
                for i in range(len(query_json)):
                    one_data = query_json[i]
                    cap = one_data['desc'].rstrip()
                    self.cap2queryinfo[cap] = [one_data['duration'], one_data['ts'][0], one_data['ts'][1]]
            self.use_st_ed = True

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        if self.open_file:
            self.open_file = True
        else:
            self.text_feat = h5py.File(self.text_feat_path, 'r')
            self.open_file = True

        cap_feat = self.text_feat[cap_id][...]

        cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
        if self.cap2queryinfo is not None:
            if self.cfg['collection'] == 'qvhighlight':
                st, ed = 0., 0.
            else:
                du, st, ed = self.cap2queryinfo[self.captions[cap_id]]

                if math.ceil(st) * 3 < self.max_ctx_len:
                    st = int(math.floor(st * 3))
                    ed = int(math.ceil(ed * 3))

                else:
                    st = int(math.floor(st / du * self.max_ctx_len))
                    ed = int(math.ceil(ed / du * self.max_ctx_len))
        else: 
            st, ed, du = 0., 0., 0.

        return cap_tensor, index, cap_id, st, ed

    def __len__(self):
        return self.length


if __name__ == '__main__':
    pass


