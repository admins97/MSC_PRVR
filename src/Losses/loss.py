import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from Models.MSC.model_components import clip_nce, frame_nce
from Utils.utils import pdist
import ipdb

class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        '''
        student: q_bsz dim [end token]
        teacher: q_bsz dim [attention pooling token]
        '''
        
        with torch.no_grad():
            t_d = pdist(teacher, squared=False) # q_bsz q_bsz
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss
    
class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss

class loss(nn.Module):
    def __init__(self, cfg):
        super(loss, self).__init__()
        self.cfg = cfg
        self.clip_nce_criterion = clip_nce(reduction='mean')
        self.video_nce_criterion = clip_nce(reduction='mean')

        self.rkd_a_loss = RKdAngle()
        self.rkd_d_loss = RkdDistance()

    def forward(self, input_list, batch):
        '''
        param: query_labels: List[int]
        param: clip_scale_scores.shape = [5*bs,bs]
        param: frame_scale_scores.shape = [5*bs,5*bs]
        param: clip_scale_scores_.shape = [5*bs,bs]
        param: frame_scale_scores_.shape = [5*bs,5*bs]
        param: label_dict: Dict[List]
        '''

        query_labels = batch['text_labels']
        merged_frame_label = batch['merged_frame_label']
        
        clip_scale_scores = input_list[0]
        clip_scale_scores_ = input_list[1]
        label_dict = input_list[2]
        frame_scale_scores = input_list[3]
        frame_scale_scores_ = input_list[4]
        query = input_list[5]
        clip_query = input_list[6]
        encoded_clip_feat = input_list[7]
        encoded_frame_feat = input_list[8]
        intra_diff = input_list[9]
        merge_size = input_list[10]
        clip_video_feat = input_list[11]

        clip_nce_loss = self.cfg['loss_factor'][0] * self.clip_nce_criterion(query_labels, label_dict, clip_scale_scores_)
        clip_trip_loss = self.get_clip_triplet_loss(clip_scale_scores, query_labels)

        frame_nce_loss = self.cfg['loss_factor'][1] * self.video_nce_criterion(query_labels, label_dict, frame_scale_scores_)
        frame_trip_loss = self.get_clip_triplet_loss(frame_scale_scores, query_labels)

        text_rkd_loss = self.cfg['rkd_d_coef'] * self.rkd_d_loss(query, clip_query) + self.cfg['rkd_a_coef'] * self.rkd_a_loss(query, clip_query)

        vid_lvl_ctlloss, clip_num = self.compute_vid_level_contrastive_loss(encoded_frame_feat, encoded_clip_feat, merged_frame_label, intra_diff, merge_size, clip_video_feat)
        vid_lvl_ctlloss = vid_lvl_ctlloss *self.cfg['vl_coef']
        
        loss = clip_nce_loss + clip_trip_loss + frame_nce_loss + frame_trip_loss + vid_lvl_ctlloss + text_rkd_loss
        return loss, {'clip_nce': clip_nce_loss, 'clip_trip': clip_trip_loss, 'frame_nce': frame_nce_loss, 'frame_trip': frame_trip_loss,
                      'vidlvlctl': vid_lvl_ctlloss, 'txt_rkd': text_rkd_loss, 'loss_overall': float(loss), 'clip_num': clip_num}

    def get_clip_triplet_loss(self, query_context_scores, labels):
        v2t_scores = query_context_scores.t()
        t2v_scores = query_context_scores
        labels = np.array(labels)

        # cal_v2t_loss
        v2t_loss = 0
        for i in range(v2t_scores.shape[0]):
            pos_pair_scores = torch.mean(v2t_scores[i][np.where(labels == i)])


            neg_pair_scores, _ = torch.sort(v2t_scores[i][np.where(labels != i)[0]], descending=True)
            if self.cfg['use_hard_negative']:
                sample_neg_pair_scores = neg_pair_scores[0]
            else:
                v2t_sample_max_idx = neg_pair_scores.shape[0]
                sample_neg_pair_scores = neg_pair_scores[
                    torch.randint(0, v2t_sample_max_idx, size=(1,)).to(v2t_scores.device)]

            v2t_loss += (self.cfg['margin'] + sample_neg_pair_scores - pos_pair_scores).clamp(min=0).sum()

        # cal_t2v_loss
        text_indices = torch.arange(t2v_scores.shape[0]).to(t2v_scores.device)
        t2v_pos_scores = t2v_scores[text_indices, labels]
        mask_score = copy.deepcopy(t2v_scores.data)
        mask_score[text_indices, labels] = 999
        _, sorted_scores_indices = torch.sort(mask_score, descending=True, dim=1)
        t2v_sample_max_idx = min(1 + self.cfg['hard_pool_size'],
                                 t2v_scores.shape[1]) if self.cfg['use_hard_negative'] else t2v_scores.shape[1]
        sample_indices = sorted_scores_indices[
            text_indices, torch.randint(1, t2v_sample_max_idx, size=(t2v_scores.shape[0],)).to(t2v_scores.device)]

        t2v_neg_scores = t2v_scores[text_indices, sample_indices]

        t2v_loss = (self.cfg['margin'] + t2v_neg_scores - t2v_pos_scores).clamp(min=0)

        return t2v_loss.sum() / len(t2v_scores) + v2t_loss / len(v2t_scores)

    def get_frame_trip_loss(self, query_context_scores):
        """ ranking loss between (pos. query + pos. video) and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (N, N), cosine similarity [-1, 1],
                Each row contains the scores between the query to each of the videos inside the batch.
        """

        bsz = len(query_context_scores)

        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]  # (N, )
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        # impossibly large for cosine similarity, the copy is created as modifying the original will cause error
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(query_context_scores, query_context_scores_masked)
        neg_query_pos_context_scores = self.get_neg_scores(query_context_scores.transpose(0, 1),
                                                           query_context_scores_masked.transpose(0, 1))
        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores)
        return loss_neg_ctx + loss_neg_q

    def get_neg_scores(self, scores, scores_masked):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos. Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores, except that the diagonal (positive) positions
            are masked with a large value.
        """

        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)

        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)

        sample_min_idx = 1  # skip the masked positive

        sample_max_idx = min(sample_min_idx + self.cfg['hard_pool_size'], bsz) if self.cfg['use_hard_negative'] else bsz

        # sample_max_idx = 2

        # (N, )
        sampled_neg_score_indices = sorted_scores_indices[batch_indices, torch.randint(sample_min_idx, sample_max_idx,
                                                                                       size=(bsz,)).to(scores.device)]

        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]  # (N, )
        return sampled_neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        """ Note here we encourage positive scores to be larger than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        return torch.clamp(self.cfg['margin'] + neg_score - pos_score, min=0).sum() / len(pos_score)

    def compute_vid_level_contrastive_loss(self, frame, clip, merged_frame_label, intra_diff, merge_size, clip_video_feat):
        """
        Args:
            clip_video_features (torch.Tensor): (B, 128, D) - frame
            vid (torch.Tensor): (B, event, D) - segment 
            event_lvl_idx = [32, 31, 29, 25, 17, 1]
        Returns:
            torch.Tensor: contrastive loss
        """
        device = frame.device
        def token_merge(x, clip_x, target_len, merge_size, merged_frame_idx):
            """
            x:                  Tensor[N, D]
            target_len:         int, desired output length
            merge_size:         Tensor[N], size weights for mixing
            merged_frame_idx:   Tensor[N], original frame indices

            Returns:
              merged:                   Tensor[target_len, D]
              merged_size:              Tensor[target_len]
              adjusted_merged_frame_idx:Tensor[target_len]
            """
            N, D = x.shape
            assert N % 2 == 0, "Need even N"
            num_keep = target_len
            merge_count = N - target_len
            metric = F.normalize(clip_x, dim=-1)

            # 1) build bipartite sets A/B
            idx = torch.arange(N, device=x.device)
            A_idx = idx[::2]  # even positions
            B_idx = idx[1::2]  # odd positions
            A, B = metric[A_idx], metric[B_idx]  # [N/2, D] each
            wA, wB = merge_size[A_idx].unsqueeze(1), merge_size[B_idx].unsqueeze(1)  # [N/2,1]


            sim_matrix = torch.matmul(A, B.t())  # [N/2, N/2]
            num_pairs = sim_matrix.size(0)

            # build a flat list of (score, a, b)
            candidates = []
            for a in range(num_pairs):
                for b in range(num_pairs):
                    candidates.append((float(sim_matrix[a, b].item()), a, b))
            # sort descending
            candidates.sort(key=lambda x: x[0], reverse=True)

            merge_pairs = []
            used_A = set()
            used_B = set()
            for score, a, b in candidates:
                if len(merge_pairs) >= merge_count:
                    break
                if a in used_A or b in used_B:
                    continue
                # record the *indices in the original x*
                merge_pairs.append((int(A_idx[a].item()), int(B_idx[b].item()), score))
                used_A.add(a)
                used_B.add(b)


            to_merge = {a: b for a, b, _ in merge_pairs}

            result_feats = []
            results_clip_feats = []
            result_sizes = []
            new_frame_idx = torch.zeros_like(merged_frame_idx).to(x.device)

            clip_idx = 0
            for orig_i in range(N):
                if orig_i in to_merge:
                    j = to_merge[orig_i]
                    # weights
                    w1 = merge_size[orig_i]
                    w2 = merge_size[j]
                    merged_feat = (x[orig_i] * w1 + x[j] * w2) / (w1 + w2)
                    merged_clip_feat = (clip_x[orig_i] * w1 + clip_x[j] * w2) / (w1 + w2)
                    result_feats.append(merged_feat.unsqueeze(0))
                    results_clip_feats.append(merged_clip_feat.unsqueeze(0))
                    result_sizes.append((w1 + w2).unsqueeze(0))
                    # take the frame_idx of the “anchor” orig_i
                    new_frame_idx[torch.where(merged_frame_idx == orig_i)[0]] = clip_idx
                    new_frame_idx[torch.where(merged_frame_idx == j)[0]] = clip_idx
                    # anchor_label = merged_frame_idx[orig_i].item()
                    clip_idx += 1
                elif orig_i in to_merge.values():
                    # skip the matched B token
                    continue
                else:
                    # keep untouched
                    result_feats.append(x[orig_i].unsqueeze(0))
                    results_clip_feats.append(clip_x[orig_i].unsqueeze(0))
                    result_sizes.append(merge_size[orig_i].unsqueeze(0))
                    # new_frame_idx.append(merged_frame_idx[orig_i].item())
                    new_frame_idx[torch.where(merged_frame_idx == orig_i)[0]] = clip_idx
                    clip_idx += 1

            merged = torch.cat(result_feats, dim=0)  # [target_len, D]
            merged_clip = torch.cat(results_clip_feats, dim=0)  # [target_len, D]
            merged_size = torch.cat(result_sizes, dim=0)  # [target_len]
            return merged, merged_clip, merged_size, new_frame_idx

        def rank_contrastive_loss(video_features, vid_segments, merged_frame_label, intra_diff, merge_size, clip_video_feat):
            batch_loss = 0.0
            B, num_segments, D = vid_segments.shape
            L = video_features.shape[1]  # 
            num_segments_list = []
            for b in range(B):
                frame_feats = video_features[b]  # (L, D) 128 384
                seg_feats = vid_segments[b]  # (num_segments, D) 32 384
                clip_seg_feats = clip_video_feat[b]
                seg_diff = intra_diff[b]
                frame_label = merged_frame_label[b]
                clip_size = merge_size[b]
                # Contrastive Stage I : Within Event-Positive, otherwise Negative**
                ### Adaptive Clip Length
                if seg_diff >= 0.8:  # querys are all same
                    rank = 5
                elif seg_diff >= 0.6:
                    rank = 8
                elif seg_diff >= 0.4:
                    rank = 12
                elif seg_diff >= 0.2:
                    rank = 20
                else: #
                    rank = 32

                if rank == 1:
                    loss_stage1 = torch.tensor(0.0).to(device)
                else:
                    if rank == 32:
                        rank_vid_segments = seg_feats
                    else:
                        rank_vid_segments = seg_feats
                        # merge_size = None
                        # 32 -> 20 -> 12 -> 8 -> 5
                        for target in [20, 12, 8, 5]: # 75% merging.
                            rank_vid_segments, clip_seg_feats, clip_size, frame_label = token_merge(rank_vid_segments, clip_seg_feats, target, clip_size, frame_label)
                            if target == rank:
                                break
                    num_segments, D = rank_vid_segments.shape
                    num_segments_list.append(num_segments)
                    L = video_features.shape[1]
                    event_labels = torch.arange(num_segments, device=seg_feats.device)
                    event_mapping = frame_label.to(seg_feats.device)
                    pos_mask = event_mapping.unsqueeze(1) == event_labels.unsqueeze(0)  # (L, num_segments)
                    neg_mask = ~pos_mask  # (L, num_segments)

                    # Frame (L, D) vs. Segment (num_segments, D)
                    sim_matrix = torch.matmul(frame_feats, rank_vid_segments.detach().t())  # (L, num_segments)
                    max_val = sim_matrix.max(dim=1, keepdim=True)[0].detach()  # (L,1)
                    sim_matrix = torch.exp(sim_matrix - max_val) #* temperature
                    sim_pos = (sim_matrix * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)  # (L,)
                    sim_neg = (sim_matrix * neg_mask).sum(dim=1) / (neg_mask.sum(dim=1) + 1e-8)  # (L,)
                    loss_stage1 = -torch.log(sim_pos / (sim_pos + sim_neg + 1e-8)).mean()

                    # Clip 2 Frame Retrieval # #
                    sim_matrix = torch.matmul(frame_feats.detach(), rank_vid_segments.t())  # (L, num_segments)
                    max_val = sim_matrix.max(dim=0, keepdim=True)[0].detach()  # (1, S)
                    sim_matrix = torch.exp(sim_matrix - max_val) #* temperature
                    c2f_sim_pos = (sim_matrix * pos_mask).sum(dim=0) / (pos_mask.sum(dim=0) + 1e-8)  # (L,)
                    c2f_sim_neg = (sim_matrix * neg_mask).sum(dim=0) / (neg_mask.sum(dim=0) + 1e-8)  # (L,)
                    max_val = torch.max(c2f_sim_pos, c2f_sim_neg)
                    c2f_loss_stage1 = -torch.log((c2f_sim_pos) / (c2f_sim_pos + c2f_sim_neg)).mean()
                    loss_stage1 = loss_stage1 + c2f_loss_stage1

                batch_loss += loss_stage1
                
            return batch_loss / B, num_segments_list

        loss, num_segments_list = rank_contrastive_loss(frame, clip, merged_frame_label, intra_diff, merge_size, clip_video_feat)
        clip_num = torch.tensor(num_segments_list, dtype=torch.float32).mean().item()

        return loss, clip_num
