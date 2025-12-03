import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import numpy as np
from tqdm import tqdm
import torch
from Utils.utils import gpu

def get_gt(video_metas, query_metas):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            if query_id.split('#', 1)[0] == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt

def get_gt_act(video_metas, query_metas):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            if query_id.split('#', 1)[0][2:] == vid_id:

                v2t_gt[-1].append(i)
    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt

def eval_q2m(scores, q2m_gts):

    n_q, n_m = scores.shape

    gt_ranks = torch.zeros((n_q), dtype=torch.int32).cuda()
    aps = torch.zeros(n_q).cuda()
    for i in range(n_q):
        s = scores[i]
        sorted_idxs = torch.argsort(s)
        rank = n_m + 1
        tmp_set = []
        for k in q2m_gts[i]:
            tmp = torch.where(sorted_idxs == k)[0][0] + 1
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank

    # compute metrics
    r1 = 100.0 * len(torch.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(torch.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(torch.where(gt_ranks <= 10)[0]) / n_q
    r100 = 100.0 * len(torch.where(gt_ranks <= 100)[0]) / n_q

    return (r1, r5, r10, r100)


def cal_perf(t2v_all_errors, t2v_gt):

    # video retrieval
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100) = eval_q2m(t2v_all_errors, t2v_gt)

    return (t2v_r1, t2v_r5, t2v_r10, t2v_r100)

def text_dist(video_metas, query_metas, video_query_, clip_query_, collection, logger):
    
    clip_sim_mean_per_vid = []
    text_sim_mean_per_vid = []
    for vid_id in video_metas:
        clip_feat = []
        text_feat = []
        for i, query_id in enumerate(query_metas):
            if collection == 'activitynet':
                if query_id.split('#', 1)[0][2:] == vid_id:
                    clip_feat.append(clip_query_[i].unsqueeze(0))
                    text_feat.append(video_query_[i].unsqueeze(0))
            else:
                if query_id.split('#', 1)[0] == vid_id:
                    clip_feat.append(clip_query_[i].unsqueeze(0))
                    text_feat.append(video_query_[i].unsqueeze(0))
        
        clip_feat = torch.cat(clip_feat, dim=0)
        text_feat = torch.cat(text_feat, dim=0)
        if len(clip_feat) <= 1:
            continue
        
        clip_sim_per_vid = torch.matmul(clip_feat, clip_feat.t())
        mask = ~torch.eye(clip_sim_per_vid.size(0), dtype=torch.bool, device=clip_sim_per_vid.device)
        clip_sim_per_vid_mean = clip_sim_per_vid[mask].mean()
        
        text_sim_per_vid = torch.matmul(text_feat, text_feat.t())
        mask = ~torch.eye(text_sim_per_vid.size(0), dtype=torch.bool, device=text_sim_per_vid.device)
        text_sim_per_vid_mean = text_sim_per_vid[mask].mean()
        
        clip_sim_mean_per_vid.append(clip_sim_per_vid_mean)
        text_sim_mean_per_vid.append(text_sim_per_vid_mean)
    
    clip_sim_mean_per_vid = torch.tensor(clip_sim_mean_per_vid)
    text_sim_mean_per_vid = torch.tensor(text_sim_mean_per_vid)
    
    logger.info(f"CLIP inner video text sim mean: {clip_sim_mean_per_vid.mean()}")
    logger.info(f"FINE inner video text sim mean: {text_sim_mean_per_vid.mean()}")

        
    clip_sim = torch.matmul(clip_query_, clip_query_.t())
    mask = ~torch.eye(clip_sim.size(0), dtype=torch.bool, device=clip_sim.device)
    clip_mean_similarity = clip_sim[mask].mean()
    
    video_sim = torch.matmul(video_query_, video_query_.t())
    mask = ~torch.eye(video_sim.size(0), dtype=torch.bool, device=video_sim.device)
    video_mean_sim = video_sim[mask].mean()
    
    logger.info(f"CLIP total text sim mean: {clip_mean_similarity}")
    logger.info(f"FINE total text sim mean: {video_mean_sim}")
    
    return

def vid_dist(video_feat, clip_feat, logger):
    '''
    vid feat: bsz, len, dm
    clip feat: bsz, len, dm
    '''
    with torch.no_grad():
        vid_sim_mean_per_vid = []
        clip_sim_mean_per_vid = []

        video_feat = F.normalize(video_feat, dim=-1)
        clip_feat = F.normalize(clip_feat, dim=-1)

        vid_sim = torch.matmul(video_feat, video_feat.transpose(1, 2))  # bsz len len
        mask = ~torch.eye(vid_sim.size(1), dtype=torch.bool, device=vid_sim.device)
        mask = mask.unsqueeze(0).expand(vid_sim.size(0), -1, -1)
        vid_mean_sim = vid_sim[mask].mean()  # 640

        clip_sim = torch.matmul(clip_feat, clip_feat.transpose(1, 2))  # bsz len len
        mask = ~torch.eye(clip_sim.size(1), dtype=torch.bool, device=clip_sim.device)
        mask = mask.unsqueeze(0).expand(clip_sim.size(0), -1, -1)
        clip_mean_sim = clip_sim[mask].mean()  # 640

        logger.info(f"CLIP inner video clip feat sim mean: {clip_mean_sim}")
        logger.info(f"FINE inner video clip feat sim mean: {vid_mean_sim}")

        chunk_size = 400

        bsz, c_len, dm = clip_feat.shape
        clip_total_feat = clip_feat.reshape(bsz * c_len, dm)
        video_feat = video_feat[:, :c_len]
        bsz, c_len, dm = video_feat.shape
        vid_total_feat = video_feat.reshape(bsz * c_len, dm)

        vid_total_sim_sum = 0.0
        clip_total_sim_sum = 0.0
        num_elements = 0

        for i in range(0, vid_total_feat.shape[0], chunk_size):
            vid_chunk = vid_total_feat[i: i + chunk_size]  # chunk
            clip_chunk = clip_total_feat[i: i + chunk_size]

            vid_sim_chunk = torch.matmul(vid_chunk, vid_total_feat.t())  # (chunk_size, bsz*c_len)
            clip_sim_chunk = torch.matmul(clip_chunk, clip_total_feat.t())  # (chunk_size, bsz*c_len)

            mask = torch.ones_like(vid_sim_chunk, dtype=torch.bool, device=vid_sim_chunk.device)
            mask[:, i: i + chunk_size] = False  

            vid_sim_sum = vid_sim_chunk.masked_select(mask).sum()
            clip_sim_sum = clip_sim_chunk.masked_select(mask).sum()

            chunk_elements = mask.sum().item()   

            vid_total_sim_sum += vid_sim_sum
            clip_total_sim_sum += clip_sim_sum
            num_elements += chunk_elements

        vid_mean_sim = vid_total_sim_sum / num_elements
        clip_mean_sim = clip_total_sim_sum / num_elements

        logger.info(f"CLIP total sim mean: {clip_mean_sim}")
        logger.info(f"FINE total sim mean: {vid_mean_sim}")

    return

class validations(nn.Module):
    def __init__(self, cfg):
        super(validations, self).__init__()
        self.cfg = cfg

    def forward(self, model, context_dataloader, query_eval_loader, logger):

        model.eval()

        context_info = self.compute_context_info(model, context_dataloader)
        score_sum, query_metas, video_query, clip_query, score_clip, score_frame = self.compute_query2ctx_info(model,
                                                             query_eval_loader,
                                                             context_info)
        video_metas = context_info['video_metas']

        if self.cfg['collection'] == 'activitynet':
            v2t_gt, t2v_gt = get_gt_act(video_metas, query_metas)
        else:
            v2t_gt, t2v_gt = get_gt(video_metas, query_metas)

        text_dist(video_metas, query_metas, video_query, clip_query, self.cfg['collection'], logger)
        vid_dist(context_info['video_feat'], context_info['clip_clip_feat'], logger)
        t2v_r1, t2v_r5, t2v_r10, t2v_r100 = cal_perf(-1 * score_sum, t2v_gt)
        t2v_rsum = 0
        t2v_rsum += (t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100)

        clip_t2v_r1, clip_t2v_r5, clip_t2v_r10, clip_t2v_r100 = cal_perf(-1 * score_clip, t2v_gt)
        clip_t2v_rsum = 0
        clip_t2v_rsum += (clip_t2v_r1 + clip_t2v_r5 + clip_t2v_r10 + clip_t2v_r100)

        frame_t2v_r1, frame_t2v_r5, frame_t2v_r10, frame_t2v_r100 = cal_perf(-1 * score_frame, t2v_gt)
        frame_t2v_rsum = 0
        frame_t2v_rsum += (frame_t2v_r1 + frame_t2v_r5 + frame_t2v_r10 + frame_t2v_r100)

        return [t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_rsum], [clip_t2v_r1, clip_t2v_r5, clip_t2v_r10, clip_t2v_r100, clip_t2v_rsum], [frame_t2v_r1, frame_t2v_r5, frame_t2v_r10, frame_t2v_r100, frame_t2v_rsum]

    def compute_query2ctx_info(self, model, query_eval_loader, ctx_info):

        query_metas = []
        score_sum = []
        vid_q = []
        clip_q = []
        score_clip = []
        score_frame = []
        for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding",
                               total=len(query_eval_loader)):
            batch = gpu(batch)
            # for q in batch[-3]:
            #     for_similarity.append(raw_texts[q])
            query_metas.extend(batch[-3])
            query_feat = batch[0]
            query_mask = batch[1]
            video_query, encoded_query = model.encode_query(query_feat, query_mask)
            clip_query = model.encode_clip_query(query_feat, query_mask)
            _clip_scale_scores, _frame_scale_scores = model.get_pred_from_raw_query(
                video_query, None, ctx_info["video_proposal_feat"], ctx_info["video_feat"])
            _score_sum = self.cfg['clip_scale_w'] * _clip_scale_scores + self.cfg['frame_scale_w'] * _frame_scale_scores

            score_sum.append(_score_sum)
            score_clip.append(_clip_scale_scores)
            score_frame.append(_frame_scale_scores)
            vid_q.append(video_query)
            clip_q.append(clip_query)

        score_sum = torch.cat(score_sum, dim=0)
        score_clip = torch.cat(score_clip, dim=0)
        score_frame = torch.cat(score_frame, dim=0)
        vid_q = torch.cat(vid_q, dim=0)
        clip_q = torch.cat(clip_q, dim=0)
        vid_q = F.normalize(vid_q, dim=-1)
        clip_q = F.normalize(clip_q, dim=-1)
        return score_sum, query_metas, vid_q, clip_q, score_clip, score_frame

    def compute_context_info(self, model, context_dataloader):

        n_total_vid = len(context_dataloader.dataset)
        bsz = self.cfg['eval_context_bsz']
        metas = []  # list(dicts)
        vid_proposal_feat = []
        clip_clip_feat = []
        frame_feat, frame_mask = [], []
        for idx, batch in tqdm(enumerate(context_dataloader), desc="Computing query2video scores",
                               total=len(context_dataloader)):
            batch = gpu(batch)
            metas.extend(batch[-2])
            clip_video_feat_ = batch[0]
            frame_video_feat_ = batch[1]
            frame_mask_ = batch[2]
            merge_size = batch[-1].unsqueeze(-1)
            _frame_feat, _video_proposal_feat = model.encode_context(clip_video_feat_, frame_video_feat_, frame_mask_, eval=True, epoch=idx, merge_size=merge_size)

            frame_feat.append(_frame_feat)
            frame_mask.append(frame_mask_)

            vid_proposal_feat.append(_video_proposal_feat)
            clip_clip_feat.append(clip_video_feat_) # 64

        vid_proposal_feat = torch.cat(vid_proposal_feat, dim=0)
        clip_clip_feat = torch.cat(clip_clip_feat, dim=0)

        def cat_tensor(tensor_list):
            if len(tensor_list) == 0:
                return None
            else:
                seq_l = [e.shape[1] for e in tensor_list]
                b_sizes = [e.shape[0] for e in tensor_list]
                b_sizes_cumsum = np.cumsum([0] + b_sizes)
                if len(tensor_list[0].shape) == 3:
                    hsz = tensor_list[0].shape[2]
                    res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l), hsz)
                elif len(tensor_list[0].shape) == 2:
                    res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l))
                else:
                    raise ValueError("Only support 2/3 dimensional tensors")
                for i, e in enumerate(tensor_list):
                    res_tensor[b_sizes_cumsum[i]:b_sizes_cumsum[i + 1], :seq_l[i]] = e
                return res_tensor

        return dict(
            video_metas=metas,  # list(dict) (N_videos)
            video_proposal_feat=vid_proposal_feat,
            video_feat=cat_tensor(frame_feat),
            video_mask=cat_tensor(frame_mask),
            clip_clip_feat=clip_clip_feat
        )