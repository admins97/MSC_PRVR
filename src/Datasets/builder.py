import os
import ipdb
from torch.utils.data import DataLoader

from Utils.basic_utils import BigFile, read_dict
from Datasets.data_provider import Dataset4PRVR, VisDataSet4PRVR, TxtDataSet4PRVR, \
    collate_train, collate_frame_val, collate_text_val, read_video_ids


def get_datasets(cfg):
    rootpath = cfg['data_root']
    collection = cfg['collection']
    trainCollection = '%strain' % collection
    valCollection = '%sval' % collection
    cap_file = {
        'train': '%s.caption.txt' % trainCollection,
        'val': '%s.caption.txt' % valCollection,
    }
    # caption
    caption_files = {x: os.path.join(rootpath, collection, 'TextData', cap_file[x]) for x in cap_file}

    # Load visual features
    if cfg['visual_feature'] == 'clip':
        visual_feats = os.path.join(rootpath, collection, 'FeatureData', '%s_clip_L14.h5' % collection)
        text_feat_path = os.path.join(rootpath, collection, 'TextData', '%s_clip_L14.h5' % collection)
        video2frames = None
        test_visual_feats = os.path.join(rootpath, collection, 'FeatureData', '%s_clip_L14.h5' % collection)
        test_video2frames = None
        is_clip = True

    elif cfg['visual_feature'] == 'clipsf': #qvhighlights
        visual_feats = os.path.join(rootpath, collection, 'FeatureData', '%s_slowfast_clip.h5' % collection)
        text_feat_path = os.path.join(rootpath, collection, 'TextData', '%s_slowfast_clip_text.h5' % collection)
        cfg['visual_feat_dim'] = 2816
        cfg['q_feat_size'] = 512
        video2frames = None
        test_visual_feats = os.path.join(rootpath, collection, 'FeatureData', '%s_slowfast_clip.h5' % collection)
        test_video2frames = None
        is_clip = True

    else:
        visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'])
        visual_feats = BigFile(visual_feat_path)
        cfg['visual_feat_dim'] = visual_feats.ndims

        text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)
        video2frames = read_dict(
            os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'], 'video2frames.txt'))

        test_visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'])
        test_visual_feats = BigFile(test_visual_feat_path)
        test_video2frames = read_dict(
            os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'], 'video2frames.txt'))
        is_clip = False

    if collection == 'tvr':
        train_dataset = Dataset4PRVR(caption_files['train'], visual_feats, text_feat_path, cfg,
                                     video2frames=video2frames, is_clip=is_clip,
                                     path_query_json='/project/prvr/dataset/tvr/tvr_train_release.jsonl')

        val_text_dataset = TxtDataSet4PRVR(caption_files['val'], text_feat_path, cfg,
                                           path_query_json='/project/prvr/dataset/tvr/tvr_val_release.jsonl')

    elif collection == 'activitynet':
        train_dataset = Dataset4PRVR(caption_files['train'], visual_feats, text_feat_path, cfg,
                                     video2frames=video2frames, is_clip=is_clip,
                                     path_query_json=None)

        val_text_dataset = TxtDataSet4PRVR(caption_files['val'], text_feat_path, cfg,
                                           path_query_json=None)
    elif collection == 'qvhighlight':
        train_dataset = Dataset4PRVR(caption_files['train'], visual_feats, text_feat_path, cfg,
                                     video2frames=video2frames, is_clip=is_clip,
                                     path_query_json='/project/prvr/dataset/qvhighlight/highlight_train_release.jsonl')

        val_text_dataset = TxtDataSet4PRVR(caption_files['val'], text_feat_path, cfg,
                                           path_query_json='/project/prvr/dataset/qvhighlight/highlight_val_release.jsonl')
    elif collection == 'charades':
        train_dataset = Dataset4PRVR(caption_files['train'], visual_feats, text_feat_path, cfg,
                                     video2frames=video2frames, is_clip=is_clip,
                                     path_query_json=None)

        val_text_dataset = TxtDataSet4PRVR(caption_files['val'], text_feat_path, cfg,
                                           path_query_json=None)
    else:
        raise ValueError("inapposite collection")

    val_video_ids_list = read_video_ids(caption_files['val'])
    val_video_dataset = VisDataSet4PRVR(visual_feats, video2frames, cfg, video_ids=val_video_ids_list, is_clip=is_clip)

    testCollection = '%stest' % collection
    test_cap_file = {'test': '%s.caption.txt' % testCollection}

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg['batchsize'],
                              shuffle=True,
                              pin_memory=cfg['pin_memory'],
                              num_workers=cfg['num_workers'],
                              collate_fn=collate_train)
    context_dataloader = DataLoader(val_video_dataset,
                                    collate_fn=collate_frame_val,
                                    batch_size=cfg['eval_context_bsz'],
                                    num_workers=cfg['num_workers'],
                                    shuffle=False,
                                    pin_memory=cfg['pin_memory'])
    query_eval_loader = DataLoader(val_text_dataset,
                                   collate_fn=collate_text_val,
                                   batch_size=cfg['eval_query_bsz'],
                                   num_workers=cfg['num_workers'],
                                   shuffle=False,
                                   pin_memory=cfg['pin_memory'])
    return cfg, train_loader, context_dataloader, query_eval_loader