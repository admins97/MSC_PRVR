# [NeurIPS 2025] Mitigating Semantic Collapse in Partially Relevant Video Retrieval

> WonJun Moon*, MinSeok Jung*, Gilhan Park, Tae-Young Kim, Cheol-Ho Cho, Woojin Jun, Jae-Pil Heo <br>
> Sungkyunkwan University

##### [[Arxiv](https://arxiv.org/abs/2311.08835)] [[OpenReview](https://openreview.net/forum?id=Wlpf0Vg4yU)]

## 🔖 Abstract
Partially Relevant Video Retrieval (PRVR) seeks videos where only part of the content matches a text query. Existing methods treat every annotated text–video pair as a positive and all others as negatives, ignoring the rich semantic variation both within a single video and across different videos. Consequently, embeddings of both queries and their corresponding video‐clip segments for distinct events within the same video collapse together, while embeddings of semantically similar queries and segments from different videos are driven apart. This limits retrieval performance when videos contain multiple, diverse events. This paper addresses the aforementioned problems, termed as semantic collapse, in both the text and video embedding spaces. We first introduce Text Correlation Preservation Learning, which preserves the semantic relationships encoded by the foundation model across text queries. To address collapse in video embeddings, we propose Cross-Branch Video Alignment (CBVA), a contrastive alignment method that disentangles hierarchical video representations across temporal scales. Subsequently, we introduce order-preserving token merging and adaptive CBVA to enhance alignment by producing video segments that are internally coherent yet mutually distinctive. Extensive experiments on PRVR benchmarks demonstrate that our framework effectively prevents semantic collapse and substantially improves retrieval accuracy.

## 📑 Datasets (CLIP Features)
> <b> [TVR](link)</b>  <br>
> <b> [ActivityNet-Captions](link)</b>  <br>
> <b> [Charades](link)</b>  <br>
> <b> [QVHighlights](link)</b>  <br>

## 🚀 Training
To train our MSC, use follow instructions:
```
cd src

python main.py -d qvhighlight --rkd_d_coef 15 --rkd_a_coef 30 --vl_coef 0.1 --sim_thr 0.7 --model_name qv_model
python main.py -d tvr --rkd_d_coef 15 --rkd_a_coef 30 --vl_coef 0.1 --sim_thr 0.8 --model_name tvr_model
python main.py -d act --rkd_d_coef 15 --rkd_a_coef 30 --vl_coef 0.1 --sim_thr 0.8 --model_name act_model
python main.py -d cha --rkd_d_coef 15 --rkd_a_coef 30 --vl_coef 0.1 --sim_thr 0.85 --model_name cha_model

```

## 👀 Evaluation
To evaluate our MSC with pretrained ckpt, use follow instructions:
```
cd src
python main.py -d qvhighlight --model_name *** --eval --resume path_to_ckpt/best.ckpt
```


## 📖 BibTeX 
If you find the repository or the paper useful, please use the following entry for citation.
```
@inproceedings{
moon2025mitigating,
title={Mitigating Semantic Collapse in Partially Relevant Video Retrieval},
author={WonJun Moon and MinSeok Jung and Gilhan Park and Tae-Young Kim and Cheol-Ho Cho and Woojin Jun and Jae-Pil Heo},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=Wlpf0Vg4yU}
}
```
