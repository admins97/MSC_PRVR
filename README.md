# [NeurIPS 2025] Mitigating Semantic Collapse in Partially Relevant Video Retrieval

## Train
To train our MSC, use follow instructions:
```
cd src

python main.py -d qvhighlight --rkd_d_coef 15 --rkd_a_coef 30 --vl_coef 0.1 --sim_thr 0.7 --model_name qv_model
python main.py -d tvr --rkd_d_coef 15 --rkd_a_coef 30 --vl_coef 0.1 --sim_thr 0.8 --model_name tvr_model
python main.py -d act --rkd_d_coef 15 --rkd_a_coef 30 --vl_coef 0.1 --sim_thr 0.8 --model_name act_model
python main.py -d cha --rkd_d_coef 15 --rkd_a_coef 30 --vl_coef 0.1 --sim_thr 0.85 --model_name cha_model

```
### eval
To evaluate our MSC with pretrained ckpt, use follow instructions:
```
cd src
python main.py -d qvhighlight --model_name *** --eval --resume path_to_ckpt/best.ckpt
```
