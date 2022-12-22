### File structures and formats
1. .../\*.mp4
2. \_data/
    + imgs/
        + 20220826\/\*.imgs
        + 20221124\/\*.imgs
    + labels/
        + labelv1.txt
            + \[video_20220810082929.mp4\]\n00:04:27.75-00:04:30.25, 00:04:41.00-00:04:44.00, -----------------------\n...
        + labelv1.json
            + {videoPath:list\[list\[cid:int, startFrame:str, endFrame:str\], ...\], ...}
    + csvPretext/
        + 20220826_v1/: 
            + train.csv, val.csv, test.csv
                + imgPath: str, class: int
    + csvDownstream/
        + 20220826_v2/
            + train.csv, val.csv, test.csv
    + csvUnlabeled/
        + 20220826_all/
            + train.csv, val.csv, test.csv
3. \_exps/:
    + pretext_0826_v1/:
        + checkpoints, log.txt
    + downstream_0826_v2/:
        + checkpoints, log.txt, eval.txt
        + result.json
            + list\[list\[c0:float, c1:float, c2:float, c3:float\]\]
        + wrong/GT_gt_PD_pd_frameName/\*.jpg (16 imgs)
    + unlabeled_0826_all/:
        + result.json
            + list\[list\[c0:float, c1:float, c2:float, c3:float\]\]
        + result.csv
            + imgPath:str, c0:float, c1:float, c2:float, c3:float, entropy:float, pd_cls:int
        + active_entropy.csv, active_entropy_fixed.csv
            + imgPath:str, entropy:float, time:str, gt:int
        + active_entropy.csv, active_entropy_fixed.csv
            + imgPath:str, pd_cls:int, time:str, gt:int
        + active_complete.csv
            + imgPath: str, class: int
        + confusion.jpg
    
### Processes in \_code/
1. preprocess_videos2imgs.ipynb
    + convert 1-day videos to images \_data/imgs/20220826/\*.jpg in 32-way multiprocessing
2. preprocess_pretext.ipynb
    + 3 ways to collect data
        + inherit previous csv: \_data/csvPretext/\*/train.csv
        + labeled videos: \_data/labels/\*.txt
        + all image path: list[str]
    + output \_data/csvPretext/20220826_v1/\*.csv for pretext training (n rows generate n//16 data in kinetics.py)
3. \_pretrain.sh
    + specify the following then do pretext training -> log.txt
        + outputDir: \_exps/pretext_0826_v1
        + dataPath: \_data/csvPretext/20220826_v1/train.csv
4. plotLoss.ipynb
    + plotPretext: load \_exps/pretext_0826_v1/log.txt and plot loss curve
5. preprocess_downstream.ipynb
    + 2 ways to collect data
        + inherit previous csv: \_data/csvDownstream/\*/train.csv
        + labeled videos: \_data/labels/\*.txt
    + valCsvPath: inherit if not none else auto split 20% from training data
    + output \_data/csvDownstream/20220826_v2/\*.csv for downstream training (n rows generate n data in kinetics.py)
6. \_finetune.sh
    + specify the following then do downstream training -> log.txt
        + outputDir: \_exps/downstream_0826_v2
        + dataPath: \_data/csvDownstream/20220826_v2
        + modelPath: \_exps/pretext_0826_v1/checkpoint-3199.pth
    + comment --eval
7. \_multi_eval.sh
    + specify the following then find best weight -> eval.txt
        + outputDir: \_exps/downstream_0826_v2
        + dataPath: \_data/csvDownstream/20220826_v2
8. plotLoss.ipynb
    + plotDownstream: load \_exps/downstream_0826_v2/log.txt and plot loss curve
    + getBest: load \_exps/downstream_0826_v2/eval.txt and get best accuracy weight
9. \_finetune.sh
    + specify the following then do downstream training -> result.json
        + outputDir: \_exps/downstream_0826_v2
        + dataPath: \_data/csvDownstream/20220826_v2
        + modelPath: \_exps/downstream_0826_v2/\*.pth
    + turn on --eval
10. result.ipynb
    + load the following then plot confusion matrix and save 
        + labelPath: \_data/csvDownstream/20220826_v2/val.csv
        + resultPath: \_exps/downstream_0826_v2/
<!--Active learning-->
11. preprocessUnlabeled.ipynb
    + specify imgFolderL then output \_data/csvUnlabeled/20220826_all/\*.csv
12. \_finetune.sh
    + specify the following then inference on unlabeled data -> result.json
        + outputDir: \_exps/unlabeled_0826_all
        + dataPath: \_data/csvUnlabeled/20220826_all
        + modelPath: \_exps/downstream_0826_v2/\*.pth
    + turn on --eval
13. resultUnlabeled.ipynb
    + specify the following
        + testCsvPath: \_data/csvUnlabeled/20220826_all/test.csv
        + resultFolder: \_exps/unlabeled_0826_all
    + then generate the following
        + \_exps/unlabeled_0826_all/result.csv for comparison
        + \_exps/unlabeled_0826_all/active_entropy.csv for manually fill in gt labels
        + \_exps/unlabeled_0826_all/active_series.csv for manually fill in gt labels
14. resultUnlabeledFixed.ipynb
    + specify the path of active_entropy.csv and active_series.csv
    + then generate \_exps/unlabeled_0826_all/active_complete.csv for next active cycle

### Others
+ Modified permanantly:
    + run_mae_pretraining.py: line213-221 lr-scheduler
    + kinetics.py:
        + (downstream) line81-95   prepare dataset; line103-106 load image instead of videos; line312-314 length of dataset
        + (pretext)    line510-519 prepare dataset; line538-540 load image instead of videos; line550-552 length of dataset
    + run_mae_finetuning.py: line486-492 evaluation
    + engine_for_finetuning.py: line176-201 evaluation
+ Modify before downstream training:
    + run_class_finetuning.py: line453-461 lr_scheduler
    + run_class_finetuning.py: line476 weighted loss
    + datasets.py: line149 class
    + engine_for_finetuning.py: line163,193 top-K
    + kinetics.py: line85 imgFolderL
