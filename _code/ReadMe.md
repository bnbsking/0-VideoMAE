Formats:
1. raw data *.mp4
2. frames *.jpg
3. label *.txt || e.g. \[video_20220810082929.mp4\]\n00:04:27.75-00:04:30.25, 00:04:41.00-00:04:44.00, -----------------------
4. train.csv\/val.csv\/test.csv || /path/to/jpg class
5. prediction result.json || list\[[s0,s1,s2,s3]\]

Files
1. preprocess_videos2imgs.ipynb || convert 1-day videos to images in 32-way multiprocessing
2. preprocess_csv.ipynb
    + UnlabeledCsv: from "previous csv", "labeled txt" or "imgs from a folder"
    + LabeledCsv: from "previous train.csv", "labeled txt", then copy "previous val.csv" or move 20% data from train.csv to val.csv
3. _pretrain.sh
4. plotLoss.ipynb
5. _finetune.sh: train or eval, ep, bs, nc, data-path, model-path
6. _multieval.sh: ep, bs, nc, data-path, model-path
7. result.ipynb: compare val.csv and result.json, then get block image stacks
8. active.ipynb:
    + convert result.json to result.csv
    + plot sampled time series
    + get abnormal frames w.r.t time series
    + pick easy and hard samples