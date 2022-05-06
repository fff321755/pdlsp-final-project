# Commends

```

mmf_convert_hm --zip_file hm.zip --password 12345  --bypass_checksum=1
mmf_run config=finetune_visual_bert.ymal model=visual_bert dataset=hateful_memes

mmf_predict config=finetune_visual_bert.yaml model=visual_bert dataset=hateful_memes run_type=val checkpoint.resume_zoo=save/visual_bert_final.pth
mmf_predict config=infer_visual_bert.yaml model=visual_bert dataset=hateful_memes run_type=test checkpoint.resume_zoo=visual_bert.finetuned.hateful_memes.from_coco

mmf_predict config=finetune_visual_bert.ymal \
    datasets=hateful_memes \
    model=visual_bert \
    run_type=val \
    checkpoint=save/visual_bert_final.pth

```