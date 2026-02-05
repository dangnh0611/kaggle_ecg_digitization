*WIP, update if needed*


## Others

#### ALL DATA, COAT 512, SEED 1022, LR 3e-4
```bash
python3 -m yagm.run -m local=local 'cv.fold_idx=0' cv.train_on=all cv.val_on=val exp_name=ALLDATA_FINAL_COAT512_GT500_LR3en4_SEED1022 exp=final_coat512_gt500 optim.lr=3e-4 scheduler.min_lr_factor=5e-4 loader.train_batch_size=4 trainer.accumulate_grad_batches=2 loader.train_num_workers=32 'loggers=[csv]' trainer.compile.mode=model seed=1022
```

#### ALL DATA, CONVNEXT SMALL 1024, LR 5e-5
```bash
python3 -m yagm.run -m local=local 'cv.fold_idx=0' cv.train_on=all cv.val_on=val exp_name=ALLDATA_FINAL_CONVNEXTSMALL1024_GT1000_LR5en5_SEED1998 exp=final_convnextsmall1024_gt1000 optim.lr=5e-5 scheduler.min_lr_factor=5e-4 loader.train_batch_size=4 trainer.accumulate_grad_batches=2 loader.train_num_workers=32 'loggers=[csv,wandb]' trainer.compile.mode=model seed=1998
```

#### FOLD 0 CONVNEXT SMALL 1024, SMP DECODER (BATCH NORM), KEYPOINT BY ROUND 1, NO TEMPLATE (IN_CHANS=3), EMA 0.995
```bash
python3 -m yagm.run -m local=local 'cv.fold_idx=0' exp_name=IMG1024_GT1024_SIG exp=signal_heatmap_exp optim.lr=5e-5 scheduler.min_lr_factor=1e-3 loader.train_batch_size=2 trainer.accumulate_grad_batches=4 loader.train_num_workers=16 loader.val_num_workers=16 'loggers=[csv,wandb]' 'loss.mtl.loss_weights=[1,0,0]' loss.mtl.weight_method=gls loss.heatmap_loss=jsd data.sigma=2 model/encoder=convnext model.decoder.n_blocks=4 'model.decoder.decoder_channels=[-1,256,224,192,160]' data.gt_name=keypoints_by_round1 trainer.max_steps=60000 trainer.val_check_interval=10000 trainer.compile.mode=model 'data.heatmap_stride=[1,1]' 'data.heatmap_hwl=[1024,1024,1000]' 'loggers=[csv]'
```


## Used in final submission

#### ALL DATA, FINE CONVNEXT LARGE 512 + VGG19 1024 + GT 1000, LR 1e-4
```bash
python3 -m yagm.run -m local=local 'cv.fold_idx=0' cv.train_on=all cv.val_on=val exp_name=ALLDATA_FINAL_CONVNEXT512_VGG1024_GT1000_LR1en4_SEED981022 exp=final_fine_convnext512_vgg1024_gt1000 optim.lr=1e-4 scheduler.min_lr_factor=5e-4 loader.train_batch_size=2 trainer.accumulate_grad_batches=4 loader.train_num_workers=16 'loggers=[csv]' trainer.compile.mode=model seed=981022
```

#### ALL DATA, FINE COAT 512 + VGG19 1024 + GT1000, LR 2e-4
```bash
python3 -m yagm.run -m local=local 'cv.fold_idx=0' cv.train_on=all cv.val_on=val exp_name=ALLDATA_FINAL_COAT512_VGG1024_GT1000_LR2en4_SEED980611 exp=final_fine_coat512_vgg1024_gt1000 optim.lr=2e-4 scheduler.min_lr_factor=5e-4 loader.train_batch_size=4 trainer.accumulate_grad_batches=2 loader.train_num_workers=32 'loggers=[csv,wandb]' trainer.compile.mode=model seed=980611
```