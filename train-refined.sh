CUDA_VISIBLE_DEVICES=0 python hrnet.py --name refined --inet refined --iters 1 --suffix iters1 --enhance de --freq 0.25 --noise True --lambda_coarse 0.6 --lambda_detect 1.0 --batchSize 8 --nThreads 24 --fliplr 0.5 --flipud 0.5 --nEpochs 100