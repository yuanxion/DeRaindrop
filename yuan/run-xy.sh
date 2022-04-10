
# run
	python3 -m pip install torch torchvision
	CUDA_VISIBLE_DEVICES=0 python3 predict.py --mode demo --input_dir ./demo/input/ --output_dir ./demo/output/

	CUDA_VISIBLE_DEVICES=gpu_id python3 predict.py --mode test --input_dir ./test_a/data/ --gt_dir ./test_a/gt/


# train xy
	CUDA_VISIBLE_DEVICES=0 python3 train-xy.py
