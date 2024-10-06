python3 -m sample.create_synthesis_dataset --dataset humanml --sin_path dataset/humanml/0000.npy --model_path ../CoSMoS/save/humanml/0000/model000020000.pt --cls_model_path save/densecls/humanml/0000/model-1.pt --num_samples 100 --batch_size 100 --device 0
# python3 -m sample.create_synthesis_dataset --dataset mixamo --sin_path dataset/mixamo/0000.bvh --model_path ../CoSMoS/save/mixamo/0000/model000060000.pt --cls_model_path save/densecls/mixamo/0000/model-1.pt --num_samples 100 --batch_size 100 --device 0

