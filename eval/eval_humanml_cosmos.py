from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from data_utils.humanml.utils.metrics import *
from data_utils.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from data_utils.humanml.scripts.motion_process import *
from data_utils.humanml.utils.utils import *
from utils.model_util import create_model_and_diffusion, load_model
import numpy as np
from scipy import linalg

from diffusion import logger
from utils import dist_util
from data_utils.get_data import get_dataset_loader
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy('file_system')

def slice_motion_sample(sample, window_size, step_size=10):
    # sample [nframes, nfeats]
    windows = []
    max_offset = sample.shape[0] - window_size + 1
    for offset_i in np.arange(max_offset)[0::step_size]:
        windows.append(sample[offset_i:offset_i+window_size].unsqueeze(0))
    return torch.cat(windows, dim=0)



def get_sample(path, device):
    try:
        return torch.tensor(np.load(path), device=device).squeeze()#.permute(1, 0)
    except:
        return np.load(path, allow_pickle=True)[None][0]['motion_raw'][0].to(device).squeeze().permute(1, 0)  # benchmark npy

def generate_eval_samples(model, diffusion, num_samples):
    sample_fn = diffusion.p_sample_loop
        
    frame_length = 196

    # 주어진 비율 리스트
    ratios_list = [
        [0.3, 0.3, 0.4], 
        [0.4, 0.3, 0.3], 
        [0.3, 0.4, 0.3], 
        [0.2, 0.3, 0.5], 
        [0.2, 0.5, 0.3], 
        [0.5, 0.3, 0.2], 
        [0.5, 0.2, 0.3], 
        [0.3, 0.2, 0.5], 
        [0.3, 0.5, 0.2], 
        [0.33, 0.33, 0.33]
    ]

    # 비율별 길이 계산 함수
    def calculate_labels(ratios, frame_length):
        labels_index = [int(frame_length * ratio) for ratio in ratios]
        remaining_length = frame_length - sum(labels_index)
        min_index = labels_index.index(min(labels_index))
        labels_index[min_index] += remaining_length
        labels_0 = labels_index[0]
        labels_1 = labels_index[1]
        labels_2 = labels_index[2]
        labels_org = [0] * labels_0 + [1] * labels_1 + [2] * labels_2
        return labels_org


    # 비율 별로 5번씩 반복하여 레이블 계산 및 원-핫 인코딩
    all_labels_one_hot = []
    for ratios in ratios_list:
        for _ in range(5):
            labels = calculate_labels(ratios, frame_length)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            labels_one_hot = F.one_hot(labels_tensor, num_classes=3)
            all_labels_one_hot.append(labels_one_hot)

    # all_labels_one_hot을 단일 텐서로 결합
    all_labels_one_hot_tensor = torch.cat(all_labels_one_hot, dim=0)
    all_labels_one_hot_tensor = all_labels_one_hot_tensor.view(num_samples, frame_length, 3).permute(0,2,1).float().to(args.device)
    # labels_index = [int(n_frames*0.3)+1, int(n_frames*0.3)+1, int(n_frames*0.4)]
    # print(labels_index)
    # labels_0 = int(labels_index[0]) 
    # labels_1 = int(labels_index[1]) 
    # labels_2 = int(labels_index[2]) 
    
    # ratio_0 = 1.0
    # ratio_2 = 1.0
    # labels_0 = int(labels_index[0]) 
    # labels_1 = int(labels_index[1]) 
    # labels_2 = int(labels_index[2]) 
    
    # labels_org = [0]*round(labels_0*ratio_0)+[1]*(labels_1+round(labels_0*(1-ratio_0))+round(labels_2*(1-ratio_2)))+[2]*round(labels_2*ratio_2)
    # print('all_labels_one_hot_tensor.permute', all_labels_one_hot_tensor.shape)
    # labels = all_labels_one_hot_tensor.permute(0,2,1).float()
    # labels = F.one_hot(torch.Tensor(all_labels).to(torch.int64), num_classes=3)
    # print(labels.shape)
    # labels = labels.to(args.device).repeat(args.batch_size,1).reshape(args.batch_size, n_frames,3).permute(0,2,1).float()

    
    model_kwargs = {'y': {'dense_label': all_labels_one_hot_tensor}}
    print('model_kwargs', model_kwargs)
    sample = sample_fn(
        model,
        (num_samples, model.njoints, model.nfeats, 196),  # FIXME - 196?
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )
    sample = sample.squeeze().permute(0, 2, 1)

    return sample

def calc_inter_diversity(eval_wrapper, samples):
    motion_emb = eval_wrapper.get_motion_embeddings(samples, torch.tensor([samples.shape[1]] * samples.shape[0])).cpu().numpy()
    dist = linalg.norm(motion_emb[:samples.shape[0]//2] - motion_emb[samples.shape[0]//2:], axis=1)  # FIXME - will not work for odd bs
    return dist.mean()

def calc_sifid(eval_wrapper, gen_samples, gt_sample, window_size=20):

    gt_slices = slice_motion_sample(gt_sample, window_size)

    def get_stats(_samples):
        _motion_embeddings = eval_wrapper.get_motion_embeddings(_samples, torch.tensor([_samples.shape[1]] * _samples.shape[0])).cpu().numpy()
        _mu, _cov = calculate_activation_statistics(_motion_embeddings)
        return _mu, _cov

    sifid = []

    gt_mu, gt_cov = get_stats(gt_slices)

    for sampe_i in range(gen_samples.shape[0]):
        gen_slices = slice_motion_sample(gen_samples[sampe_i], window_size)
        gen_mu, gen_cov = get_stats(gen_slices)
        sifid.append(calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov))

    return np.array(sifid).mean()


def calc_intra_diversity(eval_wrapper, samples, window_size=20):
    max_offset = samples.shape[1] - window_size
    dist = []
    for sample_i in range(samples.shape[0]):
        offsets = np.random.randint(max_offset, size=2)
        window0 = samples[[sample_i], offsets[0]:offsets[0]+window_size, :]
        window1 = samples[[sample_i], offsets[1]:offsets[1]+window_size, :]
        motion_emb = eval_wrapper.get_motion_embeddings(torch.cat([window0, window1]), torch.tensor([window_size] * 2)).cpu().numpy()
        dist.append(linalg.norm(motion_emb[0] - motion_emb[1]))
    return np.array(dist).mean()


def evaluate(args, model, diffusion, eval_wrapper, num_samples_limit, replication_times):

    results = {}

    for window_size in [20]:
        print(f'===Starting [window_size={window_size}]===')
        intra_diversity = []
        gt_intra_diversity = []
        intra_diversity_gt_diff = []
        inter_diversity = []
        sifid = []
        for rep_i in range(replication_times):
            print('agr',args.sin_path)
            gt_samples = get_sample(args.sin_path, args.device)
            gen_samples = generate_eval_samples(model, diffusion, num_samples_limit)
            print(f'===REP[{rep_i}]===')
            _intra_diversity = calc_intra_diversity(eval_wrapper, gen_samples, window_size=window_size)
            intra_diversity.append(_intra_diversity)
            print('intra_diversity [{:.3f}]'.format(_intra_diversity))
            _gt_intra_diversity = calc_intra_diversity(eval_wrapper, torch.tile(gt_samples[None], (gen_samples.shape[0], 1, 1)), window_size=window_size)
            gt_intra_diversity.append(_gt_intra_diversity)
            print('gt_intra_diversity [{:.3f}]'.format(_gt_intra_diversity))
            _intra_diversity_gt_diff = abs(_intra_diversity - _gt_intra_diversity)
            intra_diversity_gt_diff.append(_intra_diversity_gt_diff)
            print('intra_diversity_gt_diff [{:.3f}]'.format(_intra_diversity_gt_diff))
            _inter_diversity = calc_inter_diversity(eval_wrapper, gen_samples)
            inter_diversity.append(_inter_diversity)
            print('inter_diversity [{:.3f}]'.format(_inter_diversity))
            _sifid = calc_sifid(eval_wrapper, gen_samples, gt_samples, window_size=window_size)
            sifid.append(_sifid)
            print('SiFID [{:.3f}]'.format(_sifid))

        results[window_size] = {
            'intra_diversity': {'mean': np.mean(intra_diversity), 'std': np.std(intra_diversity)},
            'gt_intra_diversity': {'mean': np.mean(gt_intra_diversity), 'std': np.std(gt_intra_diversity)},
            'intra_diversity_gt_diff': {'mean': np.mean(intra_diversity_gt_diff), 'std': np.std(intra_diversity_gt_diff)},
            'inter_diversity': {'mean': np.mean(inter_diversity), 'std': np.std(inter_diversity)},
            'sifid': {'mean': np.mean(sifid), 'std': np.std(sifid)},
        }

        print(f'===Summary [window_size={window_size}]===')
        print('intra_diversity [{:.3f}±{:.3f}]'.format(results[window_size]['intra_diversity']['mean'], results[window_size]['intra_diversity']['std']))
        print('gt_intra_diversity [{:.3f}±{:.3f}]'.format(results[window_size]['gt_intra_diversity']['mean'], results[window_size]['gt_intra_diversity']['std']))
        print('intra_diversity_gt_diff [{:.3f}±{:.3f}]'.format(results[window_size]['intra_diversity_gt_diff']['mean'], results[window_size]['intra_diversity_gt_diff']['std']))
        print('inter_diversity [{:.3f}±{:.3f}]'.format(results[window_size]['inter_diversity']['mean'], results[window_size]['inter_diversity']['std']))
        print('SiFID [{:.3f}±{:.3f}]'.format(results[window_size]['sifid']['mean'], results[window_size]['sifid']['std']))

    return results


def main(args):
    fixseed(args.seed)
    num_samples_limit = 50  # 100
    args.batch_size = num_samples_limit
    replication_times = 2  # 5
    dist_util.setup_dist(args.device)
    logger.configure()
    logger.log("creating data loader...")
    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, None)
    logger.log(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)
    if not hasattr(args, 'log_file'):
        log_file = os.path.join(os.path.dirname(args.model_path),
                                os.path.basename(args.model_path).replace('model', 'eval_').replace('.pt', '.log'))
    else:
        log_file = args.log_file
    print(f'Will save to log file [{log_file}]')
    model.to(dist_util.dev())
    diffusion.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()  # disable random masking
    logger.log(f"Loading evaluator")
    eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
    eval_dict = evaluate(args, model, diffusion, eval_wrapper, num_samples_limit, replication_times)
    with open(log_file, 'w') as fw:
        fw.write(str(eval_dict))
    np.save(log_file.replace('.log', '.npy'), eval_dict)


if __name__ == '__main__':
    args = evaluation_parser()
    main(args)