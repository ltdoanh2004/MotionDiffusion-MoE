from datetime import datetime
import numpy as np
import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
 # Add parent directory to path
from datasets1 import get_dataset_motion_loader, get_motion_loader
from models import MotionTransformer
from utils.get_opt import get_opt
from utils.metrics import *
from datasets1 import EvaluatorModelWrapper
from collections import OrderedDict
from utils.plot_script import *
from utils.utils import *
from trainers import DDPMTrainer
from sklearn.metrics import mean_absolute_error
from utils.motion_process import recover_from_ric
from os.path import join as pjoin
from utils.word_vectorizer import WordVectorizer
import sys
import tqdm
from torch.utils.data import Dataset, DataLoader
from options.evaluate_options import TestOptions
def build_models(opt, dim_pose):
    encoder = MotionTransformer(
        input_feats=dim_pose,
        num_frames=opt.max_motion_length,
        latent_dim=opt.latent_dim,
        ff_size=256,
        num_layers=opt.num_layers,
        num_heads=4,
        dropout=0.1,
        text_latent_dim=128,
        moe_num_experts=4,
        model_size="small",   # e.g., double dims
        chunk_size=256
    )
    
    return encoder



torch.multiprocessing.set_sharing_strategy('file_system')
# 
def score(trainers, gt_dataset, dim=3, mm_num_samples=100, mm_num_repeats=10):
    """Compute Mean Absolute Error (MAE), Velocity, and Jerk

    Args:
        original:     array containing joint positions of original gesture
        predicted:    array containing joint positions of predicted gesture
        dim:          gesture dimensionality

    Returns:
        mae:          MAE between original and predicted for each joint
        velocity:     Velocity error between original and predicted for each joint
        jerk:         Jerk error between original and predicted for each joint
    """
    dataloader = DataLoader(gt_dataset, batch_size=1, num_workers=0, shuffle=True)
    epoch, it = trainers.load("/iridisfs/scratch/tvtn1c23/ckpt/t2m/test/model/latest.tar")
    trainers.eval_mode()
    trainers.to(opt.device)
    mean = np.load("/iridisfs/scratch/tvtn1c23/ckpt/t2m/test/meta/mean.npy")
    std = np.load("/iridisfs/scratch/tvtn1c23/ckpt/t2m/test/meta/mean.npy")

    mm_generated_motions = []
    mm_idxs = np.random.choice(len(gt_dataset), mm_num_samples, replace=False)
    mm_idxs = np.sort(mm_idxs)
    all_caption = []
    all_m_lens = []
    all_data = []
    all_motion = []
    min_mov_length = 10
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
            # word_emb, pos_ohot, caption, cap_lens, motions, m_lens = data
            all_data.append(data)
            # tokens = tokens[0].split('_')
            mm_num_now = len(mm_generated_motions)
            is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False
            repeat_times = mm_num_repeats if is_mm else 1
            # repeat_times = 1
            m_lens = max(m_lens // opt.unit_length * opt.unit_length, min_mov_length * opt.unit_length)
            m_lens = min(m_lens, opt.max_motion_length)
            if isinstance(m_lens, int):
                m_lens = torch.LongTensor([m_lens]).to(opt.device)
            else:
                m_lens = m_lens.to(opt.device)
            for t in range(repeat_times):
                all_m_lens.append(m_lens)
                all_caption.extend(caption)
                all_motion.append(motions)  # Adjusting motion dimensions to match predicted
            if is_mm:
                mm_generated_motions.append(0)
    all_m_lens = torch.stack(all_m_lens)

    # Generate all sequences
    with torch.no_grad():
        predicted = trainers.generate(all_caption, all_m_lens, opt.dim_pose)
        predicted = torch.stack(predicted , dim=0).cpu().numpy()
        predicted = predicted * std + mean
        predicted = recover_from_ric(torch.from_numpy(predicted).float(), 22).numpy()
        original = np.concatenate(all_motion, axis=0)
        original = original * std + mean
        original = recover_from_ric(torch.from_numpy(original).float(), 22).numpy()
        print(original.shape) 
        # num_frames, num_joints, dim = predicted.shape[1], predicted.shape[2], predicted.shape[3]
        assert not np.isnan(predicted).any(), "NaN values detected in predicted array"
        assert not np.isnan(original).any(), "NaN values detected in original array"

        # MAE calculation
        mae = np.mean(np.abs(predicted - original), axis=(1, 2, 3))
        pae = np.mean(np.abs(predicted - original), axis=(0, 1, 2))

        
        predicted = torch.from_numpy(predicted)
        original = torch.from_numpy(original)


        velocity_predicted = torch.diff(predicted, dim=1)  # Shape: [batch_size, 195, 22, 3]
        velocity_original = torch.diff(original, dim=1)

        # Calculate Velocity Error (MAE between velocity_predicted and velocity_original)
        velocity_error = torch.mean(torch.abs(velocity_predicted - velocity_original), dim=(1, 2, 3))
        velocity_error = torch.mean(velocity_error)  # Averaging over the batch

        # Jerk Calculation (Third Derivative)
        # Apply diff three times to get the third derivative
        jerk_predicted = torch.diff(velocity_predicted, dim=1)  # Shape: [batch_size, 194, 22, 3]
        jerk_original = torch.diff(velocity_original, dim=1)

        # Calculate Jerk Error (MAE between jerk_predicted and jerk_original)
        jerk_error = torch.mean(torch.abs(jerk_predicted - jerk_original), dim=(1, 2, 3))
        jerk_error = torch.mean(jerk_error)  # Averaging over the batch


   
    return mae, velocity_error, jerk_error, pae



def evaluate_matching_score(motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    # print(motion_loaders.keys())
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                assert not np.isnan(motion_embeddings.cpu().numpy()).any(), "Motion embeddings contain NaN values"
                assert not np.isnan(word_embeddings.cpu().numpy()).any(), "word_embeddings contain NaN values"
                assert not np.isnan(pos_one_hots.cpu().numpy()).any(), "pos_one_hots contain NaN values"
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                assert not np.isnan(text_embeddings.cpu().numpy()).any(), "Text embeddings contain NaN values"
                assert not np.isnan(motion_embeddings.cpu().numpy()).any(), "Motion embeddings contain NaN values"
                assert not np.isinf(text_embeddings.cpu().numpy()).any(), "Text embeddings contain Inf values"
                assert not np.isinf(motion_embeddings.cpu().numpy()).any(), "Motion embeddings contain Inf values"

                # Ensure distance matrix is valid
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(), motion_embeddings.cpu().numpy())
                assert not np.isnan(dist_mat).any(), "Distance matrix contains NaN values"
                assert not np.isinf(dist_mat).any(), "Distance matrix contains Inf values"

                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)
                print(text_embeddings.shape)
                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')  # Start of FID evaluation

    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            try:
                # Extract the batch data
                _, _, _, sent_lens, motions, m_lens, _ = batch

                # Debug: Check shapes and data types of inputs
                print(f"Batch {idx} - motions shape: {motions.shape}, m_lens: {m_lens}")

                # Compute motion embeddings
                motion_embeddings = eval_wrapper.get_motion_embeddings(
                    motions=motions,
                    m_lens=m_lens
                )

                # Debug: Check for NaN or Inf values in embeddings
                assert not torch.isnan(motion_embeddings).any(), f"NaN detected in motion_embeddings for batch {idx}"
                assert not torch.isinf(motion_embeddings).any(), f"Inf detected in motion_embeddings for batch {idx}"

                # Collect embeddings
                gt_motion_embeddings.append(motion_embeddings.cpu().numpy())

            except Exception as e:
                print(f"Error processing batch {idx}: {e}")
                continue  # Skip this batch if there's an error

    try:
        # Concatenate all motion embeddings
        gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)

        # Debug: Check shape of concatenated embeddings
        print(f"Concatenated gt_motion_embeddings shape: {gt_motion_embeddings.shape}")

        # Calculate mean and covariance of ground truth embeddings
        gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

        # Debug: Check for NaN or Inf values in statistics
        assert not np.isnan(gt_mu).any(), "NaN detected in gt_mu"
        assert not np.isinf(gt_mu).any(), "Inf detected in gt_mu"
        assert not np.isnan(gt_cov).any(), "NaN detected in gt_cov"
        assert not np.isinf(gt_cov).any(), "Inf detected in gt_cov"

    except Exception as e:
        print(f"Error calculating ground truth statistics: {e}")
        return eval_dict  # Return empty dict if there's an error

    # Iterate through each model's motion embeddings
    for model_name, motion_embeddings in activation_dict.items():
        try:
            # Debug: Check shape of model embeddings
            print(f"Model '{model_name}' - motion_embeddings shape: {motion_embeddings.shape}")

            # Calculate mean and covariance of model embeddings
            mu, cov = calculate_activation_statistics(motion_embeddings)

            # Debug: Check for NaN or Inf values in statistics
            assert not np.isnan(mu).any(), f"NaN detected in mu for model '{model_name}'"
            assert not np.isinf(mu).any(), f"Inf detected in mu for model '{model_name}'"
            assert not np.isnan(cov).any(), f"NaN detected in cov for model '{model_name}'"
            assert not np.isinf(cov).any(), f"Inf detected in cov for model '{model_name}'"

            # Calculate FID
            fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
            print(f'---> [{model_name}] FID: {fid:.4f}')
            print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
            eval_dict[model_name] = fid

        except Exception as e:
            print(f"Error calculating FID for model '{model_name}': {e}")
            continue  # Skip this model if there's an error

    return eval_dict



def evaluate_diversity(activation_dict, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(mm_motion_loaders, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(log_file):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({})})
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mm_score_dict = evaluate_multimodality(mm_motion_loaders, f)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]

            for key, item in mm_score_dict.items():
                if key not in all_metrics['MultiModality']:
                    all_metrics['MultiModality'][key] = [item]
                else:
                    all_metrics['MultiModality'][key] += [item]


        # print(all_metrics['Diversity'])
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values))
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)


if __name__ == '__main__':
    mm_num_samples = 100
    mm_num_repeats = 30
    mm_num_times = 10
    diversity_times = 300
    replication_times = 20
    batch_size = 512
    # opt_path = sys.argv[1]
    parser = TestOptions()
    opt = parser.parse()
    # opt_path = '//iridisfs/scratch/tvtn1c23//HumanML3D/HumanML3D/test.txt
    # dataset_opt_path = opt_path
    dataset_opt_path =  "/iridisfs/scratch/tvtn1c23/ckpt/t2m/test/opt.txt"
    opt_path = "/iridisfs/scratch/tvtn1c23/ckpt/t2m/test/opt.txt"
    # dataset_opt_path ='//iridisfs/scratch/tvtn1c23//HumanML3D/HumanML3D/test.txt'

    # try:
    #     device_id = int(sys.argv[2])
    # except:
    #     device_id = 0
    device_id = 0
    device = torch.device('cuda:%d' % device_id if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)
    mean = np.load('/iridisfs/scratch/tvtn1c23/HumanML3D/HumanML3D/Mean.npy')
    std = np.load('/iridisfs/scratch/tvtn1c23/HumanML3D/HumanML3D/Std.npy')
    opt = get_opt(opt_path, device)
    encoder = build_models(opt, opt.dim_pose)
    trainer = DDPMTrainer(opt, encoder)
    val_split_file = '/iridisfs/scratch/tvtn1c2/HumanML3D/HumanML3D/val.txt'
    path = '/iridisfs/scratch/tvtn1c23/MotionDiffuse/glove'
    w_vectorizer = WordVectorizer(path, 'our_vab')
    # gt_dataset = Text2MotionDataset(opt, mean, std, val_split_file, 50 ,w_vectorizer = w_vectorizer ,eval_mode=True)
    gt_loader, gt_dataset = get_dataset_motion_loader(dataset_opt_path, batch_size, device)
    with open('/iridisfs/scratch/tvtn1c23/MotionDiffuse/text2motion/output.txt', 'a') as f:
        print("BEGIN_____________________________________________________BEGIN", file=f)
        print("_____________________________________________________________", file=f)
        print("MAE score", file=f)
        mae , velocity_error, jerk_error, pae = score(trainer, gt_dataset, dim =3)
        print(mae, file=f)
        print("_____________________________________________________________", file=f)
        print("velocity_error ", file=f)
        print(velocity_error, file=f)
        print("_____________________________________________________________", file=f)
        print("jerk_error", file=f)
        print( jerk_error , file=f)
        print("_____________________________________________________________", file=f)
        print("PAE score", file=f)
        print(pae, file=f)
        print("END_____________________________________________________END", file=f)
    
    
    
    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    eval_motion_loaders = {
        'text2motion': lambda: get_motion_loader(
            opt,
            batch_size,
            trainer,
            gt_dataset,
            mm_num_samples,
            mm_num_repeats
        )
    }
    
    log_file = '/iridisfs/scratch/tvtn1c23/MotionDiffuse/t2m_evaluation_512.log'
    evaluation(log_file)
    

    
