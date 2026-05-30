import os
import ast
import json
import numpy as np
import torch

from torch.utils.data import Dataset


def ids_to_multinomial(id, categories):
    id_to_idx = {id: index for index, id in enumerate(categories)}
    return id_to_idx[id]


class AVQA_dataset(Dataset):
    def __init__(self, args, label,
                 audios_feat_dir, visual_feat_dir,
                 audios_patch_dir, visual_patch_dir,
                 qst_prompt_dir, qst_feat_dir,
                 transform=None, mode_flag='train'):

        self.args = args
        self.mode_flag = mode_flag

        if mode_flag == 'train':
            vocab_label_path = args.label_train
        elif mode_flag == 'val':
            vocab_label_path = args.label_val
        else:
            vocab_label_path = args.label_test

        with open(vocab_label_path, 'r') as f:
            samples = json.load(f)

        ques_vocab = ['<pad>']
        ans_vocab = []

        for sample in samples:
            question = sample['question_content'].rstrip().split(' ')
            question[-1] = question[-1][:-1]

            p = 0
            templ_values = ast.literal_eval(sample['templ_values'])
            for pos in range(len(question)):
                if '<' in question[pos]:
                    question[pos] = templ_values[p]
                    p += 1

            for wd in question:
                if wd not in ques_vocab:
                    ques_vocab.append(wd)

            if sample['anser'] not in ans_vocab:
                ans_vocab.append(sample['anser'])

        self.ques_vocab = ques_vocab
        self.ans_vocab = ans_vocab
        self.word_to_ix = {word: i for i, word in enumerate(self.ques_vocab)}

        with open(label, 'r') as f:
            self.samples = json.load(f)

        self.max_len = 14

        self.audios_feat_dir = audios_feat_dir
        self.visual_feat_dir = visual_feat_dir
        self.audios_patch_dir = audios_patch_dir
        self.visual_patch_dir = visual_patch_dir
        self.qst_prompt_dir = "/DATA/ramendra_2511ai39/TSPM/clip_feats/qaPrompt_ViT-L14@336px"
        self.qst_feat_dir = "/DATA/ramendra_2511ai39/TSPM/clip_feats/qaPrompt_ViT-L14@336px"

        self.transform = transform

        # Probe directories to find prototype shapes for missing-file fallbacks
        def _probe_shape(directory):
            try:
                for fn in os.listdir(directory):
                    if fn.endswith('.npy'):
                        arr = np.load(os.path.join(directory, fn))
                        return arr.shape
            except Exception:
                return None

        self.prototype_shapes = {
            'audio': _probe_shape(self.audios_feat_dir),
            'visual': _probe_shape(self.visual_feat_dir),
            'audio_patch': _probe_shape(self.audios_patch_dir),
            'visual_patch': _probe_shape(self.visual_patch_dir),
            'qst_prompt': _probe_shape(self.qst_prompt_dir),
            'qst_feat': _probe_shape(self.qst_feat_dir),
        }

    def __len__(self):
        return len(self.samples)

    def get_lstm_embeddings(self, question_input, sample):
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        p = 0
        templ_values = ast.literal_eval(sample['templ_values'])
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = templ_values[p]
                p += 1

        if len(question) < self.max_len:
            question.extend(['<pad>'] * (self.max_len - len(question)))

        idxs = [self.word_to_ix[w] for w in question]
        ques = torch.tensor(idxs, dtype=torch.long)
        return ques

    def __getitem__(self, idx):
        sample = self.samples[idx]
        name = str(sample['video_id'])
        question_id = str(sample['question_id'])

        audio_feat_path = os.path.join(self.audios_feat_dir, name + '.npy')
        visual_feat_path = os.path.join(self.visual_feat_dir, name + '.npy')
        audio_patch_path = os.path.join(self.audios_patch_dir, name + '.npy')
        visual_patch_path = os.path.join(self.visual_patch_dir, name + '.npy')

        qst_prompt_path = os.path.join(self.qst_prompt_dir, question_id + '.npy')
        qst_feat_path = os.path.join(self.qst_feat_dir, question_id + '.npy')

        # Load features, falling back to zero arrays when files are missing
        def _load_or_zero(path, proto_key):
            if os.path.exists(path):
                return np.load(path)
            proto_shape = self.prototype_shapes.get(proto_key)
            # Expected feature last-dim sizes according to model inputs
            expected_last_dim = {
                'audio': 128,
                'visual': 768,
                'audio_patch': 768,
                'visual_patch': 768,
                'qst_prompt': 768,
                'qst_feat': 768,
            }

            exp_ld = expected_last_dim.get(proto_key)
            if proto_shape is not None:
                # Build a fallback shape that preserves temporal/patch dimensions
                fallback_shape = list(proto_shape)
                # If proto is 1D-like (e.g., question vectors), ensure at least 2D
                if len(fallback_shape) == 1:
                    fallback_shape = [1, fallback_shape[0]]

                # Replace last dim with expected if known, otherwise keep proto
                if exp_ld is not None:
                    fallback_shape[-1] = exp_ld

                print(f"Warning: missing {path}, using zeros with shape {tuple(fallback_shape)}")
                return np.zeros(tuple(fallback_shape), dtype=np.float32)
            return None

        def _pad_truncate_time(arr, target_T):
            # Ensure numpy array and at least 2D
            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            cur_T = arr.shape[0]
            if target_T is None:
                return arr
            if cur_T == target_T:
                return arr
            if cur_T > target_T:
                return arr[:target_T]
            pad_shape = (target_T - cur_T,) + arr.shape[1:]
            return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)
            raise FileNotFoundError(f"Missing feature file and no prototype available: {path}")

        audios_feat = _load_or_zero(audio_feat_path, 'audio')
        visual_feat = _load_or_zero(visual_feat_path, 'visual')
        audios_patch_feat = _load_or_zero(audio_patch_path, 'audio_patch')
        visual_patch_feat = _load_or_zero(visual_patch_path, 'visual_patch')
        question_feat = _load_or_zero(qst_feat_path, 'qst_feat')
        question_prompt = _load_or_zero(qst_prompt_path, 'qst_prompt')

        # Pad/truncate temporal dimension to prototype T when available to allow batching
        proto_visual = self.prototype_shapes.get('visual')
        target_T = None
        if proto_visual is not None and len(proto_visual) >= 1:
            target_T = proto_visual[0]

        if audios_feat is not None:
            audios_feat = _pad_truncate_time(audios_feat, target_T)
        if visual_feat is not None:
            visual_feat = _pad_truncate_time(visual_feat, target_T)
        if audios_patch_feat is not None:
            audios_patch_feat = _pad_truncate_time(audios_patch_feat, target_T)
        if visual_patch_feat is not None:
            visual_patch_feat = _pad_truncate_time(visual_patch_feat, target_T)

        # If patch features are per-frame vectors (T, C), add a patch dimension N=1 -> (T, 1, C)
        if audios_patch_feat is not None and getattr(audios_patch_feat, 'ndim', 0) == 2:
            audios_patch_feat = audios_patch_feat[:, None, :]
        if visual_patch_feat is not None and getattr(visual_patch_feat, 'ndim', 0) == 2:
            visual_patch_feat = visual_patch_feat[:, None, :]

        # Enforce strict feature shapes by last-dim (coerce to expected sizes)
        def _enforce_shape(arr, key_name, expected_ld):
            if arr is None:
                return None
            arr = np.asarray(arr)
            if arr.ndim == 0:
                return None
            last_dim = arr.shape[-1]
            if last_dim == expected_ld:
                return arr
            if last_dim > expected_ld:
                # Slice to expected last-dim
                slices = [slice(None)] * (arr.ndim - 1) + [slice(expected_ld)]
                return arr[tuple(slices)]
            else:
                # Pad to expected last-dim
                pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, expected_ld - last_dim)]
                return np.pad(arr, pad_width, mode='constant', constant_values=0.0)

        audios_feat = _enforce_shape(audios_feat, 'audio', 128)
        visual_feat = _enforce_shape(visual_feat, 'visual', 768)
        audios_patch_feat = _enforce_shape(audios_patch_feat, 'audio_patch', 768)
        visual_patch_feat = _enforce_shape(visual_patch_feat, 'visual_patch', 768)
        question_feat = _enforce_shape(question_feat, 'qst_feat', 768)
        question_prompt = _enforce_shape(question_prompt, 'qst_prompt', 768)

        answer = sample['anser']
        answer_label = ids_to_multinomial(answer, self.ans_vocab)
        answer_label = torch.tensor(answer_label, dtype=torch.long)

        item = {
            'video_name': name,
            'audios_feat': audios_feat,
            'visual_feat': visual_feat,
            'audios_patch_feat': audios_patch_feat,
            'visual_patch_feat': visual_patch_feat,
            'question_feat': question_feat,
            'question_prompt': question_prompt,
            'answer_label': answer_label,
            'question_id': question_id
        }

        if self.transform:
            item = self.transform(item)

        return item


class ToTensor(object):
    def __call__(self, sample):
        return {
            'video_name': sample['video_name'],
            'audios_feat': torch.from_numpy(sample['audios_feat']).float(),
            'visual_feat': torch.from_numpy(sample['visual_feat']).float(),
            'audios_patch_feat': torch.from_numpy(sample['audios_patch_feat']).float(),
            'visual_patch_feat': torch.from_numpy(sample['visual_patch_feat']).float(),
            'question_feat': torch.from_numpy(sample['question_feat']).float(),
            'question_prompt': torch.from_numpy(sample['question_prompt']).float(),
            'answer_label': sample['answer_label'].long(),
            'question_id': sample['question_id']
        }