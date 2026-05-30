import os
import torch
import numpy as np
import json
import ast
import clip

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
# model.eval()


# def qst_feat_extract(qst):
#     text = clip.tokenize([qst]).to(device)
#     with torch.no_grad():
#         text_features = model.encode_text(text)
#     return text_features

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model.eval()

def qst_feat_extract(qst):
    text = clip.tokenize([qst]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features


def QstCLIP_feat(json_path, dst_qst_path):
    samples = json.load(open(json_path, 'r'))
    ques_vocab = ['<pad>']

    # for sample in samples:
    #     question = sample['question_content'].rstrip().split(' ')
    #     question[-1] = question[-1][:-1]

    #     question_id = sample['question_id']
    #     print("\n")
    #     print("question id:", question_id)

    #     save_file = os.path.join(dst_qst_path, str(question_id) + '.npy')

    #     if os.path.exists(save_file):
    #         print(question_id, "already exists!")
    #         continue

    #     p = 0
    #     for pos in range(len(question)):
    #         if '<' in question[pos]:
    #             question[pos] = ast.literal_eval(sample['templ_values'])[p]
    #             p += 1

    #     for wd in question:
    #         if wd not in ques_vocab:
    #             ques_vocab.append(wd)

    #     question = ' '.join(question)
    #     print(question)

    #     qst_feat = qst_feat_extract(question)
    #     print(qst_feat.shape)

    #     qst_features = qst_feat.float().cpu().numpy()
    #     np.save(save_file, qst_features)
    for sample in samples:
        try:
            question = sample['question_content'].rstrip().split(' ')
            question[-1] = question[-1][:-1]

            question_id = sample['question_id']
            print("\nquestion id:", question_id)

            save_file = os.path.join(dst_qst_path, str(question_id) + '.npy')
            if os.path.exists(save_file):
                print(question_id, "already exists!")
                continue

            p = 0
            templ_values = ast.literal_eval(sample['templ_values'])
            for pos in range(len(question)):
                if '<' in question[pos]:
                    question[pos] = templ_values[p]
                    p += 1

            question = ' '.join(question)
            print(question)

            qst_feat = qst_feat_extract(question)
            print(qst_feat.shape)

            qst_features = qst_feat.float().cpu().numpy()
            np.save(save_file, qst_features)

        except Exception as e:
            print("failed on question_id:", sample.get("question_id"), "error:", e)
            continue


# if __name__ == "__main__":
#     json_path = '/DATA/ramendra_2511ai39/TSPM/dataset/split_que_id/music_avqa.json'
#     dst_qst_path = "/DATA/ramendra_2511ai39/TSPM/clip_word/"
#     os.makedirs(dst_qst_path, exist_ok=True)
#     QstCLIP_feat(json_path, dst_qst_path)
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dataset/split_que_id"))
    dst_qst_path = os.path.abspath("./clip_word")
    os.makedirs(dst_qst_path, exist_ok=True)

    for name in ["music_avqa_train.json", "music_avqa_val.json", "music_avqa_test.json"]:
        json_path = os.path.join(base_dir, name)
        if os.path.exists(json_path):
            print("processing:", json_path)
            QstCLIP_feat(json_path, dst_qst_path)
        else:
            print("missing:", json_path)