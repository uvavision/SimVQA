# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------
## base code modified to train using synthetic generated images, questions and answers

from core.data.data_utils import img_feat_path_load, img_feat_load, ques_load, tokenize, ans_stat
from core.data.data_utils import proc_img_feat, proc_ques, proc_ans

import numpy as np
import glob, json, torch, time
import torch.utils.data as Data
import pickle
import marshal


class DataSet(Data.Dataset):
    def __init__(self, __C):
        self.__C = __C


        # --------------------------
        # ---- Raw data loading ----
        # --------------------------

        # Loading all image paths
        # if self.__C.PRELOAD:
        self.img_feat_path_list = []
        # pcascante: instead, always add preprocessed FRCNN paths
        # pcascante: add paths for bottom up features for all splits, even if not used due to new skill-sets and augmented set
        for split in ['train', 'val', 'test']:
            self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH[split] + '*.npz')

        # Loading question word list
        self.stat_ques_list = \
            json.load(open(__C.QUESTION_PATH['train'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['val'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['test'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['vg'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['hypersim_whole_counting'], 'r'))['questions'] # + \
            # json.load(open(__C.QUESTION_PATH['hypersim_train_position'], 'r'))['questions']

        # Loading answer word list
        # self.stat_ans_list = \
        #     json.load(open(__C.ANSWER_PATH['train'], 'r'))['annotations'] + \
        #     json.load(open(__C.ANSWER_PATH['val'], 'r'))['annotations']

        # breakpoint()
        # Loading question and answer list
        self.ques_list = []
        self.ans_list = []

        ## original: base code
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        ## pcascante: use only train split 
        print ('split_list', split_list)
        print ('split_list', split_list)
        print ('split_list', split_list)
        for split in split_list:
            self.ques_list += json.load(open(__C.QUESTION_PATH[split], 'r'))['questions']
            # if __C.RUN_MODE in ['train']:
            # pcascante: load ans always
            self.ans_list += json.load(open(__C.ANSWER_PATH[split], 'r'))['annotations']

        # breakpoint()
        # Define run data size
        if __C.RUN_MODE in ['train']:
            self.data_size = self.ans_list.__len__()
        else:
            self.data_size = self.ques_list.__len__()

        print('== Dataset size:', self.data_size)
        # breakpoint()

        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        
        # {image id} -> {image feature absolutely path}
        if self.__C.PRELOAD:
            print('==== Pre-Loading features ...')
            time_start = time.time()
            self.iid_to_img_feat = img_feat_load(self.img_feat_path_list)
            time_end = time.time()
            print('==== Finished in {}s'.format(int(time_end-time_start)))
        else:
            ## original: base code
            self.iid_to_img_feat_path = img_feat_path_load(self.img_feat_path_list)
            ## pcascante: override this with a custom dictionary of ids and paths from hypersym
            ## eg: {'233961': '/bigtemp/pc9za/VQA/VQAv2/coco_extract/val2014/COCO_val2014_000000233961.jpg.npz', '11184': '/bigtemp/pc9za/VQA/VQAv2/coco_extract/val2014/COCO_val2014_000000011184.jpg.npz', '455225': '/bigtemp/pc9za/VQA/VQAv2/coco_extract/val2014/COCO_val2014_000000455225.jpg.npz', '384136': '/bigtemp/pc9za/VQA/VQAv2/coco_extract/val2014/COCO_val2014_000000384136.jpg.npz'}
            # self.iid_to_img_feat_path_hypersim = pickle.load( open( "~/SyntheticBeta/annotations_ui/hypersym_id_feats_mapping_onlyCounts.p", "rb" ))
            self.iid_to_img_feat_path_hypersim = pickle.load( open( "synth_mapping/hypersym_id_feats_mapping_onlyCounts_450370.p", "rb" ))
            self.iid_to_img_feat_path_hypersim_position = pickle.load( open( "synth_mapping/hypersym_id_feats_mapping_positions_closest.p", "rb" ))
            # self.iid_to_img_feat_path_hypersim = pickle.load( open( "~/SyntheticBeta/annotations_ui/hypersym_id_feats_mapping_onlyCounts_nyu40.p", "rb" ))
            self.iid_to_img_feat_path_tdw = pickle.load( open( "synth_mapping/TDW_id_feats_mapping_onlyCounts_33264.p", "rb" ))
            

        # {question id} -> {question}
        self.qid_to_ques = ques_load(self.ques_list)

        # Tokenize
        self.token_to_ix, self.pretrained_emb = tokenize(self.stat_ques_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)

        # Answers statistic
        # Make answer dict during training does not guarantee
        # the same order of {ans_to_ix}, so we published our
        # answer dict to ensure that our pre-trained model
        # can be adapted on each machine.

        # Thanks to Licheng Yu (https://github.com/lichengunc)
        # for finding this bug and providing the solutions.

        # self.ans_to_ix, self.ix_to_ans = ans_stat(self.stat_ans_list, __C.ANS_FREQ)
        self.ans_to_ix, self.ix_to_ans = ans_stat('core/data/answer_dict.json')
        self.ans_size = self.ans_to_ix.__len__()
        print('== Answer vocab size (occurr more than {} times):'.format(8), self.ans_size)
        print('Finished!')
        print('')

        inf = open('~/BOTTOM_UP/bottom-up-attention/train_coco_pt1_marshal.p', 'rb')
        train_coco_pt1 = pickle.load(inf)
        inf.close()
        inf = open('~/BOTTOM_UP/bottom-up-attention/train_coco_pt2_marshal.p', 'rb')
        train_coco_pt2 = pickle.load(inf)
        inf.close()
        inf = open('~/BOTTOM_UP/bottom-up-attention/train_coco_pt3_marshal.p', 'rb')
        train_coco_pt3 = pickle.load(inf)
        inf.close()

        # merge multiple dictionaries into one
        self.coco_features = {**train_coco_pt1, **train_coco_pt2, **train_coco_pt3}
        # self.coco_features = train_coco_pt1
        self.list_coco_objects = list(self.coco_features.keys())

        inf = open('~/BOTTOM_UP/bottom-up-attention/hypersim_pt1_marshal.p', 'rb')
        hypersim_pt1 = pickle.load(inf)
        inf.close()
        inf = open('~/BOTTOM_UP/bottom-up-attention/hypersim_pt2_marshal.p', 'rb')
        hypersim_pt2 = pickle.load(inf)
        inf.close()
        inf = open('~/BOTTOM_UP/bottom-up-attention/hypersim_pt3_marshal.p', 'rb')
        hypersim_pt3 = pickle.load(inf)
        inf.close()
        inf = open('~/BOTTOM_UP/bottom-up-attention/hypersim_pt4_marshal.p', 'rb')
        hypersim_pt4 = pickle.load(inf)
        inf.close()
        # with open('~/BOTTOM_UP/bottom-up-attention/TDW_Objects_mm_craftroom_1a.p', 'rb') as f:
        #     u = pickle._Unpickler(f)
        #     u.encoding = 'latin1'
        #     hypersim_pt1 = u.load()

        # merge multiple dictionaries into one
        self.synth_features = {**hypersim_pt1, **hypersim_pt2, **hypersim_pt3, **hypersim_pt4}
        # self.synth_features = {**hypersim_pt1}
        # self.synth_features = hypersim_pt1
        self.list_synth_objects = list(self.synth_features.keys())

        # breakpoint()
        # user press anykey to continue
        # input("Press any key to continue...")

    def __getitem__(self, idx):

        # For code safety
        img_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)
        # this code does a lazy try/catch/try/catch between sets -- should avoid this and use switch/case impl.

        # Process ['train'] and ['val', 'test'] respectively
        if self.__C.RUN_MODE in ['train']:
            # Load the run data from list
            ans = self.ans_list[idx]
            ques = self.qid_to_ques[str(ans['question_id'])]

            try:
                with open("~/BOTTOM_UP/bottom-up-attention/COCO_FEATS/{}.p".format(ans['image_id']), 'rb') as f:
                    img_feat_x = pickle.load(f, encoding='latin1')
                # create a copy of img_feat_orig, then work on the copy
                # img_feat_x = img_feat_orig.copy()
                alpha = 1.0
                lam = np.random.beta(alpha, alpha)
                if lam > 0.5:
                    # take objects that exists in both sets
                    real_objects = [k for k in img_feat_x.keys() if k in self.list_synth_objects]      
                    # randmly select 20% of the objects to be replaced
                    random_indexes = np.random.choice(len(real_objects), int(len(real_objects)*0.2), replace=False)
                    # replace the selected objects with the objects from the other set
                    for rnd_idx in random_indexes:
                        len_of_possibilities = len(self.synth_features[real_objects[rnd_idx]])
                        len_of_features = len(img_feat_x[real_objects[rnd_idx]])
                        indexes_to_replace = np.random.choice(len_of_possibilities, len_of_features)
                        for replace_i in range (len_of_features):
                            img_feat_x[real_objects[rnd_idx]][replace_i] = self.synth_features[real_objects[rnd_idx]][indexes_to_replace[replace_i]]

                # now append all features to an array:
                img_feat_x_tmp = []
                for k,v in img_feat_x.items():
                    for iv in v:
                        img_feat_x_tmp.append(iv[2])
                img_feat_x_tmp = np.array(img_feat_x_tmp)
                img_feat_iter = proc_img_feat(img_feat_x_tmp, self.__C.IMG_FEAT_PAD_SIZE)
            except:
                try:
                    with open("~/BOTTOM_UP/bottom-up-attention/HYPERSIM_FEATS/{}".format(self.iid_to_img_feat_path_hypersim[str(ans['image_id'])]), 'rb') as f:
                        img_feat_x = pickle.load(f, encoding='latin1')
                    alpha = 1.0
                    lam = np.random.beta(alpha, alpha)
                    if lam > 0.5:
                        # take objects that exists in both sets
                        real_objects = [k for k in img_feat_x.keys() if k in self.list_coco_objects]      
                        # randmly select 20% of the objects to be replaced
                        random_indexes = np.random.choice(len(real_objects), int(len(real_objects)*0.2), replace=False)
                        # replace the selected objects with the objects from the other set
                        for rnd_idx in random_indexes:
                            len_of_possibilities = len(self.coco_features[real_objects[rnd_idx]])
                            len_of_features = len(img_feat_x[real_objects[rnd_idx]])
                            indexes_to_replace = np.random.choice(len_of_possibilities, len_of_features)
                            for replace_i in range (len_of_features):
                                img_feat_x[real_objects[rnd_idx]][replace_i] = self.coco_features[real_objects[rnd_idx]][indexes_to_replace[replace_i]]

                    # now append all features to an array:
                    img_feat_x_tmp = []
                    for k,v in img_feat_x.items():
                        for iv in v:
                            img_feat_x_tmp.append(iv[2])

                    img_feat_x_tmp = np.array(img_feat_x_tmp)
                    img_feat_iter = proc_img_feat(img_feat_x_tmp, self.__C.IMG_FEAT_PAD_SIZE)
                except:
                    img_feat_x = pickle.load( open('~/BOTTOM_UP/bottom-up-attention/TDW_FEATS2/' + self.iid_to_img_feat_path_tdw[str(ans['image_id'])], "rb"), encoding='latin1')

                    alpha = 1.0
                    lam = np.random.beta(alpha, alpha)
                    if lam > 0.5:
                        # take objects that exists in both sets
                        real_objects = [k for k in img_feat_x.keys() if k in self.list_coco_objects]      
                        # randmly select 20% of the objects to be replaced
                        random_indexes = np.random.choice(len(real_objects), int(len(real_objects)*0.2), replace=False)
                        # replace the selected objects with the objects from the other set
                        for rnd_idx in random_indexes:
                            len_of_possibilities = len(self.coco_features[real_objects[rnd_idx]])
                            len_of_features = len(img_feat_x[real_objects[rnd_idx]])
                            indexes_to_replace = np.random.choice(len_of_possibilities, len_of_features)
                            for replace_i in range (len_of_features):
                                img_feat_x[real_objects[rnd_idx]][replace_i] = self.coco_features[real_objects[rnd_idx]][indexes_to_replace[replace_i]]

                    # now append all features to an array:
                    img_feat_x_tmp = []
                    for k,v in img_feat_x.items():
                        for iv in v:
                            img_feat_x_tmp.append(iv[2])

                    img_feat_x_tmp = np.array(img_feat_x_tmp)
                    img_feat_iter = proc_img_feat(img_feat_x_tmp, self.__C.IMG_FEAT_PAD_SIZE)

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)

            # Process answer
            ans_iter = proc_ans(ans, self.ans_to_ix)

        else:
            # Load the run data from list
            ques = self.ques_list[idx]
            # pcascante: get corresponding answer from list
            ans = self.ans_list[idx]

            # Process image feature from (.npz) file
            if self.__C.PRELOAD:
                img_feat_x = self.iid_to_img_feat[str(ques['image_id'])]
            else:
                try:
                    img_feat = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
                    img_feat_x = img_feat['x'].transpose((1, 0))
                except:
                    ## pcascante: when loading hypersim features, add path to filename
                    ## pcascante: when loading hypersim features, no transpose is needed
                    try:
                        img_feat_x = np.load('~/HYPERSIM/bottom_up_features/hypersim/' + self.iid_to_img_feat_path_hypersim[str(ans['image_id'])])
                    except:
                        img_feat_x = np.load('~/HYPERSIM/bottom_up_features/hypersim/' + self.iid_to_img_feat_path_hypersim_position[str(ans['image_id'])])
            img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)


        return torch.from_numpy(img_feat_iter), \
               torch.from_numpy(ques_ix_iter), \
               torch.from_numpy(ans_iter)


    def __len__(self):
        return self.data_size


