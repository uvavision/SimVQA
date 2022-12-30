# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os

class PATH:
    def __init__(self):

        # vqav2 dataset root path
        # self.DATASET_PATH = './datasets/vqa/'
        self.DATASET_PATH = '~/data/VQAv2/vqa/'

        # bottom up features root path
        # self.FEATURE_PATH = './datasets/coco_extract/'
        self.FEATURE_PATH = '~/data/VQAv2/coco_extract/'

        self.init_path()


    def init_path(self):

        self.IMG_FEAT_PATH = {
            ## original: base code
            'train': self.FEATURE_PATH + 'train2014/',
            ## pcascante: add feature path for hypersym
            'hypersim_train': '~/data/HYPERSIM/bottom_up_features/hypersim/',
            'val': self.FEATURE_PATH + 'val2014/',
            'test': self.FEATURE_PATH + 'test2015/',
        }

        self.QUESTION_PATH = {
            ## original: base code
            'train': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions.json',
            # 'train': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions_no_counting.json',
            'train_only_counting_v1': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions_only_counting.json',
            'train_no_counting_v1': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions_no_counting.json',
            'train_only_counting_v2': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions_only_counting_v2.json',
            'train_no_counting_v2': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions_no_counting_v2.json',
            'train_only_counting_v3': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions_only_counting_v3.json',
            'train_no_counting_v3': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions_no_counting_v3.json',
            'train_only_counting_0.01': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions_only_counting_chunk_0.01.json',
            'train_only_counting_0.05': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions_only_counting_chunk_0.05.json',
            'train_only_counting_0.1': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions_only_counting_chunk_0.1.json',
            'train_only_counting_0.25': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions_only_counting_chunk_0.25.json',
            'train_only_counting_0.5': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions_only_counting_chunk_0.5.json',
            'train_only_counting_0.75': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions_only_counting_chunk_0.75.json',
            'train_only_counting_0.9': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions_only_counting_chunk_0.9.json',
            'train_only_nyu40Labels': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions_with_nyu40Labels.json', # only counting
            'val_only_nyu40Labels': self.DATASET_PATH + 'v2_OpenEnded_mscoco_val2014_questions_with_nyu40Labels.json', # only counting
            ## pcascante: add question path for hypersym
            'hypersim_whole_counting': "~/data/hypersym_train_onlyCounts_450370.json",
            'hypersim_count_train': "~/data/hypersym_train_questions_onlyCounts_450370.json",
            'hypersim_count_val': "~/data/hypersym_val_questions_onlyCounts_450370.json",
            'hypersim_nyu40_count_train': "~/data/hypersym_train_questions_onlyCounts_nyu40.json",
            'hypersim_nyu40_count_val': "~/data/hypersym_val_questions_onlyCounts_nyu40.json",
            'hypersim_train_position': "~/data/hypersym_train_positions_closest.json",
            'tdw_count_train': "~/data/tdw_train_questions_onlyCounts.json",
            'tdw_count_val': "~/data/tdw_val_questions_onlyCounts.json",
            'tdw_count_train_33264': "~/data/tdw_train_questions_onlyCounts_33264.json",
            'tdw_count_val_33264': "~/data/tdw_val_questions_onlyCounts_33264.json",
            'val': self.DATASET_PATH + 'v2_OpenEnded_mscoco_val2014_questions.json',
            'test': self.DATASET_PATH + 'v2_OpenEnded_mscoco_test2015_questions.json',
            'vg': self.DATASET_PATH + 'VG_questions.json',
        }

        self.ANSWER_PATH = {
            ## original: base code
            'train': self.DATASET_PATH + 'v2_mscoco_train2014_annotations.json',
            # 'train': self.DATASET_PATH + 'v2_mscoco_train2014_annotations_no_counting.json',
            'train_only_counting_v1': self.DATASET_PATH + 'v2_mscoco_train2014_annotations_only_counting.json',
            'train_no_counting_v1': self.DATASET_PATH + 'v2_mscoco_train2014_annotations_no_counting.json',
            'train_only_counting_v2': self.DATASET_PATH + 'v2_mscoco_train2014_annotations_only_counting_v2.json',
            'train_no_counting_v2': self.DATASET_PATH + 'v2_mscoco_train2014_annotations_no_counting_v2.json',
            'train_only_counting_v3': self.DATASET_PATH + 'v2_mscoco_train2014_annotations_only_counting_v3.json',
            'train_no_counting_v3': self.DATASET_PATH + 'v2_mscoco_train2014_annotations_no_counting_v3.json',
            'train_only_counting_0.01': self.DATASET_PATH + 'v2_mscoco_train2014_annotations_only_counting_chunk_0.01.json',
            'train_only_counting_0.05': self.DATASET_PATH + 'v2_mscoco_train2014_annotations_only_counting_chunk_0.05.json',
            'train_only_counting_0.1': self.DATASET_PATH + 'v2_mscoco_train2014_annotations_only_counting_chunk_0.1.json',
            'train_only_counting_0.25': self.DATASET_PATH + 'v2_mscoco_train2014_annotations_only_counting_chunk_0.25.json',
            'train_only_counting_0.5': self.DATASET_PATH + 'v2_mscoco_train2014_annotations_only_counting_chunk_0.5.json',
            'train_only_counting_0.75': self.DATASET_PATH + 'v2_mscoco_train2014_annotations_only_counting_chunk_0.75.json',
            'train_only_counting_0.9': self.DATASET_PATH + 'v2_mscoco_train2014_annotations_only_counting_chunk_0.9.json',
            'train_only_nyu40Labels': self.DATASET_PATH + 'v2_mscoco_train2014_annotations_with_nyu40Labels.json', # only counting
            'val_only_nyu40Labels': self.DATASET_PATH + 'v2_mscoco_val2014_annotations_with_nyu40Labels.json', # only counting
            ## pcascante: add question path for hypersym
            'hypersim_whole_counting': "~/data/hypersym_answers_onlyCounts_450370.json",
            'hypersim_count_train': "~/data/hypersym_train_answers_onlyCounts_450370.json",
            'hypersim_count_val': "~/data/hypersym_tval_answers_onlyCounts_450370.json",
            'hypersim_nyu40_count_train': "~/data/hypersym_train_answers_onlyCounts_nyu40.json",
            'hypersim_nyu40_count_val': "~/data/hypersym_val_answers_onlyCounts_nyu40.json",
            'hypersim_train_position': "~/data/hypersym_answers_positions_closest.json",
            'tdw_count_train': "~/data/tdw_train_answers_onlyCounts.json",
            'tdw_count_val': "~/data/tdw_val_answers_onlyCounts.json",
            'tdw_count_train_33264': "~/data/tdw_train_answers_onlyCounts_33264.json",
            'tdw_count_val_33264': "~/data/tdw_val_answers_onlyCounts_33264.json",
            'val': self.DATASET_PATH + 'v2_mscoco_val2014_annotations.json',
            'vg': self.DATASET_PATH + 'VG_annotations.json',
        }

        self.RESULT_PATH = './results/result_test/'
        self.PRED_PATH = './results/pred/'
        self.CACHE_PATH = './results/cache/'
        self.LOG_PATH = './results/log/'
        self.CKPTS_PATH = './ckpts/'

        if 'result_test' not in os.listdir('./results'):
            os.mkdir('./results/result_test')

        if 'pred' not in os.listdir('./results'):
            os.mkdir('./results/pred')

        if 'cache' not in os.listdir('./results'):
            os.mkdir('./results/cache')

        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')

        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')


    def check_path(self):
        print('Checking dataset ...')

        for mode in self.IMG_FEAT_PATH:
            if not os.path.exists(self.IMG_FEAT_PATH[mode]):
                print(self.IMG_FEAT_PATH[mode] + 'NOT EXIST')
                exit(-1)

        for mode in self.QUESTION_PATH:
            if not os.path.exists(self.QUESTION_PATH[mode]):
                print(self.QUESTION_PATH[mode] + 'NOT EXIST')
                exit(-1)

        for mode in self.ANSWER_PATH:
            if not os.path.exists(self.ANSWER_PATH[mode]):
                print(self.ANSWER_PATH[mode] + 'NOT EXIST')
                exit(-1)

        print('Finished')
        print('')

