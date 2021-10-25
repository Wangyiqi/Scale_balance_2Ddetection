
from .builder import DATASETS,PIPELINES
from .coco import CocoDataset
from .pipelines import Compose
from numpy import random


@DATASETS.register_module()
class CocoDatasetWithMosaic(CocoDataset):
    def __init__(self,
                 before_mosaic_load_pipeline,
                 mosaic_pipeline,
                 after_mosaic_load_pipeline,
                 prob_mosaic=0.5,
                 *ori_args, **ori_kwargs):
        super(CocoDatasetWithMosaic, self).__init__(*ori_args, **ori_kwargs)
        self._before_mosaic_load_pipeline = Compose(before_mosaic_load_pipeline)
        self._mosaic_pipeline = Compose(mosaic_pipeline)
        self._after_mosaic_load_pipeline = Compose(after_mosaic_load_pipeline)
        self.prob_mosaic = prob_mosaic
        self.num_sample = len(self.img_ids)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        assert self.proposals is None
        if random.rand()<self.prob_mosaic:
            idx_list = [idx]+[random.randint(self.num_sample) for _ in range(3)]
            result_list = []
            for idx_i in idx_list:
                img_info = self.data_infos[idx_i]
                ann_info = self.get_ann_info(idx_i)
                results_t = dict(img_info=img_info, ann_info=ann_info)
                self.pre_pipeline(results_t)
                results_t = self._before_mosaic_load_pipeline(results_t)
                result_list.append(results_t)
            results = self._mosaic_pipeline(result_list)
            if results is None:
                return super(CocoDatasetWithMosaic, self).prepare_train_img(idx)
            try:
                return self._after_mosaic_load_pipeline(results)
            except:
                return super(CocoDatasetWithMosaic, self).prepare_train_img(idx)
        else:
            return super(CocoDatasetWithMosaic, self).prepare_train_img(idx)

@DATASETS.register_module()
class CocoDatasetWithSynthesis(CocoDataset):
    def __init__(self,
                 before_synthesis_load_pipeline,
                 after_synthesisi_load_pipeline,
                 mosaic_pipeline=list(),
                 copypaste_pipeline=list(),
                 prob_mosaic=0.0,
                 prob_copypaste=0.0,
                 *ori_args, **ori_kwargs):
        super(CocoDatasetWithSynthesis, self).__init__(*ori_args, **ori_kwargs)
        self._before_synthesis_load_pipeline = Compose(before_synthesis_load_pipeline)
        self._mosaic_pipeline = Compose(mosaic_pipeline)
        self._copypaste_pipeline = Compose(copypaste_pipeline)
        self._after_synthesisi_load_pipeline = Compose(after_synthesisi_load_pipeline)
        self.prob_mosaic = prob_mosaic
        self.prob_copypaste = prob_copypaste
        assert self.prob_mosaic+self.prob_copypaste <= 1.0
        self.num_sample = len(self.img_ids)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        assert self.proposals is None
        seed_ = random.rand()
        if seed_ < self.prob_mosaic+self.prob_copypaste:
            if seed_<self.prob_mosaic:
                num_extra_img = 3
                synthesis_pipeline = self._mosaic_pipeline
            else:
                num_extra_img = 1
                synthesis_pipeline = self._copypaste_pipeline
            idx_list = [idx]+[random.randint(self.num_sample) for _ in range(num_extra_img)]
            result_list = []
            for idx_i in idx_list:
                img_info = self.data_infos[idx_i]
                ann_info = self.get_ann_info(idx_i)
                results_t = dict(img_info=img_info, ann_info=ann_info)
                self.pre_pipeline(results_t)
                results_t = self._before_synthesis_load_pipeline(results_t)
                result_list.append(results_t)
            results = synthesis_pipeline(result_list)
            if results is not None:
                return self._after_synthesisi_load_pipeline(results)
        return super(CocoDatasetWithSynthesis, self).prepare_train_img(idx)
