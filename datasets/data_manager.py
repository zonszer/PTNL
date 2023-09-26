from dassl.data import DataManager
from dassl.data.data_manager import DatasetWrapper
from dassl.data.transforms import build_transform
from dassl.data.samplers import build_sampler
from dassl.data.datasets import build_dataset
from ..vision_benchmark.evaluation import construct_dataloader, construct_multitask_dataset

from torch.utils.data import DataLoader, WeightedRandomSampler
import torch

def build_data_loader(          #re-implementation
    cfg,
    sampler_type="RandomSampler",   
    sampler=None,       
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None,
    tag=None,
):  
    assert sampler_type is not None and sampler is None            #HACK to prevent possible bugs
    # Build sampler
    if sampler_type is not None:
        sampler = build_sampler(
            sampler_type,
            cfg=cfg,
            data_source=data_source,
            batch_size=batch_size,
            n_domain=n_domain,
            n_ins=n_ins,
        )
    else:
        sampler = sampler

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    if tag is None:
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=is_train and len(data_source) >= batch_size,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
            batch_size=batch_size,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=False,
            pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
        )

    return data_loader

class ElevaterDataManager(DataManager):
    def __init__(self, cfg):
        # Load dataset:
        train_loader_x, val_loader, test_loader, class_map, train_dataset = construct_dataloader(cfg)       #train_dataset=500

        # self._metric = get_metric(class_map_metric[cfg.DATASET.DATASET])
        # self._metric_name = class_map_metric[cfg.DATASET.DATASET]
        # Attributes:
        self._num_classes = len(class_map)
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = {}
        # random.seed(cfg.DATASET.RANDOM_SEED_SAMPLING)
        for key, value in enumerate(class_map):
            if isinstance(value, list):
                value = value[0] #random.choice(value)
            self._lab2cname[key] = value

        # Dataset and data-loaders:
        self.train_loader_x = train_loader_x
        self.train_loader_u = None
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_loader_sstrain = train_loader_x

        if cfg.VERBOSE:
            pass
            # self.show_dataset_summary(cfg)

    def update_ssdateloader(self, predict_label_dict: None) -> None:
        """
        This function is used to update the train_loader_sstrain to add labels.

        Args:
            predict_label_dict (dict): A dictionary containing image paths as keys and corresponding labels as values.
        """
        assert predict_label_dict is None
        self.train_loader_sstrain.dataset.labels = self.partialY         #TODO check 
        print('ElevaterDataset sstrain: len()==', len(self.train_loader_sstrain.dataset))


class UPLDataManager(DataManager):
    def __init__(self,
                cfg,
                custom_tfm_train=None,
                custom_tfm_test=None,
                dataset_wrapper=None):
        # super().__init__(cfg, custom_tfm_train, custom_tfm_test, dataset_wrapper)   #get the train_loader_x, test_loader, and val_loader
        dataset = build_dataset(cfg)

        # 1. Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname
        self.cfg = cfg
        self.dataset_wrapper = dataset_wrapper

        # 2. build transform 
        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        # 3. build loaders: Build val_loader
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )
        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )
        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )
        # Build train_loader_u
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

        if dataset.sstrain:
            train_loader_sstrain = build_data_loader(
                cfg,
                sampler_type="SequentialSampler",
                data_source=dataset.sstrain,
                batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                tfm=tfm_test,       # both is tfm_test
                is_train=False,
                dataset_wrapper=dataset_wrapper,
                tag='sstrain'
            )

            # re-build train_loader_x
            train_loader_x = build_data_loader(
                cfg,
                sampler_type="SequentialSampler",
                data_source=dataset.train_x,
                batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                tfm=tfm_test,       # both is tfm_test
                is_train=False,
                dataset_wrapper=dataset_wrapper,
                tag='sstrain'
            )

        # Build test_novel_loader
        if cfg.DATALOADER.OPEN_SETTING:
            test_novel_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.novel,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper,
            )

            test_base_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.base,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper,
            )

        # Dataset and data-loaders
        self.dataset = dataset
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_loader_x = train_loader_x
        self.tfm_train = tfm_train
        for loader_name in ['test_novel_loader', 'test_base_loader', 'train_loader_sstrain', 'train_loader_u']:
            try:
                loader = eval(loader_name)
            except NameError:
                loader = None
            exec(f'self.{loader_name}=loader')
        
        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    def update_ssdateloader(self, predict_label_dict):
        """update the train_loader_sstrain to add labels

        Args:
            predict_label_dict ([dict]): [a dict {'imagepath': 'label'}]
        """

        sstrain = self.dataset.add_label(predict_label_dict, self.cfg.DATASET.NAME)
        print('sstrain', len(sstrain))

        # train_sampler = WeightedRandomSampler(weights, len(sstrain))
        train_loader_sstrain = build_data_loader(
            self.cfg,
            sampler_type="RandomSampler",
            sampler=None,
            data_source=sstrain,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=1,
            tfm=self.tfm_train,
            is_train=True,
            dataset_wrapper=self.dataset_wrapper,
        )
        self.train_loader_sstrain = train_loader_sstrain
