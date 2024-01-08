from torch.utils.data import DataLoader
from datasets.coco_dataset import COCO_Dataset
import datasets.transforms_ as T
from torch.utils.data.distributed import DistributedSampler


def build_dataloader(opts):

    transform_test = T.Compose([
        T.Resize([800, 800]),
        T.ToTensor(),
    ])

    test_set = COCO_Dataset(data_root=opts.data_root,
                            split='val',
                            download=True,
                            transform=transform_test,
                            boxes_coord=opts.boxes_coord,
                            visualization=False)

    test_loader = DataLoader(test_set,
                             batch_size=opts.batch_size,
                             collate_fn=test_set.collate_fn,
                             shuffle=False,
                             num_workers=opts.num_workers,
                             pin_memory=True)

    if opts.distributed:
        test_loader = DataLoader(test_set,
                                 batch_size=int(opts.batch_size / opts.world_size),
                                 collate_fn=test_set.collate_fn,
                                 shuffle=False,
                                 num_workers=int(opts.num_workers / opts.world_size),
                                 pin_memory=True,
                                 sampler=DistributedSampler(dataset=test_set, shuffle=False),
                                 drop_last=False)

    opts.num_classes = 91
    return test_loader



