import time
import torch
from tqdm import tqdm
from evaluation.coco_eval import CocoEvaluator


@torch.no_grad()
def test_and_eval(device, test_loader, model):
    iou_types = tuple(['bbox'])  # 'bbox'
    coco_evaluator = CocoEvaluator(test_loader.dataset.coco, iou_types)
    model.eval()

    tic = time.time()
    for idx, data in enumerate(tqdm(test_loader)):

        images = data[0]
        info = data[3]

        images = images.to(device)
        info = [{k: v.to(device) for k, v in i.items()} for i in info]
        pred = model(images)
        '''
        pred_ 
        [{'boxes' : tensor,
         'labels' : tensor,
         'scores' : tensor}, 
         ...]
        '''

        # 2. batch result
        results = pred
        res = {i['image_id'].item(): output for i, output in zip(info, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)
        toc = time.time()

        # ---------- print ----------
        if idx % 100 == 0 or idx == len(test_loader) - 1:
            print('Step: [{0}/{1}]\t'
                  'Time : {time:.4f}\t'
                  .format(idx,
                          len(test_loader),
                          time=toc - tic))

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = coco_evaluator.coco_eval['bbox'].stats.tolist()
    mAP = stats[0]
    print("mAP : ", mAP)
    print("Eval Time : {:.4f}".format(time.time() - tic))


if __name__ == '__main__':
    import argparse
    import torchvision
    from datasets.build_test import build_dataloader

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data_root', type=str, default='D:\data\coco')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--boxes_coord', type=str, default='cxywh')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--distributed', action='store_true')
    opts = parser.parse_args()
    print(opts)

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_loader = build_dataloader(opts)
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True).to(device)
    test_and_eval(device=device, model=model, test_loader=test_loader)




