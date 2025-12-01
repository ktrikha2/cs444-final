import argparse
import yaml
import torch
from PIL import Image
import torchvision.transforms as T
from src.models.detector_head import make_model


def load_image(path):
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.ToTensor()
    ])
    return transform(img), img


def run(model, device, image_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model([image_tensor.to(device)])
    return outputs[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--save', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = make_model(cfg)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    img_tensor, raw_img = load_image(args.image)
    preds = run(model, device, img_tensor)

    print("Predictions:")
    for b, s, l in zip(preds['boxes'], preds['scores'], preds['labels']):
        print(f"class={l.item()} score={s.item():.3f} box={b.tolist()}")

    if args.save:
        from tools.visualize import draw_boxes
        draw_boxes(args.image, preds, args.save)
        print(f"Saved visualization to {args.save}")


if __name__ == '__main__':
    main()
