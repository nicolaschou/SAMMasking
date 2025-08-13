# SAM Point Mask
Interactive tool for creating binary masks from images using Meta’s [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything).

## Requirements
- Python 3.9+
- PyTorch, OpenCV, NumPy, Matplotlib, mpl-interactions, Tkinter
- SAM (install from GitHub):
  ```bash
  pip install git+https://github.com/facebookresearch/segment-anything.git
  ```

## Model Weights
Download sam_vit_h_4b8939.pth (~2.5 GB) from the [official SAM repo](https://github.com/facebookresearch/segment-anything#model-checkpoints) and place it in the project root folder.

## Usage
```bash
python sam_point_mask.py
```
- Select images.
- Enter an output filename prefix or type quit to exit.
- Click points on the object to mask(Backspace = undo, Enter = finish).
- Masks are saved as PNGs.

## Citation
This project uses the Segment Anything Model (SAM) by Meta AI. If you also use SAM, please cite:
```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```