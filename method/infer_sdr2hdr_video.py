from ffpipe.ffpipe import video_infer
from video_process_lib import lib_numpy, lib_cupy
import cupy as cp
from torch.utils.dlpack import to_dlpack,from_dlpack
from cupy import fromDlpack
import torch
import data.util as util
import numpy as np
from network import TriSegNet
import argparse
# logger = trt.Logger(trt.Logger.WARNING)
class S2H_video_infer(video_infer):
    def __init__(self, file_name, save_name, encode_params, model=None, scale=1, in_pix_fmt="yuv444p", out_pix_fmt="yuv444p10le", decode_out_pix_fmt="yuv420p10le" ,use_fp16=True, use_unshuffle=True, **decode_param_dict) -> None:
        super().__init__(file_name, save_name, encode_params, model, scale, in_pix_fmt, out_pix_fmt, decode_out_pix_fmt, **decode_param_dict)
        self.model = model
        self.cond_scale = 4
        
        
    
    def forward(self, x):
        x = lib_numpy.yuv_to_1_domain(x, offset=16, y_max_value=235, uv_max_value=240)
        x = lib_numpy.yuv2rgb_709(x)
        # cond_img = util.imresize_np(x, 1/self.cond_scale)
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(x, (2, 0, 1)))).float().unsqueeze(0)
        # cond = torch.from_numpy(np.ascontiguousarray(np.transpose(cond_img, (2, 0, 1)))).float().unsqueeze(0)
        img_LQ = img_LQ.cuda()
        img_LQ = img_LQ.half()
        # cond = cond.cuda()
        # data = {'LQ': img_LQ, 'cond':cond}
        output = self.model(img_LQ)
        
        output = output.float().squeeze()
        # self.model.feed_data(data, need_GT=False)
        # self.model.test()
        # visuals = self.model.get_current_visuals_gpu(need_GT=False)

        # sr_img = util.tensor2img(visuals['SR'], np.uint16)  # uint16
        # output = visuals['SR'].squeeze()

        output = fromDlpack(to_dlpack(output))
        x = lib_cupy.rgb2yuv_2020_inplace(output)

        x = lib_cupy.yuv_to_1_domain_inverse(x, offset=64, y_max_value=940, uv_max_value=960)

        x = cp.asnumpy(x.astype(cp.uint16))
        return x
    
if __name__ == '__main__':
    # file_name = "/mnt/ec-data2/ivs/1080p/zyh/SR/badcase/yuexia3_4_2min.mp4"
    # # file_name = "/mnt/ec-data2/ivs/1080p/zyh/SR/badcase/yuexia3_4_5s.mp4"
    # save_name = "/mnt/ec-data2/ivs/1080p/zyh/SR/res/yuexia3_4_2min_sr.mp4"
    # engine_path = "/data/yh/SR2023/KAIR/weight/swinir_v2_x2_1088x1920_fp16_op17.engine"
    # encode_params = ("libx264", "x264opts", "qp=12:bframes=3")
    import os.path as osp
    import logging
    import time
    import argparse
    from collections import OrderedDict

    import options.options as option
    # import utils.util as util
    # from data import create_dataset, create_dataloader
    from models import create_model

    import numpy as np

#### options


    # 创建解析器对象
    parser = argparse.ArgumentParser(description='Your program description')

    # 添加命令行参数
    parser.add_argument('--file_name', type=str, help='Input file name')
    parser.add_argument('--save_name', type=str, help='Output file name')
    # parser.add_argument('--engine_path', type=str, help='Engine path')
    # parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
    # opt = option.parse(parser.parse_args().opt, is_train=False)
    # opt = option.dict_to_nonedict(opt)
    # model = create_model(opt)
    
    
    # 解析命令行参数
    args = parser.parse_args()

    # 使用命令行参数
    file_name = args.file_name
    save_name = args.save_name
    # engine_path = args.engine_path
    # encode_params = ("libx264", "x264opts", "qp=12:bframes=3")
    # encode_params = ("libx265", "x265-params", "qp=12:bframes=3")
    
    ### Load network ###
    model = TriSegNet().half()
    # net.load_state_dict(torch.load('params_3DM.pth', map_location=lambda s, l: s))
    model.load_state_dict(torch.load('params_DaVinci.pth', map_location=lambda s, l: s))
    
    encode_params = encode_params = ("libx265", "x265-params", "colorprim='bt2020':transfer='smpte2084':colormatrix='bt2020nc':qp=12:bframes=3")

    infer = S2H_video_infer(file_name, save_name, encode_params, model=model, scale=1)
    infer.infer()
    # infer.run()
    # print(infer.model)
    # print(infer.model)