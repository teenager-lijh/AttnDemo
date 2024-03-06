import os
import json
import torch
import logging


def save_checkpoint(model, fname, checkpoint_home):
    # 保存模型权重
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(checkpoint_home, fname))
    logging.info(f'Checkpoint {fname} saved!')

    # # 保存模型的配置文件
    # with open(os.path.join(checkpoint_home, f'{fname}.conf'), 'w') as obj:
    #     json.dump(model.conf, obj, indent=4, ensure_ascii=False)
    # logging.info(f'model conf {fname}.conf saved!')


def load_checkpoint(model_cls, fname, checkpoint_home):
    with open(os.path.join(checkpoint_home, f'{fname}.conf'), 'r') as obj:
        conf = json.load(obj)
    logging.info(f'loading {fname} ok!')

    model = model_cls(conf)
    state_dict = torch.load(os.path.join(checkpoint_home, fname), map_location='cpu')
    model.load_state_dict(state_dict)
    logging.info(f'loading {fname}.conf ok!')

    return model
