# models/select_model.py
import torch
from torch.optim.swa_utils import AveragedModel


def get_model(model_name, config):
    model_name = model_name.lower()

    print(config)

    if model_name == "srgan":
        from models.model_srgan import build_srgan
        model = build_srgan(config.upscale_factor)
        model = model.to(device=config.device,
                         memory_format=torch.channels_last)
        return model

    if model_name == "drln":
        from models.model_drln import DRLN
        model = DRLN(config.upscale_factor)
        model = model.to(device=config.device,
                         memory_format=torch.channels_last)

        return model

    if model_name == "srcnn":
        from models.model_srcnn import SRCNN, SRCNNwithPSF, SRCNNwithRL
        if config.arch == "psf":
            model = SRCNNwithPSF(config.in_chans, config.psf_size)
        elif config.arch == "rl":
            model = SRCNNwithRL(
                config.in_chans, config.psf_size, config.rl_iters)
        else:
            model = SRCNN(config.in_chans)

        model = model.to(device=config.device,
                         memory_format=torch.channels_last)
        return model

    if model_name == "edsr":
        from models.model_edsr import EDSR, EDSRwithRL, EDSRwithWiener
        if config.arch == "rl":
            model = EDSRwithRL(config.upscale_factor,
                               config.in_chans, config.rl_iters)
        elif config.arch == "wiener":
            model = EDSRwithWiener(config.upscale_factor, config.in_chans)
        else:
            model = EDSR(config.upscale_factor, config.in_chans)

        model = model.to(device=config.device,
                         memory_format=torch.channels_last)
        return model

    if model_name == "rcan":
        from models.model_rcan import RCAN
        model = RCAN(config.in_chans, config.out_chans, 64, 16, 20, 10,
                     config.upscale_factor,
                     rgb_mean=[0.5])
        # model_arch_name = config.model_arch.lower()
        # sr_model = model.__dict__[model_arch_name]()

        # Generate exponential average model, stabilize model training
        def ema_avg_fn(averaged_model_parameter, model_parameter, num_averaged): return (
            1 - config.model_ema_decay) * averaged_model_parameter + config.model_ema_decay * model_parameter
        ema_sr_model = AveragedModel(model, avg_fn=ema_avg_fn)

        sr_model = model.to(device=config.device)
        ema_sr_model = ema_sr_model.to(device=config.device)

        return sr_model, ema_sr_model

    raise ValueError(
        f"Model '{model_name}' is not supported. Please add it to select_model.py.")
