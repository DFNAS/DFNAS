class Diffusion_Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        '''
        self.device = torch.device(config.MODEL.DEVICE)
        self.in_features = config.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = config.MODEL.DiffusionDet.NUM_CLASSES
        self.num_proposals = config.MODEL.DiffusionDet.NUM_PROPOSALS
        self.hidden_dim = config.MODEL.DiffusionDet.HIDDEN_DIM
        self.num_heads = config.MODEL.DiffusionDet.NUM_HEADS
        '''	

        # Build Backbone.
        #self.backbone = build_backbone(config)
        #self.size_divisibility = self.backbone.size_divisibility

        add_diffusiondet_config(config)

        # build diffusion
        timesteps = 10
        sampling_timesteps = None #config.MODEL.DiffusionDet.SAMPLE_STEP
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        #self.self_condition = False
        self.scale = config.MODEL.DiffusionDet.SNR_SCALE
        #self.box_renewal = True
        #self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build Dynamic Head.
        #self.head = DynamicHead(config=config, roi_input_shape=self.backbone.output_shape())
        # Loss parameters:
        #class_weight = config.MODEL.DiffusionDet.CLASS_WEIGHT
        #giou_weight = config.MODEL.DiffusionDet.GIOU_WEIGHT
        #l1_weight = config.MODEL.DiffusionDet.L1_WEIGHT
        #no_object_weight = config.MODEL.DiffusionDet.NO_OBJECT_WEIGHT
        #self.deep_supervision = config.MODEL.DiffusionDet.DEEP_SUPERVISION
        #self.use_focal = config.MODEL.DiffusionDet.USE_FOCAL
        #self.use_fed_loss = config.MODEL.DiffusionDet.USE_FED_LOSS
        #self.use_nms = config.MODEL.DiffusionDet.USE_NMS

        # Build Criterion.
        #matcher = HungarianMatcherDynamicK(
        #    config=config, cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight, use_focal=self.use_focal
        #)
        #weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        '''
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        '''

        #losses = ["labels", "boxes"]

        '''
        self.criterion = SetCriterionDynamicK(
            config=config, num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses, use_focal=self.use_focal,)
        '''
        self.criterion = nn.MSELoss()
        self.to(device)

        self.model = MyTransformer(
                num_decoders = config.network.num_decoders,
                num_layers = config.network.num_layers,
                r1 = config.network.r1,
                r2 = config.network.r2,
                num_f_maps = config.network.num_f_maps,
                input_dim = config.network.features_dim,
                num_classes = config.data.num_classes,
                channel_masking_rate = config.network.channel_masking_rate,
            )

        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = config.data.num_classes

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    @torch.no_grad()
    #def ddim_sample(self, batched_inputs, backbone_feats, images_whwh, images, clip_denoised=True, do_postprocess=True):
    def ddim_sample(self, batched_inputs, mask):
        cached_features = self.model(batched_inputs, mask, None, None, cache_from_encoder=True)

        batch = batched_inputs.shape[0]
        shape = (batch, self.num_classes, batched_inputs.shape[-1])
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        z_t = torch.randn(shape, device=device)

        ensemble_targets = []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            #self_cond = x_start if self.self_condition else None
            self_cond = None

            z_n = torch.clamp(z_t, min=-1 * self.scale, max=self.scale)
            #z_n = ((z_n / self.scale) + 1) / 2
            z_n = self.model(cached_features, mask, z_n, time_cond)[-1]

            #z_n = (z_n * 2 - 1.) * self.scale
            z_n = torch.clamp(z_n, min=-1 * self.scale, max=self.scale)
            z_noise = self.predict_noise_from_start(z_t, time_cond, z_n)

            pred_noise, x_start = z_noise, z_n

            if time_next < 0:
                z_t = z_n
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(z_t)

            z_t = z_n * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        #print(z_t.shape, z_t)
        return z_t
        preds = (z_t * (self.num_classes-1)).round().long()
        return preds

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def prepare_diffusion_concat(self, seq):
        """
        :param seq: (T, ), decrete / bit
        """
        num_frames = seq.shape[-1]

        t = torch.randint(0, self.num_timesteps, (1,), device=device).long()
        noise = torch.randn(num_frames, device=device)

        '''
        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4,
                                          device=self.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes
        '''
        x_start = seq

        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        #x = ((x / self.scale) + 1) / 2.

        return x, noise, t

    def prepare_targets(self, targets):
        new_targets = []
        diffused_seqs = []
        noises = []
        ts = []
        for seq in targets:
            # seq = self.from_decrete_to_bit(seq)
            d_seq, d_noise, d_t = self.prepare_diffusion_concat(seq)
            diffused_seqs.append(d_seq)
            noises.append(d_noise)
            ts.append(d_t)

        return torch.stack(diffused_seqs), torch.stack(noises), torch.stack(ts)
