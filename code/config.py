
import warnings


class Config():

    def __init__(self) -> None:

        # dataset setting
        self.dataset='KineticSound'
        self.num_classes={'VGGSound':309,'KineticSound':31,'CREMAD':6,'AVE':28}
        self.modality=['audio','visual']
        self.fps = 1
        self.use_video_frames = 3

        # backbone setting
        self.in_c=3
        self.out_c=64

        # train setting
        self.train = False
        self.batch_size = 32
        self.epochs=100
        self.optimizer='Adamw'

        self.learning_rate=5e-5
        self.lr_decay_ratio=0.1
        # self.lr_decay_step=[30,50,70]
        self.lr_decay_step=40

        # modulation setting
        self.use_modulation=False
        self.modulation = 'OGM_GE'
        self.modulation_starts = 0
        self.modulation_ends = 80

        self.alpha = 1

        # fusion setting
        self.fusion_method = 'concat'
        self.d = [512, 512]
        # gated_fusion
        self.mid_c=512
        self.x_gated=False

        # adam-drop lambda setting
        self.use_adam_drop = False
        self.key=50
        self.sigma=2

        self.p_exe=0.7
        self.q_base=0.4
        self.lam=0.5

        # other setting
        self.checkpoint_path = 'result'

        self.resume_model=False
        self.resume_model_path=None

        self.use_tensorboard = True

        self.random_seed = 0
        self.gpu_ids = [0,1]

        self.func='tanh'
        self.form='/'

        self.device=0


        # transforms setting
        self.decrease_epoch=10
        self.sample_size=112
        self.sample_t_stride=1
        self.train_crop='random'
        self.value_scale=1
        self.scale_h=128
        self.scale_w=171
        self.train_crop_min_scale=0.25
        self.train_crop_min_ratio=0.75
        self.no_hflip=False
        self.colorjitter=False
        self.train_t_crop='random'

        self.audio_drop=0.0
        self.visual_drop=0.0
        

    def parse(self,kwargs):
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn('has not attribute %s'%k)
            setattr(self,k,v)

        # print('config info:')
        # for k,v in self.__dict__.items():
        #     if not k.startswith('__'):
        #         print(k,getattr(self,k))

if __name__ == "__main__":
    import argparse
    cfg=Config()

    parser = argparse.ArgumentParser()

    parser.add_argument('--use_modulation',action='store_true',help='use gradient modulation')
    parser.add_argument('--use_adam_drop',action='store_true',help='use adam-drop')
    parser.add_argument('--modulation', default='OGM_GE', type=str,choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='concat', type=str,choices=['sum', 'concat', 'gated'])
    parser.add_argument('--train', action='store_true', help='turn on train mode')
    parser.add_argument('--resume_model',action='store_true',help='whether to resume model')
    parser.add_argument('--checkpoint_path',type=str,help='load checkpoints')

    args=parser.parse_args()
    cfg.parse(vars(args))

