import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, IterableDataset, DataLoader
from timeit import default_timer
from functools import partial
from transformers import AutoModel, AutoConfig, SwinForImageClassification, SwinForMaskedImageModeling, RobertaForTokenClassification, AutoTokenizer, DataCollatorWithPadding
from transformers.models.roberta.modeling_roberta import RobertaLayer
#from otdd.pytorch.distance import DatasetDistance, FeatureCost

from task_configs import get_data, get_optimizer_scheduler, get_config, get_metric
from utils import conv_init, embedder_init, embedder_placeholder, adaptive_pooler, to_2tuple, set_grad_state, create_position_ids_from_inputs_embeds, l2, MMD_loss, get_params_to_update 
import copy
from resnet import resnet32, resnet20
from fno import Net2d
# from data_loaders import load_deepsea 
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer 
from genomic_benchmarks.dataset_getters.utils import  LetterTokenizer, build_vocab, check_seq_lengths 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def otdd(feats, ys=None, src_train_dataset=None, exact=True):
#     ys = torch.zeros(len(feats)) if ys is None else ys

#     if not torch.is_tensor(feats):
#         feats = torch.from_numpy(feats).to('cpu')
#         ys = torch.from_numpy(ys).long().to('cpu')

#     dataset = torch.utils.data.TensorDataset(feats, ys)

#     dist = DatasetDistance(src_train_dataset, dataset,
#                                     inner_ot_method = 'exact' if exact else 'gaussian_approx',
#                                     debiased_loss = True, inner_ot_debiased=True,
#                                     p = 2, inner_ot_p=2, entreg = 1e-1, ignore_target_labels = False,
#                                     device=feats.device, load_prev_dyy1=None)
                
#     d = dist.distance(maxsamples = len(src_train_dataset))
#     return d


class wrapper2D(torch.nn.Module):
    def __init__(self, input_shape, output_shape, use_embedder=True, weight='base', train_epoch=0, activation=None, target_seq_len=None, drop_out=None, from_scratch=False, args=None, wavelen=None):
        super().__init__()
        self.classification = (not isinstance(output_shape, tuple)) and (output_shape != 1)
        self.output_raw = True
        self.use_embedder = use_embedder

        if weight == 'tiny':
            arch_name = "microsoft/swin-tiny-patch4-window7-224"
            embed_dim = 96
            output_dim = 768
            img_size = 224
        elif weight == 'base':
            arch_name = "microsoft/swin-base-patch4-window7-224-in22k"
            embed_dim = 128
            output_dim = 1024
            img_size = 224
            patch_size = 4

        if self.classification:
            modelclass = SwinForImageClassification
        else:
            modelclass = SwinForMaskedImageModeling
            
        self.model = modelclass.from_pretrained(arch_name)
        self.model.config.image_size = img_size
        if drop_out is not None:
            self.model.config.hidden_dropout_prob = drop_out 
            self.model.config.attention_probs_dropout_prob = drop_out

        self.model = modelclass.from_pretrained(arch_name, config=self.model.config) if not from_scratch else modelclass(self.model.config)

        if self.classification:
            self.model.pooler = nn.AdaptiveAvgPool1d(1)
            self.model.classifier = nn.Identity()
            self.predictor = nn.Linear(in_features=output_dim, out_features=output_shape)
        else:
            self.pool_seq_dim = adaptive_pooler(output_shape[1] if isinstance(output_shape, tuple) else 1)
            self.pool = nn.AdaptiveAvgPool2d(output_shape[-2:])
            self.predictor = nn.Sequential(self.pool_seq_dim, self.pool)

        set_grad_state(self.model, False)
        set_grad_state(self.predictor, False)

        if use_embedder:
            self.embedder = Embeddings2D(input_shape, patch_size=patch_size, config=self.model.config, embed_dim=embed_dim, img_size=img_size, raw = self.output_raw, args = args, output_shape=output_shape, wavelen=wavelen)
            #embedder_init(self.model.swin.embeddings, self.embedder, train_embedder=train_epoch > 0)
            set_grad_state(self.embedder, True)
            self.model.swin.embeddings = self.embedder
        self.resizer = transforms.Resize((img_size, img_size))


    def forward(self, x):
        #x = self.resizer(x)
        if self.output_raw:
            if self.use_embedder:
                return self.model.swin.embeddings(x)
            return self.model.swin.embeddings(x)[0]
        x = self.model(x).logits

        return self.predictor(x)

    
class Embeddings2D(nn.Module):

    def __init__(self, input_shape, patch_size=4, embed_dim=96, img_size=224, config=None, raw=True, args=None, output_shape=None, wavelen=None):
        super().__init__()
        self.wavelen = torch.tensor(wavelen).reshape(input_shape[1], 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to('cuda')
        self.raw = raw
        self.resize, self.input_dimensions = transforms.Resize((img_size, img_size)), (img_size, img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patched_dimensions = (self.input_dimensions[0] // self.patch_size[0], self.input_dimensions[1] // self.patch_size[1])
        ks = self.patch_size
        self.embedder_type = args.embedder_type if args is not None else None
        self.dense = isinstance(output_shape, tuple)
        if self.dense:
            output_shape = output_shape[1]
        if self.embedder_type == 'random':
            self.projection = nn.Conv2d(input_shape[1], embed_dim, kernel_size=ks, stride=self.patch_size, padding=(ks[0]-self.patch_size[0]) // 2)
            conv_init(self.projection)
        # elif self.embedder_type == 'unet':
        #     self.projection = nn.Conv2d(input_shape[1], embed_dim, kernel_size=1, stride=1) 
        #     self.fno = Encoder_v2(embed_dim,f_channel=input_shape[1],num_class=1,ks=None,ds=None)
        elif self.embedder_type == 'resnet':
            in_channel=input_shape[1]
            num_classes=64
            self.dash = resnet20(in_channel = 2, num_classes = 16, remain_shape = True)
            #print("kernel size:", ks, self.patch_size, embed_dim)
            self.fusion = nn.Conv2d(16*input_shape[1], 64, kernel_size=3, padding=1)
            self.conv = nn.Conv2d(64, embed_dim, kernel_size = ks, stride = self.patch_size, padding = 0)
            conv_init(self.conv)
            # self.final = nn.Sequential(
            #     nn.Conv2d(64, 256, kernel_size=1),
            #     nn.Conv2d(256, output_shape, kernel_size=1),
            #     nn.AdaptiveAvgPool2d(1),
            # )
            self.final = nn.Sequential(
                nn.Conv2d(64, output_shape, kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
            )
        elif self.embedder_type == 'fno':
            self.fno = Net2d(modes=10, width=64, input_channels= 2, output_channels = 64) # input channel is 3: (a(x, y), x, y)
            self.fusion = nn.Conv2d(64*input_shape[1], 64, kernel_size=3, padding=1)
            self.conv = nn.Conv2d(64, embed_dim, kernel_size = ks, stride = self.patch_size, padding = 0)
            conv_init(self.conv)
            self.final = nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=1),
                nn.Conv2d(256, output_shape, kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
            )


        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.LayerNorm(embed_dim)
        num_patches = (self.input_dimensions[1] // self.patch_size[1]) * (self.input_dimensions[0] // self.patch_size[0])
        
        

        
    def maybe_pad(self, x, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            x = nn.functional.pad(x, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            x = nn.functional.pad(x, pad_values)
        return x


    def forward(self, x, *args, **kwargs):
        #x = self.resize(x)
        b, c, height, width = x.shape
        wave = self.wavelen.repeat(b, 1, 1, height, width) # (b, c, 1, h, w)
        x = x.unsqueeze(2)
        x = torch.cat([x, wave], 2) # (b, c, 2, h, w)
        x = x.flatten(start_dim=0, end_dim=1) # (b*c, 2, h, w)
        #x = self.maybe_pad(x, height, width)
        if self.embedder_type == 'random':
            x = self.projection(x)
            x = x.flatten(2).transpose(1, 2)
            
            x = self.norm(x)
        elif self.embedder_type == 'resnet':
            x = self.dash(x) #(b*c, 16, h, w)
            x = x.reshape(b, c, 16, height, width) #(b, c, 16, h, w)
            x = x.flatten(start_dim=1, end_dim=2) #(b, c*16, h, w)
            x = self.fusion(x) #(b, 64, h, w)
            x = self.norm1(x)
            if self.dense:
                xfno = self.final(x.permute(0,2,3,1)).permute(0,3,1,2)
            else:
                xfno = self.final(x)
            x = self.conv(x).flatten(2).transpose(1, 2)
            x = self.norm2(x)
        elif self.embedder_type == 'fno':
            x = x.permute(0,2,3,1) #(b*c, h, w, 2)
            x = self.fno(x) #(b*c, h, w, 64)
            x = x.permute(0,3,1,2) #(b*c, 64, h, w)
            x = x.reshape(b, c, 64, height, width) #(b, c, 64, h, w)
            x = x.flatten(start_dim=1, end_dim=2) #(b, c*64, h, w)
            x = self.fusion(x) #(b, 64, h, w)
            x = self.norm1(x)
            if self.dense:
                xfno = self.final(x.permute(0,2,3,1)).permute(0,3,1,2)
            else:
                xfno = self.final(x)
            x = self.conv(x).flatten(2).transpose(1, 2)
            x = self.norm2(x)
            


        if self.raw and self.embedder_type == 'random':
            return x
        elif self.raw:
            return x, xfno
        else:  
            return x, self.patched_dimensions
        



# class EmbeddingWave(nn.Module):



####################################################
def get_tgt_model(args, root, sample_shape, num_classes, loss, wavelen, add_loss=False, use_determined=False, context=None, opid=0):
    src_train_loader, _, _, _, _, _, _ = get_data(root, args.embedder_dataset, args.batch_size, False, maxsize=2000)
    
    
    IMG_SIZE = 224 if args.weight == 'tiny' or args.weight == 'base' or args.weight == 'ViT' else 196
        
    src_model = wrapper2D(sample_shape, num_classes, use_embedder=False, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, drop_out=args.drop_out, wavelen=wavelen)
    src_model = src_model.to(args.device).eval()
        
    src_feats = []
    src_ys = []
    for i, data in enumerate(src_train_loader):
        x_, y_ = data 
        x_ = x_.to(args.device)
        x_ = transforms.Resize((IMG_SIZE, IMG_SIZE))(x_)
        out = src_model(x_)
        if len(out.shape) > 2:
            out = out.mean(1)

        src_ys.append(y_.detach().cpu())
        src_feats.append(out.detach().cpu())
    src_feats = torch.cat(src_feats, 0)
    src_ys = torch.cat(src_ys, 0).long()
    src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)        
    del src_model    


    # tgt_train_loader, _, _, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, False, get_shape=True)
    tgt_train_loader, tgt_val_loader, _, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, args.valid_split)
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None
    joint_optim = True if hasattr(args,'joint_optim') and args.joint_optim else False
    
    for batch in tgt_train_loader: 
        x, y = batch
        print('x:',x.size())
        print('y:',y.size())
        break

    if args.infer_label:
        tgt_train_loader, num_classes_new = infer_labels(tgt_train_loader)
    else:
        num_classes_new = num_classes

    # if args.objective=='otdd-exact' or args.objective=='otdd':
    #     print("src feat shape", src_feats.shape, src_ys.shape, "num classes", num_classes_new) 
    #     tgt_train_loaders, tgt_class_weights = load_by_class(tgt_train_loader, num_classes_new)
    print("src feat shape", src_feats.shape)
  
    wrapper_func = wrapper2D
    tgt_model = wrapper_func(sample_shape, num_classes, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, target_seq_len=args.target_seq_len, drop_out=args.drop_out, args=args, wavelen=wavelen)
    
    # if hasattr(args, 'data_parallel') and args.data_parallel:
    #     tgt_model = nn.DataParallel(tgt_model) 
    # if hasattr(args, 'quantize') and args.quantize:
    #     tgt_model.to(torch.bfloat16)
    tgt_model = tgt_model.to(args.device).train()
    #print(get_optimizer_scheduler(args, tgt_model, module='embedder'))
    args, tgt_model, tgt_model_optimizer, tgt_model_scheduler = get_optimizer_scheduler(args, tgt_model, module='embedder-with-linear') # only update embedder
    tgt_model_optimizer.zero_grad()


    # if args.objective == 'otdd-exact':
    #     score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=True)
    # elif args.objective == 'otdd-gaussian':
    #     score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=False)
    # elif args.objective == 'l2':
    #     score_func = partial(l2, src_train_dataset=src_train_dataset)
    # else:
    score_func = MMD_loss(src_data=src_feats, maxsamples=args.maxsamples)
    
    #joint_optim = False
    if joint_optim:
        total_losses, total_MMD_losses, total_second_losses, times, embedder_stats = [], [], [], [], []
        alpha = args.alpha if hasattr(args,'alpha') else 1
        beta = args.beta if hasattr(args,'beta') else 1
        # Classification loss
        _, _, _, second_loss, args, _ = get_config(root, args)
        metric = get_metric(args.dataset)[0]
        second_loss = second_loss.to(args.device)

        for ep in range(args.embedder_epochs):   
            total_loss,total_MMD_loss,total_second_loss = 0,0,0    
            time_start = default_timer()
            
            feats, feats2, ys = [],[],[]
            datanum = 0
            # for j, data in enumerate(tgt_train_loaders[i]): # for otdd
            for j, data in enumerate(tgt_train_loader): #
                
                x, y = data 
                x = x.to(args.device) 
        
                out, xfno = tgt_model(x)
                feats.append(out)
                feats2.append(xfno)
                ys.append(y.to(args.device))

                datanum += x.shape[0]
                # print("datanum:", datanum)
                
                if datanum > args.maxsamples or j == len(tgt_train_loader) - 1: # accumulate samples until reach maxsamples
                    # print("maxsample!")
                    feats = torch.cat(feats, 0).mean(1) # can be improved?
                    feats2 = torch.cat(feats2, 0)
                    ys = torch.cat(ys, 0)
                    # print('538', feats.shape)
                    # print('538', feats2.shape)
                    # print('538', ys.shape)
                    loss1 = score_func(feats)
                    loss2 = second_loss(feats2, ys)
                    loss = alpha * loss1 + beta * loss2
                    loss.backward()

                    tgt_model_optimizer.step()
                    tgt_model_optimizer.zero_grad()

                    total_loss += loss.item()
                    total_MMD_loss += loss1.item()
                    total_second_loss += loss2.item()

                    feats, feats2, ys = [],[],[]
                    datanum = 0

            time_end = default_timer()  
            times.append(time_end - time_start) 

            total_losses.append(total_loss) #
            total_MMD_losses.append(total_MMD_loss) #
            total_second_losses.append(total_second_loss) #
            embedder_stats.append([total_losses[-1], total_MMD_losses[-1], total_second_losses[-1],times[-1]])
            print("[train embedder", ep, "%.6f" % tgt_model_optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (times[-1]), "\ttotal loss:", "%.4f" % total_losses[-1], "\tMMD loss:", "%.4f" % total_MMD_losses[-1], "\tCE loss:", "%.4f" % total_second_losses[-1]) 

            tgt_model_scheduler.step()

        with torch.no_grad():
            feats, ys = [],[]
            for j, data in enumerate(tgt_train_loader):
                x, y = data 
                x = x.to(args.device) 
                _, xfno = tgt_model(x)
                feats.append(xfno)
                ys.append(y.to(args.device))
            feats = torch.cat(feats, 0)
            ys = torch.cat(ys, 0)
            print(feats.shape, ys.shape)
            train_score = metric(feats, ys)
            feats, ys = [],[]
            for j, data in enumerate(tgt_val_loader):
                x, y = data 
                x = x.to(args.device) 
                _, xfno = tgt_model(x)
                feats.append(xfno)
                ys.append(y.to(args.device))
            feats = torch.cat(feats, 0)
            ys = torch.cat(ys, 0)
            val_score = metric(feats, ys)
            print("final train score:", train_score, "final val score:", val_score)


    # else: # not joint optimization
    #     total_losses, times, embedder_stats = [], [], []
    #     for ep in range(args.embedder_epochs):   
    #         total_loss = 0    
    #         time_start = default_timer()

            
    #         feats, ys = [],[]
    #         datanum = 0
    #         # for j, data in enumerate(tgt_train_loaders[i]): # for otdd
    #         for j, data in enumerate(tgt_train_loader): # for other losses such as MMD
    #             # if transform is not None:
    #             #     x, y, z = data
    #             # else:
    #             x, y = data 
    #             x = x.to(args.device) 
    #             ys.append(y.to(args.device))
    #             out = tgt_model(x)
    #             feats.append(out)
    #             datanum += x.shape[0]  
    #             if datanum > args.maxsamples or j == len(tgt_train_loader) - 1: # accumulate samples until reach maxsamples
    #                 feats = torch.cat(feats, 0).mean(1) # can be improved?
    #                 ys = torch.cat(ys, 0)
                    
    #                 loss = score_func(feats)
    #                 loss.backward()

    #                 tgt_model_optimizer.step()
    #                 tgt_model_optimizer.zero_grad()
    #                 total_loss += loss.item()

    #                 feats, ys = [],[]
    #                 datanum = 0

    #             # feats = torch.cat(feats, 0).mean(1)
    #             # if feats.shape[0] > 1:
    #             #     loss = tgt_class_weights[i] * score_func(feats) #
    #             #     loss.backward()
    #             #     total_loss += loss.item()

    #         time_end = default_timer()  
    #         times.append(time_end - time_start) 

    #         total_losses.append(total_loss)
    #         embedder_stats.append([total_losses[-1], times[-1]])
    #         print("[train embedder", ep, "%.6f" % tgt_model_optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (times[-1]), "\tMMD loss:", "%.4f" % total_losses[-1])

    #         tgt_model_optimizer.step()
    #         tgt_model_scheduler.step()
    #         tgt_model_optimizer.zero_grad()

    del tgt_train_loader #
    torch.cuda.empty_cache()
    
    tgt_model.output_raw = False 
    tgt_model.embedder.raw = False 

    return tgt_model, embedder_stats


def infer_labels(loader, k = 10):
    from sklearn.cluster import k_means, MiniBatchKMeans
    
    if hasattr(loader.dataset, 'tensors'):
        X, Y = loader.dataset.tensors[0].cpu(), loader.dataset.tensors[1].cpu().numpy()
        try:
            Z = loader.dataset.tensors[2].cpu()
        except:
            Z = None
    else:
        X, Y, Z = get_tensors(loader.dataset)

    Y = Y.reshape(len(Y), -1)

    if len(Y) <= 10000:
        labeling_fun = lambda Y: torch.LongTensor(k_means(Y, k)[1])
        Y = labeling_fun(Y).unsqueeze(1)
    else:
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10000).fit(Y)
        Y = torch.LongTensor(kmeans.predict(Y)).unsqueeze(1)

    if Z is None:
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True), k
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y, Z), batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True), k



def get_tensors(dataset):
    xs, ys, zs = [], [], []
    for i in range(dataset.__len__()):
        data = dataset.__getitem__(i)
        xs.append(np.expand_dims(data[0], 0))
        ys.append(np.expand_dims(data[1], 0))
        if len(data) == 3:
            zs.append(np.expand_dims(data[2], 0))

    xs = torch.from_numpy(np.array(xs)).squeeze(1)
    ys = torch.from_numpy(np.array(ys)).squeeze(1)

    if len(zs) > 0:
        zs = torch.from_numpy(np.array(zs)).squeeze(1)
    else:
        zs = None

    return xs, ys, zs


def load_by_class(loader, num_classes):
    train_set = loader.dataset
    subsets = {}
    # print(len(train_set.__getitem__(0)))
    if len(train_set.__getitem__(0)) == 3:
        try:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y == target]) for target in range(num_classes)}
        except:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}
    else:
        try:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y == target]) for target in range(num_classes)}
        except:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}
    loaders = {target: torch.utils.data.DataLoader(subset, batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True) for target, subset in subsets.items()}
    class_weights = {target: len(subset)/len(train_set) for target, subset in subsets.items()}
    
    print("class weights")
    for target, subset in subsets.items():
        print(target, len(subset), len(train_set), len(subset)/len(train_set))

    return loaders, class_weights