import torch, time, pdb, os, random
import numpy as np
import torch.nn as nn
from utils import *
from nn_layers import *
from parameters import *
import matplotlib.pyplot as plt
import numpy as np


##################### Author @Emre Ozfatura  @ Yulin Shao ###################################################

######################### Inlcluded modules and options #######################################
#1) Feature extracture 
#2) Successive decoding option 
#3) Vector embedding option 
#4) Belief Modulate

################################# Guideline #####################################
#Current activation is GELU
#trainining for 120000 epoch


################################## Distributed training approach #######################################################



def ModelAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg




########################## This is the overall AutoEncoder model ########################


class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()
        self.args = args
        ################## We use learnable positional encoder which can be removed later ######################################
        #self.pe = PositionalEncoder(SeqLen=self.args.K+1, lenWord=args.d_model_trx) # learnable PE
        self.pe = PositionalEncoder_fixed()
        ########################################################################################################################
        if args.embedding == True:
            self.Tmodel = BERT("trx", args.clas+2*(args.T-1), args.m, args.d_model_trx, args.N_trx, args.heads_trx, args.dropout, args.custom_attn,args.multclass, args.NS_model)
        else:
            self.Tmodel = BERT("trx", args.m+2*(args.T-1), args.m, args.d_model_trx, args.N_trx, args.heads_trx, args.dropout, args.custom_attn,args.multclass, args.NS_model)
        self.Rmodel = BERT("rec", args.T, args.m, args.d_model_trx, args.N_trx+1, args.heads_trx, args.dropout, args.custom_attn,args.multclass, args.NS_model)
        if args.rev_iter > 0: ###################### Here we perform succesive refinement ######################################
            self.RmodelB = BERT("rec", args.T+args.m, args.m, args.d_model_trx, args.N_trx, args.heads_trx, args.dropout, args.custom_attn,args.multclass, args.NS_model)
        ########## Power Reallocation as in deepcode work ###############
        if self.args.reloc == 1:
            self.total_power_reloc = Power_reallocate(args)

    def power_constraint(self, inputs, isTraining, eachbatch, idx = 0): # Normalize through batch dimension
        # this_mean = torch.mean(inputs, 0)
        # this_std  = torch.std(inputs, 0)
        if isTraining == 1:
            # training
            this_mean = torch.mean(inputs, 0)
            this_std  = torch.std(inputs, 0)
        elif isTraining == 0:
            # test
            if eachbatch == 0:
                this_mean = torch.mean(inputs, 0)
                this_std  = torch.std(inputs, 0)
                if not os.path.exists('statistics'):
                    os.mkdir('statistics')
                torch.save(this_mean, 'statistics/this_mean' + str(idx))
                torch.save(this_std, 'statistics/this_std' + str(idx))
                print('this_mean and this_std saved ...')
            else:
                this_mean = torch.load('statistics/this_mean' + str(idx))
                this_std = torch.load('statistics/this_std' + str(idx))

        outputs = (inputs - this_mean)*1.0/ (this_std + 1e-8)
        return outputs

    ########### IMPORTANT ##################
    # We use unmodulated bits at encoder
    #######################################
    def forward(self, eachbatch, bVec_md, fwd_noise_par, fb_noise_par, table = None, isTraining = 1):
        ###############################################################################################################################################################
        combined_noise_par = fwd_noise_par + fb_noise_par # The total noise for parity bits
        for idx in range(self.args.T): # Go through T interactions
            if idx == 0: # phase 0 
            	src = torch.cat([bVec_md, torch.zeros(self.args.batchSize, self.args.ell, 2*(self.args.T-1)).to(self.args.device)],dim=2)
            elif idx == self.args.T-1:
            	src = torch.cat([bVec_md, parity_all, combined_noise_par[:,:,:idx]],dim=2)
            else:
            	src = torch.cat([bVec_md, parity_all, torch.zeros(self.args.batchSize, args.ell, self.args.T-(idx+1) ).to(self.args.device),combined_noise_par[:,:,:idx],torch.zeros(self.args.batchSize, args.ell, self.args.T-(idx+1) ).to(self.args.device)],dim=2)
            ############# Generate the output ###################################################
            output = self.Tmodel(src, None, self.pe)
            parity = self.power_constraint(output, isTraining, eachbatch, idx) 
            parity = self.total_power_reloc(parity,idx)
            if idx == 0:
                parity_fb = parity + combined_noise_par[:,:,idx].unsqueeze(-1)
                parity_all = parity
                received = parity + fwd_noise_par[:,:,0].unsqueeze(-1)
            else:
                parity_fb = torch.cat([parity_fb, parity + combined_noise_par[:,:,idx].unsqueeze(-1)],dim=2) 
                parity_all = torch.cat([parity_all, parity], dim=2)     
                received = torch.cat([received, parity + fwd_noise_par[:,:,idx].unsqueeze(-1)], dim = 2)

        # ------------------------------------------------------------ receiver
        #print(received.shape)
        decSeq = self.Rmodel(received, None, self.pe) # Decode the sequence
        if args.rev_iter > 0:
            for i in range (args.rev_iter):
                if args.belief_modulate == True: # Modulate belief to align with the transmitted symbol power
                    belief = 2*torch.matmul(decSeq, table)-1
                else:
                     belief = torch.matmul(decSeq, table)
                received_wp = torch.cat([received,belief],dim=2)# received with prior
                decseq = self.RmodelB(received_wp, None, self.pe)
        return decSeq




############################################################################################################################################################################








def train_model(model, args):
    print("-->-->-->-->-->-->-->-->-->--> start training ...")
    model.train()
    start = time.time()
    epoch_loss_record = []
    flag = 0
    map_vec = 2**(torch.arange(args.m))
    ################################### Distance based vector embedding ####################
    A_blocks = torch.tensor([[0,0,0], [0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]],requires_grad=False).float() #Look up table for blocks
    Embed = torch.zeros(args.clas,args.batchSize, args.ell, args.clas)
    for i in range(args.clas):
        embed = torch.zeros(args.clas)
        for j in range(args.clas): ###### normalize vector embedding #########
            if args.embed_normalize == True:
                embed[j] = (torch.sum(torch.abs(A_blocks[i,:]-A_blocks[j,:]))-3/2)/0.866 # normalize embedding
            else:
                embed[j] = torch.sum(torch.abs(A_blocks[i,:]-A_blocks[j,:]))
        Embed[i,:,:,:]= embed.repeat(args.batchSize, args.ell, 1)
    #########################################################################################
    for eachbatch in range(args.totalbatch):
        if args.embedding == False:
            # BPSK modulated representations 
            bVec = torch.randint(0, 2, (args.batchSize, args.ell, args.m))
            bVec_md = 2*bVec-1
        else: # vector embedding
            bVec = torch.randint(0, args.clas, (args.batchSize, args.ell, 1))
            bVec_md = torch.zeros((args.batchSize, args.ell,args.clas), requires_grad=False) # generated data in terms of distance embeddings
            for i in range(args.clas):
                mask = (bVec == i).long()
                bVec_md= bVec_md + (mask * Embed[i,:,:,:])
        #################################### Generate noise sequence ##################################################
        ###############################################################################################################
        ###############################################################################################################
        ################################### Curriculum learning strategy ##############################################
        snr2=args.snr2
        if eachbatch < args.core * 30000:
           snr1=4* (1-eachbatch/(args.core * 30000))+ (eachbatch/(args.core * 30000)) * args.snr1
        else:
           snr1=args.snr1
        ################################################################################################################
        std1 = 10 ** (-snr1 * 1.0 / 10 / 2) #forward snr
        std2 = 10 ** (-snr2 * 1.0 / 10 / 2) #feedback snr
        # Noise values for the parity bits
        fwd_noise_par = torch.normal(0, std=std1, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        fb_noise_par = torch.normal(0, std=std2, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        if args.snr2 == 100:
            fb_noise_par = 0* fb_noise_par
        if np.mod(eachbatch, args.core) == 0:
            w_locals = []
            w0 = model.state_dict()
            w0 = copy.deepcopy(w0)
        else:
            # Use the common model to have a large batch strategy
            model.load_state_dict(w0)

        # feed into model to get predictions
        preds = model(eachbatch, bVec_md.to(args.device), fwd_noise_par.to(args.device), fb_noise_par.to(args.device), A_blocks.to(args.device), isTraining=1)

        args.optimizer.zero_grad()
        if args.multclass:
           if args.embedding == False:
               bVec_mc = torch.matmul(bVec,map_vec)
               ys = bVec_mc.long().contiguous().view(-1)
           else:
               ys = bVec.contiguous().view(-1)
        else:
        # expand the labels (bVec) in a batch to a vector, each word in preds should be a 0-1 distribution
           ys = bVec.long().contiguous().view(-1)
        preds = preds.contiguous().view(-1, preds.size(-1)) #=> (Batch*K) x 2
        preds = torch.log(preds)
        loss = F.nll_loss(preds, ys.to(args.device))########################## This should be binary cross-entropy loss
        loss.backward()
        ####################### Gradient Clipping optional ###########################
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_th)
        ##############################################################################
        args.optimizer.step()
        # Save the model
        w1 = model.state_dict()
        w_locals.append(copy.deepcopy(w1))
        ###################### untill core number of iterations are completed ####################
        if np.mod(eachbatch, args.core) != args.core - 1:
            continue
        else:
            ########### When core number of models are obtained #####################
            w2 = ModelAvg(w_locals) # Average the models
            model.load_state_dict(copy.deepcopy(w2))
            ##################### change the learning rate ##########################
            if args.use_lr_schedule:
                args.scheduler.step()
        ################################ Observe test accuracy ##############################
        with torch.no_grad():
            probs, decodeds = preds.max(dim=1)
            succRate = sum(decodeds == ys.to(args.device)) / len(ys)
            print('GBAF_FESv1','Idx,lr,snr1,snr2,BS,loss,BER,num=', (
            eachbatch, args.lr,args.snr1, args.snr2, args.batchSize, round(loss.item(), 4), round(1 - succRate.item(), 6),
            sum(decodeds != ys.to(args.device)).item()))
        ####################################################################################
        # if np.mod(eachbatch, args.core * 50) == args.core - 1:
        #     epoch_loss_record.append(loss.item())
        #     if not os.path.exists('weights'):
        #         os.mkdir('weights')
        #     torch.save(epoch_loss_record, 'weights/loss')

        if np.mod(eachbatch, args.core * 5000) == args.core - 1 and eachbatch >= 80000:
            if not os.path.exists('weights'):
                os.mkdir('weights')
            saveDir = 'weights/model_weights' + str(eachbatch)
            torch.save(model.state_dict(), saveDir)


def EvaluateNets(model, args):
    checkpoint = torch.load(args.saveDir)
    # # ======================================================= load weights
    model.load_state_dict(checkpoint)
    model.eval()
    map_vec = 2**(torch.arange(args.m))

    args.numTestbatch = 100000000
    ################################### Distance based vector embedding ####################
    A_blocks = torch.tensor([[0,0,0], [0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]],requires_grad=False).float() #Look up table for blocks
    Embed = torch.zeros(args.clas,args.batchSize, args.ell, args.clas)
    for i in range(args.clas):
        embed = torch.zeros(args.clas)
        for j in range(args.clas):
            if args.embed_normalize == True:
                embed[j] = (torch.sum(torch.abs(A_blocks[i,:]-A_blocks[j,:]))-3/2)/0.866
            else:
                embed[j] = torch.sum(torch.abs(A_blocks[i,:]-A_blocks[j,:]))
        Embed[i,:,:,:]= embed.repeat(args.batchSize, args.ell, 1)
    # failbits = torch.zeros(args.K).to(args.device)
    bitErrors = 0
    pktErrors = 0
    for eachbatch in range(args.numTestbatch):
        if args.embedding == False:
            # BPSK modulated representations 
            bVec = torch.randint(0, 2, (args.batchSize, args.ell, args.m))
            bVec_md = 2*bVec-1
        else: # vector embedding
            bVec = torch.randint(0, args.clas, (args.batchSize, args.ell, 1))
            bVec_md = torch.zeros((args.batchSize, args.ell,args.clas), requires_grad=False) # generated data in terms of distance embeddings
            for i in range(args.clas):
                mask = (bVec == i).long()
                bVec_md= bVec_md + (mask * Embed[i,:,:,:])
        # generate n sequence
        std1 = 10 ** (-args.snr1 * 1.0 / 10 / 2)
        std2 = 10 ** (-args.snr2 * 1.0 / 10 / 2)
        fwd_noise_par = torch.normal(0, std=std1, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        fb_noise_par = torch.normal(0, std=std2, size=(args.batchSize, args.ell, args.T), requires_grad=False)
        if args.snr2 == 100:
            fb_noise_par = 0* fb_noise_par

        # feed into model to get predictions
        with torch.no_grad():
            preds = model(eachbatch, bVec_md.to(args.device), fwd_noise_par.to(args.device), fb_noise_par.to(args.device), A_blocks.to(args.device), isTraining=0)
            if args.multclass:
                if args.embedding == False:
                    bVec_mc = torch.matmul(bVec,map_vec)
                    ys = bVec_mc.long().contiguous().view(-1)
                else:
                    ys = bVec.contiguous().view(-1)
            else:
                ys = bVec.long().contiguous().view(-1)
            preds1 =  preds.contiguous().view(-1, preds.size(-1))
            #print(preds1.shape)
            probs, decodeds = preds1.max(dim=1)
            decisions = decodeds != ys.to(args.device)
            bitErrors += decisions.sum()
            BER = bitErrors / (eachbatch + 1) / args.batchSize / args.ell
            pktErrors += decisions.view(args.batchSize, args.ell).sum(1).count_nonzero()
            PER = pktErrors / (eachbatch + 1) / args.batchSize
            print('GBAF_FESv1','num, BER, errors, PER, errors = ', eachbatch, round(BER.item(), 10), bitErrors.item(),
                  round(PER.item(), 10), pktErrors.item(), )

    BER = bitErrors.cpu() / (args.numTestbatch * args.batchSize * args.K)
    PER = pktErrors.cpu() / (args.numTestbatch * args.batchSize)
    print(BER)
    print("Final test BER = ", torch.mean(BER).item())
    pdb.set_trace()


if __name__ == '__main__':
    # ======================================================= parse args
    args = args_parser()
    #args.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    ########### path for saving model checkpoints ################################
    args.saveDir = 'weights/model_weights120000'  # path to be saved to
    ################## Model size part ###########################################
    args.d_model_trx = args.heads_trx * args.d_k_trx # total number of features
    # ======================================================= Initialize the model
    model = AE(args).to(args.device)
    if args.device == 'cuda':
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    # ======================================================= run
    if args.train == 1:
        if args.opt_method == 'adamW':
        	args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
        elif args.opt_method == 'lamb':
        	args.optimizer = optim.Lamb(model.parameters(),lr= 1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
        else:
        	args.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        if args.use_lr_schedule:
        	lambda1 = lambda epoch: (1-epoch/args.totalbatch)
        	args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambda1)
        	######################## huggingface library ####################################################
        	#args.scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=args.optimizer, warmup_steps=1000, num_training_steps=args.totalbatch, power=0.5)


        if 0:
            checkpoint = torch.load(args.saveDir)
            model.load_state_dict(checkpoint)
            print("================================ Successfully load the pretrained data!")

        train_model(model, args)
    else:
        EvaluateNets(model, args)
