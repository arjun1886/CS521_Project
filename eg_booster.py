from absl import app, flags
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle
from captum.attr import GradientShap
import os,sys
import time
import matplotlib.pyplot as plt
import random
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.spsa import spsa
from robustness.datasets import audiomnist
import h5py

FLAGS = flags.FLAGS

# Get current working directory
cwd = os.getcwd()

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
    
batch_size = 64
num_workers = 2
use_cuda = False
device = torch.cuda.device("cuda" if use_cuda else "cpu")


def ld_data(train=True):
  """Load test data."""
  
  transform_test = transforms.Compose([
  transforms.ToTensor()])
  
    
  audiomnist_testset = datasets.AudioMNIST(root='./data', train=False, download=True, transform=transform_test)
  test_loader = torch.utils.data.DataLoader(audiomnist_testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

  return test_loader

def load_model():
    with h5py.File('weights-improvement_0.94.hdf5', 'r') as hdf5_file:
        # Extract architecture and weights
        architecture = hdf5_file.attrs['model_config']
        weights = hdf5_file['model_weights']

        # Convert architecture from bytes to string and eval
        architecture = architecture.decode('utf-8')
        model = eval(architecture)

        # Load weights into the PyTorch model
        model.load_state_dict({k: torch.from_numpy(v.value) for k, v in weights.items()})

    return model
    
def test_accuracy(net, testset_loader):
    # Test the model
    net.eval()
    correct = 0
    total = 0

    for data in testset_loader:
        images, labels = data
        output = net(images)
        
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the network is: ' + str(100 * correct / total))
    del images
    del labels
    torch.cuda.empty_cache()
    
def predict(model,img):
    

    img = img.cuda()    
    output = model(img)
    _, predicted = torch.max(output.data, 1)
    del img
    torch.cuda.empty_cache()
    return predicted.item()


def prtedict_batch(model, x_batch):
    
    loader = torch.utils.data.DataLoader(x_batch,batch_size)
    
    s = 0
    for batch in loader:
      
        output = model(batch)
        if s==0:
            _, predicted = torch.max(output.data, 1)
        else:
            _, predicted_batch = torch.max(output.data, 1)
            predicted = torch.cat((predicted,predicted_batch),0)
        s+=batch_size
    
    del batch
    del predicted_batch
    torch.cuda.empty_cache()
    
    return predicted
        
def evasion_rate(model, adv_imgs,labels):
    
    model.eval()
    evaded = 0
    total = 0

    loader = torch.utils.data.DataLoader(adv_imgs,batch_size)
    
    s = 0
    e = batch_size
    for batch in loader:
      
        output = model(batch)
        _, predicted = torch.max(output.data, 1)
        total += labels[s:e].size(0)
        evaded += (predicted != labels[s:e].cuda()).sum()
        s+=batch_size
        e+=batch_size
        
    del batch
    del predicted
    torch.cuda.empty_cache()
    
    return 100 * evaded / total
    

def HSJ(model,test_loader,norm,size,defended=False):
    
    '''perform hop_skip_jump_attack'''
    
    print('Starting hop_skip_jump_attack attack...')
    i=0
    for data in test_loader:
        x_test, y_test = data
        x_test = x_test.cuda()        
        
        if i == 0:
            print('perturbing batch ',i+1)
            x_adv = hop_skip_jump_attack(model, x_test,batch_size=batch_size,norm = norm,verbose=0).cpu()#
            labels = y_test.cpu()
            torch.cuda.empty_cache()
        else:
            print('perturbing batch ',i+1)
            x_adv = torch.cat((x_adv, hop_skip_jump_attack(model, x_test,batch_size=batch_size, norm = norm,verbose=0).cpu()),0)#,batch_size=batch_size
            labels = torch.cat((labels ,y_test.cpu()),(y_test+1)%10) # target it to 1 more than y_test
            torch.cuda.empty_cache()
        i+=1
        if i>0:
            print('The average evasion rate of HSJ is :',evasion_rate(model, x_adv.cuda(),labels.cuda()))
            torch.cuda.empty_cache()
        if x_adv.shape[0] >= size:
            break
        #print(x_cw.shape)
        

    f = open(os.path.join(cwd,'audiomnist','undefended','audiomnist-HSJ'+str(norm)), 'wb')
    pickle.dump(x_adv, f)
    f.close()
    
    return x_adv

   
def FGS(model,test_loader,norm,size,eps,defended=False):
    '''Perform FGSM attack'''
    
    
    print('Starting FGS attack...')
    i=0
    
    for data in test_loader:
        x_test, y_test = data
        
        x_test=x_test.cuda()
        #y_test=y_test.cuda()
        
        if i == 0:
            x_fgm = fast_gradient_method(model, x_test,eps=eps, norm=norm)
            labels = y_test
        else:
            x_fgm = torch.cat((x_fgm, fast_gradient_method(model, x_test, eps = eps, norm = norm)),0)
            labels = torch.cat((labels ,y_test),(y_test+1)%10) # target it to 1 more than y_test
        i+=1
        
        print(x_fgm.shape)
        print('The current evasion rate of FGS is :',evasion_rate(model, x_fgm,labels.cuda()))
        if x_fgm.shape[0]>=size:
            break
        
        
    # save fgm data
    del labels
    del x_test
    torch.cuda.empty_cache()
    
    f = open(os.path.join(cwd,'audiomnist','defended','audiomnist-FGSL'+str(norm)+str(eps)), 'wb')
    
    pickle.dump(x_fgm, f)
    f.close()
    
    return x_fgm


####################################################### EG-Booster Functions ##################################################
def deepexplain(test_loader,path,size, model):
    
    
    gradient_shap = GradientShap(model)
    
    shp =[]
    start_time = time.time()
    j=0
    for data in test_loader:
        org_img, labels = data
        del data
        torch.cuda.empty_cache()
        org_img = torch.cuda.FloatTensor(org_img.cuda())
        print('Explainig batch {}'.format(j+1))
        for i in range(labels.shape[0]):
            rand_img_dist = torch.cat([org_img[i:i+1] * 0, org_img[i:i+1] * 255])
            attributions_gs = gradient_shap.attribute(org_img[i:i+1],
                                          n_samples=100,
                                          stdevs=0.0001,
                                          baselines=rand_img_dist,
                                          target=labels[i].item())
            
            shp_i = attributions_gs.squeeze().cpu().detach().numpy()
            shp.append(shp_i)
        if j>=int(size/batch_size):
            break
        j+=1
    
    print((time.time() - start_time)/3600,' hrs')
    shp = np.array(shp)
    np.save(path, shp)
    torch.cuda.empty_cache()
    
    return shp



def One_EG_Attack(org_img,adv_img,shaply,label,model,eps,norm_type):

    
    lines = 32
    cols = 32
    channels = 3
    ege_best = adv_img.detach().clone()
    pos_unperturbed = 0
    tot = 0
    pert_change = 0
    out_bound = 0
    attmpt = 0
    
    norm = round(torch.norm((adv_img.cpu() - org_img.cpu()).cuda(), norm_type).item(),2)
    
    if norm > eps:
        raise ValueError("norm {} cannot be used : Choose the same norm order used for initial attack".format(norm_type))
    
    for i in range(channels):
        for j in range(cols):
            for k in range(lines):
                
                if adv_img[i,j,k].item() != org_img[i,j,k].item():
                    tot+=1
                    
                    if shaply[i,j,k] < 0:
                        
                        ege_best[i,j,k] = org_img[i,j,k].item()
                        if predict(model,adv_img) != label and predict(model,ege_best) == label:
                            ege_best[i,j,k] = adv_img[i,j,k].item()
                        else:
                            pert_change-=1
    
    if predict(model,ege_best) == label:
        
        for i in range(channels):
            for j in range(cols):
                for k in range(lines):
                            
                    if shaply[i,j,k] >= 0:
                        attmpt+=1
                        
                        if adv_img[i,j,k].item() == org_img[i,j,k].item():
                            pos_unperturbed+=1
                        
                        if predict(model,ege_best) == label:
                            pert = random.random()
                            ege_best[i,j,k] += pert
                            
                            
                            attempt = 0
                            reduce = 1
                            while attempt < 20 and round(torch.norm((ege_best.cpu()- org_img.cpu()).cuda(), norm_type).item(),2) > eps:
                                print("{} > {} : perturbation out of bound! reducing the perturbation...".format(round(torch.norm((ege_best.cpu()- org_img.cpu()).cuda(), norm_type).item(),2), eps))
                                reduce +=1
                                pert = pert/reduce
                                ege_best[i,j,k] = adv_img[i,j,k] + pert
                                attempt += 1
                            
                            if round(torch.norm((ege_best.cpu() - org_img.cpu()).cuda(), norm_type).item(),2) > eps:
                                ege_best[i,j,k] = adv_img[i,j,k]
                                out_bound+=1
                            
                        else:
                            break
    
    
    if attmpt != 0:
        out_bound_rate = out_bound/attmpt
    else:
        out_bound_rate = 0
    if tot !=0:
        change_rate = pert_change/float(tot)
    else:
        change_rate = 0
    return ege_best,change_rate  , pos_unperturbed,out_bound_rate
    
    
def EG_Attack(start,test_loader,adv_loader,model,shp,attack_name,eps, norm_type, run,size,defended=False):
    
    
    if norm_type not in [np.inf, 1, 2]:
        raise ValueError(
            "Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm_type)
        )
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    out_boundAll = []
    # Get original data
    i=0
    start_time = time.time()
    max_change = 0
    tot_change = []
    tot_pos =0
    
    for data,adv_imgs  in zip(test_loader,adv_loader):
        x_batch, y_batch = data
        if i==0:
            labels = y_batch
        else:
            labels = torch.cat((labels, y_batch))
        
        print('Performing EG-Booster on batch {} ...'.format(i+1))
        for ind in range(batch_size):
            
            new_adv , pert_change, pos_unpert, out_bound = One_EG_Attack(x_batch[ind],adv_imgs[ind],shp[ind+i*batch_size],y_batch[ind].item(),model,eps,norm_type)
            
            
            if abs(pert_change) > max_change:
                max_change = pert_change
            tot_change.append(pert_change)
            tot_pos += pos_unpert
            out_boundAll.append(out_bound)
            if ind == 0 and i==0:
                best_adv=new_adv.reshape((1,3,32,32))
            else:
                best_adv = torch.cat((best_adv,new_adv.reshape((1,3,32,32))),0)
            
        print('Processed shape:',best_adv.shape)  
        print('Current EG-Booster Evasion rate on {} is {} %'.format(attack_name,evasion_rate(model, torch.cuda.FloatTensor(best_adv.cuda()),labels.cuda())))      
        if labels.shape[0] == size:
            break
        i+=1
    
    

    
    print('Average Evasion Rate Change of EG-Booster attack is {} %'.format(evasion_rate(model, torch.cuda.FloatTensor(best_adv.cuda()),labels.cuda())))
    

    print('Avg perturbation change {}%'.format(np.mean(out_boundAll)*100))

        
def main(_):
  
  torch.cuda.empty_cache()
  print('Loading data...')
  test_loader = ld_data(train=False)
  
  print('Loading model...')
  model = load_model()
  model.cuda()
  model.load_state_dict(torch.load(os.path.join(cwd,'audiomnist',"model_with_epoch" + str(50) + ".pth")))
  
  #do FGSM attack
  x_fgm = FGS(model,test_loader,FLAGS.norm,FLAGS.size,FLAGS.eps,defended=FLAGS.defended)
  
  
  # Create a loader for baseline attack data
  loader = torch.utils.data.DataLoader(x_fgm,batch_size)
  del x_fgm
  torch.cuda.empty_cache()
  
  print('\n Performing EG-Booster on FGSM...')
  _ = EG_Attack(0,test_loader,loader,model,'','FGS',FLAGS.eps, FLAGS.norm,1,FLAGS.size,defended=FLAGS.defended)

  #do HSJA attack
  x_hsj = HSJ(model,test_loader,FLAGS.norm,FLAGS.size,FLAGS.eps,defended=FLAGS.defended)
  
  
  # Create a loader for baseline attack data
  loader = torch.utils.data.DataLoader(x_hsj,batch_size)
  del x_hsj
  torch.cuda.empty_cache()
  
  print('\n Performing EG-Booster on HSJA...')
  _ = EG_Attack(0,test_loader,loader,model,'','HSJ',FLAGS.eps, FLAGS.norm,1,FLAGS.size,defended=FLAGS.defended)

  
if __name__ == '__main__':
    flags.DEFINE_float("eps", 0.3, "Total epsilon for FGS and HSJA attacks.")
    flags.DEFINE_integer("norm", 2, "Used distance metric.")
    flags.DEFINE_bool(
        "defended", False, "Use the undefended model"
    )

    app.run(main)