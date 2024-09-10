import torch
import numpy as np
from loss.metric import *
from tqdm import tqdm

def train_one_epoch(epoch, train_loader, net, criterion, sgdr, optimizer, config):
    sgdr.step()
    lr = optimizer.param_groups[0]['lr']
    print('epoch-{:}: lr={:.4f}'.format(epoch, lr))
    batch_loss = np.zeros(6, np.float32)
    r1, r2 = config.bce, config.pwl
    r3 = config.cmfl
    optimizer.zero_grad()
    for input, mask, truth in train_loader:
        # one iteration update
        net.train()
        input = input.cuda()
        truth = truth.cuda()
        mask = mask.cuda()
        logit, depth_logits, color_logits, ir_logits, x_map = net.forward(input)
        truth = truth.view(logit.shape[0])

        x_map = x_map.to(torch.float32)
        mask = mask.to(torch.float32)
        loss_pwl = criterion['PWL'](x_map, mask)
        loss1 = criterion['BCE'](logit, truth)
        # loss = r1*loss1 + r2*loss_pwl

        # loss2 = criterion['BCE'](depth_logits, truth)
        # loss3 = criterion['BCE'](color_logits, truth)
        # loss4 = criterion['BCE'](ir_logits, truth)
        # loss = loss2 + loss3 + loss4
        # loss2 = 0.5 * loss2 + 0.5 * loss3 + 0.5 * loss4
        # loss = loss2 + loss3 + loss4
        # loss = r1*loss1 + loss2 + r2*loss_pwl

        
        # loss_w, loss_r, loss2 = criterion['CMFL'](color_logits, depth_logits, truth)
        # loss = r1*loss1 + r2*loss_pwl + r3 * loss2

        loss_r,loss_d,loss_i,loss2 = criterion['ICMFL'](color_logits, depth_logits, ir_logits, truth)
        loss = r1*loss1 + r2*loss_pwl + r3 * loss2
        # loss = loss1 + r3 * loss2
        # logit = (depth_logits + color_logits + ir_logits) / 3.0

        precision, _ = metric(logit, truth)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_loss[:2] = np.array((loss.item(), precision.item(),))  # 损失函数和准确率 [2.6929493 0.671875 ]

    return batch_loss, lr

def do_test(net, test_loader, criterion, config):
    valid_num = 0
    losses = []
    corrects = []
    probs = []
    probs1, probs2, probs3, probs4 = [], [], [], []
    labels = []
    # for i, (input, truth) in enumerate(tqdm(test_loader)):
    for i, (input, mask, truth) in enumerate(test_loader):
        b, n, c, w, h = input.size()
        input = input.view(b * n, c, w, h)
        input = input.cuda()
        truth = truth.cuda()
        with torch.no_grad():
            logit, depth_logits, color_logits, ir_logits, x_map = net(input)
            logit = logit.view(b, n, logit.shape[1])

            # color_logits = color_logits.view(b, n, color_logits.shape[1])
            # depth_logits = depth_logits.view(b, n, depth_logits.shape[1])
            # ir_logits = ir_logits.view(b, n, ir_logits.shape[1])

            logit = torch.mean(logit, dim=1, keepdim=False)

            # color_logits = torch.mean(color_logits, dim=1, keepdim=False)
            # depth_logits = torch.mean(depth_logits, dim=1, keepdim=False)
            # ir_logits = torch.mean(ir_logits, dim=1, keepdim=False)

            # logit = (depth_logits + color_logits + ir_logits) / 3.0

            truth = truth.view(b)
            loss = criterion['BCE'](logit, truth)
            correct, prob = metric(logit, truth)
            # correct1, prob1 = metric(depth_logits, truth)
            # correct2, prob2 = metric(color_logits, truth)
            # correct3, prob3 = metric(ir_logits, truth)
            

        valid_num += len(input)
        losses.append(loss.data.cpu().numpy())
        corrects.append(np.asarray(correct).reshape([1]))
        probs.append(prob.data.cpu().numpy())
        labels.append(truth.data.cpu().numpy())

        # probs1.append(prob1.data.cpu().numpy())
        # probs2.append(prob2.data.cpu().numpy())
        # probs3.append(prob3.data.cpu().numpy())
        

    correct = np.concatenate(corrects)
    correct = np.mean(correct)
    loss = np.array(losses)
    loss = loss.mean()
    probs = np.concatenate(probs)
    probs = probs[:, 1]
    labels = np.concatenate(labels)

    
    """
    probs1 = np.concatenate(probs1)
    probs1 = probs1[:, 1].T
    test_eval1, test_TPR_FPRS1, threshold1 = model_performances(probs1, labels)

    probs2 = np.concatenate(probs2)
    probs2 = probs2[:, 1].T
    test_eval_rgb, test_TPR_FPRS_rgb, threshold_rgb = model_performances(probs2, labels)

    probs3 = np.concatenate(probs3)
    probs3 = probs3[:, 1].T
    test_eval_ir, test_TPR_FPRS_ir, threshold_ir = model_performances(probs3, labels)
    
    
    print(f"Depth分支===ACC: {test_eval1['ACC']}, ACER: {test_eval1['ACER']}, APCER: {test_eval1['APCER']}, BPCER: {test_eval1['BPCER']}\n"
          f"color分支===ACC: {test_eval_rgb['ACC']}, ACER: {test_eval_rgb['ACER']}, APCER: {test_eval_rgb['APCER']}, BPCER: {test_eval_rgb['BPCER']}\n"
          f"ir分支===   ACC: {test_eval_ir['ACC']}, ACER: {test_eval_ir['ACER']}, APCER: {test_eval_ir['APCER']}, BPCER: {test_eval_ir['BPCER']}\n"
          )
    """
    return loss, correct, probs, labels