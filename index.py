import os
import torch
import copy
import time
import datetime
import pickle
import numpy as np
# from tqdm import tqdm
from torch import nn
from utils import iprint
from cli import get_args
from data_preprocess import get_all_data
from models.trade import Trade
from config import PAD_TOKEN, SLOT_GATE_DICT, SLOT_GATE_DICT_INVERSE
from masked_cross_entropy import masked_cross_entropy_for_value

datetimestr = datetime.datetime.now().strftime('%Y-%b-%d_%H:%M:%S')
saving_dir = './save/{}/'.format(datetimestr)
if not os.path.exists('./save'):
    os.mkdir(path='./save')
os.mkdir(path=saving_dir)

def train_model(model, device, dataloaders, slots_dict, criterion_ptr, criterion_gate, optimizer, scheduler, clip, num_epochs, print_iter, patience):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_joint_acc = 0.0

    patience_counter = 0

    # Statistics
    results = {
        'train': {
            'loss_ptr': [],
            'loss_gate': [],
            'joint_acc': [],
            'turn_acc': [],
            'f1': []
        },
        'val': {
            'loss_ptr': [],
            'loss_gate': [],
            'joint_acc': [],
            'turn_acc': [],
            'f1': []
        }
    }

    for n_epoch in range(args['epoch']):
        print('Epoch {}'.format(n_epoch))
        print('=' * 20)

        for phase in ['train', 'val']:
            print('Phase [{}]'.format(phase))
            print('-' * 10)

            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode

            predictions = {}

            dataloader = dataloaders[phase]
            for iteration, data in enumerate(dataloader):
                data['context'] = data['context'].to(device=device)
                data['generate_y'] = data['generate_y'].to(device=device)
                data['y_lengths'] = data['y_lengths'].to(device=device)
                data['turn_domain'] = data['turn_domain'].to(device=device)
                data['gating_label'] = data['gating_label'].to(device=device)
                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    all_point_outputs, all_gate_outputs, words_point_out = model(data=data, slots_type=phase)

                    # logits = all_point_outputs.transpose(0, 1).transpose(1, 3).transpose(2, 3).contiguous()
                    # targets = data["generate_y"].contiguous()
                    # loss_ptr = criterion_ptr(logits, targets)

                    loss_ptr = masked_cross_entropy_for_value(
                        all_point_outputs.transpose(0, 1).contiguous(),
                        data["generate_y"].contiguous(), # [:,:len(self.point_slots)].contiguous(),
                        data["y_lengths"]
                    )

                    logits_gate = all_gate_outputs.transpose(1, 2).contiguous()
                    targets_gate = data["gating_label"].t().contiguous()
                    loss_gate = criterion_gate(logits_gate, targets_gate)

                    loss = loss_ptr + loss_gate

                    if phase == 'train' and iteration % print_iter == 0:
                        print('Iteration {}: loss_ptr = {:4f}, loss_gate = {:4f}'.format(iteration, loss_ptr, loss_gate))

                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                        optimizer.step()

            # Calculate evaluation metrics and save statistics to file
            accumulate_result(predictions, data, slots_dict[phase], all_gate_outputs, words_point_out)
            joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr = evaluate_metrics(
                predictions,
                "pred_bs_ptr",
                slots_dict[phase]
            )
            results[phase]['loss_ptr'].append(loss_ptr.item())
            results[phase]['loss_gate'].append(loss_gate.item())
            results[phase]['joint_acc'].append(joint_acc_score_ptr)
            results[phase]['turn_acc'].append(turn_acc_score_ptr)
            results[phase]['f1'].append(F1_score_ptr)

            print("Joint Acc: {:.4f}".format(joint_acc_score_ptr))
            print("Turn Acc: {:.4f}".format(turn_acc_score_ptr))
            print("Joint F1: {:.4f}".format(F1_score_ptr))

            pickle.dump(results, open(os.path.join(saving_dir, 'results.p'), 'wb'))

            # deep copy the model
            if phase == 'val':
                scheduler.step(joint_acc_score_ptr)
                if joint_acc_score_ptr > best_joint_acc:
                    patience_counter = 0
                    best_joint_acc = joint_acc_score_ptr
                    best_model_wts = copy.deepcopy(model.state_dict())
                    model_save_path = os.path.join(saving_dir, 'model-joint_acc-{:.4f}.pt'.format(best_joint_acc))
                    torch.save(model, model_save_path)

        if patience_counter == patience:
            iprint('Early stop at epoch {}'.format(n_epoch))
            break

        print()

    time_elapsed = time.time() - since
    iprint('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    iprint('Best validation joint_accurate: {:4f}'.format(best_joint_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def accumulate_result(predictions, data, slots, all_gate_outputs, words_point_out):
    batch_size = len(data['context_len'])
    for bi in range(batch_size):
        if data["ID"][bi] not in predictions.keys():
            predictions[data["ID"][bi]] = {}
        predictions[data["ID"][bi]][data["turn_id"][bi]] = {"turn_belief": data["turn_belief"][bi]}
        predict_belief_bsz_ptr = []
        gate = torch.argmax(all_gate_outputs.transpose(0, 1)[bi], dim=1)

        # pointer-generator results
        if args["use_gate"]:
            for si, sg in enumerate(gate):
                if sg == SLOT_GATE_DICT["none"]:
                    continue
                elif sg == SLOT_GATE_DICT["ptr"]:
                    pred = np.transpose(words_point_out[si])[bi]
                    st = []
                    for e in pred:
                        if e == 'EOS':
                            break
                        else:
                            st.append(e)
                    st = " ".join(st)
                    if st == "none":
                        continue
                    else:
                        predict_belief_bsz_ptr.append(slots[si] + "-" + str(st))
                else:
                    predict_belief_bsz_ptr.append(slots[si] + "-" + SLOT_GATE_DICT_INVERSE[sg.item()])
        else:
            for si, _ in enumerate(gate):
                pred = np.transpose(words_point_out[si])[bi]
                st = []
                for e in pred:
                    if e == 'EOS':
                        break
                    else:
                        st.append(e)
                st = " ".join(st)
                if st == "none":
                    continue
                else:
                    predict_belief_bsz_ptr.append(slots[si] + "-" + str(st))

        predictions[data["ID"][bi]][data["turn_id"][bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr


def evaluate_metrics(all_prediction, from_which, slot_temp):
    total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
    for d, v in all_prediction.items():
        for t in range(len(v)):
            cv = v[t]
            if set(cv["turn_belief"]) == set(cv[from_which]):
                joint_acc += 1
            total += 1

            # Compute prediction slot accuracy
            temp_acc = compute_acc(set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
            turn_acc += temp_acc

            # Compute prediction joint F1 score
            temp_f1, temp_r, temp_p, count = compute_prf(set(cv["turn_belief"]), set(cv[from_which]))
            F1_pred += temp_f1
            F1_count += count

    joint_acc_score = joint_acc / float(total) if total != 0 else 0
    turn_acc_score = turn_acc / float(total) if total != 0 else 0
    F1_score = F1_pred / float(F1_count) if F1_count != 0 else 0
    return joint_acc_score, F1_score, turn_acc_score


def compute_acc(gold, pred, slot_temp):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC

def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count


if __name__ == '__main__':
    args = get_args()
    # Only set the GPU to be used visible, and so just specify cuda:0 as the device
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    iprint('Using device = {}'.format(device))

    (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        test_special,
        vocabs,
        slots_dict,
        max_word
    ) = get_all_data(args=args, training=True, batch_size=args['batch'])

    model = Trade(
        args=args,
        device=device,
        slots_dict=slots_dict,
        vocabs=vocabs
    )
    model = model.to(device=device)

    criterion_ptr = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_gate = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, min_lr=0.0001, verbose=True)

    train_model(
        model=model,
        device=device,
        dataloaders={
            'train': train_dataloader,
            'val': dev_dataloader
        },
        slots_dict=slots_dict,
        criterion_ptr=criterion_ptr,
        criterion_gate=criterion_gate,
        optimizer=optimizer,
        scheduler=scheduler,
        clip=args['clip'],
        num_epochs=args['epoch'],
        print_iter=args['print_iter'],
        patience=args['patience']
    )

    # for data in train_dataloader:
    #     print(model(data=data, slots_type='train'))
    #     # print(data)
    #     exit(1)

    # TODO: make sure all tensors are properly set using .to(device) or .cuda()
