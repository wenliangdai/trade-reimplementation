import os
import torch
import copy
import time
import numpy as np
from tqdm import tqdm
from torch import cuda
from torch import nn
from utils import iprint
from cli import get_args
from data_preprocess import get_all_data
from models.trade import Trade
from config import PAD_TOKEN, SLOT_GATE_DICT, SLOT_GATE_DICT_INVERSE


def train_model(model, device, dataloaders, slots_dict, criterion_ptr, criterion_gate, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    total_progress_bar = tqdm(range(args['epoch']))
    for n_epoch in total_progress_bar:
        # iprint('Epoch {}'.format(n_epoch))
        # print('-' * 10)
        total_progress_bar.set_description('Training progress (current epoch {})'.format(n_epoch + 1))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode

            running_loss_ptr = 0.0
            running_loss_gate = 0.0

            predictions = {}

            dataloader = dataloaders[phase]
            current_epoch_progress_bar = tqdm(dataloader, total=len(dataloader))
            current_epoch_progress_bar.set_description(phase)
            for data in tqdm(dataloader, total=len(dataloader)):
                data['context'].to(device=device)
                data['generate_y'].to(device=device)
                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    all_point_outputs, all_gate_outputs, words_point_out = model(data=data, slots_type=('train' if phase == 'train' else 'dev'))

                    logits = all_point_outputs.transpose(1, 3).transpose(0, 2).contiguous()
                    targets = data["generate_y"].transpose(0, 2).contiguous()
                    loss_ptr = criterion_ptr(logits, targets)

                    logits_gate = all_gate_outputs.transpose(1, 2).contiguous()
                    targets_gate = data["gating_label"].t().contiguous()
                    loss_gate = criterion_gate(logits_gate, targets_gate)

                    loss = loss_ptr + loss_gate

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # Calculate validation metrics
                    if phase == 'val':
                        # b_gates = all_gate_outputs.transpose(0, 1).argmax(dim=2).numpy()
                        # for i, gates in enumerate(b_gates):
                        #     b_gates[i] = list(filter(lambda n: n != SLOT_GATE_DICT['none'], gates))
                        # for i, gates in enumerate(b_gates):
                        #     for j, gate_val in enumerate(gates):
                        #         b_gates[i, j] = np.tranpose(words_point_out[j])[i]
                        #         b_gates[i, j] = ' '.join(list(filter(lambda w: w != 'EOS', b_gates[i, j])))
                        #     b_gates[i] = list(filter(lambda s: s != 'none', b_gates[i]))
                        # data['turn_belief']
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
                                            predict_belief_bsz_ptr.append(slots_dict['dev'][si] + "-" + str(st))
                                    else:
                                        predict_belief_bsz_ptr.append(slots_dict['dev'][si] + "-" + SLOT_GATE_DICT_INVERSE[sg.item()])
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
                                        predict_belief_bsz_ptr.append(slots_dict['dev'][si] + "-" + str(st))

                            predictions[data["ID"][bi]][data["turn_id"][bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr

                # statistics
                running_loss_ptr += loss_ptr.item() * np.sum(list(targets.size()))
                running_loss_gate += loss_gate.item() * np.sum(list(targets_gate.size()))

            if phase == 'train':
                scheduler.step()

            # deep copy the model
            if phase == 'val':
                joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr = evaluate_metrics(predictions, "pred_bs_ptr", slots_dict['dev'])
                evaluation_metrics = {"Joint Acc": joint_acc_score_ptr, "Turn Acc": turn_acc_score_ptr, "Joint F1": F1_score_ptr}
                print(evaluation_metrics)
                if F1_score_ptr > best_acc:
                    best_acc = F1_score_ptr
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    iprint('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    iprint('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def evaluate_metrics(all_prediction, from_which, slot_temp):
    total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
    for d, v in all_prediction.items():
        for t in range(len(v)):
            cv = v[t]
            if set(cv["turn_belief"]) == set(cv[from_which]):
                joint_acc += 1
            total += 1

            # Compute prediction slot accuracy
            temp_acc = self.compute_acc(set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
            turn_acc += temp_acc

            # Compute prediction joint F1 score
            temp_f1, temp_r, temp_p, count = self.compute_prf(set(cv["turn_belief"]), set(cv[from_which]))
            F1_pred += temp_f1
            F1_count += count

    joint_acc_score = joint_acc / float(total) if total != 0 else 0
    turn_acc_score = turn_acc / float(total) if total != 0 else 0
    F1_score = F1_pred / float(F1_count) if F1_count != 0 else 0
    return joint_acc_score, F1_score, turn_acc_score


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
    model.to(device=device)

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
        num_epochs=args['epoch']
    )

    # for data in train_dataloader:
    #     print(model(data=data, slots_type='train'))
    #     # print(data)
    #     exit(1)

    # TODO: make sure all tensors are properly set using .to(device) or .cuda()


