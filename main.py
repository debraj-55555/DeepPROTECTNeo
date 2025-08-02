#!/usr/bin/env python3

import sys

import os
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve, auc, average_precision_score
)
from sklearn.model_selection import GroupShuffleSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader.dataset import DeepProtectNeoDataset, DeepProtectNeoDatasetTest
from dataloader.tokenizer import TCRBertTokenizer
from scripts.main_utils import EarlyStopping, LabelSmoothingCrossEntropy, plot_radar_chart, plot_auprc_auroc, compute_auprc_auroc
from models.layer import DeepProtectNeo

# ---------------------------------------
# Training and Evaluation
# ---------------------------------------
def train_and_evaluate(model, train_loader, val_loader, optimizer, loss_fn, device,
                       epochs, save_path, use_early_stopping=True):
    best_val_acc = 0.0
    best_epoch = 0
    stopper = EarlyStopping(patience=3) if use_early_stopping else None

    for epoch in range(1, epochs + 1):
        # Training loop
        model.train()
        total_train_loss = 0.0
        all_preds, all_labels = [], []
        for batch in tqdm(train_loader):
            tcr_tokens, pep_tokens, tcr_blos, pep_blos, \
            tcr_phys, pep_phys, tcr_hand, pep_hand, labels = batch
            # Move to device
            tcr_tokens, pep_tokens = tcr_tokens.to(device), pep_tokens.to(device)
            tcr_blos, pep_blos = tcr_blos.to(device), pep_blos.to(device)
            tcr_phys, pep_phys = tcr_phys.to(device), pep_phys.to(device)
            tcr_hand, pep_hand = tcr_hand.to(device), pep_hand.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(tcr_tokens, pep_tokens, tcr_blos, pep_blos,
                            tcr_phys, pep_phys, tcr_hand, pep_hand,
                            return_attention=False)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        val_preds, val_labels, val_probs = [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                tcr_tokens, pep_tokens, tcr_blos, pep_blos, \
                tcr_phys, pep_phys, tcr_hand, pep_hand, labels = batch
                tcr_tokens, pep_tokens = tcr_tokens.to(device), pep_tokens.to(device)
                tcr_blos, pep_blos = tcr_blos.to(device), pep_blos.to(device)
                tcr_phys, pep_phys = tcr_phys.to(device), pep_phys.to(device)
                tcr_hand, pep_hand = tcr_hand.to(device), pep_hand.to(device)
                labels = labels.to(device)

                outputs = model(tcr_tokens, pep_tokens, tcr_blos, pep_blos,
                                tcr_phys, pep_phys, tcr_hand, pep_hand,
                                return_attention=False)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                val_probs.extend(probs)

        # Compute metrics
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        prec = precision_score(val_labels, val_preds, average='macro', zero_division=0)
        rec = recall_score(val_labels, val_preds, average='macro')
        fpr, tpr, prec_vals, rec_vals, auroc, auprc = compute_auprc_auroc(
            np.array(val_labels), np.array(val_probs)
        )

        # Logging and plots
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | "
              f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
        plot_auprc_auroc(fpr, tpr, prec_vals, rec_vals, auroc, auprc, save_path)
        metrics = {'accuracy': val_acc, 'f1': val_f1, 'precision': prec, 'recall': rec, 'auc': auroc, 'auprc': auprc}
        plot_radar_chart(metrics, epoch, save_path)

        # Check for new best and early stop
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))

        if stopper is not None:
            stopper(avg_val_loss, model, os.path.join(save_path, 'early_stop_model.pth'))
            if stopper.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

    return best_val_acc, best_epoch

# ---------------------------------------
# Test Logic
# ---------------------------------------
def run_test():
    # load test data
    df = pd.read_csv(args.test_file)
    tokenizer = TCRBertTokenizer()
    dataset = DeepProtectNeoDatasetTest(
        df[['TCR','epitope']], df['Label'], align=True, tokenizer=tokenizer
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # instantiate and load model
    model = DeepProtectNeo(
        d_model=params['d_model'], n_heads=params['n_heads'],
        skip_dim=params.get('skip_dim',0), phys_dim=params.get('phys_dim',16),
        cross_attn_heads=params.get('cross_attn_heads',4),
        sa_n_heads=params.get('sa_n_heads',4), use_ablation=params.get('use_ablation',False),
        use_cross_attn=params.get('use_cross_attn',False), learned_pos=params.get('learned_pos',True),
        max_tcr_len=dataset.max_tcr_len, max_pep_len=dataset.max_pep_len,
        cnn_dim=params.get('cnn_dim',64), cnn_blocks=params.get('cnn_blocks',4),
        cnn_ch=params.get('cnn_ch',64), cnn_ks=params.get('cnn_ks',5),
        cnn_attn_heads=params.get('cnn_attn_heads',4)
    ).to(device)
    state = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state)
    model.eval()

    records, all_preds, all_labels, all_probs = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            tcrs, peps, tcr_toks, pep_toks, tcr_blos, pep_blos, tcr_phys, pep_phys, tcr_hand, pep_hand, labels = batch
            inputs = [
                tcr_toks.to(device), pep_toks.to(device),
                tcr_blos.to(device), pep_blos.to(device),
                tcr_phys.to(device), pep_phys.to(device),
                tcr_hand.to(device), pep_hand.to(device)
            ]
            labels_np = labels.cpu().numpy()
            logits = model(*inputs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_probs.extend(probs)

            for tcr, pep, prob, pred, label in zip(tcrs, peps, probs, preds, labels_np):
                records.append({
                    'TCR': tcr, 'Epitope': pep,
                    'Binding_Score': float(prob[1]),
                    'Predicted_Label': int(pred), 'label': int(label)
                })

    # save predictions
    pd.DataFrame(records).to_csv(args.predictions_file, index=False)

    # compute and save metrics
    all_probs_arr = np.array(all_probs)
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro'),
        'auroc': roc_auc_score(all_labels, all_probs_arr[:,1]),
        'auprc': average_precision_score(all_labels, all_probs_arr[:,1]),
        
    }
    with open(args.metric_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Test completed. Metrics:")
    print(json.dumps(metrics, indent=2))

# ---------------------------------------
# Main
# ---------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train/Test DeepProtectNeo Model")
    # common args
    parser.add_argument('--train_file', type=str, help='CSV file in data/ (for train).')
    parser.add_argument('--val_file', type=str, default=None, help='optional val CSV (TCR,epitope,Label)')
    parser.add_argument('--mode', choices=['train','test'], default='train', help='train or test')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 or cpu')
    parser.add_argument('--epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--early_stopping', type=lambda x: x.lower() in ('true','1'), default=True, help='enable early stopping')
    parser.add_argument('--run_dir', type=str, default='./run', help='base run directory')
    parser.add_argument('--save_dir', type=str, default='results', help='save subdirectory')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for testing')
    # test-specific args
    parser.add_argument('--predictions_file', type=str, default='predictions.csv', help='where to save predictions')
    parser.add_argument('--model_path', type=str, help='path to best_model.pth')
    parser.add_argument('--test_file', type=str, help='CSV with columns TCR,epitope,Label')
    parser.add_argument('--metric_file', type=str, default='metrics.json', help='where to save metrics')

    args = parser.parse_args()

    # prepare directories
    os.makedirs(args.run_dir, exist_ok=True)
    save_path = os.path.join(args.run_dir, args.save_dir)
    os.makedirs(save_path, exist_ok=True)

    # seeds and device
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # load hyperparameters
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, 'params.json'), 'r') as fp:
        params = json.load(fp)

    if args.mode == 'test':
        run_test()
        exit(0)

    # TRAINING
    df_train = pd.read_csv(os.path.join('data', args.train_file))
    if args.val_file:
        df_val = pd.read_csv(os.path.join('data', args.val_file))
    else:
        X = df_train[['TCR','epitope']]
        y = df_train['Label']
        splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=params.get('modelseed',42))
        train_idx, val_idx = next(splitter.split(X, y, groups=X['TCR']))
        df_val = df_train.iloc[val_idx].copy()
        df_train = df_train.iloc[train_idx].copy()

    y_train = df_train['Label']
    y_val = df_val['Label']

    tokenizer = TCRBertTokenizer()
    dataset_train = DeepProtectNeoDataset(df_train[['TCR','epitope']], y_train, align=True, tokenizer=tokenizer)
    dataset_val = DeepProtectNeoDataset(df_val[['TCR','epitope']], y_val, align=True, tokenizer=tokenizer)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=2)


    model = DeepProtectNeo(
        d_model=params['d_model'], n_heads=params['n_heads'], skip_dim=params.get('skip_dim',0),
        phys_dim=params.get('phys_dim',16), cross_attn_heads=params.get('cross_attn_heads',4),
        sa_n_heads=params.get('sa_n_heads',4), use_ablation=params.get('use_ablation',False),
        use_cross_attn=params.get('use_cross_attn',False), learned_pos=params.get('learned_pos',True),
        max_tcr_len=dataset_train.max_tcr_len, max_pep_len=dataset_train.max_pep_len,
        cnn_dim=params.get('cnn_dim',64), cnn_blocks=params.get('cnn_blocks',4),
        cnn_ch=params.get('cnn_ch',64), cnn_ks=params.get('cnn_ks',5),
        cnn_attn_heads=params.get('cnn_attn_heads',4)
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=params.get('learning_rate',1e-3))
    loss_fn = LabelSmoothingCrossEntropy(smoothing=params.get('smoothing',0.1)).to(device)

    best_acc, best_epoch = train_and_evaluate(
        model, train_loader, val_loader, optimizer, loss_fn, device,
        args.epochs, save_path, use_early_stopping=args.early_stopping
    )

    print(f"Training completed. Best val accuracy {best_acc:.4f} at epoch {best_epoch}.")
