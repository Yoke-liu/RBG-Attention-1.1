import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

sys.stdout.reconfigure(encoding='utf-8')

class Config:
    # 数据参数
    num_stations    = 24
    spatial_features = 1
    temporal_features = 10
    seq_len         = 24
    pred_horizon    = 1

    num_outputs     = 26

    spatial_hidden  = 64
    temporal_hidden = 128
    fusion_dim      = 128
    num_heads       = 8

    batch_size      = 128
    epochs          = 100
    lr              = 1e-4
    weight_decay    = 1e-5
    patience        = 30
    warmup_epochs   = 10
    grad_clip       = 1.0

    k_folds         = 5
    test_ratio      = 0.2

    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed            = 42

    data_path        = r'E:\1_LIAO\software\pycharm\workspace\1duo\data\55082A00.csv'
    model_save_dir   = './checkpoints/'
    prediction_dir   = './predictions/'

    station_names = [
        'XIGONG','DONGSI','YUNGANG','XIAOTUN','NONGZHAN','GUCHENG',
        'JIUGONG','TIANTAN','AOTI','GUANYUAN','DINGLING','MIYUNXINCHENG',
        'MIYUNZHEN','PINGGUXINCHENG','YANQINGXIADU','YANQINGSHIHEYING',
        'HUAIROUXINCHENG','HUAIROUZHEN','FANGSHAN','CHANGPING',
        'HAIDIANWANLIU','TONGZHOUDONGGUAN','MEMTOUGOU','SHUNYIXINCHENG'
    ]


class AirQualityDataset(Dataset):
    def __init__(self, X_spatial, X_temporal, y):
        self.X_spatial  = torch.FloatTensor(X_spatial)
        self.X_temporal = torch.FloatTensor(X_temporal)
        self.y          = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_spatial[idx], self.X_temporal[idx], self.y[idx]


def handle_missing_values(df):
    df = df.interpolate(method='linear', limit=3, limit_direction='both')
    df = df.fillna(df.mean())
    return df


def build_sequences(feat, targets, seq_len, pred_horizon):
    X, y = [], []
    for i in range(len(feat) - seq_len - pred_horizon + 1):
        X.append(feat[i : i + seq_len])
        y.append(targets[i + seq_len + pred_horizon - 1])
    return np.array(X), np.array(y)


def load_and_preprocess_data(config):
    print(f"Loading data: {config.data_path}")
    df = pd.read_csv(config.data_path, encoding='gbk')
    print(f"Raw shape: {df.shape}  columns: {list(df.columns)}")

    temporal_cols   = df.columns[:10].tolist()
    target_pm25_col = df.columns[10]
    spatial_cols    = df.columns[11:35].tolist()

    df = handle_missing_values(df)

    temporal_data  = df[temporal_cols].values.astype(np.float32)
    spatial_raw    = df[spatial_cols].values.astype(np.float32)
    pm25_region    = df[target_pm25_col].values.astype(np.float32)
    o3_region      = df[temporal_cols[3]].values.astype(np.float32)

    T = len(df)

    targets = np.concatenate([
        spatial_raw,
        pm25_region[:, np.newaxis],
        o3_region[:, np.newaxis]
    ], axis=1).astype(np.float32)

    spatial_data = spatial_raw[:, :, np.newaxis]

    sp_flat   = spatial_data.reshape(-1, 1)
    sp_scaler = MinMaxScaler()
    spatial_norm = sp_scaler.fit_transform(sp_flat).reshape(T, config.num_stations, 1)

    t_scaler     = MinMaxScaler()
    temporal_norm = t_scaler.fit_transform(temporal_data)

    y_scaler  = MinMaxScaler()
    targets_norm = y_scaler.fit_transform(targets)

    X_spatial,  y  = build_sequences(spatial_norm,  targets_norm,
                                     config.seq_len, config.pred_horizon)
    X_temporal, _  = build_sequences(temporal_norm, targets_norm,
                                     config.seq_len, config.pred_horizon)

    print(f"X_spatial  : {X_spatial.shape}")
    print(f"X_temporal : {X_temporal.shape}")
    print(f"y          : {y.shape}  (24 stations + PM2.5_region + O3)")

    return X_spatial, X_temporal, y, (sp_scaler, t_scaler, y_scaler)


class ChannelAwareResNet(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        red = max(1, out_channels // reduction)
        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.fc1     = nn.Linear(out_channels, red)
        self.fc2     = nn.Linear(red, out_channels)
        self.sigmoid = nn.Sigmoid()
        self.shortcut = (nn.Identity() if in_channels == out_channels
                         else nn.Conv1d(in_channels, out_channels, 1))

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        se  = self.gap(out).squeeze(-1)
        se  = self.sigmoid(self.fc2(self.relu(self.fc1(se)))).unsqueeze(-1)
        return self.relu(out * se + identity)


class SpatialFeatureExtractor(nn.Module):
    def __init__(self, num_stations, input_dim, hidden_dim=64):
        super().__init__()
        in_ch = num_stations * input_dim
        self.resnet1 = ChannelAwareResNet(in_ch,      hidden_dim)
        self.resnet2 = ChannelAwareResNet(hidden_dim,  hidden_dim)
        self.resnet3 = ChannelAwareResNet(hidden_dim,  hidden_dim)
        self.gap     = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        b, l, s, f = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, s * f, l)
        return self.gap(self.resnet3(self.resnet2(self.resnet1(x)))).squeeze(-1)


class BiGRUModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.bigru = nn.GRU(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True)
        self.proj  = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        out, _ = self.bigru(x)
        return self.proj(out[:, -1, :])


class DynamicWeightedFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc      = nn.Linear(feature_dim * 2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, s, t):
        w = self.softmax(self.fc(torch.cat([s, t], dim=1)))
        return w[:, 0:1] * s + w[:, 1:2] * t


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(embed_dim, num_heads,
                                             dropout=dropout, batch_first=True)
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        o, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(o))


class RBGAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spatial_extractor  = SpatialFeatureExtractor(
            config.num_stations, config.spatial_features, config.spatial_hidden)
        self.temporal_extractor = BiGRUModule(
            config.temporal_features, config.temporal_hidden)
        self.spatial_proj  = nn.Linear(config.spatial_hidden,  config.fusion_dim)
        self.temporal_proj = nn.Linear(config.temporal_hidden, config.fusion_dim)
        self.dwf       = DynamicWeightedFusion(config.fusion_dim)
        self.attention = MultiHeadAttentionModule(config.fusion_dim, config.num_heads)
        self.predictor = nn.Sequential(
            nn.Linear(config.fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config.num_outputs)   # 26
        )

    def forward(self, spatial_input, temporal_input):
        s = self.spatial_proj(self.spatial_extractor(spatial_input))
        t = self.temporal_proj(self.temporal_extractor(temporal_input))
        f = self.dwf(s, t).unsqueeze(1)
        o = self.attention(f).squeeze(1)
        return self.predictor(o)


class MLP(nn.Module):
    def __init__(self, input_dim, num_outputs, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, num_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, num_outputs, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, num_outputs)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_dim, num_outputs, hidden_dim=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, num_outputs)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device,
                grad_clip=1.0, is_rbg=True):
    model.train()
    total = 0.0
    for batch in loader:
        if is_rbg:
            Xs, Xt, y = [b.to(device) for b in batch]
            pred = model(Xs, Xt)
        else:
            X, y = [b.to(device) for b in batch]
            pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


def evaluate(model, loader, device, is_rbg=True, station_names=None, num_stations=24):

    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for batch in loader:
            if is_rbg:
                Xs, Xt, y = [b.to(device) for b in batch]
                pred = model(Xs, Xt)
            else:
                X, y = [b.to(device) for b in batch]
                pred = model(X)
            preds_list.append(pred.cpu().numpy())
            targets_list.append(y.cpu().numpy())

    preds   = np.vstack(preds_list)    # (N, 26)
    targets = np.vstack(targets_list)  # (N, 26)

    def met(t, p):
        mae  = mean_absolute_error(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        r2   = r2_score(t, p)
        return mae, rmse, r2

    station_metrics = {}
    names = station_names if station_names else [f'S{i}' for i in range(num_stations)]
    for i, name in enumerate(names):
        mae, rmse, r2 = met(targets[:, i], preds[:, i])
        station_metrics[name] = {'mae': mae, 'rmse': rmse, 'r2': r2}

    mae_pm, rmse_pm, r2_pm = met(targets[:, 24], preds[:, 24])
    mae_o3, rmse_o3, r2_o3 = met(targets[:, 25], preds[:, 25])

    avg_mae  = np.mean([v['mae']  for v in station_metrics.values()])
    avg_rmse = np.mean([v['rmse'] for v in station_metrics.values()])
    avg_r2   = np.mean([v['r2']   for v in station_metrics.values()])

    val_mae = avg_mae

    val_loss = np.mean((preds - targets) ** 2)

    return {
        'val_loss':        val_loss,
        'val_mae':         val_mae,
        'station_metrics': station_metrics,
        'mae_pm25':        mae_pm,  'rmse_pm25': rmse_pm, 'r2_pm25': r2_pm,
        'mae_o3':          mae_o3,  'rmse_o3':   rmse_o3, 'r2_o3':   r2_o3,
        'avg_mae':         avg_mae, 'avg_rmse':  avg_rmse,'avg_r2':  avg_r2,
        'preds':  preds,  'targets': targets
    }


def train_model(model, train_loader, val_loader, config, is_rbg=True, station_names=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 betas=(0.9, 0.999), weight_decay=config.weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs - config.warmup_epochs), eta_min=1e-7)

    best_val_mae = float('inf')
    best_state   = None
    patience_cnt = 0
    history      = {'train_loss': [], 'val_mae': [], 'lr': []}

    for epoch in range(config.epochs):
        if epoch < config.warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = config.lr * (epoch + 1) / config.warmup_epochs

        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 config.device, config.grad_clip, is_rbg)
        val_m      = evaluate(model, val_loader, config.device, is_rbg,
                              station_names, config.num_stations)
        val_mae    = val_m['val_mae']

        if epoch >= config.warmup_epochs:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_mae'].append(val_mae)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch+1:3d}/{config.epochs} | "
              f"Train MSE: {train_loss:.4f} | "
              f"Val MAE(24stn avg): {val_mae:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state   = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= config.patience:
                print(f"Early stopping at epoch {epoch+1}  (best Val MAE={best_val_mae:.4f})")
                break

    model.load_state_dict(best_state)
    return model, history


class BaselineDataset(Dataset):
    def __init__(self, X_spatial, X_temporal, y, model_type='lstm'):
        self.y = torch.FloatTensor(y)
        N, L, S, Fs = X_spatial.shape
        _, _, Ft     = X_temporal.shape
        if model_type == 'mlp':
            self.X = torch.FloatTensor(
                np.concatenate([X_spatial.reshape(N, -1),
                                X_temporal.reshape(N, -1)], axis=1))
        else:
            self.X = torch.FloatTensor(
                np.concatenate([X_spatial.reshape(N, L, -1), X_temporal], axis=2))
        self.model_type = model_type

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def print_station_report(metrics, station_names, model_name, fold=None):
    tag = f"  [Fold {fold}]" if fold is not None else ""
    print(f"\n{'─'*72}")
    print(f"  {model_name}{tag}  —  Per-Station PM2.5 Metrics (normalized space)")
    print(f"{'─'*72}")
    print(f"  {'Station':<22} {'MAE':>8}  {'RMSE':>8}  {'R2':>8}")
    print(f"  {'─'*22} {'─'*8}  {'─'*8}  {'─'*8}")
    for name, m in metrics['station_metrics'].items():
        print(f"  {name:<22} {m['mae']:8.4f}  {m['rmse']:8.4f}  {m['r2']:8.4f}")
    print(f"  {'─'*22} {'─'*8}  {'─'*8}  {'─'*8}")
    print(f"  {'24-Station Average':<22} "
          f"{metrics['avg_mae']:8.4f}  {metrics['avg_rmse']:8.4f}  {metrics['avg_r2']:8.4f}")
    print(f"  {'Region PM2.5':<22} "
          f"{metrics['mae_pm25']:8.4f}  {metrics['rmse_pm25']:8.4f}  {metrics['r2_pm25']:8.4f}")
    print(f"  {'O3':<22} "
          f"{metrics['mae_o3']:8.4f}  {metrics['rmse_o3']:8.4f}  {metrics['r2_o3']:8.4f}")
    print(f"{'─'*72}")


def save_predictions(preds_norm, targets_norm, y_scaler, station_names,
                     model_name, fold, save_dir, print_n=24):

    preds_orig   = y_scaler.inverse_transform(preds_norm)    # (N, 26)
    targets_orig = y_scaler.inverse_transform(targets_norm)  # (N, 26)

    N = len(preds_orig)

    cols = {}
    cols['timestep'] = np.arange(1, N + 1)

    for i, name in enumerate(station_names):
        cols[f'{name}_pred'] = preds_orig[:, i].round(2)
        cols[f'{name}_true'] = targets_orig[:, i].round(2)

    cols['PM2.5_region_pred'] = preds_orig[:, 24].round(2)
    cols['PM2.5_region_true'] = targets_orig[:, 24].round(2)

    cols['O3_pred'] = preds_orig[:, 25].round(2)
    cols['O3_true'] = targets_orig[:, 25].round(2)

    df_out = pd.DataFrame(cols)

    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f'{model_name}_fold{fold}_predictions.csv')
    df_out.to_csv(fname, index=False, encoding='utf-8-sig')
    print(f"\n  [Saved] Predictions -> {fname}  ({N} rows x {len(df_out.columns)} cols)")

    preview_stations = station_names[:4]
    preview_df = df_out.head(print_n)
    print(f"\n  Preview (first {print_n} timesteps, showing 4 stations + PM2.5_region + O3):")
    header = f"  {'Step':>5}"
    for s in preview_stations:
        header += f"  {(s[:8]+' pred'):>14}  {(s[:8]+' true'):>14}"
    header += f"  {'Reg_pred':>10}  {'Reg_true':>10}  {'O3_pred':>9}  {'O3_true':>9}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for _, row in preview_df.iterrows():
        line = f"  {int(row['timestep']):>5}"
        for s in preview_stations:
            line += f"  {row[s+'_pred']:>14.2f}  {row[s+'_true']:>14.2f}"
        line += (f"  {row['PM2.5_region_pred']:>10.2f}  {row['PM2.5_region_true']:>10.2f}"
                 f"  {row['O3_pred']:>9.2f}  {row['O3_true']:>9.2f}")
        print(line)

    return df_out


def run_experiment(config, model_name='RBG-Attention'):
    set_seed(config.seed)
    print(f"\n{'='*20} Running {model_name} {'='*20}")

    X_spatial, X_temporal, y, scalers = load_and_preprocess_data(config)
    sp_scaler, t_scaler, y_scaler = scalers

    if model_name == 'RBG-Attention':
        dataset = AirQualityDataset(X_spatial, X_temporal, y)
        is_rbg  = True
    elif model_name == 'MLP':
        dataset = BaselineDataset(X_spatial, X_temporal, y, model_type='mlp')
        is_rbg  = False
    elif model_name in ['LSTM', 'GRU']:
        dataset = BaselineDataset(X_spatial, X_temporal, y, model_type='lstm')
        is_rbg  = False
    else:
        raise ValueError(f"Unknown model: {model_name}")

    test_size      = int(config.test_ratio * len(dataset))
    train_val_size = len(dataset) - test_size
    train_val_ds, test_ds = random_split(
        dataset, [train_val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed))

    test_loader = DataLoader(test_ds, batch_size=config.batch_size)

    kfold        = KFold(n_splits=config.k_folds, shuffle=True,
                         random_state=config.seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(train_val_size))):
        print(f"\n----- Fold {fold+1}/{config.k_folds} -----")

        train_loader = DataLoader(Subset(train_val_ds, train_idx),
                                  batch_size=config.batch_size, shuffle=True)
        val_loader   = DataLoader(Subset(train_val_ds, val_idx),
                                  batch_size=config.batch_size)

        n_out = config.num_outputs
        if model_name == 'RBG-Attention':
            model = RBGAttention(config).to(config.device)
        elif model_name == 'MLP':
            model = MLP(dataset.X.shape[1], n_out).to(config.device)
        elif model_name == 'LSTM':
            model = LSTMModel(dataset.X.shape[2], n_out).to(config.device)
        elif model_name == 'GRU':
            model = GRUModel(dataset.X.shape[2], n_out).to(config.device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")

        model, history = train_model(
            model, train_loader, val_loader, config, is_rbg,
            station_names=config.station_names)

        test_m = evaluate(model, test_loader, config.device, is_rbg,
                          config.station_names, config.num_stations)

        print_station_report(test_m, config.station_names, model_name, fold=fold+1)

        save_predictions(
            preds_norm   = test_m['preds'],
            targets_norm = test_m['targets'],
            y_scaler     = y_scaler,
            station_names= config.station_names,
            model_name   = model_name,
            fold         = fold + 1,
            save_dir     = config.prediction_dir,
            print_n      = 24
        )

        fold_results.append(test_m)

        os.makedirs(config.model_save_dir, exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(config.model_save_dir,
                                f"{model_name}_fold{fold+1}.pth"))

    def cv_mean_std(key):
        vals = [r[key] for r in fold_results]
        return np.mean(vals), np.std(vals)

    print(f"\n{'='*60}")
    print(f"  {model_name}  Cross-Validation Summary (5-fold)")
    print(f"{'='*60}")

    print(f"\n  Per-Station MAE (mean +/- std over folds):")
    print(f"  {'Station':<22} {'MAE mean':>10}  {'MAE std':>9}")
    for name in config.station_names:
        maes = [r['station_metrics'][name]['mae'] for r in fold_results]
        print(f"  {name:<22} {np.mean(maes):10.4f}  {np.std(maes):9.4f}")

    for key_label, mk, rk, r2k in [
        ('24-Stn Avg PM2.5', 'avg_mae', 'avg_rmse', 'avg_r2'),
        ('Region PM2.5',     'mae_pm25','rmse_pm25','r2_pm25'),
        ('O3',               'mae_o3',  'rmse_o3',  'r2_o3'),
    ]:
        m_mae, s_mae   = cv_mean_std(mk)
        m_rmse, s_rmse = cv_mean_std(rk)
        m_r2, s_r2     = cv_mean_std(r2k)
        print(f"\n  [{key_label}]")
        print(f"    MAE : {m_mae:.4f} +/- {s_mae:.4f}")
        print(f"    RMSE: {m_rmse:.4f} +/- {s_rmse:.4f}")
        print(f"    R2  : {m_r2:.4f} +/- {s_r2:.4f}")

    return fold_results


def compare_all_models(config):
    models      = ['RBG-Attention', 'MLP', 'LSTM', 'GRU']
    all_results = {}
    for m in models:
        all_results[m] = run_experiment(config, m)

    print("\n" + "="*90)
    print("  Final Model Comparison  (Mean +/- Std over 5 folds,  normalized space)")
    print("="*90)
    print(f"  {'Model':<16} "
          f"{'Stn-Avg MAE':>14}  {'Stn-Avg RMSE':>14}  {'Stn-Avg R2':>12}  "
          f"{'PM2.5 MAE':>12}  {'O3 MAE':>10}")
    print(f"  {'-'*16} {'-'*14}  {'-'*14}  {'-'*12}  {'-'*12}  {'-'*10}")
    for m in models:
        res = all_results[m]
        def ms(k): return np.mean([r[k] for r in res]), np.std([r[k] for r in res])
        am, as_ = ms('avg_mae');   ar, ars = ms('avg_rmse'); a2, a2s = ms('avg_r2')
        pm, ps  = ms('mae_pm25'); om, os_ = ms('mae_o3')
        print(f"  {m:<16} "
              f"{am:.4f}+/-{as_:.4f}  "
              f"{ar:.4f}+/-{ars:.4f}  "
              f"{a2:.4f}+/-{a2s:.4f}  "
              f"{pm:.4f}+/-{ps:.4f}  "
              f"{om:.4f}+/-{os_:.4f}")
    print("="*90)


if __name__ == "__main__":
    config = Config()

    compare_all_models(config)