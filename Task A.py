# dynamic_conv_improved.py

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score


class DynamicCNN(nn.Module):
    def __init__(self, max_in_channels=3, num_classes=100, base_channels=64, n_layers=3, 
                 channels_multiplier=1.5, expansion=4, use_se=True, dropout_rate=0.3):
        super().__init__()
        self.max_in_channels = max_in_channels
        self.num_classes = num_classes  # 新增：儲存 num_classes 作為類屬性
        
        # Calculate channels for each layer and store as attribute
        self.channels = [int(base_channels * (channels_multiplier ** i)) for i in range(n_layers + 1)]  # 修改：儲存為 self.channels
        
        # Initial convolution
        self.initial_conv = ImprovedDynamicConvModule(
            max_in_channels, self.channels[0], kernel_size=3, use_se=use_se
        )
        self.bn1 = nn.BatchNorm2d(self.channels[0])
        
        # Residual layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            stride = 2 if i > 0 else 1
            self.layers.append(ImprovedDynamicResidualBlock(
                self.channels[i], self.channels[i+1], stride=stride, expansion=expansion, use_se=use_se
            ))
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.channels[-1], self.num_classes)  # 修改：使用 self.channels 和 self.num_classes
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.initial_conv(x)))
        for layer in self.layers:
            out = layer(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def get_flops(self, input_shape):
        total_flops = 0
        B, C, H, W = input_shape
        
        total_flops += self.initial_conv.get_flops(input_shape)
        
        current_shape = (B, self.initial_conv.out_channels, H, W)
        for layer in self.layers:
            conv1_shape = current_shape
            conv1_out_channels = layer.dynamic_conv1.out_channels
            conv2_out_channels = layer.dynamic_conv2.out_channels
            conv3_out_channels = layer.dynamic_conv3.out_channels
            stride = layer.dynamic_conv2.stride
            
            total_flops += layer.dynamic_conv1.get_flops(conv1_shape)
            conv2_shape = (B, conv1_out_channels, H // stride, W // stride)
            total_flops += layer.dynamic_conv2.get_flops(conv2_shape)
            total_flops += layer.dynamic_conv3.get_flops(conv2_shape)
            
            # 添加 shortcut FLOPs
            if layer.shortcut:
                max_in_channels = current_shape[1]
                shortcut_flops = max_in_channels * conv3_out_channels * (H // stride) * (W // stride)
                total_flops += shortcut_flops
            
            current_shape = (B, conv3_out_channels, H // stride, W // stride)
            H, W = H // stride, W // stride
        
        total_flops += self.channels[-1] * self.num_classes * 2 + self.num_classes  # 修改：考慮乘加和偏置
        
        return total_flops

    def get_params_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class ImprovedDynamicConvModule(nn.Module):
    def __init__(self, max_in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, use_se=True):
        super().__init__()
        self.max_in_channels = max_in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_se = use_se
        
        # Channel encoding network
        self.channel_encoder = nn.Sequential(
            nn.Linear(max_in_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # Weight generation network
        self.weight_generator = nn.Linear(
            256, 
            out_channels * max_in_channels * kernel_size * kernel_size
        )
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # SE attention module
        if self.use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, max(1, out_channels // 16), 1),
                nn.ReLU(),
                nn.Conv2d(max(1, out_channels // 16), out_channels, 1),
                nn.Sigmoid()
            )
        
        # Initialize weights
        nn.init.kaiming_normal_(self.weight_generator.weight)

    def forward(self, x):
        B, C, H, W = x.shape
        
        input_vector = torch.zeros(self.max_in_channels).to(x.device)
        input_vector[:C] = 1.0
        
        channel_encoding = self.channel_encoder(input_vector)
        weight = self.weight_generator(channel_encoding)
        weight = weight.view(self.out_channels, self.max_in_channels, self.kernel_size, self.kernel_size)
        weight = weight[:, :C, :, :]
        
        out = F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding)
        
        if self.use_se:
            se_weight = self.se(out)
            out = out * se_weight
        
        return out

    def get_flops(self, input_shape):
        B, C, H, W = input_shape
        encoder_flops = C * 128 + 128 * 256
        generator_flops = 256 * (self.out_channels * self.max_in_channels * self.kernel_size * self.kernel_size)
        conv_flops = 2 * C * self.kernel_size * self.kernel_size * H * W * self.out_channels
        total_flops = encoder_flops + generator_flops + conv_flops
        
        if self.use_se:
            se_flops = H * W * self.out_channels  # 全局平均池化
            se_flops += self.out_channels * (self.out_channels // 16)  # 第一層卷積
            se_flops += (self.out_channels // 16) * self.out_channels  # 第二層卷積
            total_flops += se_flops
        
        return total_flops

    def get_params_count(self):
        # （保持原有的參數計數邏輯）
        encoder_params = (self.max_in_channels * 128) + 128 + (128 * 256) + 256
        generator_params = 256 * (self.out_channels * self.max_in_channels * self.kernel_size * self.kernel_size) + \
                           (self.out_channels * self.max_in_channels * self.kernel_size * self.kernel_size)
        bias_params = self.out_channels
        return encoder_params + generator_params + bias_params


class ImprovedDynamicResidualBlock(nn.Module):
    def __init__(self, max_in_channels, out_channels, stride=1, expansion=4, use_se=True):
        super().__init__()
        bottleneck_channels = out_channels // expansion
        
        self.dynamic_conv1 = ImprovedDynamicConvModule(max_in_channels, bottleneck_channels, kernel_size=1, padding=0, use_se=use_se)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.dynamic_conv2 = ImprovedDynamicConvModule(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, use_se=use_se)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.dynamic_conv3 = ImprovedDynamicConvModule(bottleneck_channels, out_channels, kernel_size=1, padding=0, use_se=use_se)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or max_in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(max_in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.dynamic_conv1(x)))
        out = F.relu(self.bn2(self.dynamic_conv2(out)))
        out = self.bn3(self.dynamic_conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



# 3. 新增：標籤平滑交叉熵損失
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class EarlyStopping:
    """
    早停機制，用於監控驗證損失或準確率
    """
    def __init__(self, patience=10, min_delta=0.001, mode='min', verbose=True):
        """
        Args:
            patience (int): 容忍的epoch數量
            min_delta (float): 最小改進量
            mode (str): 'min' 或 'max'，表示監控指標的方向
            verbose (bool): 是否顯示訊息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, current_value, epoch):
        """
        檢查是否應該早停
        
        Args:
            current_value: 當前的監控值（例如驗證損失或準確率）
            epoch: 當前的 epoch 數
            
        Returns:
            bool: 是否應該停止訓練
        """
        if self.mode == 'min':
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.counter = 0
                self.best_epoch = epoch
                if self.verbose:
                    print(f"Early stopping: improved to {current_value:.4f}")
            else:
                self.counter += 1
                if self.verbose:
                    print(f"Early stopping: no improvement for {self.counter} epochs")
        else:  # mode == 'max'
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.counter = 0
                self.best_epoch = epoch
                if self.verbose:
                    print(f"Early stopping: improved to {current_value:.4f}")
            else:
                self.counter += 1
                if self.verbose:
                    print(f"Early stopping: no improvement for {self.counter} epochs")
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"Early stopping triggered! Best value: {self.best_value:.4f} at epoch {self.best_epoch}")
        
        return self.early_stop

# 4. MiniImageNetDataset with channel selection
class MiniImageNetDataset(Dataset):
    """
    Dataset for Mini-ImageNet with support for different channel combinations
    """
    def __init__(self, txt_file, img_dir="", channels="RGB", transform=None):
        self.img_dir = img_dir
        self.channels = channels
        self.transform = transform
        self.samples = []
        
        # Read image paths and labels
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # 最後一個元素是標籤，前面的所有部分是路徑
                    label = int(parts[-1])
                    img_path = " ".join(parts[:-1])
                    self.samples.append((img_path, label))
        
        # Map channel names to indices
        self.channel_idx_map = {
            "R": [0], "G": [1], "B": [2],
            "RG": [0, 1], "RB": [0, 2], "GB": [1, 2], "RGB": [0, 1, 2]
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # 如果提供了img_dir，則將其與路徑結合，否則直接使用文件中的路徑
        if self.img_dir:
            img_path = os.path.join(self.img_dir, img_path)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Select only the specified channels
        channel_indices = self.channel_idx_map[self.channels]
        image = image[channel_indices, :, :]
        
        return image, label

def define_model():
    
    base_channels = 48
    n_layers = 4
    channels_multiplier = 1.5
    expansion = 4
    use_se = True
    dropout_rate = 0.2
    
    # Fixed parameters
    model = DynamicCNN(
        max_in_channels=3,
        num_classes=100,
        base_channels=base_channels,
        n_layers=n_layers,
        channels_multiplier=channels_multiplier,
        expansion=expansion,
        use_se=use_se,
        dropout_rate=dropout_rate
    )
    
    return model

def create_data_loaders(args):
    train_transform = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.RandomCrop(64),          # 增加隨機裁剪
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),      # 增加旋轉角度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),  # 增強顏色抖動
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3),    # 新增隨機擦除
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MiniImageNetDataset(
        os.path.join(args.data_dir, 'train.txt'), "", channels="RGB", transform=train_transform
    )
    val_dataset = MiniImageNetDataset(
        os.path.join(args.data_dir, 'val.txt'), "", channels="RGB", transform=test_transform
    )
    test_dataset = MiniImageNetDataset(
        os.path.join(args.data_dir, 'test.txt'), "", channels="RGB", transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, test_transform  


# 5. Training function
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)  # 修改：考慮批次大小
        total += inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            print(f'Train Epoch: {epoch} [{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {running_loss/total:.4f} '
                  f'Acc: {100.*correct/total:.2f}% ({correct}/{total})')
    
    return running_loss / total, 100. * correct / total  

def evaluate(model, loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)  # 修改：考慮批次大小
            
            _, predicted = outputs.max(1)
            total += inputs.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / total  # 修改：正確平均
    
    print(f'Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}% ({correct}/{total})')
    
    return avg_loss, accuracy



# 7. Evaluation function across all channel combinations
def evaluate_all_channels(model, txt_file, img_dir, transform, device, batch_size=64, use_se=True):
    """
    評估模型在不同通道組合下的性能
    可以指定是否使用SE模組
    """
    channel_combos = ["RGB", "RG", "RB", "GB", "R", "G", "B"]
    results = {}
    
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    # 保存原始的SE設定
    original_se_settings = {}
    if not use_se:
        # 暫時關閉所有SE模組
        for name, module in model.named_modules():
            if isinstance(module, ImprovedDynamicConvModule):
                original_se_settings[name] = module.use_se
                module.use_se = False
    
    with torch.no_grad():
        for ch_combo in channel_combos:
            print(f"\n評估通道組合: {ch_combo}" + (f" (SE模組關閉)" if not use_se else ""))
            dataset = MiniImageNetDataset(txt_file, img_dir, channels=ch_combo, transform=transform)
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
            
            test_loss = 0
            correct = 0
            total = 0
            
            for inputs, targets in tqdm(loader, desc=f"測試 {ch_combo}"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                
                _, predicted = outputs.max(1)
                total += inputs.size(0)
                correct += predicted.eq(targets).sum().item()
            
            accuracy = 100. * correct / total
            avg_loss = test_loss / total
            
            results[ch_combo] = {
                'accuracy': accuracy,
                'loss': avg_loss
            }
            
            print(f'{ch_combo} - 損失: {avg_loss:.4f}, 準確率: {accuracy:.2f}% ({correct}/{total})')
    
    # 恢復原始SE設定
    if not use_se:
        for name, use_se_val in original_se_settings.items():
            module = dict(model.named_modules())[name]
            module.use_se = use_se_val
    
    return results

def ablation_study_se(model, test_file, img_dir, transform, device, batch_size=64, output_dir='.'):
    """
    進行SE模組的消融研究，比較啟用和禁用SE模組時的性能差異
    """
    print("\n=== SE模組消融研究 ===")
    
    # 啟用SE模組的結果
    print("評估啟用SE模組的性能...")
    results_with_se = evaluate_all_channels(model, test_file, img_dir, transform, device, batch_size, use_se=True)
    
    # 禁用SE模組的結果
    print("\n評估禁用SE模組的性能...")
    results_without_se = evaluate_all_channels(model, test_file, img_dir, transform, device, batch_size, use_se=False)
    
    # 計算差異並整理結果
    comparison_results = {}
    for ch_combo in results_with_se.keys():
        acc_with_se = results_with_se[ch_combo]['accuracy']
        acc_without_se = results_without_se[ch_combo]['accuracy']
        diff = acc_with_se - acc_without_se
        
        comparison_results[ch_combo] = {
            'with_se': acc_with_se,
            'without_se': acc_without_se,
            'diff': diff
        }
    
    # 顯示結果表格
    print("\n=== SE模組消融研究結果 ===")
    print(f"{'通道組合':>8} | {'啟用SE':>8} | {'禁用SE':>8} | {'差異':>8}")
    print("-" * 40)
    for combo, metrics in comparison_results.items():
        print(f"{combo:>8} | {metrics['with_se']:>8.2f}% | {metrics['without_se']:>8.2f}% | {metrics['diff']:>+8.2f}%")
    
    # 儲存結果為CSV
    df = pd.DataFrame.from_dict(comparison_results, orient='index')
    df.to_csv(os.path.join(output_dir, 'se_ablation_study.csv'))
    
    return comparison_results

# 8. Model comparison function (Dynamic vs Traditional)
def compare_models():
    input_shape = (1, 3, 64, 64)
    
    # Dynamic model
    dynamic_model = DynamicCNN(max_in_channels=3, num_classes=100)
    dynamic_params = dynamic_model.get_params_count()
    dynamic_flops = dynamic_model.get_flops(input_shape)
    
    # Traditional model (保持原有的 TraditionalCNN 定義)
    class TraditionalCNN(nn.Module):
        def __init__(self, in_channels=3, num_classes=100):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128)
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(128, num_classes)
            
        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            identity = x
            out = self.layer1(x)
            out += identity
            out = F.relu(out)
            out = self.layer2(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out
    
    traditional_model = TraditionalCNN()
    traditional_params = sum(p.numel() for p in traditional_model.parameters() if p.requires_grad)
    
    # （保持原有的 FLOPs 計算和比較邏輯）
    traditional_flops = 2 * 3 * 3 * 3 * 64 * 64 * 64
    traditional_flops += 2 * 64 * 3 * 3 * 64 * 64 * 64 * 2
    traditional_flops += 2 * 64 * 3 * 3 * 32 * 32 * 128 + 2 * 128 * 3 * 3 * 32 * 32 * 128
    traditional_flops += 128 * 100
    
    traditional_models_required = 7
    traditional_total_params = traditional_params * traditional_models_required
    
    comparison = {
        'dynamic_model': {
            'params': dynamic_params,
            'flops': dynamic_flops,
            'models_required': 1
        },
        'traditional_model': {
            'params_per_model': traditional_params,
            'flops_per_model': traditional_flops,
            'models_required': traditional_models_required,
            'total_params': traditional_total_params
        },
        'savings': {
            'params_percent': (1 - dynamic_params / traditional_total_params) * 100,
            'models_percent': (1 - 1 / traditional_models_required) * 100
        }
    }
    
    df = pd.DataFrame(comparison)
    print(df)

    return comparison


# 9. Main training and evaluation pipeline
def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")
    
    train_loader, val_loader, test_loader, test_transform = create_data_loaders(args)
    
    model = define_model().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if isinstance(model, nn.DataParallel):
        print(f"模型參數數量: {model.module.get_params_count():,}")
    else:
        print(f"模型參數數量: {model.get_params_count():,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=args.scheduler_factor, patience=args.scheduler_patience, verbose=True
    )
    early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='max', verbose=True)
    
    if args.train:
        best_val_acc = 0
        best_epoch = 0
        train_losses, train_accs, val_losses, val_accs = [], [], [], []
        
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            scheduler.step(val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                print(f"儲存新的最佳模型，epoch {epoch}，驗證準確率: {val_acc:.2f}%")
            
            if epoch % args.save_freq == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'best_epoch': best_epoch,
                    'hyperparameters': {
                        'lr': args.lr,
                        'weight_decay': args.weight_decay,
                        'batch_size': args.batch_size,
                        'label_smoothing': args.label_smoothing,
                        'optimizer_name': args.optimizer_name,
                        'rotation_degrees': args.rotation_degrees,
                        'color_jitter': args.color_jitter,
                        'scheduler_factor': args.scheduler_factor,
                        'scheduler_patience': args.scheduler_patience
                    }
                }
                torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))
            
            if early_stopping(val_acc, epoch):
                print(f"訓練在 epoch {epoch} 提前停止")
                break
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss vs. Epoch')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Acc')
        plt.plot(range(1, len(val_accs) + 1), val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Accuracy vs. Epoch')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
        plt.close()
        
        print(f"訓練完成，最佳驗證準確率: {best_val_acc:.2f}%，在 epoch {best_epoch}")
    
    if args.train or args.eval or args.test_channels:
        if args.load_model:
            checkpoint = torch.load(args.load_model)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'hyperparameters' in checkpoint:
                    print(f"載入模型的超參數: {checkpoint['hyperparameters']}")
            else:
                model.load_state_dict(checkpoint)
            print(f"從 {args.load_model} 載入模型")
        elif args.train:
            model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
            print("載入訓練中的最佳模型")
        
        if args.eval:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            print(f"測試準確率: {test_acc:.2f}%")
        
        if args.test_channels:
            print("\n=== 通道組合性能比較 ===")
            channel_results = evaluate_all_channels(
                model, os.path.join(args.data_dir, 'test.txt'), "", test_transform, device, args.batch_size
            )
            
            # 輸出表格格式
            print(f"\n{'通道組合':>8} | {'準確率':>10} | {'損失':>8}")
            print("-" * 30)
            for combo, metrics in channel_results.items():
                print(f"{combo:>8} | {metrics['accuracy']:>10.2f}% | {metrics['loss']:>8.4f}")
            
            # 儲存為CSV
            channel_df = pd.DataFrame.from_dict(channel_results, orient='index')
            channel_df.to_csv(os.path.join(args.output_dir, 'channel_performance.csv'))
            
            # 繪製通道比較圖
            combos = list(channel_results.keys())
            accuracies = [channel_results[c]['accuracy'] for c in combos]
            
            plt.figure(figsize=(10, 6))
            plt.bar(combos, accuracies)
            plt.xlabel('Channel Combinations')
            plt.ylabel('Accuracy (%)')
            plt.title('Comparison of Different Channel Combinations')
            plt.ylim(0, 100)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'channel_comparison.png'))
            plt.close()

    # 3.3.2 動態與傳統模型比較
    if args.compare_models:
        comparison = compare_models()
        print("\n=== 動態與傳統模型比較 ===")
        print(f"{'指標':>20} | {'動態模型':>15} | {'傳統模型':>20}")
        print("-" * 60)
        print(f"{'總參數量':>20} | {comparison['dynamic_model']['params']:>15,} | {comparison['traditional_model']['total_params']:>20,}")
        print(f"{'相對參數節省':>20} | {'-':>15} | {comparison['savings']['params_percent']:>19.2f}%")
        print(f"{'模型數量':>20} | {comparison['dynamic_model']['models_required']:>15} | {comparison['traditional_model']['models_required']:>20}")
        print(f"{'相對模型數量節省':>20} | {'-':>15} | {comparison['savings']['models_percent']:>19.2f}%")
        
        # 儲存比較結果為CSV
        comparison_data = {
            'indicators': ['Total Parameters', 'Relative Parameter Savings', 'Model Count', 'Relative Model Count Savings'],
            'dynamic_model': [
                comparison['dynamic_model']['params'],
                '...',
                comparison['dynamic_model']['models_required'],
                '...'
            ],
            'traditional_model': [
                comparison['traditional_model']['total_params'],
                f"{comparison['savings']['params_percent']:.2f}%",
                comparison['traditional_model']['models_required'],
                f"{comparison['savings']['models_percent']:.2f}%"
            ]
        }
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(os.path.join(args.output_dir, 'model_comparison.csv'), index=False)

    # 3.4.1 SE模組消融研究
    if args.ablation_se:
        se_results = ablation_study_se(
            model, os.path.join(args.data_dir, 'test.txt'), "", test_transform, device, args.batch_size, args.output_dir
        )

# 10. CLI arguments
if __name__ == "__main__":
    # 替代參數解析器的簡單類
    class Args:
        def __init__(self):
            # 數據和目錄
            self.data_dir = '.'  # 數據集目錄路徑
            self.output_dir = './output'  # 儲存檢查點和結果的目錄
            
            # 訓練參數
            self.batch_size = 64          # 訓練批次大小
            self.epochs = 30              # 訓練總輪數
            self.lr = 0.001                # 固定學習率
            self.weight_decay = 5e-4     # 固定權重衰減
            self.num_classes = 100        # 數據集中的類別數
            self.num_workers = 0          # 數據加載的工作線程數
            self.save_freq = 5            # 保存檢查點頻率 (輪數)
            self.seed = 42                # 隨機種子
            
            # 模型參數
            self.load_model = None   #'./output_v5_epoch25/best_model.pth'       # 加載模型的路徑
            
            # 執行模式
            self.train = True            # 是否訓練模型（預設為False，僅評估）
            self.eval = True              # 是否在測試集上評估
            self.test_channels = True     # 是否測試不同通道組合
            self.compare_models = True    # 是否比較動態與傳統模型
            self.ablation_se = True       # 是否執行SE模組消融研究
            
            # 固定超參數
            self.label_smoothing = 0.1
            self.optimizer_name = 'AdamW'
            self.rotation_degrees = 15
            self.color_jitter = 0.2
            self.scheduler_factor = 0.2
            self.scheduler_patience = 5

    # 創建參數對象
    args = Args()

    # 創建輸出目錄
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # 運行主函數
    main(args)