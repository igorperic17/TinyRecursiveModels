# Weights & Biases Integration

This guide explains how to use Weights & Biases (wandb) logging with the TRM language training.

## 🚀 Quick Start

### 1. Enable Wandb Logging (Default)
```bash
# Wandb is enabled by default
python train_language.py
```

### 2. Disable Wandb Logging
```bash
# Disable wandb logging
USE_WANDB=false python train_language.py
```

### 3. Login to Wandb (Optional)
```bash
# Login to wandb for cloud logging
wandb login
```

## 📊 What Gets Logged

### **Training Metrics:**
- `batch_loss`: Loss for each training batch
- `epoch_loss`: Average loss per epoch
- `learning_rate`: Current learning rate
- `num_batches`: Number of batches per epoch

### **Model Information:**
- `model_parameters`: Total number of model parameters
- `vocab_size`: Vocabulary size of the tokenizer
- `dataset_size`: Number of training examples

### **Generated Text:**
- `generated_texts`: Sample text generation results
- `generation_epoch`: Epoch when text was generated

### **Configuration:**
- Model architecture parameters
- Training hyperparameters
- Device information

## 🔧 Configuration

### **Environment Variables:**
```bash
# Enable/disable wandb
export USE_WANDB=true   # Enable (default)
export USE_WANDB=false  # Disable

# Wandb project settings
export WANDB_PROJECT="trm-language-generation"
export WANDB_ENTITY="your-username"
```

### **Project Settings:**
- **Project**: `trm-language-generation`
- **Run Name**: `trm-lang-experiment` (configurable)
- **Tags**: Automatically added based on model type

## 📈 Monitoring Training

### **Real-time Monitoring:**
1. **Local Dashboard**: Wandb runs locally by default
2. **Cloud Dashboard**: Login with `wandb login` for cloud access
3. **Metrics**: Loss curves, learning rate, generated text
4. **System**: GPU/CPU usage, memory consumption

### **Key Metrics to Watch:**
- **Loss Trends**: Should decrease over epochs
- **Learning Rate**: Should follow the schedule
- **Generated Text**: Quality should improve over time
- **Batch Performance**: Consistent batch processing

## 🎯 Advanced Usage

### **Custom Run Names:**
```bash
# Set custom run name
python train_language.py run_name="my-experiment"
```

### **Multiple Experiments:**
```bash
# Run different experiments
python train_language.py run_name="small-model"
python train_language.py run_name="large-model"
```

### **Offline Mode:**
```bash
# Run offline (sync later)
WANDB_MODE=offline python train_language.py
```

## 🔍 Troubleshooting

### **Common Issues:**

1. **Wandb not logging:**
   ```bash
   # Check if wandb is installed
   pip install wandb
   
   # Check environment variable
   echo $USE_WANDB
   ```

2. **Login issues:**
   ```bash
   # Re-login to wandb
   wandb login --relogin
   ```

3. **Permission errors:**
   ```bash
   # Check wandb directory permissions
   ls -la ~/.wandb/
   ```

### **Debug Mode:**
```bash
# Enable debug logging
WANDB_DEBUG=true python train_language.py
```

## 📊 Dashboard Features

### **Available Views:**
- **Overview**: Run summary and key metrics
- **Charts**: Loss curves and learning rate plots
- **System**: Hardware utilization graphs
- **Logs**: Training output and generated text
- **Files**: Model checkpoints and artifacts

### **Comparing Runs:**
- **Runs Table**: Compare multiple experiments
- **Parallel Coordinates**: Hyperparameter analysis
- **Scatter Plots**: Metric correlations

## 🎉 Best Practices

### **Experiment Tracking:**
1. **Use descriptive run names**
2. **Tag related experiments**
3. **Save important checkpoints**
4. **Document configuration changes**

### **Performance Monitoring:**
1. **Watch loss convergence**
2. **Monitor generated text quality**
3. **Track training speed**
4. **Check for overfitting**

The wandb integration provides comprehensive monitoring and logging for your TRM language training experiments! 🚀
