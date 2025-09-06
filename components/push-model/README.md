# Model Pusher Component

This component handles model registration, comparison, and production promotion in MLflow with DagsHub integration.

## Features

- **Secure DagsHub Authentication**: Uses pipeline parameters for authentication without exposing secrets
- **Automatic Model Comparison**: Compares new models with production models
- **Intelligent Production Promotion**: Promotes models to production only if they meet improvement thresholds
- **Automatic Model Retirement**: Retires old production models when new ones are promoted

## Secure Credential Handling

**Important**: This component uses secure credential passing through pipeline parameters instead of environment variables to avoid exposing secrets in your code repository.

### DagsHub Credentials

The component receives DagsHub credentials as arguments from the Kubeflow pipeline:

- `dagshub_username`: Your DagsHub username
- `dagshub_token`: Your DagsHub access token

### Getting Your DagsHub Token

1. **Log in** to your DagsHub account
2. **Go to Settings** → **Access Tokens**
3. **Click "Generate New Token"**
4. **Set permissions**:
   - `repo:read` - Read repository data
   - `repo:write` - Write to repository
   - `repo:admin` - Admin access (if needed)
5. **Copy the token** (you won't see it again!)

### Security Benefits

- ✅ **No secrets in code repository**
- ✅ **Credentials provided at runtime**
- ✅ **Kubeflow audit trail**
- ✅ **No environment variable exposure**

## Configuration

The model comparison behavior is configured in `params.yaml`:

```yaml
model_comparison:
  improvement_threshold: 0.05  # 5% improvement required to promote to production
  primary_metric: "accuracy"    # Primary metric to compare models
  comparison_metrics:          # Additional metrics to log for comparison
    - "precision"
    - "recall"
    - "f1_score"
```

### Configuration Parameters

- **improvement_threshold**: Minimum percentage improvement required to promote a new model to production (default: 0.05 = 5%)
- **primary_metric**: The metric used for comparison (default: "accuracy")
- **comparison_metrics**: Additional metrics to log for comprehensive comparison

## Model Promotion Logic

The component follows this decision flow:

1. **Load new model metrics** from the evaluation component
2. **Retrieve production model metrics** from MLflow registry
3. **Compare performance** using the configured primary metric
4. **Make promotion decision**:
   - If no production model exists → Promote new model to production
   - If new model improves by threshold → Promote to production and retire old model
   - If improvement below threshold → Register as staging only

## Usage

The component is called with the following arguments:

```bash
python model_pusher.py \
  --repo_owner_name "your_username" \
  --repo_name "your_repo" \
  --model_name "spam_detection_model" \
  --stage "Production" \
  --param_path "/path/to/params.yaml" \
  --model_path "/path/to/model" \
  --metrics_path "/path/to/metrics" \
  --dagshub_username "your_dagshub_username" \
  --dagshub_token "your_dagshub_token"
```

### Pipeline Integration

In Kubeflow pipelines, credentials are passed as parameters:

```python
push_op = push_model(
    model=train_op.outputs['model'],
    metrics=evaluate_op.outputs['metrics'],
    repo_owner_name=repo_owner_name,
    repo_name=repo_name,
    model_name=model_name,
    stage=stage,
    param_path=param_file_path,
    dagshub_username=dagshub_username,  # Provided at runtime
    dagshub_token=dagshub_token         # Provided at runtime
)
```

## Model Stages

- **Production**: Currently deployed model serving live traffic
- **Staging**: Candidate models that meet quality standards but haven't been promoted
- **Archived**: Retired models that are no longer in active use

## Logging

The component provides comprehensive logging including:
- Model comparison results
- Promotion decisions
- Performance metrics
- Error handling

Logs are written to both console and `logs/Model_Pusher.log`.

## Error Handling

The component includes robust error handling for:
- Missing DagsHub credentials
- MLflow registry connectivity issues
- Model loading failures
- Metric comparison errors

## Integration with Kubeflow

This component is designed to work seamlessly with Kubeflow pipelines:
- Non-interactive authentication prevents pipeline failures
- Configurable thresholds allow for different promotion strategies
- Comprehensive logging aids in pipeline debugging
