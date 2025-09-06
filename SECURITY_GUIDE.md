# Secure Credential Handling in Kubeflow Pipelines

This document explains how to securely handle DagsHub credentials in Kubeflow pipelines without exposing sensitive information in your code repository.

## Security Approach

Instead of hardcoding credentials or using environment variables that might be exposed in logs, we pass credentials as pipeline parameters. This approach allows you to:

1. **Keep secrets out of your code repository**
2. **Provide credentials at runtime** when executing the pipeline
3. **Use Kubeflow's built-in secret management** features
4. **Maintain audit trails** of credential usage

## Implementation Details

### 1. Pipeline Parameters

The pipeline accepts DagsHub credentials as parameters:

```python
def spam_detection_pipeline(
    # ... other parameters ...
    dagshub_username: str = 'your_dagshub_username',
    dagshub_token: str = 'your_dagshub_token'
):
```

### 2. Component Integration

The `push_model` component receives credentials as arguments:

```python
@dsl.container_component
def push_model(
    # ... other parameters ...
    dagshub_username: str,
    dagshub_token: str,
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        # ... container spec ...
        args=[
            # ... other args ...
            dagshub_username,
            dagshub_token
        ]
    )
```

### 3. Model Pusher Authentication

The model pusher component uses the provided credentials:

```python
def setup_dagshub_auth(dagshub_username: str, dagshub_token: str):
    """Setup DagsHub authentication using provided credentials."""
    dagshub.auth.add_app_token(dagshub_token)
```

## Running the Pipeline Securely

### Option 1: Kubeflow UI (Recommended)

1. **Upload the pipeline** to Kubeflow
2. **Create a new run** from the pipeline
3. **Override parameters** in the UI:
   - Set `dagshub_username` to your actual DagsHub username
   - Set `dagshub_token` to your actual DagsHub token
4. **Execute the pipeline**

### Option 2: Kubeflow SDK

```python
from kfp import Client

# Initialize Kubeflow client
client = Client()

# Create experiment
experiment = client.create_experiment(name='spam-detection')

# Run pipeline with credentials
run = client.run_pipeline(
    experiment_id=experiment.id,
    job_name='spam-detection-run',
    pipeline_package_path='spam_detection_pipeline.yaml',
    params={
        'dagshub_username': 'your_actual_username',
        'dagshub_token': 'your_actual_token',
        'repo_owner_name': 'your_username',
        'repo_name': 'your_repo_name',
        'model_name': 'spam_detection_model'
    }
)
```

### Option 3: Using Kubeflow Secrets (Advanced)

For production environments, consider using Kubeflow's secret management:

1. **Create a secret** in Kubeflow:
   ```bash
   kubectl create secret generic dagshub-credentials \
     --from-literal=username=your_username \
     --from-literal=token=your_token
   ```

2. **Mount the secret** in your component:
   ```python
   @dsl.container_component
   def push_model(...):
       return dsl.ContainerSpec(
           # ... other config ...
           env=[
               dsl.EnvVar(name='DAGSHUB_USERNAME', 
                         value_from=dsl.EnvVarSource(
                             secret_key_ref=dsl.SecretKeySelector(
                                 name='dagshub-credentials',
                                 key='username'
                             )
                         )),
               dsl.EnvVar(name='DAGSHUB_TOKEN',
                         value_from=dsl.EnvVarSource(
                             secret_key_ref=dsl.SecretKeySelector(
                                 name='dagshub-credentials',
                                 key='token'
                             )
                         ))
           ]
       )
   ```

## Getting DagsHub Credentials

### 1. Username
Your DagsHub username is the same as your GitHub username (if using GitHub integration) or the username you created on DagsHub.

### 2. Access Token
To create a DagsHub access token:

1. **Log in** to your DagsHub account
2. **Go to Settings** → **Access Tokens**
3. **Click "Generate New Token"**
4. **Set permissions**:
   - `repo:read` - Read repository data
   - `repo:write` - Write to repository
   - `repo:admin` - Admin access (if needed)
5. **Copy the token** (you won't see it again!)

## Security Best Practices

### 1. Token Permissions
- **Use minimal permissions** - only grant what's necessary
- **Set expiration dates** for tokens when possible
- **Rotate tokens regularly** in production

### 2. Pipeline Execution
- **Never commit credentials** to version control
- **Use parameter overrides** at runtime
- **Monitor pipeline logs** for credential exposure
- **Use Kubeflow's audit logging** to track credential usage

### 3. Environment Separation
- **Use different tokens** for different environments (dev/staging/prod)
- **Separate repositories** for different environments
- **Implement proper access controls** per environment

## Troubleshooting

### Common Issues

1. **Authentication Failed**
   - Verify username and token are correct
   - Check token permissions
   - Ensure token hasn't expired

2. **Repository Access Denied**
   - Verify token has appropriate permissions
   - Check repository ownership/permissions
   - Ensure repository name is correct

3. **Pipeline Parameter Issues**
   - Verify parameter names match exactly
   - Check parameter types (all strings)
   - Ensure no extra spaces in credentials

### Debug Mode

Enable debug logging in the model pusher:

```python
# In model_pusher.py
logger.setLevel('DEBUG')
```

This will provide detailed authentication logs (without exposing credentials).

## Example: Complete Pipeline Execution

```bash
# 1. Compile the pipeline
python pipeline.py

# 2. Upload to Kubeflow UI or use SDK
# In Kubeflow UI, set these parameters:
# dagshub_username: your_actual_username
# dagshub_token: your_actual_token
# repo_owner_name: your_username
# repo_name: your_repo_name
# model_name: spam_detection_model
```

This approach ensures your credentials remain secure while enabling automated model deployment through Kubeflow pipelines.
