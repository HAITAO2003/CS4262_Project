# Track 2: Customer Chat

## Overview
This starter kit provides a scaffold for building a high-performance chat engine compatible with the OpenAI API format. Your goal is to integrate a serving engine (like vLLM) to handle chat completions efficiently.

### Project Structure
```text
.
├── app/
│   ├── main.py          # FastAPI entry point.
│   ├── constants.py     # Necessary constants
│   ├── schemas.py       # Pydantic models (OpenAI compatible).
│   └── chat_engine.py   # MAIN LOGIC GOES HERE.
├── Dockerfile           # Environment definition.
├── pyproject.toml       # Python dependencies.
└── README.md
```

### Your Task
Modify `app/chat_engine.py` to replace the mock generation logic with a real inference engine. The benchmark evaluates throughput, latency, and perplexity (output quality).

### API Specification
*   **Endpoint**: `POST /v1/chat/completions`
*   **Input**: Standard OpenAI Chat Completion JSON (messages, model, etc.).
*   **Output**: Standard OpenAI Chat Completion Response.

## Local Development & Testing

### 1. Setup
The `docker-compose.yaml` in the repository specifies the AMD64 platform to align with the test environment. If you are developing locally on a non-AMD64 machine, please comment out the `platform: linux/amd64` line in `docker-compose.yaml` to avoid architecture errors.

```yaml
services:
  chat-engine:
    # ...
    # platform: linux/amd64  <-- Comment out for local ARM64 testing
```

### 2. Build and Run
We recommend using Docker Compose for local testing. The service will listen on http://localhost:8000.

```bash
# Build the image, and start a container in detach mode
docker compose up --build -d
```

To shutdown the engine, 

```bash
docker compose stop
```

### 3. Check Status
*   Liveness: `GET http://localhost:8000/health`
*   Readiness: `GET http://localhost:8000/ready`

## Deployment on Vast.ai

To test your engine on Vast.ai, you need to push your Docker image to a container registry (GHCR) and create a Template.

### 1. Push Docker Image to GHCR (Private)

1.  **Login to GHCR**
    Create a GitHub Personal Access Token under https://github.com/settings/tokens with `write:packages` and `read:packages` permissions.
```bash
export CR_PAT=YOUR_TOKEN
export GITHUB_USERNAME=YOUR_GITHUB_USERNAME
echo $CR_PAT | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin
```

2.  **Build & Push**
    
```bash
docker compose build
docker tag chat-engine:latest ghcr.io/$GITHUB_USERNAME/chat-engine:latest
docker push ghcr.io/$GITHUB_USERNAME/chat-engine:latest
```

We suggest remembering the SHA256 digest of the image to ensure reproducibility. You can find it at the end of output of `docker push`, or use the following command:

```bash
docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/$GITHUB_USERNAME/chat-engine:latest
```

### 2. Create a Private Template on Vast.ai

1.  Log in to Vast.ai and go to the Templates page (https://cloud.vast.ai/templates/).
2. Click **Create New Template**, set the relevant fields as following:
    -  **Image Path:** `ghcr.io/YOUR_GITHUB_USERNAME/chat-engine@sha256:YOUR_DIGEST`
    -  **Docker Options:** Add `-p 8000:8000` to expose the port.
    -  **Launch Mode:** `Docker ENTRYPOINT`.
    -  **Docker Repository Authentication:** Set Server as `ghcr.io` and input your github username and PAT accordingly. 
    -  **Disk Space:** We suggest 128GB as the image and weights take around 40GB. 

### 3. Launch & Test
Rent a GPU instance using your template. We suggest using NVIDIA RTX 5080 as this aligns with the test environment. 
Note that if you want to use Flash attention, please use Vast AI instances with Max CUDA: 13.0. 

Once running, you can open the **IP & Port Info** tab of the instance by clicking on the IP address, which provide the external port `$VAST_PORT` mapping to the internal 8000 port (which the engine is listening on). 
Then you should be able to access the engine at `http://$VAST_IP:$VAST_PORT`.

## Submission Instructions

Submit the following items to Canvas:
1.  **Full Image Name with Digest**: (e.g., `ghcr.io/username/track2-chat@sha256:abcdef...`)
2.  **Read-Only PAT** and the related **Github username**: A GitHub Personal Access Token with `read:packages` permission, plus the associated Github username.
3.  **Archived Code**: A `.zip` file of your project code.

### 1. Get Image Digest & Check Architecture
Ensure your image is built for `linux/amd64` and get its digest.

```bash
# Pull the image (if not local)
docker pull ghcr.io/${GITHUB_USERNAME}/chat-engine:latest

# Inspect to get Digest and Architecture
docker inspect ghcr.io/${GITHUB_USERNAME}/chat-engine:latest --format 'Digest: {{.RepoDigests}} | Arch: {{.Architecture}}'
```
*   **Verify**: Architecture must be `amd64`.
*   **Copy**: The full string `ghcr.io/YOUR_GITHUB_USERNAME/chat-engine@sha256:YOUR_DIGEST`.

### 2. Generate Read-Only PAT
To allow TAs to grade your submission without full access to your account:
1.  Go to GitHub Settings -> Developer Settings -> Personal access tokens (Classic).
2.  Generate new token (classic).
3.  **Scopes**: Select **ONLY** `read:packages`.
4.  **Expiration**: Set to at least 2 weeks after the submission deadline.
5.  Generate and copy the token.

### 3. Test Your Submission Credentials
**Crucial:** Verify that the PAT works for pulling the image.

```bash
# 1. Logout of your current session
docker logout ghcr.io

# 2. Login with the READ-ONLY PAT
echo "YOUR_READ_ONLY_PAT" | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# 3. Try to pull your image
docker pull ghcr.io/YOUR_USERNAME/track2-chat@sha256:<YOUR_DIGEST>
```
If the pull succeeds, your credentials are correct.

### 4. Archive Code
Zip your `track2_chat` folder (excluding `.venv`, `__pycache__`, and large model files).
```bash
zip -r track2_chat_submission.zip . -x "*.venv*" -x "*__pycache__*"
```

Submit the Image URI (with digest), the Read-Only PAT, and the Zip file to Canvas.