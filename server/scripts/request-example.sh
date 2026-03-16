#!/bin/bash
# Test script for Track 2: Chat Engine
BASE_URL=${1:-"http://localhost:8000"}

echo "--- Testing Track 2: Chat ---"
echo "Health check:"
curl -s "$BASE_URL/health" | uv run python -m json.tool
echo ""

echo "Waiting for engine to be ready..."
max_retries=30
count=0
while [ $count -lt $max_retries ]; do
  status_code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/ready")
  if [ "$status_code" -eq 200 ]; then
    echo "Engine is ready!"
    break
  fi
  echo "Engine not ready yet (HTTP $status_code)... retrying in 2s ($((count+1))/$max_retries)"
  sleep 2
  count=$((count+1))
done

if [ $count -eq $max_retries ]; then
  echo "Timeout waiting for engine to be ready."
  exit 1
fi

echo ""
echo "Chat completion:"
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7
  }' | uv run python -m json.tool
echo ""
