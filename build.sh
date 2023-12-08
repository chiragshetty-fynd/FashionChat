#!/bin/bash
docker build -t fashion_chat:v1 --build-arg OPENAI_API_KEY=${OPENAI_API_KEY} -f Dockerfile .