#!/bin/bash
docker build -t fashion_chat:v0 --build-arg OPENAI_API_KEY=${OPENAI_API_KEY} -f Dockerfile .