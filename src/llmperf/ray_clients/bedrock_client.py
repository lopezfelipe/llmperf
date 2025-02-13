import os
import time
from typing import Any, Dict

import boto3
import ray

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics


@ray.remote
class BedrockClient(LLMClient):
    """Client for the Bedrock Converse Stream API."""

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            raise ValueError("AWS_ACCESS_KEY_ID must be set.")
        if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
            raise ValueError("AWS_SECRET_ACCESS_KEY must be set.")
        if not os.environ.get("AWS_REGION_NAME"):
            raise ValueError("AWS_REGION_NAME must be set.")

        prompt = request_config.prompt
        prompt, prompt_len = prompt
        model = request_config.model

        br_runtime = boto3.client(
            "bedrock-runtime", region_name=os.environ.get("AWS_REGION_NAME")
        )

        #system_prompts = [{"text": " "}]
        messages = [
            {
                "role": "user", "content": [{"text": prompt}]
            }
        ]

        sampling_params = request_config.sampling_params
        inference_config = {
            "maxTokens": sampling_params["max_tokens"]
        }

        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = None
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0
        metrics = {}

        start_time = time.monotonic()
        most_recent_received_token_time = start_time

        try:
            response = br_runtime.converse_stream(
                modelId=model,
                messages=messages,
                #system=system_prompts,
                inferenceConfig=inference_config
            )
            generated_text = ""
            tokens_received = 0
            for chunk in response['stream']:
                if "contentBlockDelta" in chunk:
                    text = chunk["contentBlockDelta"]["delta"]["text"]
                    generated_text += text
                    if not ttft:
                        ttft = time.monotonic() - start_time
                        time_to_next_token.append(ttft)
                    else:
                        time_to_next_token.append(
                            time.monotonic() - most_recent_received_token_time
                        )
                    most_recent_received_token_time = time.monotonic()
                if "metadata" in chunk:
                    tokens_received += chunk["metadata"]["usage"]["outputTokens"]

            ttft = ttft - start_time
            total_request_time = time.monotonic() - start_time
            output_throughput = tokens_received / total_request_time

        except Exception as e:
            print(f"Warning Or Error: {e}")
            print(error_response_code)
            error_msg = str(e)
            error_response_code = 500

        metrics[common_metrics.ERROR_MSG] = error_msg
        metrics[common_metrics.ERROR_CODE] = error_response_code
        metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token)
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config