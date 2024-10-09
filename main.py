#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Author: David Manouchehri
import asyncio

import logging
import httpx
import anthropic
import os
import hashlib

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

c_handler = logging.StreamHandler()
logger.addHandler(c_handler)

BASE_URL = os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com")

MODEL_TO_USE = "claude-3-5-sonnet-20240620"

async def main():
    client = anthropic.AsyncAnthropic(
        http_client=httpx.AsyncClient(
            http2=True,
            limits=httpx.Limits(
                max_connections=None,
                max_keepalive_connections=None,
                keepalive_expiry=None,
            ),
        ),
        default_headers={
            "Priority": "u=0",
            "Accept-Encoding": "zstd;q=1.0, br;q=0.9, gzip;q=0.8, deflate;q=0.7",
            "cf-skip-cache": "true",
        },
        max_retries=0,
        timeout=3600,
        base_url=BASE_URL,
    )

    with open("long_prompt.md", "r") as file:
        long_content = file.read()

    # create MD5 hash of the content
    long_content_md5_hash = hashlib.md5(long_content.encode()).hexdigest()

    questions_to_ask = [
        "What companies are mentioned in this blog post?",
        "What are the key takeaways from this blog post?",
        "List all the people mentioned in this blog post.",
    ]

    requests = []
    for question in questions_to_ask:
        custom_id = f"{long_content_md5_hash}{hashlib.md5(question.encode()).hexdigest()}"
        user_id = f"ai.moda-dev-{custom_id}"
        request = {
            "custom_id": custom_id,
            "params": {
                "model": MODEL_TO_USE,
                "max_tokens": 8192,
                "temperature": 0.0,
                "metadata": {"user_id": user_id},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": long_content,
                                "cache_control": {"type": "ephemeral"},
                            },
                            {
                                "type": "text",
                                "text": question,
                            },
                        ],
                    }
                ],
            },
        }
        requests.append(request)

    # Send one message to insert the content into the cache
    message = await client.messages.create(
        model=MODEL_TO_USE,
        max_tokens=1,
        temperature=0.0,
        metadata={"user_id": f"ai.moda-dev-{long_content_md5_hash}"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": long_content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
        extra_headers={
            "anthropic-beta": "prompt-caching-2024-07-31",
        },
    )

    logger.debug(message.model_dump_json(indent=2))

    batch_create_result = await client.beta.messages.batches.create(
        requests=requests,
        extra_headers={
            "anthropic-beta": "prompt-caching-2024-07-31,message-batches-2024-09-24"
        },
    )
    logger.debug(batch_create_result.model_dump_json(indent=2))

    processing_status = batch_create_result.processing_status

    while processing_status != "ended":
        await asyncio.sleep(10)
        try:
            batch_response = await client.beta.messages.batches.retrieve(
                batch_create_result.id,
            )
            processing_status = batch_response.processing_status
            logger.debug(
                f"File Id: {batch_create_result.id}, Status: {processing_status}"
            )
        except Exception as e:
            logger.debug(f"An error occurred: {e}")

    if (
        batch_response.processing_status != "ended"
        or batch_response.results_url is None
    ):
        logger.debug("Batch processing failed.")
        logger.debug(batch_response.model_dump_json(indent=2))
        return

    logger.debug(batch_response.model_dump_json(indent=2))

    batch_result = await client.beta.messages.batches.results(
        batch_create_result.id,
    )

    async for result in batch_result:
        json_string = result.model_dump_json(indent=2)
        logger.info(json_string)
        with open(f"output/{result.custom_id}.json", "w") as f:
            f.write(json_string)

    return


if __name__ == "__main__":
    asyncio.run(main())
