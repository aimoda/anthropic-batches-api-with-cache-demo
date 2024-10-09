#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Author: David Manouchehri

import asyncio  # Asynchronous I/O library
import logging  # Logging facility
import httpx  # Asynchronous HTTP client
import anthropic  # Anthropics' API client library
import os  # Operating system utilities
import hashlib  # Hashing algorithms

# Set up the logger for debugging
logger = logging.getLogger(__name__)  # Create a logger for this module
logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

c_handler = logging.StreamHandler()  # Create a console handler
logger.addHandler(c_handler)  # Add the handler to the logger

# Base URL for the Anthropics API, defaulting to 'https://api.anthropic.com' if not set in environment
BASE_URL = os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com")

# Specify the model to be used for processing
MODEL_TO_USE = "claude-3-5-sonnet-20240620"


async def main():
    # Initialize an asynchronous Anthropics client with custom HTTP settings
    client = anthropic.AsyncAnthropic(
        http_client=httpx.AsyncClient(
            http2=True,  # Use HTTP/2 protocol
            limits=httpx.Limits(
                max_connections=None,  # No limit on maximum connections
                max_keepalive_connections=None,  # No limit on keep-alive connections
                keepalive_expiry=None,  # No expiration for keep-alive connections
            ),
        ),
        default_headers={
            "Priority": "u=0",  # Set priority header for request
            "Accept-Encoding": "zstd;q=1.0, br;q=0.9, gzip;q=0.8, deflate;q=0.7",  # Specify accepted encodings
            "cf-skip-cache": "true",  # Bypass Cloudflare cache
        },
        max_retries=0,  # Disable automatic retries
        timeout=3600,  # Set a timeout of 1 hour
        base_url=BASE_URL,  # Use the base URL specified above
    )

    # Read the long content from a markdown file
    with open("long_prompt.md", "r") as file:
        long_content = file.read()

    # Create an MD5 hash of the content for unique identification
    long_content_md5_hash = hashlib.md5(long_content.encode()).hexdigest()

    # Define the list of questions to ask the model
    questions_to_ask = [
        "What companies are mentioned in this blog post?",
        "What are the key takeaways from this blog post?",
        "List all the people mentioned in this blog post.",
    ]

    # Prepare a list to hold all the request payloads
    requests = []
    for question in questions_to_ask:
        # Create a unique custom ID by hashing the content and question
        custom_id = (
            f"{long_content_md5_hash}{hashlib.md5(question.encode()).hexdigest()}"
        )
        # Create a user ID for metadata
        user_id = f"ai.moda-dev-{custom_id}"
        # Build the request dictionary with parameters
        request = {
            "custom_id": custom_id,  # Unique identifier for the request
            "params": {
                "model": MODEL_TO_USE,  # Specify the model to use
                "max_tokens": 8192,  # Maximum number of tokens in the response
                "temperature": 0.0,  # Sampling temperature for determinism
                "metadata": {"user_id": user_id},  # Metadata including user ID
                "messages": [
                    {
                        "role": "user",  # Role of the message sender
                        "content": [
                            {
                                "type": "text",
                                "text": long_content,  # The main content to process
                                "cache_control": {
                                    "type": "ephemeral"
                                },  # Cache control settings
                            },
                            {
                                "type": "text",
                                "text": question,  # The question to ask about the content
                            },
                        ],
                    }
                ],
            },
        }
        requests.append(request)  # Add the request to the list

    # Send a single message to cache the long content
    message = await client.messages.create(
        model=MODEL_TO_USE,  # Specify the model to use
        max_tokens=1,  # Minimal tokens since we're just caching
        temperature=0.0,  # Deterministic output
        metadata={
            "user_id": f"ai.moda-dev-{long_content_md5_hash}"
        },  # Metadata with user ID
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": long_content,  # The content to cache
                        "cache_control": {"type": "ephemeral"},  # Ephemeral caching
                    }
                ],
            }
        ],
        extra_headers={
            "anthropic-beta": "prompt-caching-2024-07-31",  # Use beta features for prompt caching
        },
    )

    # Log the response from caching the content
    logger.debug(message.model_dump_json(indent=2))

    # Send the batch of questions to the API
    batch_create_result = await client.beta.messages.batches.create(
        requests=requests,  # The list of request payloads
        extra_headers={
            "anthropic-beta": "prompt-caching-2024-07-31,message-batches-2024-09-24"  # Use beta features for batching
        },
    )
    # Log the result of the batch creation
    logger.debug(batch_create_result.model_dump_json(indent=2))

    # Check the processing status of the batch
    processing_status = batch_create_result.processing_status

    # Poll the API until the batch processing has ended
    while processing_status != "ended":
        await asyncio.sleep(10)  # Wait for 10 seconds before polling again
        try:
            # Retrieve the latest batch response
            batch_response = await client.beta.messages.batches.retrieve(
                batch_create_result.id,  # Use the batch ID from the creation result
            )
            processing_status = batch_response.processing_status  # Update the status
            # Log the current status of the batch
            logger.debug(
                f"File Id: {batch_create_result.id}, Status: {processing_status}"
            )
        except Exception as e:
            # Log any exceptions that occur during retrieval
            logger.debug(f"An error occurred: {e}")

    # Check if the batch processing ended successfully and results are available
    if (
        batch_response.processing_status != "ended"
        or batch_response.results_url is None
    ):
        # Log an error message if batch processing failed
        logger.debug("Batch processing failed.")
        logger.debug(batch_response.model_dump_json(indent=2))
        return  # Exit the function

    # Log the final batch response after processing has ended
    logger.debug(batch_response.model_dump_json(indent=2))

    # Retrieve the results from the batch processing
    batch_result = await client.beta.messages.batches.results(
        batch_create_result.id,  # Use the batch ID to get results
    )

    # Iterate asynchronously over each result in the batch
    async for result in batch_result:
        # Dump the result to a JSON-formatted string
        json_string = result.model_dump_json(indent=2)
        # Log the result
        logger.info(json_string)
        # Write the result to a JSON file named after the custom ID
        with open(f"output/{result.custom_id}.json", "w") as f:
            f.write(json_string)

    return  # End of the main function


# Check if the script is being run directly
if __name__ == "__main__":
    # Run the main asynchronous function using asyncio
    asyncio.run(main())
