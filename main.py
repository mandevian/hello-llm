# Import os so the script can read settings from environment variables.
import os
# Import time so the script can measure calls and wait between retries.
import time
# Import load_dotenv so values in the local .env file are made available.
from dotenv import load_dotenv
# Import Google's official Gen AI client library.
from google import genai
# Import the SDK options used to set the request timeout.
from google.genai import types
# Import the exception used by the Google Gen AI SDK for API errors.
from google.genai.errors import APIError
# Import httpx so request timeouts can be recognized and retried.
import httpx
# Import Ollama's client for local-model requests.
import ollama

# Load settings from .env and let its values override existing environment variables.
load_dotenv(override=True)
# Read the selected provider from the loaded settings.
provider = os.getenv("LLM_PROVIDER")
# Stop with a clear error unless the provider is gemini or ollama.
if provider not in ("gemini", "ollama"):
    # Explain the accepted provider values.
    raise ValueError("LLM_PROVIDER must be set to either 'gemini' or 'ollama' in .env.")

# Ask for a question in the terminal and store what you type.
question = input("Enter your question: ")
# Save the Gemini model name used for requests and output.
gemini_model = "gemini-3.8-flash"
# Read the local model name, defaulting to the installed llama3.1 model.
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
# Start measuring the selected provider call, including retries and fallback.
call_started_at = time.perf_counter()
# Prepare the response objects before choosing which provider to call.
gemini_response = None
# Prepare the Ollama response for either direct use or Gemini fallback.
ollama_response = None

# Send the question to Ollama directly when it is the selected provider.
if provider == "ollama":
    # Create a client that connects to the local Ollama service.
    ollama_client = ollama.Client()
    # Start measuring the direct Ollama request.
    attempt_started_at = time.perf_counter()
    # Send the question to the selected local model.
    ollama_response = ollama_client.generate(model=ollama_model, prompt=question)
    # Print the time taken by the Ollama request.
    print(f"Ollama attempt 1/1 took {time.perf_counter() - attempt_started_at:.2f} seconds.")
# Otherwise, try Gemini and use Ollama only if all retries fail.
else:
    # Read the Gemini API key from the .env file.
    api_key = os.getenv("GEMINI_API_KEY")
    # Stop with a helpful message if the Gemini API key is not set.
    if not api_key:
        # Explain which setting must be added before a Gemini request can run.
        raise ValueError("GEMINI_API_KEY was not found; add it to your .env file.")
    # Create a client that gives each Gemini HTTP request a 30-second timeout.
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30_000))
    # Try Gemini once, then make up to three retry attempts.
    for retry_number in range(4):
        # Start timing this individual Gemini attempt.
        attempt_started_at = time.perf_counter()
        # Send the question to the requested Gemini model.
        try:
            # Request an answer from Gemini.
            gemini_response = client.models.generate_content(model=gemini_model, contents=question)
            # Stop retrying after Gemini successfully answers.
            break
        # Handle Gemini API errors and request timeouts.
        except (APIError, httpx.TimeoutException) as error:
            # Treat request timeouts as errors that can be retried.
            if isinstance(error, httpx.TimeoutException):
                # Give timeout errors a readable description in retry messages.
                failure_reason = "timeout"
            # Retry API responses that indicate rate limiting or service unavailability.
            elif error.code in (429, 503):
                # Include the response code in the retry message.
                failure_reason = f"HTTP {error.code}"
            # Re-raise other API errors instead of silently hiding them.
            else:
                # Keep unexpected API failures visible to the user.
                raise
            # Use Ollama after the last Gemini attempt fails.
            if retry_number == 3:
                # Tell the user that the local Ollama model will be used.
                print("Falling back to local model")
                # Leave the retry loop and proceed to the Ollama request.
                break
            # Calculate the exponential delay before the next retry.
            delay = 2**retry_number
            # Tell the user why the request failed and when the next retry happens.
            print(f"Gemini request failed with {failure_reason}; retry {retry_number + 1}/3 in {delay}s.")
            # Wait before sending the next Gemini request.
            time.sleep(delay)
        # Always report how long this individual Gemini attempt took.
        finally:
            # Print this attempt's elapsed time, even if it failed.
            print(f"Gemini attempt {retry_number + 1}/4 took {time.perf_counter() - attempt_started_at:.2f} seconds.")
    # Call Ollama only if every Gemini attempt failed with a retryable error.
    if gemini_response is None:
        # Create a client that connects to the local Ollama service.
        ollama_client = ollama.Client()
        # Start measuring the fallback Ollama request.
        attempt_started_at = time.perf_counter()
        # Send the question to the selected local model.
        ollama_response = ollama_client.generate(model=ollama_model, prompt=question)
        # Print the time taken by the fallback Ollama request.
        print(f"Ollama attempt 1/1 took {time.perf_counter() - attempt_started_at:.2f} seconds.")

# Print the answer and usage details returned by Gemini.
if gemini_response is not None:
    # Print the answer returned by Gemini.
    print(f"\nAnswer:\n{gemini_response.text}")
    # Identify the provider that returned the answer.
    print("Provider: gemini")
    # Show the exact Gemini model name used for the request.
    print(f"Model: {gemini_model}")
    # Show the total time including failed attempts and retry waits.
    print(f"Total time: {time.perf_counter() - call_started_at:.2f} seconds")
    # Get the token usage information returned with the Gemini answer.
    usage = gemini_response.usage_metadata
    # Print how many input tokens were used.
    print(f"\nInput tokens: {usage.prompt_token_count}")
    # Print how many output tokens were used.
    print(f"Output tokens: {usage.candidates_token_count}")
# Otherwise, print the answer and usage details returned by Ollama.
else:
    # Print the answer returned by Ollama.
    print(f"\nAnswer:\n{ollama_response.response}")
    # Identify the provider that returned the answer.
    print("Provider: ollama")
    # Show the exact Ollama model name used for the request.
    print(f"Model: {ollama_response.model}")
    # Show the total time including any Gemini retries and the Ollama request.
    print(f"Total time: {time.perf_counter() - call_started_at:.2f} seconds")
    # Print how many input tokens the local model used.
    print(f"\nInput tokens: {ollama_response.prompt_eval_count}")
    # Print how many output tokens the local model generated.
    print(f"Output tokens: {ollama_response.eval_count}")
