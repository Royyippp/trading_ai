import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict
from dotenv import load_dotenv
import json
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the state structure
class State(TypedDict):
    user_query: str
    technical_series: dict


# Initialize the real OpenAI model
model = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key)


# Function to test
def determine_technical_series(state: State):
    """
    Use the LLM to determine which technical series are needed based on the query.
    """
    # Create the prompt
    prompt = f"""
    A user asked: "{state['user_query']}"
    Determine which technical series to generate (e.g., MACD, RSI).
    Respond with a JSON object listing the series names and any required parameters, like:
    {{
        "series": [
            {{
                "name": "MACD",
                "parameters": {{"fast": 12, "slow": 26, "signal": 9}}
            }},
            {{
                "name": "RSI",
                "parameters": {{"length": 14}}
            }}
        ]
    }}
    """

    # Use model.invoke to get the response
    response = model.invoke([HumanMessage(content=prompt)])
    print("Model Response:", response.content)

    # Parse the JSON content from the AIMessage
    try:
        response_data = json.loads(response.content.strip('```json\n'))
        state['technical_series'] = response_data.get('series', [])
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        state['technical_series'] = []  # Set to empty if parsing fails

    return state


# Test function
def test_determine_technical_series_real():
    # Initialize a dummy state
    text = 'What is the percentage of winning using MACD crossing strategy?'
    initial_state = State(
        user_query=HumanMessage(text),
        technical_series={}
    )

    # Measure execution time
    start_time = time.time()
    updated_state = determine_technical_series(initial_state)
    end_time = time.time()

    # Print execution time and response
    print(f"Function executed successfully in {end_time - start_time:.4f} seconds.")
    print("Generated Technical Series:", updated_state['technical_series'])


# Run the test
if __name__ == "__main__":
    test_determine_technical_series_real()