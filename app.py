import json
from flask import Flask, request, jsonify
import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure OpenAI API Key from .env
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_prompt_turkish(sentence):
    return f"""
    Please break down the following Turkish sentence "{sentence}" grammatically and return the result **only** as a valid JSON object with no explanation, no introductory text, and no other output:

    The JSON structure should be:
    {{
        "sentence": {{
            "word": {{
                "position": "The position of the word in the sentence",
                "part_of_speech": "The part of speech",
                "root": "The root of the word",
                "noun_components": {{
                    "stem": "The stem of the noun",
                    "suffixes": "The suffixes of the noun"
                }},
                "noun_case": "The noun case",
                "noun_case_components": "The suffix that marks the noun case",
                "verb_tense": "The verb tense",
                "verb_tense_components": "The components that make up the verb tense"
            }},
            "another_word": {{
                ...
            }}
        }},
        "relationship_matrix": [[...]] 1 for modifies, 0 for null
    }}

    The key for each word in the sentence should be the actual word itself (e.g., "Bisiklete" or "binen") and not a description like "first word" or "second word."
    """

def generate_prompt_italian(sentence):
    return f"""
    Please break down the following Italian sentence "{sentence}" grammatically and return the result **only** as a valid JSON object with no explanation, no introductory text, and no other output:

    The JSON structure should be:
    {{
        "sentence": {{
            "word": {{
                "position": "The position of the word in the sentence",
                "part_of_speech": "The part of speech",
                "root": "The root of the word",
                "noun_components": {{
                    "stem": "The stem of the noun",
                    "suffixes": "The suffixes of the noun"
                }},
                "noun_case": "The noun case",
                "noun_case_components": "The suffix that marks the noun case",
                "verb_tense": "The verb tense",
                "verb_tense_components": "The components that make up the verb tense"
            }},
            "another_word": {{
                ...
            }}
        }},
        "relationship_matrix": [[...]] 1 for modifies, 0 for null
    }}

    The key for each word in the sentence should be the actual word itself (e.g., "Io" or "faccio") and not a description like "first word" or "second word."
    """

@app.route('/grammar-breakdown', methods=['POST'])
def grammar_breakdown():
    data = request.json
    sentence = data.get('sentence', "")
    language = data.get('language', "").capitalize()  # Case-insensitive check

    if not sentence or language not in ['Turkish', 'Italian']:
        return jsonify({"error": "Please provide a valid sentence and language (Turkish or Italian)"}), 400

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides detailed grammatical breakdowns of sentences."},
                {"role": "user", "content": generate_prompt_turkish(sentence) if language == 'Turkish' else generate_prompt_italian(sentence)}
            ],
            max_tokens=1500,  # Increased token limit
            temperature=0
        )

        # Log the entire OpenAI response
        breakdown_text = response['choices'][0]['message']['content'].strip()
        print("Raw OpenAI Response:", breakdown_text)

        # Ensure the response is properly formatted and ends with a valid closing bracket
        json_start = breakdown_text.find('{')
        if json_start == -1:
            raise ValueError("No JSON found in OpenAI response")
        
        # Try to fix incomplete JSON (e.g., missing closing brackets)
        breakdown_text = breakdown_text[json_start:].strip()
        
        if not breakdown_text.endswith('}'):
            print("Detected incomplete JSON, attempting to fix it")
            breakdown_text += '}'  # This is a simplistic approach; you may need more sophisticated fixes

        # Convert the stringified JSON to a dictionary
        breakdown_json = json.loads(breakdown_text)

        # Extract the sentence object and convert it to a sorted list based on 'position'
        sentence_structure = breakdown_json.get("sentence", {})
        sorted_sentence = sorted(sentence_structure.items(), key=lambda x: x[1]['position'])

        # Replace the sentence object with the sorted list
        breakdown_json["sentence"] = [{"word": k, **v} for k, v in sorted_sentence]

        # Return the sorted JSON response
        return jsonify(breakdown_json)

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")  # Debugging output for JSON error
        return jsonify({"error": "Failed to parse JSON from OpenAI response", "details": str(e)}), 500

    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging output for other errors
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
