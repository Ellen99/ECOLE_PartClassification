from openai import OpenAI
from secret import KEY

def filter_relevant_parts_through_api(query_parts_string, new_concept, client):
    # new_concept = "awl tool"
    prompt_to_gpt = f"Which of the following parts are essential for identifying an {new_concept} specifically?\
 Focus on parts that are visually distinctive and most critical to the {new_concept}'s function. Only select the parts that you would use to describe the {new_concept}.\
 If multiple parts seem similar, select the one most specific to most instances of {new_concept} concept. Provide only those for which you are more than 90% sure.\
 Provide the output only as a comma-separated list of those critical less than 8 parts.\n\
    Parts: {{{query_parts_string}}}"

    print("Prompt to GPT: ", prompt_to_gpt)
    print("\n")
    response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt_to_gpt,
        }
    ],
    model="gpt-4o-mini",
    stream=False
    )
    return response.choices[0].message.content

# example usage
# implement main function to test the function
def main():
    ''' Main function to test the function '''
    client = OpenAI(api_key=KEY)
    parts_to_consider = ['head', 'screwdriver', 'long arm', 'spring', 'handle', 'blade', 'point', 'grip', 'shaft', 'chisel']
    parts_to_consider = ', '.join([s.replace("'", "") for s in parts_to_consider])
    result = filter_relevant_parts_through_api(parts_to_consider, "awl tool", client)
    print("API response: ", result)