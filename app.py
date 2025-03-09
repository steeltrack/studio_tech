# TODO Store retrieved documents in one array, then rerank collection and truncate to top 20
# TODO Ask claude if the existing documents can answer the question, if not request document retrieval

import re
import chainlit as cl
import anthropic
import voyageai
import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter, HybridFusion, Metrics


SYSTEM_PROMPT = "You are a specialized studio assistant for a busy music producer. Your primary purpose is to provide accurate, concise technical information to maximize the producer's efficiency in the studio. You have access to a RAG (Retrieval-Augmented Generation) system containing a comprehensive index of technical manuals for available studio equipment."
INITIAL_PROMPT = """
You can expect retrieved documents to be located in <retrieved_documents> tags and the user's query to be located in <user_query> tags.

Core Responsibilities:

1. Answer technical questions about studio equipment using the retrieved manual excerpts
2. Use your general knowledge and, if they're relevant, the retrieved documents, to answer questions about music theory and studio tasks like mixing and mastering.
3. Provide troubleshooting assistance based on technical documentation
4. Suggest optimal equipment settings and configurations
5. Offer workflow tips to improve productivity
6. Translate technical jargon into clear, actionable instructions

Interaction Guidelines:

- Keep responses brief and focused on the immediate need
- Prioritize actionable information over theoretical explanations
- Acknowledge when information is incomplete or unclear in the retrieved documents
- Use appropriate technical terminology but explain it when necessary
- Format responses for quick scanning (concise paragraphs, occasional bullet points)
- Include exact page/section references from manuals when relevant
- When suggesting alternatives, focus only on what's feasible with the existing equipment

Response Structure:

- Direct answer to the question (1-2 sentences)
- Supporting details from relevant manual(s)
- Practical next steps or troubleshooting sequence (when applicable)
- Optional: Quick tip for improved workflow

Remember, you can expect retrieved documents to be located in <retrieved_documents> tags and the user's query in <user_query> tags.
"""

TURN_TEMPLATE = """
For this turn of the conversation, the following documents have been retrieved from a RAG (Retrieval-Augmented Generation) system:

<retrieved_documents>
{RETRIEVED_DOCUMENTS}
</retrieved_documents>

Here is the user's query for this turn of the conversation:

<user_query>
{USER_QUERY}
</user_query>
"""

# Set up needed clients
claude_client = anthropic.AsyncAnthropic()
weaviate_client = weaviate.connect_to_local()
embedding_client = voyageai.Client()

def extract_tag_values(text, tag_name):
    """
    Extract values from XML-like tags in a text.
    
    Args:
        text (str): Text to search in
        tag_name (str): Name of the tag to extract from
        
    Returns:
        list: Extracted values or empty list if no match or value is "none"
    """
    pattern = f'<{tag_name}>(.*?)</{tag_name}>'
    match = re.search(pattern, text, re.DOTALL)
    
    if not match:
        return []
    
    match_values = match.group(1).strip()
    
    if match_values.lower() == "none":
        return []
    
    # Split by lines and filter out empty strings
    return [line.strip() for line in match_values.splitlines() if line.strip()]

async def get_filters(query):

    weaviate_client = weaviate.connect_to_local()
    collection = weaviate_client.collections.get("Manuals")

    brand_response = collection.aggregate.over_all(
        return_metrics=Metrics("brand").text(
            top_occurrences_count=True,
            top_occurrences_value=True,
            min_occurrences=5
        )
    )

    brands = []
    for occurence in brand_response.properties['brand'].top_occurrences:
        brands.append(occurence.value)

    model_response = collection.aggregate.over_all(
        return_metrics=Metrics("model").text(
            top_occurrences_count=True,
            top_occurrences_value=True,
            min_occurrences=5
        )
    )

    models = []
    for occurence in model_response.properties['model'].top_occurrences:
        models.append(occurence.value)

    anthropic_client = anthropic.Client()

    MODEL_CLASSIFIER_PROMPT = """
    You will be given a list of brands and models, followed by a user's query. Your task is to determine if the user's query contains mentions of any of the brands or models from the list. Exact matches are not necessary; you should look for close matches or variations as well. Consider common misspellings, abbreviations, or partial matches.

    First, here is the list of brands and models:
    <brands>
    {BRANDS}
    </brands>

    <models>
    {MODELS}
    </models>

    Now, here is the user's query:
    <user_query>
    {USER_QUERY}
    </user_query>

    Provide your response in the following format:

    <brands>
    List the matched brands here, one per line. If no matches were found, write "none"
    </brands>

    <models>
    List the matched models here, one per line. If no matches were found, write "none"
    </models>
    """

    llm_response = anthropic_client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=1024,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": MODEL_CLASSIFIER_PROMPT.format(BRANDS="\n".join(brands), MODELS="\n".join(models), USER_QUERY=query)
                    }
                ]
            }
        ]
    )

    filters = {"brands": [], "models": []}

    brands = extract_tag_values(llm_response.content[0].text, "brands")
    if brands:
        filters["brands"].extend(brands)

    models = extract_tag_values(llm_response.content[0].text, "models")
    if models:
        filters["models"].extend(models)

    return filters

# Get studio documentation, passing property values as filters
async def get_documentation(query = "", filters = {"brands": [], "models": []}):

    query_embeddings = embedding_client.embed(
        [query],
        model="voyage-3",
        input_type="query"
    ).embeddings[0]

    collection = weaviate_client.collections.get("Manuals")

    filterset = []

    if len(filters["brands"]) > 0:
        filterset.append(Filter.by_property("brand").contains_any(filters["brands"]))
    if len(filters["models"]) > 0:
        filterset.append(Filter.by_property("model").contains_any(filters["models"]))

    if filterset:
        response = collection.query.hybrid(
            query=query,
            filters=(
                Filter.any_of(filterset)
            ),
            vector=query_embeddings,
            limit=5,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            return_metadata=wvc.query.MetadataQuery(certainty=True)
        )
    else:
        response = collection.query.hybrid(
            query=query,
            vector=query_embeddings,
            limit=5,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            return_metadata=wvc.query.MetadataQuery(certainty=True)
        )

    documents = {}
    for object in response.objects:
        documents[object.uuid] = object.properties['content']

    return documents

# Call to Claude
async def call_claude(query: str, documents = {}):

    messages = cl.user_session.get("messages")

    messages.append({"role": "user", "content": TURN_TEMPLATE.format(RETRIEVED_DOCUMENTS="\n".join(documents.values()), USER_QUERY=query)})

    response = cl.Message(content="", author="Claude")

    stream = await claude_client.messages.create(
        model="claude-3-7-sonnet-latest",
        system=SYSTEM_PROMPT,
        messages=messages,
        max_tokens=8192,
        stream=True
    )

    async for data in stream:
        if data.type == "content_block_delta":
            await response.stream_token(data.delta.text)

    await response.send()
    messages.append({"role": "assistant", "content": response.content})
    cl.user_session.set("messages", messages)

# Start up session
@cl.on_chat_start
async def start_chat():

    start_messages = []
    start_messages.append({"role": "user", "content": INITIAL_PROMPT})
    start_messages.append({"role": "assistant", "content": "Understood"})

    cl.user_session.set("messages", start_messages)
    cl.user_session.set("entities", {"brands": [], "models": []})

# Send message to Claude
@cl.on_message
async def chat(message: cl.Message):

    # Only query documentation if the session has mentioned specific entities
    filters = await get_filters(message.content)
    entities = cl.user_session.get("entities")
    entities["brands"].extend(filters["brands"])
    entities["models"].extend(filters["models"])
    cl.user_session.set("entities", entities)

    if entities["brands"] or entities["models"]:
        documents = await get_documentation(message.content, entities)
    else:
        documents = {}

    # Send a response back to the user
    await call_claude(message.content, documents)

# Enable debugger by running when file is called directly
if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
