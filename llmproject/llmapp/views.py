from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from . import clip_vectorization,searchSimilarPaper
import json
import re
from django.views.decorators.csrf import csrf_exempt
from groq import Groq
from django.conf import settings
import os
# Create your views here.

def home(request):
    return render(request,'home.html',{"message" : "Welcome to your page!!"})

groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
client = Groq(api_key=groq_api_key) if groq_api_key else None


def _fallback_answer(query: str, context: str) -> str:
    """Return a concise, readable answer when the hosted LLM is unavailable."""
    context = context.strip()
    if not context:
        return (
            "I could not find matching paper content for that query yet. "
            "Try a more specific question or add the Groq API key to enable LLM answers."
        )

    cleaned_context = re.sub(r"\n{3,}", "\n\n", context)
    return (
        f"Based on the retrieved paper content, here is the most relevant context for: {query}\n\n"
        f"{cleaned_context[:3000]}"
    )


def _extract_query_and_context(prompt: str) -> tuple[str, str]:
    query_marker = "|| User Query:"
    context_marker = "Fetched From Database :"

    if query_marker in prompt and prompt.startswith(context_marker):
        context_part, query_part = prompt.split(query_marker, 1)
        context = context_part.replace(context_marker, "", 1).strip()
        query = query_part.strip()
        return query, context

    return prompt, prompt

@csrf_exempt
def getDataFromOpenAIAPI(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            query = data.get("query", "")
            answer_query, answer_context = _extract_query_and_context(query)

            if client is None:
                return JsonResponse({"response": _fallback_answer(answer_query, answer_context)})

            completion = client.chat.completions.create(
                    model=groq_model,
                    messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a PDF RAG assistant. You will assist users by responding to their queries "
                            "using the relevant data fetched from external databases. Provide accurate and concise answers. "
                        )
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                    temperature=1,
                    max_tokens=1024,
                    top_p=1,
                    stream=True,
                    stop=None,
                )
            ans = ""
            for chunk in completion:
                ans += chunk.choices[0].delta.content or ""
            if not ans.strip():
                ans = _fallback_answer(answer_query, answer_context)
            return JsonResponse({'response': ans})
        except Exception as e:
            data = {}
            try:
                data = json.loads(request.body)
            except Exception:
                pass
            query = data.get("query", "") if isinstance(data, dict) else ""
            answer_query, answer_context = _extract_query_and_context(query)
            return JsonResponse({'response': _fallback_answer(answer_query, answer_context)})
    return JsonResponse({'error': 'Invalid request method.'}, status=400)

@csrf_exempt
def uploadFile(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # Retrieve the uploaded file
        uploaded_file = request.FILES['image']
        # Define the local path to save the image
        save_dir = os.path.join(settings.MEDIA_ROOT, 'images')
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        file_path = os.path.join(save_dir, uploaded_file.name)

        try:
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            return JsonResponse({'message': 'Image uploaded successfully', 'file_path': file_path})
        except Exception as e:
            return JsonResponse({'error': f'Failed to save image: {str(e)}'}, status=500)
    else:
        return JsonResponse({'error': 'No image file provided'}, status=400)
    
@csrf_exempt
def getEmbedding(request):
    if request.method == 'POST':
        try:
            textEmbedding = []
            imageEmbedding = []
            data = json.loads(request.body)
            type = data.get("type", "")
            print(type)
            if type == "text":
                text = data.get("text","")
                textEmbedding = clip_vectorization.vectorize_text(text)
                return JsonResponse({'response': textEmbedding})
            else:
                imagePath = data.get("imageFilePath","")
                imageEmbedding = clip_vectorization.vectorize_image(imagePath)
                return JsonResponse({'response': imageEmbedding})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method.'}, status=400)


@csrf_exempt
def getSimilarContent(request):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            query_embedding = data.get("embedding", None)
            print(query_embedding)
            if query_embedding is None:
                return JsonResponse({'error': 'Embedding is required.'}, status=400)

            # Assuming the embeddings are stored somewhere (e.g., in a database or a file)
            # and we have a function to find the most similar content based on the embedding
            similar_content = searchSimilarPaper.search_similar_papers(query_embedding)

            # Assuming the find_similar_content_by_embedding returns a list of similar content in plain text
            return JsonResponse({'response': similar_content})
        
        except Exception as e:
            print(f"Error in getSimilarContent: {e}")
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=400)
