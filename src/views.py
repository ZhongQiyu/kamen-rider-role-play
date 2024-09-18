# views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def agent_a(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        question = data.get('question', '')
        answer = call_agent_b(question)
        return JsonResponse({'answer': answer})

@csrf_exempt
def agent_b(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        question = data.get('question', '')
        answer = generate_answer(question)
        return JsonResponse({'answer': answer})

def call_agent_b(question):
    return "这是对问题的回答"

def generate_answer(question):
    return "这是生成的答案"
