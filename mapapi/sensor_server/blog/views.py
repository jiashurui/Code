import json

from django.http import JsonResponse
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt


def hello_world(request):
    return HttpResponse("Hello World")


@csrf_exempt
def send_data(request):
    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body_data = json.loads(body_unicode)

        name = body_data.get('name')
        age = body_data.get('age')


        response_data = {
            'status': 'success',
            'message': 'ok',
            'received_data': {
                'name': name,
                'age': age
            }
        }
        return JsonResponse(response_data)