from django.shortcuts import render
import random

# Create your views here.
def index(request):
    context = {
        'name' : 'yusin',
        'for' : [1,2,3,4,5],
    }
    return render(request, 'articles/index.html', context)



def hello(request):
    context = {
        'name' : 'Django!',
    }
    return render(request, 'articles/hello.html', context)


def dinner(request):
    foods = ['국밥', '국수', '카레', '탕수육', '스팸']
    pick = random.choice(foods)
    context = {
        'name' : foods,
        'plate' : pick,
        'plate_len' : len(pick),
    }
    return render(request, 'articles/dinner.html', context)

def search(request):
    context = {
    
    }
    return render(request, 'articles/search.html', context)

def throw(request):

    return render(request, 'articles/throw.html')

def catch(request):
    context = {
        "message" : request.GET.get('message')
    }
    return render(request, 'articles/catch.html', context)


def detail(request, num):
    context = {
        'num' : num
    }
    return render(request, 'articles/detail.html', context)