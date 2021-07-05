from django.shortcuts import render,redirect
from django.contrib import messages
from django.contrib import auth
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from IPSproject.settings import BASE_DIR
from .models import Account, Diagnosis 
from konlpy.tag import Okt
from konlpy.tag import Kkma

import sys
import os
import pandas as pd
import numpy as np

# Create your views here.


import csv
from django.http import HttpResponse
from django.template import loader
import json
import requests

# 얼굴인식
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np   
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import pandas as pd

from bs4 import BeautifulSoup



# Create your views here.

def face(request):
    return render(request, 'facegame.html')

def face_1(request):

    # Face detection XML load and trained model loading
    face_detection = cv2.CascadeClassifier('/Users/serinkim/Desktop/IPS_django/IPSproject/bloomingapp/haarcascade_frontalface_default.xml')
    emotion_classifier = load_model('/Users/serinkim/Desktop/IPS_django/IPSproject/bloomingapp/emotion_model.hdf5', compile=False)
    EMOTIONS = ["Angry" ,"Disgusting","Fearful", "Happy", "Sad", "Surpring", "Neutral"]

    # Video capture using webcam
    camera = cv2.VideoCapture(0)

    while True:

        # Capture image from camera
        ret, frame = camera.read() # 카메라 상태 및 프레임
        
        # Convert color to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face detection in frame
        faces = face_detection.detectMultiScale(gray, # 검출하고자 하는 원본 이미지
                                                scaleFactor=1.1, # 검색 윈도우 확대 비율
                                                minNeighbors=5, # 얼굴 사이 최소 간격(픽셀)
                                                minSize=(30,30)) # 얼굴 최소 크기
        
        # Create empty image
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        
        # Perform emotion recognition only when face is detected
        if len(faces) > 0:
            # For the largest image
            face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = face
            # Resize the image to 48x48 for neural network
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            # Emotion predict
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            
            # Assign labeling
            cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
    
            # Label printing
            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)    
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        # Open two windows
        ## Display image ("Emotion Recognition")
        ## Display probabilities of emotion
        cv2.imshow('Emotion_Recognition', frame)
        cv2.imshow("Probabilities", canvas)
        
        # q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Clear program and close windows
    camera.release()
    cv2.destroyAllWindows()

    return render(request, 'facegame.html')



def apiapi(request): 

    file_path = os.path.join(BASE_DIR, 'bloomingapp/seoulgil.csv')
    csv = pd.read_csv(file_path, encoding = 'cp949')

    name = csv['코스명']
    name = name.values
    name = name.tolist()

    xx = csv['X 좌표']
    xx = xx.values
    xx = xx.tolist()

    yy = csv['Y 좌표']
    yy = yy.values
    yy = yy.tolist()


    context = {
        'name' : name,
        'xx' : xx,
        'yy' : yy
    }
    
    return HttpResponse(render(request, 'apiapi.html', context))



def home(request):
    return render(request, 'home.html')

def signup(request):
    context = {
        'error': {
            'state': False,
            'msg': ''
        }
    }
    if request.method == 'POST':
        
        user_id = request.POST['user_id']
        user_pw = request.POST['user_pw']
        user_pw_check = request.POST['user_pw_check']
        # add
        user_name = request.POST['name']
        user_email = request.POST['email']
        user_gender = request.POST['gender']
        user_age = request.POST['age']

        if (user_id and user_pw):
            user = User.objects.filter(username=user_id)
            if len(user) == 0:
                if (user_pw == user_pw_check):
                    created_user = User.objects.create_user(
                        username=user_id,
                        password=user_pw
                    )
                    # add
                    Account.objects.create(
                        user = created_user,
                        name = user_name,
                        email = user_email,
                        gender = user_gender,
                        age = user_age,
                    )

                    auth.login(request, created_user)
                    return redirect('home')
                else:
                    context['error']['state'] = True
                    context['error']['msg'] = ERROR_MSG['PW_CHECK']
            else:
                context['error']['state'] = True
                context['error']['msg'] = ERROR_MSG['ID_EXIST']
        else:
            context['error']['state'] = True
            context['error']['msg'] = ERROR_MSG['ID_PW_MISSING']

    return render(request, 'signup.html', context)

def que1(request, user_pk):
    Users = User.objects.get(pk=user_pk)
    
    if request.POST:
        diag = Diagnosis()
        diag.diagnosis = request.user
        diag.image = request.FILES['images']
        diag.save()
        
        context = {
            'user': Users,
            'data': diag
        }
        return render(request, 'que1.html', context)
        
    context = {
        'user': Users,
    }

    return render(request, 'que1.html', context)

def que2(request, user_pk, dia_pk):
    Users = User.objects.get(pk=user_pk)
    if request.POST:
        diag = Diagnosis.objects.get(pk=dia_pk)
        diag.selection = request.POST['selection']
        diag.save()
        
        context = {
            'user': Users,
            'data': diag
        }
        return render(request, 'que2.html', context)
        
    context = {
        'user': Users,
        'data': diag
    }

    return render(request, 'que2.html', context)

def que3(request, user_pk, dia_pk):
    Users = User.objects.get(pk=user_pk)
    if request.POST:
        diag = Diagnosis.objects.get(pk=dia_pk)
        diag.doing = request.POST['doing']
        diag.feel = request.POST['feeling']
        diag.save()
        
        context = {
            'user': Users,
            'data': diag
        }
        return render(request, 'que3.html', context)
        
    context = {
        'user': Users,
        'data': diag
    }
    return render(request, 'que3.html', context)

def que4(request, user_pk, dia_pk):
    Users = User.objects.get(pk=user_pk)
    if request.POST:
        diag = Diagnosis.objects.get(pk=dia_pk)
        diag.tree_state = request.POST['tree_state']
        diag.tree_forward = request.POST['tree_forward']
        diag.save()
        
        context = {
            'user': Users,
            'data': diag
        }
        return render(request, 'que4.html', context)
        
    context = {
        'user': Users,
        'data': diag
    }
    return render(request, 'que4.html', context)

def que5(request, user_pk, dia_pk):
    Users = User.objects.get(pk=user_pk)
    
    if request.POST:
        diag = Diagnosis.objects.get(pk=dia_pk)
        diag.home_mood = request.POST['home_mood']
        diag.home_door = request.POST['home_door']
        diag.save()
        
        context = {
            'user': Users,
            'data': diag
        }
        return render(request, 'que5.html', context)
        
    context = {
        'user': Users,
        'data': diag
    }

    return render(request, 'que5.html', context)

def result(request, user_pk, dia_pk):
    Users = User.objects.get(pk=user_pk)
    
    if request.POST:
        diag = Diagnosis.objects.get(pk=dia_pk)
        diag.last_q = request.POST['last_q']
        diag.save()

        f_draw = diag.selection
        doing = diag.doing
        feel = diag.feel
        tree_state = diag.tree_state
        tree_forward = diag.tree_forward
        home_mood = diag.home_mood
        home_door = diag.home_door
        last_q = diag.last_q
        

        file_path = os.path.join(BASE_DIR, 'bloomingapp/sentiment.csv')
        sentiment = pd.read_csv(file_path)
        POS = sentiment['value'] == 'POS'
        NEG = sentiment['value'] == 'NEG'
        is_POS = sentiment[POS]
        is_NEG = sentiment[NEG]
        POS_words = is_POS['word'].values.tolist()
        NEG_words = is_NEG['word'].values.tolist()

        kkma = Kkma()
        morphs = []
        pos = kkma.pos(doing) 
        # print('kkma형태소 분석 :  ', pos)
        for i in pos: # ['JKS','JKC','JKG','JKO','JKM','JKI','JKQ','JC','EPH','EPT','EPP','EFN','EFQ','EFO','EFA','EFI','EFR','ECE','ECD','ECS','ETN','ETD','XPN','XPV','XSN','XSV','XSA','SF','SP','SS','SE','SO','SW']
            if i[1] not in ['JKS','JKC','JKG','JKO','JKM','JKI','JKQ','JC','EPH','EPT','EPP','EFN','EFQ','EFO','EFA','EFI','EFR','ECE','ECD','ECS','ETN','ETD','XPN','XPV','XSN','XSV','XSA','SF','SP','SS','SE','SO','SW']:
                morphs.append(i[0])
            else:
                pass
        total = len(morphs)

        POS_score = 0
        count = 0
        match_list = []
        for i in range(0,total):
            if morphs[i] in POS_words:
                POS_score += 1
                count += 1
                match_list.append(morphs[i])
            elif morphs[i] in NEG_words:
                POS_score += -1
                count += 1
                match_list.append(morphs[i])
            else:
                pass
        print('POS_score :   ',POS_score)
        print(total, count)
        print('match_list :  ', match_list)


        file_path = os.path.join(BASE_DIR, 'bloomingapp/emotion_dic.xlsx')
        emotion_dic = pd.read_excel(file_path)

        score_1 = emotion_dic['score'] == 1
        words_1 = emotion_dic[score_1]['word'].values.tolist()
        score_2 = emotion_dic['score'] == 2
        words_2 = emotion_dic[score_2]['word'].values.tolist()
        score_4 = emotion_dic['score'] == 4
        words_4 = emotion_dic[score_4]['word'].values.tolist()
        score_5 = emotion_dic['score'] == 5
        words_5 = emotion_dic[score_5]['word'].values.tolist()

        okt = Okt()
        pos = okt.pos(doing, stem=True)
        morphs = []
        for i in range(0, len(pos)):
            morphs.append(pos[i][0])
        # print('morphs :  ',morphs)

        if POS_score>0:
            count_4 = 0
            count_5 = 0
            for i in range(0, len(morphs)):
                if morphs[i] in words_4:
                    count_4 += 1
                elif morphs[i] in words_5:
                    count_5 += 1
                else:
                    pass
            # print('긍 : ', count_4, count_5)
            if count_4 < count_5:
                doing_score = 5
            else:
                doing_score = 4

        elif POS_score<0:
            count_1 = 0
            count_2 = 0
            for i in range(0, len(morphs)):
                if morphs[i] in words_1:
                    count_1 += 1
                elif morphs[i] in words_2:
                    count_2 += 1
                else:
                    pass
            # print('부 : ', count_1, count_2)
            if count_2 < count_1:
                doing_score = 1
            else:
                doing_score = 2     
        else:
            doing_score = 3  

        print("사람 1 우울 점수 ", doing_score)
        
        # -------doing 끝---------

        # tree_state

        morphs = []
        pos = kkma.pos(tree_state) 
        # print('kkma형태소 분석 :  ', pos)
        for i in pos: # ['JKS','JKC','JKG','JKO','JKM','JKI','JKQ','JC','EPH','EPT','EPP','EFN','EFQ','EFO','EFA','EFI','EFR','ECE','ECD','ECS','ETN','ETD','XPN','XPV','XSN','XSV','XSA','SF','SP','SS','SE','SO','SW']
            if i[1] not in ['JKS','JKC','JKG','JKO','JKM','JKI','JKQ','JC','EPH','EPT','EPP','EFN','EFQ','EFO','EFA','EFI','EFR','ECE','ECD','ECS','ETN','ETD','XPN','XPV','XSN','XSV','XSA','SF','SP','SS','SE','SO','SW']:
                morphs.append(i[0])
            else:
                pass
        total = len(morphs)

        POS_score = 0
        count = 0
        match_list = []
        for i in range(0,total):
            if morphs[i] in POS_words:
                POS_score += 1
                count += 1
                match_list.append(morphs[i])
            elif morphs[i] in NEG_words:
                POS_score += -1
                count += 1
                match_list.append(morphs[i])
            else:
                pass
        #print('POS_score :   ',POS_score)
        #print(total, count)
        #print('match_list :  ', match_list)


        pos = okt.pos(tree_state, stem=True)
        morphs = []
        for i in range(0, len(pos)):
            morphs.append(pos[i][0])
        # print('morphs :  ',morphs)

        if POS_score>0:
            count_4 = 0
            count_5 = 0
            for i in range(0, len(morphs)):
                if morphs[i] in words_4:
                    count_4 += 1
                elif morphs[i] in words_5:
                    count_5 += 1
                else:
                    pass
            # print('긍 : ', count_4, count_5)
            if count_4 < count_5:
                tstate_score = 5
            else:
                tstate_score = 4

        elif POS_score<0:
            count_1 = 0
            count_2 = 0
            for i in range(0, len(morphs)):
                if morphs[i] in words_1:
                    count_1 += 1
                elif morphs[i] in words_2:
                    count_2 += 1
                else:
                    pass
            # print('부 : ', count_1, count_2)
            if count_2 < count_1:
                tstate_score = 1
            else:
                tstate_score = 2     
        else:
            tstate_score = 3  

        print("tree_state 우울 점수 ", tstate_score)

        # tree_forward
        morphs = []
        pos = kkma.pos(tree_forward) 
        # print('kkma형태소 분석 :  ', pos)
        for i in pos: # ['JKS','JKC','JKG','JKO','JKM','JKI','JKQ','JC','EPH','EPT','EPP','EFN','EFQ','EFO','EFA','EFI','EFR','ECE','ECD','ECS','ETN','ETD','XPN','XPV','XSN','XSV','XSA','SF','SP','SS','SE','SO','SW']
            if i[1] not in ['JKS','JKC','JKG','JKO','JKM','JKI','JKQ','JC','EPH','EPT','EPP','EFN','EFQ','EFO','EFA','EFI','EFR','ECE','ECD','ECS','ETN','ETD','XPN','XPV','XSN','XSV','XSA','SF','SP','SS','SE','SO','SW']:
                morphs.append(i[0])
            else:
                pass
        total = len(morphs)

        POS_score = 0
        count = 0
        match_list = []
        for i in range(0,total):
            if morphs[i] in POS_words:
                POS_score += 1
                count += 1
                match_list.append(morphs[i])
            elif morphs[i] in NEG_words:
                POS_score += -1
                count += 1
                match_list.append(morphs[i])
            else:
                pass
        #print('POS_score :   ',POS_score)
        #print(total, count)
        #print('match_list :  ', match_list)


        pos = okt.pos(tree_forward, stem=True)
        morphs = []
        for i in range(0, len(pos)):
            morphs.append(pos[i][0])
        # print('morphs :  ',morphs)

        if POS_score>0:
            count_4 = 0
            count_5 = 0
            for i in range(0, len(morphs)):
                if morphs[i] in words_4:
                    count_4 += 1
                elif morphs[i] in words_5:
                    count_5 += 1
                else:
                    pass
            # print('긍 : ', count_4, count_5)
            if count_4 < count_5:
                tforward_score = 5
            else:
                tforward_score = 4

        elif POS_score<0:
            count_1 = 0
            count_2 = 0
            for i in range(0, len(morphs)):
                if morphs[i] in words_1:
                    count_1 += 1
                elif morphs[i] in words_2:
                    count_2 += 1
                else:
                    pass
            # print('부 : ', count_1, count_2)
            if count_2 < count_1:
                tforward_score = 1
            else:
                tforward_score = 2     
        else:
            tforward_score = 3  

        print("tree_forward 우울 점수 ", tforward_score)

        # home_mood
        morphs = []
        pos = kkma.pos(home_mood) 
        # print('kkma형태소 분석 :  ', pos)
        for i in pos: # ['JKS','JKC','JKG','JKO','JKM','JKI','JKQ','JC','EPH','EPT','EPP','EFN','EFQ','EFO','EFA','EFI','EFR','ECE','ECD','ECS','ETN','ETD','XPN','XPV','XSN','XSV','XSA','SF','SP','SS','SE','SO','SW']
            if i[1] not in ['JKS','JKC','JKG','JKO','JKM','JKI','JKQ','JC','EPH','EPT','EPP','EFN','EFQ','EFO','EFA','EFI','EFR','ECE','ECD','ECS','ETN','ETD','XPN','XPV','XSN','XSV','XSA','SF','SP','SS','SE','SO','SW']:
                morphs.append(i[0])
            else:
                pass
        total = len(morphs)

        POS_score = 0
        count = 0
        match_list = []
        for i in range(0,total):
            if morphs[i] in POS_words:
                POS_score += 1
                count += 1
                match_list.append(morphs[i])
            elif morphs[i] in NEG_words:
                POS_score += -1
                count += 1
                match_list.append(morphs[i])
            else:
                pass
        #print('POS_score :   ',POS_score)
        #print(total, count)
        #print('match_list :  ', match_list)


        pos = okt.pos(home_mood, stem=True)
        morphs = []
        for i in range(0, len(pos)):
            morphs.append(pos[i][0])
        # print('morphs :  ',morphs)

        if POS_score>0:
            count_4 = 0
            count_5 = 0
            for i in range(0, len(morphs)):
                if morphs[i] in words_4:
                    count_4 += 1
                elif morphs[i] in words_5:
                    count_5 += 1
                else:
                    pass
            # print('긍 : ', count_4, count_5)
            if count_4 < count_5:
                hmood_score = 5
            else:
                hmood_score = 4

        elif POS_score<0:
            count_1 = 0
            count_2 = 0
            for i in range(0, len(morphs)):
                if morphs[i] in words_1:
                    count_1 += 1
                elif morphs[i] in words_2:
                    count_2 += 1
                else:
                    pass
            # print('부 : ', count_1, count_2)
            if count_2 < count_1:
                hmood_score = 1
            else:
                hmood_score = 2     
        else:
            hmood_score = 3  

        print("home_mood 우울 점수 ", hmood_score)


        # last_q
        morphs = []
        pos = kkma.pos(last_q) 
        # print('kkma형태소 분석 :  ', pos)
        for i in pos: # ['JKS','JKC','JKG','JKO','JKM','JKI','JKQ','JC','EPH','EPT','EPP','EFN','EFQ','EFO','EFA','EFI','EFR','ECE','ECD','ECS','ETN','ETD','XPN','XPV','XSN','XSV','XSA','SF','SP','SS','SE','SO','SW']
            if i[1] not in ['JKS','JKC','JKG','JKO','JKM','JKI','JKQ','JC','EPH','EPT','EPP','EFN','EFQ','EFO','EFA','EFI','EFR','ECE','ECD','ECS','ETN','ETD','XPN','XPV','XSN','XSV','XSA','SF','SP','SS','SE','SO','SW']:
                morphs.append(i[0])
            else:
                pass
        total = len(morphs)

        POS_score = 0
        count = 0
        match_list = []
        for i in range(0,total):
            if morphs[i] in POS_words:
                POS_score += 1
                count += 1
                match_list.append(morphs[i])
            elif morphs[i] in NEG_words:
                POS_score += -1
                count += 1
                match_list.append(morphs[i])
            else:
                pass
        #print('POS_score :   ',POS_score)
        #print(total, count)
        #print('match_list :  ', match_list)


        pos = okt.pos(last_q, stem=True)
        morphs = []
        for i in range(0, len(pos)):
            morphs.append(pos[i][0])
        # print('morphs :  ',morphs)

        if POS_score>0:
            count_4 = 0
            count_5 = 0
            for i in range(0, len(morphs)):
                if morphs[i] in words_4:
                    count_4 += 1
                elif morphs[i] in words_5:
                    count_5 += 1
                else:
                    pass
            # print('긍 : ', count_4, count_5)
            if count_4 < count_5:
                lastq_score = 5
            else:
                lastq_score = 4

        elif POS_score<0:
            count_1 = 0
            count_2 = 0
            for i in range(0, len(morphs)):
                if morphs[i] in words_1:
                    count_1 += 1
                elif morphs[i] in words_2:
                    count_2 += 1
                else:
                    pass
            # print('부 : ', count_1, count_2)
            if count_2 < count_1:
                lastq_score = 1
            else:
                lastq_score = 2     
        else:
            lastq_score = 3  

        print("last_q 우울 점수 ", lastq_score)

        score = (doing_score + feel + tstate_score + tforward_score + hmood_score +  lastq_score)/7
        score = round(score, 2)

        if (score <= 1) : level = "first_level"
        elif (score <= 2) : level = "second_level"
        elif (score <= 3) : level = "third_level"
        elif (score <= 4) : level = "fourth_level"
        elif (score <= 5) : level = "fifth_level"
             
       
        print("score = ", score)
        print("level = ", level)


        
        context = {
            'user': Users,
            'data': diag,
            'f_draw' : f_draw,
            'score' : score,
            'level' : level
        }
        return render(request, 'result.html', context)
        
    context = {
        'user': Users,
        'data': diag,
        'f_draw' : f_draw,
    }

    return render(request, 'result.html', context)


def draw(request, user_pk):
    Users = User.objects.get(pk=user_pk)
    context = {
        'user': Users,
    }
    return render(request, 'draw.html', context)

def diagnosis_intro(request, user_pk):
    Users = User.objects.get(pk=user_pk)

    context = {
        'user': Users
    }

    return render(request, 'diagnosis_intro.html', context)

def service_intro(request, user_pk):
    Users = User.objects.get(pk=user_pk)

    context = {
        'user': Users
    }

    return render(request, 'service_intro.html', context)

def login(request):
    context = {
        'error': {
            'state': False,
            'msg': ''
        },
    }
    if request.method == 'POST':
        user_id = request.POST['user_id']
        user_pw = request.POST['user_pw']

        user = User.objects.filter(username=user_id)
        if (user_id and user_pw):
            if len(user) != 0:
                user = auth.authenticate(
                    username=user_id,
                    password=user_pw
                )
                if user != None:
                    auth.login(request, user)

                    return redirect('home')
                else:
                    context['error']['state'] = True
                    context['error']['msg'] = ERROR_MSG['PW_CHECK']
            else:
                context['error']['state'] = True
                context['error']['msg'] = ERROR_MSG['ID_NOT_EXIST']
        else:
            context['error']['state'] = True
            context['error']['msg'] = ERROR_MSG['ID_PW_MISSING']

    return render(request, 'login.html', context)

def mypage(request):
    return render(request, 'mypage.html')

def logout(request):
    auth.logout(request)
    return redirect('home')

def music(request, user_pk):
    Users = User.objects.get(pk = user_pk)
    
    context = {
        'user' : Users,
        
    }
    return render(request, 'music.html', context)

def music_1(request, user_pk):
    Users = User.objects.get(pk = user_pk)

    # 크롤링
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Trident/7.0; rv:11.0) like Gecko'}
    response = requests.get('https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=444702404', headers=header)  # 멜론차트는 헤더정보를 입력해줘야함

    html = response.text
    soup = BeautifulSoup(html, 'html.parser')

    img= soup.find_all("img", {"width":"60"})
    titles = soup.find_all("div", {"class": "ellipsis rank01"})
    singers = soup.find_all("div", {"class": "ellipsis rank02"})

    title = []
    singer = []
    imgURL = []

    for i in titles:
        title.append(i.find('a').text)

    for j in singers:
        singer.append(j.find('a').text)

    for k in img:
        imgURL.append(k.get('src'))
    music_data = [title, singer, imgURL]
    print(type(title))
    print(imgURL[1])
    # len = len(title)
    context = {
        'user' : Users,
        'title' : title,
        'singer' : singer,
        'imgURL' : imgURL,
    }
    return render(request, 'music_1.html', context)