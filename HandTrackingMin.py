import cv2
import mediapipe as mp
import time
import csv
import pandas as pd

video_file = "Video/selamat_sore/MP4/selamat_sore6.mp4"
cap = cv2.VideoCapture(video_file)
total_id = 21


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

filename = "selamat_sore6.csv"
text_data = open((filename), "w")
text_data.write('Hasil'+'*')
data=[]

for i in range(total_id):
    data.append(str(i))
    data.append(',')
    data.append(str(i))
    data.append('*')

data.append('\n')
hasil = ''.join(str(e) for e in data)
text_data.write(hasil)
text_data.write('Frame'+'*')
data=[]

for i in range(total_id):
    data.append('x')
    data.append(',')
    data.append('y')
    data.append('*')

data.append('\n')
hasil = ''.join(str(e) for e in data)
text_data.write(hasil)
# with open(filename, 'w') as csvfile:
#     csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

#     csvwriter.writerow(fields)

  #  tes_writer = csv.writer(tes_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

   # tes_writer.writerow(['John Smith', 'Accounting', 'November'])
   # tes_writer.writerow(['Erica Meyers', 'IT', 'March'])

pTime = 0
cTime = 0
prame=0
total_data = pd.DataFrame()
hasil_hasil = []
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    hasil = pd.DataFrame()
    #print(results.multi_hand_landmarks)



    	

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)

                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # data = [cx,cy]

                # df1 = pd.DataFrame(data,
                                   # columns=[str(id)])
                # hasil = hasil.join(df1)
                # text_data.write(str(id) + '\n')
                print(prame)
                # print(id)
                if id==0:
                    text_data.write(str(prame)+'*'+str(cx)+','+str(cy))

                if id>0 and id<20:
                    text_data.write('*' + str(cx) + ',' + str(cy))


                if id==20:
                    text_data.write('*' + str(cx) + ',' + str(cy)+'\n')
                # print(id, cx, cy)
                if id== 0:
                    cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    # hasil_hasil.append(hasil.values)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)


    # df2 =df1 = pd.DataFrame(hasil, rows=[str(prame)])
    # total_data = total_data.join(df2)
    cv2.imshow("Image", img)
    prame +=1
    cv2.waitKey(1)

# total_data.to_excel('coba.xlsx')
text_data.close()

# print(hasil_hasil)
