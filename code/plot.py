import gmplot
from gmplot import GoogleMapPlotter as gmp
import gen
from sklearn.externals import joblib
from collections import Counter
from sklearn.preprocessing import minmax_scale

clf = joblib.load('classifier.pkl')

cameras = ['116.avi','117.avi','118.avi']
traffic = []

for c in cameras:
    count, density = gen.generate_features(c)
    traffic_stat = clf.predict([count,density])
    if traffic_stat == 2:
        print(c, ' : High traffic')
    elif traffic_stat == 1:
        print(c,' : Medium traffic')
    else:
        print(c,' : Low traffic')
    traffic.append(traffic_stat)

count = [traffic.count(0), traffic.count(1), traffic.count(2)]
status = count.index(max(count))

color = ['green','yellow','red']
gmap = gmp.from_geocode("Veermata Jijabai Technological Institute, Mumbai, Maharashtra")
gmap.scatter([19.018892, 19.024714], [72.855786, 72.856964], color[status], size=40, marker=False)
gmap.scatter([19.019693, 19.020433, 19.021336, 19.022432, 19.024055,19.023030], [72.856173, 72.856409, 72.856570, 72.856763, 72.857053, 72.856914], color[status], size=10, marker=False)



##this will generate html file of google map
gmap.draw("map.html")


