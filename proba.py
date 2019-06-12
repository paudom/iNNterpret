import innterpret as inn
from innterpret.display import print_msg
from keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet',include_top=True)
classOne = './innterpret/ZUTesting/Data/Taurons'
classTwo = './innterpret/ZUTesting/Data/Taurons2'
DR2 = inn.datapoints.DistRobust(model, classOne, classTwo)
result2 = DR2.execute('cosine')
print_msg(str(result2),option='verbose')
print_msg('\n==================')