import collections

Dataset_Info = collections.namedtuple("Dataset_Info", 'data_file categories')
class PATH(object):
	def __init__(self):
		self.FER2013 = Dataset_Info(
			data_file = '/home/ddeng/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/fer2013.csv',
			categories =  ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])

		self.FERPlus = Dataset_Info(
			                        data_file = '/home/ddeng/challenges-in-representation-learning-facial-expression-recognition-challenge/FERPlus-master/fer2013new.csv',
			                        categories =['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
			                        )
