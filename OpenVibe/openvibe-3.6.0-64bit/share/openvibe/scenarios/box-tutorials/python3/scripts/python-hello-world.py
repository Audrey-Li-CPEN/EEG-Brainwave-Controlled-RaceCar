
class MyOVBox(OVBox):
	def __init__(self):
		OVBox.__init__(self)
		
	def initialize(self):
		print('Initialize Hello World Script') 
		# nop
		return
		
	def process(self):
		# print the string specified in the box configuration. 
		# 'Message' is the name of the config entry.
		print(self.setting['Message']) 
		
	def uninitialize(self):
		print('Uninitialize Hello World Script') 
		# nop
		return

box = MyOVBox()
