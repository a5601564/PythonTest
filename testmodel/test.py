from urllib import urlopen
data = urlopen('http://peak.telecommunity.com/dist/ez_setup.py')
open('ez_setup.py','wb').write(data.read())
exit