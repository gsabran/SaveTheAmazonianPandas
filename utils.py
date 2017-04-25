import subprocess
import time
import string

def getUniqName():
	"""
	A uniq name to diffenrenciate generated files
	"""
	gitHash = subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")
	gitMessage =  subprocess.check_output(["git", "log", "--format=%B", "-n", "1", "HEAD"]).strip().decode("utf-8")
	# remove bad filename characters: http://stackoverflow.com/a/295146/2054629
	valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
	gitMessage = ''.join(c for c in gitMessage if c in valid_chars)

	t = time.strftime("%Y-%m-%d.%H:%M:%S")
	return "{t}-{h}-{m}".format(t=t, h=gitHash, m=gitMessage).lower().replace(" ", "_")
