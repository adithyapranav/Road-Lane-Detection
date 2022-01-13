import cv2
import matplotlib.pyplot as plt
import numpy as np

def grey(image): # convert color image into grescale image
	image=np.asarray(image)
	return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

def guass(image): # perform gaussian blur to reduce noice and smoothen the image as canny is noise sensitive
	return cv2.GaussianBlur(image,(5,5),0) # params img,ksize-kernel dimension,sigma-sd along x axis

def canny(image,x,y): 
	'''# canny edge detection any change in luminocity(black to white vice verse) is an edge'''
	edges=cv2.Canny(image,50,150) 
	'''# params img,threshold-1 filters all gradient lower than this no, threshold-2 determine value for which edge is considered valid'''
	return edges

def region(image): # isolate edges that corresponds with lane lines, inp is canny image
	height,width=image.shape #extract image dim
	triangle=np.array([[(100,height),(475,325),(width,height)]]) #define triangle dim(region to isolate)
	mask= np.zeros_like(image) # create a black plane
	mask=cv2.fillPoly(mask,triangle,255) # define a white triangle
	mask=cv2.bitwise_and(image,mask) # bitwise and which isolates the lanes
	return mask

# hough line transform- turns clusters of white pixels into actual lines	
'''lines = cv2.HoughLinesP(isolated, rho=2, theta=np.pi/180, threshold=100, np.array([]), minLineLength=40, maxLineGap=5) #params 1-ioslated gradient,(2,3)-bin size,4-min intersection needed per bin to be considered a line(100 in our case),5-placeholder array,6-min line length,7-max line gap'''

def average(image,lines): # average the lines
	left=[]
	right=[]
	for line in lines: # loop through array of lines
		print(line) 
		x1,y1,x2,y2=line.reshape(4) # extract the (x,y) values of the 2 points from each line segment
		parameters=np.polyfit((x1,x2),(y1,y2),1)  
		slope=parameters[0] # find slope
		y_int=parameters[1] # find y intersetp
		if slope<0:
			left.append((slope,y_int)) # negative slope so add to left line list
		else:
			right.append((slope,y_int)) # positive slope so add to right line list
	#taking average of slopes and y intercepts from both list
	right_avg= np.average(right, axis=0) #avg of left line segment
	left_avg=np.average(left, axis=0) # avg of right line segment
	left_line=make_points(image,left_avg) # calculate start and end point of left line
	right_line=make_points(image,right_avg) # calculate start and end point of right line
	return np.array([left_line,right_line]) # output the two coordinates
	
def make_points(image,average):
	slope,y_int=average# get average slope and y intercept
	y1=image.shape[0] # define height of lines(same for left and right)
	y2=int(y1*(3/5)) # define height of lines(same for left and right)
	x1=int((y1-y_int)//slope) # calculate x by rearranging y=mx+b to x=(y-b)/m
	x2=int((y2-y_int)//slope)# calculate x by rearranging y=mx+b to x=(y-b)/m 
	return np.array([x1,y1,x2,y2]) #output the coordinate set
	
def display_lines(image,lines):
	lines_image=np.zeros_like(image) #creates a blacked out image with same dim as original image
	if lines is not None: # ensures list with line points arent empty
		for line in lines: # loops throught that list
			x1,y1,x2,y2=line # extracting (x,y)
			cv2.line(lines_image, (x1,y1),(x2,y2),(255,0,0),10) #creates line and pastes it onto blacked out image
	return lines_image #output black image with lines

'''
lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1) 
# adds weight of 0.8 pixel to image making them darker
we then give weight 1 to blacked out image with all lane lines so all pixel there keep same intensity making the line stand out
why make them darker 
as  real image is clourful so making it darker will make it easy to display line
little more efficient'''

#putting all function together
image1=cv2.imread("/home/pes2ug19cs015/Car Lane Detection Project/lane1.jpeg")
copy = np.copy(image1)
grey = grey(copy)
gaus = guass(grey)
edges = canny(gaus,50,150)
isolated = region(edges)
lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average(copy, lines)
black_lines = display_lines(copy, averaged_lines)
lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
cv2.imshow("lanes", lanes)
cv2.waitKey(0)
