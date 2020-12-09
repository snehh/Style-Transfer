#Broad categories will be taken as content images

from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
from GoogleImageScraper import GoogleImageScraper
import os

# If no. of images you want to get is 50 then you will have to enter 100 as num of images all the images wont get downloaded due to various issues i.e permissions,etc

content=[['redbull f1 car','lamborghini','ferrari','bugatti'],['lebron','virat kohli','ronaldo','lewis hamliton'],['museum italy','museum india','museum australia','museum africa']]
directory = ['cars','players','museum']

for i in range(0,3):
	os.mkdir(os.getcwd()+"\\dataset\\images\\content_images\\"+directory[i])
	for j in range(0,len(content[i])):
		webdriver_path = os.getcwd() + "\\webdriver\\chromedriver.exe"
		image_path = os.getcwd()+"\\dataset\\images\\content_images\\"+directory[i]
		search_key = content[i][j]
		num_of_images = 100
		image_scrapper = GoogleImageScraper(webdriver_path,image_path,search_key,num_of_images)
		image_urls = image_scrapper.find_image_urls()
		image_scrapper.save_images(image_urls)


