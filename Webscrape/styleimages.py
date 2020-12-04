#Famous painters painting will be taken as style image

from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
from GoogleImageScraper import GoogleImageScraper
import os

my_url = 'http://www.theartwolf.com/articles/most-important-painters.htm'

# opens the connection and downloads html page from url
uClient = uReq(my_url)
page_html = uClient.read()
uClient.close()

# parses html into a soup data structure to traverse html
# as if it were a json data type.
page_soup = soup(page_html, "html.parser")

#finds each painter div
containers = page_soup.findAll("div",{"class":"noticiacentro"})

#Fetching all the p tags from the object
all_p_tags = page_soup.find_all('p')

#All the name of the painters lie in the range 94->195
painters=[]
for i in range(94,195):
	#print(all_p_tags[i].strong.text)
	painters.append(all_p_tags[i].strong.text)



#style images web scraping of approximately 50 painters paintings
#making a directory and storing those images in them
#some of the  images arent in jpg or png format and some are not accessible hence when I download lets say 10 only 4 /5 get downloaded

for i in range(0,50):
	webdriver_path = os.getcwd() + "\\webdriver\\chromedriver.exe"
	os.mkdir(os.getcwd()+"\\dataset\\images\\style_images\\"+painters[i])
	image_path = os.getcwd()+"\\dataset\\images\\style_images\\"+painters[i]
	search_key = painters[i]+" paintings"
	num_of_images = 100
	image_scrapper = GoogleImageScraper(webdriver_path,image_path,search_key,num_of_images)
	image_urls = image_scrapper.find_image_urls()
	image_scrapper.save_images(image_urls)
