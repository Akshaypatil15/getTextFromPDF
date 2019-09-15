'''
Copyright 2019 Patil Akshay Vilas Vaishali

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from pdf2image import convert_from_path, convert_from_bytes
from google_vision_ocr import get_text_from_image
from time import time
from PIL import Image
import configparser
import requests
import logging
import os
import io

class multipage_pdf_to_images:

	def __init__(self, gobj_logger=None):
		try:
			if gobj_logger is None:
				ROOT_DIR  = os.path.dirname(os.path.realpath(__file__))
				logging.basicConfig(filename=ROOT_DIR+"/Logs.log", 
				                	format='%(asctime)s %(message)s', 
				            		filemode='a')
				self.pobj_logger = logging.getLogger()
				self.pobj_logger.setLevel(logging.DEBUG)
			else:
				self.pobj_logger = gobj_logger
		except Exception as e:
			print("Configuration error...\n"+str(e))
			raise

	def download_pdf(self, lstr_pdf_source_url, lstr_pdf_save_path=None):
		try:
			r = requests.get(lstr_pdf_source_url, stream=True)
			if lstr_pdf_save_path is not None:
				with open(lstr_pdf_save_path, 'wb') as fd:
					for chunk in r.iter_content(chunk_size=2000):
						fd.write(chunk)
				return lstr_pdf_save_path
			else:
				return r.content
		except Exception:
			self.pobj_logger.error("Method Name: download_pdf(lstr_pdf_source_url, lstr_pdf_save_path=None)", exc_info=True)
			raise

	def convert_pdf(self, file_path, output_path=None):
		try:
			images = []		
			if type(file_path) == str and output_path is not None:
				if os.path.isfile(file_path):
					# convert pdf to multiple image
					images = convert_from_path(file_path, output_folder=None)
					file_name = ".".join(os.path.basename(file_path).split(".")[:-1])
					for i in range(len(images)):
						image_path = f'{output_path}/{file_name}_{i}.jpg'
						images[i].save(image_path, 'JPEG')
						print("Successfully image saved in ", image_path)
						return image_path
			
			elif isinstance(file_path, (bytes, bytearray)) and output_path is None:
				llst_tmp_images = convert_from_bytes(file_path)
				for i in llst_tmp_images:
					imgByteArr = io.BytesIO()
					i.save(imgByteArr, format='PNG')
					images.append(imgByteArr.getvalue())
				return images
		except Exception:
			self.pobj_logger.error("Method Name: convert_pdf(self, file_path, output_path=None)", exc_info=True)
			raise

def main_module():
	try:
		lobj_parser = configparser.ConfigParser()
		lobj_parser.read('config.ini')
		lstr_save_path = str(lobj_parser['PATHS']['SAVE_PATH'])
		lstr_pdf_save_path = str(lobj_parser['PATHS']['PDF_SAVE_PATH'])
		lstr_pdf_url = str(lobj_parser['URLS']['PDF_URL'])
		lstr_google_cloud_platform_api_key = str(lobj_parser['GOOGLE_VISION_API_SETTING']['GOOGLE_CLOUD_PLATFORM_API_KEY'])
		lstr_vision_api_url = str(lobj_parser['GOOGLE_VISION_API_SETTING']['VISION_API_URL'])

		# Creating object of multipage_pdf_to_images class
		pobj_multipage_pdf_to_images = multipage_pdf_to_images()
		
		# Let's download pdf from given url
		# If you want to save pdf file pass variable lstr_pdf_save_path
		llst_images = pobj_multipage_pdf_to_images.download_pdf(lstr_pdf_url)
		
		# Convert pdf to image
		llst_result = pobj_multipage_pdf_to_images.convert_pdf(llst_images)
		
		# Initialize get_text_from_image class object
		lobj_get_text_from_image = get_text_from_image(lstr_vision_api_url, lstr_google_cloud_platform_api_key, 
														pobj_multipage_pdf_to_images.pobj_logger)
		for idx in range(len(llst_result)):
			print("Calling Google Vision API")
			ldict_result = lobj_get_text_from_image.ocr_using_google_vision_api(llst_result[idx])
			image_name = ".".join(os.path.basename(lstr_pdf_url).split(".")[:-1])
			
			print("Saving text and coordinates in csv file")
			# Saving text and coordinates in csv file.
			lstr_csv_save_path = os.path.join(lstr_save_path, "%s%s.csv" % (image_name, idx))
			lobj_get_text_from_image.write_dict_to_csv(ldict_result, lstr_csv_save_path)
			
			print("Saving Image...")
			# Draw coordinates on image (Optional)
			lstr_image_save_path = os.path.join(lstr_save_path, "%s%s.jpg" % (image_name, idx))
			lobj_get_text_from_image.write_bounding_boxes_on_image(ldict_result, llst_result[idx], lstr_image_save_path)
			pobj_multipage_pdf_to_images.pobj_logger.info("Processed %s..." % (image_name))
	except Exception as e:
			print("Error...\n"+str(e))

if __name__ == "__main__":
	startTime = time()
	main_module()
	print("Execution time is {0:.2f} seconds".format((time()-startTime)))
