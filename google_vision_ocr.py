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
from time import time
import configparser
import numpy as np
import requests
import logging
import base64
import json
import cv2
import os

class get_text_from_image:

	def __init__(self, vision_api_url, google_cloud_platform_api_key, gobj_logger=None):
		try:
			self.pstr_google_cloud_platform_api_key = google_cloud_platform_api_key
			self.pstr_text_detection_address = vision_api_url
			if gobj_logger is None:
				ROOT_DIR  = os.path.dirname(os.path.realpath(__file__))
				logging.basicConfig(filename=ROOT_DIR+"/Logs.log", 
				                	format='%(asctime)s %(message)s', 
				            		filemode='a')
				self.pobj_logger = logging.getLogger()
				self.pobj_logger.setLevel(logging.DEBUG)
			else:
				print("continuing already created logger object")
				self.pobj_logger = gobj_logger
		except Exception as e:
			print("Configuration error...\n"+str(e))
			raise

	def ocr_using_google_vision_api(self, pstr_input_image_path):
		'''
			method description :
					This method uses google vision api - OCR to find any text in an image.
			Input :
					pstr_input_image_path : Input image path
			Output : 
					{
					"responses": [
						{
						"textAnnotations": [
							{
								"locale": "en",
								"description": "WAITING?\nPLEASE\nTURN OFF\nYOUR\nENGINE\n",
								"boundingPoly": {
								"vertices": [
								{ "x": 341, "y": 828
								},
								{ "x": 2249, "y": 828
								},
								{ "x": 2249, "y": 1993
								},
								{ "x": 341, "y": 1993
								}			]
							  			}
							},{
								"description": "WAITING?",
								"boundingPoly": {
								"vertices": [
								{ "x": 352, "y": 828
								},
								{ "x": 2248, "y": 911
								},
								{ "x": 2238, "y": 1148
								},
								{ "x": 342, "y": 1065
								}			]
							  			}
							},
							"fullTextAnnotation": {
								"pages": [ ]
										},
										"paragraphs": [ ]
											},
											"words":  [
											},
											"symbols":[
											}
											]
											}
										],
										"blockType": "TEXT"
									},	]
								}	],
								"text": "WAITING?\nPLEASE\nTURN OFF\nYOUR\nENGINE\n"
						}	}	]	}																	
		'''
		try:
			ldict_detected_text = {}
			lint_respose_code = 0
			lstr_request_address = self.pstr_text_detection_address+self.pstr_google_cloud_platform_api_key
			ljson_request_header = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
			lstr_encode_image = ""
			
			if type(pstr_input_image_path) == str:
				if True in [pstr_input_image_path.endswith(a) for a in ["jpg", "JPG", "png", "PNG", "jpeg", "JPEG"]]:
					imageFile = open(pstr_input_image_path, "rb")
					lstr_encode_image = base64.b64encode(imageFile.read()).decode()
					imageFile.close()
			
			elif isinstance(pstr_input_image_path, (bytes, bytearray)):
				lstr_encode_image = base64.b64encode(pstr_input_image_path).decode()

			assert (len(lstr_encode_image) != 0), "Invalid Image Content"

			ljson_request_payload = {'requests': 
						[
							{
								"image":
								{"content":lstr_encode_image
								},
								"features": 
								[
									{'type': 'TEXT_DETECTION'
									}
								]
									# "imageContext": 
									# 		{ 'languageHints': [
									# 			'en' ]
									# 		}
							}    
						]
						}

			try:
				lstr_http_response = requests.post(lstr_request_address, data=json.dumps(ljson_request_payload), headers=ljson_request_header)
				lint_respose_code = lstr_http_response.status_code
				if lint_respose_code != 200:
					self.pobj_logger.info("Method Name: ocr_using_google_vision_api(self, pstr_input_image_path)"+
														"\n response code: "+str(lint_respose_code))
					return ldict_detected_text
				else:
					ljson_response_data = json.loads(lstr_http_response.text)
					if ljson_response_data['responses'][0]:
						ldict_detected_text = ljson_response_data['responses'][0]["textAnnotations"] #['fullTextAnnotation']['text']
					return ldict_detected_text
			except Exception:
				self.pobj_logger.error("Method Name: ocr_using_google_vision_api(self, pstr_input_image_path)", exc_info=True)
				return ldict_detected_text
					
		except Exception:
			self.pobj_logger.error("Method Name: ocr_using_google_vision_api(self, pstr_input_image_path)", exc_info=True)
			raise

	def write_dict_to_csv(self, ldict_input, lstr_output_csv_path):
		try:
			llst_text_to_write = [["xmin", "ymin", "xmax", "ymax", "text"]]
			
			if not os.path.exists(os.path.dirname(lstr_output_csv_path)):
				os.makedirs(os.path.dirname(lstr_output_csv_path))
			
			for idx in range (1, len(ldict_input)):
				lstr_text = ldict_input[idx]['description']
				xmin, ymin = list(ldict_input[idx]['boundingPoly']['vertices'][0].values())
				xmax, ymax = list(ldict_input[idx]['boundingPoly']['vertices'][2].values())
				llst_text_to_write.append([xmin, ymin, xmax, ymax, lstr_text])
			llst_text_to_write = np.array(llst_text_to_write)
			np.savetxt(lstr_output_csv_path, llst_text_to_write, fmt='%s', delimiter=',')
		except Exception:
			self.pobj_logger.error("Method Name: write_dict_to_csv(self, ldict_input, lstr_output_csv_path)", exc_info=True)
			raise

	def write_bounding_boxes_on_image(self, ldict_input, lstr_input_image_path, lstr_image_save_path):
		try:
			line_size = 1
			if type(lstr_input_image_path) == str:
				larr_image = cv2.imread(lstr_input_image_path)

			elif isinstance(lstr_input_image_path, (bytes, bytearray)):
				nparr = np.fromstring(lstr_input_image_path, np.uint8)
				larr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

			line_size = 2 if (max(larr_image.shape) > 1000) else line_size
			
			for idx in range (1, len(ldict_input)):
				xmin, ymin = list(ldict_input[idx]['boundingPoly']['vertices'][0].values())
				xmax, ymax = list(ldict_input[idx]['boundingPoly']['vertices'][2].values())
				cv2.rectangle(larr_image, (xmin, ymin), (xmax, ymax), (255,0,0), line_size)
			cv2.imwrite(lstr_image_save_path, larr_image)
		except Exception:
			self.pobj_logger.error("Method Name: write_bounding_boxes_on_image(self, ldict_input,"+
									"lstr_input_image_path, lstr_image_save_path)", exc_info=True)
			raise

def main_module():
	try:
		lobj_parser = configparser.ConfigParser()
		lobj_parser.read('config.ini')
		lstr_save_path = str(lobj_parser['PATHS']['SAVE_PATH'])
		lstr_input_img_path = str(lobj_parser['PATHS']['INPUT_IMG_PATH'])
		lstr_google_cloud_platform_api_key = str(lobj_parser['GOOGLE_VISION_API_SETTING']['GOOGLE_CLOUD_PLATFORM_API_KEY'])
		lstr_vision_api_url = str(lobj_parser['GOOGLE_VISION_API_SETTING']['VISION_API_URL'])
		image_name = ".".join(os.path.basename(lstr_input_img_path).split(".")[:-1])

		# Initialize get_text_from_image class object
		lobj_get_text_from_image = get_text_from_image(lstr_vision_api_url, lstr_google_cloud_platform_api_key)

		print("Calling Google Vision API")
		ldict_result = lobj_get_text_from_image.ocr_using_google_vision_api(lstr_input_img_path)
		
		print("Saving text and coordinates in csv file")
		# Saving text and coordinates in csv file.
		lstr_csv_save_path = os.path.join(lstr_save_path, "%s.csv" % image_name)
		lobj_get_text_from_image.write_dict_to_csv(ldict_result, lstr_csv_save_path)
		
		print("Saving Image %s.jpg" % image_name)
		# Draw coordinates on image (Optional)
		lstr_image_save_path = os.path.join(lstr_save_path, "%s.jpg" % image_name)
		lobj_get_text_from_image.write_bounding_boxes_on_image(ldict_result, lstr_input_img_path, lstr_image_save_path)
		lobj_get_text_from_image.pobj_logger.info("Processed %s..." % (image_name))
	except Exception as e:
			print("Error...\n"+str(e))

if __name__ == "__main__":
	startTime = time()
	main_module()
	print("Execution time is {0:.2f} seconds".format((time()-startTime)))
