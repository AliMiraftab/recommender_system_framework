import logging
import os
import json

from predict import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# read-test-files
TEST_INPUT_OW = None
TEST_OUTPUT_OW = None
cur_path = os.path.dirname(os.path.abspath(__file__))
logger.info("testing...current path:" + str(cur_path))
try:
  logger.info("loading test files:" + str(cur_path))
  TEST_INPUT_OW_FILENAME = os.path.join(cur_path, "input_unit_test.json")
  TEST_OUTPUT_OW_FILENAME = os.path.join(cur_path, "output_unit_test.json")
  with open(TEST_INPUT_OW_FILENAME) as f:
   TEST_INPUT_OW = json.load(f)
  with open(TEST_OUTPUT_OW_FILENAME) as f:
   TEST_OUTPUT_OW = json.load(f)
  logger.info("test files loaded successfully.")
except:
  logger.exception("test files load failed.")

def test_predict():		
	assert predict(TEST_INPUT_OW) == TEST_OUTPUT_OW