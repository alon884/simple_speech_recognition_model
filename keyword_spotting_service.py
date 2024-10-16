

import tensorflow.keras as keras


MODEL_PATH =  "/media/alon/DATA/ProjectsForCV/proj1/model.h5"

class _Keyword_Spotting_Service:

	model = None
	_mappings = [
        "down",
        "go",
        "left",
        "no",
        "off",
        "on",
        "right",
        "stop",
        "up",
        "yes"
    ]
	_instance = None


def _Keyword_Spotting_Service():

	# ensure that we only have 1 instance of KSS
	if _Keyword_Spotting_Service._instance is None: 
		_Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
		_Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)

	return _Keyword_Spotting_Service._instance

	