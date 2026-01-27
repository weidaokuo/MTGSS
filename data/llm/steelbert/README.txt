

# Download

	Please download the SteelBERT model from the website https://huggingface.co/MGE-LLMs/SteelBERT/tree/main and save it to the /data/llm/steelbert folder.





# Usage and validation

	First: Replace the model_path in the Model_Select class within /data/tools/LLM_encode.py with the local path to your downloaded SteelBERT model.
	Second: Ensure that the transformers library is installed by running pip install transformers. The version used in this work is 4.57.3.
	Third ：Running LLM_encode.py, SteelBERT can be used normally.